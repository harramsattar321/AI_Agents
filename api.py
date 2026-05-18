from flask import Flask, request, jsonify
from flask_cors import CORS
import jwt
import io
import os
import pdfplumber
from main import Router
from report_analyzer import ReportAnalyzer
from report_chatter import ReportChatter
from db import HospitalDB
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)

CORS(app,
     origins=["http://localhost:4200"],
     allow_headers=["Authorization", "Content-Type"],
     methods=["GET", "POST", "OPTIONS"],
     supports_credentials=True)

@app.after_request
def add_cors_headers(response):
    response.headers["Access-Control-Allow-Origin"]  = "http://localhost:4200"
    response.headers["Access-Control-Allow-Headers"] = "Authorization, Content-Type"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    return response

# ── Config ────────────────────────────────────────────────────────────────────

JWT_SECRET = "your-secret-key-change-in-production-12345"
MONGO_URI  = os.getenv("MONGO_URI")

# Shared DB — one connection reused across all agents
_db = HospitalDB(uri=MONGO_URI)

_routers:         dict[str, Router] = {}
_report_analyzer = ReportAnalyzer(groq_api_key=os.getenv("GROQ_KEY_REPORT"))
_report_chatter  = ReportChatter(
    groq_api_key  = os.getenv("GROQ_KEY_REPORT"),   # used for report Q&A calls
    db            = _db,
    mongo_uri     = MONGO_URI,                        # passed to HospitalChatbot for hospital queries
    groq_key_chat = os.getenv("GROQ_KEY_CHAT"),       # passed to HospitalChatbot
)

# Per-patient report Q&A sessions
# Structure: { patient_name: {"report_text": str, "history": list[dict]} }
_report_sessions: dict[str, dict] = {}


# ── Helpers ───────────────────────────────────────────────────────────────────

def _get_patient_name(auth_header: str) -> str | None:
    if not auth_header or not auth_header.startswith("Bearer "):
        print("❌ No auth header or wrong format")
        return None
    token = auth_header.split(" ")[1]
    print("🔑 Token received:", token[:30], "...")
    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=["HS256"])
        print("✅ Decoded payload:", payload)
        return payload.get("userId") or payload.get("email")
    except jwt.ExpiredSignatureError:
        print("❌ Token EXPIRED")
        return None
    except jwt.InvalidTokenError as e:
        print("❌ Invalid token:", str(e))
        return None


def _extract_text_from_pdf(file_bytes: bytes) -> str:
    try:
        with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
            return "\n".join(
                page.extract_text() or "" for page in pdf.pages
            ).strip()
    except Exception:
        return ""


# ── Routes ────────────────────────────────────────────────────────────────────

@app.route("/debug", methods=["GET", "POST", "OPTIONS"])
def debug():
    return jsonify({
        "auth_header":      request.headers.get("Authorization", "MISSING"),
        "headers_received": dict(request.headers),
    })


@app.route("/chat", methods=["POST", "OPTIONS"])
def chat():
    if request.method == "OPTIONS":
        return jsonify({}), 200

    patient_name = _get_patient_name(request.headers.get("Authorization", ""))
    if not patient_name:
        return jsonify({"error": "Unauthorized"}), 401

    data         = request.get_json()
    user_input   = (data or {}).get("message", "").strip()
    display_name = (data or {}).get("patientName", patient_name)

    if not user_input:
        return jsonify({"error": "Empty message"}), 400

    if display_name not in _routers:
        _routers[display_name] = Router(patient_name=display_name)

    router = _routers[display_name]
    reply  = router.handle(user_input)

    return jsonify({"reply": reply, "state": router.state})


@app.route("/chat/reset", methods=["POST", "OPTIONS"])
def reset():
    if request.method == "OPTIONS":
        return jsonify({}), 200

    patient_name = _get_patient_name(request.headers.get("Authorization", ""))
    if patient_name and patient_name in _routers:
        _routers[patient_name].close()
        del _routers[patient_name]
    return jsonify({"status": "reset"})


# ── /analyze-report ───────────────────────────────────────────────────────────
# Response shape unchanged: {"patient": ..., "analysis": ...}
# Also saves extracted text into _report_sessions for follow-up Q&A.

@app.route("/analyze-report", methods=["POST", "OPTIONS"])
def analyze_report():
    if request.method == "OPTIONS":
        return jsonify({}), 200

    patient_name = _get_patient_name(request.headers.get("Authorization", ""))
    if not patient_name:
        return jsonify({"error": "Unauthorized"}), 401

    file = request.files.get("file")
    success, status_code, result = _report_analyzer.analyze(file)

    if not success:
        return jsonify({"error": result}), status_code

    # Re-read after analyze() consumed the stream
    file.seek(0)
    report_text = _extract_text_from_pdf(file.read())

    _report_sessions[patient_name] = {
        "report_text": report_text,
        "history":     [],
    }

    return jsonify({"patient": patient_name, "analysis": result})


# ── /chat-report ──────────────────────────────────────────────────────────────
# Request:  {"message": "<question>", "patientName": "<display name>"}
# Response: {"reply": "<answer>"}  — same shape as /chat

@app.route("/chat-report", methods=["POST", "OPTIONS"])
def chat_report():
    if request.method == "OPTIONS":
        return jsonify({}), 200

    patient_name = _get_patient_name(request.headers.get("Authorization", ""))
    if not patient_name:
        return jsonify({"error": "Unauthorized"}), 401

    data         = request.get_json()
    question     = (data or {}).get("message", "").strip()
    display_name = (data or {}).get("patientName", patient_name)

    if not question:
        return jsonify({"error": "Empty message"}), 400

    session = _report_sessions.get(patient_name)
    if not session or not session.get("report_text"):
        return jsonify({
            "error": "No report found. Please upload your medical report first."
        }), 400

    try:
        answer = _report_chatter.chat(
            report_text  = session["report_text"],
            question     = question,
            history      = session["history"],
            patient_name = display_name,
        )
    except Exception as e:
        print(f"❌ /chat-report error: {e}")
        return jsonify({"error": "Could not process your question. Please try again."}), 500

    # Update conversation history (bounded to last 6 turns)
    session["history"].append({"role": "user",      "content": question})
    session["history"].append({"role": "assistant",  "content": answer})
    if len(session["history"]) > 12:
        session["history"] = session["history"][-12:]

    return jsonify({"reply": answer})


# ── /clear-report ─────────────────────────────────────────────────────────────

@app.route("/clear-report", methods=["POST", "OPTIONS"])
def clear_report():
    if request.method == "OPTIONS":
        return jsonify({}), 200

    patient_name = _get_patient_name(request.headers.get("Authorization", ""))
    if patient_name and patient_name in _report_sessions:
        del _report_sessions[patient_name]

    return jsonify({"status": "cleared"})


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)