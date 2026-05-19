"""
api.py — Flask API
==================
One /chat endpoint for everything.
The Router in main.py uses an LLM to decide per-message which agent handles
the turn — general chat, report Q&A, insurance, or booking — invisibly.

KEY FIX: Router is always keyed on JWT userId (patient_name from token).
The patientName field from the request body is used only for display/greeting —
never as the session key. This ensures /analyze-report and /chat always
hit the same router instance regardless of what the frontend sends in the body.
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import jwt
import io
import os
import pdfplumber

from main            import Router
from report_analyzer import ReportAnalyzer
from db              import HospitalDB
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)

CORS(
    app,
    origins=["http://localhost:4200", "https://ambitious-wave-0575e9603.7.azurestaticapps.net"],
    allow_headers=["Authorization", "Content-Type"],
    methods=["GET", "POST", "OPTIONS"],
    supports_credentials=True,
)

ALLOWED_ORIGINS = [
    "http://localhost:4200",
    "https://ambitious-wave-0575e9603.7.azurestaticapps.net",
]

@app.after_request
def add_cors_headers(response):
    origin = request.headers.get("Origin", "")
    if origin in ALLOWED_ORIGINS:
        response.headers["Access-Control-Allow-Origin"]  = origin
    response.headers["Access-Control-Allow-Headers"] = "Authorization, Content-Type"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    return response


# ─────────────────────────────────────────────────────────────────────────────
#  CONFIG
# ─────────────────────────────────────────────────────────────────────────────
JWT_SECRET = "your-secret-key-change-in-production-12345"
MONGO_URI  = os.getenv("MONGO_URI")

_db              = HospitalDB(uri=MONGO_URI)
_report_analyzer = ReportAnalyzer(groq_api_key=os.getenv("GROQ_KEY_REPORT"))

# One Router instance per patient — keyed on JWT userId, never display name
_routers: dict[str, Router] = {}


# ─────────────────────────────────────────────────────────────────────────────
#  AUTH HELPER
# ─────────────────────────────────────────────────────────────────────────────
def _get_patient_id(auth_header: str) -> str | None:
    """
    Extract patient ID from JWT token.
    Returns userId (e.g. PAT1776188487656870) or email as fallback.
    This is the single source of truth for session keying.
    """
    if not auth_header or not auth_header.startswith("Bearer "):
        print("❌ No auth header or wrong format")
        return None
    token = auth_header.split(" ")[1]
    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=["HS256"])
        return payload.get("userId") or payload.get("email")
    except jwt.ExpiredSignatureError:
        print("❌ Token expired")
        return None
    except jwt.InvalidTokenError as e:
        print(f"❌ Invalid token: {e}")
        return None


def _get_router(patient_id: str) -> Router:
    """Always keyed on JWT userId — never on display name."""
    if patient_id not in _routers:
        _routers[patient_id] = Router(patient_name=patient_id)
    return _routers[patient_id]


def _extract_text_from_pdf(file_bytes: bytes) -> str:
    try:
        with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
            return "\n".join(
                page.extract_text() or "" for page in pdf.pages
            ).strip()
    except Exception:
        return ""


# ─────────────────────────────────────────────────────────────────────────────
#  ROUTES
# ─────────────────────────────────────────────────────────────────────────────

@app.route("/debug", methods=["GET", "POST", "OPTIONS"])
def debug():
    return jsonify({
        "auth_header":      request.headers.get("Authorization", "MISSING"),
        "headers_received": dict(request.headers),
    })


# ── /chat — unified endpoint for ALL messages ─────────────────────────────────
# Request:  {"message": "...", "patientName": "..."}
# Response: {"reply": "...", "has_report": true/false}

@app.route("/chat", methods=["POST", "OPTIONS"])
def chat():
    if request.method == "OPTIONS":
        return jsonify({}), 200

    patient_id = _get_patient_id(request.headers.get("Authorization", ""))
    if not patient_id:
        return jsonify({"error": "Unauthorized"}), 401

    data       = request.get_json()
    user_input = (data or {}).get("message", "").strip()

    if not user_input:
        return jsonify({"error": "Empty message"}), 400

    router = _get_router(patient_id)
    reply  = router.handle(user_input)

    return jsonify({
        "reply":      reply,
        "has_report": router.has_report,
    })


# ── /analyze-report — upload PDF, store in router session ────────────────────
# Request:  multipart/form-data with "file" field
# Response: {"patient": ..., "analysis": ..., "has_report": true}

@app.route("/analyze-report", methods=["POST", "OPTIONS"])
def analyze_report():
    if request.method == "OPTIONS":
        return jsonify({}), 200

    patient_id = _get_patient_id(request.headers.get("Authorization", ""))
    if not patient_id:
        return jsonify({"error": "Unauthorized"}), 401

    file = request.files.get("file")
    success, status_code, result = _report_analyzer.analyze(file)

    if not success:
        return jsonify({"error": result}), status_code

    # Re-read after analyze() consumed the stream
    file.seek(0)
    file_bytes  = file.read()
    report_text = _extract_text_from_pdf(file_bytes)

    if not report_text:
        return jsonify({"error": "Could not extract text from PDF (may be scanned)."}), 422

    # Store in the router keyed on JWT userId — same key /chat uses
    router = _get_router(patient_id)
    router._report_text    = report_text
    router._report_history = []

    return jsonify({
        "patient":    patient_id,
        "analysis":   result,
        "has_report": True,
    })


# ── /chat/reset — clear the router session for a patient ─────────────────────
@app.route("/chat/reset", methods=["POST", "OPTIONS"])
def reset():
    if request.method == "OPTIONS":
        return jsonify({}), 200

    patient_id = _get_patient_id(request.headers.get("Authorization", ""))
    if not patient_id:
        return jsonify({"error": "Unauthorized"}), 401

    if patient_id in _routers:
        _routers[patient_id].close()
        del _routers[patient_id]

    return jsonify({"status": "reset"})


# ── /clear-report — remove report from session but keep chat history ──────────
@app.route("/clear-report", methods=["POST", "OPTIONS"])
def clear_report():
    if request.method == "OPTIONS":
        return jsonify({}), 200

    patient_id = _get_patient_id(request.headers.get("Authorization", ""))
    if not patient_id:
        return jsonify({"error": "Unauthorized"}), 401

    if patient_id in _routers:
        router = _routers[patient_id]
        router._report_text    = ""
        router._report_history = []

    return jsonify({"status": "cleared", "has_report": False})


# ── /chat-report — DEPRECATED: kept for backwards compatibility ───────────────
@app.route("/chat-report", methods=["POST", "OPTIONS"])
def chat_report():
    if request.method == "OPTIONS":
        return jsonify({}), 200

    patient_id = _get_patient_id(request.headers.get("Authorization", ""))
    if not patient_id:
        return jsonify({"error": "Unauthorized"}), 401

    data     = request.get_json()
    question = (data or {}).get("message", "").strip()

    if not question:
        return jsonify({"error": "Empty message"}), 400

    router = _get_router(patient_id)

    if not router.has_report:
        return jsonify({
            "error": "No report found. Please upload your medical report first."
        }), 400

    reply = router.handle(question)
    return jsonify({"reply": reply})


# ─────────────────────────────────────────────────────────────────────────────
#  ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
