from flask import Flask, request, jsonify
from flask_cors import CORS
import jwt
import os
from main import Router

app = Flask(__name__)

CORS(app,
     origins=["http://localhost:4200"],
     allow_headers=["Authorization", "Content-Type"],
     methods=["GET", "POST", "OPTIONS"],
     supports_credentials=True)

@app.after_request
def add_cors_headers(response):
    response.headers["Access-Control-Allow-Origin"] = "http://localhost:4200"
    response.headers["Access-Control-Allow-Headers"] = "Authorization, Content-Type"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    return response

# ── Config ────────────────────────────────────────────────────────────────────
JWT_SECRET = "your-secret-key-change-in-production-12345"

_routers: dict[str, Router] = {}

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

# ── Routes ────────────────────────────────────────────────────────────────────
@app.route("/debug", methods=["GET", "POST", "OPTIONS"])
def debug():
    return jsonify({
        "auth_header": request.headers.get("Authorization", "MISSING"),
        "headers_received": dict(request.headers),
    })

@app.route("/chat", methods=["POST", "OPTIONS"])
def chat():
    if request.method == "OPTIONS":
        return jsonify({}), 200

    patient_name = _get_patient_name(request.headers.get("Authorization", ""))
    if not patient_name:
        return jsonify({"error": "Unauthorized"}), 401

    data = request.get_json()
    user_input = (data or {}).get("message", "").strip()
    display_name = (data or {}).get("patientName", patient_name)

    if not user_input:
        return jsonify({"error": "Empty message"}), 400

    if display_name not in _routers:
        _routers[display_name] = Router(patient_name=display_name)

    router = _routers[display_name]
    reply = router.handle(user_input)

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

if __name__ == "__main__":
    app.run(port=5000, debug=True)