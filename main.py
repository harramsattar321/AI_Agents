"""
main.py — Invisible LLM Router
===============================
One LLM call per message decides which agent handles the turn.
No explicit state machine. No "exit report" commands. No mode labels.

The user just talks. The router reads conversation history + session context
and silently picks the right agent every single turn.

Agents:
  HospitalChatbot   (chatbot.py)         — general receptionist
  BookingAgent      (booking_agent.py)   — appointment booking
  InsuranceAgent    (insurance_agent.py) — insurance / coverage
  ReportChatter     (report_chatter.py)  — medical report Q&A

Routing signals (returned by LLM as JSON):
  {"route": "chat"}      — general hospital question
  {"route": "report"}    — question about the loaded report
  {"route": "insurance"} — insurance / coverage question
  {"route": "booking"}   — appointment booking intent
"""

import re
import os
import io
import json
import pdfplumber
import requests

from chatbot         import HospitalChatbot
from booking_agent   import BookingAgent
from insurance_agent import InsuranceAgent
from report_chatter  import ReportChatter
from report_analyzer import ReportAnalyzer
from db              import HospitalDB
from dotenv import load_dotenv

load_dotenv()

# ─────────────────────────────────────────────────────────────────────────────
#  CONFIG
# ─────────────────────────────────────────────────────────────────────────────
MONGO_URI    = os.getenv("MONGO_URI")
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
ROUTER_MODEL = "llama-3.3-70b-versatile"

# ─────────────────────────────────────────────────────────────────────────────
#  KEY POOLS
# ─────────────────────────────────────────────────────────────────────────────
def _load_key_pool(prefix: str) -> list[str]:
    """Collect PREFIX_1, PREFIX_2, ... from .env and return as a list."""
    keys = []
    i = 1
    while True:
        val = os.getenv(f"{prefix}_{i}")
        if not val:
            break
        keys.append(val)
        i += 1
    return keys

KEY_POOLS: dict[str, list[str]] = {
    "chat":      _load_key_pool("GROQ_KEY_CHAT"),
    "booking":   _load_key_pool("GROQ_KEY_BOOK"),
    "insurance": _load_key_pool("GROQ_KEY_INS"),
    "report":    _load_key_pool("GROQ_KEY_REPORT"),
}

# Current active index per pool
_key_index: dict[str, int] = {k: 0 for k in KEY_POOLS}


def get_key(pool: str) -> str:
    """Return the currently active key for this pool."""
    keys = KEY_POOLS[pool]
    if not keys:
        raise ValueError(f"No API keys loaded for pool: '{pool}'. Check your .env file.")
    return keys[_key_index[pool]]


def rotate_key(pool: str) -> str | None:
    """
    Advance to the next key in the pool.
    Returns the new key, or None if all keys are exhausted.
    """
    keys  = KEY_POOLS[pool]
    current = _key_index[pool]
    if current + 1 < len(keys):
        _key_index[pool] = current + 1
        print(f"[KeyRotator] ↻  '{pool}' rotated to key index {_key_index[pool]}")
        return keys[_key_index[pool]]
    print(f"[KeyRotator] ⚠️  All keys exhausted for pool: '{pool}'")
    return None


def groq_request_with_fallback(pool: str, payload: dict, timeout: int = 10) -> dict:
    """
    POST to Groq API using the active key for `pool`.
    On 429 (rate-limit) or 401 (bad/expired key), rotate to the next key
    and retry automatically — until all keys in the pool are exhausted.
    Raises the last HTTPError if every key fails.
    """
    while True:
        key = get_key(pool)
        try:
            response = requests.post(
                GROQ_API_URL,
                headers={
                    "Authorization": f"Bearer {key}",
                    "Content-Type":  "application/json",
                },
                json=payload,
                timeout=timeout,
            )

            if response.status_code in (429, 401):
                print(
                    f"[KeyRotator] '{pool}' key failed "
                    f"(HTTP {response.status_code}), trying next key..."
                )
                new_key = rotate_key(pool)
                if new_key is None:
                    response.raise_for_status()   # all keys dead → propagate
                continue                           # retry with rotated key

            response.raise_for_status()
            return response.json()

        except requests.exceptions.Timeout:
            print(f"[KeyRotator] '{pool}' key timed out, trying next key...")
            new_key = rotate_key(pool)
            if new_key is None:
                raise
            # loop continues with new key


# ─────────────────────────────────────────────────────────────────────────────
#  ROUTER SYSTEM PROMPT
# ─────────────────────────────────────────────────────────────────────────────
_ROUTER_SYSTEM = """You are a silent routing assistant for a hospital AI chatbot.
Your ONLY job is to read the conversation and decide which agent should handle
the latest patient message. Reply with ONLY a JSON object — nothing else.

AGENTS:
  "chat"      — General hospital questions: doctors, timings, fees, departments,
                lab tests, hospital info, greetings, anything not covered below.
  "report"    — Questions about the patient's uploaded medical report: values,
                what's abnormal, severity, what a result means, which doctor to
                see based on the report findings.
  "insurance" — Insurance, coverage, claims, health plans, panel hospitals,
                policy, co-pay, reimbursement.
  "booking"   — Booking, scheduling, or reserving a new appointment with a doctor.

RULES:
  1. If no report has been uploaded, NEVER route to "report".
  2. "book", "schedule", "appointment" → "booking".
  3. Insurance keywords → "insurance".
  4. If the patient is mid-conversation with one agent and asks a follow-up
     that clearly belongs there, keep routing to that agent.
  5. When in doubt → "chat".
  6. Output ONLY: {{"route": "<agent>"}}  — no explanation, no other keys.

REPORT LOADED: {report_loaded}
"""

# ─────────────────────────────────────────────────────────────────────────────
#  CANCEL / BOOKING TERMINAL HELPERS
# ─────────────────────────────────────────────────────────────────────────────
_CANCEL_BOOKING_RE = re.compile(
    r"\b(cancel appointment|cancel my appointment|reschedule|re.?schedule"
    r"|change appointment|reappointment)\b",
    re.IGNORECASE,
)

def _format_booking_terminal(reply: str, patient_name: str) -> str:
    if reply.startswith("BOOKING_COMPLETE"):
        detail = reply.replace("BOOKING_COMPLETE:", "").strip()
        return (
            f"✅ {detail}\n\n"
            f"Is there anything else I can help you with, {patient_name}?"
        )
    if reply.startswith("BOOKING_CANCELLED"):
        return (
            f"No problem, {patient_name}. Your booking has been cancelled. "
            f"Feel free to ask me anything else!"
        )
    return reply


# ─────────────────────────────────────────────────────────────────────────────
#  ROUTER CLASS
# ─────────────────────────────────────────────────────────────────────────────
class Router:

    def __init__(self, patient_name: str):
        self.patient_name = patient_name

        self.db = HospitalDB(MONGO_URI)

        # ── Agents — each receives its pool's current active key ──────────────
        self.chat_agent = HospitalChatbot(
            mongo_uri    = MONGO_URI,
            groq_api_key = get_key("chat"),
            patient_name = patient_name,
        )
        self.chat_agent.db = self.db

        self.booking_agent = BookingAgent(
            groq_api_key_2 = get_key("booking"),
            db             = self.db,
            patient_name   = patient_name,
        )

        self.insurance_agent = InsuranceAgent(
            groq_api_key = get_key("insurance"),
            mongo_uri    = MONGO_URI,
            patient_name = patient_name,
        )

        self.report_chatter = ReportChatter(
            groq_api_key  = get_key("report"),
            db            = self.db,
            mongo_uri     = MONGO_URI,
            groq_key_chat = get_key("chat"),
        )

        self.report_analyzer = ReportAnalyzer(groq_api_key=get_key("report"))

        # Report context — loaded once, lives for the whole session
        self._report_text:    str        = ""
        self._report_history: list[dict] = []

        # Full conversation history for the router LLM
        self._history: list[dict] = []

        # Track last route so agents retain context on fallback
        self._last_route: str = "chat"

    # ── Report loaded? ────────────────────────────────────────────────────────
    @property
    def has_report(self) -> bool:
        return bool(self._report_text)

    # ── LLM Router ────────────────────────────────────────────────────────────
    def _route(self, user_input: str) -> str:
        """
        Ask the router LLM which agent should handle this message.
        Uses the 'chat' key pool with automatic fallback.
        Returns one of: "chat", "report", "insurance", "booking".
        Falls back to last known route on any error.
        """
        system = _ROUTER_SYSTEM.format(
            report_loaded="YES" if self.has_report else "NO"
        )

        recent_history = self._history[-10:]
        messages = [{"role": "system", "content": system}]
        messages.extend(recent_history)
        messages.append({"role": "user", "content": user_input})

        try:
            data = groq_request_with_fallback(
                pool    = "chat",
                payload = {
                    "model":       ROUTER_MODEL,
                    "messages":    messages,
                    "temperature": 0.0,
                    "max_tokens":  20,
                },
            )
            raw   = data["choices"][0]["message"]["content"].strip()
            clean = re.sub(r"```(?:json)?|```", "", raw).strip()
            route = json.loads(clean).get("route", "chat").lower()

            if route not in ("chat", "report", "insurance", "booking"):
                route = "chat"
            if route == "report" and not self.has_report:
                route = "chat"

            return route

        except Exception as e:
            print(f"[Router fallback] {e}")
            return self._last_route

    # ── Load PDF report ───────────────────────────────────────────────────────
    def load_report(self, pdf_path: str) -> str:
        """
        Analyze a PDF, store extracted text, return formatted summary string.
        Report context persists for the whole session.
        """
        if not os.path.isfile(pdf_path):
            return f"❌ File not found: {pdf_path}"

        with open(pdf_path, "rb") as f:
            file_bytes = f.read()

        try:
            with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
                self._report_text = "\n".join(
                    p.extract_text() or "" for p in pdf.pages
                ).strip()
        except Exception as e:
            return f"❌ Could not read PDF: {e}"

        if not self._report_text:
            return "❌ Could not extract text from this PDF (may be a scanned image)."

        print("⏳ Analyzing report...")

        class _FakeFile:
            def __init__(self, data, name):
                self.filename = name
                self._stream  = io.BytesIO(data)
            def read(self):    return self._stream.read()
            def seek(self, n): self._stream.seek(n)

        fake_file = _FakeFile(file_bytes, os.path.basename(pdf_path))
        success, _, result = self.report_analyzer.analyze(fake_file)

        if not success:
            self._report_text = ""
            return f"❌ Analysis failed: {result}"

        self._report_history = []

        abnormal     = result.get("abnormal_values", [])
        observations = result.get("key_observations", [])
        advice       = result.get("advice", "")

        lines = [
            "",
            "═" * 56,
            "  📋  REPORT ANALYSIS",
            "═" * 56,
            f"  Type    : {result.get('report_type', 'Unknown')}",
            f"  Summary : {result.get('summary', '')}",
        ]
        if abnormal:
            lines.append("\n  ⚠️  Abnormal Values:")
            for v in abnormal:
                lines.append(
                    f"     • {v['name']}: {v['value']} "
                    f"(normal: {v['normal_range']}) [{v['status']}]"
                )
        if observations:
            lines.append("\n  🔍  Key Observations:")
            for obs in observations:
                lines.append(f"     • {obs}")
        if advice:
            lines.append(f"\n  💡  Advice : {advice}")
        lines += [
            "",
            f"  {result.get('disclaimer', '')}",
            "═" * 56,
            "",
            "  ✅  Report loaded. Ask me anything about it,",
            "  or continue with any other hospital questions.",
            "═" * 56,
        ]
        return "\n".join(lines)

    # ── Single turn handler ───────────────────────────────────────────────────
    def handle(self, user_input: str) -> str:
        if len(self._history) > 40:
            self._history = self._history[-40:]

        # ── Cancel/reschedule shortcut ────────────────────────────────────────
        if _CANCEL_BOOKING_RE.search(user_input):
            self.booking_agent.reset()
            reply = (
                "\n" + "═" * 50 + "\n"
                "        🔄  MANAGE YOUR APPOINTMENT\n"
                + "═" * 50 + "\n\n"
                " Please click the 'Appointment Button'\n"
                " to open the Appointment Management page.\n\n"
                " There you can:\n"
                "   • Cancel Appointment\n"
                "   • Re-schedule Appointment\n\n"
                " Quick • Simple • Hassle-Free\n\n"
                f"Need anything else, {self.patient_name}?\n"
            )
            self._record(user_input, reply)
            return reply

        # ── LLM decides the route ─────────────────────────────────────────────
        route = self._route(user_input)
        self._last_route = route

        # ── Dispatch ──────────────────────────────────────────────────────────
        if route == "booking":
            reply = self._handle_booking(user_input)

        elif route == "insurance":
            reply = self.insurance_agent.respond(user_input)

        elif route == "report":
            reply = self.report_chatter.chat(
                report_text  = self._report_text,
                question     = user_input,
                history      = self._report_history,
                patient_name = self.patient_name,
            )
            self._report_history.append({"role": "user",      "content": user_input})
            self._report_history.append({"role": "assistant",  "content": reply})
            if len(self._report_history) > 12:
                self._report_history = self._report_history[-12:]

        else:  # "chat"
            reply = self.chat_agent.ask(user_input)

        self._record(user_input, reply)
        return reply

    # ── Booking sub-handler ───────────────────────────────────────────────────
    def _handle_booking(self, user_input: str) -> str:
        booking_reply, is_terminal = self.booking_agent.respond(user_input)
        if is_terminal:
            self.booking_agent.reset()
            return _format_booking_terminal(booking_reply, self.patient_name)
        return booking_reply

    # ── History recorder ──────────────────────────────────────────────────────
    def _record(self, user_input: str, reply: str):
        self._history.append({"role": "user",      "content": user_input})
        self._history.append({"role": "assistant",  "content": reply})

    def close(self):
        self.db.close()


# ─────────────────────────────────────────────────────────────────────────────
#  CLI ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────
def main():
    print("\n" + "═" * 56)
    print("        🏥  HARRAM HOSPITAL AI ASSISTANT")
    print("═" * 56)

    name = input("Enter Patient Name: ").strip()
    if not name:
        print("No name entered. Exiting.")
        return

    router = Router(patient_name=name)

    print(f"\nWelcome, {name}! I'm your hospital assistant.")
    print("I can help with general queries, appointments, insurance, and medical reports.")
    print("\nCommands:")
    print("  Type a message     → chat normally")
    print("  /report <path>     → upload and analyze a PDF report")
    print("                       e.g.  /report CBC_Test.pdf")
    print("  exit / quit / bye  → quit")
    print("─" * 56)

    try:
        while True:
            user_input = input(f"\n{name} > ").strip()

            if not user_input:
                continue

            if user_input.lower() in ("exit", "quit", "bye"):
                print(f"\nGoodbye, {name}! Stay healthy. 👋")
                break

            if user_input.lower().startswith("/report"):
                parts    = user_input.split(maxsplit=1)
                pdf_path = parts[1].strip() if len(parts) > 1 else ""
                if not pdf_path:
                    print("Usage: /report <path to PDF file>")
                    print("Example: /report CBC_Test.pdf")
                    continue
                print("⏳ Analyzing report...")
                summary = router.load_report(pdf_path)
                print(f"\n{summary}")
                continue

            print("⏳ Processing...")
            reply = router.handle(user_input)
            print(f"\n🤖  {reply}")

    finally:
        router.close()


if __name__ == "__main__":
    main()
