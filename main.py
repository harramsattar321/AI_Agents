"""
main.py — The Router
====================
Three-agent orchestration layer for Harram Hospital AI Assistant.

Agents:
  Agent 1 — HospitalChatbot   (chatbot.py)         General receptionist
  Agent 2 — BookingAgent      (booking_agent.py)   Appointment booking
  Agent 3 — InsuranceAgent    (insurance_agent.py) Insurance / coverage queries
  Agent 4 — ReportChatter     (report_chatter.py)  Medical report Q&A

State machine:
    CHAT      ──(booking intent)──►   BOOKING   ──(terminal)──► CHAT
    CHAT      ──(insurance intent)──► INSURANCE ──(done/exit)──► CHAT
    CHAT      ──(report loaded)────► REPORT    ──(exit report)──► CHAT
    BOOKING   ──(insurance intent)──► INSURANCE
    BOOKING   ──(general intent)───►  CHAT
    INSURANCE ──(booking intent)───►  BOOKING
    INSURANCE ──(general intent)───►  CHAT
"""

import re
import os
import io
import pdfplumber

from chatbot         import HospitalChatbot    # Agent 1
from booking_agent   import BookingAgent       # Agent 2
from insurance_agent import InsuranceAgent     # Agent 3
from report_chatter  import ReportChatter      # Agent 4
from report_analyzer import ReportAnalyzer     # PDF summarizer
from db              import HospitalDB
from dotenv import load_dotenv
load_dotenv()

# ─────────────────────────────────────────────────────────────────────────────
#  CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────
MONGO_URI     = os.getenv("MONGO_URI")
GROQ_KEY_CHAT = os.getenv("GROQ_KEY_CHAT")
GROQ_KEY_BOOK = os.getenv("GROQ_KEY_BOOK")
GROQ_KEY_INS  = os.getenv("GROQ_KEY_INS")
GROQ_KEY_RPT  = os.getenv("GROQ_KEY_REPORT")

# ─────────────────────────────────────────────────────────────────────────────
#  INTENT DETECTORS
# ─────────────────────────────────────────────────────────────────────────────
_CANCEL_BOOKING_RE = re.compile(
    r"\b(cancel appointment|cancel my appointment|reschedule|re.?schedule|change appointment|reappointment)\b",
    re.IGNORECASE
)

_BOOKING_RE = re.compile(
    r"\b("
    r"book|appointment|appoint|schedule|reserve"
    r"|fix an? appoint"
    r"|book me|make an? appoint|set up an? appoint"
    r"|visit the doctor"
    r")\b",
    re.IGNORECASE
)

_ADVISORY_RE = re.compile(
    r"\b("
    r"whom (?:should|do|to|can|must) i (?:consult|see|visit|go to|contact)"
    r"|who (?:should|can|do) i (?:consult|see|visit)"
    r"|which doctor"
    r"|what (?:doctor|specialist|department)"
    r"|i want to (?:know|find|ask about) (?:a |the )?doctor"
    r"|recommend (?:a |the )?doctor"
    r"|suggest (?:a |the )?doctor"
    r"|consult(?:ation)?"
    r"|i (?:need|want) to see(?: a)?(?: doctor)?"
    r")\b",
    re.IGNORECASE
)

_INSURANCE_RE = re.compile(
    r"\b("
    r"insur(?:ance|ed|er)?|coverage|cover|claim"
    r"|efu|jubilee|adamjee|state life"
    r"|cashless|panel|in.?network|out.?of.?network"
    r"|policy|policies|premium|deductible|co.?pay"
    r"|reimburs(?:e|ement)|health.?plan|health plan"
    r"|sehat|rahbar|mukammal|lifestyle care|personal health"
    r"|admission fee|file fee"
    r")\b",
    re.IGNORECASE
)

_GENERAL_RE = re.compile(
    r"\b("
    r"hospital|clinic|harram"
    r"|timings?|hours?|opening|closing|open"
    r"|location|address|directions?|where (?:is|are)"
    r"|facilities|services|departments?"
    r"|doctors? list|staff|specialists?"
    r"|emergency|ward|icu|lab(?:oratory)?|pharmacy"
    r"|parking|visiting hours?|contact|phone|number"
    r"|something else|other question|different question"
    r"|general query|general question|main menu|receptionist"
    r"|go back to|take me back|return to"
    r"|forget (?:the )?(?:booking|insurance|it)"
    r"|not about (?:booking|insurance|appointment)"
    r"|tell me about (?:the )?hospital"
    r"|info(?:rmation)? about"
    r"|what (?:do you offer|can you do|are the)"
    r")\b",
    re.IGNORECASE
)

def _is_booking_intent(text: str) -> bool:
    return bool(_BOOKING_RE.search(text)) and not bool(_ADVISORY_RE.search(text))

def _is_insurance_intent(text: str) -> bool:
    return bool(_INSURANCE_RE.search(text))

def _is_general_intent(text: str) -> bool:
    return (
        bool(_GENERAL_RE.search(text))
        and not _is_booking_intent(text)
        and not _is_insurance_intent(text)
    )

def _is_cancel_or_reschedule(text: str) -> bool:
    return bool(_CANCEL_BOOKING_RE.search(text))

# ─────────────────────────────────────────────────────────────────────────────
#  TERMINAL SIGNAL HELPERS
# ─────────────────────────────────────────────────────────────────────────────
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
#  ROUTER
# ─────────────────────────────────────────────────────────────────────────────
class Router:

    CHAT      = "CHAT"
    BOOKING   = "BOOKING"
    INSURANCE = "INSURANCE"
    REPORT    = "REPORT"

    def __init__(self, patient_name: str):
        self.patient_name = patient_name
        self.state        = self.CHAT

        self.db = HospitalDB(MONGO_URI)

        self.chat_agent = HospitalChatbot(
            mongo_uri    = MONGO_URI,
            groq_api_key = GROQ_KEY_CHAT,
            patient_name = patient_name
        )
        self.chat_agent.db = self.db

        self.booking_agent = BookingAgent(
            groq_api_key_2 = GROQ_KEY_BOOK,
            db             = self.db,
            patient_name   = patient_name
        )

        self.insurance_agent = InsuranceAgent(
            groq_api_key = GROQ_KEY_INS,
            mongo_uri    = MONGO_URI,
            patient_name = patient_name
        )

        self.report_chatter = ReportChatter(
            groq_api_key  = GROQ_KEY_RPT,
            db            = self.db,
            mongo_uri     = MONGO_URI,
            groq_key_chat = GROQ_KEY_CHAT,
        )

        self.report_analyzer = ReportAnalyzer(groq_api_key=GROQ_KEY_RPT)

        # Report session state
        self._report_text:    str        = ""
        self._report_history: list[dict] = []

        self.display_history: list[dict] = []

    # ── Cancel shortcut ───────────────────────────────────────────────────────
    _CANCEL_RE = re.compile(
        r"\b(cancel|stop|quit|exit|never ?mind|forget it|go back|main menu)\b",
        re.IGNORECASE
    )

    def _wants_to_cancel(self, text: str) -> bool:
        return bool(self._CANCEL_RE.search(text))

    # ── Load a PDF report ─────────────────────────────────────────────────────
    def load_report(self, pdf_path: str) -> str:
        """
        Analyze a PDF file, print the summary, and switch to REPORT state.
        Returns the summary string.
        """
        if not os.path.isfile(pdf_path):
            return f"❌ File not found: {pdf_path}"

        with open(pdf_path, "rb") as f:
            file_bytes = f.read()

        # Extract text
        try:
            with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
                self._report_text = "\n".join(
                    p.extract_text() or "" for p in pdf.pages
                ).strip()
        except Exception as e:
            return f"❌ Could not read PDF: {e}"

        if not self._report_text:
            return "❌ Could not extract text from this PDF (may be a scanned image)."

        # Summarize
        print("⏳ Analyzing report...")

        class _FakeFile:
            """Wraps raw bytes to look like a Flask file upload."""
            def __init__(self, data, name):
                self.filename = name
                self._data    = data
                self._stream  = io.BytesIO(data)
            def read(self):   return self._stream.read()
            def seek(self, n): self._stream.seek(n)

        fake_file = _FakeFile(file_bytes, os.path.basename(pdf_path))
        success, _, result = self.report_analyzer.analyze(fake_file)

        if not success:
            return f"❌ Analysis failed: {result}"

        # Switch to REPORT state
        self.state = self.REPORT
        self._report_history = []

        # Format summary for terminal
        lines = [
            "",
            "═" * 56,
            "  📋  REPORT ANALYSIS",
            "═" * 56,
            f"  Type    : {result.get('report_type', 'Unknown')}",
            f"  Summary : {result.get('summary', '')}",
        ]

        abnormal = result.get("abnormal_values", [])
        if abnormal:
            lines.append("\n  ⚠️  Abnormal Values:")
            for v in abnormal:
                lines.append(
                    f"     • {v['name']}: {v['value']} "
                    f"(normal: {v['normal_range']}) [{v['status']}]"
                )

        observations = result.get("key_observations", [])
        if observations:
            lines.append("\n  🔍  Key Observations:")
            for obs in observations:
                lines.append(f"     • {obs}")

        advice = result.get("advice", "")
        if advice:
            lines.append(f"\n  💡  Advice : {advice}")

        lines += [
            "",
            f"  {result.get('disclaimer', '')}",
            "═" * 56,
            "",
            "  You are now in REPORT mode.",
            "  Ask me anything about this report.",
            "  Type 'exit report' to return to general chat.",
            "═" * 56,
        ]

        return "\n".join(lines)

    # ── Single turn dispatcher ────────────────────────────────────────────────
    def handle(self, user_input: str) -> str:
        if len(self.display_history) > 40:
            self.display_history = self.display_history[-40:]

        self.display_history.append({"role": "user", "content": user_input})

        # ── Universal cancel ──────────────────────────────────────────────────
        if self.state != self.CHAT and self._wants_to_cancel(user_input):
            self.state = self.CHAT
            self.booking_agent.reset()
            self.insurance_agent.reset()
            self._report_text    = ""
            self._report_history = []
            reply = (
                f"No problem, {self.patient_name}. "
                f"I'm back as your general receptionist — how can I help you?"
            )
            self.display_history.append({"role": "assistant", "content": reply})
            return reply

        # ══════════════════════════════════════════════════════════════════════
        #  STATE: REPORT
        # ══════════════════════════════════════════════════════════════════════
        if self.state == self.REPORT:

            # Exit report mode explicitly
            if re.search(r"\b(exit report|leave report|done with report|back to chat)\b",
                         user_input, re.IGNORECASE):
                self.state           = self.CHAT
                self._report_text    = ""
                self._report_history = []
                reply = (
                    f"You've exited report mode, {self.patient_name}. "
                    f"I'm back as your general receptionist — how can I help you?"
                )

            else:
                reply = self.report_chatter.chat(
                    report_text  = self._report_text,
                    question     = user_input,
                    history      = self._report_history,
                    patient_name = self.patient_name,
                )
                # Update report conversation history
                self._report_history.append({"role": "user",      "content": user_input})
                self._report_history.append({"role": "assistant",  "content": reply})
                if len(self._report_history) > 12:
                    self._report_history = self._report_history[-12:]

        # ══════════════════════════════════════════════════════════════════════
        #  STATE: CHAT
        # ══════════════════════════════════════════════════════════════════════
        elif self.state == self.CHAT:

            if _is_cancel_or_reschedule(user_input):
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

            elif _is_booking_intent(user_input):
                reply = (
                    "\n" + "═" * 50 + "\n"
                    "            APPOINTMENT BOOKING\n"
                    + "═" * 50 + "\n\n"
                    "Please click the 'Appointment Button' to begin.\n\n"
                    "Follow these simple steps:\n"
                    "   1️⃣ Select your preferred doctor\n"
                    "   2️⃣ Choose a suitable date\n"
                    "   3️⃣ Pick an available time slot\n"
                    "   4️⃣ Click 'Confirm Appointment'\n\n"
                    " Your appointment will be successfully scheduled.\n\n"
                    "⚡ Fast • Easy • Secure\n\n"
                    f"How else can I assist you, {self.patient_name}?\n"
                )

            elif _is_insurance_intent(user_input):
                self.state = self.INSURANCE
                self.insurance_agent.reset()
                reply = self.insurance_agent.respond(user_input)

            else:
                reply = self.chat_agent.ask(user_input)

        # ══════════════════════════════════════════════════════════════════════
        #  STATE: BOOKING
        # ══════════════════════════════════════════════════════════════════════
        elif self.state == self.BOOKING:

            if _is_general_intent(user_input):
                self.state = self.CHAT
                self.booking_agent.reset()
                reply = self.chat_agent.ask(user_input)

            elif _is_insurance_intent(user_input):
                self.state = self.INSURANCE
                self.booking_agent.reset()
                reply = self.insurance_agent.respond(user_input)

            else:
                booking_reply, is_terminal = self.booking_agent.respond(user_input)
                reply = self._wrap_booking(booking_reply, is_terminal)

        # ══════════════════════════════════════════════════════════════════════
        #  STATE: INSURANCE
        # ══════════════════════════════════════════════════════════════════════
        elif self.state == self.INSURANCE:

            if _is_general_intent(user_input):
                self.state = self.CHAT
                self.insurance_agent.reset()
                reply = self.chat_agent.ask(user_input)

            elif _is_booking_intent(user_input):
                self.state = self.BOOKING
                self.booking_agent.reset()
                booking_reply, is_terminal = self.booking_agent.respond(user_input)
                reply = (
                    booking_reply
                    if not is_terminal
                    else self._wrap_booking(booking_reply, is_terminal)
                )

            else:
                ins_reply = self.insurance_agent.respond(user_input)
                if re.search(
                    r"\b(thank(?:s| you)|that'?s? all|no more questions?|got it|perfect|great|bye)\b",
                    user_input, re.IGNORECASE
                ):
                    self.state = self.CHAT
                    self.insurance_agent.reset()
                    reply = (
                        f"{ins_reply}\n\n"
                        f"Feel free to ask me anything else, {self.patient_name}! "
                        f"I'm back as your general receptionist."
                    )
                else:
                    reply = ins_reply

        else:
            reply = self.chat_agent.ask(user_input)

        self.display_history.append({"role": "assistant", "content": reply})
        return reply

    def _wrap_booking(self, reply: str, is_terminal: bool) -> str:
        if is_terminal:
            self.state = self.CHAT
            self.booking_agent.reset()
            return _format_booking_terminal(reply, self.patient_name)
        return reply

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
    print("I can help with general queries, appointments, and insurance.")
    print("\nCommands:")
    print("  Type a message       → chat normally")
    print("  /report <path>       → upload and analyze a PDF report")
    print("                         e.g.  /report CBC_Test.pdf")
    print("  exit report          → leave report mode, return to general chat")
    print("  exit / quit / bye    → quit the program")
    print("─" * 56)

    _STATE_LABELS = {
        Router.CHAT:      "[CHAT]     ",
        Router.BOOKING:   "[BOOKING]  ",
        Router.INSURANCE: "[INSURANCE]",
        Router.REPORT:    "[REPORT]   ",
    }

    try:
        while True:
            tag        = _STATE_LABELS.get(router.state, "[CHAT]     ")
            user_input = input(f"\n{tag} {name} > ").strip()

            if not user_input:
                continue

            if user_input.lower() in ("exit", "quit", "bye"):
                print(f"\nGoodbye, {name}! Stay healthy. 👋")
                break

            # ── /report <path> command ────────────────────────────────────────
            if user_input.lower().startswith("/report"):
                parts    = user_input.split(maxsplit=1)
                pdf_path = parts[1].strip() if len(parts) > 1 else ""
                if not pdf_path:
                    print("Usage: /report <path to PDF file>")
                    print("Example: /report CBC_Test.pdf")
                    continue
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