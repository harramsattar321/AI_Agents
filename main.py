"""
main.py — The Router
====================
Three-agent orchestration layer for Harram Hospital AI Assistant.

Agents:
  Agent 1 — HospitalChatbot   (chatbot.py)         General receptionist
  Agent 2 — BookingAgent      (booking_agent.py)   Appointment booking
  Agent 3 — InsuranceAgent    (insurance_agent.py) Insurance / coverage queries

State machine:
    CHAT      ──(booking intent)──►   BOOKING   ──(terminal)──► CHAT
    CHAT      ──(insurance intent)──► INSURANCE ──(done/exit)──► CHAT
    BOOKING   ──(insurance intent)──► INSURANCE
    BOOKING   ──(general intent)───►  CHAT       (BUG FIX #3)
    INSURANCE ──(booking intent)───►  BOOKING
    INSURANCE ──(general intent)───►  CHAT       (BUG FIX #3)

Intent detection — keyword-only (zero tokens, zero latency):
  Booking  : book, appointment, schedule, reserve, consult, etc.
  Insurance: insurance, insur, coverage, claim, EFU, Jubilee, cashless, etc.
  General  : hospital info, timings, location, services, departments, etc.
             (BUG FIX #3 — new detector to route back to general agent)
"""

import re
from chatbot         import HospitalChatbot    # Agent 1
from booking_agent   import BookingAgent       # Agent 2
from insurance_agent import InsuranceAgent     # Agent 3
from db              import HospitalDB
import os
from dotenv import load_dotenv
load_dotenv()

# ─────────────────────────────────────────────────────────────────────────────
#  CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────
MONGO_URI = os.getenv("MONGO_URI")
GROQ_KEY_CHAT = os.getenv("GROQ_KEY_CHAT")
GROQ_KEY_BOOK =os.getenv("GROQ_KEY_BOOK")
GROQ_KEY_INS = os.getenv("GROQ_KEY_INS")
# ─────────────────────────────────────────────────────────────────────────────
#  INTENT DETECTORS  (keyword-only, zero tokens)
# ─────────────────────────────────────────────────────────────────────────────
_BOOKING_RE = re.compile(
    r"\b("
    r"book|appointment|appoint|schedule|reserve"
    r"|fix an? appoint"
    r"|book me|make an? appoint|set up an? appoint"
    r"|visit the doctor"
    r")\b",
    re.IGNORECASE
)

# ── Advisory intent: patient asks WHO to see / WHICH doctor — stays in CHAT ──
# These are recommendation questions, NOT booking requests.
# e.g. "my kid is sick, whom should I consult?" → general agent answers
# e.g. "I want to see a doctor" alone → general agent, NOT booking
_ADVISORY_RE = re.compile(
    r"\b("
    r"whom (?:should|do|to|can|must) i (?:consult|see|visit|go to|contact)"
    r"|who (?:should|can|do) i (?:consult|see|visit)"
    r"|which doctor"
    r"|what (?:doctor|specialist|department)"
    r"|i want to (?:know|find|ask about) (?:a |the )?doctor"
    r"|recommend (?:a |the )?doctor"
    r"|suggest (?:a |the )?doctor"
    r"|consult(?:ation)?"          # "consult" alone → advisory
    r"|i (?:need|want) to see(?: a)?(?: doctor)?"  # "I need to see a doctor" → advisory
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

# ─────────────────────────────────────────────────────────────────────────────
# BUG FIX #3: General intent detector.
# Triggers a switch back to the general receptionist (CHAT state) when the
# user asks about hospital info, services, location, etc. while stuck in
# BOOKING or INSURANCE state.
# ─────────────────────────────────────────────────────────────────────────────
_GENERAL_RE = re.compile(
    r"\b("
    # Hospital info keywords
    r"hospital|clinic|harram"
    r"|timings?|hours?|opening|closing|open"
    r"|location|address|directions?|where (?:is|are)"
    r"|facilities|services|departments?"
    r"|doctors? list|staff|specialists?"
    r"|emergency|ward|icu|lab(?:oratory)?|pharmacy"
    r"|parking|visiting hours?|contact|phone|number"
    # Explicit "go to general" phrases
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
    # Must match booking keywords AND must NOT be a pure advisory/recommendation query
    return bool(_BOOKING_RE.search(text)) and not bool(_ADVISORY_RE.search(text))

def _is_insurance_intent(text: str) -> bool:
    return bool(_INSURANCE_RE.search(text))

def _is_general_intent(text: str) -> bool:
    """
    Returns True only when the message looks like a general hospital query
    AND is NOT also a booking or insurance query.
    This prevents false positives like "what are the available booking slots"
    triggering a switch to general.
    """
    return (
        bool(_GENERAL_RE.search(text))
        and not _is_booking_intent(text)
        and not _is_insurance_intent(text)
    )


# ─────────────────────────────────────────────────────────────────────────────
#  TERMINAL SIGNAL HELPERS  (booking only — insurance has no terminal signal)
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
    """
    Owns one shared DB connection and all three agents.
    Routes each user message to the right agent based on state + intent.
    """

    CHAT      = "CHAT"
    BOOKING   = "BOOKING"
    INSURANCE = "INSURANCE"

    def __init__(self, patient_name: str):
        self.patient_name = patient_name
        self.state        = self.CHAT

        # Shared DB — one connection for all agents
        self.db = HospitalDB(MONGO_URI)

        # Agent 1: General Receptionist
        self.chat_agent = HospitalChatbot(
            mongo_uri    = MONGO_URI,
            groq_api_key = GROQ_KEY_CHAT,
            patient_name = patient_name
        )
        self.chat_agent.db = self.db   # inject shared connection

        # Agent 2: Booking Clerk
        self.booking_agent = BookingAgent(
            groq_api_key_2 = GROQ_KEY_BOOK,
            db             = self.db,
            patient_name   = patient_name
        )

        # Agent 3: Insurance Clerk
        self.insurance_agent = InsuranceAgent(
            groq_api_key = GROQ_KEY_INS,
            mongo_uri    = MONGO_URI,
            patient_name = patient_name
        )

        # Human-readable display log
        self.display_history: list[dict] = []

    # ── Cancel shortcut — works from any state ────────────────────────────────
    _CANCEL_RE = re.compile(
        r"\b(cancel|stop|quit|exit|never ?mind|forget it|go back|main menu)\b",
        re.IGNORECASE
    )

    def _wants_to_cancel(self, text: str) -> bool:
        return bool(self._CANCEL_RE.search(text))

    # ── Single turn dispatcher ────────────────────────────────────────────────
    def handle(self, user_input: str) -> str:
        # Keep display_history bounded to avoid memory bloat
        if len(self.display_history) > 40:
            self.display_history = self.display_history[-40:]

        self.display_history.append({"role": "user", "content": user_input})

        # ── Universal cancel (from BOOKING or INSURANCE back to CHAT) ─────────
        if self.state != self.CHAT and self._wants_to_cancel(user_input):
            self.state = self.CHAT
            self.booking_agent.reset()
            self.insurance_agent.reset()
            reply = (
                f"No problem, {self.patient_name}. "
                f"I'm back as your general receptionist — how can I help you?"
            )
            self.display_history.append({"role": "assistant", "content": reply})
            return reply

        # ══════════════════════════════════════════════════════════════════════
        #  STATE: CHAT
        # ══════════════════════════════════════════════════════════════════════
        if self.state == self.CHAT:

            # CHAT → BOOKING
            if _is_booking_intent(user_input):
                self.state = self.BOOKING
                self.booking_agent.reset()
                booking_reply, is_terminal = self.booking_agent.respond(user_input)
                reply = self._wrap_booking(booking_reply, is_terminal)

            # CHAT → INSURANCE
            elif _is_insurance_intent(user_input):
                self.state = self.INSURANCE
                self.insurance_agent.reset()
                ins_reply = self.insurance_agent.respond(user_input)
                reply = f"[INSURANCE DESK]: {ins_reply}"

            # Stay in CHAT (general query)
            else:
                reply = self.chat_agent.ask(user_input)

        # ══════════════════════════════════════════════════════════════════════
        #  STATE: BOOKING
        # ══════════════════════════════════════════════════════════════════════
        elif self.state == self.BOOKING:

            # ─────────────────────────────────────────────────────────────────
            # BUG FIX #3: General intent while in BOOKING → return to CHAT.
            # Example triggers: "tell me about the hospital", "what are your
            # timings", "where are you located", "something else".
            # ─────────────────────────────────────────────────────────────────
            if _is_general_intent(user_input):
                self.state = self.CHAT
                self.booking_agent.reset()
                bridge = (
                    f"Sure, {self.patient_name}! Let me pass you back to our "
                    f"general receptionist.\n{'─' * 48}"
                )
                chat_reply = self.chat_agent.ask(user_input)
                reply = f"{bridge}\n{chat_reply}"

            # BOOKING → INSURANCE (patient asks about insurance mid-booking)
            elif _is_insurance_intent(user_input):
                self.state = self.INSURANCE
                self.booking_agent.reset()
                bridge = (
                    f"Of course, {self.patient_name}! Let me connect you with our "
                    f"Insurance Desk. Your booking session has been paused.\n{'─' * 48}"
                )
                ins_reply = self.insurance_agent.respond(user_input)
                reply = f"{bridge}\n[INSURANCE DESK]: {ins_reply}"

            # Continue BOOKING conversation
            else:
                booking_reply, is_terminal = self.booking_agent.respond(user_input)
                reply = self._wrap_booking(booking_reply, is_terminal)

        # ══════════════════════════════════════════════════════════════════════
        #  STATE: INSURANCE
        # ══════════════════════════════════════════════════════════════════════
        elif self.state == self.INSURANCE:

            # ─────────────────────────────────────────────────────────────────
            # BUG FIX #3: General intent while in INSURANCE → return to CHAT.
            # Example triggers: "tell me about the hospital", "what facilities
            # do you have", "what are your opening hours", "something else".
            # ─────────────────────────────────────────────────────────────────
            if _is_general_intent(user_input):
                self.state = self.CHAT
                self.insurance_agent.reset()
                bridge = (
                    f"Sure, {self.patient_name}! Let me pass you back to our "
                    f"general receptionist.\n{'─' * 48}"
                )
                chat_reply = self.chat_agent.ask(user_input)
                reply = f"{bridge}\n{chat_reply}"

            # INSURANCE → BOOKING (patient wants to book mid-insurance)
            elif _is_booking_intent(user_input):
                self.state = self.BOOKING
                self.booking_agent.reset()
                bridge = (
                    f"Sure, {self.patient_name}! Let me connect you with our "
                    f"Booking Clerk.\n{'─' * 48}"
                )
                booking_reply, is_terminal = self.booking_agent.respond(user_input)
                reply = (
                    f"{bridge}\n[BOOKING CLERK]: {booking_reply}"
                    if not is_terminal
                    else self._wrap_booking(booking_reply, is_terminal)
                )

            # Continue INSURANCE conversation
            else:
                ins_reply = self.insurance_agent.respond(user_input)

                # Detect natural conversation endings → return to CHAT
                if re.search(
                    r"\b(thank(?:s| you)|that'?s? all|no more questions?|got it|perfect|great|bye)\b",
                    user_input, re.IGNORECASE
                ):
                    self.state = self.CHAT
                    self.insurance_agent.reset()
                    reply = (
                        f"[INSURANCE DESK]: {ins_reply}\n\n"
                        f"Feel free to ask me anything else, {self.patient_name}! "
                        f"I'm back as your general receptionist."
                    )
                else:
                    reply = f"[INSURANCE DESK]: {ins_reply}"

        else:
            # Safety fallback — should never be reached
            reply = self.chat_agent.ask(user_input)

        self.display_history.append({"role": "assistant", "content": reply})
        return reply

    # ── Booking terminal wrapper ──────────────────────────────────────────────
    def _wrap_booking(self, reply: str, is_terminal: bool) -> str:
        if is_terminal:
            self.state = self.CHAT
            self.booking_agent.reset()
            return _format_booking_terminal(reply, self.patient_name)
        return f"[BOOKING CLERK]: {reply}"

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
    print("Type 'exit' to quit, or 'cancel' / 'go back' at any time.")
    print("\nTips:")
    print("  • Ask 'which doctor for X?' or 'whom should I consult?' → general advice")
    print("  • Say 'book appointment' or 'schedule appointment' → booking desk")
    print("  • Say 'tell me about the hospital' / 'what are timings' → general desk")
    print("─" * 56)

    _STATE_LABELS = {
        Router.CHAT:      "[CHAT]     ",
        Router.BOOKING:   "[BOOKING]  ",
        Router.INSURANCE: "[INSURANCE]",
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

            print("⏳ Processing...")
            reply = router.handle(user_input)
            print(f"\n🤖  {reply}")

    finally:
        router.close()


if __name__ == "__main__":
    main()