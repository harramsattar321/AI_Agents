"""
report_chatter.py — Medical Report Q&A Agent
=============================================
The LLM decides routing — no fragile regex intent detection.

The system prompt instructs the model to return one of three signal types:
  1. {"action": "recommend_doctor", ...}  → fetch doctors from DB, format reply
  2. {"action": "hospital_query"}          → delegate to HospitalChatbot
  3. Plain text                            → answer from report

This handles conversational follow-ups correctly because the LLM has
the full conversation history and understands context.
"""

import json
import re
import requests
from chatbot import HospitalChatbot
from db import HospitalDB


# ─────────────────────────────────────────────────────────────────────────────
#  SYSTEM PROMPT
# ─────────────────────────────────────────────────────────────────────────────

_SYSTEM_PROMPT = """You are a medical report assistant for Harram Hospital.
A patient has uploaded their medical report and you help them understand it.

You also act as a smart router. For every patient message, decide which of
these three responses to give:

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CASE 1 — REPORT QUESTION
The patient asks about their report values, symptoms, severity, what is
abnormal, what something means, or whether something is serious.
→ Answer in plain conversational English using ONLY the report text.
→ Do NOT invent values or ranges not in the report.
→ End with: "⚠️ This is informational only — please consult your physician for a formal diagnosis."

CASE 2 — DOCTOR RECOMMENDATION
The patient asks which doctor or specialist to see based on their report,
or who to contact for their specific condition.
→ Respond with ONLY this JSON (no other text):
{"action": "recommend_doctor", "specialty": "<specialty>", "reason": "<one sentence from report findings>"}

CASE 3 — HOSPITAL QUERY
The patient asks anything about the hospital itself that is NOT answered
by the report — for example:
  • Who is Dr. X? / Tell me about Dr. X
  • What are the timings of Dr. X?
  • Can I book an appointment?
  • Do you have a [specialty] doctor?
  • What are your opening hours / location / services?
  • Any follow-up to a hospital question (e.g. patient just said "General Physician"
    after being asked about specialty — that context means they want hospital info)
→ Respond with ONLY this JSON (no other text):
{"action": "hospital_query"}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

CRITICAL RULES:
- Never mix JSON signals with plain text in the same response.
- Never answer hospital questions from the report text — always signal hospital_query.
- Never answer report questions with a JSON signal.
- Use conversation history to understand follow-up context.
"""

MAX_HISTORY_TURNS = 6
MAX_REPORT_CHARS  = 6000


# ─────────────────────────────────────────────────────────────────────────────
#  AGENT
# ─────────────────────────────────────────────────────────────────────────────

class ReportChatter:

    GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
    GROQ_MODEL   = "llama-3.3-70b-versatile"

    def __init__(
        self,
        groq_api_key:  str,
        db:            HospitalDB,
        mongo_uri:     str,
        groq_key_chat: str,
    ):
        self.groq_api_key  = groq_api_key
        self.db            = db
        self.mongo_uri     = mongo_uri
        self.groq_key_chat = groq_key_chat

    def _get_chatbot(self, patient_name: str) -> HospitalChatbot:
        return HospitalChatbot(
            mongo_uri    = self.mongo_uri,
            groq_api_key = self.groq_key_chat,
            patient_name = patient_name,
        )

    # ── Public entry point ────────────────────────────────────────────────────

    def chat(
        self,
        report_text:  str,
        question:     str,
        history:      list[dict] | None = None,
        patient_name: str = "there",
    ) -> str:
        history = (history or [])[-MAX_HISTORY_TURNS * 2:]

        if len(report_text) > MAX_REPORT_CHARS:
            report_text = report_text[:MAX_REPORT_CHARS] + "\n\n[Report truncated]"

        system_msg = (
            _SYSTEM_PROMPT
            + f"\n\n--- PATIENT'S MEDICAL REPORT ---\n{report_text}\n--- END OF REPORT ---"
        )

        messages = [{"role": "system", "content": system_msg}]
        messages.extend(history)
        messages.append({"role": "user", "content": question})

        raw = self._call_groq(messages)

        # ── Parse LLM signal ──────────────────────────────────────────────────
        action = self._try_parse_action(raw)

        if action:
            kind = action.get("action")

            # Case 2: Doctor recommendation
            if kind == "recommend_doctor":
                specialty = action.get("specialty", "specialist")
                reason    = action.get("reason", "")
                doctors   = self.db.get_doctors(search_term=specialty)
                return self._format_doctor_reply(specialty, reason, doctors, patient_name)

            # Case 3: Hospital query → delegate to HospitalChatbot
            if kind == "hospital_query":
                return self._get_chatbot(patient_name).ask(question)

        # Case 1: Plain report answer
        return raw

    # ── Groq call ─────────────────────────────────────────────────────────────

    def _call_groq(self, messages: list[dict]) -> str:
        headers = {
            "Authorization": f"Bearer {self.groq_api_key}",
            "Content-Type":  "application/json",
        }
        payload = {
            "model":       self.GROQ_MODEL,
            "messages":    messages,
            "temperature": 0.2,
            "max_tokens":  512,
        }
        response = requests.post(
            self.GROQ_API_URL, headers=headers, json=payload, timeout=30
        )
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"].strip()

    # ── Action parser ─────────────────────────────────────────────────────────

    def _try_parse_action(self, text: str) -> dict | None:
        """Extract any JSON action signal from the LLM response."""
        try:
            clean = re.sub(r"```(?:json)?|```", "", text).strip()
            # Find first JSON object in the text
            match = re.search(r'\{[^{}]*"action"\s*:[^{}]*\}', clean, re.DOTALL)
            if match:
                data = json.loads(match.group())
                if isinstance(data, dict) and "action" in data:
                    return data
        except (json.JSONDecodeError, ValueError):
            pass
        return None

    # ── Doctor reply formatter ────────────────────────────────────────────────

    def _format_doctor_reply(
        self,
        specialty:    str,
        reason:       str,
        doctors:      list[dict],
        patient_name: str,
    ) -> str:
        lines = [
            f"Based on your report, {patient_name}, you should consult a {specialty}.",
            f"{reason}",
            "",
        ]

        if doctors:
            lines.append(
                f"We have the following {specialty}(s) available at Harram Hospital:\n"
            )
            for doc in doctors:
                name       = doc.get("name", "N/A")
                department = doc.get("department", "")
                timings    = doc.get("timings", "") or doc.get("schedule", "")
                line       = f"  • Dr. {name}"
                if department:
                    line += f" ({department})"
                if timings:
                    line += f" — {timings}"
                lines.append(line)
        else:
            # Fallback to general physician
            fallback_doctors = []
            for fallback in ["General Physician", "Internal Medicine", "General"]:
                fallback_doctors = self.db.get_doctors(search_term=fallback)
                if fallback_doctors:
                    break

            if fallback_doctors:
                lines.append(
                    f"We don't currently have a {specialty} listed, but a General Physician "
                    f"can evaluate you and refer you further if needed. "
                    f"Here are our available doctors:\n"
                )
                for doc in fallback_doctors:
                    name       = doc.get("name", "N/A")
                    department = doc.get("department", "")
                    timings    = doc.get("timings", "") or doc.get("schedule", "")
                    line       = f"  • Dr. {name}"
                    if department:
                        line += f" ({department})"
                    if timings:
                        line += f" — {timings}"
                    lines.append(line)
            else:
                lines.append(
                    f"We don't currently have a {specialty} listed. "
                    f"Please visit our reception — our team will refer you to the right specialist."
                )

        lines += [
            "",
            "⚠️ This is informational only — please consult your physician for a formal diagnosis.",
        ]
        return "\n".join(lines)