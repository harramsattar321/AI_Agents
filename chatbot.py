"""
chatbot.py — HospitalChatbot (General Receptionist)
====================================================
Keyword detection decides which DB method to call.
DB results are injected into the LLM prompt as plain context.
LLM formats a natural language reply from that context only.
"""

import re
from groq import Groq

from db import HospitalDB


class HospitalChatbot:

    MODEL = "llama-3.3-70b-versatile"

    def __init__(self, mongo_uri: str, groq_api_key: str, patient_name: str):
        self.client       = Groq(api_key=groq_api_key)
        self.db           = HospitalDB(mongo_uri)
        self.patient_name = patient_name

    # ── Public entry point ────────────────────────────────────────────────────

    def ask(self, query: str) -> str:
        context = self._fetch_context(query)
        return self._generate_reply(query, context)

    # ── Step 1: DB fetch ──────────────────────────────────────────────────────

    def _fetch_context(self, query: str) -> str:
        q = query.lower()

        # Doctor name lookup
        name_match = re.search(
            r'\bdr\.?\s+([a-zA-Z]+(?:\s+[a-zA-Z]+){0,2})',
            query, re.IGNORECASE
        )
        if name_match:
            name    = name_match.group(1).strip()
            doctors = self.db.get_doctors(search_term=name)
            if doctors:
                return self._format_doctors(doctors)

        # Specialty / department keywords
        specialty = self._detect_specialty(q)
        if specialty:
            doctors = self.db.get_doctors(search_term=specialty)
            return self._format_doctors(doctors) if doctors else f"No {specialty} found in records."

        # General doctor list
        if any(x in q for x in [
            "doctor", "doctors", "specialist", "physician",
            "who do you have", "available doctor", "list doctor",
            "how many doctor", "all doctor", "your doctor"
        ]):
            doctors = self.db.get_doctors()
            return self._format_doctors(doctors) if doctors else "No doctors found in records."

        # Test / lab pricing
        if any(x in q for x in ["test", "lab", "price", "cost", "fee", "cbc",
                                  "blood test", "urine", "xray", "x-ray"]):
            test_name = self._extract_test_name(q)
            tests     = self.db.get_tests(test_name)
            return self._format_tests(tests) if tests else "No tests found in records."

        # Departments
        if any(x in q for x in ["department", "ward", "unit", "section"]):
            departments = self.db.get_departments()
            return self._format_departments(departments) if departments else "No departments found."

        return ""

    # ── Step 2: LLM reply ─────────────────────────────────────────────────────

    def _generate_reply(self, query: str, context: str) -> str:
        context_block = f"\n\nHOSPITAL DATABASE RESULTS:\n{context}\n" if context else ""

        system = (
            f"You are the front-desk receptionist at Harram Hospital. "
            f"Patient name: {self.patient_name}.\n\n"

            "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            "WHAT YOU CAN DO — nothing else, ever:\n"
            "  1. Answer questions about doctors, specialties, fees, and timings\n"
            "  2. Answer questions about hospital departments, services, and facilities\n"
            "  3. Answer questions about lab tests and their prices\n"
            "  4. Provide general hospital information (location, hours, contact)\n"
            "  5. Direct the patient to the right section of the app\n\n"

            "WHAT YOU CANNOT DO — never suggest, offer, or imply these:\n"
            "  ✗ Book, cancel, or reschedule appointments\n"
            "  ✗ Call or contact any doctor or staff on the patient's behalf\n"
            "  ✗ Contact any insurance company or manager\n"
            "  ✗ Send emails, messages, or make calls of any kind\n"
            "  ✗ Access or retrieve any patient records\n"
            "  ✗ Perform any action outside of providing information\n\n"

            "If asked for something outside this list, say:\n"
            "  'I'm not able to do that, but I can point you in the right direction.\n"
            "   For [their request], please [visit reception / use the booking section / "
            "contact the billing desk].'\n"
            "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"

            "STRICT DATA RULES:\n"
            "1. Only use information from HOSPITAL DATABASE RESULTS below.\n"
            "2. Never invent doctor names, prices, timings, or services.\n"
            "3. If the database has no result, tell the patient politely.\n"
            "4. Use Rs. for all prices.\n"
            "5. Never output raw JSON, function tags, or technical syntax.\n"
            "6. Be warm, helpful, and concise.\n"
            f"{context_block}"
        )

        messages = [
            {"role": "system", "content": system},
            {"role": "user",   "content": query},
        ]

        try:
            response = self.client.chat.completions.create(
                model       = self.MODEL,
                messages    = messages,
                temperature = 0.3,
                max_tokens  = 512,
            )
            raw = response.choices[0].message.content or ""
            return re.sub(r'<[^>]+>', '', raw).strip()
        except Exception as e:
            return f"I'm sorry, I encountered an error. Please try again. ({e})"

    # ── Specialty detector ────────────────────────────────────────────────────

    def _detect_specialty(self, q: str) -> str:
        mapping = [
            (["cardiolog", "heart", "cardio", "cardiac", "chest pain",
              "palpitation", "blood pressure"], "Cardiologist"),
            (["neurolog", "brain", "neuro", "headache", "migraine",
              "seizure", "epilepsy", "stroke", "dizziness"], "Neurologist"),
            (["pediatric", "child", "kids", "baby", "infant",
              "vaccination"], "Pediatrician"),
            (["dermatolog", "skin", "derm", "rash", "acne",
              "eczema", "psoriasis", "itching"], "Dermatologist"),
            (["orthopedic", "ortho", "bone", "joint", "fracture",
              "back pain", "knee", "shoulder", "arthritis"], "Orthopedic Surgeon"),
            (["general surgeon", "surgery", "operation",
              "appendix", "hernia", "gallbladder"], "General Surgeon"),
            (["general physician", "general doctor", "gp",
              "hematolog", "blood doctor", "anemia"], "General Physician"),
            (["psychiatr", "mental", "psycholog",
              "anxiety", "depression"], "Psychiatrist"),
            (["gynecolog", "obstetr", "women",
              "pregnancy", "maternity"], "Gynecologist"),
        ]
        for keywords, specialty in mapping:
            if any(kw in q for kw in keywords):
                return specialty
        return ""

    def _extract_test_name(self, q: str) -> str:
        known = [
            "cbc", "complete blood count", "urine", "urine analysis",
            "xray", "x-ray", "mri", "ct scan", "ultrasound", "ecg",
            "blood sugar", "glucose", "cholesterol", "liver function",
            "kidney function", "thyroid", "hepatitis"
        ]
        for t in known:
            if t in q:
                return t
        return ""

    # ── Context formatters ────────────────────────────────────────────────────

    def _format_doctors(self, doctors: list) -> str:
        if not doctors:
            return "No doctors found."
        lines = []
        for d in doctors:
            name       = d.get("name", "N/A")
            specialty  = d.get("specialty", "")
            dept       = d.get("department", "")
            timings    = d.get("timings", "") or d.get("schedule", "")
            experience = d.get("experience", "")
            fee        = d.get("fee", "") or d.get("consultationFee", "")

            line = f"- Dr. {name}"
            if specialty:  line += f" | Specialty: {specialty}"
            if dept:       line += f" | Dept: {dept}"
            if timings:    line += f" | Timings: {timings}"
            if experience: line += f" | Experience: {experience} years"
            if fee:        line += f" | Fee: Rs. {fee}"
            lines.append(line)
        return "\n".join(lines)

    def _format_tests(self, tests: list) -> str:
        if not tests:
            return "No tests found."
        lines = []
        for t in tests:
            name  = t.get("name", "N/A")
            price = t.get("price", "") or t.get("fee", "")
            line  = f"- {name}"
            if price: line += f": Rs. {price}"
            lines.append(line)
        return "\n".join(lines)

    def _format_departments(self, departments: list) -> str:
        if not departments:
            return "No departments found."
        return "\n".join(f"- {d.get('name', 'N/A')}" for d in departments)
