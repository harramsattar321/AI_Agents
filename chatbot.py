import json
import re
from groq import Groq
from db import HospitalDB

class HospitalChatbot:
    def __init__(self, mongo_uri, groq_api_key, patient_name):
        self.client = Groq(api_key=groq_api_key)
        self.db = HospitalDB(mongo_uri)
        self.patient_name = patient_name
        self.current_dept = None 

    def ask(self, query):
        q_low = query.lower()

        # ── ENHANCED CONTEXT & DEPARTMENT MAPPING ──
        if any(x in q_low for x in [
            "heart", "cardio", "cardiac", "chest pain", "bp", "blood pressure",
            "heartbeat", "palpitation"
        ]):
            self.current_dept = "Cardiologist"

        elif any(x in q_low for x in [
            "skin", "derm", "rash", "acne", "pimple", "allergy", "itching",
            "eczema", "psoriasis"
        ]):
            self.current_dept = "Dermatologist"

        elif any(x in q_low for x in [
            "kid", "kids", "child", "children", "baby", "infant",
            "growth", "vaccination", "fever child"
        ]):
            self.current_dept = "Pediatrician"

        elif any(x in q_low for x in [
            "brain", "neuro", "headache", "migraine", "tremor",
            "seizure", "epilepsy", "dizziness", "stroke"
        ]):
            self.current_dept = "Neurologist"

        elif any(x in q_low for x in [
            "bone", "joint", "ortho", "fracture", "back pain",
            "knee pain", "shoulder pain", "arthritis"
        ]):
            self.current_dept = "Orthopedic Surgeon"

        elif any(x in q_low for x in [
            "surgery", "operation", "operate", "cut", "appendix",
            "hernia", "gallbladder"
        ]):
            self.current_dept = "General Surgeon"

        tools = [
            {
                "type": "function",
                "function": {
                    "name": "get_doctor_info",
                    "description": "Search for doctors by specialty. USE THIS if the user asks 'how many doctors' or asks for doctors in a specific department.",
                    "parameters": {"type": "object", "properties": {"search_term": {"type": "string"}}},
                },
            },
            # Add to tools list in both chatbot.py AND booking_agent.py
{
    "type": "function",
    "function": {
        "name": "get_available_slots",
        "description": (
            "Get all bookable 15-minute time slots for a specific doctor on a specific date. "
            "Call this when the patient asks 'when is the doctor free', "
            "'what times are available', or 'which slots can I book'. "
            "Returns free slots (0 bookings) and partial slots (some spots left)."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "doctor_id": {
                    "type": "integer",
                    "description": "Doctor ID from get_doctor_info"
                },
                "appointment_date": {
                    "type": "string",
                    "description": "Date in YYYY-MM-DD format"
                }
            },
            "required": ["doctor_id", "appointment_date"]
        }
    }
},
            {
                "type": "function",
                "function": {
                    "name": "get_test_info",
                    "description": "Get test pricing. Leave blank to list all tests.",
                    "parameters": {"type": "object", "properties": {"test_name": {"type": "string"}}},
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "get_department_info",
                    "description": "List departments. Leave blank to list all departments.",
                    "parameters": {"type": "object", "properties": {"search_term": {"type": "string"}}},
                },
            }
        ]

        messages = [
            {
                "role": "system",
                "content": (
                    f"You are the empathetic front-desk manager at Vitual Hospital. Patient: {self.patient_name}. "
                    f"ACTIVE CONTEXT: {self.current_dept or 'General'}. "
                    "1. NO TAGS: Never type <function> or technical tags in your final answer. "
                    "2. NO GUESSING: Only use prices from the tool results. Use Rs. only. "
                    "3. APPOINTMENTS: Always end with: 'I am here for you and Please click the Appointment Button to proceed with booking. It will guide you through the process quickly and easily.' "
                    "4. TEST LIST: If the user asks for 'all tests' or 'how many', call get_test_info with an empty string. "
                    "5. DATABASE ONLY: Use only doctor names and test prices provided in tool results. Do not hallucinate names"
                    "6. IF EMPTY: If the tool returns an empty list [], you MUST say: 'I'm sorry, I don't have a specialist listed for that in my records yet.' "
                    "7. NO HALLUCINATIONS: Never mention names of famous doctor from your own research. Only use the names from the tool."
                    "8. CONTEXT MEMORY: If the user asks a follow-up question like 'and price?' or 'what about timing?', apply it to the test or doctor you were JUST discussing. Do not fetch all records unless explicitly asked to 'list all'."
                )
            },
            {"role": "user", "content": query}
        ]

        # ── Department synonyms ──
        dept_synonyms = {
            "neuro": "Neurologist",
            "neurology": "Neurologist",
            "brain": "Neurologist",
            "child": "Pediatrician",
            "kids": "Pediatrician",
            "baby": "Pediatrician",
            "pediatric": "Pediatrician",
            "ortho": "Orthopedic Surgeon",
            "bone": "Orthopedic Surgeon",
            "joint": "Orthopedic Surgeon",
            "skin": "Dermatologist",
            "derm": "Dermatologist",
            "rash": "Dermatologist",
            "acne": "Dermatologist",
            "surgery": "General Surgeon",
            "operate": "General Surgeon",
            "appendix": "General Surgeon",
            "hernia": "General Surgeon",
            "gallbladder": "General Surgeon",
            "heart": "Cardiologist",
            "cardio": "Cardiologist",
            "cardiac": "Cardiologist",
            "chest": "Cardiologist"
        }

        try:
            response = self.client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=messages,
                tools=tools,
                tool_choice="auto"
            )

            response_message = response.choices[0].message
            tool_calls = response_message.tool_calls

            if tool_calls:
                messages.append(response_message)
                for tool_call in tool_calls:
                    func_name = tool_call.function.name
                    args = json.loads(tool_call.function.arguments)

                    if func_name == "get_doctor_info":
                        raw_val = args.get("search_term", "")
                        if isinstance(raw_val, dict):
                            raw_val = ""
                        else:
                            raw_val = str(raw_val).lower()
                        if any(x in raw_val for x in ["all", "list", "every", "full", "show all", "everything"]):
                            val = ""
                        else:
                            # Map synonyms to official doctor specialties
                            synonym_map = {
                                "heart": "Cardiologist", "cardio": "Cardiologist",
                                "brain": "Neurologist", "neuro": "Neurologist",
                                "skin": "Dermatologist", "derm": "Dermatologist",
                                "child": "Pediatrician", "kids": "Pediatrician", "baby": "Pediatrician",
                                "bone": "Orthopedic Surgeon", "ortho": "Orthopedic Surgeon",
                                "surgery": "General Surgeon", "operation": "General Surgeon"
                            }
                            val = synonym_map.get(raw_val, raw_val)
                        result = self.db.get_doctors(val)

                    elif func_name == "get_test_info":
                        raw_val = args.get("test_name", "")
                        if isinstance(raw_val, dict):
                            raw_val = ""
                        else:
                            raw_val = str(raw_val).lower()
                        # ... (same as your existing test mapping logic)
                        val = "" if any(x in raw_val for x in ["all", "list", "every", "full", "show all"]) else raw_val
                        result = self.db.get_tests(val)

                    elif func_name == "get_available_slots":
                        result = self.db.get_available_slots(
                            doctor_id=args.get("doctor_id"),
                            appointment_date=args.get("appointment_date")
                            )

                    elif func_name == "get_department_info":
                        raw_val = args.get("search_term", "")
                        if isinstance(raw_val, dict):
                            raw_val = ""
                        else:
                            raw_val = str(raw_val).lower()
                        if any(x in raw_val for x in ["all", "list", "everything", "show all"]):
                            val = ""
                        else:
                            # Multi-keyword mapping like doctors/tests
                            keywords = [k for k in re.split(r"\s+", raw_val) if len(k) > 2]
                            val = {"$or": []}
                            for kw in keywords:
                                dept_name = dept_synonyms.get(kw, kw)
                                val["$or"].append({"name": {"$regex": re.escape(dept_name), "$options": "i"}})
                            if not val["$or"]:
                                val = {}
                        result = self.db.get_departments(val)

                    messages.append({"tool_call_id": tool_call.id, "role": "tool", "name": func_name, "content": json.dumps(result, default=str)})

                messages = messages[-5:]
                final_res = self.client.chat.completions.create(model="llama-3.1-8b-instant", messages=messages)
                raw_answer = final_res.choices[0].message.content
            else:
                raw_answer = response_message.content

            return re.sub(r'<.*?>|function=.*?>|\{.*?\}', '', raw_answer).strip()

        except Exception as e:
            return f"I am here for you, but I encountered a slight error. Let's try again! ({str(e)})"