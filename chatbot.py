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

        # ── Department context detection ───────────────────────────────────────
        if any(x in q_low for x in ["heart", "cardio", "cardiac", "chest pain", "bp", "blood pressure", "heartbeat", "palpitation"]):
            self.current_dept = "Cardiologist"
        elif any(x in q_low for x in ["skin", "derm", "rash", "acne", "pimple", "allergy", "itching", "eczema", "psoriasis"]):
            self.current_dept = "Dermatologist"
        elif any(x in q_low for x in ["kid", "kids", "child", "children", "baby", "infant", "growth", "vaccination", "fever child"]):
            self.current_dept = "Pediatrician"
        elif any(x in q_low for x in ["brain", "neuro", "headache", "migraine", "tremor", "seizure", "epilepsy", "dizziness", "stroke"]):
            self.current_dept = "Neurologist"
        elif any(x in q_low for x in ["bone", "joint", "ortho", "fracture", "back pain", "knee pain", "shoulder pain", "arthritis"]):
            self.current_dept = "Orthopedic Surgeon"
        elif any(x in q_low for x in ["surgery", "operation", "operate", "cut", "appendix", "hernia", "gallbladder"]):
            self.current_dept = "General Surgeon"

        # ── If asking about a specific doctor by name, fetch all and let LLM filter ──
        # This handles "who is Dr. Dawood Khan" / "timings of Dr. X" correctly
        doctor_name_match = re.search(
            r'\bdr\.?\s+([a-z]+(?:\s+[a-z]+)?)',
            q_low
        )

        tools = [
            {
                "type": "function",
                "function": {
                    "name": "get_doctor_info",
                    "description": (
                        "Search for doctors by name or specialty. "
                        "Pass the doctor's name to find a specific doctor. "
                        "Pass a specialty (e.g. 'Cardiologist') to find doctors in that specialty. "
                        "Pass an empty string to list all doctors."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "search_term": {
                                "type": "string",
                                "description": "Doctor name, specialty, or empty string for all doctors"
                            }
                        }
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "get_available_slots",
                    "description": (
                        "Get available time slots for a specific doctor on a specific date. "
                        "Call when patient asks about doctor availability or free slots."
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
                    "description": "Get test pricing. Pass empty string to list all tests.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "test_name": {"type": "string"}
                        }
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "get_department_info",
                    "description": "List hospital departments. Pass empty string to list all.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "search_term": {"type": "string"}
                        }
                    },
                },
            }
        ]

        system_message = {
            "role": "system",
            "content": (
                f"You are the empathetic front-desk manager at Harram Hospital. Patient: {self.patient_name}. "
                f"ACTIVE CONTEXT: {self.current_dept or 'General'}. "
                "RULES: "
                "1. NO TAGS: Never output <function> tags or raw JSON in your final answer. "
                "2. NO GUESSING: Only use data from tool results. Use Rs. for prices. "
                "3. DATABASE ONLY: Only use doctor names from tool results. Never invent names. "
                "4. IF EMPTY: If tool returns [], say: 'I'm sorry, I don't have that listed in my records yet.' "
                "5. DOCTOR BY NAME: If asked about a specific doctor (e.g. 'who is Dr. Dawood Khan'), "
                "   call get_doctor_info with that doctor's name as search_term. "
                "6. CONTEXT MEMORY: Apply follow-up questions to the last discussed doctor/test."
            )
        }

        messages = [
            system_message,
            {"role": "user", "content": query}
        ]

        dept_synonyms = {
            "neuro": "Neurologist", "neurology": "Neurologist", "brain": "Neurologist",
            "child": "Pediatrician", "kids": "Pediatrician", "baby": "Pediatrician", "pediatric": "Pediatrician",
            "ortho": "Orthopedic Surgeon", "bone": "Orthopedic Surgeon", "joint": "Orthopedic Surgeon",
            "skin": "Dermatologist", "derm": "Dermatologist", "rash": "Dermatologist", "acne": "Dermatologist",
            "surgery": "General Surgeon", "operate": "General Surgeon", "appendix": "General Surgeon",
            "hernia": "General Surgeon", "gallbladder": "General Surgeon",
            "heart": "Cardiologist", "cardio": "Cardiologist", "cardiac": "Cardiologist", "chest": "Cardiologist"
        }

        try:
            response = self.client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=messages,
                tools=tools,
                tool_choice="auto"
            )

            response_message = response.choices[0].message
            tool_calls = response_message.tool_calls

            if tool_calls:
                # Always keep system message + append assistant message with tool calls
                messages.append(response_message)

                for tool_call in tool_calls:
                    func_name = tool_call.function.name
                    args = json.loads(tool_call.function.arguments)

                    if func_name == "get_doctor_info":
                        raw_val = args.get("search_term", "")
                        if isinstance(raw_val, dict):
                            raw_val = ""
                        else:
                            raw_val = str(raw_val).strip()

                        if any(x in raw_val.lower() for x in ["all", "list", "every", "full", "show all", "everything"]):
                            val = ""
                        else:
                            # Try synonym map first, then use as-is (handles names + specialties)
                            synonym_map = {
                                "heart": "Cardiologist", "cardio": "Cardiologist",
                                "brain": "Neurologist", "neuro": "Neurologist",
                                "skin": "Dermatologist", "derm": "Dermatologist",
                                "child": "Pediatrician", "kids": "Pediatrician", "baby": "Pediatrician",
                                "bone": "Orthopedic Surgeon", "ortho": "Orthopedic Surgeon",
                                "surgery": "General Surgeon", "operation": "General Surgeon",
                                "general physician": "General Physician",
                            }
                            val = synonym_map.get(raw_val.lower(), raw_val)
                        result = self.db.get_doctors(val)

                    elif func_name == "get_test_info":
                        raw_val = args.get("test_name", "")
                        if isinstance(raw_val, dict):
                            raw_val = ""
                        else:
                            raw_val = str(raw_val).lower()
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
                            keywords = [k for k in re.split(r"\s+", raw_val) if len(k) > 2]
                            val = {"$or": []}
                            for kw in keywords:
                                dept_name = dept_synonyms.get(kw, kw)
                                val["$or"].append({"name": {"$regex": re.escape(dept_name), "$options": "i"}})
                            if not val["$or"]:
                                val = {}
                        result = self.db.get_departments(val)

                    else:
                        result = []

                    messages.append({
                        "role":         "tool",
                        "tool_call_id": tool_call.id,
                        "content":      json.dumps(result, default=str)
                    })

                # ── Final call: always include system message at index 0 ────────
                # Never slice off the system message — keep [system, assistant+tools, tool_results]
                final_messages = [system_message] + messages[1:]

                final_res = self.client.chat.completions.create(
                    model="llama-3.3-70b-versatile",
                    messages=final_messages
                )
                raw_answer = final_res.choices[0].message.content

            else:
                raw_answer = response_message.content

            return re.sub(r'<function=.*?>|<[^>]+>', '', raw_answer).strip()

        except Exception as e:
            return f"I am here for you, but I encountered a slight error. Let's try again! ({str(e)})"