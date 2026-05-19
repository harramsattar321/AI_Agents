"""
booking_agent.py — Agent 2: The Booking Clerk
=============================================
All slot conflict logic lives in db.py now.
This agent handles:
  - Conversation flow (5 steps: doctor → day/date → time → reason → book)
  - Tool execution (calls db.py methods)
  - Emergency alt-doctor suggestion when db returns needs_alt_doctor=True
  - Patient conflict messaging
  - Groq LLM for natural language only — not for slot logic
"""

import json
import re
from datetime import datetime, timedelta
from groq import Groq


# ── Date/time helpers ─────────────────────────────────────────────────────────

def _today_str() -> str:
    return datetime.today().strftime("%Y-%m-%d")

def _today_display() -> str:
    return datetime.today().strftime("%A, %d %B %Y")

def _current_year() -> int:
    return datetime.today().year


# ── Build 15-min slots from a doctor's timeSlots array ───────────────────────

def build_slots_for_doctor(doctor: dict) -> list[str]:
    slots_set = set()
    for window in (doctor.get("timeSlots") or []):
        try:
            start = datetime.strptime(window["startTime"], "%H:%M")
            end   = datetime.strptime(window["endTime"],   "%H:%M")
        except (KeyError, ValueError):
            continue
        current = start
        while current < end:
            slots_set.add(current.strftime("%I:%M %p").lstrip("0") or "12:00 AM")
            current += timedelta(minutes=15)

    def _sort_key(s):
        try:    return datetime.strptime(s, "%I:%M %p")
        except: return datetime.min

    return sorted(slots_set, key=_sort_key)


def _all_slots_for_day(doctor: dict, day_name: str) -> list[str]:
    day_name  = day_name.strip().capitalize()
    slots_set = set()
    for window in (doctor.get("timeSlots") or []):
        if window.get("day", "").strip().capitalize() != day_name:
            continue
        try:
            start = datetime.strptime(window["startTime"], "%H:%M")
            end   = datetime.strptime(window["endTime"],   "%H:%M")
        except (KeyError, ValueError):
            continue
        current = start
        while current < end:
            slots_set.add(current.strftime("%I:%M %p").lstrip("0") or "12:00 AM")
            current += timedelta(minutes=15)

    def _sort_key(s):
        try:    return datetime.strptime(s, "%I:%M %p")
        except: return datetime.min

    return sorted(slots_set, key=_sort_key)


# ── Fallback slots ────────────────────────────────────────────────────────────
from slots import VALID_SLOTS, VALID_SLOTS_SET


# ── Time normaliser ───────────────────────────────────────────────────────────
_TIME_CLEAN = re.compile(
    r'(\d{1,2})\s*[:.]\s*(\d{2})\s*([AaPp]\.?[Mm]\.?)', re.IGNORECASE
)
_TIME_24 = re.compile(r'^(\d{1,2}):(\d{2})$')

def normalise_time(raw: str) -> str | None:
    raw = raw.strip()
    m24 = _TIME_24.match(raw)
    if m24:
        h, mn = int(m24.group(1)), int(m24.group(2))
        if 0 <= h <= 23 and 0 <= mn <= 59:
            ampm = "AM" if h < 12 else "PM"
            h12  = h if h <= 12 else h - 12
            h12  = 12 if h12 == 0 else h12
            return f"{h12:02d}:{mn:02d} {ampm}"
    m = _TIME_CLEAN.search(raw)
    if m:
        h, mn = int(m.group(1)), int(m.group(2))
        ampm  = m.group(3).upper().replace('.','').replace(' ','')
        ampm  = "AM" if "A" in ampm else "PM"
        return f"{h:02d}:{mn:02d} {ampm}"
    return None


# ── Date validator ────────────────────────────────────────────────────────────

def validate_date(date_str: str) -> dict:
    try:
        date_obj = datetime.strptime(date_str, "%Y-%m-%d")
    except ValueError:
        parts = date_str.split("-")
        hint  = ""
        if len(parts) == 3:
            try:
                y, m, d = parts
                if len(y) == 4 and int(y) < _current_year():
                    hint = f" The year {y} is in the past — did you mean {_current_year()}?"
                elif int(m) > 12:
                    hint = f" Month {m} is invalid. Did you swap month and day?"
                elif int(d) > 31:
                    hint = f" Day {d} is too large."
            except Exception:
                pass
        return {
            "valid":   False,
            "message": f"'{date_str}' is not a valid date.{hint} Please use YYYY-MM-DD."
        }

    today = datetime.today().replace(hour=0, minute=0, second=0, microsecond=0)
    if date_obj < today:
        return {
            "valid":   False,
            "message": (
                f"'{date_str}' is in the past. Today is {today.strftime('%Y-%m-%d')}. "
                f"Please choose a future date."
            )
        }
    return {
        "valid":    True,
        "date":     date_str,
        "date_obj": date_obj.strftime("%A, %d %B %Y")
    }


# ── Day verification ──────────────────────────────────────────────────────────

def verify_day_matches_date(day_name: str, date_str: str) -> dict:
    check = validate_date(date_str)
    if not check["valid"]:
        return {"matched": False, "message": check["message"]}
    try:
        date_obj   = datetime.strptime(date_str, "%Y-%m-%d")
        actual_day = date_obj.strftime("%A")
        matched    = day_name.strip().lower() == actual_day.lower()
        return {
            "matched":    matched,
            "given_day":  day_name.strip().capitalize(),
            "actual_day": actual_day,
            "date":       date_str,
            "message": (
                f"Confirmed — {date_str} is {actual_day}." if matched
                else (
                    f"{date_str} is actually {actual_day}, not {day_name.strip().capitalize()}. "
                    f"Please correct the date or the day name."
                )
            )
        }
    except ValueError:
        return {"matched": False, "message": f"Bad date '{date_str}'. Use YYYY-MM-DD."}


# ── Priority classifier ───────────────────────────────────────────────────────
_HIGH_RE = re.compile(
    r"\b(heart attack|cardiac|chest pain|can.t breathe|accident|trauma|fracture|broken"
    r"|stroke|unconscious|faint|collapse|severe|extreme|unbearable|critical|emergency|urgent"
    r"|bleeding|vomit.*blood|cough.*blood|seizure|convulsion|anaphylaxis|high fever"
    r"|appendicitis|kidney stone|cancer|tumou?r|paralys|difficulty breathing"
    r"|shortness of breath|severe pain|head injury)\b",
    re.IGNORECASE
)

def classify_priority(reason: str) -> str:
    return "High" if _HIGH_RE.search(reason) else "Normal"


# ── Safe JSON parse ───────────────────────────────────────────────────────────

def safe_parse_args(raw: str) -> dict:
    raw = raw.strip()
    if not raw.startswith("{"):
        brace_idx = raw.find("{")
        if brace_idx != -1:
            raw = raw[brace_idx:]
        else:
            return {}
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return {}


# ── System prompt ─────────────────────────────────────────────────────────────

def build_system_prompt(patient_name: str) -> str:
    today_disp = _today_display()
    year       = _current_year()
    return f"""You are the Booking Clerk at Harram Hospital. Be concise. No greetings or filler.
TODAY: {today_disp}. Current year is {year}. NEVER use any year before {year}.

STRICT BOOKING STEPS — follow in exact order, never skip:

STEP 1 — DOCTOR
  Call get_doctor_info with the doctor name or specialty the patient mentioned.
  Confirm doctor name with the patient.

STEP 2 — DAY + DATE
  Ask patient for both day name AND date together (e.g. "Wednesday 2026-04-16").
  Call verify_day_date. If mismatch or past date → tell patient and ask again.

STEP 3 — REASON (MANDATORY — NEVER SKIP)
  ALWAYS ask: "What is the reason for your visit?"
  Wait for the reply. Then call classify_priority with exactly what the patient said.
  NEVER ask the patient whether priority is Normal or High — it is auto-detected.

STEP 4 — TIME
  Now that priority is known, call get_doctor_slots with day_name, appointment_date,
  AND priority — this returns only genuinely free slots for the patient's priority.
  Show these slots to the patient and ask which one they prefer.
  If filtered=False in the result (no date/priority passed), remind the LLM to
  pass both next time — but still show the slots and proceed.

STEP 5 — BOOK
  Call book_appointment only after classify_priority has been called.
  Use the exact doctor_id from get_doctor_info. Never invent it.

SPECIAL CASE — patient asks for slots BEFORE giving reason:
  Call get_doctor_slots with only day_name (no priority, no date).
  Show the structural slots returned, then immediately ask: "What is the reason for your visit?"
  Once reason is given, call classify_priority, then call get_doctor_slots again
  with day_name + appointment_date + priority to show the accurate filtered list.

HANDLING BOOK RESULTS:
  Success → reply ONLY:
    "BOOKING_COMPLETE: Appointment confirmed for {patient_name} with {{doctor}} on {{date}} at {{confirmed_time_slot}} [{{priority}} priority]."
    Always use confirmed_time_slot from the result — not the time_slot you passed in.

  error = patient_clash →
    "You already have an appointment at that time. Please choose a different slot."
    Then call get_doctor_slots again and ask for a new time.

  error = slot_full →
    "That slot is fully booked."
    Then call get_doctor_slots to show remaining free slots.

  error = no_emergency_slot + needs_alt_doctor = true →
    Call get_alt_doctors with the doctor's department/specialty.
    Present the alternatives to the patient and ask which one they prefer.
    Once patient picks one, restart from STEP 2 with the new doctor.

  override_used = true in success result →
    Inform patient: "A previously scheduled Normal appointment in that slot was
    cancelled to accommodate your emergency. Your appointment is confirmed."

  Cancellation → reply ONLY: "BOOKING_CANCELLED"

RULES:
  - One question at a time.
  - Never invent or assume a doctor_id.
  - Never ask patient about priority — classify_priority decides it.
"""


# ── Model chain ───────────────────────────────────────────────────────────────
_MODEL_CHAIN = [
    "llama-3.3-70b-versatile",
    "llama-3.1-70b-versatile",
    "llama-3.1-8b-instant",
]


# ─────────────────────────────────────────────────────────────────────────────
#  BOOKING AGENT
# ─────────────────────────────────────────────────────────────────────────────

class BookingAgent:

    def __init__(self, groq_api_key_2: str, db, patient_name: str):
        self.client              = Groq(api_key=groq_api_key_2)
        self.db                  = db
        self.patient_name        = patient_name   # stores patient ID
        self.history: list[dict] = []
        self._resolved_doctor    = None
        self._detected_priority  = None
        self._pending_next_week  = None
        self._model              = _MODEL_CHAIN[0]

    @staticmethod
    def _int(val, default=0) -> int:
        try:    return int(val)
        except: return default

    def _doctor_id(self) -> int:
        if self._resolved_doctor:
            return self._int(self._resolved_doctor.get("id", 0))
        return 0

    def _doctor_day_slots(self, day_name: str) -> list[str]:
        if self._resolved_doctor:
            slots = _all_slots_for_day(self._resolved_doctor, day_name)
            if slots:
                return slots
            return build_slots_for_doctor(self._resolved_doctor) or VALID_SLOTS
        return VALID_SLOTS

    # ── Tool definitions ──────────────────────────────────────────────────────

    @property
    def _tools(self):
        return [
            {
                "type": "function",
                "function": {
                    "name": "get_doctor_info",
                    "description": "Search doctor by name or specialty. Returns list with id, name, specialty, timeSlots, availableDays.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "search_term": {"type": "string"}
                        },
                        "required": ["search_term"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "get_doctor_slots",
                    "description": (
                        "Get slots for the resolved doctor on a specific day. "
                        "If priority is known (classify_priority already called), pass it — "
                        "returns only free slots for that priority. "
                        "If priority is not yet known, omit it — returns all structural slots."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "day_name": {"type": "string"},
                            "appointment_date": {
                                "type": "string",
                                "description": "YYYY-MM-DD — required when priority is passed so DB availability can be checked."
                            },
                            "priority": {
                                "type": "string",
                                "enum": ["Normal", "High"],
                                "description": "Only pass if classify_priority has already been called."
                            }
                        },
                        "required": ["day_name"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "verify_day_date",
                    "description": "Verify the day name matches the calendar date.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "day_name": {"type": "string"},
                            "date_str": {"type": "string", "description": "YYYY-MM-DD"}
                        },
                        "required": ["day_name", "date_str"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "classify_priority",
                    "description": "Detect High or Normal priority from the patient's visit reason. MUST be called before book_appointment.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "reason": {"type": "string"}
                        },
                        "required": ["reason"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "get_alt_doctors",
                    "description": (
                        "Get alternative doctors when the primary doctor has no emergency slot available. "
                        "Searches by department or specialty first, then broadly."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "department": {
                                "type": "string",
                                "description": "Department or specialty of the original doctor."
                            }
                        },
                        "required": ["department"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "book_appointment",
                    "description": (
                        "Book the appointment. Only call AFTER classify_priority. "
                        "For High priority the DB finds the nearest available slot automatically — "
                        "pass the patient's requested time as time_slot. "
                        "The result contains confirmed_time_slot — always use that in your reply."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "doctor_id":        {"type": "integer"},
                            "doctor_name":      {"type": "string"},
                            "appointment_date": {"type": "string", "description": "YYYY-MM-DD"},
                            "time_slot":        {"type": "string"},
                            "priority":         {"type": "string", "enum": ["Normal", "High"]}
                        },
                        "required": ["doctor_id", "doctor_name", "appointment_date", "time_slot", "priority"]
                    }
                }
            }
        ]

    # ── Tool execution ────────────────────────────────────────────────────────

    def _execute_tool(self, func_name: str, args: dict) -> str:
        try:

            # ── get_doctor_info ───────────────────────────────────────────────
            if func_name == "get_doctor_info":
                results = self.db.get_doctors(args.get("search_term", ""))
                if results:
                    self._resolved_doctor = results[0]
                    self._resolved_doctor["id"] = self._int(self._resolved_doctor.get("id", 0))
                return json.dumps(results, default=str)

            # ── get_doctor_slots ──────────────────────────────────────────────
            elif func_name == "get_doctor_slots":
                day_name = args.get("day_name", "")
                if not self._resolved_doctor:
                    return json.dumps({
                        "error":   "no_doctor",
                        "message": "No doctor resolved. Call get_doctor_info first."
                    })

                avail_days  = self._resolved_doctor.get("availableDays", [])
                all_slots   = _all_slots_for_day(self._resolved_doctor, day_name)

                if not all_slots:
                    return json.dumps({
                        "error":          "no_slots_on_day",
                        "day":            day_name,
                        "available_days": avail_days,
                        "message": (
                            f"Dr. {self._resolved_doctor.get('name','?')} is not available "
                            f"on {day_name}. Available days: {', '.join(avail_days)}."
                        )
                    })

                # Priority known + date provided → filter by actual DB availability
                priority         = args.get("priority") or self._detected_priority
                appointment_date = args.get("appointment_date", "")

                if priority and appointment_date:
                    date_chk = validate_date(appointment_date)
                    if date_chk["valid"]:
                        doctor_id = self._doctor_id()
                        free_slots = [
                            s for s in all_slots
                            if not self.db.check_slots(
                                doctor_id, appointment_date, s, priority,
                                self.patient_name
                            ).get("slot_full")
                        ]
                        return json.dumps({
                            "doctor":          self._resolved_doctor.get("name"),
                            "doctor_id":       doctor_id,
                            "day":             day_name,
                            "date":            appointment_date,
                            "priority":        priority,
                            "slots":           free_slots,
                            "count":           len(free_slots),
                            "filtered":        True,
                            "note": (
                                "These are genuinely free slots for your priority. "
                                if free_slots else
                                "No free slots on this day for your priority."
                            )
                        })

                # Priority not yet known — return structural slots, no DB check
                return json.dumps({
                    "doctor":    self._resolved_doctor.get("name"),
                    "doctor_id": self._doctor_id(),
                    "day":       day_name,
                    "slots":     all_slots,
                    "count":     len(all_slots),
                    "filtered":  False,
                    "note":      "Slots shown without availability filter — reason not yet collected."
                })

            # ── verify_day_date ───────────────────────────────────────────────
            elif func_name == "verify_day_date":
                result = verify_day_matches_date(
                    args.get("day_name", ""), args.get("date_str", "")
                )
                if result.get("matched") and self._resolved_doctor:
                    actual_day = result.get("actual_day", "")
                    avail_days = [
                        d.strip() for d in
                        (self._resolved_doctor.get("availableDays") or [])
                    ]
                    if avail_days and actual_day not in avail_days:
                        result["doctor_available"] = False
                        result["message"] += (
                            f" However, Dr. {self._resolved_doctor.get('name','?')} "
                            f"is not available on {actual_day}. "
                            f"Available days: {', '.join(avail_days)}."
                        )
                    else:
                        result["doctor_available"] = True
                return json.dumps(result)

            # ── classify_priority ─────────────────────────────────────────────
            elif func_name == "classify_priority":
                p = classify_priority(args.get("reason", ""))
                self._detected_priority = p
                return json.dumps({
                    "priority": p,
                    "message":  f"Priority auto-detected as {p}."
                })

            # ── get_alt_doctors ───────────────────────────────────────────────
            elif func_name == "get_alt_doctors":
                department = args.get("department", "")
                current_id = self._doctor_id()

                # Try same department first
                alts = self.db.get_doctors_by_department(
                    department, exclude_doctor_id=current_id
                )

                if not alts:
                    # Broaden — get all doctors except the current one
                    all_docs = self.db.get_doctors()
                    alts = [d for d in all_docs if d.get("id") != current_id]

                if not alts:
                    return json.dumps({
                        "found":   False,
                        "message": "No alternative doctors available at this time."
                    })

                # Format for the LLM to present to patient
                formatted = []
                for d in alts[:5]:   # cap at 5 suggestions
                    formatted.append({
                        "id":          d.get("id"),
                        "name":        d.get("name"),
                        "specialty":   d.get("specialty", ""),
                        "department":  d.get("department", ""),
                        "fee":         d.get("consultationFee", d.get("fee", "")),
                        "availableDays": d.get("availableDays", []),
                    })

                return json.dumps({
                    "found":    True,
                    "same_dept": bool(
                        self.db.get_doctors_by_department(
                            department, exclude_doctor_id=current_id
                        )
                    ),
                    "doctors":  formatted,
                    "message":  (
                        f"Found {len(formatted)} alternative doctor(s). "
                        f"Present these to the patient and ask which one they prefer."
                    )
                })

            # ── book_appointment ──────────────────────────────────────────────
            elif func_name == "book_appointment":

                # Guard: reason must be collected first
                if self._detected_priority is None:
                    return json.dumps({
                        "error":   "reason_required",
                        "message": "MUST call classify_priority before book_appointment."
                    })

                date_str = args.get("appointment_date", "")
                date_chk = validate_date(date_str)
                if not date_chk["valid"]:
                    return json.dumps({"error": "invalid_date", "message": date_chk["message"]})

                # Always use DB doctor_id — never trust LLM arg
                doctor_id = self._doctor_id()
                if not doctor_id:
                    return json.dumps({
                        "error":   "doctor_not_resolved",
                        "message": "Call get_doctor_info first to resolve the doctor."
                    })

                priority = self._detected_priority   # always from classify_priority
                raw_slot = args.get("time_slot", "")
                slot     = normalise_time(raw_slot) or raw_slot

                date_obj  = datetime.strptime(date_str, "%Y-%m-%d")
                day_name  = date_obj.strftime("%A")
                day_slots = self._doctor_day_slots(day_name)

                doctor_name = args.get(
                    "doctor_name",
                    self._resolved_doctor.get("name", "") if self._resolved_doctor else ""
                )

                result = self.db.book_appointment(
                    patient_name     = self.patient_name,
                    doctor_id        = doctor_id,
                    doctor_name      = doctor_name,
                    appointment_date = date_str,
                    time_slot        = slot,
                    priority         = priority,
                    reason           = args.get("reason"),
                    doctor_slots     = day_slots,     # needed for emergency scan
                )

                # If emergency needs a different doctor, store for follow-up
                if result.get("needs_alt_doctor"):
                    self._pending_alt_doctor_dept = (
                        self._resolved_doctor.get("specialty", "")
                        or self._resolved_doctor.get("department", "")
                        if self._resolved_doctor else ""
                    )

                return json.dumps(result, default=str)

        except Exception as e:
            return json.dumps({"error": str(e)})

    # ── Trimmed history for LLM ───────────────────────────────────────────────

    def _trimmed_history(self, max_turns: int = 6) -> list[dict]:
        return self.history[-(max_turns * 2):]

    # ── Main respond loop ─────────────────────────────────────────────────────

    def respond(self, user_message: str) -> tuple[str, bool]:

        # Normalise times in user message
        def _sub_time(m):
            fixed = normalise_time(m.group(0))
            return fixed if fixed else m.group(0)

        user_message_clean = _TIME_CLEAN.sub(_sub_time, user_message)

        # Next-week confirmation shortcut
        if self._pending_next_week:
            low = user_message_clean.lower()
            if any(w in low for w in ["yes","confirm","ok","sure","fine","go ahead","yeah","alright"]):
                nw_date, nw_time      = self._pending_next_week
                self._pending_next_week = None
                user_message_clean    = f"Yes, please book for {nw_date} at {nw_time}."
            elif any(w in low for w in ["no","cancel","don't","different","other"]):
                self._pending_next_week = None
                user_message_clean    = "Patient declined next-week suggestion. Ask for a different date or time."

        self.history.append({"role": "user", "content": user_message_clean})

        messages = [
            {"role": "system", "content": build_system_prompt(self.patient_name)}
        ] + self._trimmed_history()

        for _ in range(10):
            response = None
            last_err = ""

            for model_candidate in _MODEL_CHAIN:
                if _MODEL_CHAIN.index(model_candidate) < _MODEL_CHAIN.index(self._model):
                    continue
                try:
                    response = self.client.chat.completions.create(
                        model       = model_candidate,
                        messages    = messages,
                        tools       = self._tools,
                        tool_choice = "auto",
                        temperature = 0.0,
                        max_tokens  = 500,
                    )
                    self._model = model_candidate
                    break
                except Exception as api_err:
                    last_err = str(api_err)
                    if "429" in last_err or "rate_limit" in last_err:
                        continue
                    if "tool_use_failed" in last_err or "failed_generation" in last_err:
                        try:
                            recovery = self.client.chat.completions.create(
                                model       = model_candidate,
                                messages    = messages,
                                temperature = 0.0,
                                max_tokens  = 200,
                            )
                            reply = (recovery.choices[0].message.content or "").strip()
                        except Exception:
                            reply = "Which doctor would you like to see?"
                        self.history.append({"role": "assistant", "content": reply})
                        return reply, False
                    break

            if response is None:
                fallback = (
                    "All models are currently rate-limited. Please wait a moment and try again."
                    if ("429" in last_err or "rate_limit" in last_err)
                    else "I had a technical issue. Please repeat your last message."
                )
                self.history.append({"role": "assistant", "content": fallback})
                return fallback, False

            response_msg = response.choices[0].message
            tool_calls   = response_msg.tool_calls

            if not tool_calls:
                reply = (response_msg.content or "").strip()
                self.history.append({"role": "assistant", "content": reply})
                is_terminal = (
                    reply.startswith("BOOKING_COMPLETE") or
                    reply.startswith("BOOKING_CANCELLED")
                )
                return reply, is_terminal

            messages.append(response_msg)
            for tc in tool_calls:
                args        = safe_parse_args(tc.function.arguments)
                tool_result = self._execute_tool(tc.function.name, args)
                messages.append({
                    "role":         "tool",
                    "tool_call_id": tc.id,
                    "name":         tc.function.name,
                    "content":      tool_result,
                })

        fallback = "Something went wrong after too many steps. Please try again."
        self.history.append({"role": "assistant", "content": fallback})
        return fallback, False

    def reset(self):
        self.history                = []
        self._resolved_doctor       = None
        self._detected_priority     = None
        self._pending_next_week     = None
        self._pending_alt_doctor_dept = None