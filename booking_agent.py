"""
booking_agent.py — Agent 2: The Booking Clerk
=============================================
FIXES in this version:
  1. Doctor slots come FROM DB (timeSlots field), not hardcoded 9-5
  2. High priority ALWAYS books at the FIRST slot of doctor's day window
     (even if a Normal appointment already exists there — doctor sees both)
  3. Doctor ID ALWAYS taken from DB _resolved_doctor["id"] — NEVER from LLM args
     (fixes wrong doctor_id stored in DB)
  4. confirmed_time_slot returned in book_appointment result so LLM
     displays the CORRECT time to user (fixes wrong time shown to user)
  5. Year forced to current year in system prompt (never past years)
  6. Priority auto-detected from reason — patient is NEVER asked for it
  7. Day/date verification works correctly
  8. Malformed tool JSON handled gracefully
"""

import json
import re
from datetime import datetime, timedelta
from groq import Groq


# ── Current date helpers ──────────────────────────────────────────────────────
def _today_str() -> str:
    return datetime.today().strftime("%Y-%m-%d")

def _today_display() -> str:
    return datetime.today().strftime("%A, %d %B %Y")

def _current_year() -> int:
    return datetime.today().year


# ── Build valid 15-min slots from a doctor's timeSlots array ─────────────────
def build_slots_for_doctor(doctor: dict) -> list[str]:
    """
    Given a doctor document (with timeSlots list), build all valid 15-min
    sub-slots across ALL of the doctor's time windows.

    timeSlots entry example:
      {"day": "Wednesday", "startTime": "09:00", "endTime": "12:00", ...}

    Returns sorted unique list of "HH:MM AM/PM" strings, e.g.
      ["09:00 AM", "09:15 AM", ..., "11:45 AM"]
    """
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
        try:
            return datetime.strptime(s, "%I:%M %p")
        except ValueError:
            return datetime.min

    return sorted(slots_set, key=_sort_key)


def _first_slot_for_day(doctor: dict, day_name: str) -> str | None:
    """
    Return the very first 15-min slot for a specific day from the doctor's
    timeSlots. Used for High Priority booking.
    """
    day_name = day_name.strip().capitalize()
    windows_today = [
        w for w in (doctor.get("timeSlots") or [])
        if w.get("day", "").strip().capitalize() == day_name
    ]
    if not windows_today:
        windows_today = doctor.get("timeSlots") or []
    if not windows_today:
        return None

    earliest = None
    for w in windows_today:
        try:
            t = datetime.strptime(w["startTime"], "%H:%M")
            if earliest is None or t < earliest:
                earliest = t
        except (KeyError, ValueError):
            continue

    if earliest is None:
        return None

    return earliest.strftime("%I:%M %p").lstrip("0") or "12:00 AM"


def _all_slots_for_day(doctor: dict, day_name: str) -> list[str]:
    """All 15-min slots available for a doctor on a specific day name."""
    day_name = day_name.strip().capitalize()
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
        try:
            return datetime.strptime(s, "%I:%M %p")
        except ValueError:
            return datetime.min

    return sorted(slots_set, key=_sort_key)


# ── Fallback global slots (only used when no doctor resolved yet) ─────────────
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
        ampm = m.group(3).upper().replace('.', '').replace(' ', '')
        ampm = "AM" if "A" in ampm else "PM"
        return f"{h:02d}:{mn:02d} {ampm}"
    return None


# ── Date validator ────────────────────────────────────────────────────────────
def validate_date(date_str: str) -> dict:
    try:
        date_obj = datetime.strptime(date_str, "%Y-%m-%d")
    except ValueError:
        parts = date_str.split("-")
        hint = ""
        if len(parts) == 3:
            try:
                y, m, d = parts
                if len(y) == 4 and int(y) < _current_year():
                    hint = f" The year {y} is in the past — did you mean {_current_year()}?"
                elif int(m) > 12:
                    hint = f" Month {m} is invalid (must be 01-12). Did you swap month and day?"
                elif int(d) > 31:
                    hint = f" Day {d} is too large."
            except Exception:
                pass
        return {
            "valid":   False,
            "message": f"'{date_str}' is not a valid date.{hint} Please use YYYY-MM-DD (e.g. {_current_year()}-04-14)."
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
                f"Confirmed — {date_str} is indeed {actual_day}." if matched
                else (
                    f"{date_str} is actually {actual_day}, not {day_name.strip().capitalize()}. "
                    f"Please give the correct date for a {day_name.strip().capitalize()}, "
                    f"or correct the day name."
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
TODAY: {today_disp}. Current year is {year}. NEVER use any year before {year}. All dates must be real calendar dates in {year} or later.

STRICT BOOKING STEPS — follow in exact order, never skip:

STEP 1 — DOCTOR
  Call get_doctor_info with the doctor name or specialty the patient mentioned.
  Confirm doctor name and ID with the patient.

STEP 2 — DAY + DATE
  Ask patient for both day name AND date together (e.g. "Wednesday 2026-04-16").
  Call verify_day_date to confirm the day matches the date.
  If mismatch → tell the patient exactly what the actual day is and ask to correct.
  If date is in the past or invalid → tell patient and ask again.
  Always use year {year} or later. NEVER suggest or accept dates in past years.

STEP 3 — TIME
  Show the patient the doctor's available time slots for that day (call get_doctor_slots).
  Ask the patient which time slot they prefer.
  Call check_slots to verify availability.
  If check_slots or book_appointment returns "slot_full" error with a "next_free_slot" field:
    → Immediately call book_appointment again using next_free_slot as the time_slot.
    → Do NOT ask the patient again — just book the next available slot automatically.
    → Inform the patient AFTER booking: "Slot X was taken, so I've booked you at Y instead."
  If the whole day is fully booked → suggest the same time next week.

STEP 4 — REASON (MANDATORY — NEVER SKIP)
  ALWAYS ask: "What is the reason for your visit?"
  Wait for the patient to reply.
  Then call classify_priority with exactly what the patient said.
  NEVER ask the patient whether their priority is Normal or High — it is auto-detected.
  NEVER call book_appointment before classify_priority has been called.

STEP 5 — BOOK
  Only after classify_priority has been called, call book_appointment.
  Use the exact doctor_id returned by get_doctor_info (the "id" field).
  CRITICAL: After book_appointment returns, read the "confirmed_time_slot" field
  from the tool result. Use THAT value as the time in your reply — NEVER use the
  time_slot argument you passed in, as the system may have adjusted it for priority.

RULES:
  - One question at a time.
  - NEVER invent or assume a doctor_id — always use the id from get_doctor_info result.
  - High priority patients are automatically booked at the FIRST slot of the doctor's
    day window regardless of normal appointments in that slot.
  - On successful booking reply ONLY:
    "BOOKING_COMPLETE: Appointment confirmed for {patient_name} with {{doctor}} on {{date}} at {{confirmed_time_slot}} [{{priority}} priority]."
    Where {{confirmed_time_slot}} is taken from the "confirmed_time_slot" field of the
    book_appointment tool result — NOT from the time_slot argument you passed in.
  - On cancellation reply ONLY: "BOOKING_CANCELLED"
"""


# ── Model fallback chain ──────────────────────────────────────────────────────
_MODEL_CHAIN = [
    "llama-3.3-70b-versatile",
    "llama-3.1-70b-versatile",
    "llama-3.1-8b-instant",
]


class BookingAgent:
    def __init__(self, groq_api_key_2: str, db, patient_name: str):
        self.client              = Groq(api_key=groq_api_key_2)
        self.db                  = db
        self.patient_name        = patient_name
        self.history: list[dict] = []
        self._resolved_doctor    = None   # full doctor document from DB
        self._detected_priority  = None   # "Normal" | "High"
        self._pending_next_week  = None   # (date_str, time_slot) for next-week confirm
        self._model              = _MODEL_CHAIN[0]

    # ── Coerce to int safely ──────────────────────────────────────────────────
    @staticmethod
    def _int(val, default=0) -> int:
        try:
            return int(val)
        except (TypeError, ValueError):
            return default

    # ─────────────────────────────────────────────────────────────────────────
    # BUG FIX #2: _doctor_id() is now the SINGLE source of truth.
    # It ONLY reads from self._resolved_doctor (set by get_doctor_info from DB).
    # We NEVER fall back to LLM-provided doctor_id anywhere in the code.
    # ─────────────────────────────────────────────────────────────────────────
    def _doctor_id(self) -> int:
        if self._resolved_doctor:
            return self._int(self._resolved_doctor.get("id", 0))
        return 0

    # ── Doctor's slots for a specific day ────────────────────────────────────
    def _doctor_day_slots(self, day_name: str) -> list[str]:
        if self._resolved_doctor:
            slots = _all_slots_for_day(self._resolved_doctor, day_name)
            if slots:
                return slots
            return build_slots_for_doctor(self._resolved_doctor) or VALID_SLOTS
        return VALID_SLOTS

    @property
    def _tools(self):
        return [
            {
                "type": "function",
                "function": {
                    "name": "get_doctor_info",
                    "description": "Search doctor by name or specialty. Returns doctor list with id, name, specialty, timeSlots, availableDays.",
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
                    "description": "Get all valid 15-minute time slots for the resolved doctor on a specific day name.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "day_name": {
                                "type": "string",
                                "description": "Day of week e.g. Wednesday"
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
                    "description": "Verify that the day name matches the calendar date. Call after patient gives day + date.",
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
                    "description": "Auto-detect High or Normal priority from the patient's stated reason for visit. MUST be called before book_appointment.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "reason": {
                                "type": "string",
                                "description": "Exact reason the patient gave for their visit."
                            }
                        },
                        "required": ["reason"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "check_slots",
                    "description": "Check if a specific 15-min slot is free for the doctor on a date.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "doctor_id":        {"type": "integer"},
                            "appointment_date": {"type": "string", "description": "YYYY-MM-DD"},
                            "time_slot":        {"type": "string"},
                            "priority":         {"type": "string", "enum": ["Normal", "High"]}
                        },
                        "required": ["doctor_id", "appointment_date", "time_slot", "priority"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "get_available_slots",
                    "description": "Get all free 15-min slots for a doctor/date/priority.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "doctor_id":        {"type": "integer"},
                            "appointment_date": {"type": "string"},
                            "priority":         {"type": "string", "enum": ["Normal", "High"]}
                        },
                        "required": ["doctor_id", "appointment_date", "priority"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "book_appointment",
                    "description": (
                        "Book the appointment. Only call AFTER classify_priority has been called. "
                        "The result contains 'confirmed_time_slot' — ALWAYS use that field "
                        "for the time shown to the patient, not the time_slot argument you passed in."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "doctor_id":        {"type": "integer"},
                            "doctor_name":      {"type": "string"},
                            "appointment_date": {"type": "string"},
                            "time_slot":        {"type": "string"},
                            "priority":         {"type": "string", "enum": ["Normal", "High"]}
                        },
                        "required": ["doctor_id", "doctor_name", "appointment_date", "time_slot", "priority"]
                    }
                }
            }
        ]

    def _execute_tool(self, func_name: str, args: dict) -> str:
        try:
            # ── get_doctor_info ───────────────────────────────────────────────
            if func_name == "get_doctor_info":
                results = self.db.get_doctors(args.get("search_term", ""))
                if results:
                    # Store the FIRST matched doctor — id comes from DB only
                    self._resolved_doctor = results[0]
                    # Ensure id is always an int
                    if "id" in self._resolved_doctor:
                        self._resolved_doctor["id"] = self._int(self._resolved_doctor["id"])
                return json.dumps(results, default=str)

            # ── get_doctor_slots ──────────────────────────────────────────────
            elif func_name == "get_doctor_slots":
                day_name = args.get("day_name", "")
                if not self._resolved_doctor:
                    return json.dumps({
                        "error":   "no_doctor",
                        "message": "No doctor resolved yet. Call get_doctor_info first."
                    })
                slots = _all_slots_for_day(self._resolved_doctor, day_name)
                if not slots:
                    avail_days = self._resolved_doctor.get("availableDays", [])
                    return json.dumps({
                        "error":           "no_slots_on_day",
                        "day":             day_name,
                        "available_days":  avail_days,
                        "message": (
                            f"Dr. {self._resolved_doctor.get('name','?')} is not available on {day_name}. "
                            f"Available days: {', '.join(avail_days)}."
                        )
                    })
                return json.dumps({
                    "doctor":    self._resolved_doctor.get("name"),
                    "doctor_id": self._doctor_id(),
                    "day":       day_name,
                    "slots":     slots,
                    "count":     len(slots)
                })

            # ── verify_day_date ───────────────────────────────────────────────
            elif func_name == "verify_day_date":
                result = verify_day_matches_date(
                    args.get("day_name", ""), args.get("date_str", "")
                )
                if result.get("matched") and self._resolved_doctor:
                    actual_day  = result.get("actual_day", "")
                    avail_days  = [d.strip() for d in (self._resolved_doctor.get("availableDays") or [])]
                    if avail_days and actual_day not in avail_days:
                        result["doctor_available"] = False
                        result["message"] += (
                            f" However, Dr. {self._resolved_doctor.get('name','?')} "
                            f"is NOT available on {actual_day}. "
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
                    "message":  f"Priority auto-detected as {p} based on patient's reason."
                })

            # ── check_slots ───────────────────────────────────────────────────
            elif func_name == "check_slots":
                date_str  = args.get("appointment_date", "")
                date_chk  = validate_date(date_str)
                if not date_chk["valid"]:
                    return json.dumps({"error": "invalid_date", "message": date_chk["message"]})

                slot      = normalise_time(args.get("time_slot", "")) or args.get("time_slot", "")
                priority  = args.get("priority", self._detected_priority or "Normal")

                # BUG FIX #2: Always use DB doctor_id — never trust LLM args
                doctor_id = self._doctor_id()
                if not doctor_id:
                    return json.dumps({
                        "error":   "doctor_not_resolved",
                        "message": "Doctor not resolved. Please call get_doctor_info first."
                    })

                result = self.db.check_slots(
                    doctor_id=doctor_id,
                    appointment_date=date_str,
                    time_slot=slot,
                    priority=priority
                )

                if result.get("slot_full"):
                    date_obj  = datetime.strptime(date_str, "%Y-%m-%d")
                    day_name  = date_obj.strftime("%A")
                    day_slots = self._doctor_day_slots(day_name)
                    alts = []
                    try:
                        try:
                            idx = day_slots.index(slot)
                        except ValueError:
                            # Fallback: match by parsed time value
                            try:
                                slot_dt = datetime.strptime(slot, "%I:%M %p")
                            except ValueError:
                                slot_dt = None
                            idx = -1
                            if slot_dt:
                                for i, s in enumerate(day_slots):
                                    try:
                                        if datetime.strptime(s, "%I:%M %p") == slot_dt:
                                            idx = i
                                            break
                                    except ValueError:
                                        continue
                        if idx >= 0:
                            alts = day_slots[idx + 1: idx + 4]
                    except Exception:
                        alts = []
                    result["suggested_alternatives"] = alts

                return json.dumps(result, default=str)

            # ── get_available_slots ───────────────────────────────────────────
            elif func_name == "get_available_slots":
                date_str  = args.get("appointment_date", "")
                date_chk  = validate_date(date_str)
                if not date_chk["valid"]:
                    return json.dumps({"error": "invalid_date", "message": date_chk["message"]})

                priority  = args.get("priority", self._detected_priority or "Normal")

                # BUG FIX #2: Always use DB doctor_id
                doctor_id = self._doctor_id()
                if not doctor_id:
                    return json.dumps({
                        "error":   "doctor_not_resolved",
                        "message": "Doctor not resolved. Please call get_doctor_info first."
                    })

                date_obj  = datetime.strptime(date_str, "%Y-%m-%d")
                day_name  = date_obj.strftime("%A")
                day_slots = self._doctor_day_slots(day_name)

                available = []
                full      = []
                for slot in day_slots:
                    info = self.db.check_slots(doctor_id, date_str, slot, priority)
                    (full if info["slot_full"] else available).append(slot)

                next_week     = (date_obj + timedelta(days=7)).strftime("%Y-%m-%d")
                next_week_day = (date_obj + timedelta(days=7)).strftime("%A")

                return json.dumps({
                    "doctor_id":      doctor_id,
                    "date":           date_str,
                    "priority":       priority,
                    "free_slots":     available,
                    "full_slots":     full,
                    "total_free":     len(available),
                    "next_week_date": next_week,
                    "next_week_day":  next_week_day
                }, default=str)

            # ── book_appointment ──────────────────────────────────────────────
            elif func_name == "book_appointment":

                # HARD GUARD 1: reason must be collected first
                if self._detected_priority is None:
                    return json.dumps({
                        "error":   "reason_required",
                        "message": "You MUST ask the patient for their visit reason and call classify_priority first. Do NOT book without it."
                    })

                date_str  = args.get("appointment_date", "")
                date_chk  = validate_date(date_str)
                if not date_chk["valid"]:
                    return json.dumps({"error": "invalid_date", "message": date_chk["message"]})

                # Always use auto-detected priority — never trust LLM arg
                priority  = self._detected_priority

                # ─────────────────────────────────────────────────────────────
                # BUG FIX #2: STRICT doctor_id — only from DB, NEVER from LLM.
                # If _resolved_doctor is not set, refuse to book.
                # ─────────────────────────────────────────────────────────────
                doctor_id = self._doctor_id()
                if not doctor_id:
                    return json.dumps({
                        "error":   "doctor_not_resolved",
                        "message": "Doctor ID could not be determined. Please call get_doctor_info first to resolve the doctor from the database."
                    })

                raw_slot  = args.get("time_slot", "")
                slot      = normalise_time(raw_slot) or raw_slot

                date_obj  = datetime.strptime(date_str, "%Y-%m-%d")
                day_name  = date_obj.strftime("%A")
                day_slots = self._doctor_day_slots(day_name)

                # ── HIGH PRIORITY: always use the FIRST slot of the day ───────
                if priority == "High":
                    first_slot = (
                        _first_slot_for_day(self._resolved_doctor, day_name)
                        if self._resolved_doctor
                        else (day_slots[0] if day_slots else slot)
                    )
                    # Override whatever slot was passed in — High always starts at day's first slot
                    slot = first_slot or slot

                    # Check if this High slot is already taken by another High
                    chk = self.db.check_slots(doctor_id, date_str, slot, "High")
                    if chk.get("slot_full"):
                        # Move to next available slot for High priority
                        # BUG FIX #4 applied here too: use time-value matching fallback
                        try:
                            try:
                                idx = day_slots.index(slot)
                            except ValueError:
                                try:
                                    slot_dt = datetime.strptime(slot, "%I:%M %p")
                                except ValueError:
                                    slot_dt = None
                                idx = -1
                                if slot_dt:
                                    for i, s in enumerate(day_slots):
                                        try:
                                            if datetime.strptime(s, "%I:%M %p") == slot_dt:
                                                idx = i
                                                break
                                        except ValueError:
                                            continue

                            next_high_slot = None
                            if idx >= 0:
                                for candidate in day_slots[idx + 1:]:
                                    c2 = self.db.check_slots(doctor_id, date_str, candidate, "High")
                                    if not c2.get("slot_full"):
                                        next_high_slot = candidate
                                        break

                            if next_high_slot:
                                slot = next_high_slot
                            else:
                                # Whole day full for High — suggest next week
                                nw_date = (date_obj + timedelta(days=7)).strftime("%Y-%m-%d")
                                nw_day  = (date_obj + timedelta(days=7)).strftime("%A")
                                self._pending_next_week = (nw_date, slot)
                                return json.dumps({
                                    "success":        False,
                                    "error":          "day_full_for_high",
                                    "message":        f"All High-priority slots on {date_str} are full. Suggest: {slot} on {nw_date} ({nw_day}). Confirm with patient.",
                                    "next_week_date": nw_date,
                                    "next_week_day":  nw_day
                                })
                        except Exception:
                            pass

                    # High priority books regardless of Normal in same slot
                    result = self.db.book_appointment(
                        patient_name=self.patient_name,
                        doctor_id=doctor_id,
                        doctor_name=args.get("doctor_name", self._resolved_doctor.get("name", "") if self._resolved_doctor else ""),
                        appointment_date=date_str,
                        time_slot=slot,
                        priority="High"
                    )

                    # ─────────────────────────────────────────────────────────
                    # BUG FIX #1: Add confirmed_time_slot to result so the LLM
                    # displays the ACTUAL booked time, not its own argument.
                    # ─────────────────────────────────────────────────────────
                    if isinstance(result, dict):
                        result["confirmed_time_slot"] = slot
                        result["confirmed_priority"]  = "High"
                    else:
                        result = {
                            "success":              True,
                            "confirmed_time_slot":  slot,
                            "confirmed_priority":   "High"
                        }

                    return json.dumps(result, default=str)

                # ── NORMAL PRIORITY ───────────────────────────────────────────
                chk = self.db.check_slots(doctor_id, date_str, slot, "Normal")
                if chk.get("slot_full"):
                    # ─────────────────────────────────────────────────────────
                    # BUG FIX #4: slot format mismatch caused ValueError on
                    # day_slots.index(slot), which fell through to "day full".
                    # Fix: try both the normalised slot AND a stripped version,
                    # and scan all slots by parsed time value as fallback.
                    # ─────────────────────────────────────────────────────────
                    free = None
                    try:
                        # Try exact match first
                        try:
                            idx = day_slots.index(slot)
                        except ValueError:
                            # Fallback: match by parsed time value
                            try:
                                slot_dt = datetime.strptime(slot, "%I:%M %p")
                            except ValueError:
                                slot_dt = None

                            idx = -1
                            if slot_dt:
                                for i, s in enumerate(day_slots):
                                    try:
                                        if datetime.strptime(s, "%I:%M %p") == slot_dt:
                                            idx = i
                                            break
                                    except ValueError:
                                        continue

                        if idx >= 0:
                            for candidate in day_slots[idx + 1:]:
                                c2 = self.db.check_slots(doctor_id, date_str, candidate, "Normal")
                                if not c2.get("slot_full"):
                                    free = candidate
                                    break
                    except Exception:
                        free = None

                    if free:
                        # AUTO-BOOK the next free slot — don't just suggest it,
                        # tell the LLM exactly which slot to use so it books correctly.
                        return json.dumps({
                            "success":        False,
                            "error":          "slot_full",
                            "message": (
                                f"Slot {slot} is already booked (Normal). "
                                f"Next available slot is {free}. "
                                f"Please book {free} instead — call book_appointment with time_slot={free}."
                            ),
                            "next_free_slot":          free,
                            "original_requested_slot": slot
                        })

                    # Only reach here if ALL slots on the day are genuinely full
                    nw_date = (date_obj + timedelta(days=7)).strftime("%Y-%m-%d")
                    nw_day  = (date_obj + timedelta(days=7)).strftime("%A")
                    self._pending_next_week = (nw_date, slot)
                    return json.dumps({
                        "success":        False,
                        "error":          "day_full",
                        "message":        f"All slots on {date_str} are fully booked. Suggest: {slot} on {nw_date} ({nw_day}). Confirm with patient.",
                        "next_week_date": nw_date,
                        "next_week_day":  nw_day
                    })

                result = self.db.book_appointment(
                    patient_name=self.patient_name,
                    doctor_id=doctor_id,
                    doctor_name=args.get("doctor_name", self._resolved_doctor.get("name", "") if self._resolved_doctor else ""),
                    appointment_date=date_str,
                    time_slot=slot,
                    priority="Normal"
                )

                # ─────────────────────────────────────────────────────────────
                # BUG FIX #1: Add confirmed_time_slot to Normal result too.
                # ─────────────────────────────────────────────────────────────
                if isinstance(result, dict):
                    result["confirmed_time_slot"] = slot
                    result["confirmed_priority"]  = "Normal"
                else:
                    result = {
                        "success":             True,
                        "confirmed_time_slot": slot,
                        "confirmed_priority":  "Normal"
                    }

                return json.dumps(result, default=str)

        except Exception as e:
            return json.dumps({"error": str(e)})

    def _trimmed_history(self, max_turns: int = 6) -> list[dict]:
        return self.history[-(max_turns * 2):]

    def respond(self, user_message: str) -> tuple[str, bool]:
        # Normalise any times in user message
        def _sub_time(m):
            fixed = normalise_time(m.group(0))
            return fixed if fixed else m.group(0)

        user_message_clean = _TIME_CLEAN.sub(_sub_time, user_message)

        # Handle next-week confirmation
        if self._pending_next_week:
            low = user_message_clean.lower()
            if any(w in low for w in ["yes", "confirm", "ok", "sure", "fine", "go ahead", "yeah", "alright"]):
                nw_date, nw_time = self._pending_next_week
                self._pending_next_week = None
                user_message_clean = f"Yes, please book for {nw_date} at {nw_time}."
            elif any(w in low for w in ["no", "cancel", "don't", "different", "other"]):
                self._pending_next_week = None
                user_message_clean = "Patient declined next-week suggestion. Ask for a different date or time."

        self.history.append({"role": "user", "content": user_message_clean})

        messages = [
            {"role": "system", "content": build_system_prompt(self.patient_name)}
        ] + self._trimmed_history()

        for _ in range(10):
            response  = None
            last_err  = ""
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
                        max_tokens  = 500
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
                                max_tokens  = 200
                            )
                            reply = (recovery.choices[0].message.content or "").strip()
                        except Exception:
                            reply = "Which doctor would you like to see?"
                        self.history.append({"role": "assistant", "content": reply})
                        return reply, False
                    break

            if response is None:
                if "429" in last_err or "rate_limit" in last_err:
                    fallback = "All models are currently rate-limited. Please wait a moment and try again."
                else:
                    fallback = "I had a technical issue. Please repeat your last message."
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
                    "content":      tool_result
                })

        fallback = "Something went wrong after too many steps. Please try again."
        self.history.append({"role": "assistant", "content": fallback})
        return fallback, False

    def reset(self):
        self.history             = []
        self._resolved_doctor    = None
        self._detected_priority  = None
        self._pending_next_week  = None