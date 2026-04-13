"""
db.py — HospitalDB
==================
MongoDB data layer for Harram Hospital AI Assistant.

SLOT RULES:
  - Each 15-min slot allows: 1 Normal + 1 High (independently)
  - High does NOT block Normal and vice versa
  - Doctor sees both, decides who to attend

FIX: Removed circular import of VALID_SLOTS from booking_agent.
     Now imports from slots.py instead.
"""

from pymongo import MongoClient
from datetime import datetime, timedelta
from slots import VALID_SLOTS


MAX_NORMAL_PER_SLOT = 1
MAX_HIGH_PER_SLOT   = 1


class HospitalDB:

    def __init__(self, uri: str):
        self.client = MongoClient(uri)
        self.db = self.client["hospital_db"]

    # ── Internal helpers ───────────────────────────────────────────────────────
    def _serialize_dates(self, data):
        if isinstance(data, list):
            for item in data:
                self._serialize_dates(item)
        elif isinstance(data, dict):
            for k, v in data.items():
                if isinstance(v, datetime):
                    data[k] = v.strftime("%Y-%m-%d %H:%M")
        return data

    def _day_range(self, date_str: str):
        date_obj = datetime.strptime(date_str, "%Y-%m-%d")
        start    = date_obj.replace(hour=0,  minute=0,  second=0,  microsecond=0)
        end      = date_obj.replace(hour=23, minute=59, second=59, microsecond=999999)
        return start, end

    def _slot_to_datetime(self, date_str: str, time_slot: str) -> datetime:
        combined = f"{date_str} {time_slot}"
        try:
            return datetime.strptime(combined, "%Y-%m-%d %I:%M %p")
        except ValueError:
            return datetime.strptime(combined, "%Y-%m-%d %H:%M")

    def _next_appointment_id(self) -> int:
        last = self.db.appointments.find_one(
            sort=[("id", -1)],
            projection={"id": 1, "_id": 0}
        )
        return (last["id"] + 1) if last and "id" in last else 1

    # ── DOCTORS ───────────────────────────────────────────────────────────────
    def get_doctors(self, search_term: str = "", limit: int = 20) -> list:
        search_term = (search_term or "").lower().strip()
        query = (
            {}
            if not search_term
            else {
                "$or": [
                    {"name":       {"$regex": search_term, "$options": "i"}},
                    {"specialty":  {"$regex": search_term, "$options": "i"}},
                    {"department": {"$regex": search_term, "$options": "i"}}
                ]
            }
        )
        results = list(self.db.doctors.find(query, {"_id": 0}).limit(limit))
        return self._serialize_dates(results) if results else []

    # ── TESTS ─────────────────────────────────────────────────────────────────
    def get_tests(self, test_name: str = "", limit: int = 20) -> list:
        test_name = (test_name or "").lower().strip()
        query = {} if not test_name else {"name": {"$regex": test_name, "$options": "i"}}
        results = list(self.db.healthtests.find(query, {"_id": 0}).limit(limit))
        return self._serialize_dates(results) if results else []

    # ── DEPARTMENTS ───────────────────────────────────────────────────────────
    def get_departments(self, search_term="", limit: int = 20) -> list:
        if isinstance(search_term, dict):
            query = search_term
        else:
            search_term = (search_term or "").lower().strip()
            query = (
                {}
                if not search_term
                else {"name": {"$regex": search_term, "$options": "i"}}
            )
        results = list(self.db.departments.find(query, {"_id": 0}).limit(limit))
        return self._serialize_dates(results) if results else []

    # ── CORE SLOT QUERY (timezone-safe) ───────────────────────────────────────
    def _get_slot_bookings(self, doctor_id: int, date_str: str, time_slot: str) -> list:
        start, end = self._day_range(date_str)
        return list(self.db.appointments.find(
            {
                "doctorId":        doctor_id,
                "appointmentDate": {"$gte": start, "$lte": end},
                "time":            time_slot,
                "status":          {"$nin": ["Cancelled", "cancelled"]}
            },
            {"_id": 0, "priority": 1, "patientName": 1}
        ))

    # ── CHECK SLOTS ───────────────────────────────────────────────────────────
    def check_slots(
        self,
        doctor_id: int,
        appointment_date: str,
        time_slot: str,
        priority: str = "Normal"
    ) -> dict:
        priority = (priority or "Normal").capitalize()

        existing      = self._get_slot_bookings(doctor_id, appointment_date, time_slot)
        high_booked   = sum(1 for a in existing if a.get("priority") == "High")
        normal_booked = sum(1 for a in existing if a.get("priority") == "Normal")
        total_booked  = len(existing)

        normal_full = normal_booked >= MAX_NORMAL_PER_SLOT
        high_full   = high_booked   >= MAX_HIGH_PER_SLOT
        slot_full   = high_full if priority == "High" else normal_full

        return {
            "slot":             time_slot,
            "date":             appointment_date,
            "doctor_id":        doctor_id,
            "priority_checked": priority,
            "total_booked":     total_booked,
            "high_booked":      high_booked,
            "normal_booked":    normal_booked,
            "max_normal":       MAX_NORMAL_PER_SLOT,
            "max_high":         MAX_HIGH_PER_SLOT,
            "normal_full":      normal_full,
            "high_full":        high_full,
            "slot_full":        slot_full,
            "normal_available": not normal_full,
            "high_available":   not high_full,
            "message": (
                f"Slot {time_slot} on {appointment_date} for {priority}: "
                f"{'TAKEN' if slot_full else 'AVAILABLE'}. "
                f"(Normal: {normal_booked}/{MAX_NORMAL_PER_SLOT}, "
                f"High: {high_booked}/{MAX_HIGH_PER_SLOT})"
            )
        }

    # ── GET AVAILABLE SLOTS ───────────────────────────────────────────────────
    def get_available_slots(
        self,
        doctor_id: int,
        appointment_date: str,
        priority: str = "Normal"
    ) -> dict:
        priority  = (priority or "Normal").capitalize()
        available = []
        full      = []

        for slot in VALID_SLOTS:
            info = self.check_slots(doctor_id, appointment_date, slot, priority)
            (full if info["slot_full"] else available).append(slot)

        try:
            date_obj      = datetime.strptime(appointment_date, "%Y-%m-%d")
            next_week     = (date_obj + timedelta(days=7)).strftime("%Y-%m-%d")
            next_week_day = (date_obj + timedelta(days=7)).strftime("%A")
        except Exception:
            next_week = next_week_day = None

        return {
            "doctor_id":      doctor_id,
            "date":           appointment_date,
            "priority":       priority,
            "free_slots":     available,
            "full_slots":     full,
            "total_free":     len(available),
            "next_week_date": next_week,
            "next_week_day":  next_week_day
        }

    # ── BOOK APPOINTMENT ──────────────────────────────────────────────────────
    def book_appointment(                        # ✅ properly indented inside class
        self,
        patient_name:     str,
        doctor_id:        int,
        doctor_name:      str,
        appointment_date: str,
        time_slot:        str,
        priority:         str = "Normal"
    ) -> dict:
        priority = (priority or "Normal").capitalize()
        if priority not in ("Normal", "High"):
            priority = "Normal"

        # ── Check slot availability ───────────────────────────────────────
        availability = self.check_slots(doctor_id, appointment_date, time_slot, priority)
        if availability["slot_full"]:
            return {
                "success": False,
                "error":   "slot_full",
                "message": (
                    f"The {time_slot} slot on {appointment_date} is already taken "
                    f"for a {priority} priority patient."
                )
            }

        # ── Parse date/time ───────────────────────────────────────────────
        try:
            slot_dt = self._slot_to_datetime(appointment_date, time_slot)
        except ValueError as e:
            return {"success": False, "error": f"Invalid date/time: {e}"}

        # ── Save to MongoDB ───────────────────────────────────────────────
        new_id = self._next_appointment_id()
        self.db.appointments.insert_one({
            "id":              new_id,
            "doctorId":        doctor_id,
            "doctorName":      doctor_name,
            "patientName":     patient_name,
            "appointmentDate": slot_dt,
            "time":            time_slot,
            "priority":        priority,
            "status":          "pending",
            "createdAt":       datetime.utcnow()
        })

        # ── Return success ────────────────────────────────────────────────
        return {
            "success":    True,
            "id":         new_id,
            "doctorId":   doctor_id,
            "doctorName": doctor_name,
            "patient":    patient_name,
            "date":       appointment_date,
            "time":       time_slot,
            "priority":   priority,
            "status":     "Confirmed",
            "message": (
                f"Appointment #{new_id} confirmed for {patient_name} "
                f"with {doctor_name} on {appointment_date} at {time_slot} "
                f"[{priority} priority]."
            )
        }

    def close(self):
        self.client.close()