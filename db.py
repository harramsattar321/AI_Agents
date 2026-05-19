"""
db.py — HospitalDB
==================
MongoDB data layer for Harram Hospital AI Assistant.

SLOT RULES:
  - Each 15-min slot allows: 1 Normal + 1 High (independently)
  - High does NOT block Normal and vice versa UNLESS it is an emergency override

BOOKING RULES:
  - Patient cannot have two appointments at the same date + time (any doctor)
  - Doctor cannot have two Normal or two High in the same slot
  - Emergency (High priority) override:
      1. Scan forward from requested time in 15-min intervals (within doctor's timeSlots)
      2. If slot is free for High → book directly
      3. If slot has a Normal → cancel that Normal, book emergency there
      4. If slot already has a High → skip, try next interval
      5. If no slot found for this doctor → caller handles doctor suggestion
"""

from pymongo import MongoClient
from datetime import datetime, timedelta
from slots import VALID_SLOTS


MAX_NORMAL_PER_SLOT = 1
MAX_HIGH_PER_SLOT   = 1


class HospitalDB:

    def __init__(self, uri: str):
        self.client = MongoClient(uri)
        self.db     = self.client["hospital_db"]

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

    def get_doctors_by_department(self, department: str, exclude_doctor_id: int = None) -> list:
        """
        Fetch all doctors in a department, optionally excluding one by ID.
        Used for emergency fallback suggestions.
        """
        query = {"$or": [
            {"specialty":  {"$regex": department, "$options": "i"}},
            {"department": {"$regex": department, "$options": "i"}}
        ]}
        results = list(self.db.doctors.find(query, {"_id": 0}).limit(20))
        if exclude_doctor_id is not None:
            results = [d for d in results if d.get("id") != exclude_doctor_id]
        return self._serialize_dates(results) if results else []

    # ── TESTS ─────────────────────────────────────────────────────────────────

    def get_tests(self, test_name: str = "", limit: int = 20) -> list:
        test_name = (test_name or "").lower().strip()
        query     = {} if not test_name else {"name": {"$regex": test_name, "$options": "i"}}
        results   = list(self.db.healthtests.find(query, {"_id": 0}).limit(limit))
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

    # ── CORE SLOT QUERY ───────────────────────────────────────────────────────

    def _get_slot_bookings(self, doctor_id: int, date_str: str, time_slot: str) -> list:
        """All non-cancelled appointments for a doctor at a specific date + slot."""
        start, end = self._day_range(date_str)
        return list(self.db.appointments.find(
            {
                "doctorId":        doctor_id,
                "appointmentDate": {"$gte": start, "$lte": end},
                "time":            time_slot,
                "status":          {"$nin": ["Cancelled", "cancelled"]}
            },
            {"_id": 0, "priority": 1, "patientName": 1, "id": 1}
        ))

    def _patient_has_appointment_at(
        self, patient_id: str, date_str: str, time_slot: str
    ) -> bool:
        """
        Returns True if this patient already has any non-cancelled appointment
        on date_str at time_slot (with ANY doctor).
        patientName in DB stores the patient ID.
        """
        start, end = self._day_range(date_str)
        clash = self.db.appointments.find_one({
            "patientName":     patient_id,
            "appointmentDate": {"$gte": start, "$lte": end},
            "time":            time_slot,
            "status":          {"$nin": ["Cancelled", "cancelled"]}
        })
        return clash is not None

    def _cancel_appointment_by_id(self, appointment_id: int) -> bool:
        """
        Cancel a single appointment by its numeric ID.
        The email system picks up the Cancelled status change automatically.
        Returns True if a document was actually updated.
        """
        result = self.db.appointments.update_one(
            {"id": appointment_id},
            {"$set": {"status": "Cancelled"}}
        )
        return result.modified_count > 0

    # ── CHECK SLOTS ───────────────────────────────────────────────────────────

    def check_slots(
        self,
        doctor_id:        int,
        appointment_date: str,
        time_slot:        str,
        priority:         str = "Normal",
        patient_id:       str = None,
    ) -> dict:
        """
        Check availability for a doctor slot.
        Also checks patient conflict if patient_id is supplied.
        """
        priority = (priority or "Normal").capitalize()

        existing      = self._get_slot_bookings(doctor_id, appointment_date, time_slot)
        high_booked   = sum(1 for a in existing if a.get("priority") == "High")
        normal_booked = sum(1 for a in existing if a.get("priority") == "Normal")

        normal_full = normal_booked >= MAX_NORMAL_PER_SLOT
        high_full   = high_booked   >= MAX_HIGH_PER_SLOT
        slot_full   = high_full if priority == "High" else normal_full

        # Patient conflict check
        patient_clash = False
        if patient_id and not slot_full:
            patient_clash = self._patient_has_appointment_at(
                patient_id, appointment_date, time_slot
            )
            if patient_clash:
                slot_full = True

        return {
            "slot":              time_slot,
            "date":              appointment_date,
            "doctor_id":         doctor_id,
            "priority_checked":  priority,
            "normal_booked":     normal_booked,
            "high_booked":       high_booked,
            "max_normal":        MAX_NORMAL_PER_SLOT,
            "max_high":          MAX_HIGH_PER_SLOT,
            "normal_full":       normal_full,
            "high_full":         high_full,
            "patient_clash":     patient_clash,
            "slot_full":         slot_full,
            "normal_available":  not normal_full,
            "high_available":    not high_full,
            "message": (
                f"Patient already has an appointment at {time_slot} on {appointment_date}."
                if patient_clash else
                f"Slot {time_slot} on {appointment_date} for {priority}: "
                f"{'TAKEN' if slot_full else 'AVAILABLE'}. "
                f"(Normal: {normal_booked}/{MAX_NORMAL_PER_SLOT}, "
                f"High: {high_booked}/{MAX_HIGH_PER_SLOT})"
            )
        }

    # ── GET AVAILABLE SLOTS ───────────────────────────────────────────────────

    def get_available_slots(
        self,
        doctor_id:        int,
        appointment_date: str,
        priority:         str = "Normal",
        patient_id:       str = None,
    ) -> dict:
        priority  = (priority or "Normal").capitalize()
        available = []
        full      = []

        for slot in VALID_SLOTS:
            info = self.check_slots(
                doctor_id, appointment_date, slot, priority, patient_id
            )
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
            "next_week_day":  next_week_day,
        }

    # ── EMERGENCY SLOT FINDER ─────────────────────────────────────────────────

    def find_emergency_slot(
        self,
        doctor_id:        int,
        appointment_date: str,
        requested_slot:   str,
        doctor_slots:     list[str],
    ) -> dict:
        """
        For High/emergency priority — scan forward from requested_slot through
        doctor_slots (already filtered to this day, sorted by time).

        For each candidate slot:
          - If High slot is free → return it directly (no override needed)
          - If Normal is booked there → mark it for cancellation, return it
          - If High is already booked → skip (can't displace another emergency)

        Returns:
          {
            "found": True,
            "slot": "09:15 AM",
            "override": False,           # True if we need to cancel a Normal
            "displaced_appointment_id": 12,   # only when override=True
            "displaced_patient": "PAT..."     # only when override=True
          }
          or {"found": False} if no slot works for this doctor.
        """
        # Find start index — nearest slot at or after requested_slot
        start_idx = 0
        try:
            req_dt = datetime.strptime(requested_slot, "%I:%M %p")
        except ValueError:
            req_dt = None

        if req_dt:
            for i, s in enumerate(doctor_slots):
                try:
                    if datetime.strptime(s, "%I:%M %p") >= req_dt:
                        start_idx = i
                        break
                except ValueError:
                    continue

        for slot in doctor_slots[start_idx:]:
            existing = self._get_slot_bookings(doctor_id, appointment_date, slot)
            high_count   = sum(1 for a in existing if a.get("priority") == "High")
            normal_appts = [a for a in existing if a.get("priority") == "Normal"]

            # Already has a High → skip, can't displace another emergency
            if high_count >= MAX_HIGH_PER_SLOT:
                continue

            # Slot free for High → book directly
            if not normal_appts:
                return {
                    "found":    True,
                    "slot":     slot,
                    "override": False,
                }

            # Normal is here → cancel it, take the slot
            displaced = normal_appts[0]
            return {
                "found":                    True,
                "slot":                     slot,
                "override":                 True,
                "displaced_appointment_id": displaced.get("id"),
                "displaced_patient":        displaced.get("patientName"),
            }

        return {"found": False}

    # ── BOOK APPOINTMENT ──────────────────────────────────────────────────────

    def book_appointment(
        self,
        patient_name:     str,      # stores patient ID
        doctor_id:        int,
        doctor_name:      str,
        appointment_date: str,
        time_slot:        str,
        priority:         str = "Normal",
        reason:           str = None,
        doctor_slots:     list[str] = None,   # required for emergency override
    ) -> dict:
        """
        Book an appointment.

        Normal priority:
          - Checks patient conflict first
          - Checks slot availability
          - Books if free

        High priority (emergency):
          - Uses find_emergency_slot() to get nearest available slot
          - If override needed: cancels the displaced Normal first, then books
          - If no slot found for this doctor: returns needs_alt_doctor signal
        """
        priority = (priority or "Normal").capitalize()
        if priority not in ("Normal", "High"):
            priority = "Normal"

        # ── Validate date ─────────────────────────────────────────────────────
        try:
            datetime.strptime(appointment_date, "%Y-%m-%d")
        except ValueError as e:
            return {"success": False, "error": f"Invalid date: {e}"}

        # ── HIGH PRIORITY — emergency override flow ───────────────────────────
        if priority == "High":
            slots_to_scan = doctor_slots or VALID_SLOTS

            result = self.find_emergency_slot(
                doctor_id        = doctor_id,
                appointment_date = appointment_date,
                requested_slot   = time_slot,
                doctor_slots     = slots_to_scan,
            )

            if not result["found"]:
                # No slot available for this doctor at all
                return {
                    "success":           False,
                    "error":             "no_emergency_slot",
                    "needs_alt_doctor":  True,
                    "message": (
                        f"No available emergency slot for Dr. {doctor_name} "
                        f"on {appointment_date}. An alternative doctor is needed."
                    )
                }

            confirmed_slot = result["slot"]

            # Cancel displaced Normal if needed
            displaced_info = None
            if result["override"]:
                appt_id = result.get("displaced_appointment_id")
                if appt_id is not None:
                    self._cancel_appointment_by_id(appt_id)
                displaced_info = {
                    "displaced_appointment_id": appt_id,
                    "displaced_patient":        result.get("displaced_patient"),
                }

            # Book emergency
            try:
                slot_dt = self._slot_to_datetime(appointment_date, confirmed_slot)
            except ValueError as e:
                return {"success": False, "error": f"Invalid time: {e}"}

            new_id = self._next_appointment_id()
            self.db.appointments.insert_one({
                "id":              new_id,
                "doctorId":        doctor_id,
                "doctorName":      doctor_name,
                "patientName":     patient_name,
                "appointmentDate": slot_dt,
                "time":            confirmed_slot,
                "priority":        "High",
                "status":          "pending",
                "reason":          reason,
                "createdAt":       datetime.utcnow(),
            })

            response = {
                "success":             True,
                "id":                  new_id,
                "doctorId":            doctor_id,
                "doctorName":          doctor_name,
                "patient":             patient_name,
                "date":                appointment_date,
                "time":                confirmed_slot,
                "confirmed_time_slot": confirmed_slot,
                "priority":            "High",
                "status":              "pending",
                "override_used":       result["override"],
                "message": (
                    f"Emergency appointment #{new_id} confirmed for {patient_name} "
                    f"with {doctor_name} on {appointment_date} at {confirmed_slot} [High priority]."
                    + (
                        f" A Normal appointment was cancelled to accommodate this emergency."
                        if result["override"] else ""
                    )
                )
            }
            if displaced_info:
                response["displaced"] = displaced_info

            return response

        # ── NORMAL PRIORITY ───────────────────────────────────────────────────

        # 1. Patient conflict check
        if self._patient_has_appointment_at(patient_name, appointment_date, time_slot):
            return {
                "success": False,
                "error":   "patient_clash",
                "message": (
                    f"You already have an appointment on {appointment_date} at {time_slot}. "
                    f"Please choose a different time slot."
                )
            }

        # 2. Slot availability check
        availability = self.check_slots(
            doctor_id, appointment_date, time_slot, "Normal"
        )
        if availability["slot_full"]:
            return {
                "success": False,
                "error":   "slot_full",
                "message": (
                    f"The {time_slot} slot on {appointment_date} is already fully booked."
                )
            }

        # 3. Book it
        try:
            slot_dt = self._slot_to_datetime(appointment_date, time_slot)
        except ValueError as e:
            return {"success": False, "error": f"Invalid time: {e}"}

        new_id = self._next_appointment_id()
        self.db.appointments.insert_one({
            "id":              new_id,
            "doctorId":        doctor_id,
            "doctorName":      doctor_name,
            "patientName":     patient_name,
            "appointmentDate": slot_dt,
            "time":            time_slot,
            "priority":        "Normal",
            "status":          "pending",
            "reason":          reason,
            "createdAt":       datetime.utcnow(),
        })

        return {
            "success":             True,
            "id":                  new_id,
            "doctorId":            doctor_id,
            "doctorName":          doctor_name,
            "patient":             patient_name,
            "date":                appointment_date,
            "time":                time_slot,
            "confirmed_time_slot": time_slot,
            "priority":            "Normal",
            "status":              "pending",
            "message": (
                f"Appointment #{new_id} confirmed for {patient_name} "
                f"with {doctor_name} on {appointment_date} at {time_slot} [Normal priority]."
            )
        }

    def close(self):
        self.client.close()