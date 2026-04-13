"""
slots.py — Shared constants for Harram Hospital
================================================
Imported by both booking_agent.py and db.py to avoid circular imports.
"""

from datetime import datetime, timedelta

# Fallback global 15-min slots (9 AM – 5 PM)
# These are only used when no doctor document is available.
# The real slots come from each doctor's timeSlots array in MongoDB.
VALID_SLOTS = []
for _hour in range(9, 17):
    for _minute in [0, 15, 30, 45]:
        _dt = datetime(2000, 1, 1, _hour, _minute)
        _s  = _dt.strftime("%I:%M %p").lstrip("0")
        VALID_SLOTS.append(_s)

VALID_SLOTS_SET = set(VALID_SLOTS)