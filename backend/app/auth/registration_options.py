"""Radio-button options on the reviewer registration form (DEVIATIONS.md
#282; operator: "use defaults"). Stored values are the keys; the frontend
shows the labels, served by `GET /auth/registration/options`. Edit here to
change the form."""

from __future__ import annotations

SEX_OPTIONS: dict[str, str] = {
    "female": "Female",
    "male": "Male",
}

CADRE_OPTIONS: dict[str, str] = {
    "medical_officer": "Medical officer",
    "paediatrician": "Paediatrician",
    "clinical_officer": "Clinical officer",
    "nurse": "Nurse",
    "pharmacist": "Pharmacist",
}

EXPERTISE_OPTIONS: dict[str, str] = {
    "general_practitioner": "General practitioner (post-internship)",
    "registrar": "Registrar / specialist in training",
    "specialist": "Specialist / consultant",
}

YEARS_MIN, YEARS_MAX = 0, 60
TEXT_MIN, TEXT_MAX = 4, 30
# Kenyan mobile, Safaricom or Airtel: 07XXXXXXXX or 01XXXXXXXX (DEVIATIONS.md #287).
PHONE_PATTERN = r"^0[17][0-9]{8}$"
