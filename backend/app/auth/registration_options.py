"""Radio-button options on the reviewer registration form (DEVIATIONS.md
#282; operator: "use defaults"). Stored values are the keys; the frontend
shows the labels, served by `GET /auth/registration/options`. Edit here to
change the form."""

from __future__ import annotations

SEX_OPTIONS: dict[str, str] = {
    "female": "Female",
    "male": "Male",
}

# Operator list, 2026-10-12 (DEVIATIONS.md #288).
CADRE_OPTIONS: dict[str, str] = {
    "neonatologist": "Neonatologist",
    "paediatrician": "Paediatrician",
    "clinical_pharmacist": "Clinical Pharmacist",
    "medical_officer": "Medical Officer",
    "other": "Other",
}

EXPERTISE_OPTIONS: dict[str, str] = {
    "general_practitioner": "General practitioner (post-internship)",
    "registrar": "Registrar / specialist in training",
    "specialist": "Specialist / consultant",
}

# The cadre that asks the reviewer to specify (`cadre_other`, DEVIATIONS.md #289).
CADRE_OTHER = "other"

# Expertise levels open only to some cadres; a level not listed here is open
# to every cadre (DEVIATIONS.md #288).
EXPERTISE_CADRES: dict[str, frozenset[str]] = {
    "general_practitioner": frozenset({"medical_officer", "other"}),
}

YEARS_MIN, YEARS_MAX = 0, 60
TEXT_MIN, TEXT_MAX = 4, 30
# Kenyan mobile, Safaricom or Airtel: 07XXXXXXXX or 01XXXXXXXX (DEVIATIONS.md #287).
PHONE_PATTERN = r"^0[17][0-9]{8}$"
