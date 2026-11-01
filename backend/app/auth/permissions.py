"""The permission matrix from docs/workflows-and-permissions.md as code.

Public access needs no permission. Ownership of patient records is checked separately, because
"own records only" depends on the record, not just the role.
"""

from enum import StrEnum


class Permission(StrEnum):
    VIEW_OWN_PROFILE = "VIEW_OWN_PROFILE"
    VIEW_OWN_APPOINTMENTS = "VIEW_OWN_APPOINTMENTS"
    MANAGE_OWN_APPOINTMENTS = "MANAGE_OWN_APPOINTMENTS"
    MANAGE_OWN_WAITLIST = "MANAGE_OWN_WAITLIST"
    REQUEST_PATIENT_SPECIFIC_ASSISTANCE = "REQUEST_PATIENT_SPECIFIC_ASSISTANCE"
    REQUEST_GENERAL_ASSISTANCE = "REQUEST_GENERAL_ASSISTANCE"
    ACCESS_STAFF_DASHBOARD = "ACCESS_STAFF_DASHBOARD"
    VIEW_CLINIC_SCHEDULES = "VIEW_CLINIC_SCHEDULES"
    VIEW_PATIENT_DIRECTORY = "VIEW_PATIENT_DIRECTORY"
    MANAGE_DOCTOR_AVAILABILITY = "MANAGE_DOCTOR_AVAILABILITY"
    VIEW_HANDOFF_TICKETS = "VIEW_HANDOFF_TICKETS"
    UPDATE_HANDOFF_TICKETS = "UPDATE_HANDOFF_TICKETS"


ROLE_PERMISSIONS: dict[str, frozenset[Permission]] = {
    "PATIENT": frozenset(
        {
            Permission.VIEW_OWN_PROFILE,
            Permission.VIEW_OWN_APPOINTMENTS,
            Permission.MANAGE_OWN_APPOINTMENTS,
            Permission.MANAGE_OWN_WAITLIST,
            Permission.REQUEST_PATIENT_SPECIFIC_ASSISTANCE,
            Permission.REQUEST_GENERAL_ASSISTANCE,
        }
    ),
    # Staff deliberately hold no appointment, waitlist, or offer mutation permission in v1.
    "STAFF": frozenset(
        {
            Permission.REQUEST_GENERAL_ASSISTANCE,
            Permission.ACCESS_STAFF_DASHBOARD,
            Permission.VIEW_CLINIC_SCHEDULES,
            Permission.VIEW_PATIENT_DIRECTORY,
            Permission.MANAGE_DOCTOR_AVAILABILITY,
            Permission.VIEW_HANDOFF_TICKETS,
            Permission.UPDATE_HANDOFF_TICKETS,
        }
    ),
}


def has_permission(actor_type: str, permission: Permission) -> bool:
    return permission in ROLE_PERMISSIONS.get(actor_type, frozenset())
