from enum import StrEnum


class SlotStatus(StrEnum):
    AVAILABLE = "AVAILABLE"
    HELD = "HELD"
    BOOKED = "BOOKED"


class AppointmentStatus(StrEnum):
    CONFIRMED = "CONFIRMED"
    CANCELLED = "CANCELLED"


class AppointmentAction(StrEnum):
    BOOKED = "BOOKED"
    RESCHEDULED = "RESCHEDULED"
    CANCELLED = "CANCELLED"


class HistoryActorType(StrEnum):
    PATIENT = "PATIENT"
    STAFF = "STAFF"
    SYSTEM = "SYSTEM"


class WaitlistStatus(StrEnum):
    ACTIVE = "ACTIVE"
    FULFILLED = "FULFILLED"
    CANCELLED = "CANCELLED"


class TimePreference(StrEnum):
    ANYTIME = "ANYTIME"
    MORNING = "MORNING"
    AFTERNOON = "AFTERNOON"


class OfferStatus(StrEnum):
    ACTIVE = "ACTIVE"
    ACCEPTED = "ACCEPTED"
    DECLINED = "DECLINED"
    EXPIRED = "EXPIRED"


class ActorType(StrEnum):
    PATIENT = "PATIENT"
    STAFF = "STAFF"


class SessionStatus(StrEnum):
    ACTIVE = "ACTIVE"
    EXPIRED = "EXPIRED"
    LOGGED_OUT = "LOGGED_OUT"
    REVOKED = "REVOKED"


class PendingActionType(StrEnum):
    BOOK = "BOOK"
    RESCHEDULE = "RESCHEDULE"
    CANCEL = "CANCEL"
    JOIN = "JOIN"
    ACCEPT = "ACCEPT"
    DECLINE = "DECLINE"


class PendingActionStatus(StrEnum):
    ACTIVE = "ACTIVE"
    CONSUMED = "CONSUMED"
    INVALIDATED = "INVALIDATED"
    EXPIRED = "EXPIRED"


class HandoffStatus(StrEnum):
    OPEN = "OPEN"
    IN_PROGRESS = "IN_PROGRESS"
    RESOLVED = "RESOLVED"


class HandoffCategory(StrEnum):
    SCHEDULING = "SCHEDULING"
    CLINIC_INFORMATION = "CLINIC_INFORMATION"
    ACCESSIBILITY = "ACCESSIBILITY"
    TECHNICAL_SUPPORT = "TECHNICAL_SUPPORT"
    OTHER_ADMINISTRATIVE = "OTHER_ADMINISTRATIVE"


class IdempotencyActorType(StrEnum):
    PATIENT = "PATIENT"
    STAFF = "STAFF"
    ANONYMOUS = "ANONYMOUS"


class NotificationStatus(StrEnum):
    UNREAD = "UNREAD"
    READ = "READ"


class OutboxStatus(StrEnum):
    PENDING = "PENDING"
    PROCESSING = "PROCESSING"
    PROCESSED = "PROCESSED"
