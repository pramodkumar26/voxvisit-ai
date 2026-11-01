"""A synthetic, read-only, FHIR-like view of the scheduling data.

This is NOT an HL7 FHIR implementation and makes no conformance, certification, or
interoperability claim. It borrows FHIR's resource vocabulary (Patient, Practitioner, Schedule,
Slot, Appointment) and some of its field names to show where a structured healthcare-scheduling
integration boundary would sit. The deliberate simplifications:

- Only five resource types, each with a handful of fields. There is no Location, Organization,
  CapabilityStatement, OperationOutcome, history, or write interaction. A clinic is a logical
  reference (an identifier plus display name), not a served resource.
- Search uses this project's own parameters (`clinic`, `start_date`, `limit`, `cursor`) rather
  than FHIR search parameters, and errors use the project's shared error body.
- Responses are plain `application/json` and tagged `synthetic-fhir-like` instead of declaring
  any FHIR profile or media type.
- All data is fictional. No real EHR (Epic or otherwise) is contacted.

PostgreSQL stays authoritative: every resource is built on request from the same tables and
trusted read services as the rest of the API, and nothing is stored or cached. The adapter only
reads. It cannot book, reschedule, cancel, or confirm anything; those stay on the prepare and
confirm endpoints with their authorization, confirmation, transaction, and idempotency rules.
"""
