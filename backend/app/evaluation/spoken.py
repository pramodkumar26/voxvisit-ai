"""Reading dates and times the way people say and write them.

Pure functions, no clock of their own and no I/O: the caller supplies "today" (in the right time
zone) and receives structured readings. Three jobs:

1. `normalize_spoken` rewrites spoken number words in date and time phrases into digits
   ("November seventeenth" to "November 17", "quarter past two" to "2:15", "eleven thirty in the
   morning" to "11:30 am"), and nothing else. A number word that is not part of such a phrase is
   left alone, so "Two Rivers Clinic" stays a name.
2. `date_mentions` finds every date a message states. A date that depends on today ("tomorrow",
   "next Tuesday", "November 17" with no year) keeps a `spec`, so it can be read again against
   the clinic's own calendar day once the clinic is known. A phrase with two reasonable meanings
   ("next Tuesday" said on a Monday) is reported with both, never resolved by a guess.
3. `clock_mentions` finds every clock time. A time with no am/pm, and no morning or afternoon
   word to settle it, is ambiguous and says so.

Nothing here is a guess. When a phrase has more than one reasonable meaning, the reading says so
and the agent asks.
"""

import re
from dataclasses import dataclass
from datetime import date, timedelta

MONTHS = [
    "january",
    "february",
    "march",
    "april",
    "may",
    "june",
    "july",
    "august",
    "september",
    "october",
    "november",
    "december",
]
WEEKDAYS = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]

_CARDINALS = {
    **{
        w: i
        for i, w in enumerate(
            [
                "zero",
                "one",
                "two",
                "three",
                "four",
                "five",
                "six",
                "seven",
                "eight",
                "nine",
                "ten",
                "eleven",
                "twelve",
                "thirteen",
                "fourteen",
                "fifteen",
                "sixteen",
                "seventeen",
                "eighteen",
                "nineteen",
            ]
        )
    },
    "twenty": 20,
    "thirty": 30,
    "forty": 40,
    "fifty": 50,
}
_ORDINALS = {
    **{
        w: i
        for i, w in enumerate(
            [
                "zeroth",
                "first",
                "second",
                "third",
                "fourth",
                "fifth",
                "sixth",
                "seventh",
                "eighth",
                "ninth",
                "tenth",
                "eleventh",
                "twelfth",
                "thirteenth",
                "fourteenth",
                "fifteenth",
                "sixteenth",
                "seventeenth",
                "eighteenth",
                "nineteenth",
            ]
        )
    },
    "twentieth": 20,
    "thirtieth": 30,
}
_ONES = "one|two|three|four|five|six|seven|eight|nine"
_TEENS = "ten|eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen"
_TENS = "twenty|thirty|forty|fifty"
CARD = rf"(?:(?:{_TENS})(?:[\s-](?:{_ONES}))?|{_TEENS}|{_ONES})"
_ORD_ONES = "first|second|third|fourth|fifth|sixth|seventh|eighth|ninth"
_ORD_TEENS = (
    "tenth|eleventh|twelfth|thirteenth|fourteenth|fifteenth|sixteenth|seventeenth|"
    "eighteenth|nineteenth"
)
ORD = rf"(?:(?:twenty|thirty)[\s-](?:{_ORD_ONES})|twentieth|thirtieth|{_ORD_TEENS}|{_ORD_ONES})"
HOUR = r"(?:one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve)"
MONTH = (
    r"(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|june?|july?|aug(?:ust)?|"
    r"sep(?:t(?:ember)?)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)\.?"
)


def word_value(words: str) -> int | None:
    """The value of a spoken number or ordinal ("twenty-fourth" is 24, "forty five" is 45)."""
    parts = [p for p in re.split(r"[\s-]+", words.strip().casefold()) if p]
    if not parts:
        return None
    if len(parts) == 1:
        return _ORDINALS.get(parts[0], _CARDINALS.get(parts[0]))
    if len(parts) == 2 and parts[0] in ("twenty", "thirty", "forty", "fifty"):
        tail = _ORDINALS.get(parts[1], _CARDINALS.get(parts[1]))
        if tail is not None and 1 <= tail <= 9:
            return _CARDINALS[parts[0]] + tail
    return None


def _suffix(day: int) -> str:
    return "th" if 10 <= day % 100 <= 20 else {1: "st", 2: "nd", 3: "rd"}.get(day % 10, "th")


# --- spoken numbers to digits ------------------------------------------------------------------


def _date_words(text: str) -> str:
    def month_then_day(match: re.Match) -> str:
        value = word_value(match.group(2))
        return f"{match.group(1)} {value}" if value and 1 <= value <= 31 else match.group(0)

    def day_of_month(match: re.Match) -> str:
        value = word_value(match.group(2))
        if not value or not 1 <= value <= 31:
            return match.group(0)
        return f"{match.group(1) or ''}{value} of {match.group(3)}"

    def day_alone(match: re.Match) -> str:
        value = word_value(match.group(1))
        # "the first" to "the fifth" are, far more often, a choice from a list: not read as a day.
        return f"the {value}{_suffix(value)}" if value and 6 <= value <= 31 else match.group(0)

    text = re.sub(
        rf"\b({MONTH})\s+(?:the\s+)?({ORD}|{CARD})\b", month_then_day, text, flags=re.IGNORECASE
    )
    text = re.sub(
        rf"\b(the\s+)?({ORD}|{CARD})\s+of\s+({MONTH})\b", day_of_month, text, flags=re.IGNORECASE
    )
    return re.sub(
        rf"\bthe\s+({ORD})\b(?!\s+(?:one|option|choice|available|appointment|opening|slot|time|visit)\b)",
        day_alone,
        text,
        flags=re.IGNORECASE,
    )


def _year_words(text: str) -> str:
    """ "twenty twenty-six" and "two thousand and twenty six" to 2026, as a year after a date."""

    def teens_and_twenties(match: re.Match) -> str:
        tail = word_value(match.group(2)) or 0
        return f"{match.group(1)} 20{tail:02d}" if 0 <= tail <= 39 else match.group(0)

    def two_thousand(match: re.Match) -> str:
        tail = word_value(match.group(2) or "") or 0
        return f"{match.group(1)} {2000 + tail}"

    text = re.sub(
        rf"(\b\d{{1,2}}(?:st|nd|rd|th)?\b|\b{MONTH}\b)[,\s]+twenty[\s-]+((?:twenty|thirty)"
        rf"(?:[\s-](?:{_ONES}))?)\b",
        lambda m: (
            f"{m.group(1)} 20{word_value(m.group(2)):02d}" if word_value(m.group(2)) else m.group(0)
        ),
        text,
        flags=re.IGNORECASE,
    )
    return (
        re.sub(
            rf"(\b\d{{1,2}}(?:st|nd|rd|th)?\b)[,\s]+two thousand(?:\s+and)?(?:[\s-]+({CARD}))?\b",
            two_thousand,
            text,
            flags=re.IGNORECASE,
        )
        if "thousand" in text.casefold()
        else text
    )


def _time_words(text: str) -> str:
    def hour(word: str) -> int:
        return _CARDINALS[word.casefold()]

    def clock(h: int, m: int) -> str:
        return f"{h}:{m:02d}"

    # "quarter past two", "half past three", "quarter to three", "ten past two", "twenty to four".
    def quarter_half(match: re.Match) -> str:
        kind, direction, h = (
            match.group(1).casefold(),
            match.group(2).casefold(),
            hour(match.group(3)),
        )
        if kind == "half":
            return clock(h, 30)
        if direction in ("past", "after"):
            return clock(h, 15)
        return clock(h - 1 or 12, 45)

    text = re.sub(
        rf"\b(?:a\s+)?(quarter|half)\s+(past|after|to|till|before)\s+({HOUR})\b",
        quarter_half,
        text,
        flags=re.IGNORECASE,
    )

    def minutes_past_to(match: re.Match) -> str:
        minutes, direction, h = (
            word_value(match.group(1)),
            match.group(2).casefold(),
            hour(match.group(3)),
        )
        if minutes not in (5, 10, 20, 25):
            return match.group(0)
        if direction in ("past", "after"):
            return clock(h, minutes)
        return clock(h - 1 or 12, 60 - minutes)

    text = re.sub(
        rf"\b({CARD})\s+(past|after|to|till)\s+({HOUR})\b",
        minutes_past_to,
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(
        rf"\b(1[0-2]|[1-9]|{HOUR})\s+o['\u2019]?\s?clock\b",
        lambda m: clock(int(m.group(1)) if m.group(1)[0].isdigit() else hour(m.group(1)), 0),
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(
        rf"\b({HOUR})\s+(?:oh|o)\s+({_ONES})\b",
        lambda m: clock(hour(m.group(1)), hour(m.group(2))),
        text,
        flags=re.IGNORECASE,
    )

    def hour_minutes(match: re.Match) -> str:
        minutes = word_value(match.group(2))
        if minutes is None or not 10 <= minutes <= 59:
            return match.group(0)
        return clock(hour(match.group(1)), minutes)

    text = re.sub(rf"\b({HOUR})\s+({CARD})\b", hour_minutes, text, flags=re.IGNORECASE)

    # A meridiem, said or implied by a part of the day, on a clock written either way.
    spoken_hour = rf"(\d{{1,2}}:\d{{2}}|(?<![\d/-])\d{{1,2}}(?![\d/-])|{HOUR})"

    def meridiem_unless_a_date(match: re.Match) -> str:
        before = match.string[: match.start()]
        if re.search(
            rf"\b(?:{'|'.join(m[:3] for m in MONTHS)})[a-z]*\.?\s*$", before, re.IGNORECASE
        ):
            return match.group(0)  # "November 12 in the morning": a day of the month, not an hour.
        return _hour_meridiem(match)

    text = re.sub(
        rf"\b{spoken_hour}\s*(?:in the (morning|afternoon|evening)|([ap])\.?m\b\.?)",
        meridiem_unless_a_date,
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(r"\bmid-?day\b", "noon", text, flags=re.IGNORECASE)
    # A bare hour after "at", "after", "before" and the like, only when the phrase ends there.
    text = re.sub(
        rf"\b(at|after|before|from|until|by)\s+({HOUR})\b"
        rf"(?=\s*(?:$|[.,;:!?]|(?:in|on|at|please|tomorrow|today|then|or|and|if|that|for|with|so|but)\b))",
        lambda m: f"{m.group(1)} {hour(m.group(2))}",
        text,
        flags=re.IGNORECASE,
    )
    return text


def _hour_meridiem(match: re.Match) -> str:
    token = match.group(1)
    digits = token if token[0].isdigit() else str(_CARDINALS[token.casefold()])
    part = (match.group(2) or "").casefold()
    meridiem = (
        "am" if part == "morning" else "pm" if part else (match.group(3) or "a").casefold() + "m"
    )
    return f"{digits} {meridiem}"


def normalize_spoken(text: str) -> str:
    """The message with spoken dates and times written in digits. Other words are unchanged."""
    return _time_words(_year_words(_date_words(text)))


# --- dates --------------------------------------------------------------------------------------

_ISO = re.compile(r"\b(\d{4})-(\d{2})-(\d{2})\b")
_MONTH_FIRST = re.compile(
    rf"\b({MONTH})\s+(\d{{1,2}})(?:st|nd|rd|th)?(?!\d|:)(?:,?\s+(\d{{4}})\b)?", re.IGNORECASE
)
_DAY_FIRST = re.compile(
    rf"\b(\d{{1,2}})(?:st|nd|rd|th)?\s+(?:of\s+)?({MONTH})(?:,?\s+(\d{{4}})\b)?", re.IGNORECASE
)
_DAY_ONLY = re.compile(r"\bthe\s+(\d{1,2})(?:st|nd|rd|th)\b", re.IGNORECASE)
_RELATIVE = re.compile(
    r"\b(day after tomorrow|tomorrow|today|tonight|(next\s+|this\s+|on\s+)?"
    r"(monday|mon|tuesday|tues|tue|wednesday|wed|thursday|thurs|thur|thu|friday|fri|"
    r"saturday|sunday))\b",
    re.IGNORECASE,
)
_WEEKDAY_ABBREVIATIONS = {
    "mon": "monday",
    "tue": "tuesday",
    "tues": "tuesday",
    "wed": "wednesday",
    "thu": "thursday",
    "thur": "thursday",
    "thurs": "thursday",
    "fri": "friday",
}
_NUMBER = r"(\d{1,2}|an?|one|two|three|four|five|six|seven|eight|nine|ten)"
_OFFSET = re.compile(
    rf"\b(?:in\s+{_NUMBER}\s+(days?|weeks?)|{_NUMBER}\s+(days?|weeks?)\s+(?:from\s+)?"
    r"(?:now|today))\b",
    re.IGNORECASE,
)
_NUMBER_WORDS = {
    "a": 1, "an": 1, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
}  # fmt: skip


def without_dates(text: str) -> str:
    """The message with every date written in it blanked out, so the numbers left are not days."""
    for pattern in (_ISO, _MONTH_FIRST, _DAY_FIRST, _DAY_ONLY):
        text = pattern.sub(" ", text)
    return text


@dataclass(frozen=True)
class DateMention:
    """One date a message states.

    `iso` is the day on `today`'s calendar, or "" when the phrase has two reasonable meanings, in
    which case `options` lists them. `spec` is how to read it again against another "today"
    (the clinic's own day): "md:11:17", "rel:tomorrow", "rel:next:tuesday". It is empty for a
    date written in full, which means the same thing everywhere.
    """

    position: int
    iso: str
    spec: str = ""
    options: tuple[str, ...] = ()


def _iso(year: int, month: int, day: int) -> str:
    try:
        return date(year, month, day).isoformat()
    except ValueError:
        return ""


def _month_number(word: str) -> int:
    return next(i for i, name in enumerate(MONTHS, 1) if name.startswith(word.casefold()[:3]))


def resolve_spec(spec: str, today: date) -> tuple[str, ...]:
    """The day or days a spec means on `today`'s calendar: one, or two when it is ambiguous."""
    kind, _, rest = spec.partition(":")
    if kind == "md":
        month, day = (int(p) for p in rest.split(":"))
        for year in (today.year, today.year + 1, today.year + 2):
            found = _iso(year, month, day)
            if found and found >= today.isoformat():
                return (found,)
        return ()
    if kind == "rel":
        words = rest.split(":")
        if words[0] == "today":
            return (today.isoformat(),)
        if words[0] == "tonight":
            return (today.isoformat(),)
        if words[0] == "tomorrow":
            return ((today + timedelta(days=1)).isoformat(),)
        if words[0] == "dayafter":
            return ((today + timedelta(days=2)).isoformat(),)
        if words[0] == "plus":
            return ((today + timedelta(days=int(words[1]))).isoformat(),)
        qualifier, name = (words[0], words[1]) if len(words) == 2 else ("", words[0])
        target = WEEKDAYS.index(name)
        ahead = (target - today.weekday()) % 7 or 7
        soonest = (today + timedelta(days=ahead)).isoformat()
        if qualifier == "next" and target > today.weekday():
            # Later this week: "next Tuesday" can mean the coming one or the one after.
            return (soonest, (today + timedelta(days=ahead + 7)).isoformat())
        if qualifier == "next":
            return ((today + timedelta(days=ahead)).isoformat(),)
        return (soonest,)
    return ()


def date_mentions(request: str, today: date, anchor: date | None = None) -> list[DateMention]:
    """Every date the message states, in the order said, whichever way it is written."""
    found: dict[int, DateMention] = {}
    taken: list[range] = []

    def add(position: int, mention: DateMention) -> None:
        found.setdefault(position, mention)

    for match in _ISO.finditer(request):
        value = _iso(*map(int, match.groups()))
        if value:
            add(match.start(), DateMention(match.start(), value))
            taken.append(range(match.start(), match.end()))
    for pattern, order in ((_MONTH_FIRST, "md"), (_DAY_FIRST, "dm")):
        for match in pattern.finditer(request):
            first, second, year = match.groups()
            month_word, day_word = (first, second) if order == "md" else (second, first)
            month, day = _month_number(month_word), int(day_word)
            taken.append(range(match.start(), match.end()))
            if year:
                value = _iso(int(year), month, day)
                if value:
                    add(match.start(), DateMention(match.start(), value))
            else:
                spec = f"md:{month}:{day}"
                resolved = resolve_spec(spec, today)
                if resolved:
                    add(match.start(), DateMention(match.start(), resolved[0], spec))
    for match in _OFFSET.finditer(request):
        count, unit = (match.group(1), match.group(2)) if match.group(1) else match.groups()[2:4]
        number = int(count) if count.isdigit() else _NUMBER_WORDS.get(count.casefold(), 0)
        days = number * (7 if unit.casefold().startswith("week") else 1)
        if 0 < days <= 60:
            spec = f"rel:plus:{days}"
            resolved = resolve_spec(spec, today)
            add(match.start(), DateMention(match.start(), resolved[0], spec))
            taken.append(range(match.start(), match.end()))
    for match in _RELATIVE.finditer(request):
        if any(match.start() in span for span in taken):
            continue  # "Ten days from today" is one date, not two.
        word = match.group(1).casefold()
        if match.group(3):
            name = match.group(3).casefold()
            name = _WEEKDAY_ABBREVIATIONS.get(name, name)
            qualifier = (match.group(2) or "").strip().casefold()
            qualifier = qualifier if qualifier in ("next", "this") else ""
            spec = f"rel:{qualifier}:{name}" if qualifier else f"rel:{name}"
        else:
            spec = "rel:" + word.replace("day after tomorrow", "dayafter")
        resolved = resolve_spec(spec, today)
        if len(resolved) == 1:
            add(match.start(), DateMention(match.start(), resolved[0], spec))
        elif len(resolved) == 2:
            add(match.start(), DateMention(match.start(), "", spec, resolved))
    for match in _DAY_ONLY.finditer(request):
        if any(match.start() in span for span in taken):
            continue
        base = anchor or today
        value = _iso(base.year, base.month, int(match.group(1)))
        if value:
            add(match.start(), DateMention(match.start(), value))
    return [found[position] for position in sorted(found)]


def stated_dates(request: str, today: date, anchor: date | None = None) -> set[str]:
    """The dates the message settles. An ambiguous phrase settles none: it must be asked about."""
    return {m.iso for m in date_mentions(request, today, anchor) if m.iso}


# --- clock times --------------------------------------------------------------------------------

_CLOCK = re.compile(
    r"\b(\d{1,2})(?::(\d{2}))?\s*([ap])\.?m\b\.?"  # 1-3: written with am or pm
    r"|\b(0\d|1[3-9]|2[0-3]):([0-5]\d)\b"  # 4-5: 24-hour, unambiguous
    r"|\b(noon)\b"  # 6
    r"|\b([1-9]|1[0-2]):([0-5]\d)\b(?!\s*[ap]\.?m)"  # 7-8: 12-hour with no am or pm
    r"|\b(at|after|before|from|until|by|do|try|say|take|go with|make it)\s+([1-9]|1[0-2])"  # 9-10
    r"\b(?![\d:]|\s*(?:[ap]\.?m|st|nd|rd|th))",
    re.IGNORECASE,
)
_PARTS = {
    "morning": re.compile(r"\bmorning\b", re.IGNORECASE),
    "afternoon": re.compile(r"\bafternoon\b", re.IGNORECASE),
    "evening": re.compile(r"\b(?:evening|tonight)\b", re.IGNORECASE),
    "night": re.compile(r"\bnight\b", re.IGNORECASE),
}
_AFTER = {"after", "from"}
_BEFORE = {"before", "until", "by"}


@dataclass(frozen=True)
class ClockMention:
    start: int
    end: int
    hour: int  # As said, 0-23 when written in 24-hour form, otherwise 1-12.
    minute: int
    meridiem: str  # "am", "pm", or "" when not said.
    twenty_four: bool = False
    kind: str = "at"  # "at", "after" or "before".

    @property
    def clock_text(self) -> str:
        return f"{self.hour}:{self.minute:02d}"


def parts_of_day(text: str) -> set[str]:
    return {name for name, pattern in _PARTS.items() if pattern.search(text)}


def clock_mentions(text: str) -> list[ClockMention]:
    found = []
    for match in _CLOCK.finditer(text):
        g = match.groups()
        if g[0] is not None:
            hour, minute, meridiem, twenty_four, kind = (
                int(g[0]),
                int(g[1] or 0),
                g[2].lower() + "m",
                False,
                "at",
            )
            if not 1 <= hour <= 12 or minute > 59:
                continue
        elif g[3] is not None:
            hour, minute, meridiem, twenty_four, kind = int(g[3]), int(g[4]), "", True, "at"
        elif g[5] is not None:
            hour, minute, meridiem, twenty_four, kind = 12, 0, "pm", False, "at"
        elif g[6] is not None:
            hour, minute, meridiem, twenty_four, kind = int(g[6]), int(g[7]), "", False, "at"
        else:
            hour, minute, meridiem, twenty_four = int(g[9]), 0, "", False
            kind = (
                "after" if g[8].lower() in _AFTER else "before" if g[8].lower() in _BEFORE else "at"
            )
        if g[0] is not None or g[3] is not None or g[5] is not None or g[6] is not None:
            before = text[: match.start()].rstrip().rsplit(" ", 1)[-1].casefold()
            kind = "after" if before in _AFTER else "before" if before in _BEFORE else "at"
        found.append(
            ClockMention(match.start(), match.end(), hour, minute, meridiem, twenty_four, kind)
        )
    return found


def minutes_of_mention(mention: ClockMention, parts: set[str] | None = None) -> int | None:
    """Minutes after midnight, or None if the time is ambiguous.

    A time with no am or pm is settled only by a single part of the day in the same message:
    "in the morning" is am, "in the afternoon" or "in the evening" is pm. Twelve in the morning
    is left ambiguous, because people mean both midnight and the late morning.
    """
    if mention.twenty_four:
        return mention.hour * 60 + mention.minute
    meridiem = mention.meridiem
    if not meridiem and parts:
        day_parts = parts & {"morning", "afternoon", "evening"}
        if len(day_parts) == 1:
            meridiem = "am" if day_parts == {"morning"} else "pm"
            if meridiem == "am" and mention.hour == 12:
                return None
    if not meridiem:
        return None
    hour = mention.hour % 12 + (12 if meridiem == "pm" else 0)
    return hour * 60 + mention.minute


def minutes_of(text: str | None) -> int | None:
    """A single clock time as minutes ("10:30 AM", "2 pm", "14:30", "noon"), or None if there is
    none or it is ambiguous."""
    mentions = clock_mentions(normalize_spoken(text or ""))
    if len(mentions) != 1:
        return None
    return minutes_of_mention(mentions[0], parts_of_day(text or ""))


def stated_clocks(request: str) -> set[int]:
    """The definite times the message states (those with an am or pm, a part of the day that
    settles them, or 24-hour form)."""
    parts = parts_of_day(request)
    return {
        m
        for m in (minutes_of_mention(mention, parts) for mention in clock_mentions(request))
        if m is not None
    }


def label(minutes: int) -> str:
    return f"{minutes // 60 % 12 or 12}:{minutes % 60:02d} {'AM' if minutes < 720 else 'PM'}"
