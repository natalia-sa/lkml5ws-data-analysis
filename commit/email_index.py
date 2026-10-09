from dataclasses import dataclass
from datetime import datetime


import re

_PATCH_PREFIX_RE = re.compile(r"^\[PATCH(?:\s+[^]]+)?\]\s*", re.IGNORECASE)


def normalize_subject(subject: str) -> str:
    subject = subject.strip()
    subject = _PATCH_PREFIX_RE.sub("", subject)
    return " ".join(subject.split()).casefold()


@dataclass
class EmailInfo:
    message_id: str
    sender_email: str
    subject: str
    date: datetime


class EmailIndex:
    def __init__(self):
        self.by_author_date: dict[tuple[str, datetime], list[EmailInfo]] = {}
        self.by_author_subject: dict[tuple[str, str], list[EmailInfo]] = {}

        self.total_patch_emails = 0

    def add(self, email: EmailInfo) -> None:
        self.total_patch_emails += 1

        if email.date is None:
            return

        key = (email.sender_email.casefold(), email.date)
        self.by_author_date.setdefault(key, []).append(email.message_id)

        key = (email.sender_email.casefold(), normalize_subject(email.subject))
        self.by_author_subject.setdefault(key, []).append(email.message_id)

    def find_by_author_date(self, author_email: str, date: datetime) -> list[str]:
        return self.by_author_date.get((author_email.casefold(), date), [])

    def find_by_author_subject(self, author_email: str, subject: str) -> list[str]:
        return self.by_author_subject.get((author_email.casefold(), normalize_subject(subject)), [])
