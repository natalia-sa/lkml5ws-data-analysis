from dataclasses import dataclass
from datetime import datetime


@dataclass
class EmailInfo:
    message_id: str
    sender_email: str
    date: datetime


class EmailIndex:
    def __init__(self):
        self.by_author_date: dict[tuple[str, datetime], list[EmailInfo]] = {}
        self.total_patch_emails = 0

    def add(self, email: EmailInfo) -> None:
        self.total_patch_emails += 1

        if email.date is None:
            return

        key = (email.sender_email.lower(), email.date)
        self.by_author_date.setdefault(key, []).append(email)

    def find_by_author_date(self, author_email: str, date: datetime) -> list[EmailInfo]:
        return self.by_author_date.get((author_email.lower(), date), [])
