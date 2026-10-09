from dataclasses import dataclass
from datetime import datetime


@dataclass
class EmailInfo:
    message_id: str
    sender_email: str
    date: datetime


class EmailIndex:
    def __init__(self):
        self.by_author_date: dict[tuple[str, datetime], list[str]] = {}
        self.message_ids: set[str] = set()

    @property
    def total_patch_emails(self) -> int:
        return len(self.message_ids)

    def add(self, email: EmailInfo) -> None:
        if email.message_id in self.message_ids:
            return

        self.message_ids.add(email.message_id)

        key = (email.sender_email.casefold(), email.date)
        self.by_author_date.setdefault(key, []).append(email.message_id)

    def find_by_author_date(self, author_email: str, date: datetime) -> list[str]:
        return self.by_author_date.get((author_email.casefold(), date), [])
