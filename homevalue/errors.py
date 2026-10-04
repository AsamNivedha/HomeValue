"""User-facing error type for HomeValue.

Every problem a normal user can run into (bad CSV, missing columns, too few
records, a failed training run) is raised as a HomeValueError so the UI can show
a structured explanation instead of a Python traceback.
"""
from __future__ import annotations


class HomeValueError(Exception):
    """An error that can be explained to a non-technical user."""

    def __init__(
        self,
        title: str,
        what: str,
        expected: list[str] | None = None,
        action: str = "",
    ) -> None:
        super().__init__(title)
        self.title = title
        self.what = what
        self.expected = expected or []
        self.action = action
