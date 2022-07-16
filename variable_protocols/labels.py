from __future__ import annotations
from typing import NamedTuple, Protocol, runtime_checkable, Optional


class Labels(NamedTuple):
    tags: frozenset[str]

    def fmt(self) -> str:
        return ', '.join(self.tags)

    def check(self) -> None:
        if not isinstance(self.tags, frozenset):
            raise ValueError(f"tags must be a frozenset, not {type(self.tags)}")

    def __len__(self) -> int:
        return len(self.tags)


class NoLabels(Protocol):
    labels: None


@runtime_checkable
class WithLabels(Protocol):
    labels: Optional[Labels]

    def clear(self) -> NoLabels:
        ...
