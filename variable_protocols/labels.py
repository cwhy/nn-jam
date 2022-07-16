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

    @classmethod
    def empty(cls):
        return cls(frozenset())

    @classmethod
    def from_strs(cls, *tags: str) -> Labels:
        return cls(frozenset(tags))


L = Labels.from_strs


@runtime_checkable
class WithLabels(Protocol):
    labels: Labels

    def clear_labels(self) -> WithLabels:
        ...
