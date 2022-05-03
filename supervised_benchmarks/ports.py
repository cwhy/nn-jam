from __future__ import annotations

from typing import Literal, NamedTuple

# If the port is both Input and Output, type it as Input
PortType = Literal['Input', 'Output', 'Context']


class Port(NamedTuple):
    type: PortType
    name: str


Input = Port(type='Input', name='Input')
Output = Port(type='Output', name='Output')
Context = Port(type='Context', name='Context')
