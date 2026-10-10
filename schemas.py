"""Small contracts shared by the agent, tools, runner, and UI."""

from dataclasses import dataclass
from typing import Literal

from pydantic import BaseModel, Field

from workspace import Workspace

Operation = Literal[
    "trim_text", "normalize_currency", "normalize_amounts", "normalize_dates",
    "remove_exact_duplicates", "flag_unresolved",
]


class CleaningSummary(BaseModel):
    """The model's explanation. Completion is verified separately in Python."""

    summary: str
    changes: list[str] = Field(default_factory=list)
    uncertainties: list[str] = Field(default_factory=list)


@dataclass
class RunContext:
    """Private per-run dependency; the model never supplies this object."""

    workspace: Workspace
    inspected: bool = False
