"""SoftSkill compatibility package.

The public project is SoftSkill, but the import package remains ``skillopt`` so
existing configs, scripts, and SkillOpt-derived checkpoints continue to work.

Pipeline stages:
  1. Rollout   — execute episodes with current skill
  2. Reflect   — analyze trajectories, generate patches
  3. Aggregate — hierarchical merge of patches
  4. Select    — rank and select top edits
  5. Update    — apply edits to skill document
  6. Evaluate  — validate candidate skill, accept/reject
"""

__version__ = "0.1.0"

from skillopt.types import (  # noqa: F401
    BatchSpec,
    Edit,
    EditOp,
    FailureSummaryEntry,
    GateAction,
    GateResult,
    Patch,
    RawPatch,
    RolloutResult,
    SlowUpdateResult,
)
