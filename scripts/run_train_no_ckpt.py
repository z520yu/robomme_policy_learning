#!/usr/bin/env python3
"""Run RoboMME training without checkpoint saves for resource ramp tests."""

from __future__ import annotations

import logging
from pathlib import Path
import sys


sys.path.insert(0, str(Path(__file__).resolve().parent))

import train as train_mod  # noqa: E402
import mme_vla_suite.training.config as config_mod  # noqa: E402


def _skip_save(_checkpoint_manager, _state, _data_loader, step: int) -> None:
    logging.info("Skipping checkpoint save at step %s for resource ramp.", step)


def main() -> int:
    train_mod._checkpoints.save_state = _skip_save
    train_mod.main(config_mod.cli(), tentative_run=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
