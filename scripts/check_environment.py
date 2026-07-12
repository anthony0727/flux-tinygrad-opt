from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import tinygrad


EXPECTED_TINYGRAD_REVISION = "8074c0ec8f1e3ffc5a9459294570d778b5d1fa4f"


def main() -> None:
    root = Path(tinygrad.__file__).resolve().parent.parent
    required = [
        root / "examples" / "flux1.py",
        root / "extra" / "mcts_search.py",
        root / "tinygrad" / "codegen" / "kernel.py",
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise RuntimeError(
            "tinygrad must be used from its full source checkout; missing: "
            + ", ".join(missing)
        )

    revision = subprocess.check_output(
        ["git", "-C", str(root), "rev-parse", "HEAD"], text=True
    ).strip()
    if revision != EXPECTED_TINYGRAD_REVISION:
        raise RuntimeError(
            f"expected tinygrad {EXPECTED_TINYGRAD_REVISION}, found {revision}"
        )

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from utils import get_sched_dummy

    schedule = get_sched_dummy()
    if not schedule:
        raise RuntimeError("dummy tinygrad schedule is empty")
    print({"tinygrad_revision": revision, "dummy_schedule_items": len(schedule)})


if __name__ == "__main__":
    main()
