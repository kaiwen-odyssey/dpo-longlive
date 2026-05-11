"""Add the cloned LongLive and Reward-Forcing repos to sys.path so we can import their modules."""
import os, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
LONGLIVE_DIR = ROOT / "LongLive"
RF_DIR = ROOT.parent / "Reward-Forcing"

# wan_models, longlive_models live under assets/
WAN_MODELS_DIR = ROOT / "assets" / "wan_models"
LONGLIVE_MODELS_DIR = ROOT / "assets" / "longlive"
VIDEOREWARD_DIR = ROOT / "assets" / "videoreward"


def add_longlive_to_path():
    p = str(LONGLIVE_DIR)
    if p not in sys.path:
        sys.path.insert(0, p)


def add_reward_forcing_to_path():
    p = str(RF_DIR)
    if p not in sys.path:
        sys.path.insert(0, p)


def ensure_wan_models_symlink():
    """LongLive code expects ./wan_models/Wan2.1-T2V-1.3B/; symlink from assets/.
    Place in both the project root (for direct runs) and inside LongLive/ (since LongLive
    code chdirs there and uses relative path)."""
    for base in (Path.cwd(), LONGLIVE_DIR):
        target = base / "wan_models"
        if target.is_symlink():
            try:
                if target.resolve() == WAN_MODELS_DIR.resolve():
                    continue
                target.unlink()
            except (OSError, RuntimeError):
                target.unlink()
        elif target.exists():
            # don't overwrite a real dir
            continue
        target.symlink_to(WAN_MODELS_DIR)
