"""Blocking pageable H2D budgets fitted from real checkpoint load ranges.

The default is an affine fit to local V100 safetensors (1B/3B/7B) and DCP
(1B) cold/warm load traces, with entire repeat groups held out for validation.
Evidence: validation/meta-h2d-sim-20261029/profile/. Evaluation does not import
CUDA, read source pages, or touch the filesystem. Optional size knots use the
same continuous interpolation and extrapolation rules as the D2H profile.
"""

import json

from .meta_d2h_profile import duration_seconds as duration_seconds
from .meta_d2h_profile import validate_profile as validate_profile

DEFAULT_PROFILE = {
    "a_seconds": 0.0001260831091564399,
    "bandwidth_bytes_per_s": 4581911166.565587,
}


def load_profile(path=None):
    """Load a JSON override or return a fresh validated default profile."""
    try:
        if not path:
            profile = dict(DEFAULT_PROFILE)
        else:
            with open(path, encoding="utf-8") as stream:
                profile = json.load(stream)
        return validate_profile(profile)
    except (OSError, ValueError, KeyError, OverflowError, TypeError) as error:
        raise ValueError(f"Invalid Meta simulated H2D profile {path!r}: {error}") from error
