"""Torch-free validation of model checkpoint scope and content checks."""


def content_audit(args):
    mode = getattr(args, "content_audit", "auto")
    if mode not in ("auto", "none"):
        raise ValueError("content-audit must be auto or none")
    if mode == "none" and (
        getattr(args, "action", None) not in ("save", "load")
        or getattr(args, "data_pattern", None) not in ("zero", "position")
        or getattr(args, "validate_values", False)
    ):
        raise ValueError("content-audit none requires zero/position save/load without validate-values")
    return mode


def dcp_file_layout(args):
    if getattr(args, "dcp_file_layout", "rank") != "rank":
        raise ValueError("gdtensor checkpoints require one file per rank")
    return "rank"


def checkpoint_state(args):
    if getattr(args, "checkpoint_state", "model") not in (None, "model"):
        raise ValueError("Only model checkpoint state is supported")
    return "model"
