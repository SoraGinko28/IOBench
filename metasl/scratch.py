"""Load planner skeletons shared by gdtensor's Meta and resident targets."""

import torch
from torch.distributed.checkpoint._nested_dict import flatten_state_dict
from torch.distributed.checkpoint.default_planner import DefaultLoadPlanner
from torch.distributed.checkpoint.metadata import STATE_DICT_TYPE, Metadata

class OptionalKeysMixin:
    """Tolerate only the listed flattened keys being absent from the checkpoint.

    Every key outside `optional_keys` stays strict. The metadata is the one
    DCP already read.
    """

    def __init__(self, *args, optional_keys=(), **kwargs):
        super().__init__(*args, **kwargs)
        self.optional_keys = frozenset(optional_keys)
        self.skipped_keys = []

    def set_up_planner(self, state_dict, metadata=None, is_coordinator=False):
        super().set_up_planner(state_dict, metadata, is_coordinator)
        self.drop_optional_missing(metadata)

    def drop_optional_missing(self, metadata):
        if metadata is None:
            return
        for name in sorted(self.optional_keys):
            if name in self.state_dict and name not in metadata.state_dict_metadata:
                del self.state_dict[name]
                self.skipped_keys.append(name)


class OptionalKeysLoadPlanner(OptionalKeysMixin, DefaultLoadPlanner):
    """Default CPU load planner with the same optional-key rule."""


class MetaLoadPlanner(OptionalKeysMixin, DefaultLoadPlanner):
    # Override default_planner.py:318-335. Do NOT invoke _init_state_dict
    # (planner_helpers.py:496-550): it materializes meta tensors and asks for a PG.
    def set_up_planner(
        self,
        state_dict: STATE_DICT_TYPE,
        metadata: Metadata | None = None,
        is_coordinator: bool = False,
    ) -> None:
        flat, mappings = flatten_state_dict(state_dict)
        if any(isinstance(t, torch.Tensor) and not t.is_meta for t in flat.values()):
            raise TypeError("MetaLoadPlanner requires a meta tensor skeleton")
        self.flatten_state_dict = True
        self.flatten_sharded_tensors = False
        self.original_state_dict = state_dict
        self.state_dict = flat
        self.mappings = mappings
        self.metadata = metadata
        self.is_coordinator = is_coordinator
        self.drop_optional_missing(metadata)
