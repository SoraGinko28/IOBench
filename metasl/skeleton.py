"""Meta Qwen skeleton: structural manifest, bundle preparation and model rebuild."""

import hashlib
import json

import torch
from transformers import AutoConfig, AutoModelForCausalLM

from metasl import TORCH_VERSION

DTYPES = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}


def require_torch():
    allowed = (TORCH_VERSION, "2.14.0+cu126")
    if torch.__version__ not in allowed:
        raise RuntimeError(
            f"Source-specific implementation requires torch {' or '.join(allowed)}; "
            f"got {torch.__version__}"
        )


def model_structure(model):
    """Compare structural semantics, not device, addresses or numerical values."""
    aliases = {}
    entries = []
    for kind, members in (
        ("parameter", model.named_parameters(remove_duplicate=False)),
        ("buffer", model.named_buffers(remove_duplicate=False)),
    ):
        for name, tensor in members:
            aliases.setdefault(id(tensor), []).append(f"{kind}:{name}")
            entries.append(
                dict(
                    kind=kind,
                    name=name,
                    shape=list(tensor.shape),
                    dtype=str(tensor.dtype),
                    stride=list(tensor.stride()),
                    requires_grad=tensor.requires_grad,
                )
            )
    return dict(
        model_class=f"{type(model).__module__}.{type(model).__qualname__}",
        modules=[
            (name, f"{type(module).__module__}.{type(module).__qualname__}")
            for name, module in model.named_modules(remove_duplicate=False)
        ],
        tensors=entries,
        aliases=sorted(sorted(names) for names in aliases.values() if len(names) > 1),
        state_keys=list(model.state_dict()),
    )


def signature(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def model_from_config(config, dtype):
    """Keep Qwen3.5's official vision and language model in its checkpoint skeleton.

    AutoModelForCausalLM accepts the composite Qwen3.5 config but extracts only
    text_config. The image/text mapping selects the official conditional model
    declared by Qwen/Qwen3.5-27B instead, without changing its configuration.
    """
    if config.model_type == "qwen3_5":
        from transformers import AutoModelForImageTextToText

        return AutoModelForImageTextToText.from_config(config, dtype=dtype)
    return AutoModelForCausalLM.from_config(config, dtype=dtype)


def decoder_layers(model):
    """Decoder blocks of the supported official Qwen model classes."""
    if model.config.model_type == "qwen3_5":
        return model.model.language_model.layers
    return model.model.layers


def build_model(bundle, device):
    """Rebuild the model from a prepared bundle and verify its structure signature."""
    manifest = json.loads((bundle / "model_manifest.json").read_text())
    config = AutoConfig.from_pretrained(bundle, local_files_only=True)
    with torch.device(device):
        model = model_from_config(config, DTYPES[manifest["dtype"]])
    model.eval()
    assert signature(model_structure(model)) == manifest["structure_sha256"]
    return model, manifest


def prepare_bundle(
    model_id, dtype, bundle, local_files_only=False, config_cache=None, verify_cpu_structure=False
):
    """Fetch the config, build the meta model and write config + structure manifest."""
    if bundle.exists() and any(bundle.iterdir()):
        raise ValueError("Prepare requires an empty bundle directory")
    bundle.mkdir(parents=True, exist_ok=True)
    config = AutoConfig.from_pretrained(
        model_id, local_files_only=local_files_only, cache_dir=config_cache
    )
    with torch.device("meta"):
        model = model_from_config(config, DTYPES[dtype])
    model.eval()
    structure = model_structure(model)
    config.save_pretrained(bundle)
    manifest = dict(
        format_version=1,
        model_id=model_id,
        dtype=dtype,
        torch_version=torch.__version__,
        structure=structure,
        structure_sha256=signature(structure),
        cpu_structure_verified=False,
    )
    (bundle / "model_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    if verify_cpu_structure:
        cpu, _ = build_model(bundle, "cpu")
        assert signature(model_structure(cpu)) == signature(structure)
        manifest["cpu_structure_verified"] = True
        (bundle / "model_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return dict(
        action="prepare",
        structure_sha256=manifest["structure_sha256"],
        cpu_structure_verified=manifest["cpu_structure_verified"],
        model_id=model_id,
    )

