# Core runner subprocess options before slimming

Recorded before removing flags or paths. The shared `phase_command` constructs the `python -m metasl.run` command.

## scripts/run_gdtensor_trace_matrix.py

`phase_command`: `--action`, `--bundle`, `--checkpoint`, `--checkpoint-format`, `--checkpoint-state`, `--compute-dtype`, `--cuda-devices`, `--cuda-share-devices`, `--data-pattern`, `--dcp-file-layout`, `--evict-cache`, `--implementation`, `--io-backend`, `--io-soft-pss-gib`, `--max-pss-gib`, `--measure-cache-residency`, `--metadata-limit`, `--min-available-gib`, `--output`, `--parallel`, `--ranks`, `--reader-limit`, `--safetensors-max-file-mib`, `--sharding`, `--standard-device`, `--threads`, `--timeout`, `--trace-io`, `--validate-values`, `--writer-limit`

Subprocess invocation source (preserved verbatim):
```python
def phase_command(
    python, bundle, checkpoint, output, ranks, action, device="meta", *,
    data_pattern="position", validate_values=True, cache_control=False,
    checkpoint_format="dcp", safetensors_max_file_mib=None,
):
    command = [
        str(python),
        "-m",
        "metasl.run",
        "--action",
        action,
        "--io-backend",
        "gdtensor",
        "--checkpoint-state",
        "model",
        "--checkpoint-format",
        checkpoint_format,
        "--dcp-file-layout",
        "rank",
        "--bundle",
        str(bundle),
        "--checkpoint",
        str(checkpoint),
        "--output",
        str(output),
        "--ranks",
        str(ranks),
        "--implementation",
        "simulated" if device == "meta" else "standard",
        "--sharding",
        "fsdp2",
        "--parallel",
        f"dp_shard={ranks}",
        "--threads",
        "1",
        "--compute-dtype",
        "bf16",
        "--data-pattern",
        data_pattern,
        "--writer-limit" if action == "save" else "--reader-limit",
        "1" if action == "save" else "0",
        "--metadata-limit",
        "4",
        "--max-pss-gib",
        "26",
        "--io-soft-pss-gib",
        "26",
        "--min-available-gib",
        "3",
        "--timeout",
        "1800",
    ]
    if safetensors_max_file_mib is not None:
        command += ["--safetensors-max-file-mib", str(safetensors_max_file_mib)]
    command += ["--trace-io"]
    if action == "load" and cache_control:
        command += ["--evict-cache", "--measure-cache-residency"]
    if action == "load" and validate_values:
        command += ["--validate-values"]
    if device == "cpu":
        command += ["--standard-device", "cpu"]
    if device == "cuda":
        command += [
            "--standard-device",
            "cuda",
            "--cuda-share-devices",
            "--cuda-devices",
            ",".join(["0"] * ranks),
        ]
    return command
```

## scripts/run_gdt_safetensors.py

`phase`: `--content-audit`, `--io-backend`

Subprocess invocation source (preserved verbatim):
```python
The safetensors runner calls phase_command with ranks=1, then overrides --io-backend (dcp or gdtensor) for old identity gates and adds --content-audit none for measurements.
```

The matrix bundle subprocess is `python -m metasl.prepare --model-id MODEL --bundle BUNDLE --dtype bf16 --local-files-only`; environment evidence runs `python -c PROBE`. Runner-only options are documented in the pre-edit help captured below.

All option strings present in scripts/run_gdtensor_trace_matrix.py: `--action`, `--bundle`, `--cache-control`, `--cache-control selects cold and conflicts with --cache-mode`, `--cache-mode`, `--checkpoint`, `--checkpoint-format`, `--checkpoint-state`, `--compute-dtype`, `--configs`, `--continuous-io`, `--continuous-io-interval`, `--cuda-devices`, `--cuda-share-devices`, `--data-pattern`, `--dcp-file-layout`, `--dtype`, `--evict-cache`, `--implementation`, `--io-backend`, `--io-soft-pss-gib`, `--keep-checkpoints`, `--local-files-only`, `--max-pss-gib`, `--measure-cache-residency`, `--metadata-limit`, `--min-available-gib`, `--model-id`, `--models`, `--output`, `--parallel`, `--python`, `--ranks`, `--reader-limit`, `--resume`, `--safetensors-max-file-mib`, `--sharding`, `--standard-device`, `--threads`, `--timeout`, `--trace-io`, `--validate-values`, `--writer-limit`

All option strings present in scripts/run_gdt_safetensors.py: `--checkpoint-format`, `--configs`, `--content-audit`, `--gate-configs`, `--gate-models`, `--gate-patterns`, `--io-backend`, `--models`, `--output`, `--repeats`, `--repeats must be positive`, `--safetensors-max-file-mib`, `--safetensors-max-file-mib must be positive`, `--safetensors-max-file-mib requires safetensors format`, `--stage`
