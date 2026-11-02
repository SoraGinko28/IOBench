"""Five-repeat save-only rerun of the 3B 4-rank shared-safetensors confirmation cell (64 MiB ring)."""
import sys
from pathlib import Path
ROOT = Path("/data00/home/wutingjun/AIStore/Meta-Device-DCP-Save-Load")
sys.path.insert(0, str(ROOT))
from scripts.run_gdt_safetensors import Runner
from scripts.run_multirank_meta_cpu import arguments

run = ROOT / "validation/rerun-3b4r-x5-20261101/3b-4r-safetensors"
args = arguments(run, "3b", 4, "safetensors", "ring", writer_limit=4)
args.repeats, args.meta_load_sink_mib = 5, 64
args.save_only = True
args.requested_writer_limit = 0
args.safetensors_layout = "shared"
args.warm_block_read_tolerance_bytes = 64 * 1024
Runner(args).run()
