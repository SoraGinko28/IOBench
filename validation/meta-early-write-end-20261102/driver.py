"""Diagnosis harness; no benchmark or system configuration changes."""
import json
import mmap
import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path('/data00/home/wutingjun/AIStore/Meta-Device-DCP-Save-Load')
OUT = ROOT / 'validation/meta-early-write-end-20261102'
KEYS = ('nr_dirty nr_writeback nr_dirty_threshold nr_dirty_background_threshold '
        'nr_free_pages nr_active_file nr_inactive_file nr_anon_pages').split()


def proc_values(path):
    return {k.rstrip(':'): int(v.split()[0]) for k, v in
            (line.split(maxsplit=1) for line in Path(path).read_text().splitlines())
            if v.split()[0].isdigit()}


def descendants(pid):
    result = []
    try:
        children = Path(f'/proc/{pid}/task/{pid}/children').read_text().split()
    except OSError:
        return result
    for child in children:
        result.append(int(child))
        result.extend(descendants(child))
    return result


def sampler(output, parent):
    running = True
    def stop(*_):
        nonlocal running
        running = False
    signal.signal(signal.SIGTERM, stop)
    deadline = time.perf_counter_ns()
    pids, scan = [], 0
    with Path(output).open('w', buffering=1) as stream:
        print(json.dumps({'ready_ns': deadline, 'clock': 'perf_counter_ns',
                          'realtime_ns': time.time_ns(), 'pid': os.getpid()}), flush=True)
        while running:
            start = time.perf_counter_ns()
            vm = proc_values('/proc/vmstat')
            mem = proc_values('/proc/meminfo')
            if start >= scan:
                pids = [p for p in descendants(int(parent)) if p != os.getpid()]
                scan = start + 500_000_000
            waits = {}
            for pid in pids:
                try:
                    waits[str(pid)] = Path(f'/proc/{pid}/wchan').read_text().strip()
                except OSError:
                    pass
            journal = Path('/proc/fs/jbd2/vdb-8/info').read_text()
            disk = next(line.split() for line in Path('/proc/diskstats').read_text().splitlines()
                        if line.split()[2] == 'vdb')
            row = dict(absolute_ns=start, realtime_ns=time.time_ns(),
                       vmstat={k: vm[k] for k in KEYS},
                       meminfo_kib={k: mem[k] for k in ('Dirty', 'Writeback', 'MemAvailable')},
                       wchan=waits, journal_info=journal, diskstats=disk,
                       read_duration_ns=time.perf_counter_ns()-start)
            stream.write(json.dumps(row, separators=(',', ':'))+'\n')
            deadline += 50_000_000
            delay = (deadline-time.perf_counter_ns())/1e9
            if delay > 0:
                time.sleep(delay)
            elif delay < -0.05:
                deadline = time.perf_counter_ns()


def ballast(size):
    size = int(size)
    data = mmap.mmap(-1, size, flags=mmap.MAP_PRIVATE | mmap.MAP_ANONYMOUS,
                     prot=mmap.PROT_READ | mmap.PROT_WRITE)
    block = b'\x01' * (1024**2)
    for offset in range(0, size, len(block)):
        data[offset:min(offset+len(block),size)] = block[:min(len(block),size-offset)]
    del block
    print(json.dumps(dict(pid=os.getpid(), bytes=size, fully_faulted_ns=time.perf_counter_ns(),
                          smaps_rollup=Path('/proc/self/smaps_rollup').read_text())), flush=True)
    sys.stdin.buffer.read(1)
    data.close()


def main():
    sys.path.insert(0, str(ROOT))
    from scripts.run_gdt_safetensors import Runner, archive_metadata
    from scripts.run_gdtensor_trace_matrix import dump, load
    from scripts.run_multirank_meta_cpu import arguments
    from statistics import median

    old = ROOT/'validation/rerun-3b4r-x5-20261101/3b-4r-safetensors/measurements/3b'
    peaks = {c: [load(p)['peak_sampled_total_pss_bytes']
                 for p in sorted(old.glob(f'r*-{c}/save/summary.json'))] for c in ('cpu','meta')}
    size = round((median(peaks['cpu'])-median(peaks['meta']))/4096)*4096
    args = arguments(OUT, '3b', 4, 'safetensors', 'ring', writer_limit=4, save_only=True)
    args.meta_load_sink_mib = 64
    args.safetensors_layout = 'shared'
    args.requested_writer_limit = 0

    class Diagnosis(Runner):
        def measurements(self):
            shutil.copyfile(__file__, self.root/'driver.py')
            failed = Path('/tmp/meta-early-write-end-failed-launch-20261102')
            if failed.exists():
                shutil.copytree(failed, self.root/'failed-sandbox-launch')
            dump(self.root/'protocol.json', dict(ballast_bytes=size, source_peak_pss_bytes=peaks,
                 derivation='page-rounded difference of prior five-run median peak PSS',
                 sample_interval_ms=50, clock='absolute perf_counter_ns plus realtime_ns',
                 order=[['cpu','meta','meta-ballast'],['meta','meta-ballast','cpu'],
                        ['meta-ballast','cpu','meta']],
                 kernel=os.uname().release,
                 ext4_options=Path('/proc/fs/ext4/vdb/options').read_text(),
                 vm_settings={k:Path('/proc/sys/vm/'+k).read_text().strip() for k in
                              ('dirty_ratio','dirty_background_ratio','dirty_writeback_centisecs',
                               'dirty_expire_centisecs','dirty_bytes','dirty_background_bytes')},
                 meminfo_before=Path('/proc/meminfo').read_text()))
            rows = []
            labels = ['cpu', 'meta', 'meta-ballast']
            for repeat in range(1,4):
                offset = repeat-1
                for label in labels[offset:]+labels[:offset]:
                    case = self.root/'measurements'/'3b'/f'r{repeat}-{label}'
                    case.mkdir(parents=True)
                    self.update(current=dict(repeat=repeat, condition=label, stage='waiting-idle'))
                    # A clean baseline without flushing or evicting unrelated files.
                    stable = 0
                    while stable < 2:
                        foreign = subprocess.run(['pgrep','-af','metasl[.]run'],capture_output=True,text=True)
                        mem = proc_values('/proc/meminfo')
                        okay = foreign.returncode == 1 and mem['Dirty'] < 65536 and mem['Writeback'] < 16384
                        stable = stable+1 if okay else 0
                        if not okay:
                            self.update(waiting_foreign=foreign.stdout, waiting_meminfo=mem)
                        time.sleep(1)
                    checkpoint = self.ckroot/f'r{repeat}-{label}'
                    self.verify_ownership()
                    checkpoint.mkdir()
                    sample = subprocess.Popen([sys.executable,__file__,'sampler',str(case/'vmstat.jsonl'),
                                               str(os.getpid())],stdout=subprocess.PIPE,text=True)
                    ready = json.loads(sample.stdout.readline())
                    ball = None
                    try:
                        diagnostic = dict(sampler=ready, baseline_meminfo=proc_values('/proc/meminfo'))
                        if label == 'meta-ballast':
                            ball = subprocess.Popen([sys.executable,__file__,'ballast',str(size)],
                                                    stdin=subprocess.PIPE,stdout=subprocess.PIPE,text=True)
                            diagnostic['ballast'] = json.loads(ball.stdout.readline())
                        diagnostic['launch_ns'] = time.perf_counter_ns()
                        dump(case/'diagnostic.json', diagnostic)
                        phase_rows = self.phase(case,checkpoint,'3b','cpu' if label=='cpu' else 'meta','save')
                        diagnostic['return_ns'] = time.perf_counter_ns()
                        if ball is not None and ball.poll() is not None:
                            raise RuntimeError('Ballast exited before save returned')
                        rows.extend(dict(row,repeat=repeat,condition=label) for row in phase_rows)
                        archive_metadata(checkpoint,case,'safetensors')
                        sizes = {p.name:p.stat().st_size for p in checkpoint.iterdir()}
                        dump(case/'completed.json',dict(rows=phase_rows,inventory_bytes=sizes,
                                                       checkpoint_removed=True))
                        dump(case/'diagnostic.json',diagnostic)
                        self.progress['completed'].append(f'r{repeat}-{label}')
                        self.update()
                        dump(self.root/'measurement-results.json',rows)
                    finally:
                        if ball is not None:
                            ball.stdin.write('x'); ball.stdin.flush(); ball.wait(timeout=30)
                        sample.terminate(); sample.wait(timeout=10)
                        self.verify_ownership()
                        if checkpoint.exists():
                            shutil.rmtree(checkpoint)
    Diagnosis(args).run()


if __name__ == '__main__':
    if len(sys.argv)>1 and sys.argv[1]=='sampler':
        sampler(sys.argv[2],sys.argv[3])
    elif len(sys.argv)>1 and sys.argv[1]=='ballast':
        ballast(sys.argv[2])
    else:
        main()
