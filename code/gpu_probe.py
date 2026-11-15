"""
How busy the GPU is with OTHER work, measured while this process is idle.

On the server the GPU is shared: another workload, outside the container,
holds 41 GB and computes on and off, and nvidia-smi inside the container
cannot list it. Its load slows our kernels and inflates the wall-clock
train/unlearn times, so every timed run records the load just before and
just after it, at moments when this process has nothing queued on the GPU.
What nvidia-smi reports then is the other workload's share.

  background_load() -> max GPU utilisation (%) over a short window, or None
                       when nvidia-smi is not available
"""

import os
import shutil
import subprocess
import time

THRESHOLD = 10      # % above which a run counts as shared


def background_load(samples=3, gap=0.4):
    if shutil.which('nvidia-smi') is None:
        return None
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.synchronize()     # nothing of ours left on the GPU
    except Exception:
        pass
    gpu = (os.environ.get('CUDA_VISIBLE_DEVICES') or '0').split(',')[0] or '0'
    vals = []
    for i in range(samples):
        try:
            out = subprocess.run(
                ['nvidia-smi', '-i', gpu, '--query-gpu=utilization.gpu',
                 '--format=csv,noheader,nounits'],
                capture_output=True, text=True, timeout=10).stdout.strip()
            vals.append(int(out.splitlines()[0]))
        except Exception:
            pass
        if i + 1 < samples:
            time.sleep(gap)
    return max(vals) if vals else None


def wait_idle(max_wait_min=None, quiet_checks=3, gap=20):
    """Hold the next timed run until the other workload is quiet.

    Waits until background_load() stays below THRESHOLD for `quiet_checks`
    checks in a row (gap seconds apart), at most max_wait_min minutes; then
    the run starts anyway (its bg-load line still flags it). Off when
    max_wait_min is 0 / unset (env WAIT_IDLE_MIN), or without nvidia-smi.
    Returns (seconds waited, last load).
    """
    if max_wait_min is None:
        max_wait_min = float(os.environ.get('WAIT_IDLE_MIN', '0') or 0)
    if max_wait_min <= 0:
        return 0.0, None
    t0, quiet, load = time.time(), 0, None
    while time.time() - t0 < max_wait_min * 60:
        load = background_load()
        if load is None:
            return 0.0, None
        quiet = quiet + 1 if load < THRESHOLD else 0
        if quiet >= quiet_checks:
            break
        time.sleep(gap)
    waited = time.time() - t0
    print(f"       [idle-wait] waited {waited / 60:.1f} min, GPU background now {fmt(load)}"
          + ("" if quiet >= quiet_checks else "  (gave up waiting, run starts anyway)"),
          flush=True)
    return waited, load


def fmt(v):
    return 'n/a' if v is None else f'{v}%'
