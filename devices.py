"""
One place to pick the training device: CUDA (NVIDIA) -> MPS (Apple Silicon) -> CPU.

    from devices import pick_device
    device = pick_device()

PYTORCH_ENABLE_MPS_FALLBACK=1 lets the few ops MPS doesn't implement run on the CPU
instead of raising an error. It only takes effect if set before torch is imported,
so it's set here (when this module is imported first) and in train_all.sh.
"""

import os

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import torch  # noqa: E402


def pick_device(verbose=False):
    if torch.cuda.is_available():
        dev = torch.device("cuda")
    elif getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        dev = torch.device("mps")
    else:
        dev = torch.device("cpu")
    if verbose:
        print(f"device: {dev}")
    return dev


def device_info():
    """What hardware this run is on, as flat values for the W&B run config:
    device, gpu_name, gpu_count, gpu_memory_gb, cuda_version, torch_version, hostname,
    slurm_job_id. On a Mac, gpu_name is the chip (e.g. "Apple M2 Pro")."""
    import platform
    import socket
    import subprocess

    dev = pick_device().type
    info = {"device": dev, "gpu_name": "none", "gpu_count": 0,
            "torch_version": torch.__version__, "hostname": socket.gethostname()}
    if dev == "cuda":
        n = torch.cuda.device_count()
        names = [torch.cuda.get_device_name(i) for i in range(n)]
        info.update(gpu_name=names[0] if len(set(names)) == 1 else ", ".join(names),
                    gpu_count=n,
                    gpu_memory_gb=round(torch.cuda.get_device_properties(0).total_memory / 2**30, 1),
                    cuda_version=torch.version.cuda)
    elif dev == "mps":
        try:
            chip = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"],
                                  capture_output=True, text=True, timeout=5).stdout.strip()
        except Exception:
            chip = ""
        info.update(gpu_name=chip or f"Apple {platform.machine()}", gpu_count=1)
    if os.environ.get("SLURM_JOB_ID"):
        info["slurm_job_id"] = os.environ["SLURM_JOB_ID"]
    return info
