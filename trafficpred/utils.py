import os
import platform
import random
import socket
import subprocess

import numpy as np
import torch


def pick_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def device_info():
    device = pick_device().type
    info = {"device": device, "gpu_name": "none", "gpu_count": 0,
            "torch_version": torch.__version__, "hostname": socket.gethostname()}
    if device == "cuda":
        names = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
        info.update(gpu_name=names[0] if len(set(names)) == 1 else ", ".join(names),
                    gpu_count=len(names),
                    gpu_memory_gb=round(torch.cuda.get_device_properties(0).total_memory / 2**30, 1),
                    cuda_version=torch.version.cuda)
    elif device == "mps":
        try:
            chip = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"],
                                  capture_output=True, text=True, timeout=5).stdout.strip()
        except Exception:
            chip = ""
        info.update(gpu_name=chip or f"Apple {platform.machine()}", gpu_count=1)
    if os.environ.get("SLURM_JOB_ID"):
        info["slurm_job_id"] = os.environ["SLURM_JOB_ID"]
    return info


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed % 2**32)
    torch.manual_seed(seed)


def rng_state():
    return {"cpu": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}


def set_rng_state(state):
    torch.set_rng_state(state["cpu"].cpu())
    if state.get("cuda") is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([s.cpu() for s in state["cuda"]])


def save_atomic(obj, path):
    if os.path.dirname(path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    torch.save(obj, tmp)
    os.replace(tmp, path)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def to_device(batch, device):
    x, y, *rest = (t.to(device, non_blocking=True) for t in batch)
    return x, y, (rest[0] if rest else None)


def predict(model, x, ext=None, **kwargs):
    return model(x, **kwargs) if ext is None else model(x, ext=ext, **kwargs)
