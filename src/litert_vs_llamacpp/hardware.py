"""Hardware detection for optimal macOS Apple Silicon and Linux settings."""

from __future__ import annotations

import dataclasses
import os
import platform
import subprocess


def _run_cmd(cmd: list[str]) -> str | None:
    try:
        out = subprocess.check_output(cmd, stderr=subprocess.DEVNULL, text=True, timeout=5)
        return out.strip()
    except (OSError, subprocess.SubprocessError):
        return None


@dataclasses.dataclass(frozen=True)
class HardwareInfo:
    os_name: str
    arch: str
    cpu_model: str
    logical_cpus: int
    perf_cpus: int
    memory_gb: float
    is_apple_silicon: bool
    gpu_summary: str

    def to_dict(self) -> dict[str, object]:
        return dataclasses.asdict(self)


def detect_hardware() -> HardwareInfo:
    os_name = platform.system()
    arch = platform.machine()
    logical_cpus = os.cpu_count() or 1
    is_apple_silicon = os_name == "Darwin" and arch == "arm64"

    cpu_model = platform.processor() or arch
    perf_cpus = logical_cpus
    memory_gb = 0.0
    gpu_summary = "unknown"

    if os_name == "Darwin":
        brand = _run_cmd(["sysctl", "-n", "machdep.cpu.brand_string"])
        if brand:
            cpu_model = brand
        # On Apple Silicon, P-cores (hw.perflevel0.physicalcpu) give the best CPU LLM
        # throughput without contention from slower E-cores.
        p_cores = _run_cmd(["sysctl", "-n", "hw.perflevel0.physicalcpu"])
        phys_cores = _run_cmd(["sysctl", "-n", "hw.physicalcpu"])
        if p_cores and p_cores.isdigit() and int(p_cores) > 0:
            perf_cpus = int(p_cores)
        elif phys_cores and phys_cores.isdigit() and int(phys_cores) > 0:
            perf_cpus = int(phys_cores)
        mem_bytes = _run_cmd(["sysctl", "-n", "hw.memsize"])
        if mem_bytes and mem_bytes.isdigit():
            memory_gb = round(int(mem_bytes) / (1024**3), 1)
        gpu_summary = f"{cpu_model} Metal GPU (unified memory)" if is_apple_silicon else "macOS GPU"
    elif os_name == "Linux":
        lscpu = _run_cmd(["lscpu"])
        if lscpu:
            models = [
                line.split(":", 1)[1].strip()
                for line in lscpu.splitlines()
                if line.startswith("Model name:")
            ]
            if models:
                cpu_model = " + ".join(dict.fromkeys(models))
        try:
            with open("/proc/meminfo", encoding="utf-8") as fh:
                for line in fh:
                    if line.startswith("MemTotal:"):
                        kb = int(line.split()[1])
                        memory_gb = round(kb / (1024**2), 1)
                        break
        except OSError:
            pass
        nvidia = _run_cmd(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"])
        gpu_summary = nvidia.splitlines()[0].strip() if nvidia else "none/unknown"

    return HardwareInfo(
        os_name=os_name,
        arch=arch,
        cpu_model=cpu_model,
        logical_cpus=logical_cpus,
        perf_cpus=perf_cpus,
        memory_gb=memory_gb,
        is_apple_silicon=is_apple_silicon,
        gpu_summary=gpu_summary,
    )
