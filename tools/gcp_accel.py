#!/usr/bin/env python3
"""gcp_accel.py -- the GCP accelerator registry for fak benchmarking.

One place that names the GCP machine types fak can benchmark on, what GPU each
carries, and how to provision it. Everything else (the probe, the provisioner,
the one-touch driver) imports this so the machine-type strings, GPU model names,
and the Blackwell-first fallback ladder live in exactly one file.

Facts here were resolved from Google Cloud's GPU machine-type documentation
(cloud.google.com/compute/docs/gpus) on 2026-06-20. They are slow-moving but not
frozen -- re-confirm machine_type / accelerator_type strings against
`gcloud compute accelerator-types list` before a real run on a new region.

Nothing in this module calls gcloud or the network; it is pure data + helpers so
it imports cleanly anywhere (including the test suite) with no auth.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class AccelTier:
    """One GCP accelerator-optimized machine type fak can benchmark on."""

    slug: str  # short fak-side id, e.g. "a4-b200"
    machine_type: str  # exact gcloud machine type, e.g. "a4-highgpu-8g"
    accelerator_type: str  # exact `--accelerator type=` value, e.g. "nvidia-b200"
    gpu_label: str  # human GPU name, e.g. "NVIDIA B200 (Blackwell)"
    gpu_count: int
    gpu_mem_gb_each: int
    vcpus: int
    host_mem_gb: int
    # Generation rank: higher = newer silicon. Drives the Blackwell-first ladder.
    gen_rank: int
    # GPU micro-architecture family, for grouping in reports.
    arch: str
    # CUDA compute capability of the GPU (sm_XX without the "sm_").
    compute_capability: str
    # Rough on-demand list price, USD/hour for the WHOLE VM. Indicative only --
    # real price varies by region/commitment/spot; the provisioner never trusts
    # this for billing, it is just so the operator sees the order of magnitude.
    approx_usd_per_hour: float
    # Zones where this type is commonly offered. Not exhaustive; the probe
    # discovers the live truth. First entry is the provisioner's default zone.
    common_zones: tuple[str, ...]
    # Is this a Blackwell-class part (the literal ask)?
    blackwell: bool
    notes: str = ""


# The registry, richest (newest Blackwell) first. The provisioner's default
# fallback ladder walks this order and stops at the first tier with live quota.
TIERS: tuple[AccelTier, ...] = (
    AccelTier(
        slug="a4-b200",
        machine_type="a4-highgpu-8g",
        accelerator_type="nvidia-b200",
        gpu_label="NVIDIA B200 (Blackwell)",
        gpu_count=8,
        gpu_mem_gb_each=180,
        vcpus=224,
        host_mem_gb=3968,
        gen_rank=50,
        arch="blackwell",
        compute_capability="100",
        approx_usd_per_hour=90.0,
        common_zones=(
            "us-central1-b",
            "us-east5-a",
            "europe-west4-b",
        ),
        blackwell=True,
        notes=(
            "The literal 'Blackwell' target. 8x B200, 1,440 GB HBM3e total. "
            "On-demand is frequently quota-gated; capacity often needs a "
            "reservation or DWS flex-start. Probe quota before launching."
        ),
    ),
    AccelTier(
        slug="a4x-gb200",
        machine_type="a4x-highgpu-4g",
        accelerator_type="nvidia-gb200",
        gpu_label="NVIDIA GB200 (Grace Blackwell)",
        gpu_count=4,
        gpu_mem_gb_each=186,
        vcpus=140,
        host_mem_gb=884,
        gen_rank=49,
        arch="blackwell",
        compute_capability="100",
        approx_usd_per_hour=85.0,
        common_zones=(
            "us-central1-a",
            "us-east5-b",
        ),
        blackwell=True,
        notes=(
            "Grace+Blackwell superchip (GB200 NVL72 building block). 4 GPUs per "
            "VM. Usually a reservation/Calendar-mode part, the hardest to get "
            "on-demand. Second rung of the Blackwell-first ladder."
        ),
    ),
    AccelTier(
        slug="a3-ultra-h200",
        machine_type="a3-ultragpu-8g",
        accelerator_type="nvidia-h200-141gb",
        gpu_label="NVIDIA H200 (Hopper)",
        gpu_count=8,
        gpu_mem_gb_each=141,
        vcpus=224,
        host_mem_gb=2952,
        gen_rank=40,
        arch="hopper",
        compute_capability="90",
        approx_usd_per_hour=60.0,
        common_zones=(
            "us-central1-a",
            "us-east5-a",
            "europe-west1-b",
            "asia-southeast1-c",
        ),
        blackwell=False,
        notes=(
            "Not Blackwell, but the most-provisionable current-gen tier and the "
            "same 224-vCPU CPU profile as A4. The pragmatic 'latest actually "
            "available' fallback when B200/GB200 quota is unavailable."
        ),
    ),
    AccelTier(
        slug="a3-high-h100",
        machine_type="a3-highgpu-8g",
        accelerator_type="nvidia-h100-80gb",
        gpu_label="NVIDIA H100 (Hopper)",
        gpu_count=8,
        gpu_mem_gb_each=80,
        vcpus=208,
        host_mem_gb=1872,
        gen_rank=30,
        arch="hopper",
        compute_capability="90",
        approx_usd_per_hour=30.0,
        common_zones=(
            "us-central1-a",
            "us-east4-a",
            "europe-west4-a",
            "asia-northeast1-a",
        ),
        blackwell=False,
        notes=(
            "Widely available Hopper baseline. Last automatic fallback before "
            "the cheap single-GPU proof tier."
        ),
    ),
    AccelTier(
        slug="g2-l4",
        machine_type="g2-standard-8",
        accelerator_type="nvidia-l4",
        gpu_label="NVIDIA L4 (Ada)",
        gpu_count=1,
        gpu_mem_gb_each=24,
        vcpus=8,
        host_mem_gb=32,
        gen_rank=12,
        arch="ada",
        compute_capability="89",
        approx_usd_per_hour=0.85,
        common_zones=(
            "us-central1-a",
            "us-east4-a",
            "europe-west4-a",
            "asia-east1-a",
        ),
        blackwell=False,
        notes=(
            "Cheap single-GPU Ada tier. Not for headline numbers -- it exists so "
            "the full provision->bench->teardown->catalog loop can be proven for "
            "a few dollars before spending on a Blackwell node."
        ),
    ),
    AccelTier(
        slug="n1-t4",
        machine_type="n1-standard-8",
        # GCP's accelerator string for the T4 is "nvidia-tesla-t4" (the Tesla
        # prefix survives only on the older Turing/Volta parts). "nvidia-t4" is
        # NOT a valid type and makes `accelerator-types list` / the probe report
        # the SKU as NOT_OFFERED. Verified live 2026-06-20 against
        # `gcloud compute accelerator-types list`.
        accelerator_type="nvidia-tesla-t4",
        gpu_label="NVIDIA T4 (Turing)",
        gpu_count=1,
        gpu_mem_gb_each=16,
        vcpus=8,
        host_mem_gb=30,
        gen_rank=10,
        arch="turing",
        compute_capability="75",
        approx_usd_per_hour=0.55,
        common_zones=(
            "us-central1-a",
            "us-east1-c",
            "europe-west1-b",
            "asia-east1-a",
        ),
        blackwell=False,
        notes=(
            "Cheapest, most-broadly-available single-GPU tier (Turing). The T4 "
            "attaches to the older N1 family, not an accelerator-optimized type. "
            "This is the de-risk-the-plumbing tier: most projects carry a "
            "default 1-T4-per-region quota, so the full provision->bench-> "
            "teardown->catalog loop can run for well under $1 before any "
            "Blackwell quota lands."
        ),
    ),
)


# Slug -> tier, for O(1) lookup.
_BY_SLUG = {t.slug: t for t in TIERS}


def by_slug(slug: str) -> Optional[AccelTier]:
    return _BY_SLUG.get(slug)


def fallback_ladder(blackwell_only: bool = False) -> list[AccelTier]:
    """The provisioner's preference order: newest silicon first.

    With blackwell_only=True, only the Blackwell-class tiers are returned -- the
    strict reading of "ideally Blackwell". The default (False) appends the
    H200/H100/L4 fallbacks so "whatever latest is actually available" still
    yields a runnable node when B200/GB200 quota is missing.
    """
    tiers = sorted(TIERS, key=lambda t: t.gen_rank, reverse=True)
    if blackwell_only:
        return [t for t in tiers if t.blackwell]
    return tiers


def proof_tier() -> AccelTier:
    """The cheapest tier, for de-risking the plumbing before a real run."""
    return min(TIERS, key=lambda t: t.approx_usd_per_hour)


def boot_image() -> tuple[str, str]:
    """The image family + project fak's GCP nodes boot from.

    Google's Deep Learning VM / CUDA images ship the NVIDIA driver + CUDA
    toolkit preinstalled, which is what the on-VM llama.cpp CUDA build needs.
    Returns (image_family, image_project).
    """
    # CUDA base image; NVIDIA driver + CUDA toolkit preinstalled. The DLVM image
    # families rev with the CUDA/driver version -- the older "common-cu124-debian-11"
    # was DELISTED (create fails "image family ... was not found"). Verified live
    # 2026-06-20: "common-cu129-ubuntu-2204-nvidia-580" (CUDA 12.9, driver 580,
    # Ubuntu 22.04) resolves to a READY image. Re-confirm with
    # `gcloud compute images list --project=deeplearning-platform-release --filter=family~cu1`
    # if a create ever fails on the image family again.
    return ("common-cu129-ubuntu-2204-nvidia-580", "deeplearning-platform-release")


def accelerator_flag(tier: AccelTier) -> str:
    """The `--accelerator` value for `gcloud compute instances create`."""
    return f"type={tier.accelerator_type},count={tier.gpu_count}"


def emit_shell(slug: str, prefix: str = "GLM") -> str:
    """Eval-able shell assignments for one tier, so a bash provisioner reads the
    machine-type / accelerator / image strings from THIS registry instead of
    re-hardcoding them (the module's whole reason for existing). Keys are
    `<PREFIX>_<FIELD>`; values are single-quote-shell-safe. Raises KeyError on an
    unknown slug. Pairs with scripts/gcp-glm-serve.sh:  eval "$(python tools/gcp_accel.py --emit-shell a3-ultra-h200)".
    """
    import shlex

    t = by_slug(slug)
    if t is None:
        raise KeyError(slug)
    fam, proj = boot_image()
    pairs = {
        "SLUG": t.slug,
        "MACHINE_TYPE": t.machine_type,
        "ACCEL_FLAG": accelerator_flag(t),
        "ACCEL_TYPE": t.accelerator_type,
        "GPU_COUNT": str(t.gpu_count),
        "GPU_LABEL": t.gpu_label,
        "COMPUTE_CAP": t.compute_capability,
        "DEFAULT_ZONE": t.common_zones[0] if t.common_zones else "",
        "IMAGE_FAMILY": fam,
        "IMAGE_PROJECT": proj,
        "APPROX_USD_HR": f"{t.approx_usd_per_hour:.2f}",
        "BLACKWELL": "1" if t.blackwell else "0",
    }
    return "\n".join(f"{prefix}_{k}={shlex.quote(v)}" for k, v in pairs.items())


def _print_table() -> None:
    """The human sanity dump: `python tools/gcp_accel.py` with no args."""
    print(f"{'slug':14} {'machine_type':18} {'gpu':28} {'cap':4} {'~$/hr':>7}")
    for t in fallback_ladder():
        tag = " [blackwell]" if t.blackwell else ""
        print(
            f"{t.slug:14} {t.machine_type:18} "
            f"{t.gpu_count}x {t.gpu_label:24} sm_{t.compute_capability:3} "
            f"{t.approx_usd_per_hour:7.2f}{tag}"
        )


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="GCP accelerator registry for fak.")
    ap.add_argument(
        "--emit-shell",
        metavar="SLUG",
        help="print eval-able shell assignments for one tier (for gcp-glm-serve.sh)",
    )
    ap.add_argument(
        "--prefix",
        default="GLM",
        help="variable-name prefix for --emit-shell (default GLM)",
    )
    args = ap.parse_args()
    if args.emit_shell:
        try:
            print(emit_shell(args.emit_shell, prefix=args.prefix))
        except KeyError:
            slugs = ", ".join(t.slug for t in TIERS)
            sys.stderr.write(f"unknown tier slug: {args.emit_shell!r} (known: {slugs})\n")
            raise SystemExit(2)
    else:
        _print_table()
