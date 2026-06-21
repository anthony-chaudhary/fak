#!/usr/bin/env python3
"""gcp_gpu_probe.py -- read-only "what can I actually launch?" probe for GCP GPUs.

The hard reality behind "benchmark on Blackwell": B200/GB200 capacity is
quota-gated and often waitlisted, so the literal machine type you want may not be
launchable in your project today. This probe answers that BEFORE the provisioner
spends money or fails halfway:

  - which credential/project gcloud is using,
  - per-tier GPU quota in candidate regions (limit vs usage),
  - any reservations that match a tier,
  - whether the accelerator type is even offered in a zone,

and folds it into one verdict per tier: PROVISIONABLE / NO_QUOTA / NOT_OFFERED /
UNKNOWN. The provisioner's fallback ladder consumes this to pick the best tier
that is actually PROVISIONABLE.

It is strictly read-only -- only `gcloud ... list/describe` and quota reads, never
a create. If auth is stale it says so with the exact `gcloud auth login` fix and
exits non-zero, rather than pretending.

Usage:
  python tools/gcp_gpu_probe.py                       # probe the Blackwell ladder
  python tools/gcp_gpu_probe.py --all-tiers           # include H200/H100/L4
  python tools/gcp_gpu_probe.py --project P --region us-central1
  python tools/gcp_gpu_probe.py --json out.json       # machine-readable verdict
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import urllib.error
import urllib.request
from typing import Optional

import gcp_accel


SCHEMA = "fak.gcp-gpu-probe.v1"

# GA Cloud Quotas API quota IDs. The legacy `gcloud compute regions describe`
# quota array does NOT carry Blackwell/Hopper-next GPU families -- it stops at
# A100/L4/T4 -- so reading it returns a FALSE limit=0 for B200/H200/H100. The
# modern truth lives in the Cloud Quotas API, dimensioned by gpu_family+region,
# plus a project-global cap (GPUS-ALL-REGIONS) that silently overrides any
# per-family grant. We read both via REST with the gcloud access token so no
# extra gcloud component (beta) install -- which needs admin on the SDK dir --
# is required.
QUOTAS_API = "https://cloudquotas.googleapis.com/v1"
QUOTA_FAMILY_REGION = "GPUS-PER-GPU-FAMILY-per-project-region"
QUOTA_GLOBAL = "GPUS-ALL-REGIONS-per-project"

# accelerator-type string -> the gpu_family dimension value the Quotas API uses.
GPU_FAMILY = {
    "nvidia-b200": "NVIDIA_B200",
    "nvidia-gb200": "NVIDIA_GB200",
    "nvidia-h200-141gb": "NVIDIA_H200",
    "nvidia-h100-80gb": "NVIDIA_H100",
    "nvidia-l4": "NVIDIA_L4",
}

# Older GPU families (L4, T4, A100, ...) are NOT carried by the unified
# GPUS-PER-GPU-FAMILY quota -- they keep their own per-family quota IDs. The
# newer Blackwell/Hopper-next families (B200/GB200/H200/H100) live ONLY in the
# unified one. We read both so a launchable L4 (the cheap proof tier) isn't
# falsely reported as limit=0.
LEGACY_FAMILY_QUOTA = {
    "NVIDIA_L4": "NVIDIA-L4-GPUS-per-project-region",
    "NVIDIA_T4": "NVIDIA-T4-GPUS-per-project-region",
    "NVIDIA_A100": "NVIDIA-A100-GPUS-per-project-region",
    "NVIDIA_A100_80GB": "NVIDIA-A100-80GB-GPUS-per-project-region",
}


def utc_now() -> str:
    return _dt.datetime.now(_dt.timezone.utc).isoformat().replace("+00:00", "Z")


def zone_region(zone: str) -> str:
    """us-central1-b -> us-central1."""
    return zone.rsplit("-", 1)[0]


class GcloudError(RuntimeError):
    """gcloud failed; .stale is True when the cause is expired credentials."""

    def __init__(self, msg: str, stale: bool = False):
        super().__init__(msg)
        self.stale = stale


def _looks_stale(text: str) -> bool:
    t = (text or "").lower()
    return (
        "reauthentication failed" in t
        or "invalid_grant" in t
        or "there was a problem refreshing" in t
        or "gcloud auth login" in t
        or "do not have permission" in t and "credential" in t
    )


def run_gcloud(args: list[str], *, project: Optional[str] = None,
               account: Optional[str] = None, timeout: int = 90) -> str:
    """Run `gcloud <args>` and return stdout, raising GcloudError on failure.

    Always appends --format=json unless the caller already set a format, so the
    result parses cleanly. Detects the stale-credential case specifically so the
    caller can surface the one fix that matters.
    """
    exe = shutil.which("gcloud") or shutil.which("gcloud.cmd")
    if not exe:
        raise GcloudError("gcloud not found on PATH -- install the Google Cloud SDK.")
    cmd = [exe, *args]
    if project:
        cmd += ["--project", project]
    if account:
        cmd += ["--account", account]
    if not any(a.startswith("--format") for a in args):
        cmd += ["--format=json"]
    try:
        proc = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout,
        )
    except subprocess.TimeoutExpired as exc:
        raise GcloudError(f"gcloud timed out after {timeout}s: {' '.join(args)}") from exc
    if proc.returncode != 0:
        err = (proc.stderr or proc.stdout or "").strip()
        raise GcloudError(err or f"gcloud failed: {' '.join(args)}", stale=_looks_stale(err))
    return proc.stdout


def access_token(account: Optional[str]) -> str:
    """A bearer token for the active (or named) account, via gcloud.

    Calls gcloud directly (NOT through run_gcloud) so no --format is appended:
    `print-access-token` emits the raw token on stdout, and any --format wrapper
    can corrupt it. We strip aggressively -- a stray newline or quote in the
    Authorization header is exactly what yields a 401.
    """
    exe = shutil.which("gcloud") or shutil.which("gcloud.cmd")
    if not exe:
        raise GcloudError("gcloud not found on PATH -- install the Google Cloud SDK.")
    cmd = [exe, "auth", "print-access-token"]
    if account:
        cmd += ["--account", account]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
    if proc.returncode != 0:
        err = (proc.stderr or proc.stdout or "").strip()
        raise GcloudError(err or "gcloud auth print-access-token failed",
                          stale=_looks_stale(err))
    return (proc.stdout or "").strip().strip('"').strip()


def _quotas_get(project: str, quota_id: str, token: str) -> dict:
    """GET one quotaInfo resource from the GA Cloud Quotas API."""
    url = (f"{QUOTAS_API}/projects/{project}/locations/global/services/"
           f"compute.googleapis.com/quotaInfos/{quota_id}")
    req = urllib.request.Request(url, headers={"Authorization": f"Bearer {token}"})
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = ""
        try:
            body = exc.read().decode("utf-8")
        except Exception:  # pragma: no cover - best-effort detail
            pass
        raise GcloudError(f"quotas API {quota_id}: HTTP {exc.code} {body[:200]}",
                          stale=(exc.code in (401, 403))) from exc
    except urllib.error.URLError as exc:
        raise GcloudError(f"quotas API {quota_id}: {exc.reason}") from exc


def gpu_quota_map(project: str, token: str) -> dict:
    """Read the live GPU quota truth from the GA Quotas API.

    Returns {"global": <int>, "by_family_region": {(family, region): <int>}}.
    The global cap is the project-wide GPUS-ALL-REGIONS limit -- the one that
    was silently =1 and made every multi-GPU tier impossible regardless of any
    per-family grant.
    """
    out = {"global": None, "by_family_region": {}}
    try:
        g = _quotas_get(project, QUOTA_GLOBAL, token)
        dims = g.get("dimensionsInfos") or []
        if dims:
            out["global"] = _limit_val(dims[0])
    except GcloudError:
        pass  # global cap unreadable; treat as unknown, not zero
    try:
        fr = _quotas_get(project, QUOTA_FAMILY_REGION, token)
        for d in (fr.get("dimensionsInfos") or []):
            dim = d.get("dimensions") or {}
            fam = dim.get("gpu_family")
            region = dim.get("region")
            if fam and region:
                out["by_family_region"][(fam, region)] = _limit_val(d)
    except GcloudError:
        pass
    # Legacy per-family quotas (L4/T4/A100) -- only fetched for families that
    # actually need them, so a project with no old GPUs pays nothing extra.
    for fam, qid in LEGACY_FAMILY_QUOTA.items():
        try:
            r = _quotas_get(project, qid, token)
        except GcloudError:
            continue
        for d in (r.get("dimensionsInfos") or []):
            region = (d.get("dimensions") or {}).get("region")
            val = _limit_val(d)
            # The legacy quota lists EVERY region (most at 0); keep only the
            # ones with a real grant so the map stays small and truthful.
            if region and val and val > 0:
                out["by_family_region"][(fam, region)] = val
    return out


def _limit_val(dim_info: dict) -> Optional[float]:
    """Pull the effective limit out of a dimensionsInfos entry."""
    details = dim_info.get("details") or {}
    v = details.get("value")
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def active_identity(account: Optional[str], project: Optional[str]) -> dict:
    """Who/where are we? Resolves the active account + project from gcloud config."""
    out = {"account": account or "", "project": project or "", "ok": False}
    try:
        if not out["account"]:
            raw = run_gcloud(["auth", "list", "--filter=status:ACTIVE",
                              "--format=value(account)"])
            out["account"] = (raw or "").strip().splitlines()[0] if raw.strip() else ""
        if not out["project"]:
            raw = run_gcloud(["config", "get-value", "project", "--format=value(.)"])
            out["project"] = (raw or "").strip()
        out["ok"] = bool(out["account"] and out["project"])
    except GcloudError as exc:
        out["error"] = str(exc)
        out["stale"] = exc.stale
    return out


def reservations(project: Optional[str], account: Optional[str]) -> list[dict]:
    """All reservations visible to the project (may be empty / may error w/o perms)."""
    try:
        raw = run_gcloud(["compute", "reservations", "list", "--format=json"],
                         project=project, account=account)
        return json.loads(raw or "[]")
    except GcloudError:
        return []


def accelerator_offered(zone: str, accel_type: str, project: Optional[str],
                        account: Optional[str]) -> Optional[bool]:
    """Is accel_type offered in zone? None when the lookup itself fails."""
    try:
        raw = run_gcloud(
            ["compute", "accelerator-types", "list",
             f"--filter=zone:( {zone} ) AND name=( {accel_type} )",
             "--format=value(name)"],
            project=project, account=account,
        )
        return bool((raw or "").strip())
    except GcloudError:
        return None


def probe_tier(tier: gcp_accel.AccelTier, *, project: Optional[str],
               account: Optional[str], quota: dict,
               reservation_list: list[dict],
               zone_override: Optional[str]) -> dict:
    """One tier -> a verdict dict, judged against the GA quota map.

    A tier is PROVISIONABLE only if ALL hold in some candidate zone:
      - the accelerator type is offered there,
      - the per-family region quota covers a full VM's GPU count (or 0/unknown
        is rescued by a matching reservation),
      - the project-global GPU cap (GPUS-ALL-REGIONS) also covers it.
    The global cap is the subtle one: it can be 1 while a family grant looks
    fine, making every multi-GPU tier silently impossible.
    """
    zones = [zone_override] if zone_override else list(tier.common_zones)
    family = GPU_FAMILY.get(tier.accelerator_type, "")
    g_cap = quota.get("global")
    result = {
        "slug": tier.slug,
        "machine_type": tier.machine_type,
        "accelerator_type": tier.accelerator_type,
        "gpu_family": family,
        "gpu_label": tier.gpu_label,
        "gpu_count": tier.gpu_count,
        "blackwell": tier.blackwell,
        "approx_usd_per_hour": tier.approx_usd_per_hour,
        "global_gpu_cap": g_cap,
        "zones_checked": [],
        "verdict": "UNKNOWN",
        "reason": "",
    }
    matched_reservation = None
    for r in reservation_list:
        sku = (((r.get("specificReservation") or {}).get("instanceProperties") or {})
               .get("machineType", ""))
        if sku == tier.machine_type:
            matched_reservation = r.get("name")
            break
    result["reservation"] = matched_reservation

    # The project-global cap gates everything; check it once up front.
    global_blocks = (g_cap is not None and g_cap < tier.gpu_count
                     and matched_reservation is None)

    best = None
    for zone in zones:
        region = zone_region(zone)
        fam_limit = quota["by_family_region"].get((family, region))
        offered = accelerator_offered(zone, tier.accelerator_type, project, account)
        zinfo = {
            "zone": zone, "region": region, "gpu_family": family,
            "family_region_limit": fam_limit, "offered": offered,
        }
        result["zones_checked"].append(zinfo)
        fam_ok = (fam_limit is not None and fam_limit >= tier.gpu_count) \
            or matched_reservation is not None
        if global_blocks:
            best = ("NO_QUOTA", zone,
                    f"global GPU cap={int(g_cap)} < need {tier.gpu_count} "
                    f"(raise GPUS-ALL-REGIONS, family {family})")
            continue
        if offered and fam_ok:
            lim = "reservation" if matched_reservation else int(fam_limit)
            best = ("PROVISIONABLE", zone,
                    f"{family} limit={lim} >= {tier.gpu_count} in {region}, "
                    f"offered, global cap={g_cap}")
            break
        if offered is False and (best is None or best[0] == "UNKNOWN"):
            best = ("NOT_OFFERED", zone, f"{tier.accelerator_type} not offered in {zone}")
        elif offered and not fam_ok and (best is None or best[0] != "NOT_OFFERED"):
            shown = 0 if fam_limit is None else int(fam_limit)
            best = ("NO_QUOTA", zone,
                    f"{family} region limit={shown} < need {tier.gpu_count} in {region}")
    if best:
        result["verdict"], result["zone"], result["reason"] = best
    else:
        result["reason"] = "no candidate zone yielded a quota/offer signal"
    return result


def probe(*, project: Optional[str], account: Optional[str], all_tiers: bool,
          zone_override: Optional[str]) -> dict:
    ident = active_identity(account, project)
    report = {
        "schema": SCHEMA,
        "generated_at": utc_now(),
        "identity": ident,
        "tiers": [],
        "recommended": None,
    }
    if not ident.get("ok"):
        report["error"] = ident.get("error", "no active gcloud credential/project")
        report["stale_auth"] = bool(ident.get("stale"))
        return report

    proj = ident["project"]
    acct = ident["account"]
    ladder = gcp_accel.fallback_ladder(blackwell_only=not all_tiers)
    if all_tiers:
        ladder = gcp_accel.fallback_ladder(blackwell_only=False)

    try:
        token = access_token(acct)
        quota = gpu_quota_map(proj, token)
    except GcloudError as exc:
        report["error"] = str(exc)
        report["stale_auth"] = bool(exc.stale)
        return report
    report["global_gpu_cap"] = quota.get("global")
    report["family_region_quota"] = {
        f"{fam}/{reg}": lim for (fam, reg), lim in quota["by_family_region"].items()
    }
    reservation_list = reservations(proj, acct)
    report["reservations_seen"] = [r.get("name") for r in reservation_list]

    for tier in ladder:
        try:
            report["tiers"].append(
                probe_tier(tier, project=proj, account=acct,
                           quota=quota,
                           reservation_list=reservation_list,
                           zone_override=zone_override))
        except GcloudError as exc:
            if exc.stale:
                report["error"] = str(exc)
                report["stale_auth"] = True
                return report
            report["tiers"].append({"slug": tier.slug, "verdict": "UNKNOWN",
                                    "reason": str(exc)})

    for t in report["tiers"]:
        if t.get("verdict") == "PROVISIONABLE":
            report["recommended"] = t["slug"]
            break
    return report


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--project", default=os.environ.get("GCP_PROJECT") or None)
    ap.add_argument("--account", default=os.environ.get("GCP_ACCOUNT") or None)
    ap.add_argument("--zone", default=None,
                    help="Probe only this zone (overrides each tier's common_zones).")
    ap.add_argument("--all-tiers", action="store_true",
                    help="Include H200/H100/L4 fallbacks, not just Blackwell.")
    ap.add_argument("--json", dest="json_out", default=None,
                    help="Write the full verdict JSON to this path.")
    args = ap.parse_args(argv)

    report = probe(project=args.project, account=args.account,
                   all_tiers=args.all_tiers, zone_override=args.zone)

    if args.json_out:
        Path(args.json_out).write_text(json.dumps(report, indent=2), encoding="utf-8")

    ident = report.get("identity", {})
    if report.get("stale_auth"):
        print("GCP credentials are stale -- nothing can be probed or launched.",
              file=sys.stderr)
        print("Fix (interactive, one time):  gcloud auth login", file=sys.stderr)
        print(f"  detail: {report.get('error', '')}", file=sys.stderr)
        return 2
    if report.get("error"):
        print(f"probe error: {report['error']}", file=sys.stderr)
        return 2

    print(f"identity: {ident.get('account')}  project={ident.get('project')}")
    g_cap = report.get("global_gpu_cap")
    if g_cap is not None:
        print(f"global GPU cap (GPUS-ALL-REGIONS): {int(g_cap)}"
              + ("   <-- this gates every multi-GPU tier" if g_cap < 8 else ""))
    if report.get("reservations_seen"):
        print(f"reservations: {', '.join(report['reservations_seen'])}")
    print(f"{'tier':14} {'machine_type':18} {'verdict':14} reason")
    for t in report["tiers"]:
        bw = " [BW]" if t.get("blackwell") else ""
        print(f"{t['slug']:14} {t.get('machine_type',''):18} "
              f"{t.get('verdict',''):14} {t.get('reason','')}{bw}")
    rec = report.get("recommended")
    if rec:
        print(f"\nrecommended: {rec}  (best provisionable tier in the ladder)")
    else:
        print("\nrecommended: NONE provisionable -- request GPU quota or a "
              "reservation, or pass --all-tiers to consider H200/H100/L4.")
    return 0 if rec else 1


if __name__ == "__main__":
    raise SystemExit(main())
