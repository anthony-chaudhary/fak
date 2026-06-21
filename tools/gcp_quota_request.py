#!/usr/bin/env python3
"""gcp_quota_request.py -- file (and read back) a GCP GPU quota-increase request.

Track B of the GCP-Blackwell benchmark effort: the global GPU cap
`GPUS-ALL-REGIONS` and the per-region per-family quota gate every Blackwell
launch, and both start at a level too low for an 8x B200 VM. This tool files a
Cloud Quotas v1 `quotaPreference` against each, records the API response as
durable evidence, and can poll the grant status.

It hits the Cloud Quotas REST API directly with a user access token
(`gcloud auth print-access-token`) because the `gcloud alpha quotas` command
group cannot be installed on this host (no admin to write the SDK dir). The two
preferences it files for a B200 launch:

  1. GPUS-ALL-REGIONS-per-project        -> preferredValue N   (dimension-less)
  2. GPUS-PER-GPU-FAMILY-per-project-region with {region, gpu_family} -> N

Both are required: (1) is the master cross-region ceiling, (2) is the specific
B200-in-a-region gate. There is NO standalone `NVIDIA_B200_GPUS` quotaId in
v1 -- B200 is a `gpu_family` DIMENSION on the family quota (verified live
2026-06-20: NVIDIA_B200 is a verbatim token, increase-eligible).

A successful POST only RECORDS the preference (reconciling:true); the actual
grant is asynchronous and may be partially granted or denied. `--status` polls.
If a preference already exists, create returns ALREADY_EXISTS; this tool then
PATCHes with the existing etag (upsert).

SAFETY: filing a quota request is outward-facing (your contact email + a
justification go to Google, visible to the org). `--dry-run` prints the exact
POST bodies and URLs and submits NOTHING. Submission requires `--submit`.

Usage:
  python tools/gcp_quota_request.py --dry-run                  # show the plan
  python tools/gcp_quota_request.py --submit                   # file the B200 request (8 GPUs, us-central1)
  python tools/gcp_quota_request.py --gpus 8 --region us-central1 --family NVIDIA_B200 --submit
  python tools/gcp_quota_request.py --status                   # poll grant status
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
from pathlib import Path
import subprocess
import sys
from typing import Any, Optional
import urllib.error
import urllib.request


ROOT = Path(__file__).resolve().parents[1]
EVIDENCE_DIR = ROOT / "fak" / "experiments" / "gcp-quota"
API = "https://cloudquotas.googleapis.com/v1"
DEFAULT_PROJECT = "example-gcp-project"
DEFAULT_CONTACT = "anthony.chaudhary@example.com"
DEFAULT_JUST = (
    "Launch a4-highgpu-8g (8x NVIDIA B200 Blackwell) for fak native-engine + "
    "llama.cpp CUDA inference benchmarking; need 8 B200 GPUs in one region and "
    "the all-regions GPU ceiling raised to match."
)


def utc_now() -> str:
    return _dt.datetime.now(_dt.timezone.utc).isoformat().replace("+00:00", "Z")


def stamp() -> str:
    return _dt.datetime.now(_dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def access_token() -> str:
    """A user access token. Must run WITHOUT --format or the header 401s.

    gcloud on Windows is gcloud.cmd/gcloud.ps1, not a bare exe, so CreateProcess
    can't find "gcloud" directly -- resolve the real launcher with shutil.which,
    and fall back to a shell invocation.
    """
    import shutil

    exe = shutil.which("gcloud") or shutil.which("gcloud.cmd") or shutil.which("gcloud.ps1")
    if exe:
        p = subprocess.run([exe, "auth", "print-access-token"],
                           capture_output=True, text=True, shell=False)
    else:
        p = subprocess.run("gcloud auth print-access-token",
                           capture_output=True, text=True, shell=True)
    tok = (p.stdout or "").strip()
    if not tok:
        raise RuntimeError(f"could not get access token: {(p.stderr or '').strip()[:200]}")
    return tok


def _req(method: str, url: str, token: str, project: str,
         body: Optional[dict] = None) -> tuple[int, dict]:
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(url, data=data, method=method)
    req.add_header("Authorization", f"Bearer {token}")
    req.add_header("x-goog-user-project", project)
    req.add_header("Content-Type", "application/json")
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            return resp.status, json.loads(resp.read().decode() or "{}")
    except urllib.error.HTTPError as e:
        detail = e.read().decode()
        try:
            return e.code, json.loads(detail)
        except json.JSONDecodeError:
            return e.code, {"_raw": detail}
    except urllib.error.URLError as e:
        return 0, {"_error": str(e)}


def preference_url(project: str, pref_id: str, *, create: bool) -> str:
    base = f"{API}/projects/{project}/locations/global/quotaPreferences"
    if create:
        return f"{base}?quotaPreferenceId={pref_id}"
    return f"{base}/{pref_id}"


def build_requests(project: str, gpus: int, region: str, family: str,
                   contact: str, just: str) -> list[dict]:
    """The two quotaPreference POSTs needed for an N-GPU family launch."""
    all_regions_id = "compute_googleapis_com-gpus-all-regions"
    family_id = f"compute_googleapis_com-gpus-{region}-{family}".replace("_", "-").lower()
    return [
        {
            "name": "GPUS-ALL-REGIONS (global ceiling)",
            "pref_id": all_regions_id,
            "body": {
                "service": "compute.googleapis.com",
                "quotaId": "GPUS-ALL-REGIONS-per-project",
                "quotaConfig": {"preferredValue": gpus},
                "contactEmail": contact,
                "justification": just,
            },
        },
        {
            "name": f"{family} in {region} (per-region family)",
            "pref_id": family_id,
            "body": {
                "service": "compute.googleapis.com",
                "quotaId": "GPUS-PER-GPU-FAMILY-per-project-region",
                "quotaConfig": {"preferredValue": gpus},
                "dimensions": {"region": region, "gpu_family": family},
                "contactEmail": contact,
                "justification": just,
            },
        },
    ]


def _find_existing(token: str, project: str, spec: dict) -> Optional[str]:
    """Return the resource id of an existing preference matching this spec's
    (quotaId, dimensions), or None. The all-regions preference often pre-exists
    under a DIFFERENT id (e.g. 'gpus-all-regions-1'), so we match on content, not
    on our chosen pref_id."""
    url = f"{API}/projects/{project}/locations/global/quotaPreferences"
    st, resp = _req("GET", url, token, project)
    want_qid = spec["body"]["quotaId"]
    want_dims = spec["body"].get("dimensions", {}) or {}
    for pref in resp.get("quotaPreferences", []) if isinstance(resp, dict) else []:
        if pref.get("quotaId") != want_qid:
            continue
        if (pref.get("dimensions") or {}) == want_dims:
            return pref["name"].rsplit("/", 1)[-1], pref.get("etag")
    return None


def submit_one(token: str, project: str, spec: dict) -> dict:
    """POST create; if a preference for this (quotaId,dimensions) already exists
    -- which the API reports as a 400 INVALID_ARGUMENT with 'already exist', NOT
    a 409 -- find the real resource id and PATCH it (upsert)."""
    url = preference_url(project, spec["pref_id"], create=True)
    status, resp = _req("POST", url, token, project, spec["body"])
    already = (
        status in (400, 409)
        and isinstance(resp, dict)
        and "already exist" in json.dumps(resp).lower()
    )
    if already:
        found = _find_existing(token, project, spec)
        if not found:
            return {"action": "create", "status": status, "response": resp}
        real_id, etag = found
        # PATCH requires service + quotaId in the body (not just the masked field).
        patch_body = dict(spec["body"])
        if etag:
            patch_body["etag"] = etag
        patch_url = (f"{API}/projects/{project}/locations/global/quotaPreferences/"
                     f"{real_id}?updateMask=quotaConfig.preferredValue")
        ps, pr = _req("PATCH", patch_url, token, project, patch_body)
        return {"action": f"patch({real_id})", "status": ps, "response": pr}
    return {"action": "create", "status": status, "response": resp}


def read_status(token: str, project: str, specs: list[dict]) -> list[dict]:
    out = []
    for spec in specs:
        # Resolve the real resource id by content (the all-regions preference
        # lives under 'gpus-all-regions-1', not our chosen pref_id).
        found = _find_existing(token, project, spec)
        pref_id = found[0] if found else spec["pref_id"]
        url = preference_url(project, pref_id, create=False)
        st, resp = _req("GET", url, token, project)
        cfg = resp.get("quotaConfig", {}) if isinstance(resp, dict) else {}
        out.append({
            "name": spec["name"],
            "pref_id": spec["pref_id"],
            "http": st,
            "preferred": cfg.get("preferredValue"),
            "granted": cfg.get("grantedValue"),
            "trace_id": cfg.get("traceId"),
            "reconciling": resp.get("reconciling") if isinstance(resp, dict) else None,
            "state": (resp.get("quotaConfig", {}) or {}).get("requestOrigin"),
            "raw": resp,
        })
    return out


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--project", default=DEFAULT_PROJECT)
    ap.add_argument("--gpus", type=int, default=8, help="preferred GPU count (a4-highgpu-8g needs 8)")
    ap.add_argument("--region", default="us-central1")
    ap.add_argument("--family", default="NVIDIA_B200")
    ap.add_argument("--contact", default=DEFAULT_CONTACT)
    ap.add_argument("--justification", default=DEFAULT_JUST)
    ap.add_argument("--submit", action="store_true", help="actually file the request")
    ap.add_argument("--status", action="store_true", help="poll grant status, file nothing")
    ap.add_argument("--dry-run", action="store_true", help="print bodies/URLs, submit nothing")
    args = ap.parse_args(argv)

    specs = build_requests(args.project, args.gpus, args.region, args.family,
                           args.contact, args.justification)

    if args.dry_run or (not args.submit and not args.status):
        print("DRY-RUN -- would POST these quotaPreferences (nothing submitted):\n")
        for spec in specs:
            print(f"# {spec['name']}")
            print(f"POST {preference_url(args.project, spec['pref_id'], create=True)}")
            print(json.dumps(spec["body"], indent=2))
            print()
        print("Re-run with --submit to file, or --status to poll.")
        return 0

    token = access_token()
    EVIDENCE_DIR.mkdir(parents=True, exist_ok=True)

    if args.status:
        rows = read_status(token, args.project, specs)
        for r in rows:
            print(f"{r['name']}: http={r['http']} preferred={r['preferred']} "
                  f"granted={r['granted']} reconciling={r['reconciling']}")
        (EVIDENCE_DIR / f"status-{stamp()}.json").write_text(
            json.dumps(rows, indent=2), encoding="utf-8")
        return 0

    # submit
    results = []
    for spec in specs:
        res = submit_one(token, args.project, spec)
        ok = res["status"] in (200, 201)
        print(f"[{'OK' if ok else 'ERR'}] {spec['name']}: "
              f"{res['action']} http={res['status']}")
        if not ok:
            print("    " + json.dumps(res["response"])[:400])
        results.append({"spec": spec, "result": res})
    evidence = {
        "schema": "fak.gcp-quota-request.v1",
        "submitted_at": utc_now(),
        "project": args.project,
        "gpus": args.gpus,
        "region": args.region,
        "family": args.family,
        "results": results,
    }
    out = EVIDENCE_DIR / f"request-{stamp()}.json"
    out.write_text(json.dumps(evidence, indent=2), encoding="utf-8")
    print(f"\nevidence -> {out}")
    print("Poll approval with: python tools/gcp_quota_request.py --status")
    return 0 if all(r["result"]["status"] in (200, 201) for r in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
