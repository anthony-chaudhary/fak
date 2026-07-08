#!/usr/bin/env python3
"""Elevated Session-0 orphan sweep lane (issue #2338).

Some processes are invisible to every non-elevated reaper: Session-0 zombies
owned by a *tombstoned* fleet account. They are DACL-protected -- a
``Stop-Process``/``taskkill /F`` from the interactive user token returns
ERROR_ACCESS_DENIED and ``GetOwner`` returns ReturnValue=2 -- but they are NOT
kernel-wedged: an elevated / SYSTEM token can terminate them. Documented case:
213 Session-0 zombies (71 opencode.exe + 71 cmd.exe + 71 python.exe triples, all
created 2026-06-29 from the tombstoned ``jack-barker`` account) that cost ~6 GB
RAM and inflate the host_cap thread-throttle baseline the dispatch concurrency
math reads.

``runaway_process_reaper.ps1`` (5 grep-family names) and ``proc_resource_guard.py``
(orphan pattern ``dos_mcp.server``) have no session-ID dimension, so neither can
see these. This lane adds that dimension.

**Safety is the whole point.** The confusion risk the issue names is killing by
image name (``/IM``), which would also hit the LIVE Session-1 fleet processes
that share those image names (opencode.exe/cmd.exe/python.exe). This lane never
does that. :func:`sweep_targets` is a pure, PID-scoped predicate that only
selects a process when ALL of these hold:

  * SessionId == 0            -- never an interactive Session-1 process
  * image name in the fleet-image set -- excludes services (svchost/services/...)
  * owner leaf in the owner set        -- excludes SYSTEM/LocalService and any
                                          live *different* account
  * (optional) created-on date match   -- date-scoped to the incident day
  * dead parent chain (ppid not live)  -- the tombstoned launcher is gone

Because the predicate is pure (rows in, targets out), the load-bearing
invariant -- it never lists a live Session-1 process or a Windows service as a
target -- is asserted directly in ``session0_orphan_sweep_test.py`` with no live
host. Enacting the kill still requires elevation (SYSTEM), which is why the lane
ships with :file:`register_session0_sweep.ps1` (a run-as-SYSTEM scheduled task);
this module runs **dry-run by default** and only calls ``taskkill /F /PID`` under
an explicit ``--enact``.

Exit code: 0 == clean / dry-run / disabled ; 1 == targets remain after an enact
(the Session-0 fleet-image census did not return empty).
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
from typing import Any, Iterable

_CREATE_NO_WINDOW = 0x08000000

# The image names a tombstoned fleet account leaves in Session-0. These are also
# used by LIVE Session-1 fleet processes, which is exactly why image name alone
# is NEVER a sufficient predicate -- the session/owner/date/parent rungs below
# are what make the sweep safe. Windows service hosts (svchost.exe, services.exe)
# are deliberately absent, so a service can never match on image name.
DEFAULT_FLEET_IMAGES: tuple[str, ...] = ("opencode.exe", "cmd.exe", "python.exe")


def _win_creationflags() -> int:
    return _CREATE_NO_WINDOW if platform.system() == "Windows" else 0


def _as_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _owner_leaf(owner: Any) -> str:
    """The account leaf of an owner string, lower-cased. ``NETRA\\jack-barker`` and
    ``jack-barker`` both read as ``jack-barker``; ``None`` (GetOwner denied) reads
    as ``""`` and therefore matches no owner set -- a denied owner is never a
    positive match, so a service that denies GetOwner is not swept."""
    s = str(owner or "").strip()
    if not s:
        return ""
    for sep in ("\\", "/"):
        if sep in s:
            s = s.rsplit(sep, 1)[-1]
    return s.lower()


def _created_date(value: Any) -> str:
    """The date part (YYYY-MM-DD) of a creation timestamp, whatever the shape:
    an ISO ``2026-06-29T04:11:07Z`` or a bare ``2026-06-29`` both yield
    ``2026-06-29``. A blank/absent value yields ``""`` (matches no date)."""
    s = str(value or "").strip()
    if not s:
        return ""
    # Split on the first date/time separator we recognise; keep the leading date.
    for sep in ("T", " "):
        if sep in s:
            s = s.split(sep, 1)[0]
            break
    return s.strip()


def sweep_targets(
    rows: Iterable[dict[str, Any]],
    *,
    fleet_images: Iterable[str] = DEFAULT_FLEET_IMAGES,
    fleet_owners: Iterable[str] = (),
    created_on: str | None = None,
    live_pids: frozenset[int] | None = None,
    require_dead_parent: bool = True,
) -> list[dict[str, Any]]:
    """Pure PID-scoped scoping predicate for the Session-0 orphan sweep.

    A ``row`` carries ``pid``, ``ppid``, ``name`` (image, e.g. ``opencode.exe``),
    ``sessionid``, ``owner`` (``DOMAIN\\user`` or ``None`` if GetOwner denied) and
    ``created`` (a timestamp or date). ``live_pids`` is the set of all currently
    resident PIDs from the same snapshot (used to prove the parent chain is dead);
    when omitted, the dead-parent rung is skipped.

    Returns the subset to terminate, one dict per PID (never an image name), with
    a ``reason`` naming every rung that matched. A process is a target only when
    EVERY enabled rung holds -- the rungs are ANDed, so tightening any one can
    only ever shrink the target set, never grow it."""
    images = {i.lower() for i in fleet_images if i}
    owners = {_owner_leaf(o) for o in fleet_owners if o}
    want_date = _created_date(created_on) if created_on else ""
    live = live_pids or frozenset()

    targets: list[dict[str, Any]] = []
    for row in rows:
        sid = _as_int(row.get("sessionid"))
        if sid != 0:
            # Rung 1: never an interactive Session-1 (or higher) process.
            continue

        name = str(row.get("name") or "")
        if name.lower() not in images:
            # Rung 2: excludes services (svchost/services) and any other image.
            continue

        owner_leaf = _owner_leaf(row.get("owner"))
        if owners and owner_leaf not in owners:
            # Rung 3: excludes SYSTEM/LocalService and any live *different* account.
            continue

        if want_date and _created_date(row.get("created")) != want_date:
            # Rung 4 (optional): date-scoped to the incident day.
            continue

        ppid = _as_int(row.get("ppid"))
        parent_dead = ppid is None or ppid <= 1 or ppid not in live
        if require_dead_parent and not parent_dead:
            # Rung 5: the tombstoned launcher is gone; a live parent is spared.
            continue

        reason = ["session0", f"image={name}"]
        if owners:
            reason.append(f"owner={owner_leaf}")
        if want_date:
            reason.append(f"created={want_date}")
        if require_dead_parent:
            reason.append(f"dead-parent(ppid={ppid})")
        targets.append(
            {
                "pid": _as_int(row.get("pid")),
                "name": name,
                "sessionid": sid,
                "owner": owner_leaf or None,
                "created": _created_date(row.get("created")) or None,
                "ppid": ppid,
                "reason": ", ".join(reason),
            }
        )
    targets.sort(key=lambda t: t["pid"] or 0)
    return targets


def target_pids(targets: Iterable[dict[str, Any]]) -> list[int]:
    """PID list for the terminate call -- PID-scoped, never ``/IM``."""
    return [t["pid"] for t in targets if _as_int(t.get("pid")) is not None]


# --------------------------------------------------------------------------- #
# Live host collection (Windows only) -- thin PowerShell shell, not unit-tested.
# The predicate above is the tested part; this just feeds it a snapshot.
# --------------------------------------------------------------------------- #
def _ps_image_array(images: Iterable[str]) -> str:
    """A PowerShell string-array literal of validated image names. Names are
    restricted to a safe charset so nothing here can inject into the script."""
    safe = [i for i in images if i and all(c.isalnum() or c in "._-" for c in i)]
    return "@(" + ",".join("'" + i.replace("'", "") + "'" for i in safe) + ")"


def collect_session0(
    images: Iterable[str] = DEFAULT_FLEET_IMAGES,
) -> tuple[list[dict[str, Any]], frozenset[int], str]:
    """Snapshot the Session-0 fleet-image candidates (with owner/created/parent)
    plus the full set of live PIDs (for the dead-parent rung).

    Only the Session-0 processes whose image is in ``images`` get a ``GetOwner``
    round-trip -- filtering in the CIM query first keeps the sweep fast even on a
    host with hundreds of processes. Owner resolves to the real account only under
    an elevated / SYSTEM token; run non-elevated it comes back ``None`` and the
    owner rung matches nothing (fail-safe)."""
    if platform.system() != "Windows":
        return [], frozenset(), "session-0 sweep is a Windows-only lane"
    imgs = _ps_image_array(images)
    script = (
        "$all = Get-CimInstance Win32_Process -ErrorAction SilentlyContinue; "
        "$live = @($all | ForEach-Object { $_.ProcessId }); "
        f"$imgs = {imgs}; "
        "$rows = @($all | Where-Object { $_.SessionId -eq 0 -and $imgs -contains $_.Name } "
        "| ForEach-Object { try { "
        "$o=Invoke-CimMethod -InputObject $_ -MethodName GetOwner -ErrorAction SilentlyContinue; "
        "$own = if ($o -and $o.ReturnValue -eq 0) { \"$($o.Domain)\\$($o.User)\" } else { $null }; "
        "$cd = if ($_.CreationDate) { $_.CreationDate.ToString('o') } else { $null }; "
        "[pscustomobject]@{ pid=$_.ProcessId; ppid=$_.ParentProcessId; name=$_.Name; "
        "sessionid=$_.SessionId; owner=$own; created=$cd } } catch {} }); "
        "[pscustomobject]@{ live=$live; rows=$rows } | ConvertTo-Json -Compress -Depth 4"
    )
    try:
        proc = subprocess.run(
            ["powershell", "-NoProfile", "-NonInteractive", "-Command", script],
            capture_output=True,
            text=True,
            timeout=120,
            check=False,
            creationflags=_win_creationflags(),
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return [], frozenset(), f"{type(exc).__name__}: {exc}"
    text = (proc.stdout or "").strip()
    if not text:
        return [], frozenset(), ""
    try:
        obj = json.loads(text)
    except json.JSONDecodeError as exc:
        return [], frozenset(), f"JSONDecodeError: {exc}"
    if not isinstance(obj, dict):
        return [], frozenset(), ""
    raw_rows = obj.get("rows") or []
    if isinstance(raw_rows, dict):
        raw_rows = [raw_rows]
    rows = [r for r in raw_rows if isinstance(r, dict)]
    raw_live = obj.get("live") or []
    if isinstance(raw_live, int):
        raw_live = [raw_live]
    live = frozenset(p for p in (_as_int(x) for x in raw_live) if p is not None)
    return rows, live, ""


def _terminate(pids: Iterable[int]) -> list[dict[str, Any]]:
    """PID-scoped ``taskkill /F /PID`` -- succeeds only under elevation. Never
    ``/IM``: every kill names exactly one PID this lane already vouched for."""
    enacted: list[dict[str, Any]] = []
    for pid in pids:
        try:
            proc = subprocess.run(
                ["taskkill", "/F", "/PID", str(pid)],
                capture_output=True,
                text=True,
                timeout=30,
                check=False,
                creationflags=_win_creationflags(),
            )
            enacted.append({"pid": pid, "rc": proc.returncode, "detail": (proc.stdout or proc.stderr or "").strip()})
        except (OSError, subprocess.SubprocessError) as exc:
            enacted.append({"pid": pid, "rc": -1, "detail": f"{type(exc).__name__}: {exc}"})
    return enacted


def _is_elevated() -> bool:
    if platform.system() != "Windows":
        return os.geteuid() == 0 if hasattr(os, "geteuid") else False
    try:
        import ctypes

        return bool(ctypes.windll.shell32.IsUserAnAdmin())
    except (OSError, AttributeError):
        return False


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Elevated Session-0 orphan sweep lane (#2338).")
    ap.add_argument("--owners", default="jack-barker",
                    help="comma-separated owner account leaves to sweep (default: jack-barker). "
                         "Empty means owner-agnostic -- refused unless --created-on is set.")
    ap.add_argument("--created-on", default="",
                    help="only sweep processes created on this YYYY-MM-DD (default: unset).")
    ap.add_argument("--images", default=",".join(DEFAULT_FLEET_IMAGES),
                    help="comma-separated fleet image names.")
    ap.add_argument("--enact", action="store_true",
                    help="actually terminate the targets (PID-scoped). Requires elevation; "
                         "default is a dry-run census.")
    args = ap.parse_args(argv)

    owners = tuple(o.strip() for o in args.owners.split(",") if o.strip())
    images = tuple(i.strip() for i in args.images.split(",") if i.strip())
    created_on = args.created_on.strip() or None

    # Refuse an over-broad sweep: with neither an owner set nor a date, the only
    # rungs left are session0+image+dead-parent, which is too coarse to enact.
    if not owners and not created_on:
        print(json.dumps({"error": "refused: need --owners or --created-on to scope the sweep"}))
        return 2

    rows, live_pids, err = collect_session0(images)
    targets = sweep_targets(
        rows,
        fleet_images=images,
        fleet_owners=owners,
        created_on=created_on,
        live_pids=live_pids,
    )

    result: dict[str, Any] = {
        "platform": platform.system(),
        "elevated": _is_elevated(),
        "collect_error": err,
        "scanned": len(rows),
        "targets": targets,
        "target_pids": target_pids(targets),
        "enacted": [],
    }

    if args.enact:
        if not _is_elevated():
            result["enact_warning"] = (
                "not elevated: these are DACL-protected another-account PIDs; "
                "taskkill will return ACCESS_DENIED. Run via register_session0_sweep.ps1 (SYSTEM)."
            )
        result["enacted"] = _terminate(target_pids(targets))
        # Re-census: after a real sweep the Session-0 fleet-image targets must be 0.
        rows2, live2, _ = collect_session0(images)
        remaining = sweep_targets(
            rows2, fleet_images=images, fleet_owners=owners, created_on=created_on, live_pids=live2
        )
        result["remaining_after"] = len(remaining)

    print(json.dumps(result, indent=2))
    if args.enact and result.get("remaining_after", 0) > 0:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
