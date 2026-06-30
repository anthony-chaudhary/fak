#!/usr/bin/env python3
"""Multi-node lease transport for fleet — the cross-host overlay on `dos lease-lane`.

## Why this exists

DOS already ships a real, WAL-backed lane-lease layer: `dos lease-lane
{acquire,release,heartbeat,spawn,live}` journals every grant to
`.dos/lane-journal.jsonl`, and `dos lease-lane live` reconstructs the held-lease
set from that WAL. A live record already carries node identity:

    {"lane":"docs","holder":"...","host_id":"<node-hostname>","mode":"exclusive",
     "loop_ts":"...","pid":25768,"proc_starttime":...,"tree":["docs/**"], ...}

So on ONE machine, two sessions are already kept off the same lane: each consults
the shared WAL, and the pure `dos arbitrate` admitter refuses an overlapping tree.

The gap is purely **multi-node**. The WAL is local and gitignored (`.dos/.gitignore`
is `*`), so node B never sees a lane node A holds — two machines can each grant the
same exclusive lane and collide. And kernel liveness uses `pid`+`proc_starttime`, a
**same-host** probe that is meaningless for a remote record, so a dead remote node's
lease can't be aged out locally.

This tool is the missing transport, and ONLY the transport. It does not reimplement
the lease store or the disjointness rule — the kernel stays the authority. It uses
the seam the kernel already provides: `dos lease-lane acquire --leases <JSON array>`
**unions extra live leases with the WAL's** before deciding. The whole multi-node
story is therefore:

    publish each node's `dos lease-lane live`  ->  a shared store
    materialize the fleet union  ->  feed it back via `--leases` at acquire time

`tools/release_lock.py` is the precedent for the *idiom* (TTL staleness not pid —
Windows `os.kill` terminates rather than probes; owner from `$CLAUDE_CODE_SESSION_ID`;
exit codes 0/2/3/4; hermetic tempfile tests). The lease *store* precedent is the
kernel's `dos lease-lane`, which this wraps — not release_lock.

## Regimes

- **Shard-by-node (primary):** each node owns disjoint lanes; a node arbitrating its
  OWN lanes needs zero network — `materialize(scope="local")` reads only the local
  WAL. This is the common, fast, always-available path.
- **Global (the 3 exclusive lanes abi/release/global):** `materialize(scope="global")`
  unions the local WAL with every other node's published set. Only this path touches
  the transport. For the `release` lane specifically, COMPOSE with
  `tools/release_lock.py` (the global lease picks the node; release_lock picks the
  session on that node) — do not re-solve releases here.

## Correctness scope (honest residuals)

- Foreign-lease liveness is a wall-clock overlay (`heartbeat_at + ttl + skew_margin`),
  because the kernel's pid/proc_starttime probe doesn't cross hosts. The
  epoch/heartbeat — not the bare clock — is the steal authority; TTL across nodes only
  *narrows* a double-grant. A `skew_margin` covers bounded clock skew.
- This MVP fences the lease *record*, not the leased *files*: an ordinary commit to
  `gateway/**` carries no epoch, so a lost-then-resumed writer is only *narrowed*, not
  closed, until a pre-commit lease gate exists (out of scope).
- The `GitRefStore` transport (a real `refs/dos-fleet-leases/<host_id>` force-with-lease
  CAS push) is IMPLEMENTED and TESTED — `GitRefStoreCASTest` exercises it against a real
  throwaway bare remote and proves a second node's stale-OID push is refused at the remote
  (the cross-node double-grant is impossible at publish time). ACTIVATED (#21): the 3 global
  exclusive lanes (abi/release/global) now default to `GitRefStore` — the lanes that need
  cross-node consensus get it with no flag; shard-by-node lanes keep the zero-network
  `LocalDirStore` fast path. An explicit `--store` / `FAK_FLEET_LEASE_STORE` still overrides
  either way. Both backends honor the same CAS contract.

## FLEET_LANE_ALLOWLIST decision (#21)

DECIDED: a `FLEET_LANE_ALLOWLIST` is NOT built as a serialization floor. It cannot be
sound from fleet code alone — the kernel's bare-autopick ladder (`arbiter.py` reads
`cfg.lanes.autopick`) exposes no per-node override flag (`cli.py`), so a fleet-level
allowlist can only filter what the supervisor *proposes*, never bind what the kernel
*picks*. The serialization floor for the 3 global lanes is therefore the GitRefStore
CAS, now wired ON by default for those lanes (see `_store_for`). Any future allowlist
is ADVISORY steering only — a best-effort steer/refuse at
`dos_supervisor_watchdog.build_plan` + `dispatch_worker.build_command` — and a real
per-node allowlist is deferred until the dos kernel grows an autopick-override seam.

Refuse reasons are REUSED from the kernel's closed vocabulary (verified known via
`dos_check_reason`), not invented: `LANE_LEASE_HELD_BY_LIVE_DISPATCH_LOOP` (a foreign
live holder has the lane — its reason string carries the holder), `SCHEMA_UNREADABLE`
(an unparseable/old published record — refuse-don't-guess), `NO_FREE_REGION` (autopick
found nothing disjoint).

Pure stdlib; off the request path (a tooling seam). Exit codes mirror release_lock:
0 ok, 2 usage, 3 contended/denied, 4 internal.
"""
from __future__ import annotations

import argparse
import json
import os
import socket
import subprocess
from dispatch_worker import install_no_window_subprocess_defaults
import sys
import time
from pathlib import Path
from typing import Any, Callable
install_no_window_subprocess_defaults(subprocess)

SCHEMA = "fleet-multinode-lease/1"

# Exit codes (shared protocol with release_lock.py).
EXIT_OK, EXIT_USAGE, EXIT_DENIED, EXIT_INTERNAL = 0, 2, 3, 4

# Reused kernel refuse tokens (verified known via dos_check_reason — NOT invented).
REASON_HELD_REMOTE = "LANE_LEASE_HELD_BY_LIVE_DISPATCH_LOOP"
REASON_STALE_RECORD = "SCHEMA_UNREADABLE"
REASON_NO_REGION = "NO_FREE_REGION"

# Default foreign-lease liveness window. A published record is considered live until
# heartbeat_at + ttl + skew_margin; past that a peer may treat the lane as free.
DEFAULT_TTL_S = 900          # 15 min — generous; heartbeats refresh it.
DEFAULT_SKEW_MARGIN_S = 30   # bounded cross-node clock skew we tolerate.

# The 3 exclusive lanes are the only ones that need the fleet-wide union; everything
# else is shard-by-node and arbitrates against the local WAL alone.
GLOBAL_LANES = ("abi", "release", "global")


# --------------------------------------------------------------------------- env


def repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def host_id() -> str:
    try:
        return socket.gethostname()
    except OSError:
        return "?"


def default_owner() -> str:
    """A token stable across this session's separate `python` invocations.

    Mirrors release_lock.default_owner: the harness sets CLAUDE_CODE_SESSION_ID once
    per session (stable across calls, unique per session); fall back to host+pid.
    """
    sid = os.environ.get("FAK_LEASE_OWNER") or os.environ.get("CLAUDE_CODE_SESSION_ID")
    if sid:
        return sid.strip()
    return f"{os.environ.get('USERNAME') or os.environ.get('USER') or 'user'}@{host_id()}:{os.getpid()}"


def now_s() -> float:
    return time.time()


# --------------------------------------------------------------------------- kernel seam


def run_text(cmd: list[str], cwd: Path, *, timeout: int = 30) -> dict[str, Any]:
    """Run a command; UTF-8 with replacement (dos/git emit non-ASCII prose)."""
    try:
        proc = subprocess.run(
            cmd, cwd=str(cwd), capture_output=True, text=True,
            encoding="utf-8", errors="replace", timeout=timeout, check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"stdout": "", "stderr": str(exc), "returncode": 1, "_error": str(exc)}
    return {"stdout": proc.stdout, "stderr": proc.stderr, "returncode": proc.returncode}


def parse_live_array(text: str) -> list[dict[str, Any]]:
    """`dos lease-lane live` emits a JSON array of lease records. Tolerant parse."""
    text = (text or "").strip()
    if not text:
        return []
    try:
        data = json.loads(text)
    except ValueError:
        # Last non-empty line is sometimes the only JSON line.
        for line in reversed(text.splitlines()):
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
            except ValueError:
                continue
            break
        else:
            return []
    return [r for r in data if isinstance(r, dict)] if isinstance(data, list) else []


# The kernel-call seam is injectable so tests never shell a real `dos`.
KernelLive = Callable[[Path], list[dict[str, Any]]]
KernelAcquire = Callable[..., dict[str, Any]]


def kernel_live(workspace: Path) -> list[dict[str, Any]]:
    # Do NOT pass --workspace: the kernel resolves its WAL from cwd, and an explicit
    # path arg here makes `dos lease-lane live` re-resolve to a different/empty .dos/
    # and emit non-JSON (verified on this host). run_text already sets cwd=workspace.
    res = run_text(["dos", "lease-lane", "live"], workspace)
    return parse_live_array(res.get("stdout", ""))


def kernel_acquire(
    workspace: Path, *, lane: str, owner: str, kind: str = "", mode: str = "exclusive",
    run_id: str = "", leases: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    cmd = ["dos", "lease-lane", "acquire", "--owner", owner]  # cwd=workspace; no --workspace (see kernel_live)
    if lane:
        cmd += ["--lane", lane]
    if kind:
        cmd += ["--kind", kind]
    if mode:
        cmd += ["--mode", mode]
    if run_id:
        cmd += ["--run-id", run_id]
    if leases:
        cmd += ["--leases", json.dumps(leases)]
    res = run_text(cmd, workspace)
    out = parse_first_obj(res.get("stdout", ""))
    out.setdefault("_returncode", res.get("returncode"))
    return out


def kernel_release(workspace: Path, *, lane: str, owner: str) -> dict[str, Any]:
    res = run_text(
        ["dos", "lease-lane", "release", "--lane", lane, "--owner", owner],  # cwd=workspace
        workspace,
    )
    return parse_first_obj(res.get("stdout", ""))


def kernel_heartbeat(workspace: Path, *, lane: str, owner: str, loop_ts: str = "") -> dict[str, Any]:
    cmd = ["dos", "lease-lane", "heartbeat", "--lane", lane, "--owner", owner]  # cwd=workspace
    if loop_ts:
        cmd += ["--loop-ts", loop_ts]
    res = run_text(cmd, workspace)
    return parse_first_obj(res.get("stdout", ""))


def parse_first_obj(text: str) -> dict[str, Any]:
    text = (text or "").strip()
    if not text:
        return {}
    try:
        obj = json.loads(text)
    except ValueError:
        for line in reversed(text.splitlines()):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except ValueError:
                continue
            if isinstance(obj, dict):
                return obj
        return {}
    if isinstance(obj, list):
        return obj[0] if obj and isinstance(obj[0], dict) else {}
    return obj if isinstance(obj, dict) else {}


# --------------------------------------------------------------------------- transport seam


class FleetStore:
    """Cross-node publish/read of per-host lease snapshots.

    Two methods only. `read_fleet()` returns {host_id: snapshot}; `compare_and_swap`
    writes THIS host's snapshot iff the store's current version matches `expected`
    (the TOCTOU guard the git backend inherits). A snapshot is:
        {"host_id", "published_at", "fleet_epoch", "leases": [<record>, ...]}
    """

    def read_fleet(self) -> dict[str, dict[str, Any]]:  # pragma: no cover - interface
        raise NotImplementedError

    def compare_and_swap(self, host: str, expected: Any, snapshot: dict[str, Any]) -> tuple[bool, Any]:  # pragma: no cover
        raise NotImplementedError


class LocalDirStore(FleetStore):
    """A directory of `<host_id>.json` snapshots. The MVP backend + test fixture.

    The "version" used for the CAS is the snapshot's `fleet_epoch` (monotone per host).
    On a single box this is a trivial filesystem store; a two-dir setup simulates two
    nodes. The git backend (below) implements the SAME contract over a per-host ref.
    """

    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, host: str) -> Path:
        safe = "".join(c if (c.isalnum() or c in "-_.") else "-" for c in host) or "host"
        return self.root / f"{safe}.json"

    def read_fleet(self) -> dict[str, dict[str, Any]]:
        out: dict[str, dict[str, Any]] = {}
        for p in sorted(self.root.glob("*.json")):
            try:
                snap = json.loads(p.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                # A corrupt snapshot is surfaced (not silently dropped) so the
                # SCHEMA_UNREADABLE rung can refuse-don't-guess on it.
                out[p.stem] = {"host_id": p.stem, "_unreadable": True}
                continue
            if isinstance(snap, dict):
                out[str(snap.get("host_id") or p.stem)] = snap
        return out

    def compare_and_swap(self, host: str, expected: Any, snapshot: dict[str, Any]) -> tuple[bool, Any]:
        path = self._path(host)
        current = None
        if path.exists():
            try:
                cur = json.loads(path.read_text(encoding="utf-8"))
                current = cur.get("fleet_epoch") if isinstance(cur, dict) else None
            except (OSError, ValueError):
                current = None
        if expected is not None and current != expected:
            return False, current
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(snapshot, indent=2) + "\n", encoding="utf-8")
        os.replace(tmp, path)  # atomic on the local FS
        return True, snapshot.get("fleet_epoch")


class GitRefStore(FleetStore):
    """Publish each host's snapshot to `refs/dos-fleet-leases/<host_id>` on `remote`.

    The per-host ref points at a BLOB whose content is the snapshot JSON, and a
    force-with-lease push IS the cross-node compare-and-swap:

        git push <remote> <blob-oid>:refs/dos-fleet-leases/<host_id> \\
                 --force-with-lease=refs/dos-fleet-leases/<host_id>:<expected-old-oid>

    The remote atomically rejects the update unless its current ref oid matches the
    lease, so two nodes racing to claim the same lane cannot both win — the remote's
    per-ref update IS the consensus, no central lock server. It deliberately uses a
    DEDICATED ref namespace, never master/the index, so it sidesteps both
    safe_ff_sync's dirty-tree refusal AND the OFF_TRUNK trunk guard. read_fleet() does
    `git ls-remote <remote> 'refs/dos-fleet-leases/*'`, fetches the blobs, and
    `git cat-file -p`s each. The CAS guard is the ref OID, which is strictly stronger
    than the `fleet_epoch` LocalDirStore compares (it also catches a same-epoch
    content change); `expected is None` still selects create-only (non-force) so a
    first publisher that loses the create race is correctly rejected.

    Honest residual (see module docstring): this fences the lease *record* across
    nodes, and foreign liveness is the wall-clock TTL overlay in `foreign_record_live`
    — the remote ref CAS makes a *double-grant* impossible at publish time, but a
    stale TTL window can still let a peer treat a crashed node's lane as free. The 3
    global exclusive lanes default to this store (`_store_for`, #21); shard lanes use
    LocalDirStore. An explicit `--store git` / FAK_FLEET_LEASE_STORE=git forces it for
    any lane.
    """

    REF_NS = "refs/dos-fleet-leases"

    def __init__(self, root: Path, remote: str = "origin") -> None:
        self.root, self.remote = Path(root), remote
        self._oid: dict[str, str] = {}  # {host: ref oid} cached from the last ls-remote

    def _git(self, args: list[str], *, stdin: str | None = None, timeout: int = 60) -> dict[str, Any]:
        """Run one git command with cwd=root; `stdin` feeds hash-object. Never raises."""
        try:
            proc = subprocess.run(
                ["git", *args], cwd=str(self.root), input=stdin,
                capture_output=True, text=True, encoding="utf-8",
                errors="replace", timeout=timeout, check=False,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            return {"stdout": "", "stderr": str(exc), "returncode": 1}
        return {"stdout": proc.stdout, "stderr": proc.stderr, "returncode": proc.returncode}

    def _ls_remote(self) -> dict[str, str]:
        res = self._git(["ls-remote", self.remote, f"{self.REF_NS}/*"])
        out: dict[str, str] = {}
        for line in (res.get("stdout") or "").splitlines():
            parts = line.split()
            if len(parts) == 2 and parts[1].startswith(self.REF_NS + "/"):
                out[parts[1][len(self.REF_NS) + 1:]] = parts[0]
        return out

    def read_fleet(self) -> dict[str, dict[str, Any]]:
        refs = self._ls_remote()
        self._oid = dict(refs)  # remember the OIDs so compare_and_swap can lease against them
        if not refs:
            return {}
        # Bring the blob objects local so cat-file can read them (refspec fetch works
        # for refs that point at blobs; objects are addressed by the ls-remote oid).
        self._git(["fetch", self.remote, f"+{self.REF_NS}/*:{self.REF_NS}/*"])
        out: dict[str, dict[str, Any]] = {}
        for host, oid in refs.items():
            blob = self._git(["cat-file", "-p", oid])
            if blob.get("returncode") != 0:
                out[host] = {"host_id": host, "_unreadable": True}
                continue
            try:
                snap = json.loads(blob.get("stdout") or "")
            except ValueError:
                # A corrupt/unparseable blob is surfaced (not dropped) so the
                # SCHEMA_UNREADABLE rung can refuse-don't-guess — mirrors LocalDirStore.
                out[host] = {"host_id": host, "_unreadable": True}
                continue
            if isinstance(snap, dict):
                out[str(snap.get("host_id") or host)] = snap
        return out

    def _current_oid(self, host: str) -> str | None:
        if host in self._oid:
            return self._oid[host]
        res = self._git(["ls-remote", self.remote, f"{self.REF_NS}/{host}"])
        for line in (res.get("stdout") or "").splitlines():
            parts = line.split()
            if len(parts) == 2:
                return parts[0]
        return None

    def _observed_epoch(self, host: str) -> Any:
        try:
            return self.read_fleet().get(host, {}).get("fleet_epoch")
        except Exception:  # noqa: BLE001 - best-effort report only
            return None

    def compare_and_swap(self, host: str, expected: Any, snapshot: dict[str, Any]) -> tuple[bool, Any]:
        ref = f"{self.REF_NS}/{host}"
        new = self._git(["hash-object", "-w", "--stdin"], stdin=json.dumps(snapshot, indent=2) + "\n")
        if new.get("returncode") != 0:
            return False, None
        new_oid = (new.get("stdout") or "").strip()
        if not new_oid:
            return False, None
        old_oid = self._current_oid(host)
        if expected is None or old_oid is None:
            # expected is None -> caller asserts no prior snapshot; a non-force push
            # of a NEW ref is rejected if a peer created it first (collision).
            # old_oid is None with expected set -> the ref vanished (release/race);
            # creating it is the right move (the lane is free now).
            push = self._git(["push", self.remote, f"{new_oid}:{ref}"])
        else:
            # Update path: the remote rejects unless its ref oid still matches old_oid.
            push = self._git(["push", self.remote, f"{new_oid}:{ref}",
                              f"--force-with-lease={ref}:{old_oid}"])
        if push.get("returncode") == 0:
            self._oid[host] = new_oid
            return True, snapshot.get("fleet_epoch")
        return False, self._observed_epoch(host)


# --------------------------------------------------------------------------- pure core


def is_global_lane(lane: str) -> bool:
    return lane in GLOBAL_LANES


def foreign_record_live(
    record: dict[str, Any], *, now: float, ttl_s: float, skew_margin_s: float
) -> bool:
    """Is a PUBLISHED (foreign) lease still live? Pure function of stamps + now.

    Liveness = local_now < heartbeat_at + ttl + skew_margin. We read heartbeat_at if
    present (refreshed by `dos lease-lane heartbeat`), else fall back to published_at,
    else acquired_at. An unreadable record is NOT live (refuse-don't-guess handles the
    refusal separately; for liveness we conservatively treat it as not-holding).
    """
    if record.get("_unreadable"):
        return False
    stamp = (
        _epoch_seconds(record.get("heartbeat_at"))
        or _epoch_seconds(record.get("published_at"))
        or _epoch_seconds(record.get("acquired_at"))
    )
    if stamp is None:
        # No usable timestamp -> cannot age it; treat as live so we don't silently
        # steal a lane whose record we merely failed to parse the time of.
        return True
    return now < stamp + ttl_s + skew_margin_s


def _epoch_seconds(value: Any) -> float | None:
    """Coerce a stamp (epoch float, or ISO-8601 'YYYY-MM-DDTHH:MM:SSZ') to epoch seconds."""
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str) and value.strip():
        s = value.strip().replace("Z", "+00:00")
        try:
            import datetime as _dt

            return _dt.datetime.fromisoformat(s).timestamp()
        except ValueError:
            return None
    return None


def union_live_leases(
    *,
    local: list[dict[str, Any]],
    fleet: dict[str, dict[str, Any]],
    self_host: str,
    now: float,
    ttl_s: float = DEFAULT_TTL_S,
    skew_margin_s: float = DEFAULT_SKEW_MARGIN_S,
) -> dict[str, Any]:
    """Build the live-lease union to feed `dos lease-lane acquire --leases`.

    Local WAL records are taken as-is (the kernel's own same-host liveness already
    filtered them). Foreign published records are included ONLY while live by the
    wall-clock overlay. Returns {leases, dropped, unreadable} so callers can report
    what was aged out and what tripped the SCHEMA_UNREADABLE rung.
    """
    leases: list[dict[str, Any]] = list(local)
    dropped: list[dict[str, Any]] = []
    unreadable: list[str] = []
    for host, snap in fleet.items():
        if host == self_host:
            continue  # our own published copy is redundant with the local WAL
        if snap.get("_unreadable"):
            unreadable.append(host)
            continue
        for rec in snap.get("leases") or []:
            if not isinstance(rec, dict):
                continue
            if foreign_record_live(rec, now=now, ttl_s=ttl_s, skew_margin_s=skew_margin_s):
                leases.append(rec)
            else:
                dropped.append({"host_id": host, "lane": rec.get("lane")})
    return {"leases": leases, "dropped": dropped, "unreadable": unreadable}


def would_collide(requested_lane: str, requested_tree: list[str], live_leases: list[dict[str, Any]]) -> dict[str, Any]:
    """Refuse-ONLY advisory pre-filter — a strict subset of the arbiter's authority.

    Catches the obvious cross-node collision (same lane name, or an exclusive-mode
    record whose tree shares a prefix) before a network round-trip. NEVER grants: a
    False here means "no obvious collision, ask the arbiter", not "go". If this ever
    disagrees with the kernel, the kernel wins.
    """
    req_tree = [t for t in (requested_tree or []) if t]
    for rec in live_leases:
        lane = rec.get("lane")
        if lane and lane == requested_lane:
            return {"collides": True, "reason": REASON_HELD_REMOTE,
                    "holder": rec.get("holder"), "host_id": rec.get("host_id"), "lane": lane}
        if str(rec.get("mode") or "").lower() == "exclusive":
            for a in req_tree:
                for b in rec.get("tree") or []:
                    if _tree_overlap(a, b):
                        return {"collides": True, "reason": REASON_HELD_REMOTE,
                                "holder": rec.get("holder"), "host_id": rec.get("host_id"),
                                "lane": rec.get("lane"), "tree": [a, b]}
    return {"collides": False}


def _tree_overlap(a: str, b: str) -> bool:
    """Conservative glob-prefix overlap: do two `dir/**` globs share a root segment chain?"""
    pa = a.replace("\\", "/").rstrip("/").removesuffix("/**")
    pb = b.replace("\\", "/").rstrip("/").removesuffix("/**")
    return pa == pb or pa.startswith(pb + "/") or pb.startswith(pa + "/")


def build_snapshot(*, host: str, leases: list[dict[str, Any]], now: float, prev_epoch: int | None) -> dict[str, Any]:
    """A publishable per-host snapshot. fleet_epoch is monotone (prev+1), the CAS key."""
    stamp = _iso(now)
    stamped = []
    for rec in leases:
        r = dict(rec)
        r.setdefault("host_id", host)
        r["published_at"] = stamp
        r.setdefault("heartbeat_at", r.get("acquired_at") or stamp)
        stamped.append(r)
    return {
        "schema": SCHEMA,
        "host_id": host,
        "published_at": stamp,
        "fleet_epoch": (int(prev_epoch) + 1) if isinstance(prev_epoch, int) else 1,
        "leases": stamped,
    }


def _iso(now: float) -> str:
    import datetime as _dt

    return _dt.datetime.fromtimestamp(now, _dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# --------------------------------------------------------------------------- verbs (wiring)


def do_publish(
    workspace: Path, store: FleetStore, *, host: str, now: float, live: KernelLive,
) -> tuple[dict[str, Any], int]:
    local = live(workspace)
    prev = store.read_fleet().get(host) or {}
    prev_epoch = prev.get("fleet_epoch") if isinstance(prev, dict) else None
    snapshot = build_snapshot(host=host, leases=local, now=now, prev_epoch=prev_epoch)
    ok, observed = store.compare_and_swap(host, prev_epoch, snapshot)
    if not ok:
        return ({"ok": False, "reason": REASON_STALE_RECORD,
                 "detail": f"fleet_epoch CAS lost (expected {prev_epoch}, store has {observed})"},
                EXIT_DENIED)
    return ({"ok": True, "host_id": host, "fleet_epoch": snapshot["fleet_epoch"],
             "published": len(snapshot["leases"])}, EXIT_OK)


def do_materialize(
    workspace: Path, store: FleetStore | None, *, scope: str, host: str, now: float,
    live: KernelLive, ttl_s: float = DEFAULT_TTL_S, skew_margin_s: float = DEFAULT_SKEW_MARGIN_S,
) -> tuple[dict[str, Any], int]:
    local = live(workspace)
    if scope == "local" or store is None:
        # Shard-by-node fast path: zero network, the local WAL is the whole truth.
        return ({"ok": True, "scope": "local", "leases": local,
                 "dropped": [], "unreadable": []}, EXIT_OK)
    u = union_live_leases(local=local, fleet=store.read_fleet(), self_host=host,
                          now=now, ttl_s=ttl_s, skew_margin_s=skew_margin_s)
    return ({"ok": True, "scope": "global", "leases": u["leases"],
             "dropped": u["dropped"], "unreadable": u["unreadable"]}, EXIT_OK)


def do_acquire(
    workspace: Path, store: FleetStore | None, *, lane: str, owner: str, kind: str, mode: str,
    run_id: str, scope: str, host: str, now: float, live: KernelLive, acquire: KernelAcquire,
) -> tuple[dict[str, Any], int]:
    mat, _ = do_materialize(workspace, store, scope=scope, host=host, now=now, live=live)
    union = mat["leases"]
    # Advisory pre-filter (refuse-only) before the network/kernel round-trip.
    pre = would_collide(lane, _lane_tree(union, lane) or [f"{lane}/**"], [r for r in union if r.get("lane") == lane])
    if pre.get("collides"):
        return ({"ok": False, "reason": pre["reason"], "holder": pre.get("holder"),
                 "host_id": pre.get("host_id"), "lane": lane,
                 "detail": "advisory pre-filter: a fleet peer holds this lane"}, EXIT_DENIED)
    res = acquire(workspace, lane=lane, owner=owner, kind=kind, mode=mode, run_id=run_id, leases=union)
    outcome = str(res.get("outcome") or "")
    if outcome != "acquire" or res.get("_returncode") not in (0, None):
        return ({"ok": False, "reason": REASON_HELD_REMOTE, "kernel": res,
                 "detail": "kernel refused the acquire"}, EXIT_DENIED)
    granted = res.get("lane")
    # A bare request (lane="") means "any free lane" — an auto-pick IS the grant.
    # A SPECIFIC request that the kernel auto-picked away from means that lane was
    # busy in the union: the caller asked for THIS lane and didn't get it. Report it
    # denied-for-the-request (they can retry bare to take any free lane) rather than
    # silently handing back a different lane than asked.
    if lane and granted != lane:
        return ({"ok": False, "reason": REASON_HELD_REMOTE, "kernel": res,
                 "lane": lane, "auto_picked": granted,
                 "detail": f"requested lane {lane!r} was busy; kernel auto-picked {granted!r} instead"},
                EXIT_DENIED)
    # Honor the grant by re-publishing this node's now-larger held set.
    if store is not None:
        do_publish(workspace, store, host=host, now=now, live=live)
    return ({"ok": True, "lane": res.get("lane"), "kernel": res}, EXIT_OK)


def _lane_tree(leases: list[dict[str, Any]], lane: str) -> list[str] | None:
    for r in leases:
        if r.get("lane") == lane and r.get("tree"):
            return list(r["tree"])
    return None


# --------------------------------------------------------------------------- cli


def _emit(payload: dict[str, Any], code: int) -> int:
    json.dump(payload, sys.stdout, indent=2)
    sys.stdout.write("\n")
    return code


def _store_for(
    args: argparse.Namespace, root: Path, *, lane: str = "", scope: str = "",
) -> FleetStore | None:
    """Pick the fleet store for an operation. The ACTIVATION seam for #21.

    An explicit ``--store`` / ``FAK_FLEET_LEASE_STORE`` always wins (``git`` ->
    GitRefStore, any other value -> a LocalDirStore at that path). With no explicit
    backend, the 3 global exclusive lanes (abi/release/global) — and an explicit
    ``materialize --scope global`` — default to the cross-node GitRefStore CAS
    (refs/dos-fleet-leases/*), because those are exactly the operations that need
    fleet-wide consensus. This wires the git transport ON for the global lanes rather
    than leaving it opt-in: a bare ``acquire --lane release`` now publishes +
    materializes-global through the remote ref, no flag. Shard-by-node lanes keep the
    zero-network LocalDirStore fast path (the common, always-available case)."""
    backend = getattr(args, "store", "") or os.environ.get("FAK_FLEET_LEASE_STORE", "")
    if backend == "git":
        return GitRefStore(root)
    if backend:
        return LocalDirStore(Path(backend))
    if scope == "global" or is_global_lane(lane):
        return GitRefStore(root)
    # Default store: a local dir under the registry (the MVP, hermetic-style).
    return LocalDirStore(root / "tools" / "_registry" / "fleet-leases")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Multi-node lease transport over `dos lease-lane` (fleet).")
    ap.add_argument("--workspace", default="", help="workspace root (default: repo root)")
    ap.add_argument("--store", default="", help="store path, or 'git' for the git-ref CAS backend (refs/dos-fleet-leases/*)")
    sub = ap.add_subparsers(dest="cmd", required=True)

    sub.add_parser("publish", help="publish this node's held leases to the fleet store")

    m = sub.add_parser("materialize", help="emit the live-lease union for `--leases`")
    m.add_argument("--scope", choices=["local", "global"], default="local")
    m.add_argument("--ttl-s", type=float, default=DEFAULT_TTL_S)
    m.add_argument("--skew-margin-s", type=float, default=DEFAULT_SKEW_MARGIN_S)

    a = sub.add_parser("acquire", help="materialize the fleet union, then acquire via the kernel")
    a.add_argument("--lane", default="", help="requested lane ('' = kernel auto-picks)")
    a.add_argument("--owner", default=None, help="lease holder tag (default: session id)")
    a.add_argument("--kind", default="", help="cluster|keyword|global|''")
    a.add_argument("--mode", default="exclusive", choices=["", "shared", "exclusive"])
    a.add_argument("--run-id", default="", help="RID-… spine id to stamp on the lease")
    a.add_argument("--scope", choices=["local", "global"], default="")

    r = sub.add_parser("release", help="release a held lane lease, then re-publish")
    r.add_argument("--lane", required=True)
    r.add_argument("--owner", default=None)

    h = sub.add_parser("heartbeat", help="refresh a held lease, then re-publish")
    h.add_argument("--lane", required=True)
    h.add_argument("--owner", default=None)
    h.add_argument("--loop-ts", default="")

    args = ap.parse_args(argv)
    root = Path(args.workspace).resolve() if args.workspace else repo_root()
    host = host_id()
    now = now_s()

    try:
        if args.cmd == "publish":
            payload, code = do_publish(root, _store_for(args, root), host=host, now=now, live=kernel_live)
            return _emit(payload, code)

        if args.cmd == "materialize":
            store = None if args.scope == "local" else _store_for(args, root, scope=args.scope)
            payload, code = do_materialize(
                root, store, scope=args.scope, host=host, now=now, live=kernel_live,
                ttl_s=args.ttl_s, skew_margin_s=args.skew_margin_s,
            )
            return _emit(payload, code)

        if args.cmd == "acquire":
            owner = args.owner or default_owner()
            scope = args.scope or ("global" if is_global_lane(args.lane) else "local")
            store = None if scope == "local" else _store_for(args, root, lane=args.lane, scope=scope)
            payload, code = do_acquire(
                root, store, lane=args.lane, owner=owner, kind=args.kind, mode=args.mode,
                run_id=args.run_id, scope=scope, host=host, now=now,
                live=kernel_live, acquire=kernel_acquire,
            )
            return _emit(payload, code)

        if args.cmd == "release":
            owner = args.owner or default_owner()
            res = kernel_release(root, lane=args.lane, owner=owner)
            do_publish(root, _store_for(args, root, lane=args.lane), host=host, now=now, live=kernel_live)
            return _emit({"ok": True, "kernel": res}, EXIT_OK)

        if args.cmd == "heartbeat":
            owner = args.owner or default_owner()
            res = kernel_heartbeat(root, lane=args.lane, owner=owner, loop_ts=args.loop_ts)
            do_publish(root, _store_for(args, root, lane=args.lane), host=host, now=now, live=kernel_live)
            return _emit({"ok": True, "kernel": res}, EXIT_OK)
    except Exception as exc:  # noqa: BLE001 - tooling seam: report, don't crash the caller
        return _emit({"ok": False, "reason": "internal", "detail": repr(exc)}, EXIT_INTERNAL)

    ap.error(f"unknown command {args.cmd!r}")
    return EXIT_USAGE


if __name__ == "__main__":
    raise SystemExit(main())
