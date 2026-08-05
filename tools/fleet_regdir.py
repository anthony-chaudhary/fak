"""fleet_regdir.py -- the ONE answer to "which fleet registry does this host use?".

Python twin of ``internal/accountprobe/regdir.go``. The Go reader learned the host
authority ladder in #5390; the Python side -- which is where the registry is actually
WRITTEN -- did not, and kept a flat two-rung fallback repeated in six modules::

    REG_DIR = os.environ.get("FLEET_REG_DIR", os.path.join(HERE, "_registry"))

That fallback is what FORKS the host. The watchdog (fleet_resume_watchdog.ps1 /
fleet_status.ps1) pins ``FLEET_REG_DIR`` at the per-user Fleet registry, so every
watchdog-launched writer lands there. Anything else that imports these modules with the
env UNSET -- an operator running ``fleet_sessions.py registry`` by hand, an agent
shelling out from the clone, a tick whose parent forgot to pin the env -- lands in the
clone-root ``tools/_registry`` instead and quietly maintains a SECOND registry beside the
live one. Both then look alive: two sessions.json files, two decision logs, two
transition trails, each a partial view, each overwriting only itself.

The damage is not that the second registry is stale. It is that the second registry is
CONFIDENT. It carries no probe ledger (the prober writes its ledger under the per-user
dir), so it can derive no seat block at all -- and a consumer that reads its silence as
"no seats blocked" routes work onto dead seats, because a 403 burns no quota and so a
dead seat reports FULL headroom (#5390). Meanwhile `fak` reports the split on every
launch, because the Go side can see both::

    accountprobe: fleet registry FORKED - reading user=...\\Fleet\\registry
    (blocks-known); also carrying state: clone=tools\\_registry (blocks-unknown)

Resolving through this module is what stops the fork from being CREATED. The rule is
AUTHORITY, then DERIVABILITY -- never recency, so the answer is a pure function of which
files exist and cannot FLAP between ticks when two writers each take a turn at being
freshest. See ``resolve`` for the ladder, rung by rung.

What this module deliberately does NOT do: pick a POLICY dir. ``accounts_policy.json`` is
operator CONFIG pinned to the repo (fleet_accounts.POLICY_DIR), not runtime state, and it
must not move with the registry -- see the comment above fleet_accounts.POLICY_DIR for the
drift that caused. Only runtime state follows this resolver.
"""
from __future__ import annotations

import os
import tempfile
from typing import Callable, NamedTuple

HERE = os.path.dirname(os.path.abspath(__file__))

# Registry dir file names, mirroring regdir.go. sessions.json is the roster
# fleet_sessions.py publishes; probe_ledger.jsonl is the append-only record
# account_probe.py writes. They are SIBLINGS by convention, not by construction -- a dir
# can carry one without the other, and that asymmetry is the whole of the grading below.
SESSIONS_FILE = "sessions.json"
LEDGER_FILE = "probe_ledger.jsonl"

# What a registry dir can honestly say about seat BLOCKS. Absence is not neutral here:
# a dir with sessions.json and no ledger beside it can derive no probe verdict, so it can
# derive no block either, and the "zero blocked seats" it reports means "cannot tell", not
# "nothing is blocked". Unknown is a THIRD state, distinct from both OK and blocked, so
# nothing gets stranded by treating unprobed as blocked.
HEALTH_BLOCKS_KNOWN = "blocks-known"      # a ledger is present; a block verdict is derivable
HEALTH_BLOCKS_UNKNOWN = "blocks-unknown"  # registry state, but no ledger -> no verdict
HEALTH_EMPTY = "empty"                    # no registry state at this path at all

# Rung names -- the fixed, ordered set of places a fleet registry may live on one host.
# They are identifiers in a report, so an operator reading a fork note can tell WHICH
# writer produced each registry without guessing from the path.
RUNG_ENV = "env"      # $FLEET_REG_DIR, named outright by the operator or the fleet
RUNG_STATE = "state"  # $FLEET_STATE_DIR/registry, named by the installed service
RUNG_USER = "user"    # the per-user Fleet registry the prober writes under
RUNG_TEMP = "temp"    # the Windows temp-root Fleet registry (last machine-local resort)
RUNG_CLONE = "clone"  # tools/_registry beside this file -- the legacy default


class RegSite(NamedTuple):
    """One candidate registry dir and what it actually holds on disk."""

    dir: str
    rung: str
    dir_exists: bool
    has_sessions: bool
    has_ledger: bool
    health: str


class RegChoice(NamedTuple):
    """The resolved registry dir plus the whole survey it was chosen from, so the
    decision is auditable and a FORK is reportable rather than silent."""

    dir: str
    rung: str
    health: str
    sites: tuple[RegSite, ...]
    forked: bool

    def blocks_derivable(self) -> bool:
        """Whether the chosen registry can derive a block verdict at all.

        False means the only honest statement about blocked seats is "unknown", and a
        caller that would otherwise publish "no seats blocked" is obliged to publish
        nothing instead."""
        return self.health == HEALTH_BLOCKS_KNOWN

    def fork_note(self) -> str:
        """The one-line operator-visible fork report, or "" on a single-registry host.

        Byte-compatible with accountprobe.RegChoice.ForkNote so an operator greps one
        string across both halves of the fleet."""
        if not self.forked:
            return ""
        others = [f"{s.rung}={s.dir} ({s.health})"
                  for s in self.sites
                  if s.dir != self.dir and s.health != HEALTH_EMPTY]
        return ("accountprobe: fleet registry FORKED — reading "
                f"{self.rung}={self.dir} ({self.health}); "
                f"also carrying state: {', '.join(others)}")


def _reg_dir_key(path: str) -> str:
    """Normalize a candidate path for duplicate detection: absolute where possible,
    cleaned, and case-folded on the one platform whose paths are case-insensitive."""
    try:
        key = os.path.abspath(path)
    except (OSError, ValueError):  # pragma: no cover - defensive; abspath is total in practice
        key = path
    key = os.path.normpath(key)
    return key.lower() if os.name == "nt" else key


def _file_present(path: str) -> bool:
    return os.path.isfile(path)


def _stat_site(rung: str, path: str) -> RegSite:
    has_sessions = _file_present(os.path.join(path, SESSIONS_FILE))
    has_ledger = _file_present(os.path.join(path, LEDGER_FILE))
    if has_ledger:
        health = HEALTH_BLOCKS_KNOWN
    elif has_sessions:
        health = HEALTH_BLOCKS_UNKNOWN
    else:
        health = HEALTH_EMPTY
    return RegSite(dir=path, rung=rung, dir_exists=os.path.isdir(path),
                   has_sessions=has_sessions, has_ledger=has_ledger, health=health)


def default_clone_dir() -> str:
    """The legacy clone-root registry: ``tools/_registry`` beside this file.

    File-relative, not cwd-relative, so a script run from anywhere in the workspace names
    the same dir it always has. (regdir.go's clone rung is cwd-relative, which coincides
    with this whenever fak is run from the clone root -- the case the fork note fires in.)"""
    return os.path.join(HERE, "_registry")


def sites(clone_dir: str | None = None) -> list[RegSite]:
    """Every candidate registry dir on this host, in fixed AUTHORITY order, each stat'd.

    The order is a constant of the program, never a function of the clock: ordering by
    modification time would let the choice FLAP between ticks, because two writers run on
    their own schedules and each is the freshest for part of every minute.

    Duplicates collapse to the highest-authority rung that names them, so the ordinary
    production wiring -- FLEET_REG_DIR pointing AT the per-user Fleet registry -- is one
    site, not a phantom fork."""
    out: list[RegSite] = []
    seen: set[str] = set()

    def add(rung: str, path: str | None) -> None:
        path = (path or "").strip()
        if not path:
            return
        key = _reg_dir_key(path)
        if key in seen:
            return
        seen.add(key)
        out.append(_stat_site(rung, path))

    add(RUNG_ENV, os.environ.get("FLEET_REG_DIR"))
    state = (os.environ.get("FLEET_STATE_DIR") or "").strip()
    if state:
        add(RUNG_STATE, os.path.join(state, "registry"))
    local_app_data = (os.environ.get("LOCALAPPDATA") or "").strip()
    if local_app_data:
        add(RUNG_USER, os.path.join(local_app_data, "Fleet", "registry"))
    if os.name == "nt":
        add(RUNG_TEMP, os.path.join(tempfile.gettempdir(), "Fleet", "registry"))
    add(RUNG_CLONE, clone_dir if clone_dir is not None else default_clone_dir())
    return out


def resolve(clone_dir: str | None = None) -> RegChoice:
    """Pick the registry dir this process reads and writes, and survey the rest.

    AUTHORITY, then DERIVABILITY -- never recency:

    1. ``$FLEET_REG_DIR``, when set, wins outright. An operator (or the fleet launcher)
       naming the dir is the highest authority there is, and every existing wiring --
       fleet_resume_watchdog.ps1, fleet_status.ps1, fleet_control_pane.py -- depends on it.
    2. Else ``$FLEET_STATE_DIR/registry``, when set: the installed service's declared home.
       Same reasoning -- a declaration outranks a discovery.
    3. Else the first DISCOVERED site, in rung order, that can actually derive a block
       (it carries a probe ledger). This is the convergence: the per-user Fleet dir, which
       has the ledger, now outranks the clone-root dir, which never does.
    4. Else the first site carrying any registry state, graded ``blocks-unknown`` so a
       caller cannot mistake its silence for "nothing blocked".
    5. Else the first site whose directory merely exists, matching what the Go resolver
       settles on so a fak process and a fleet script on one host converge.
    6. Else nothing exists anywhere and the clone-root path is returned UNCHANGED --
       exactly the pre-existing default, so a fresh checkout and CI behave as they always
       have, and no test that never touched a registry starts reading the host's.

    No branch consults a modification time. Repeated calls over an unchanged filesystem
    return the same dir, and making the wrong dir the freshest changes nothing."""
    surveyed = sites(clone_dir)
    forked = sum(1 for s in surveyed if s.health != HEALTH_EMPTY) > 1

    def take(match: Callable[[RegSite], bool]) -> RegSite | None:
        return next((s for s in surveyed if match(s)), None)

    chosen = (take(lambda s: s.rung == RUNG_ENV)
              or take(lambda s: s.rung == RUNG_STATE)
              or take(lambda s: s.health == HEALTH_BLOCKS_KNOWN)
              or take(lambda s: s.health == HEALTH_BLOCKS_UNKNOWN)
              or take(lambda s: s.dir_exists)
              # Nothing anywhere: the clone rung is always last and is the legacy default.
              or surveyed[-1])
    return RegChoice(dir=chosen.dir, rung=chosen.rung, health=chosen.health,
                     sites=tuple(surveyed), forked=forked)


def reg_dir(clone_dir: str | None = None) -> str:
    """The registry dir this process should read and write. See ``resolve``."""
    return resolve(clone_dir).dir


def fork_note(clone_dir: str | None = None) -> str:
    """The fork report for this host, or "" when it runs a single registry."""
    return resolve(clone_dir).fork_note()


def _main(argv: list[str] | None = None) -> int:
    """``python tools/fleet_regdir.py`` -- print the survey an operator needs to see when
    `fak` reports a FORKED registry: which dir wins, why, and what else is on disk."""
    import argparse
    import json

    ap = argparse.ArgumentParser(description="resolve the fleet registry dir for this host")
    ap.add_argument("--json", action="store_true", help="machine-readable survey")
    args = ap.parse_args(argv)

    choice = resolve()
    if args.json:
        print(json.dumps({"dir": choice.dir, "rung": choice.rung, "health": choice.health,
                          "forked": choice.forked, "fork_note": choice.fork_note(),
                          "sites": [s._asdict() for s in choice.sites]}, indent=2))
        return 0
    print(f"registry: {choice.dir}")
    print(f"  rung={choice.rung} health={choice.health} forked={str(choice.forked).lower()}")
    for s in choice.sites:
        mark = "->" if s.dir == choice.dir else "  "
        print(f"  {mark} {s.rung:<6} {s.health:<14} {s.dir}")
    note = choice.fork_note()
    if note:
        print()
        print(note)
        print("  remedy: delete the runtime state (sessions.json, probe_ledger.jsonl) from the"
              " registry that is NOT the one above, or pin FLEET_REG_DIR on the writer that"
              " maintains it.")
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry
    raise SystemExit(_main())
