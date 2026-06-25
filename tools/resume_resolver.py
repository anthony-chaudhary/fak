#!/usr/bin/env python3
r"""resume_resolver.py -- resolve WHICH account ``claude --resume <sid>`` should
run under, re-homing the transcript onto a healthy account when the owning
account is rate-limited / blocked.

``claude --resume <sid>`` is CLAUDE_CONFIG_DIR + cwd scoped: it only finds the
conversation under ``<config>/projects/<sanitized-cwd>/<sid>.jsonl``, and ONLY
ever under the *active* ``CLAUDE_CONFIG_DIR``. The ``c`` launcher rotates
accounts per launch, so a bare ``c --resume <id>`` would 404 unless it PINS to
the owning account. The existing PowerShell finder (Find-ClaudeSessionAccount)
pins to the owner -- but when that owner is THROTTLED, pinning to it yields a
*dead* resume: every model call is refused until the limit resets, which for a
weekly cap is days. That is the gap the operator hit on
``c --resume <sid>`` against a session owned by a rate-limited account.

This resolver closes it. It locates the owner host-LAST, newest-mtime (the same
selection rule as the PowerShell finder), checks the owner's LIVE availability
via :mod:`fleet_accounts`, and decides:

  * owner available  -> ``PIN`` to the owner (no copy; the safe default, and the
                        same answer the PS finder already gives).
  * owner blocked    -> ``REHOME``: copy the transcript (+ its ``<sid>/`` sidecar)
                        onto the least-loaded healthy Claude worker and pin THERE.
  * no healthy acct  -> ``PIN_BLOCKED``: pin to the owner anyway (best effort --
                        nothing better exists; the resume waits for the reset).

It is the interactive ``c --resume`` analogue of the headless
:mod:`fleet_resume_watchdog` re-home: the SAME locate-the-owner mechanism, the
SAME copy primitive (``rehome_transcript``), the SAME target ranking
(``_rehome_targets``). The two notes
``c-resume-cross-account-recovery`` / ``account-resume-rehome-and-dryrun``
describe exactly this split: owner reachable -> pin, don't copy; owner throttled
-> copy on purpose. Until now only the watchdog did the second half; this brings
it to the interactive path.

Output contract (so the PowerShell ``c`` can consume it):
  stdout   ONE line: the config dir to set ``CLAUDE_CONFIG_DIR`` to (nothing on
           error). With ``--json``, the full decision record instead.
  stderr   human diagnostics (the action + why, duplicate-account notes).
  exit     0 resolved, 1 session not found, 2 internal error.
  --dry-run  decide + report but do NOT copy. stdout still shows the intended
             pin dir; ``--json`` marks it ``would_rehome`` so a caller never
             pins to a target it has not actually populated.

CLI:
  python tools/resume_resolver.py <session-id>            # print pin config dir
  python tools/resume_resolver.py <session-id> --json     # full record
  python tools/resume_resolver.py <session-id> --dry-run  # decide, don't copy
"""
from __future__ import annotations

import argparse
import glob
import inspect
import json
import math
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import fleet_accounts  # noqa: E402
import fleet_resume_watchdog  # noqa: E402  (rehome_transcript -- the canonical copy)
import fleet_sessions  # noqa: E402  (_rehome_targets -- the canonical target ranking)

# How many ranked re-home candidates to live-probe before falling back to the
# best-ranked one. Bounds the cost of confirming a target actually serves; a probe
# is ~2-4s, so a handful covers the realistic healthy-account count without making a
# fully-walled fleet probe every dir.
_MAX_TARGET_PROBES = 4


def _carried_throttle_block(owner_status: dict) -> bool:
    """True when the owner is reported blocked by a usage throttle that was CARRIED
    in the registry, not confirmed by a fresh live probe.

    A carried throttle is the stale-block risk: the registry keeps a `reset` string
    from whenever the limit was last seen, and a bare-time reset ("3pm") can read as
    a future reset for hours after the limit actually cleared -- so the account is
    really serving while the roster still calls it throttled. A re-home off such an
    owner abandons a healthy account (and copies onto a target that may be no better).
    runtime_status already DEFERS to a fresh probe (status_source=="probe"), so we
    only re-probe when the block is the weaker, registry-carried usage kind."""
    if owner_status.get("available", True):
        return False
    if owner_status.get("status_source") == "probe":
        return False  # already confirmed live -- trust it, do not re-probe
    return owner_status.get("block_kind") == "usage"


def _probe_owner_available(owner: dict) -> dict | None:
    """Live-probe the owner account, returning its fresh runtime_status, or None if
    a probe could not be run. Used to confirm a CARRIED usage throttle before
    re-homing off the owner -- a fresh OK probe overrides the stale carried block
    (the same override runtime_status already applies to a probe row)."""
    try:
        import account_probe  # local import: only the re-home path needs the prober
    except ImportError:
        return None
    row = {"kind": "worker", "account": owner["account"],
           "tag": fleet_accounts.account_tag(owner["account"]),
           "dir": owner["config_dir"], "product": "claude"}
    try:
        verdict = account_probe.probe_account(row)
    except Exception:
        return None
    status = str(verdict.get("status") or "").upper()
    if status == "OK":
        return {"available": True, "block_reason": "", "status_source": "probe"}
    reason = str(verdict.get("block_reason") or verdict.get("reason") or "blocked")
    return {"available": False, "block_reason": reason, "status_source": "probe",
            "block_kind": verdict.get("block_kind")}


def _is_host(config_dir: str) -> bool:
    """True for the host ``~/.claude`` login -- the account ``c`` keeps OFF the
    rotation, so it is only ever chosen as an owner when it is the SOLE one."""
    return os.path.basename(config_dir.rstrip("\\/")) == ".claude"


def project_slug(path: str) -> str:
    """The on-disk session-store slug Claude derives from a working directory:
    drive colon, path separators, and dots all become '-' (``C:\\work\\fak`` ->
    ``C--work-fak``). Mirrors the PowerShell launcher's Get-ClaudeProjectSlug EXACTLY
    so the resolver lands a re-home copy under the same slug ``claude --resume`` will
    look under from the caller's cwd -- the cross-directory resume fix."""
    return re.sub(r"[\\/:.]", "-", path)


def _rehome(rehome_fn, src_cfg: str, dst_cfg: str, project: str, sid: str,
            dest_projects: list[str] | None = None) -> bool:
    """Call ``rehome_fn`` with ``dest_projects`` only when it accepts that parameter.

    rehome_transcript gained ``dest_projects`` for the cross-dir fix, but this module
    lives in a shared multi-session tree where a peer can transiently revert it to the
    4-arg form -- and tests inject simpler stubs. Degrade gracefully: if the callable
    can't take dest_projects, fall back to the owner-slug-only copy rather than raising."""
    if dest_projects:
        try:
            params = inspect.signature(rehome_fn).parameters
            takes = ("dest_projects" in params
                     or any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values()))
        except (TypeError, ValueError):
            takes = False
        if takes:
            return rehome_fn(src_cfg, dst_cfg, project, sid, dest_projects=dest_projects)
    return rehome_fn(src_cfg, dst_cfg, project, sid)


def locate_owner(sid: str, home: str) -> dict | None:
    """Return the on-disk owner record for session ``sid``, or ``None``.

    Scans every ``<home>/.claude*`` account dir for
    ``projects/*/<sid>.jsonl`` and selects the owner host-LAST, newest-mtime:
    among non-host accounts the freshest ``.jsonl`` wins; the host (``~/.claude``)
    is chosen only when it is the SOLE owner. This mirrors the PowerShell
    ``Find-ClaudeSessionAccount`` selection exactly, so the resolver and the
    launcher's fallback never disagree about WHO owns a session. Scanning every
    ``projects/*`` (not just the current cwd slug) also makes the lookup robust
    to a session created under a different working directory.

    The record carries the discovered ``project`` dir name (so the re-home copy
    lands at the exact path ``claude --resume`` will look under) plus a
    ``dup_count`` / ``all_accounts`` summary for the duplicate-fork note.
    """
    matches = _locate_matches(sid, home)
    if not matches:
        return None
    non_host = [m for m in matches if not m["is_host"]]
    pool = non_host or matches
    pool.sort(key=lambda m: m["mtime"], reverse=True)
    owner = dict(pool[0])
    owner["dup_count"] = len(matches)
    owner["all_accounts"] = sorted(m["account"] for m in matches)
    return owner


def _locate_matches(sid: str, home: str) -> list[dict]:
    """Every ``<home>/.claude*`` account dir holding this session's ``.jsonl``, each
    as ``{config_dir, account, project, mtime, size, is_host}``. The raw input shared
    by the owner pick and the duplicate-owner re-selection. ``size`` is the
    transcript's byte length -- the cheap content-freshness signal the re-selection
    uses to refuse pinning a copy that is missing turns the freshest one has (these
    transcripts are append-only, so fewer turns == strictly smaller)."""
    matches: list[dict] = []
    for acct_dir in glob.glob(os.path.join(home, ".claude*")):
        if not os.path.isdir(acct_dir):
            continue
        proj_root = os.path.join(acct_dir, "projects")
        if not os.path.isdir(proj_root):
            continue
        for f in glob.glob(os.path.join(proj_root, "*", sid + ".jsonl")):
            try:
                st = os.stat(f)
            except OSError:
                continue
            matches.append({
                "config_dir": acct_dir,
                "account": os.path.basename(acct_dir),
                "project": os.path.basename(os.path.dirname(f)),
                "mtime": st.st_mtime,
                "size": st.st_size,
                "is_host": _is_host(acct_dir),
            })
    return matches


def _reselect_duplicate_owner(sid: str, owner: dict, home: str,
                              probe_fn) -> dict | None:
    """For a session duplicated across accounts (a prior re-home), confirm the
    newest-mtime owner actually serves; if it does not, fall back to another copy's
    account that a live probe confirms serving.

    Returns the owner record to use (carrying ``dup_count`` / ``all_accounts``), or
    ``None`` to keep the original pick (it serves, or nothing better could be
    confirmed). Candidates are tried newest-mtime first, host LAST -- the same
    ordering as the primary owner pick -- so an older copy is chosen ONLY when the
    freshest one is provably walled. This is the operator's failure: the resume was
    owned by a limited day24 (the re-home target, stamped newest) while q-netra (the
    original copy) was serving.

    Pinning that older copy is safe ONLY while it has the SAME content as the walled
    freshest one -- the state right after a re-home, when both are byte-identical and
    only the mtime differs. Once the session is actually USED on the freshest copy it
    advances past the siblings, so a sibling that is content-behind (strictly smaller,
    these transcripts being append-only) is missing the newest turns. Pinning it then
    silently resumes a STALE transcript and loses the latest exchanges -- the
    "resume pins badly" failure.

    So this returns a TYPED decision rather than a bare owner:
      * ``None``                               freshest serves (or is unprobeable), or
                                               nothing better could be confirmed -- keep the
                                               normal pick.
      * ``{"mode": "pin", "owner": rec}``      a serving sibling whose content is at parity
                                               with (or ahead of) the walled freshest -- pin
                                               it, no copy needed.
      * ``{"mode": "rehome", "source": rec,
            "target": rec}``                   the freshest is walled and the only serving
                                               siblings are content-BEHIND it. Re-home the
                                               freshest's FULL content onto the least-stale
                                               serving sibling and pin there. That sibling
                                               already holds this session and serves, so the
                                               move carries every turn AND is exempt from the
                                               burst-spread re-home cap (it refreshes an
                                               account in place rather than admitting a new
                                               one) -- which would otherwise strand a busy
                                               operator on PIN_BLOCKED until the reset."""
    matches = _locate_matches(sid, home)
    if len(matches) <= 1:
        return None
    non_host = [m for m in matches if not m["is_host"]]
    ordered = sorted(non_host or matches, key=lambda m: m["mtime"], reverse=True)
    first = ordered[0]
    probed = probe_fn({"account": first["account"],
                       "config_dir": first["config_dir"]})
    if probed is None or probed.get("available"):
        return None  # freshest copy serves (or unprobeable) -> keep the normal pick

    def _stamp(cand: dict) -> dict:
        rec = dict(cand)
        rec["dup_count"] = len(matches)
        rec["all_accounts"] = sorted(m["account"] for m in matches)
        return rec

    behind_serving = None  # least-stale serving sibling that is content-behind the freshest
    for cand in ordered[1:]:
        p = probe_fn({"account": cand["account"], "config_dir": cand["config_dir"]})
        if p is None or not p.get("available"):
            continue
        if cand.get("size", 0) >= first.get("size", 0):
            return {"mode": "pin", "owner": _stamp(cand)}  # at parity -> pin, no copy
        if behind_serving is None:
            behind_serving = cand  # freshest serving-but-stale sibling; carry content here
    if behind_serving is not None:
        return {"mode": "rehome", "source": _stamp(first),
                "target": _stamp(behind_serving)}
    return None


def _discover_availability(home: str) -> list[dict]:
    """Live availability records for the routable Claude/opencode workers, shaped
    for :func:`fleet_sessions._rehome_targets` (account / available / live_sessions
    / active_sessions / tag / config_dir). Built from the same roster + runtime
    status the account switcher uses, so a re-home target is never an account the
    switcher itself would refuse to offer."""
    rows = fleet_accounts.annotated_roster(home)
    return [
        {
            "account": r["account"],
            "available": r.get("available"),
            "live_sessions": r.get("live_sessions"),
            "active_sessions": r.get("active_sessions"),
            "tag": r.get("tag"),
            "config_dir": r.get("dir"),
        }
        for r in rows
        if fleet_accounts.routable_worker(r)
    ]


def resolve(sid: str, home: str | None = None, *,
            availability: list[dict] | None = None,
            owner_status: dict | None = None,
            dry_run: bool = False,
            rehome_fn=None,
            probe_fn=None,
            probe_owner: bool = True,
            cwd: str | None = None) -> dict:
    """Decide where ``claude --resume <sid>`` should run.

    ``availability`` / ``owner_status`` / ``rehome_fn`` / ``probe_fn`` are injectable
    so the decision is unit-testable without a live registry or real account dirs;
    production passes none and they are read from :mod:`fleet_accounts` and
    copied with :func:`fleet_resume_watchdog.rehome_transcript`.

    ``probe_owner`` (default True) live-probes the owner before re-homing when its
    block is a CARRIED usage throttle (not a fresh probe). This catches a stale
    bare-time reset that reads as future for hours after the limit cleared -- without
    it the resolver re-homes off a healthy account onto an arbitrary target. Set
    False (or pass ``--no-probe``) to skip the probe and trust the carried status.

    ``cwd`` (default: the process cwd) is the directory ``claude --resume`` will run
    from. Its project slug is added to the re-home copy's destinations -- and, on the
    pin path, mirrored WITHIN the owner account -- so the resume works from a DIRECTORY
    DIFFERENT from where the session was created (the cross-directory fix: run
    ``c --resume <sid>`` from slack-helpers for a session born under fak).
    """
    home = home or fleet_accounts.USER
    rehome_fn = rehome_fn or fleet_resume_watchdog.rehome_transcript
    probe_fn = probe_fn or _probe_owner_available
    cwd_slug = project_slug(cwd or os.getcwd())

    owner = locate_owner(sid, home)
    if owner is None:
        return {
            "ok": False, "action": "NOT_FOUND", "session": sid,
            "pin_config_dir": None,
            "reason": "no ~/.claude* account holds this session id",
        }

    # A session present in MORE THAN ONE account dir is the signature of a prior
    # re-home: locate_owner picks the newest-mtime copy, which is the re-home TARGET,
    # not necessarily a serving account. The registry can report that target available
    # purely because no limit was ever recorded for it (absence of evidence), so a
    # naive pin lands the resume on a secretly-walled account while the true original
    # owner is still serving. When the session is duplicated, confirm the chosen owner
    # actually serves; if it does not, re-pick among the OTHER copies' accounts. This
    # is the operator's exact failure: resumed onto a limited day24 while q-netra (the
    # original owner) was serving. (#resume-resolver: bad re-home target as owner)
    forced_target = None
    if (probe_owner and owner_status is None and owner.get("dup_count", 1) > 1
            and len(owner.get("all_accounts", [])) > 1):
        decision = _reselect_duplicate_owner(sid, owner, home, probe_fn)
        if decision and decision.get("mode") == "pin":
            # A serving sibling at content-parity with the walled freshest -> pin it,
            # no copy (the original optimisation: the re-home target was stamped newest
            # but the original copy holds identical content and still serves).
            rec_owner_reselect = {"from": owner["account"],
                                  "to": decision["owner"]["account"]}
            owner = decision["owner"]
        elif decision and decision.get("mode") == "rehome":
            # The freshest copy is walled but a serving sibling already holds this
            # session AND is content-behind it. Keep the freshest as the copy SOURCE
            # (owner is unchanged -- locate_owner already picked it) so every turn is
            # carried, and force the landing onto that proven-serving sibling below.
            forced_target = decision["target"]
            rec_owner_reselect = None
        else:
            rec_owner_reselect = None
    else:
        rec_owner_reselect = None

    if owner_status is None:
        owner_status = fleet_accounts.runtime_status(owner["account"])
    owner_available = bool(owner_status.get("available", True))
    block_reason = str(owner_status.get("block_reason") or "blocked")

    rec = {
        "ok": True, "session": sid, "project": owner["project"],
        "owner_account": owner["account"], "owner_config_dir": owner["config_dir"],
        "owner_available": owner_available,
        "owner_block_reason": owner_status.get("block_reason", ""),
        "dup_count": owner.get("dup_count", 1),
        "all_accounts": owner.get("all_accounts", [owner["account"]]),
    }
    if rec_owner_reselect:
        rec["owner_reselected"] = rec_owner_reselect

    # Before trusting a CARRIED usage throttle (one the registry kept, not a fresh
    # probe), re-check the owner live -- a stale bare-time reset can keep an account
    # marked throttled for hours after it actually cleared. A fresh OK probe means the
    # owner is serving: pin to it and skip the needless re-home. (#resume-resolver
    # false throttle: re-homed off a healthy q-netra whose "12:30am"/"3pm" reset had
    # already passed.)
    if probe_owner and not owner_available and _carried_throttle_block(owner_status):
        probed = probe_fn(owner)
        if probed is not None:
            rec["owner_probe"] = {
                "available": probed.get("available"),
                "block_reason": probed.get("block_reason", ""),
            }
            owner_available = bool(probed.get("available"))
            block_reason = str(probed.get("block_reason") or block_reason)
            rec["owner_available"] = owner_available
            if owner_available:
                rec["owner_block_reason"] = ""

    # Owner reachable -> pin to it, no cross-ACCOUNT copy. This is the unchanged, safe
    # default (and exactly what the PS finder already returns) -- EXCEPT that the session
    # may be stored under a DIFFERENT cwd-slug than the one claude --resume looks up from
    # the caller's directory. The owner account is right but the transcript isn't under the
    # resume cwd's slug -> a 404 from a different folder. So mirror it WITHIN the owner
    # account into the cwd slug (same account, owner->owner). (cross-dir fix)
    if owner_available:
        pinned_reason = "owner account is available -- pin to it (no copy)"
        if rec.get("owner_probe", {}).get("available"):
            pinned_reason = ("owner's carried throttle was stale -- live probe OK, "
                             "pin to owner (no re-home)")
        if cwd_slug and cwd_slug != owner["project"] and not dry_run:
            mirrored = _rehome(rehome_fn, owner["config_dir"], owner["config_dir"],
                               owner["project"], sid, dest_projects=[cwd_slug])
            if mirrored:
                rec["mirrored_to_cwd_slug"] = cwd_slug
                pinned_reason += f" (mirrored into cwd slug {cwd_slug})"
        elif cwd_slug and cwd_slug != owner["project"]:
            rec["would_mirror_to_cwd_slug"] = cwd_slug
        rec.update({
            "action": "PIN", "rehomed": False,
            "pin_account": owner["account"], "pin_config_dir": owner["config_dir"],
            "reason": pinned_reason,
        })
        return rec

    # Owner blocked/throttled -> re-home its FULL transcript onto a healthy Claude
    # worker and pin there.
    if forced_target is not None:
        # The duplicate reselect found a serving sibling that ALREADY holds this
        # session (e.g. q-netra still serving while the freshest copy's day24 is
        # walled). It is proven-serving and already admitted, so land the freshest
        # content there directly -- skipping the roster ranking AND the burst-spread
        # re-home cap. That cap exists to stop a FLEET of throttled sessions piling
        # onto one busy account; it must not strand a single operator's interactive
        # resume on PIN_BLOCKED when an account already running their session serves.
        tgt = forced_target
        rec["rehome_to_sibling"] = forced_target["account"]
    else:
        # Re-home onto the least-loaded healthy Claude worker, the same ranking the
        # headless watchdog uses.
        if availability is None:
            availability = _discover_availability(home)
        targets = fleet_sessions._rehome_targets(availability, owner["account"])
        if not targets:
            # The fleet burst-spread cap (REHOME_CAP) just excluded every available
            # account by LOAD. That cap exists to stop a BURST of throttled sessions
            # stampeding one seat -- but this is a SINGLE interactive resume, and the
            # comment on the forced_target path above says the cap must not strand one
            # operator's resume on PIN_BLOCKED. So retry UNCAPPED: if a healthy Claude
            # account exists and was dropped only for being over the cap (not because it
            # is blocked), re-home there anyway. PIN_BLOCKED stays the verdict only when
            # even the uncapped pool is empty -- i.e. NO healthy account exists at all.
            relief = fleet_sessions._rehome_targets(
                availability, owner["account"], cap=math.inf)
            if not relief:
                rec.update({
                    "action": "PIN_BLOCKED", "rehomed": False,
                    "pin_account": owner["account"], "pin_config_dir": owner["config_dir"],
                    "reason": (f"owner blocked ({block_reason}) and no healthy Claude "
                               "worker available -- pin to owner; resume waits for reset"),
                })
                return rec
            rec["cap_relief"] = {
                "rehome_cap": fleet_sessions.REHOME_CAP,
                "note": ("all available accounts were over the fleet burst cap; a single "
                         "interactive resume relaxes it onto the least-loaded healthy seat"),
            }
            targets = relief

        # The roster's `available` is only "nothing bad was recorded" -- it can offer an
        # account that is itself limited but never probed (so no throttle row exists). A
        # re-home onto such a target just moves the session from one walled account to
        # another. So live-probe candidates top-down and pick the first PROVEN-serving one;
        # an account a probe confirms blocked is skipped. (#resume-resolver: re-homed onto
        # day24, which was itself usage-limited.) Bounded so a fully-walled fleet does not
        # probe forever; if every checked target is blocked we still fall back to the
        # best-ranked one rather than stranding the resume.
        tgt = targets[0]
        if probe_owner:
            checked: list[dict] = []
            for cand in targets[:_MAX_TARGET_PROBES]:
                probed = probe_fn({"account": cand["account"],
                                   "config_dir": cand.get("config_dir")
                                   or os.path.join(home, cand["account"])})
                if probed is None:
                    tgt = cand  # cannot probe -> trust the ranking, take it
                    break
                checked.append({"account": cand["account"],
                                "available": probed.get("available"),
                                "block_reason": probed.get("block_reason", "")})
                if probed.get("available"):
                    tgt = cand
                    break
            else:
                # ran the whole bounded slice without a proven-serving target. If EVERY
                # checked candidate probed as blocked, re-homing would only move the resume
                # from one walled account to another -- so pin to the owner and wait for a
                # reset instead (PIN_BLOCKED). Only when no candidate could be probed at all
                # (checked empty) do we fall back to the best-ranked best-effort landing.
                if checked and all(not c["available"] for c in checked):
                    rec["target_probes"] = checked
                    rec.update({
                        "action": "PIN_BLOCKED", "rehomed": False,
                        "pin_account": owner["account"],
                        "pin_config_dir": owner["config_dir"],
                        "reason": (f"owner blocked ({block_reason}) and every probed "
                                   f"re-home target is also limited -- pin to owner; "
                                   f"resume waits for reset"),
                    })
                    return rec
                tgt = targets[0]
            if checked:
                rec["target_probes"] = checked

    tgt_cfg = tgt.get("config_dir") or os.path.join(home, tgt["account"])
    # Land the copy under the owner's original slug AND the launching cwd's slug, so
    # `claude --resume` finds it whether the operator resumes from the session's birth
    # directory or a different one (the cross-directory fix).
    dest_slugs = [cwd_slug] if cwd_slug and cwd_slug != owner["project"] else []
    if dest_slugs:
        rec["dest_project_slugs"] = [owner["project"], *dest_slugs]
    if not dry_run:
        copied = _rehome(rehome_fn, owner["config_dir"], tgt_cfg, owner["project"], sid,
                         dest_projects=dest_slugs)
        if not copied:
            rec.update({
                "action": "PIN_BLOCKED", "rehomed": False,
                "pin_account": owner["account"], "pin_config_dir": owner["config_dir"],
                "reason": "re-home source transcript missing -- pin to owner",
            })
            return rec
        # shutil.copy2 preserves the SOURCE mtime, so the re-homed copy would tie
        # the throttled original -- and the "host-last, newest-mtime" owner pick
        # (here AND the PowerShell fallback finder) could then re-select the walled
        # account on the next launch. Stamp every re-homed copy (owner slug + any cwd
        # slug) as newest so the healthy target is the unambiguous owner from now on
        # (it also stops a redundant re-copy each invocation until the live resume
        # writes to it).
        for slug in (owner["project"], *dest_slugs):
            try:
                os.utime(os.path.join(tgt_cfg, "projects", slug, sid + ".jsonl"), None)
            except OSError:
                pass

    tgt_tag = tgt.get("tag") or tgt["account"]
    target_confirmed = any(
        c["account"] == tgt["account"] and c["available"]
        for c in rec.get("target_probes", []))
    confirm_note = " (live-probe OK)" if target_confirmed else ""
    rec.update({
        "action": "REHOME",
        "rehomed": not dry_run,
        "would_rehome": dry_run,
        "pin_account": tgt["account"], "pin_config_dir": tgt_cfg,
        "source_config_dir": owner["config_dir"],
        "reason": (f"owner blocked ({block_reason}) -- "
                   f"{'would re-home' if dry_run else 're-homed'} transcript onto "
                   f"{tgt_tag}{confirm_note} and pin there"),
    })
    return rec


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    ap = argparse.ArgumentParser(
        prog="resume_resolver",
        description="Resolve the CLAUDE_CONFIG_DIR for `claude --resume <sid>`, "
                    "re-homing onto a healthy account when the owner is throttled.")
    ap.add_argument("session", help="session id to resume")
    ap.add_argument("--home", default=None,
                    help="user home holding the .claude* dirs (default: ~)")
    ap.add_argument("--dry-run", action="store_true",
                    help="decide and report but do NOT copy the transcript")
    ap.add_argument("--no-probe", action="store_true",
                    help="trust the carried throttle; do NOT live-probe the owner "
                         "before re-homing (faster, but can re-home off a stale block)")
    ap.add_argument("--cwd", default=None,
                    help="directory `claude --resume` will run from (default: the "
                         "process cwd); its project slug is added to the re-home copy "
                         "so a resume works from a different folder than the session's "
                         "birth directory")
    ap.add_argument("--json", action="store_true",
                    help="emit the full decision record instead of the bare dir")
    args = ap.parse_args(argv)

    try:
        rec = resolve(args.session, args.home, dry_run=args.dry_run,
                      probe_owner=not args.no_probe, cwd=args.cwd)
    except Exception as exc:  # never crash the launcher -- it falls back on rc!=0
        print(f"[resume-resolver] internal error: {exc}", file=sys.stderr)
        return 2

    if not rec.get("ok"):
        print(f"[resume-resolver] {rec.get('reason')}", file=sys.stderr)
        if args.json:
            print(json.dumps(rec, indent=1))
        return 1

    print(f"[resume-resolver] {rec['action']}: {rec['reason']}", file=sys.stderr)
    if rec.get("dup_count", 1) > 1:
        print(f"[resume-resolver] session in {rec['dup_count']} accounts "
              f"({', '.join(rec['all_accounts'])})", file=sys.stderr)

    if args.json:
        print(json.dumps(rec, indent=1))
    else:
        print(rec["pin_config_dir"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
