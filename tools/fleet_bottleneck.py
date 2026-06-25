#!/usr/bin/env python3
r"""fleet_bottleneck.py — the fleet's "what is limiting us right now?" engine + live dashboard.

This is v1 of fleet-wide bottleneck detection and visibility for the autonomous
Claude Code worker fleet. It is a direct adaptation of three concepts proven in
the DGX-caching observability stack (C:\work\metrics-service):

  1. Bottleneck Master List  (observability/bottleneck-master-list.md)
       A *ranked* inventory of every bottleneck, scored by how likely it is to be
       the ACTUAL limiter, each with severity + symptom + fix + a decision tree.
       Here: 10 fleet-bottleneck classes (the "Top 10", mirroring the source's
       at-a-glance), scored live from on-disk telemetry, with the single most
       important one surfaced as the headline.

  2. Universal Observability Hub  (design/universal-observability-hub-implementation.md)
       Normalize *heterogeneous* sources into ONE common schema and fan out to
       sinks. The fleet's signals are fragmented across separate files/tools;
       collect() normalizes them into a single FleetSnapshot, and serve() fans it
       out to an HTML dashboard + /health + /latest + /bottlenecks JSON.

  3. Embedded zero-dependency web UI  (internal/ui)
       GitHub-dark health card, ranked tables, auto-refresh — pure stdlib
       http.server, no pip installs, runs the same on Windows/mac/Linux.

  4. Prometheus sink + Grafana dashboards  (internal/sink/prometheus, grafana/)
       The machine/agent time-series surface: render_prometheus() exposes every
       signal as Prometheus metrics (namespace fleet_) at /metrics and in the
       _registry/fleet_bottleneck.prom textfile; tools/grafana/ provisions a
       Grafana dashboard + alert rules over them.

Telemetry sources (all already produced by the existing fleet tooling — we do not
re-parse the JSONL classifier, we consume its output):

  - tools/_registry/sessions.json      <- fleet_sessions.py  (disposition, category,
                                          throttle, account, age, autonomy)
  - session_audit.py (imported)        <- EXACT token accounting (cost, cache-hit,
                                          I:O, top spenders) from the transcripts
  - tools/_registry/resume_ledger.jsonl, transitions.log, _watchdog/  <- recovery freshness

Usage:
  python fleet_bottleneck.py report                 # ranked bottlenecks + health card (text)
  python fleet_bottleneck.py json [--out FILE]      # full machine payload
  python fleet_bottleneck.py prometheus [--out FILE]# Prometheus text exposition (machine/agent)
  python fleet_bottleneck.py serve [--port 9095]    # live web dashboard + JSON API + /metrics
      [--no-audit]            skip the (slow) token-spend pass; registry signals only
      [--audit-days 1.5]      how far back the token audit looks
      [--audit-max 80]        cap transcripts analyzed per refresh
      [--interval 45]         background snapshot refresh seconds (serve mode)

The registry signals are cheap and always computed; the token-spend pass reads
full transcripts, so it is bounded and (in serve mode) refreshed on an interval —
the HTTP handlers always serve the latest cached snapshot (the memory-sink pattern).
"""
import os
import sys
import json
import argparse
import datetime as dt
import threading
import time
import html
import math

HERE = os.path.dirname(os.path.abspath(__file__))
FLEET_DIR = os.path.dirname(HERE)
REG_DIR = os.path.join(HERE, "_registry")
WATCH_DIR = os.path.join(HERE, "_watchdog")

# session_audit lives next to us; import it for EXACT token accounting (no re-parse).
sys.path.insert(0, HERE)
try:
    import fleet_version
except Exception:  # keep serving even if the version helper is unavailable (mirrors session_audit)
    class fleet_version:  # type: ignore  # minimal fallback matching app_version()'s real contract
        @staticmethod
        def app_version(*a, **k):
            return os.environ.get("FAK_APP_VERSION", "").strip() or "dev"
try:
    import session_audit  # type: ignore
except Exception:  # pragma: no cover - audit is optional, registry signals still work
    session_audit = None

def NOW():
    return dt.datetime.now(dt.timezone.utc)

# --------------------------------------------------------------------------- #
# Tunable thresholds. Every bottleneck score is a transparent function of these
# so the ranking is auditable and adjustable without touching the logic.
# --------------------------------------------------------------------------- #
CFG = {
    "stale_warn_min":      15.0,   # registry older than this => "flying blind" starts to bite
    "watchdog_stale_min":  30.0,   # recovery plumbing considered stale past this
    "hang_age_warn_h":     2.0,    # a HANGING session older than this is wasted capacity
    "cache_hit_baseline":  0.80,   # per-session cache-hit median below this = token waste
    "io_baseline":         60.0,   # per-session I:O median above this = context churn
    # severity cutoffs on the 0..100 score
    "sev_critical": 80, "sev_high": 55, "sev_medium": 30, "sev_low": 10,
}


def _reset_of(info):
    """reset string from a throttle entry, tolerant of a non-dict (corrupt registry)."""
    return info.get("reset") if isinstance(info, dict) else info


def severity_of(score):
    if score >= CFG["sev_critical"]:
        return "CRITICAL"
    if score >= CFG["sev_high"]:
        return "HIGH"
    if score >= CFG["sev_medium"]:
        return "MEDIUM"
    if score >= CFG["sev_low"]:
        return "LOW"
    return "OK"


# --------------------------------------------------------------------------- #
# 1. COLLECT — normalize the fragmented fleet signals into one FleetSnapshot.
# --------------------------------------------------------------------------- #
def _load_registry():
    p = os.path.join(REG_DIR, "sessions.json")
    if not os.path.exists(p):
        return None
    try:
        return json.load(open(p, encoding="utf-8"))
    except Exception:
        return None


def _age_min(iso):
    if not iso:
        return None
    try:
        t = dt.datetime.fromisoformat(iso.replace("Z", "+00:00"))
        if t.tzinfo is None:
            t = t.replace(tzinfo=dt.timezone.utc)
        return (NOW() - t).total_seconds() / 60.0
    except Exception:
        return None


def _file_age_min(path):
    try:
        return (NOW().timestamp() - os.stat(path).st_mtime) / 60.0
    except OSError:
        return None


def _resumed_set():
    """session-ids already auto-resumed (so a dead session isn't double-counted as a backlog)."""
    out = set()
    p = os.path.join(REG_DIR, "resume_ledger.jsonl")
    if os.path.exists(p):
        for line in open(p, encoding="utf-8", errors="replace"):
            line = line.strip()
            if not line:
                continue
            try:
                out.add(json.loads(line).get("session"))
            except Exception:
                pass
    return out


def _token_audit(days, cap):
    """Bounded EXACT token-spend pass. Returns the session_audit aggregate + a few
    derived fleet figures, or None if the auditor isn't importable."""
    if session_audit is None:
        return None
    try:
        files = session_audit.discover(session_audit.DEFAULT_ROOTS, since_days=days, ns_prefix="")
        files = files[:cap]
        sess = [session_audit.analyze(f["path"]) for f in files]
        sess = [s for s in sess if "error" not in s]
        if not sess:
            return None
        agg = session_audit.aggregate(sess)
        # window cost-rate: total cost over the wall span of the analyzed sessions
        spans = [s["wall_s"] for s in sess if s.get("wall_s")]
        cost_per_hr = None
        if agg["total_cost_usd"] and spans:
            # approximate active hours = sum of session wall spans (parallel workers overlap,
            # so this is an upper bound on machine-hours; reported as "per active session-hour")
            active_h = sum(spans) / 3600.0
            cost_per_hr = agg["total_cost_usd"] / active_h if active_h else None
        # top spenders + worst cache-hit, for the dashboard
        top = sorted(sess, key=lambda s: -s["tokens"]["output"])[:6]
        worst_cache = sorted(
            [s for s in sess if s.get("cache_hit_frac") is not None and s["n_tool_use"] >= 5],
            key=lambda s: s["cache_hit_frac"])[:6]

        def _row(s):
            ns = os.path.basename(os.path.dirname(s["path"]))
            return {"session": s["session"][:8], "ns": ns,
                    "turns": s["assistant_turns"], "tool_calls": s["n_tool_use"],
                    "output": s["tokens"]["output"], "io": s.get("io_ratio"),
                    "cache_hit": s.get("cache_hit_frac"), "cost": round(s["cost_usd"], 2)}
        return {
            "n_analyzed": len(sess),
            "days": days,
            "totals": agg["totals"], "total_cost_usd": round(agg["total_cost_usd"], 2),
            "cost_per_active_session_hr": round(cost_per_hr, 2) if cost_per_hr else None,
            "dist": agg["dist"],
            "tool_mix": dict(list(agg["tool_mix"].items())[:12]),
            "top_spenders": [_row(s) for s in top],
            "worst_cache": [_row(s) for s in worst_cache],
        }
    except Exception as e:
        return {"error": str(e)}


def collect(audit=True, audit_days=1.5, audit_max=80):
    """Build the unified FleetSnapshot from every available source."""
    snap = {
        "app_version": fleet_version.app_version(),
        "generated_utc": NOW().isoformat(timespec="seconds"),
        "fleet_dir": FLEET_DIR,
        "registry": None, "audit": None,
        "errors": [],
    }
    reg = _load_registry()
    if reg is None:
        snap["errors"].append("registry sessions.json not found — run: python tools/fleet_sessions.py registry")
        return snap

    sessions = reg.get("sessions", [])
    # normalize the throttle map so every consumer can safely call info.get("reset")
    # even on a hand-edited / schema-drifted registry (defense for a corrupt file).
    throttle = {k: (v if isinstance(v, dict) else {"reset": v})
                for k, v in (reg.get("throttle", {}) or {}).items()}
    accounts = sorted({s.get("account") for s in sessions if s.get("account")})
    resumed = _resumed_set()

    # roll up dispositions / categories / actions
    from collections import Counter, defaultdict
    cat = Counter(s.get("category") for s in sessions)
    disp = Counter(s.get("disp") for s in sessions)
    act = Counter(s.get("action") for s in sessions)

    # A worker slot is "in-flight" unless it has cleanly finished. DONE / USER_CLOSED
    # are terminal; everything else (live, hanging, blocked, crashed) is in-flight.
    TERMINAL = ("DONE", "USER_CLOSED")
    throttled_accts = set(throttle.keys())
    per_acct = defaultdict(lambda: Counter())
    for s in sessions:
        a = s.get("account")
        per_acct[a]["total"] += 1
        if s.get("disp") not in TERMINAL:
            per_acct[a]["active"] += 1

    def _of(pred):
        return [s for s in sessions if pred(s)]

    hanging = _of(lambda s: s.get("category") == "HANGING")
    dead = _of(lambda s: s.get("disp") in ("DEAD_MIDTOOL", "DEAD_KILLED"))
    live = _of(lambda s: s.get("disp") == "LIVE")
    done = _of(lambda s: s.get("disp") in TERMINAL)
    active = _of(lambda s: s.get("disp") not in TERMINAL)
    supervised = _of(lambda s: s.get("action") == "SUPERVISED")
    # The `action` field is the registry's authority on what each session NEEDS.
    # The real resume backlog is what was decided AUTO_RESUME but isn't done yet —
    # NOT every crashed session (supervised / skipped ones are already accounted for).
    auto_resume = [s for s in _of(lambda s: s.get("action") == "AUTO_RESUME")
                   if s.get("session") not in resumed]
    # Count auth pressure by the producer's DECISION, not raw disp: an INFRA_AUTH
    # session whose account is throttled/supervised/cwd-gone is deliberately decided
    # DEFER_THROTTLED/SUPERVISED/SKIP_EPHEMERAL, so the live blocker is that — not a
    # re-login. Keeps the "needs attention" buckets disjoint and action-driven.
    auth = _of(lambda s: s.get("action") == "BLOCKED_AUTH")
    deferred = _of(lambda s: s.get("action") == "DEFER_THROTTLED")
    limited = _of(lambda s: s.get("disp") == "STOPPED_LIMIT")
    # SURFACE = a crashed/stopped INTERACTIVE worker the classifier flagged for a
    # human (a person owns it, so it is deliberately NOT auto-resumed). Distinct
    # from AUTO_RESUME (autonomous crashes the watchdog relaunches).
    surface = _of(lambda s: s.get("action") == "SURFACE")
    # API-error stalls counted by DISPOSITION (an upstream-health lens): how many
    # workers halted on a transient API/transport error, regardless of whether the
    # action router then sent them to AUTO_RESUME (autonomous) or SURFACE (interactive).
    api_error = _of(lambda s: s.get("disp") == "STOPPED_APIERR")
    # fraction of in-flight workers sitting on a rate-limited account (the real hit)
    workers_on_throttled = sum(1 for s in active if s.get("account") in throttled_accts)

    snap["registry"] = {
        "generated_utc": reg.get("generated_utc"),
        "age_min": _age_min(reg.get("generated_utc")),
        "window_h": reg.get("window_h"),
        "n_sessions": len(sessions),
        "n_accounts": len(accounts),
        "accounts": accounts,
        "category": dict(cat), "disposition": dict(disp), "action": dict(act),
        "throttle": throttle, "n_throttled": len(throttle),
        "counts": {
            "live": len(live), "done": len(done), "active": len(active),
            "hanging": len(hanging), "dead": len(dead),
            "auto_resume": len(auto_resume), "supervised": len(supervised),
            "auth_blocked": len(auth), "deferred_throttle": len(deferred),
            "rate_limited": len(limited),
            "workers_on_throttled": workers_on_throttled,
            "surface": len(surface), "api_error": len(api_error),
        },
        "per_account": {a: dict(c) for a, c in per_acct.items()},
        # keep the live session list (trimmed) for the dashboard table
        "sessions": [{
            "account": s.get("account"), "project": s.get("project"),
            "session": (s.get("session") or "")[:8], "git": s.get("git"),
            "category": s.get("category"), "disp": s.get("disp"),
            "cause": s.get("cause"), "action": s.get("action"),
            "age_min": s.get("age_min"), "autonomous": s.get("autonomous"),
            "supervised": s.get("supervised"), "throttle_reset": s.get("throttle_reset"),
        } for s in sessions],
        # hanging detail (the wasted-capacity offenders), oldest first
        "hanging_detail": sorted(
            [{"account": s.get("account"), "project": s.get("project"),
              "session": (s.get("session") or "")[:8], "disp": s.get("disp"),
              "age_min": s.get("age_min")} for s in hanging],
            key=lambda x: -(x.get("age_min") or 0))[:8],
    }

    # recovery-plumbing freshness (best-effort, on-disk only — no cross-repo subprocess)
    wlog = os.path.join(WATCH_DIR, "watchdog.log")
    snap["recovery"] = {
        "watchdog_log_age_min": _file_age_min(wlog),
        "resumed_ever": len(resumed),
        "transitions_log_present": os.path.exists(os.path.join(REG_DIR, "transitions.log")),
    }

    if audit:
        snap["audit"] = _token_audit(audit_days, audit_max)
    return snap


# --------------------------------------------------------------------------- #
# 2. BOTTLENECK MASTER LIST — score each class live; rank by likely-actual-limiter.
#    Each scorer returns a dict or None. score in [0,100]; severity derives from it.
# --------------------------------------------------------------------------- #
def _b(id, title, layer, score, symptom, fix, evidence):
    score = max(0.0, min(100.0, score))
    return {"id": id, "title": title, "layer": layer, "score": round(score, 1),
            "severity": severity_of(score), "symptom": symptom, "fix": fix,
            "evidence": evidence}


def _bottlenecks(snap):
    r = snap.get("registry")
    if not r:
        return []
    c = r["counts"]
    n_acct = max(r["n_accounts"], 1)
    n_active = max(c["active"], 1)
    out = []

    # #1 Rate-limit / throttle saturation — the fleet's external hard ceiling.
    #    What binds throughput is the fraction of IN-FLIGHT WORKERS sitting on a
    #    rate-limited account (not merely the account count) — a throttled account
    #    carrying most of the fleet hurts far more than an idle one. (cf. NIC #1)
    frac_acct = r["n_throttled"] / n_acct
    frac_workers = c["workers_on_throttled"] / n_active
    score = 100 * frac_workers + 6 * c["rate_limited"] + 8 * frac_acct
    if r["n_throttled"] or c["rate_limited"]:
        out.append(_b(
            "throttle_saturation", "Rate-limit / throttle saturation", "Account ceiling",
            score,
            f"{c['workers_on_throttled']}/{n_active} in-flight workers are on a rate-limited "
            f"account ({r['n_throttled']}/{n_acct} accounts throttled, {c['rate_limited']} stopped on a limit).",
            "Spread work across more accounts or wait for reset; pause dispatch to throttled accounts. "
            "This is an external ceiling — adding workers on a throttled account does not help.",
            {"throttled_accounts": list(r["throttle"].keys()),
             "frac_accounts_throttled": round(frac_acct, 3),
             "frac_workers_throttled": round(frac_workers, 3),
             "throttle_resets": {k: (v.get("reset") if isinstance(v, dict) else v)
                                 for k, v in r["throttle"].items()}}))

    # #2 Auth-blocked workers — a hard stop that needs a human; blocks the account.
    if c["auth_blocked"]:
        score = 55 + 15 * c["auth_blocked"]
        out.append(_b(
            "auth_blocked", "Auth-blocked workers (need re-login)", "Account ceiling",
            score,
            f"{c['auth_blocked']} session(s) blocked on authentication (expired token / /login).",
            "Re-authenticate the owning account (run /login under its CLAUDE_CONFIG_DIR); "
            "until then that account's workers cannot run.",
            {"count": c["auth_blocked"]}))

    # #3 Stalled / hanging capacity — slots held by workers making no progress.
    frac_hang = c["hanging"] / n_active
    ages_h = [(x.get("age_min") or 0) / 60.0 for x in r["hanging_detail"]]
    avg_age_h = sum(ages_h) / len(ages_h) if ages_h else 0.0
    if c["hanging"]:
        score = 80 * frac_hang + min(20, avg_age_h / CFG["hang_age_warn_h"] * 12)
        out.append(_b(
            "stalled_capacity", "Stalled / hanging capacity", "Worker",
            score,
            f"{c['hanging']} of {n_active} active workers are parked/ambiguous "
            f"(avg age {avg_age_h:.1f}h) — holding a slot without progress.",
            "Inspect the parked workers: resume the ones waiting on a finished task, "
            "close the genuinely-stuck ones. Tighten the supervisor's parked-vs-stuck timeout.",
            {"hanging": c["hanging"], "active": c["active"],
             "frac_active_hanging": round(frac_hang, 3),
             "oldest": r["hanging_detail"][:3]}))

    # #4 Crash-resume backlog — workers the registry DECIDED to resume (action
    #    AUTO_RESUME) that haven't been resumed yet. Supervised/skipped crashes are
    #    handled elsewhere, so they do NOT count here.
    wd_age = snap.get("recovery", {}).get("watchdog_log_age_min")
    if c["auto_resume"]:
        stale_wd = wd_age is None or wd_age > CFG["watchdog_stale_min"]
        score = 22 * c["auto_resume"] + (20 if stale_wd else 0)
        out.append(_b(
            "crash_resume_backlog", "Crash-resume backlog", "Recovery",
            score,
            f"{c['auto_resume']} worker(s) queued for resume (action=AUTO_RESUME) not yet recovered"
            + (f"; resume watchdog last ran {wd_age:.0f}m ago." if wd_age is not None
               else "; resume-watchdog activity not seen on disk."),
            "Confirm the resume watchdog is scheduled and in LIVE mode (DRY-RUN never resumes); "
            "process resume_plan.json. Each queued worker is lost throughput until resumed.",
            {"auto_resume": c["auto_resume"], "watchdog_log_age_min":
             round(wd_age, 1) if wd_age is not None else None}))

    # #5 Recovery plumbing stale — if anything needs recovery but the watchdog is
    #    stale/absent, recovery itself may have stopped (the meta-bottleneck).
    reg_age = r.get("age_min")
    needs_recovery = c["auto_resume"] + c["hanging"] + c["deferred_throttle"]
    if needs_recovery and (wd_age is None or wd_age > CFG["watchdog_stale_min"]):
        score = 45 + (min(35, (wd_age - CFG["watchdog_stale_min"]) / 10) if wd_age else 35)
        out.append(_b(
            "recovery_stale", "Recovery plumbing stale", "Control plane",
            score,
            (f"Recovery watchdog last ran {wd_age:.0f}m ago" if wd_age is not None
             else "No recovery-watchdog activity found on disk")
            + f", but {needs_recovery} worker(s) need attention.",
            "Verify FleetSupervisorWatchdog / FleetResumeWatchdog scheduled tasks are running; "
            "the supervisor (PID-1) must outlive the launcher. If recovery is down, nothing self-heals.",
            {"watchdog_log_age_min": round(wd_age, 1) if wd_age is not None else None,
             "needs_recovery": needs_recovery}))

    # #6 Token-spend inefficiency — workers burning tokens (low cache reuse / high churn).
    a = snap.get("audit")
    if a and "error" not in a and a.get("dist"):
        chf = a["dist"]["cache_hit_frac"].get("median")
        io = a["dist"]["io_ratio"].get("median")
        score = 0.0
        bits = []
        if chf is not None and chf < CFG["cache_hit_baseline"]:
            score += 130 * (CFG["cache_hit_baseline"] - chf)
            bits.append(f"median cache-hit {chf*100:.0f}% (baseline {CFG['cache_hit_baseline']*100:.0f}%)")
        if io is not None and io > CFG["io_baseline"]:
            score += min(25, (io - CFG["io_baseline"]) / 10)
            bits.append(f"median I:O {io:.0f}:1 (baseline {CFG['io_baseline']:.0f}:1)")
        score = min(score, 70)  # cost is MEDIUM by design — it should rarely outrank an outage
        if bits:
            out.append(_b(
                "token_inefficiency", "Token-spend inefficiency", "Cost",
                score,
                "Workers are spending more than needed: " + "; ".join(bits) + ".",
                "Reduce redundant re-reads (raise prompt-cache reuse), trim context churn, "
                "and check the worst-cache sessions below for read-loops / glob-storms.",
                {"median_cache_hit": chf, "median_io": io,
                 "total_cost_usd": a.get("total_cost_usd"),
                 "cost_per_active_session_hr": a.get("cost_per_active_session_hr")}))

    # #7 Account load imbalance — work concentrated on few accounts while others idle.
    if r["n_accounts"] > 1:
        actives = {acc: v.get("active", 0) for acc, v in r["per_account"].items()}
        tot = sum(actives.values())
        if tot:
            max_share = max(actives.values()) / tot
            even = 1.0 / r["n_accounts"]
            score = min(70, 130 * max(0.0, max_share - even))
            if score >= CFG["sev_low"]:
                hot = max(actives, key=actives.get)
                out.append(_b(
                    "account_imbalance", "Account load imbalance", "Dispatch",
                    score,
                    f"{max_share*100:.0f}% of active workers are on one account ({hot}); "
                    f"even split would be {even*100:.0f}% across {r['n_accounts']} accounts.",
                    "Rebalance dispatch across accounts — concentration wastes idle accounts' "
                    "rate-limit headroom and makes one throttle event hurt more.",
                    {"active_per_account": actives, "max_share": round(max_share, 3)}))

    # #9 Dead-crash surfacing backlog — interactive crashes the classifier flagged
    #    for a human (action=SURFACE). A person owns these, so they are deliberately
    #    NOT auto-resumed — they sit as lost work until someone triages them. Distinct
    #    from #4 (autonomous crashes the watchdog relaunches). Capped at HIGH.
    if c.get("surface"):
        score = min(60, 10 * c["surface"])
        out.append(_b(
            "surface_backlog", "Dead-crash surfacing backlog", "Recovery",
            score,
            f"{c['surface']} crashed/stopped interactive worker(s) flagged for human review "
            f"(action=SURFACE) — not auto-resumable, lost until someone triages them.",
            "List them with `python tools/fleet_sessions.py resume` (the SURFACE block); resume the "
            "ones you meant to keep, close the rest. Autonomous crashes auto-resume — these are "
            "interactive, so a human must decide.",
            {"surface": c["surface"]}))

    # #10 API-error stalls — workers halted on a transient upstream API/transport
    #     error (5xx / overloaded / network blip), counted by disposition. A different
    #     lens than the recovery buckets above (those route by action): a spike here
    #     means the upstream API itself is degraded, not the fleet. Capped at HIGH.
    if c.get("api_error"):
        score = min(60, 12 * c["api_error"])
        out.append(_b(
            "api_error_stalls", "API-error stalls", "Provider",
            score,
            f"{c['api_error']} worker(s) stopped on a transient API/transport error "
            f"(disp=STOPPED_APIERR) — upstream overload / 5xx / network blip.",
            "Usually self-clears on retry (autonomous ones are queued AUTO_RESUME, interactive ones "
            "SURFACE). If the count is climbing, the upstream API is degraded — check provider status "
            "before adding load.",
            {"api_error": c["api_error"]}))

    # #8 Stale telemetry — the registry itself is old, so you're flying blind.
    if reg_age is not None and reg_age > CFG["stale_warn_min"]:
        score = min(80, (reg_age - CFG["stale_warn_min"]) * 2)
        out.append(_b(
            "stale_telemetry", "Stale fleet telemetry", "Observability",
            score,
            f"Session registry is {reg_age:.0f} min old — fleet state may have changed since.",
            "Re-run `python tools/fleet_sessions.py registry` (or schedule it) so the "
            "classification reflects the live fleet. Stale signals hide real bottlenecks.",
            {"registry_age_min": round(reg_age, 1)}))

    out.sort(key=lambda x: -x["score"])
    return out


def rank_bottlenecks(snap):
    bl = _bottlenecks(snap)
    headline = bl[0] if bl and bl[0]["score"] >= CFG["sev_low"] else None
    # overall fleet health from the worst score (+ telemetry presence)
    worst = bl[0]["score"] if bl else 0
    if snap.get("registry") is None:
        health = "down"
    elif worst >= CFG["sev_critical"]:
        health = "critical"
    elif worst >= CFG["sev_high"]:
        health = "degraded"
    else:
        health = "ok"
    return {"health": health, "headline": headline, "bottlenecks": bl}


# --------------------------------------------------------------------------- #
# 3a. TEXT REPORT (console + _registry/BOTTLENECKS.txt)
# --------------------------------------------------------------------------- #
def _bar(score, width=20):
    n = int(round(score / 100 * width))
    return "█" * n + "·" * (width - n)


def report_text(snap, ranked):
    L = []
    now = NOW().strftime("%Y-%m-%d %H:%M")
    L.append(f"==================== FLEET BOTTLENECK + VISIBILITY @ {now}Z ====================")
    r = snap.get("registry")
    if not r:
        L.append("registry: NOT FOUND — run `python tools/fleet_sessions.py registry` first.")
        for e in snap.get("errors", []):
            L.append("  ! " + e)
        L.append("=" * 78)
        return "\n".join(L)

    c = r["counts"]
    age_s = f"{r['age_min']:.0f}m old" if r.get("age_min") is not None else "age unknown"
    win_s = f"{r['window_h']}h window" if r.get("window_h") is not None else "window unknown"
    L.append(f"health: {ranked['health'].upper()}   "
             f"sessions={r['n_sessions']}  accounts={r['n_accounts']}  "
             f"(registry {age_s}, {win_s})")
    L.append("category: " + "  ".join(f"{k}={v}" for k, v in
             sorted(r["category"].items(), key=lambda kv: -kv[1])))
    L.append(f"slots: live={c['live']}  supervised={c['supervised']}  active={c['active']}  "
             f"hanging={c['hanging']}  auto_resume={c['auto_resume']}  surface={c.get('surface', 0)}  "
             f"auth_blocked={c['auth_blocked']}  api_error={c.get('api_error', 0)}  "
             f"deferred_throttle={c['deferred_throttle']}  done={c['done']}")
    if r["throttle"]:
        L.append("throttled accounts:")
        for acc, info in r["throttle"].items():
            L.append(f"  {acc}  resets {_reset_of(info)}")

    L.append("")
    if ranked["headline"]:
        h = ranked["headline"]
        L.append(f">>> #1 BOTTLENECK: {h['title']}  [{h['severity']}]")
        L.append(f"    {h['symptom']}")
        L.append(f"    FIX: {h['fix']}")
    else:
        L.append(">>> No dominant bottleneck — fleet is healthy.")
    L.append("")

    L.append("RANKED BOTTLENECKS (by likelihood of being the actual limiter):")
    if not ranked["bottlenecks"]:
        L.append("  (none scored — fleet healthy)")
    for i, b in enumerate(ranked["bottlenecks"], 1):
        L.append(f"  {i}. [{b['severity']:8}] {_bar(b['score'])} {b['score']:5.1f}  "
                 f"{b['title']}  ({b['layer']})")
        L.append(f"       {b['symptom']}")

    a = snap.get("audit")
    if a and "error" not in a:
        L.append("")
        L.append(f"token spend (last {a.get('days')}d, {a.get('n_analyzed')} transcripts, EXACT):")
        d = a["dist"]
        L.append(f"  cost=${a['total_cost_usd']:,} (assumed pricing)   "
                 f"median cache-hit={(d['cache_hit_frac'].get('median') or 0)*100:.0f}%   "
                 f"median I:O={d['io_ratio'].get('median')}:1")
        if a.get("top_spenders"):
            L.append("  top spenders (output tok):  " + "  ".join(
                f"{t['session']}={t['output']:,}" for t in a["top_spenders"][:4]))
    elif a and "error" in a:
        L.append(f"\ntoken spend: (audit error: {a['error']})")
    L.append("=" * 78)
    return "\n".join(L)


def write_artifacts(snap, ranked):
    os.makedirs(REG_DIR, exist_ok=True)
    txt = report_text(snap, ranked)
    with open(os.path.join(REG_DIR, "BOTTLENECKS.txt"), "w", encoding="utf-8") as f:
        f.write(txt)
    payload = {"generated_utc": snap["generated_utc"], **ranked, "snapshot": snap}
    with open(os.path.join(REG_DIR, "bottlenecks.json"), "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, default=str)
    # Prometheus textfile artifact (for a node_exporter textfile collector or a
    # cron scrape) — best-effort; never let it break the text/JSON artifacts.
    try:
        with open(os.path.join(REG_DIR, "fleet_bottleneck.prom"), "w", encoding="utf-8") as f:
            f.write(render_prometheus(snap, ranked))
    except Exception:
        pass
    return txt


# --------------------------------------------------------------------------- #
# 3a'. PROMETHEUS EXPOSITION (text format 0.0.4) — the machine/agent time-series
#      surface, the fleet's analog of the metrics-service prometheus sink. One
#      HELP/TYPE per family, label-escaped, bounded cardinality (per-bottleneck
#      ≤10; per-account ≤ n_accounts; leaderboards capped by the engine ≤6). Served
#      live at /metrics and written to _registry/fleet_bottleneck.prom each refresh.
# --------------------------------------------------------------------------- #
PROM_NS = "fleet"
HEALTH_STATE = {"ok": 0, "degraded": 1, "critical": 2, "down": 3}
SEV_NUM = {"OK": 0, "LOW": 1, "MEDIUM": 2, "HIGH": 3, "CRITICAL": 4}


def _iso_to_epoch(iso):
    if not iso:
        return None
    try:
        t = dt.datetime.fromisoformat(iso.replace("Z", "+00:00"))
        if t.tzinfo is None:
            t = t.replace(tzinfo=dt.timezone.utc)
        return t.timestamp()
    except Exception:
        return None


def _prom_esc(v):
    """Escape a Prometheus label value (backslash, double-quote, newline)."""
    return str(v).replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")


def _prom_labels(d):
    if not d:
        return ""
    return "{" + ",".join(f'{k}="{_prom_esc(v)}"' for k, v in d.items()) + "}"


def _fmt_prom(v):
    if isinstance(v, bool):
        return "1" if v else "0"
    if isinstance(v, int):
        return str(v)
    f = float(v)
    if not math.isfinite(f):  # defensive: never crash int(); fam() already drops these
        return "NaN" if math.isnan(f) else ("+Inf" if f > 0 else "-Inf")
    return str(int(f)) if f == int(f) else repr(f)


def render_prometheus(snap, ranked):
    """Render the FleetSnapshot + ranking as Prometheus text exposition (0.0.4)."""
    fams = []  # (name, type, help, [(labels_dict_or_None, value)])

    def fam(name, mtype, help_text, series):
        series = [(lbl, val) for lbl, val in series  # drop None / non-finite -> never emit NaN/Inf
                  if val is not None and not (isinstance(val, float) and not math.isfinite(val))]
        fams.append((f"{PROM_NS}_{name}", mtype, help_text, series))

    # scrape liveness + freshness + health (always emitted, even with no registry)
    fam("up", "gauge", "1 if the bottleneck engine produced a snapshot this scrape.", [(None, 1)])
    fam("snapshot_timestamp_seconds", "gauge",
        "Unix time the snapshot was generated (use time()-this for staleness).",
        [(None, _iso_to_epoch(snap.get("generated_utc")))])
    fam("health_state", "gauge", "Fleet health: 0 ok, 1 degraded, 2 critical, 3 down.",
        [(None, HEALTH_STATE.get(ranked.get("health"), 3))])

    # ranked bottleneck scores (the headline family). Labels are STABLE identity only
    # (id/title/layer are 1:1 with the class) — severity is value-derived, so it is
    # emitted as a SEPARATE numeric gauge below rather than a label, or the score series
    # would split into a new time series every time it crosses a severity threshold.
    fam("bottleneck_score", "gauge",
        "Per-class bottleneck score 0-100 (higher = more likely the actual limiter).",
        [({"id": b["id"], "title": b["title"], "layer": b["layer"]},
          b["score"]) for b in ranked.get("bottlenecks", [])])
    fam("bottleneck_severity", "gauge",
        "Per-class severity as a number (0 OK, 1 LOW, 2 MEDIUM, 3 HIGH, 4 CRITICAL), "
        "derived from the score — kept OFF bottleneck_score's labels to avoid series churn.",
        [({"id": b["id"], "title": b["title"], "layer": b["layer"]},
          SEV_NUM.get(b["severity"], 0)) for b in ranked.get("bottlenecks", [])])
    h = ranked.get("headline")
    fam("headline_score", "gauge", "Score of the #1 bottleneck right now (0 if none dominant).",
        [(None, h["score"] if h else 0)])
    fam("bottlenecks", "gauge", "Number of bottleneck classes currently scored (gauge, not cumulative).",
        [(None, len(ranked.get("bottlenecks", [])))])

    r = snap.get("registry")
    if r:
        c = r.get("counts", {})
        fam("sessions", "gauge", "Sessions seen in the registry window (gauge — windowed, not cumulative).", [(None, r.get("n_sessions"))])
        fam("accounts", "gauge", "Distinct accounts in the registry.", [(None, r.get("n_accounts"))])
        fam("accounts_throttled", "gauge", "Accounts currently rate-limited.", [(None, r.get("n_throttled"))])
        for mname, ckey, help_text in [
            ("workers_active", "active", "Workers in-flight (not cleanly finished)."),
            ("workers_live", "live", "Workers appended within the live window."),
            ("workers_done", "done", "Workers terminal (DONE / USER_CLOSED)."),
            ("workers_hanging", "hanging", "Workers parked/ambiguous, holding a slot without progress."),
            ("workers_auth_blocked", "auth_blocked", "Workers blocked on auth (need /login)."),
            ("workers_on_throttled_account", "workers_on_throttled", "In-flight workers on a rate-limited account."),
            ("workers_rate_limited", "rate_limited", "Workers stopped on a rate limit."),
            ("workers_supervised", "supervised", "Workers handled by the supervisor."),
            ("workers_deferred_throttle", "deferred_throttle", "Workers dispatch-deferred for throttle."),
            ("resume_queue", "auto_resume", "Workers queued for auto-resume, not yet recovered."),
            ("surface_backlog", "surface", "Interactive crashes flagged for human triage."),
            ("api_error_stalls", "api_error", "Workers stopped on a transient API/transport error."),
        ]:
            fam(mname, "gauge", help_text, [(None, c.get(ckey))])
        fam("registry_age_minutes", "gauge",
            "Age of the session registry in minutes (staleness of all signals).", [(None, r.get("age_min"))])
        fam("workers_active_per_account", "gauge",
            "In-flight workers per account (concentration drives imbalance #7).",
            [({"account": acc}, v.get("active", 0)) for acc, v in sorted(r.get("per_account", {}).items())])
        fam("account_throttled", "gauge", "1 for each currently rate-limited account.",
            [({"account": acc}, 1) for acc in sorted((r.get("throttle") or {}).keys())])

    fam("watchdog_log_age_minutes", "gauge",
        "Age of the recovery watchdog log in minutes (recovery-plumbing freshness).",
        [(None, (snap.get("recovery") or {}).get("watchdog_log_age_min"))])

    a = snap.get("audit")
    if a and "error" not in a:
        d = a.get("dist") or {}
        fam("cost_usd", "gauge", "Estimated token spend over the audit window (USD, assumed pricing).",
            [(None, a.get("total_cost_usd"))])
        fam("cost_per_active_session_hour", "gauge", "Estimated cost per active session-hour (USD).",
            [(None, a.get("cost_per_active_session_hr"))])
        fam("cache_hit_ratio_median", "gauge", "Median per-session prompt-cache-hit fraction (0-1).",
            [(None, (d.get("cache_hit_frac") or {}).get("median"))])
        fam("io_ratio_median", "gauge", "Median per-session input:output token ratio.",
            [(None, (d.get("io_ratio") or {}).get("median"))])
        fam("audit_sessions", "gauge", "Transcripts analyzed in the token-spend pass.",
            [(None, a.get("n_analyzed"))])
        fam("top_spender_output_tokens", "gauge",
            "Output tokens for the top-spending sessions (leaderboard, capped).",
            [({"session": t["session"], "ns": t["ns"]}, t["output"]) for t in a.get("top_spenders", [])])
        fam("worst_cache_hit_ratio", "gauge",
            "Cache-hit fraction for the lowest-cache sessions (leaderboard, capped).",
            [({"session": t["session"], "ns": t["ns"]}, t["cache_hit"]) for t in a.get("worst_cache", [])])

    L = []
    for name, mtype, help_text, series in fams:
        if not series:
            continue
        L.append(f"# HELP {name} {help_text}")
        L.append(f"# TYPE {name} {mtype}")
        for lbl, val in series:
            L.append(f"{name}{_prom_labels(lbl)} {_fmt_prom(val)}")
    return "\n".join(L) + "\n"


# --------------------------------------------------------------------------- #
# 3b. EMBEDDED WEB DASHBOARD (zero-dependency; GitHub-dark, auto-refresh)
# --------------------------------------------------------------------------- #
SEV_COLOR = {"CRITICAL": "#f85149", "HIGH": "#d29922", "MEDIUM": "#58a6ff",
             "LOW": "#3fb950", "OK": "#3fb950"}
HEALTH_COLOR = {"down": "#f85149", "critical": "#f85149", "degraded": "#d29922", "ok": "#3fb950"}


def _esc(x):
    return html.escape(str(x)) if x is not None else "—"


def render_html(snap, ranked, interval=10):
    r = snap.get("registry") or {}
    c = r.get("counts", {})
    a = snap.get("audit")
    hc = HEALTH_COLOR.get(ranked["health"], "#8b949e")
    parts = []
    P = parts.append

    P(f"""<!doctype html><html><head><meta charset="utf-8">
<title>Fleet Bottleneck &amp; Visibility</title>
<meta name="viewport" content="width=device-width,initial-scale=1">
<style>
:root{{--bg:#0d1117;--bg2:#161b22;--bg3:#21262d;--bd:#30363d;--tx:#c9d1d9;--tx2:#8b949e;--blue:#58a6ff;--grn:#3fb950;--org:#d29922;--red:#f85149}}
*{{margin:0;padding:0;box-sizing:border-box}}
body{{font-family:-apple-system,Segoe UI,Roboto,sans-serif;background:var(--bg);color:var(--tx);font-size:15px;line-height:1.5}}
header{{display:flex;align-items:center;gap:12px;padding:14px 22px;background:var(--bg2);border-bottom:1px solid var(--bd)}}
header h1{{font-size:20px;font-weight:600}} .sub{{color:var(--tx2);font-size:13px}}
main{{max-width:1120px;margin:0 auto;padding:22px}}
.bar{{display:flex;align-items:center;gap:10px;color:var(--tx2);font-size:13px;margin-bottom:16px}}
.dot{{width:9px;height:9px;border-radius:50%;background:var(--grn);display:inline-block;animation:pulse 2s infinite}}
@keyframes pulse{{0%,100%{{opacity:1}}50%{{opacity:.4}}}}
.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:12px;margin-bottom:18px}}
.card{{background:var(--bg2);border:1px solid var(--bd);border-radius:8px;padding:14px}}
.card h2{{font-size:12px;font-weight:600;color:var(--tx2);text-transform:uppercase;letter-spacing:.04em;margin-bottom:8px}}
.card .v{{font-size:26px;font-weight:700}}
.headline{{border-left:5px solid {hc};background:var(--bg2);border-radius:8px;padding:16px 18px;margin-bottom:18px}}
.headline .k{{font-size:12px;color:var(--tx2);text-transform:uppercase;letter-spacing:.05em}}
.headline .t{{font-size:21px;font-weight:700;margin:3px 0 6px}}
.headline .fix{{color:var(--tx2);font-size:14px;margin-top:6px}}
.badge{{display:inline-block;padding:2px 9px;border-radius:11px;font-size:11px;font-weight:700;color:#0d1117}}
table{{width:100%;border-collapse:collapse;font-size:13px}}
th,td{{text-align:left;padding:7px 9px;border-bottom:1px solid var(--bd)}}
th{{color:var(--tx2);font-weight:600;font-size:11px;text-transform:uppercase;letter-spacing:.03em}}
td.r,th.r{{text-align:right;font-variant-numeric:tabular-nums}}
.mono{{font-family:ui-monospace,Consolas,monospace}}
.meter{{height:7px;background:var(--bg3);border-radius:4px;overflow:hidden;width:120px;display:inline-block;vertical-align:middle}}
.meter>i{{display:block;height:100%}}
.sec{{background:var(--bg2);border:1px solid var(--bd);border-radius:8px;padding:14px 16px;margin-bottom:18px}}
.sec h2{{font-size:15px;font-weight:600;margin-bottom:10px}}
.pill{{display:inline-block;padding:1px 8px;border-radius:10px;font-size:11px;font-weight:600;background:var(--bg3);color:var(--tx2)}}
.muted{{color:var(--tx2)}} a{{color:var(--blue);text-decoration:none}}
</style></head><body>
<header><h1>🛰 Fleet Bottleneck &amp; Visibility</h1>
<span class="sub">v1 · adapted from metrics-service · {_esc(snap['generated_utc'])}</span></header>
<main>""")

    P(f'<div class="bar"><span class="dot"></span> LIVE · auto-refresh {interval}s · '
      f'<a href="/health">/health</a> · <a href="/latest">/latest</a> · '
      f'<a href="/bottlenecks">/bottlenecks</a> · <a href="/metrics">/metrics</a>')
    rage = r.get("age_min")
    if rage is not None and rage > CFG["stale_warn_min"]:
        P(f' · <span style="color:var(--org)">⚠ telemetry {rage:.0f}m stale</span>')
    P("</div>")

    # health + slot cards
    P('<div class="grid">')
    P(f'<div class="card" style="border-color:{hc}"><h2>Health</h2>'
      f'<span class="v" style="color:{hc}">{ranked["health"].upper()}</span></div>')
    for label, key in [("Sessions", "n_sessions")]:
        P(f'<div class="card"><h2>{label}</h2><span class="v">{_esc(r.get(key))}</span></div>')
    P(f'<div class="card"><h2>Accounts</h2><span class="v">{_esc(r.get("n_accounts"))}</span>'
      f'<div class="muted" style="font-size:12px">{c.get("rate_limited",0)} limited · {c.get("auth_blocked",0)} auth</div></div>')
    P(f'<div class="card"><h2>Active</h2><span class="v">{_esc(c.get("active"))}</span>'
      f'<div class="muted" style="font-size:12px">{c.get("live",0)} live · {c.get("done",0)} done</div></div>')
    P(f'<div class="card"><h2>Hanging</h2><span class="v" style="color:{"var(--org)" if c.get("hanging") else "inherit"}">{_esc(c.get("hanging"))}</span>'
      f'<div class="muted" style="font-size:12px">stalled slots</div></div>')
    P(f'<div class="card"><h2>Resume queue</h2><span class="v" style="color:{"var(--red)" if c.get("auto_resume") else "inherit"}">{_esc(c.get("auto_resume"))}</span>'
      f'<div class="muted" style="font-size:12px">{c.get("supervised",0)} supervised</div></div>')
    if a and "error" not in a:
        P(f'<div class="card"><h2>Spend ({a.get("days")}d)</h2><span class="v">${_esc(a.get("total_cost_usd"))}</span>'
          f'<div class="muted" style="font-size:12px">assumed pricing</div></div>')
    P('</div>')

    # headline
    h = ranked["headline"]
    if h:
        col = SEV_COLOR.get(h["severity"], "#8b949e")
        P(f'<div class="headline"><span class="k">#1 Most important bottleneck right now</span>'
          f'<div class="t">{_esc(h["title"])} '
          f'<span class="badge" style="background:{col}">{h["severity"]}</span></div>'
          f'<div>{_esc(h["symptom"])}</div>'
          f'<div class="fix"><b>Fix:</b> {_esc(h["fix"])}</div></div>')
    else:
        P('<div class="headline"><span class="k">Status</span>'
          '<div class="t">No dominant bottleneck — fleet healthy ✓</div></div>')

    # ranked bottleneck table
    P('<div class="sec"><h2>Ranked bottlenecks</h2><table>'
      '<tr><th>#</th><th>Bottleneck</th><th>Layer</th><th>Severity</th>'
      '<th>Score</th><th>Symptom</th></tr>')
    if ranked["bottlenecks"]:
        for i, b in enumerate(ranked["bottlenecks"], 1):
            col = SEV_COLOR.get(b["severity"], "#8b949e")
            P(f'<tr><td class="r">{i}</td><td><b>{_esc(b["title"])}</b></td>'
              f'<td class="muted">{_esc(b["layer"])}</td>'
              f'<td><span class="badge" style="background:{col}">{b["severity"]}</span></td>'
              f'<td><span class="meter"><i style="width:{b["score"]}%;background:{col}"></i></span> '
              f'<span class="mono">{b["score"]}</span></td>'
              f'<td class="muted">{_esc(b["symptom"])}</td></tr>')
    else:
        P('<tr><td colspan="6" class="muted">No bottlenecks scored — fleet healthy.</td></tr>')
    P('</table></div>')

    # throttle / accounts
    if r.get("throttle"):
        P('<div class="sec"><h2>Rate-limited accounts</h2><table>'
          '<tr><th>Account</th><th>Resets</th></tr>')
        for acc, info in r["throttle"].items():
            P(f'<tr><td class="mono">{_esc(acc)}</td><td>{_esc(_reset_of(info))}</td></tr>')
        P('</table></div>')

    # hanging detail
    if r.get("hanging_detail"):
        P('<div class="sec"><h2>Stalled / hanging workers (oldest first)</h2><table>'
          '<tr><th>Account</th><th>Project</th><th>Session</th><th>Disp</th><th class="r">Age (h)</th></tr>')
        for x in r["hanging_detail"]:
            age = (x.get("age_min") or 0) / 60.0
            P(f'<tr><td class="mono">{_esc(x.get("account"))}</td><td>{_esc(x.get("project"))}</td>'
              f'<td class="mono">{_esc(x.get("session"))}</td><td>{_esc(x.get("disp"))}</td>'
              f'<td class="r">{age:.1f}</td></tr>')
        P('</table></div>')

    # token spend tables
    if a and "error" not in a:
        if a.get("top_spenders"):
            P('<div class="sec"><h2>Top spenders (output tokens, EXACT)</h2><table>'
              '<tr><th>Session</th><th>NS</th><th class="r">Turns</th><th class="r">Tools</th>'
              '<th class="r">Output</th><th class="r">I:O</th><th class="r">Cache-hit</th><th class="r">Est.$</th></tr>')
            for t in a["top_spenders"]:
                io = f"{t['io']:.0f}" if t.get("io") else "—"
                ch = f"{t['cache_hit']*100:.0f}%" if t.get("cache_hit") is not None else "—"
                P(f'<tr><td class="mono">{_esc(t["session"])}</td><td class="muted">{_esc(t["ns"])}</td>'
                  f'<td class="r">{_esc(t["turns"])}</td><td class="r">{_esc(t["tool_calls"])}</td>'
                  f'<td class="r">{t["output"]:,}</td><td class="r">{io}</td><td class="r">{ch}</td>'
                  f'<td class="r">${_esc(t["cost"])}</td></tr>')
            P('</table></div>')
        if a.get("worst_cache"):
            P('<div class="sec"><h2>Lowest cache-hit sessions (token-waste suspects)</h2><table>'
              '<tr><th>Session</th><th>NS</th><th class="r">Tools</th><th class="r">Cache-hit</th>'
              '<th class="r">I:O</th><th class="r">Est.$</th></tr>')
            for t in a["worst_cache"]:
                io = f"{t['io']:.0f}" if t.get("io") else "—"
                ch = f"{t['cache_hit']*100:.0f}%" if t.get("cache_hit") is not None else "—"
                P(f'<tr><td class="mono">{_esc(t["session"])}</td><td class="muted">{_esc(t["ns"])}</td>'
                  f'<td class="r">{_esc(t["tool_calls"])}</td><td class="r">{ch}</td>'
                  f'<td class="r">{io}</td><td class="r">${_esc(t["cost"])}</td></tr>')
            P('</table></div>')
    elif a and "error" in a:
        P(f'<div class="sec muted">token audit unavailable: {_esc(a["error"])}</div>')

    P('<p class="muted" style="font-size:12px">Sources: tools/_registry/sessions.json '
      '(fleet_sessions.py) + session_audit.py · scoring tunable in fleet_bottleneck.py CFG.</p>')
    P(f'<script>setTimeout(function(){{location.reload()}},{interval*1000});</script>')
    P('</main></body></html>')
    return "".join(parts)


# Background snapshot cache for serve mode (collector -> memory-sink pattern).
class _SnapStore:
    def __init__(self, audit, audit_days, audit_max, interval):
        self.audit, self.audit_days, self.audit_max = audit, audit_days, audit_max
        self.interval = max(5, int(interval))  # bound refresh cost; avoid sleep(0) spin / sleep(neg) ValueError
        self.lock = threading.Lock()
        self.snap = None
        self.ranked = None
        try:
            self.refresh()
        except Exception as e:
            # Match _loop's swallow-and-continue: a fragile first snapshot (e.g. a
            # corrupt registry) must not abort startup before the socket binds.
            snap = {"generated_utc": NOW().isoformat(timespec="seconds"),
                    "app_version": fleet_version.app_version(),
                    "fleet_dir": FLEET_DIR, "registry": None, "audit": None,
                    "errors": [f"initial snapshot failed: {e}"]}
            with self.lock:
                self.snap, self.ranked = snap, rank_bottlenecks(snap)
        t = threading.Thread(target=self._loop, daemon=True)
        t.start()

    def refresh(self):
        snap = collect(self.audit, self.audit_days, self.audit_max)
        ranked = rank_bottlenecks(snap)
        with self.lock:
            self.snap, self.ranked = snap, ranked
        try:
            write_artifacts(snap, ranked)
        except Exception:
            pass

    def _loop(self):
        while True:
            time.sleep(self.interval)
            try:
                self.refresh()
            except Exception:
                pass

    def get(self):
        with self.lock:
            return self.snap, self.ranked


def serve(port, audit, audit_days, audit_max, interval):
    import http.server
    store = _SnapStore(audit, audit_days, audit_max, interval)

    class H(http.server.BaseHTTPRequestHandler):
        def _send(self, code, body, ctype):
            data = body.encode("utf-8") if isinstance(body, str) else body
            try:
                self.send_response(code)
                self.send_header("Content-Type", ctype)
                self.send_header("Content-Length", str(len(data)))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(data)
            except (BrokenPipeError, ConnectionResetError):
                return  # client left mid-response (routine with auto-refresh) — not an error

        def do_GET(self):
            try:
                self._route()
            except (BrokenPipeError, ConnectionResetError):
                return
            except Exception:
                try:
                    self._send(500, json.dumps({"error": "internal"}), "application/json")
                except Exception:
                    pass

        def _route(self):
            snap, ranked = store.get()
            path = self.path.split("?")[0].rstrip("/") or "/"
            if path in ("/", "/ui"):
                self._send(200, render_html(snap, ranked, interval=min(15, max(5, interval//3))), "text/html; charset=utf-8")
            elif path == "/health":
                # The service is up if it answers; `health` is the FLEET's state
                # (ok/degraded/critical, or down only when telemetry is missing).
                h = ranked["headline"]
                self._send(200, json.dumps({
                    "status": "up", "health": ranked["health"],
                    "headline_bottleneck": h["title"] if h else None,
                    "headline_severity": h["severity"] if h else None,
                    "n_bottlenecks": len(ranked["bottlenecks"]),
                    "generated_utc": snap.get("generated_utc"),
                }, default=str), "application/json")
            elif path == "/latest":
                self._send(200, json.dumps(snap, default=str, indent=2), "application/json")
            elif path == "/bottlenecks":
                self._send(200, json.dumps(ranked, default=str, indent=2), "application/json")
            elif path == "/metrics":
                # Prometheus exposition (text 0.0.4) — the machine/agent surface.
                self._send(200, render_prometheus(snap, ranked),
                           "text/plain; version=0.0.4; charset=utf-8")
            else:
                self._send(404, json.dumps({"error": "not found",
                           "paths": ["/", "/health", "/latest", "/bottlenecks", "/metrics"]}), "application/json")

        def log_message(self, *a):  # quiet
            pass

    srv = http.server.ThreadingHTTPServer(("0.0.0.0", port), H)
    print(f"fleet_bottleneck dashboard on http://localhost:{port}/  "
          f"(refresh {interval}s, audit={'on' if audit else 'off'})")
    print(f"  dashboard: http://localhost:{port}/")
    print(f"  health:    http://localhost:{port}/health")
    print(f"  metrics:   http://localhost:{port}/metrics  (Prometheus exposition)")
    print("  api:       /latest  /bottlenecks")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\nshutting down")


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    p = argparse.ArgumentParser(description="Fleet bottleneck detection + visibility (v1).")
    sub = p.add_subparsers(dest="cmd")
    for name in ("report", "json", "prometheus", "serve"):
        q = sub.add_parser(name)
        q.add_argument("--no-audit", action="store_true", help="skip the token-spend pass")
        q.add_argument("--audit-days", type=float, default=1.5)
        q.add_argument("--audit-max", type=int, default=80)
        if name in ("json", "prometheus"):
            q.add_argument("--out", default=None)
        if name == "serve":
            q.add_argument("--port", type=int, default=9095)
            q.add_argument("--interval", type=int, default=45)
    a = p.parse_args()
    cmd = a.cmd or "report"

    if cmd == "serve":
        serve(a.port, not a.no_audit, a.audit_days, a.audit_max, a.interval)
        return

    snap = collect(audit=not a.no_audit, audit_days=a.audit_days, audit_max=a.audit_max)
    ranked = rank_bottlenecks(snap)
    txt = write_artifacts(snap, ranked)
    if cmd == "json":
        payload = {"app_version": fleet_version.app_version(), "generated_utc": snap["generated_utc"], **ranked, "snapshot": snap}
        out = json.dumps(payload, default=str, indent=2)
        if a.out:
            open(a.out, "w", encoding="utf-8").write(out)
            print(f"wrote {a.out}", file=sys.stderr)
        else:
            print(out)
    elif cmd == "prometheus":
        # write_artifacts already refreshed _registry/fleet_bottleneck.prom; emit to
        # stdout (or --out) so a textfile collector / cron scrape can capture it too.
        out = render_prometheus(snap, ranked)
        if a.out:
            open(a.out, "w", encoding="utf-8").write(out)
            print(f"wrote {a.out}", file=sys.stderr)
        else:
            print(out)
    else:
        print(txt)


if __name__ == "__main__":
    main()
