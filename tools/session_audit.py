#!/usr/bin/env python3
"""
session_audit.py — reusable auditor for Claude Code session-transcript JSONL files.

The transcripts live under <claude-home>/projects/<namespace>/<session-uuid>.jsonl and
carry EXACT token accounting (the `.dos/` tool-stream telemetry only has content digests,
no token counts — so this is the tool that can answer the token-weighted questions:
real input:output ratio, real prompt-cache / KV reuse, real cost, real tool mix).

Schema (per JSONL line, one record):
  type=assistant : message.usage = {input_tokens, output_tokens,
                     cache_read_input_tokens, cache_creation_input_tokens,
                     server_tool_use{web_search_requests,web_fetch_requests}, iterations[...]}
                   message.content = [ {type: thinking|text|tool_use}, ... ]
  type=user      : message.content = str (typed prompt / slash-command wrapper / hook text)
                                   or [ {type: tool_result, content: ...}, ... ]
  plus meta records: last-prompt, mode, permission-mode, attachment, file-history-snapshot,
                     system, queue-operation, ai-title, summary

Usage:
  python session_audit.py discover [--since-days N] [--root DIR ...]
  python session_audit.py audit  [--since-days N] [--root DIR ...] [--json OUT] [--md OUT]
  python session_audit.py deep   <session.jsonl>          # follow ONE trajectory top-to-bottom

All numbers from token usage are EXACT. Cost uses the PRICING table below (an ASSUMPTION,
clearly flagged) — edit it to match current rates; token counts are the ground truth.

Besides the token/cost lens, every subcommand also runs the BEHAVIORAL lens (#2365):
per-tool call/error counts and error rate, shell timeout kills (exit 143 / "timed
out"), foreground sleep-polls, Edit/Write read-discipline churn, repeated identical
failure signatures, and per-file mutation churn — the stuck/churn half a token-only
audit can't see.

Cost is split by BILLING BUCKET (provider): claude-* is the Anthropic invoice; a gemini-*
/ gpt-* / local model is a different invoice. The auditor never sums cost across buckets
and never prices a non-Claude model at Claude rates — an unpriced model is reported with
exact tokens but no fabricated cost. <synthetic> (harness-injected) is non-billed ($0).
"""
import sys
import os
import re
import json
import glob
import argparse
import statistics
import subprocess
import collections
import datetime

# --- pricing assumption (USD per 1e6 tokens). EDIT to match real card; flagged in output. ---
# Rates are per-model-family and ANTHROPIC-ONLY. A model that matches none of these is
# NOT silently priced as Opus — it is an UNPRICED, SEPARATE billing bucket. Claude and
# Gemini are different vendors = different invoices; the audit never sums cost across
# buckets and never invents a Claude price for a non-Claude model. Add a vendor's card to
# PRICING (and its substrings to PROVIDER_BUCKETS) to price it; until then its tokens are
# reported and its cost is left blank.
PRICING = {
    # model_substring: (input, cache_write_5m, cache_read, output)
    "opus":   (15.0, 18.75, 1.50, 75.0),
    "sonnet": ( 3.0,  3.75, 0.30, 15.0),
    "haiku":  ( 0.80, 1.00, 0.08,  4.0),
    "fable":  ( 3.0,  3.75, 0.30, 15.0),
}
PRICING_IS_ASSUMPTION = True

# Harness-injected pseudo-models that never reach any vendor — never billed, never a bucket.
NONBILLED_MODELS = {"<synthetic>", "?", ""}

# Provider / billing-bucket classification. Each provider is a DISTINCT invoice; the report
# breaks cost out per bucket and refuses to sum across them. Substring-matched, first hit wins;
# Anthropic is first so a claude-* tier never falls through to another vendor's bucket.
PROVIDER_BUCKETS = [
    ("Anthropic (Claude)",  ("claude", "opus", "sonnet", "haiku", "fable")),
    ("Google (Gemini)",     ("gemini", "gemma")),
    ("OpenAI",              ("gpt", "o1-", "o3-", "o4-", "davinci")),
    ("local / self-hosted", ("qwen", "llama", "mistral", "mixtral", "phi-", "deepseek")),
]

def provider_bucket(model):
    """Which vendor invoice a model lands on — its billing bucket, not its rate."""
    if (model or "") in NONBILLED_MODELS:
        return "non-billed (harness)"
    m = (model or "").lower()
    if not m:
        return "non-billed (harness)"
    for name, subs in PROVIDER_BUCKETS:
        if any(s in m for s in subs):
            return name
    return "UNKNOWN (unpriced bucket)"

DEFAULT_ROOTS = [
    os.path.join(os.environ.get("CLAUDE_CONFIG_DIR", os.path.expanduser("~/.claude")), "projects"),
]
# transient test-fixture / temp-workspace namespaces — never "our own sessions"
EXCLUDE_NS_SUBSTR = ["pytest-of-USER", "AppData-Local-Temp", "workspace", "-ws", "test_"]
NS_INCLUDE_PREFIX = ""   # all non-excluded namespaces by default; narrow with --ns-prefix PREFIX

READ_ONLY_TOOLS = {"Read", "Glob", "Grep", "LS", "NotebookRead", "WebFetch", "WebSearch",
                   "TodoRead", "ToolSearch",
                   # observation-only harness tools: poll/query state, never mutate it.
                   "Monitor", "TaskGet", "TaskList", "TaskOutput",
                   "ReadMcpResourceTool", "ListMcpResourcesTool", "ReadMcpResourceDirTool"}
# Bash/Edit/Write/NotebookEdit/TaskCreate/TaskUpdate/TaskStop/Workflow/etc. are
# side-effecting or spawn.

# --- behavioral lens (#2365): stuck / churn detectors -------------------------
# All detectors read ONLY what the transcript already carries (tool_use inputs +
# errored tool_results); none of them re-run anything or read the process table.
SHELL_TOOLS = {"Bash", "PowerShell"}
# A shell result that was killed by the harness deadline: SIGTERM exit 143 or the
# harness "timed out" phrasing. \W{0,3} absorbs ':'/': ' variants between
# "code"/"status" and the number.
TIMEOUT_KILL_RE = re.compile(r"exit (?:code|status)\W{0,3}143\b|timed out",
                             re.IGNORECASE)
# A foreground shell call whose command *starts* with a sleep is a poll — the
# turn is blocked doing nothing (background sleeps are fine, they don't block).
SLEEP_POLL_RE = re.compile(r"^\s*(?:sleep|start-sleep)\b", re.IGNORECASE)
# INTERACTIVE_HANG (#2365 detector 3): a shell call that opened an editor/pager with
# no TTY and wedged until the turn deadline — the top recurring stall in the trajectory
# audit (15 hangs / 12 sessions). Keyed on the repo-guard's EXACT emission (the reason
# token, or its distinctive verbatim phrase), NOT a loose "editor"/"hang" substring: the
# loose form conflated this class up from 15 to 68 by catching prose that merely
# mentioned an editor. Mirrors internal/repoguard ReasonInteractiveHang.
INTERACTIVE_HANG_RE = re.compile(
    r"\bINTERACTIVE_HANG\b|waits for a human and this session has no TTY",
    re.IGNORECASE)
# A fak capability-floor refusal surfaced in a tool_result: the guard SET ASIDE the
# call (POLICY_BLOCK / REQUIRE_WITNESS / a preview-confirm gate). This is the guard
# doing its job, NOT the shell or the agent's command failing — folding it into the raw
# shell error rate is the conflation that inflated the headline PowerShell number (#2365
# finding 2). Anchored on the in-band "[fak] refused" note and the reason tokens.
POLICY_REFUSAL_RE = re.compile(
    r"\[fak\] refused\b|\bPOLICY_BLOCK\b|\bREQUIRE_WITNESS\b|preview-confirm gate",
    re.IGNORECASE)
# A usage error: the command or a path did not exist (a typo / wrong-shell mistake),
# across bash ("command not found", "No such file or directory") and PowerShell/cmd
# ("is not recognized as the name of…"/"…an internal or external command",
# CommandNotFoundException). Distinct from a command that RAN and exited nonzero.
SHELL_NOT_FOUND_RE = re.compile(
    r"command not found"
    r"|is not recognized as (?:the name of|an internal or external)"
    r"|CommandNotFound"
    r"|No such file or directory",
    re.IGNORECASE)
# A command that RAN and exited nonzero (the residual "genuine downstream failure"
# bucket) — an explicit nonzero exit-code/status phrasing, excluding the 143 deadline
# kill TIMEOUT_KILL_RE already owns.
SHELL_NONZERO_RE = re.compile(r"exit (?:code|status)\W{0,3}([0-9]+)", re.IGNORECASE)
# Edit/Write churn: mutation calls wasted on read-before-write discipline.
EDIT_CHURN_SIGNATURES = {
    "not_read":   "File has not been read yet",
    "stale_read": "has been modified since read",
}
MUTATION_TOOLS = {"Edit", "Write", "NotebookEdit"}
# The same failure repeated this often in one session is a stuck loop. Verbatim
# repeats key on (tool, args, error) — a true retry loop; the error-mass view
# keys on (tool, error text) alone — a recurring failure CLASS whose args may
# vary (e.g. rotating through wedged bridge sessions). Conflating the two
# false-alarmed on five distinct commands sharing one timeout string.
REPEAT_FAILURE_MIN = 3
# A file mutated this often is only a rewrite/flip-flop loop when the edits
# revisit the same regions or undo each other; distinct-region build-out
# (verb-per-edit-triple, helper extraction) is healthy iteration.
FILE_CHURN_MIN = 5
# A gap this long with ZERO transcript records is a harness/API stall — the
# kind of dead time the sleep-poll counter can't see.
STALL_GAP_S = 300
# not_read edit-churn sub-classes (#2375 detector 1). The raw not_read counter
# conflates three mechanically distinct causes and only the last is misbehavior:
#   post_resume      — a --resume/compaction reset the harness read-state tracker
#                      while the compacted context still believed the file was read
#                      (a prior successful Read of THIS path exists, then a restart
#                      marker, then the edit fails not-read). Not the agent's fault.
#   self_duplicate   — a forked/duplicated branch re-issued a stale Write of a file
#                      the SAME session already wrote; the guard fence caught it (a
#                      prior successful Write/Edit of THIS path exists). Not misbehavior.
#   true_never_read  — an edit of a file this session never read: the real defect.
# Precedence: self_duplicate (concrete prior write) > post_resume (prior read +
# restart) > true_never_read (default). Listed for stable render order.
NOT_READ_CLASSES = ("post_resume", "self_duplicate", "true_never_read")
# A transcript-carried restart/compaction marker: a compaction summary record, or a
# resume/clear command, or the bare "Resume" continuation prompt. Anchored so a
# prompt that merely *mentions* resume ("resume the sweep…") does NOT match.
RESTART_MARKER_RE = re.compile(
    r"session is being continued from a previous"
    r"|<command-name>\s*/(?:resume|clear)\b"
    r"|^\s*resume\s*$",
    re.IGNORECASE)
# Successful-call loop detector (#2375 detector 2): loops of SUCCESSFUL identical
# calls (read-loops / glob-storms / output-file poll loops) that the failure/mutation
# loop checks never see. Only these tools count — Read/Glob/Grep/LS + a shell reading
# a file; Monitor/TaskOutput are the SANCTIONED poll surface and are excluded.
SUCCESS_LOOP_TOOLS = {"Read", "Glob", "Grep", "LS", "Bash", "PowerShell"}
# This many identical SUCCESSFUL (tool, args-digest) calls in one session is a poll
# loop / storm, not the healthy 2-4 re-reads of iterative editing.
SUCCESS_LOOP_MIN = 8
# Suffix-cache reset detector (#3069): per billed turn the provider reports the
# size of the cached prefix it reused (cache_read_input_tokens). It CLIMBS as the
# conversation grows, then SNAPS back toward a floor when a previously-cached
# suffix is invalidated mid-session — the provider re-writes that suffix, which
# bills as a cache_create BURST. A per-turn cache_read drop larger than this many
# tokens is one such invalidation; the value it snaps TO is a "reset floor"
# (empirically ~56k: the stable system-prompt+tools prefix). Fed one turn at a
# time AFTER message.id de-dup, so a re-serialized/retried turn never double-counts.
# This is the "reset-to-a-fixed-floor" signature the read-share lens is blind to:
# a burst INFLATES read-share (the re-created suffix is read back next turn), so a
# thrashing session looks MORE cached, not less — the exact blind spot #3069 fixes.
SUFFIX_RESET_DROP_MIN = 20_000
# A "long" session for the burst-offender table (#3069) — a session with at least
# this many billed turns AND ≥1 suffix reset is a cache-CREATE-thrash offender the
# heaviest-by-output table never surfaces.
BURST_LONG_SESSION_MIN = 8

def _tool_path(tool_input):
    """The file a mutation/read call targets — its read-state identity (#2375 d1)."""
    ti = tool_input if isinstance(tool_input, dict) else {}
    return ti.get("file_path") or ti.get("notebook_path")

def _tool_label(name, tool_input):
    """A short human-readable offender label for a call (path / pattern / command)."""
    ti = tool_input if isinstance(tool_input, dict) else {}
    raw = (ti.get("file_path") or ti.get("notebook_path") or ti.get("pattern")
           or ti.get("path") or ti.get("command") or "")
    return _norm_head(raw, 120)

def _is_restart_record(r):
    """True for a session-restart / compaction marker (#2375 d1: post_resume signal)."""
    if r.get("isCompactSummary") or r.get("type") == "summary":
        return True
    lp = r.get("lastPrompt")
    if isinstance(lp, str) and RESTART_MARKER_RE.search(lp):
        return True
    if r.get("type") == "user":
        c = (r.get("message", {}) or {}).get("content")
        if isinstance(c, str) and RESTART_MARKER_RE.search(c):
            return True
    return False

def _txt_str(content, cap=4000):
    """Flatten a content field (str or list of blocks) to text, capped at `cap`."""
    if isinstance(content, str):
        return content[:cap]
    if isinstance(content, list):
        parts, n = [], 0
        for b in content:
            if n >= cap:
                break
            if isinstance(b, dict):
                s = _txt_str(b.get("content", b.get("text", "")), cap - n)
            elif isinstance(b, str):
                s = b[:cap - n]
            else:
                continue
            parts.append(s)
            n += len(s)
        return "".join(parts)
    return ""

def _norm_head(s, cap=200):
    """whitespace-collapsed head of a string, for region/signature identity."""
    return " ".join((s or "").split())[:cap]

class BehaviorLens:
    """Per-transcript stuck/churn detectors (#2365). Fed one tool_use /
    tool_result at a time (post de-dup); `summary()` emits one plain dict:
    per-tool error counts, timeout kills, foreground sleep-polls, Edit/Write
    read-discipline churn, verbatim repeat failures (same tool+args+error, a
    true retry loop) vs failure-class mass (same tool+error, args vary), and
    per-file mutation churn discriminated into rewrite-loop vs distinct-region
    build-out via edit-region identity + revert pairs."""

    def __init__(self):
        self.errors = collections.Counter()        # tool -> errored results
        self.timeout_kills = 0
        self.interactive_hangs = 0                 # editor/pager-no-TTY wedges (#2365 d3)
        self.shell_error_classes = collections.Counter()  # shell err cause breakdown (#2365 finding 2)
        self.sleep_polls = 0
        self.edit_churn = collections.Counter()    # not_read / stale_read
        self.not_read_classes = collections.Counter()  # post_resume/self_dup/true (#2375 d1)
        self.true_never_read_paths = collections.Counter()  # path -> n (#3942 offenders)
        self.verbatim_sigs = collections.Counter() # (tool, args_key, sig) -> n
        self.mass_sigs = collections.Counter()     # (tool, sig) -> n
        self.file_writes = collections.Counter()   # file_path -> mutation calls
        self.file_regions = collections.defaultdict(list)  # path -> [(old_h, new_h)]
        # not_read sub-classification signals (#2375 d1)
        self.read_paths = set()      # paths with a prior SUCCESSFUL Read in-session
        self.mutated_paths = set()   # paths with a prior SUCCESSFUL Write/Edit in-session
        self.saw_restart = False     # a --resume/compaction marker has been seen
        # successful-call loop signals (#2375 d2): count identical calls, subtract
        # the ones that errored, so a loop of SUCCESSFUL calls stands alone.
        self.call_sigs = collections.Counter()     # (tool, args_key) -> calls
        self.call_labels = {}                       # (tool, args_key) -> offender label
        self.err_sig_counts = collections.Counter() # (tool, args_key) -> errored calls
        # suffix-cache reset signals (#3069): per billed turn cache_read climbs then
        # SNAPS back when a cached suffix is invalidated mid-session — a cache_create
        # burst the read-share lens hides. Fed post-dedup, one turn at a time.
        self._prev_cache_read = None                # last billed turn's cache_read
        self.suffix_resets = 0                      # count of snap-backs > threshold
        self.reset_floors = collections.Counter()   # value snapped-TO -> times

    def see_turn_usage(self, cache_read):
        """Feed one BILLED turn's cache_read (already message.id-deduped upstream).
        A drop > SUFFIX_RESET_DROP_MIN vs the previous turn is a suffix-cache
        invalidation (#3069); the value it snaps TO is a reset floor. The first
        turn (prev is None) can never trigger, and context GROWTH (an increase)
        never does — only a genuine snap-back counts."""
        cr = int(cache_read or 0)
        if self._prev_cache_read is not None \
                and cr < self._prev_cache_read - SUFFIX_RESET_DROP_MIN:
            self.suffix_resets += 1
            self.reset_floors[cr] += 1
        self._prev_cache_read = cr

    def note_restart(self):
        """Record that a session-restart / compaction marker occurred (#2375 d1)."""
        self.saw_restart = True

    def see_tool_use(self, name, tool_input, args_key=None):
        ti = tool_input if isinstance(tool_input, dict) else {}
        if name in SHELL_TOOLS and not ti.get("run_in_background") \
                and SLEEP_POLL_RE.match(ti.get("command") or ""):
            self.sleep_polls += 1
        if name in MUTATION_TOOLS:
            path = ti.get("file_path") or ti.get("notebook_path")
            if path:
                self.file_writes[path] += 1
                # Region identity: Edit carries old/new strings; a Write is a
                # whole-file rewrite (one region), so N Writes of one file still
                # read as revisiting the same region.
                old_h = hash(_norm_head(ti.get("old_string", "")))
                new_h = hash(_norm_head(ti.get("new_string",
                                               ti.get("content", ""))))
                self.file_regions[path].append((old_h, new_h))
        # Successful-call loop tally (#2375 d2): every eligible call keys on its
        # (tool, args-digest); errored ones are subtracted at summary time.
        if name in SUCCESS_LOOP_TOOLS and args_key is not None:
            k = (name, args_key)
            self.call_sigs[k] += 1
            self.call_labels[k] = _tool_label(name, tool_input)

    def _classify_not_read(self, path):
        """Sub-classify a not_read edit-churn failure (#2375 d1). Precedence:
        a concrete prior write (self_duplicate) beats a prior read + restart
        (post_resume); a never-read edit is the real defect (true_never_read)."""
        if path and path in self.mutated_paths:
            return "self_duplicate"
        if self.saw_restart and path and path in self.read_paths:
            return "post_resume"
        return "true_never_read"

    def _classify_shell_error(self, text):
        """Sub-classify a SHELL_TOOLS error by surface cause (#2365 finding 2). The raw
        per-tool error rate conflates guard refusals and turn-deadline hangs (NOT the
        shell being flaky) with genuine command failures; this breakdown makes the number
        honest and actionable. Precedence, most-specific first:
          policy_refusal  — the capability floor set the call aside (guard did its job).
          interactive_hang— an editor/pager opened with no TTY and wedged (#2365 d3).
          timeout_kill    — the harness deadline (SIGTERM 143 / "timed out") fired.
          not_found       — command/path did not exist (a typo / wrong-shell usage error).
          nonzero_exit    — the command RAN and exited nonzero: a real downstream failure.
          other           — an errored result matching none of the above shapes."""
        if POLICY_REFUSAL_RE.search(text):
            return "policy_refusal"
        if INTERACTIVE_HANG_RE.search(text):
            return "interactive_hang"
        if TIMEOUT_KILL_RE.search(text):
            return "timeout_kill"
        if SHELL_NOT_FOUND_RE.search(text):
            return "not_found"
        m = SHELL_NONZERO_RE.search(text)
        if m and m.group(1) != "0":
            return "nonzero_exit"
        return "other"

    def see_tool_result(self, tool, is_error, text, args_key=None, path=None):
        if not is_error:
            # A success establishes read-state / a prior write for the path — the
            # signals the not_read sub-classifier reads (#2375 d1).
            if path:
                if tool == "Read":
                    self.read_paths.add(path)
                elif tool in MUTATION_TOOLS:
                    self.mutated_paths.add(path)
            return
        self.errors[tool] += 1
        if tool in SUCCESS_LOOP_TOOLS and args_key is not None:
            self.err_sig_counts[(tool, args_key)] += 1   # not a SUCCESSFUL call (#2375 d2)
        if tool in SHELL_TOOLS:
            cls = self._classify_shell_error(text)
            self.shell_error_classes[cls] += 1           # honest error breakdown (#2365 finding 2)
            if cls == "timeout_kill":
                self.timeout_kills += 1
        if INTERACTIVE_HANG_RE.search(text):
            self.interactive_hangs += 1                  # editor/pager-no-TTY wedge (#2365 d3)
        for key, sig in EDIT_CHURN_SIGNATURES.items():
            if sig in text:
                self.edit_churn[key] += 1
                if key == "not_read":
                    cls = self._classify_not_read(path)
                    self.not_read_classes[cls] += 1
                    # #3942 — capture WHICH file the genuine defect hit, so the
                    # report can name the offender instead of only counting it.
                    if cls == "true_never_read" and path:
                        self.true_never_read_paths[path] += 1
        sig = _norm_head(text, 160)
        self.verbatim_sigs[(tool, args_key, sig)] += 1
        self.mass_sigs[(tool, sig)] += 1

    def _success_loop_rows(self):
        """SUCCESSFUL identical-call loops (#2375 d2): calls minus errored calls,
        thresholded, worst-first, with the offender label."""
        rows = []
        for (tool, ak), n in self.call_sigs.items():
            succ = n - self.err_sig_counts.get((tool, ak), 0)
            if succ >= SUCCESS_LOOP_MIN:
                rows.append({"tool": tool,
                             "target": self.call_labels.get((tool, ak), ""),
                             "count": succ})
        return sorted(rows, key=lambda r: -r["count"])

    def _churn_rows(self):
        rows = []
        for f, n in self.file_writes.items():
            if n < FILE_CHURN_MIN:
                continue
            regions = self.file_regions.get(f, [])
            distinct = len({o for o, _ in regions}) or 1
            seen_old = set()
            reverts = 0
            for old_h, new_h in regions:
                if new_h in seen_old and new_h != old_h:
                    reverts += 1          # this edit restores an earlier state
                seen_old.add(old_h)
            # Rewrite loop = edits keep revisiting the same few regions
            # (distinct*2 <= n), OR undo each other WHILE regions are being
            # reused (reverts and distinct < n). A single revert amid all-
            # distinct regions (distinct == n) is NOT a loop: a long linear
            # refactor that happens to restore one earlier snippet once is
            # healthy build-out, not thrash. Requiring distinct < n on the
            # revert arm kills the b72e2808 false alarm (n=19, distinct=19,
            # reverts=1) while still catching real region-reuse thrash
            # (5c72b8ba: n=5, distinct=4, reverts=2).
            if distinct * 2 <= n or (reverts >= 1 and distinct < n):
                rows.append({"file": f, "count": n,
                             "distinct_regions": distinct, "reverts": reverts})
        return sorted(rows, key=lambda r: -r["count"])

    def summary(self):
        repeats = sorted(({"tool": t, "sig": sig, "count": n}
                          for (t, _a, sig), n in self.verbatim_sigs.items()
                          if n >= REPEAT_FAILURE_MIN),
                         key=lambda r: -r["count"])
        mass = sorted(({"tool": t, "sig": sig, "count": n}
                       for (t, sig), n in self.mass_sigs.items()
                       if n >= REPEAT_FAILURE_MIN),
                      key=lambda r: -r["count"])
        success_loops = self._success_loop_rows()
        max_success_loop = max((n - self.err_sig_counts.get(k, 0)
                                for k, n in self.call_sigs.items()), default=0)
        return {
            "tool_errors": dict(self.errors),
            "timeout_kills": self.timeout_kills,
            "interactive_hangs": self.interactive_hangs,
            "shell_error_classes": dict(self.shell_error_classes),
            "sleep_polls": self.sleep_polls,
            "edit_churn": dict(self.edit_churn),
            "not_read_classes": dict(self.not_read_classes),
            "true_never_read_paths": sorted(
                ({"path": p, "count": n}
                 for p, n in self.true_never_read_paths.items()),
                key=lambda r: (-r["count"], r["path"]))[:10],
            "repeat_failures": repeats[:10],
            "max_repeat_failure": max(self.verbatim_sigs.values(), default=0),
            "failure_mass": mass[:10],
            "max_failure_mass": max(self.mass_sigs.values(), default=0),
            "file_churn": self._churn_rows()[:10],
            "max_file_churn": max(self.file_writes.values(), default=0),
            "success_loops": success_loops[:10],
            "max_success_loop": max_success_loop,
            # suffix-cache reset burst (#3069)
            "suffix_resets": self.suffix_resets,
            "suffix_reset_floor": (self.reset_floors.most_common(1)[0][0]
                                   if self.reset_floors else None),
            "suffix_reset_floors": dict(self.reset_floors),
        }

def price_for(model):
    """Rate card for a model, or None when we hold no card for its billing bucket.
    None means: report the tokens, never invent a cost (and never default to Opus)."""
    if (model or "") in NONBILLED_MODELS:
        return None
    m = (model or "").lower()
    if not m:
        return None
    for key, rates in PRICING.items():
        if key in m:
            return rates
    return None

def cost_usd(model, inp, cwrite, cread, out):
    rates = price_for(model)
    if rates is None:
        return 0.0          # unpriced / non-Anthropic / non-billed — kept out of the total
    pi, pcw, pcr, po = rates
    return (inp*pi + cwrite*pcw + cread*pcr + out*po) / 1e6

def model_cost(model, c):
    """Cost for one model's rolled-up token Counter (input/cache_create/cache_read/output)."""
    return cost_usd(model, c.get("input", 0), c.get("cache_create", 0),
                    c.get("cache_read", 0), c.get("output", 0))

def model_tier(model):
    """Stable model tier for mix KPIs (opus/sonnet/haiku/etc.), not a full model id."""
    if (model or "") in NONBILLED_MODELS:
        return "<synthetic>"
    m = (model or "").lower()
    for key in PRICING:
        if key in m:
            return key
    return "unpriced"

def discover(roots, since_days=None, ns_prefix=NS_INCLUDE_PREFIX, include_subagents=False):
    cutoff = None
    if since_days is not None:
        cutoff = datetime.datetime.now().timestamp() - since_days*86400
    out = []
    for root in roots:
        if not os.path.isdir(root):
            continue
        for ns in os.listdir(root):
            if any(s in ns for s in EXCLUDE_NS_SUBSTR):
                continue
            if ns_prefix and not ns.startswith(ns_prefix):
                continue
            nsdir = os.path.join(root, ns)
            if not os.path.isdir(nsdir):
                continue
            # top-level session transcripts (one per conversation)
            top = set(glob.glob(os.path.join(nsdir, "*.jsonl")))
            paths = [(p, "session") for p in top]
            if include_subagents:
                # subagent / workflow transcripts live in <session-id>/**/*.jsonl —
                # SEPARATE files, so top-level session usage UNDERCOUNTS true spend.
                for p in glob.glob(os.path.join(nsdir, "**", "*.jsonl"), recursive=True):
                    if p not in top:
                        paths.append((p, "subagent"))
            for path, kind in paths:
                try:
                    st = os.stat(path)
                except OSError:
                    continue
                if cutoff and st.st_mtime < cutoff:
                    continue
                out.append({"root": root, "ns": ns, "path": path, "kind": kind,
                            "size": st.st_size, "mtime": st.st_mtime})
    out.sort(key=lambda r: r["mtime"], reverse=True)
    return out

def _parent_session(rec):
    """The TOP-LEVEL session a discovered transcript belongs to (#3226).

    A top-level transcript is its own parent. A subagent / workflow transcript lives
    at <ns>/<parent-session-id>/**/<agent>.jsonl, so its parent is the first path
    component under the namespace directory — the session that spawned the fan-out,
    and therefore the one its spend and behavioral findings are attributed to."""
    stem = os.path.splitext(os.path.basename(rec["path"]))[0]
    if rec.get("kind") != "subagent":
        return stem
    try:
        rel = os.path.relpath(rec["path"], os.path.join(rec["root"], rec["ns"]))
    except (ValueError, KeyError):      # e.g. different drives on Windows
        return stem
    head = rel.replace("\\", "/").split("/")[0]
    return os.path.splitext(head)[0] or stem

def _txt_len(content):
    """char length of a content field that may be str or list of blocks."""
    if isinstance(content, str):
        return len(content)
    if isinstance(content, list):
        n = 0
        for b in content:
            if isinstance(b, dict):
                c = b.get("content", b.get("text", ""))
                n += _txt_len(c)
            elif isinstance(b, str):
                n += len(b)
        return n
    return 0

def _looks_like_typed_prompt(s):
    """user string that is an actual prompt (typed or slash-command), not pure hook/reminder."""
    if not isinstance(s, str):
        return False
    st = s.strip()
    if not st:
        return False
    if st.startswith("<system-reminder>") or st.startswith("Caveat:"):
        return False
    return True

def _token_economics(tok):
    """The session's token economics: total ingested context, the I:O ratio, and
    the cache-read and cache-create shares of that context.

    #3069: the cache-CREATE share of all ingested context — the burst counterpart
    to cache_hit_frac, sharing its EXACT denominator (read + create + input) so the
    three shares sum to 1. A session that re-writes its cached suffix mid-run carries
    a high cc_share even while its flattering cache_hit_frac stays high; this is the
    signal the read-share lens is blind to.
    """
    total_in = tok["input"] + tok["cache_read"] + tok["cache_create"]
    io_ratio = total_in / tok["output"] if tok["output"] else None
    cache_hit = tok["cache_read"] / (tok["cache_read"] + tok["cache_create"] + tok["input"]) \
                if (tok["cache_read"] + tok["cache_create"] + tok["input"]) else None
    cc_share = tok["cache_create"] / (tok["cache_read"] + tok["cache_create"] + tok["input"]) \
               if (tok["cache_read"] + tok["cache_create"] + tok["input"]) else None
    return total_in, io_ratio, cache_hit, cc_share

def _wall_seconds(ts_min, ts_max):
    """Wall-clock seconds between the first and last record, or None when the
    transcript carries no usable timestamps."""
    wall = None
    if ts_min and ts_max:
        try:
            a = datetime.datetime.fromisoformat(ts_min.replace("Z", "+00:00"))
            b = datetime.datetime.fromisoformat(ts_max.replace("Z", "+00:00"))
            wall = (b - a).total_seconds()
        except Exception:
            pass
    return wall

def analyze(path):
    rec_types = collections.Counter()
    models = collections.Counter()
    tools = collections.Counter()
    tool_input_chars = 0
    tool_result_chars = 0
    n_tool_use = 0
    n_tool_result = 0
    n_thinking = 0
    n_text = 0
    tok = dict(input=0, output=0, cache_read=0, cache_create=0,
               web_search=0, web_fetch=0, iterations=0)
    per_model = collections.defaultdict(collections.Counter)   # model -> token Counter (per billing bucket)
    cost = 0.0
    ts_min = ts_max = None
    prompts = []          # (ts, text) — the trajectory's user asks
    assistant_turns = 0
    interrupted = 0
    dup_assistant_lines = 0
    # Claude Code writes MULTIPLE transcript lines per billed assistant turn
    # (streaming events / retries / sidechain re-serialization). Each carries the
    # SAME message.usage, so folding every line double-counts tokens/cost/turns
    # (measured ~2x, session-dependent). The model's own response id (message.id)
    # is the identity of a billed turn — verified to collapse exactly and to never
    # disagree on usage among its duplicate lines. De-dup on it; only id-less
    # lines (defensive — none seen in practice) are counted individually.
    seen_msg_ids = set()
    # Content blocks are deduped by BLOCK identity, not line identity: newer
    # transcripts stream ONE block per line under a shared message.id (skipping
    # dup lines wholesale undercounted tool calls ~6x there), while older ones
    # repeat the FULL content array on every duplicate line.
    seen_blocks = set()
    lens = BehaviorLens()
    hook_outcomes = collections.Counter()
    hook_durations = collections.defaultdict(list)
    hook_failures = collections.Counter()
    tooluse_names = {}   # tool_use id -> tool name, to attribute tool_results
    tooluse_args = {}    # tool_use id -> args digest, for verbatim-retry keying
    tooluse_paths = {}   # tool_use id -> target file path, for not_read sub-class (#2375 d1)
    prev_dt = None
    stall_gaps = 0
    max_gap_s = 0.0

    try:
        with open(path, encoding="utf-8") as f:
            lines = f.read().splitlines()
    except Exception as e:
        return {"path": path, "error": str(e)}

    for line in lines:
        line = line.strip()
        if not line:
            continue
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        t = r.get("type")
        rec_types[t] += 1
        if _is_restart_record(r):
            lens.note_restart()          # #2375 d1: a --resume/compaction boundary
        ts = r.get("timestamp")
        if ts:
            ts_min = ts if ts_min is None or ts < ts_min else ts_min
            ts_max = ts if ts_max is None or ts > ts_max else ts_max
            try:
                dt = datetime.datetime.fromisoformat(ts.replace("Z", "+00:00"))
            except Exception:
                dt = None
            if dt is not None:
                if prev_dt is not None:
                    gap = (dt - prev_dt).total_seconds()
                    if gap > max_gap_s:
                        max_gap_s = gap
                    if gap >= STALL_GAP_S:
                        stall_gaps += 1   # zero-record dead time (harness stall)
                prev_dt = dt

        attachment = r.get("attachment") if t == "attachment" else None
        if isinstance(attachment, dict) and str(attachment.get("type", "")).startswith("hook_"):
            outcome = str(attachment.get("type"))[5:] or "unknown"
            event = str(attachment.get("hookEvent") or "unknown")
            hook_outcomes[(event, outcome)] += 1
            duration = attachment.get("durationMs")
            if isinstance(duration, (int, float)) and duration >= 0:
                hook_durations[event].append(int(duration))
            if outcome in {"non_blocking_error", "cancelled"}:
                detail = attachment.get("stderr") or attachment.get("stdout") or "no diagnostic output"
                hook_failures[(event, outcome, _norm_head(str(detail), 200))] += 1

        if t == "assistant":
            msg = r.get("message", {}) or {}
            mid = msg.get("id")
            new_turn = True
            if mid is not None:
                if mid in seen_msg_ids:
                    dup_assistant_lines += 1
                    new_turn = False   # usage already folded; blocks may be new
                else:
                    seen_msg_ids.add(mid)
            if new_turn:
                assistant_turns += 1
                models[msg.get("model", "?")] += 1
                u = msg.get("usage", {}) or {}
                inp = u.get("input_tokens", 0) or 0
                out = u.get("output_tokens", 0) or 0
                cr  = u.get("cache_read_input_tokens", 0) or 0
                cc  = u.get("cache_creation_input_tokens", 0) or 0
                stu = u.get("server_tool_use", {}) or {}
                tok["input"] += inp
                tok["output"] += out
                tok["cache_read"] += cr
                tok["cache_create"] += cc
                tok["web_search"] += stu.get("web_search_requests", 0) or 0
                tok["web_fetch"]  += stu.get("web_fetch_requests", 0) or 0
                tok["iterations"] += len(u.get("iterations", []) or [])
                cost += cost_usd(msg.get("model"), inp, cc, cr, out)
                lens.see_turn_usage(cr)   # #3069: suffix-cache reset detection
                pm = per_model[msg.get("model", "?")]
                pm["turns"] += 1
                pm["input"] += inp
                pm["output"] += out
                pm["cache_read"] += cr
                pm["cache_create"] += cc
            dedup_blocks = mid is not None
            for b in (msg.get("content") or []):
                if not isinstance(b, dict):
                    continue
                bt = b.get("type")
                if bt == "tool_use":
                    key = b.get("id") or (mid, "tool_use", b.get("name"),
                                          json.dumps(b.get("input", {}),
                                                     sort_keys=True, default=str))
                    if dedup_blocks:
                        if key in seen_blocks:
                            continue
                        seen_blocks.add(key)
                    n_tool_use += 1
                    name = b.get("name", "?")
                    tools[name] += 1
                    tool_input_chars += _txt_len(b.get("input", {}))
                    ak = hash(json.dumps(b.get("input", {}), sort_keys=True, default=str))
                    if b.get("id"):
                        tooluse_names[b["id"]] = name
                        tooluse_args[b["id"]] = ak
                        tooluse_paths[b["id"]] = _tool_path(b.get("input"))
                    lens.see_tool_use(name, b.get("input"), ak)
                elif bt in ("thinking", "text"):
                    body = b.get("thinking") if bt == "thinking" else b.get("text")
                    key = (mid, bt, hash(body if isinstance(body, str) else str(body)))
                    if dedup_blocks:
                        if key in seen_blocks:
                            continue
                        seen_blocks.add(key)
                    if bt == "thinking":
                        n_thinking += 1
                    else:
                        n_text += 1
            if r.get("interruptedMessageId") or msg.get("stop_reason") == "interrupted":
                key = (mid, "interrupted")
                if not dedup_blocks or key not in seen_blocks:
                    seen_blocks.add(key)
                    interrupted += 1
        elif t == "user":
            msg = r.get("message", {}) or {}
            content = msg.get("content")
            if isinstance(content, list):
                for b in content:
                    if isinstance(b, dict) and b.get("type") == "tool_result":
                        n_tool_result += 1
                        tool_result_chars += _txt_len(b.get("content", ""))
                        lens.see_tool_result(
                            tooluse_names.get(b.get("tool_use_id"), "?"),
                            bool(b.get("is_error")),
                            _txt_str(b.get("content", "")),
                            args_key=tooluse_args.get(b.get("tool_use_id")),
                            path=tooluse_paths.get(b.get("tool_use_id")))
            elif _looks_like_typed_prompt(content) and not r.get("isMeta"):
                prompts.append((ts, content.strip()[:400]))

    total_in, io_ratio, cache_hit, cc_share = _token_economics(tok)
    wall = _wall_seconds(ts_min, ts_max)
    ro = sum(v for k, v in tools.items() if k in READ_ONLY_TOOLS)
    behavior = lens.summary()
    behavior["stall_gaps"] = stall_gaps
    behavior["max_gap_s"] = round(max_gap_s, 1)
    return {
        "path": path, "session": os.path.splitext(os.path.basename(path))[0],
        "n_records": sum(rec_types.values()), "rec_types": dict(rec_types),
        "models": dict(models), "per_model": {m: dict(c) for m, c in per_model.items()},
        "assistant_turns": assistant_turns,
        "dup_assistant_lines": dup_assistant_lines,
        "n_prompts": len(prompts), "prompts": prompts,
        "n_tool_use": n_tool_use, "n_tool_result": n_tool_result,
        "tools": dict(tools), "read_only_tool_calls": ro,
        "read_only_frac": (ro / n_tool_use) if n_tool_use else None,
        "tool_input_chars": tool_input_chars, "tool_result_chars": tool_result_chars,
        "n_thinking": n_thinking, "n_text": n_text, "interrupted": interrupted,
        "tokens": tok, "total_input_tokens": total_in,
        "io_ratio": io_ratio, "cache_hit_frac": cache_hit, "cc_share": cc_share,
        "cost_usd": cost, "ts_min": ts_min, "ts_max": ts_max, "wall_s": wall,
        "behavior": behavior,
        "hooks": {"outcomes": {"|".join(k): v for k, v in hook_outcomes.items()},
                  "durations_ms": {k: v for k, v in hook_durations.items()},
                  "failures": [{"event": k[0], "outcome": k[1], "signature": k[2], "count": n}
                               for k, n in hook_failures.most_common()]},
    }

def _pct(xs, p):
    xs = sorted(x for x in xs if x is not None)
    if not xs:
        return None
    k = max(0, min(len(xs)-1, int(round((p/100)*(len(xs)-1)))))
    return xs[k]

def _ns_of(s):
    """The namespace an analyzed transcript belongs to. A subagent transcript sits in
    a nested <parent-session>/… directory, so basename(dirname(path)) is NOT its
    namespace — prefer the ns `discover` recorded on the record (#3226)."""
    return s.get("ns") or os.path.basename(os.path.dirname(s["path"]))

def _attributed_session(s):
    """Which TOP-LEVEL session a finding belongs to. A subagent's turns (cost and
    every behavioral detector) are attributed to the parent session that spawned
    it, so a fan-out stuck in a retry loop surfaces against a session an operator
    can actually open (#3226). A top-level transcript is its own parent."""
    return s.get("parent_session") or s["session"]

def aggregate(sessions):
    S = [s for s in sessions if "error" not in s]
    tot = collections.Counter()
    for s in S:
        for k, v in s["tokens"].items():
            tot[k] += v
    tot_cost = sum(s["cost_usd"] for s in S)
    tot_tools = collections.Counter()
    for s in S:
        tot_tools.update(s["tools"])
    ns_roll = collections.defaultdict(lambda: collections.Counter())
    ns_cost = collections.Counter()
    ns_models = collections.defaultdict(collections.Counter)   # ns -> {model: output tok}
    for s in S:
        ns = _ns_of(s)
        ns_roll[ns]["sessions"] += 1
        ns_roll[ns]["output"] += s["tokens"]["output"]
        ns_roll[ns]["cache_read"] += s["tokens"]["cache_read"]
        ns_roll[ns]["tool_use"] += s["n_tool_use"]
        ns_cost[ns] += s["cost_usd"]
        for model, c in s.get("per_model", {}).items():
            ns_models[ns][model] += c.get("output", 0)
    # per-model and per-billing-bucket rollups (token-exact; cost added at render)
    pm_roll = collections.defaultdict(collections.Counter)
    for s in S:
        for model, c in s.get("per_model", {}).items():
            pm_roll[model].update(c)
    bucket_roll = collections.defaultdict(collections.Counter)
    for model, c in pm_roll.items():
        bucket_roll[provider_bucket(model)].update(c)
    tier_roll = collections.defaultdict(collections.Counter)
    for model, c in pm_roll.items():
        tier_roll[model_tier(model)].update(c)
    ns_opus_share = {}
    for ns, models in ns_models.items():
        out = sum(models.values())
        opus = sum(v for m, v in models.items() if model_tier(m) == "opus")
        ns_opus_share[ns] = (opus / out) if out else None
    calls = [s["n_tool_use"] for s in S]
    outs  = [s["tokens"]["output"] for s in S]
    ios   = [s["io_ratio"] for s in S if s["io_ratio"]]
    chf   = [s["cache_hit_frac"] for s in S if s["cache_hit_frac"] is not None]
    ccs   = [s["cc_share"] for s in S if s.get("cc_share") is not None]   # #3069
    rof   = [s["read_only_frac"] for s in S if s["read_only_frac"] is not None]
    # behavioral rollup (#2365) — tolerate sessions replayed from pre-lens JSON
    beh_errors = collections.Counter()
    beh_churn = collections.Counter()
    beh_not_read = collections.Counter()   # #2375 d1: not_read sub-classes
    beh_shell_err = collections.Counter()  # #2365 finding 2: shell err cause breakdown
    beh_timeouts = beh_sleeps = beh_hangs = 0
    beh_suffix_resets = 0                    # #3069: mid-session cache-suffix invalidations
    beh_reset_floors = collections.Counter() # #3069: value snapped-TO -> times (machine-wide mode)
    burst_rows = []                          # #3069: long sessions carrying a cache-CREATE burst
    stall_sessions = 0
    max_gap_s = 0.0
    repeat_rows, filechurn_rows, mass_rows, successloop_rows = [], [], [], []
    never_read_rows = []   # #3942: sessions with a genuine never-read defect
    hook_outcomes = collections.Counter()
    hook_durations = collections.defaultdict(list)
    hook_failures = collections.Counter()
    hook_failure_sessions = collections.defaultdict(set)
    for s in S:
        b = s.get("behavior") or {}
        h = s.get("hooks") or {}
        hook_outcomes.update(h.get("outcomes") or {})
        for event, values in (h.get("durations_ms") or {}).items(): hook_durations[event].extend(values)
        for failure in h.get("failures") or []:
            key = (failure.get("event", "unknown"), failure.get("outcome", "unknown"), failure.get("signature", ""))
            hook_failures[key] += failure.get("count", 0)
            hook_failure_sessions[key].add(_attributed_session(s))
        beh_errors.update(b.get("tool_errors", {}))
        beh_churn.update(b.get("edit_churn", {}))
        beh_not_read.update(b.get("not_read_classes", {}))
        beh_shell_err.update(b.get("shell_error_classes", {}))
        beh_timeouts += b.get("timeout_kills", 0)
        beh_hangs += b.get("interactive_hangs", 0)
        beh_sleeps += b.get("sleep_polls", 0)
        beh_suffix_resets += b.get("suffix_resets", 0)
        beh_reset_floors.update(b.get("suffix_reset_floors", {}))
        # #3069 burst offender: a LONG session that snapped its cached suffix back
        # to a floor ≥1 time mid-run — a cache-CREATE thrash the heaviest-by-output
        # table never surfaces (the re-created suffix inflates read-share instead).
        if s.get("assistant_turns", 0) >= BURST_LONG_SESSION_MIN \
                and b.get("suffix_resets", 0) >= 1:
            burst_rows.append({
                "session": _attributed_session(s),
                "ns": _ns_of(s),
                "turns": s.get("assistant_turns", 0),
                "cache_create": s["tokens"]["cache_create"],
                "cc_share": s.get("cc_share"),
                "suffix_resets": b.get("suffix_resets", 0),
                "reset_floor": b.get("suffix_reset_floor"),
                "cost_usd": s["cost_usd"],
            })
        if b.get("stall_gaps", 0):
            stall_sessions += 1
        max_gap_s = max(max_gap_s, b.get("max_gap_s", 0) or 0)
        # #3226 — a subagent's findings are labelled with the PARENT session that
        # spawned it, so the offender tables always name a session an operator can open.
        ns, sid = _ns_of(s), _attributed_session(s)
        for r in (b.get("repeat_failures") or [])[:1]:
            repeat_rows.append({"session": sid, "ns": ns, **r})
        for r in (b.get("failure_mass") or [])[:1]:
            mass_rows.append({"session": sid, "ns": ns, **r})
        for r in (b.get("file_churn") or [])[:1]:
            filechurn_rows.append({"session": sid, "ns": ns, **r})
        for r in (b.get("success_loops") or [])[:1]:
            successloop_rows.append({"session": sid, "ns": ns, **r})
        # #3942 — surface the genuine never-read defect per session with the
        # offending file(s), so the count is actionable, not just a total.
        tnr = (b.get("not_read_classes") or {}).get("true_never_read", 0)
        if tnr:
            never_read_rows.append({
                "session": sid, "ns": ns, "count": tnr,
                "paths": [p.get("path", "") for p in
                          (b.get("true_never_read_paths") or [])[:3]]})
    per_tool_beh = {t: {"calls": tot_tools.get(t, 0),
                        "errors": beh_errors.get(t, 0),
                        "error_rate": (beh_errors.get(t, 0) / tot_tools[t])
                                      if tot_tools.get(t) else None}
                    for t in set(tot_tools) | set(beh_errors)}
    # Honest shell error rate (#2365 finding 2): the raw per-tool rate conflates guard
    # refusals and turn-deadline hangs — neither the shell nor the agent's command
    # failing — with genuine command failures. "genuine" strips policy_refusal (the
    # capability floor doing its job) and interactive_hang (now fixed at the guard) so
    # the reported rate reflects errors a shell change could actually move.
    shell_calls = sum(tot_tools.get(t, 0) for t in SHELL_TOOLS)
    shell_err_raw = sum(beh_shell_err.values())
    shell_err_discounted = beh_shell_err.get("policy_refusal", 0) \
        + beh_shell_err.get("interactive_hang", 0)
    shell_err_genuine = shell_err_raw - shell_err_discounted
    shell_errors = {
        "shell_calls": shell_calls,
        "classes": dict(beh_shell_err.most_common()),
        "raw_errors": shell_err_raw,
        "raw_rate": round(shell_err_raw / shell_calls, 3) if shell_calls else None,
        "genuine_errors": shell_err_genuine,
        "genuine_rate": round(shell_err_genuine / shell_calls, 3) if shell_calls else None,
    }
    hook_events = {}
    for event in sorted(set(hook_durations) | {k.split("|", 1)[0] for k in hook_outcomes}):
        values = hook_durations.get(event, [])
        outcomes = {k.split("|", 1)[1]: n for k, n in hook_outcomes.items() if k.startswith(event + "|")}
        hook_events[event] = {"outcomes": outcomes, "total": sum(outcomes.values()),
                              "duration_ms": {"median": _pct(values, 50), "p90": _pct(values, 90),
                                              "max": max(values) if values else None}}
    hooks = {"events": hook_events,
             "failures": [{"event": k[0], "outcome": k[1], "signature": k[2], "count": n,
                           "sessions": len(hook_failure_sessions[k])} for k, n in hook_failures.most_common(15)],
             "failure_total": sum(n for k, n in hook_outcomes.items()
                                  if k.split("|", 1)[-1] in {"non_blocking_error", "cancelled"})}

    behavior = {
        "per_tool": per_tool_beh,
        "timeout_kills": beh_timeouts,
        "interactive_hangs": beh_hangs,
        "shell_errors": shell_errors,
        "sleep_polls": beh_sleeps,
        "edit_churn": dict(beh_churn),
        "not_read_classes": dict(beh_not_read),
        "wasted_mutation_calls": sum(beh_churn.values()),
        "stall_sessions": stall_sessions,
        "max_gap_s": round(max_gap_s, 1),
        "repeat_failure_sessions": sorted(repeat_rows, key=lambda r: -r["count"])[:10],
        "failure_mass_sessions": sorted(mass_rows, key=lambda r: -r["count"])[:10],
        "file_churn_sessions": sorted(filechurn_rows, key=lambda r: -r["count"])[:10],
        "success_loop_sessions": sorted(successloop_rows, key=lambda r: -r["count"])[:10],
        "never_read_sessions": sorted(never_read_rows, key=lambda r: -r["count"])[:10],
        # #3069 suffix-cache reset burst
        "suffix_resets": beh_suffix_resets,
        "suffix_reset_floor": (beh_reset_floors.most_common(1)[0][0]
                               if beh_reset_floors else None),
        "burst_sessions": sorted(burst_rows, key=lambda r: -r["cache_create"])[:10],
    }
    return {
        "n_sessions": len(S), "totals": dict(tot), "total_cost_usd": tot_cost,
        "behavior": behavior,
        "hooks": hooks,
        "tool_mix": dict(tot_tools.most_common()),
        "per_namespace": {k: dict(v) for k, v in ns_roll.items()},
        "per_namespace_cost": dict(ns_cost),
        "per_namespace_top_model": {k: (v.most_common(1)[0][0] if v else "?")
                                    for k, v in ns_models.items()},
        "per_namespace_opus_share": ns_opus_share,
        "per_model": {m: dict(c) for m, c in pm_roll.items()},
        "per_bucket": {b: dict(c) for b, c in bucket_roll.items()},
        "per_tier": {t: dict(c) for t, c in tier_roll.items()},
        "dist": {
            "calls_per_session": {"median": statistics.median(calls) if calls else None,
                                  "mean": round(statistics.mean(calls),1) if calls else None,
                                  "p90": _pct(calls,90), "max": max(calls) if calls else None},
            "output_tokens_per_session": {"median": statistics.median(outs) if outs else None,
                                  "p90": _pct(outs,90), "max": max(outs) if outs else None},
            "io_ratio": {"median": round(statistics.median(ios),1) if ios else None,
                         "p90": round(_pct(ios,90),1) if ios else None},
            "cache_hit_frac": {"median": round(statistics.median(chf),3) if chf else None,
                               "p10": round(_pct(chf,10),3) if chf else None,
                               "p90": round(_pct(chf,90),3) if chf else None},
            "cc_share": {"median": round(statistics.median(ccs),3) if ccs else None,
                         "p90": round(_pct(ccs,90),3) if ccs else None,
                         "max": round(max(ccs),3) if ccs else None},
            "read_only_frac": {"median": round(statistics.median(rof),3) if rof else None},
        },
    }

def fmt_int(n):
    return f"{n:,}"

def fmt_pct(frac):
    return "—" if frac is None else f"{frac*100:.1f}%"

def _namespace_name(path):
    return os.path.basename(os.path.dirname(path))

def _count_kinds(sessions):
    """(top-level, subagent) transcript counts for a list of analyzed sessions."""
    subs = sum(1 for s in sessions if s.get("kind") == "subagent")
    return len(sessions) - subs, subs

def _scope_line(sessions, ns_prefix, since_days, top_level_only, max_sessions):
    namespaces = sorted({_ns_of(s) for s in sessions if "error" not in s})
    if len(namespaces) > 8:
        ns_desc = ", ".join(namespaces[:8]) + f", ... (+{len(namespaces)-8} more)"
    else:
        ns_desc = ", ".join(namespaces) if namespaces else "none"
    ns_filter = ns_prefix or "all non-excluded namespaces"
    window = "all-time" if since_days is None else f"last {since_days:g} days"
    # #3226 — subagent/workflow transcripts are folded into every total BY DEFAULT;
    # --top-level-only restores the pre-#3226 view (and prints what it is hiding).
    if top_level_only:
        kinds = ("top-level session transcripts ONLY (`--top-level-only`; "
                 "subagent/workflow spend excluded — see the NOTE)")
    else:
        kinds = ("session transcripts + subagent/workflow transcripts "
                 "(folded into every total below, attributed to their parent session)")
    cap = f"; max top-level sessions: {max_sessions}" if max_sessions else ""
    return (f"{len(namespaces)} namespaces folded ({ns_desc}); "
            f"namespace filter: {ns_filter}; time window: {window}; {kinds}{cap}")

def _subagent_note(summary):
    """What a `--top-level-only` run is hiding: the subagent spend it left out (#3226)."""
    if not summary or not summary.get("count"):
        return None
    tokens = summary.get("tokens", {})
    return (f"NOTE: +{summary['count']} subagent transcripts uncounted; "
            f"drop `--top-level-only` to fold them in "
            f"(about +${summary.get('cost_usd', 0.0):,.2f} / "
            f"+{fmt_int(tokens.get('output', 0))} output tok).")

def _report_header(S, agg, ns_prefix, since_days, top_level_only, max_sessions,
                   excluded_subagents):
    """Title, generation stamp and the one-line statement of what was in scope."""
    L = []
    n_top, n_sub = _count_kinds([s for s in S if "error" not in s])
    audited = (f"{n_top} top-level + {n_sub} subagent = {agg['n_sessions']}"
               if n_sub else f"{agg['n_sessions']}")
    L.append("# Session-Transcript Audit — active scope\n")
    L.append(f"**Generated:** {datetime.datetime.now().isoformat(timespec='seconds')}  ")
    L.append(f"**Transcripts audited:** {audited}  ·  **Tool:** `tools/session_audit.py` (re-runnable)  ")
    L.append(f"**Scope:** {_scope_line(S, ns_prefix, since_days, top_level_only, max_sessions)}")
    note = _subagent_note(excluded_subagents)
    if note:
        L.append(note)
    return L

def _scope_totals(agg):
    """EXACT token totals for the scope, with the cache read/create split and cost."""
    L = []
    t = agg["totals"]
    L.append("## Scope totals (EXACT token counts)\n")
    L.append(f"- **Output tokens (the actual work generated):** {fmt_int(t['output'])}")
    L.append(f"- **Fresh input tokens (billed, non-cached):** {fmt_int(t['input'])}")
    L.append(f"- **Cache-read tokens (prompt-cache / KV reuse):** {fmt_int(t['cache_read'])}")
    L.append(f"- **Cache-creation tokens:** {fmt_int(t['cache_create'])}")
    tot_in = t['input']+t['cache_read']+t['cache_create']
    L.append(f"- **Total context ingested:** {fmt_int(tot_in)}  →  **machine-wide I:O ratio = {tot_in/max(t['output'],1):.1f} : 1**")
    chf = t['cache_read']/max(tot_in,1)
    L.append(f"- **Cache-read share of all ingested context = {chf*100:.1f}%**  (this is the prompt-cache/KV reuse the harness ALREADY captures)")
    # #3069: the burst counterpart to the flattering read-share above. Cache-CREATE
    # tokens are the cached prefix RE-written when a suffix is invalidated mid-session;
    # they bill at write rates and the read-share line hides them (a burst inflates the
    # NEXT turn's read, so a thrashing session reads as MORE cached, not less).
    ccs = t['cache_create']/max(tot_in,1)
    cr_ratio = t['cache_create']/max(t['cache_read'],1)
    L.append(f"- **Cache-CREATE burst share of all ingested context = {ccs*100:.1f}%**  ·  **create:read ratio = {cr_ratio:.3f}**  (context RE-written into the provider cache — billed at write rates, NOT captured by the read-share above)")
    # Two DIFFERENT mechanisms reach the web — report BOTH so the line can never
    # appear to contradict the tool-mix table below (which lists the CLIENT tools):
    #   - server_tool_use: the model's built-in web_search/web_fetch (billed server-side)
    #   - the client WebSearch/WebFetch tools (tool_use blocks — these are what show
    #     up in the tool mix). Counting only the former printed "0 / 0" even when a
    #     session used the client WebFetch tool, which read as "no web activity".
    ws_c = agg["tool_mix"].get("WebSearch", 0)
    wf_c = agg["tool_mix"].get("WebFetch", 0)
    L.append(f"- **Web requests — server-tool (`server_tool_use`, billed):** "
             f"search {fmt_int(t['web_search'])} / fetch {fmt_int(t['web_fetch'])}  "
             f"·  **client tool:** WebSearch {fmt_int(ws_c)} / WebFetch {fmt_int(wf_c)}")
    L.append(f"- **Multi-iteration count:** {fmt_int(t['iterations'])}")
    flag = "  _(⚠ cost uses an ASSUMED price table — edit PRICING; token counts above are exact)_" if PRICING_IS_ASSUMPTION else ""
    L.append(f"- **Estimated Anthropic-billed cost:** ${agg['total_cost_usd']:,.2f}{flag}")
    # Surface other billing buckets so the Anthropic total is never read as "the whole bill".
    buckets = agg.get("per_bucket", {})
    other = {b: c for b, c in buckets.items()
             if b not in ("Anthropic (Claude)", "non-billed (harness)") and c.get("output", 0)}
    if other:
        parts = [f"{b} ({fmt_int(c['output'])} output tok, unpriced — add its card)"
                 for b, c in sorted(other.items(), key=lambda kv: -kv[1].get("output", 0))]
        L.append("- **⚠ Other billing buckets present (NOT in the total above — different invoices):** "
                 + "; ".join(parts))
    nb = buckets.get("non-billed (harness)", {})
    if nb.get("turns"):
        L.append(f"- **Non-billed `<synthetic>` turns (harness-injected, $0):** {fmt_int(nb.get('turns',0))} "
                 f"({fmt_int(nb.get('output',0))} output tok)")
    L.append("")
    return L

def _model_mix_table(agg):
    """The headline mix KPI that makes "opus-heavy vs haiku-heavy" explicit: per-tier share."""
    L = []
    L.append("## Model-mix KPI (tier shares)\n")
    L.append("| Tier | Output tok | Output share | Est. cost | Cost share |")
    L.append("|---|---:|---:|---:|---:|")
    total_output = sum(c.get("output", 0) for c in agg.get("per_tier", {}).values())
    total_priced_cost = sum(model_cost(m, c) for m, c in agg.get("per_model", {}).items())
    for tier, c in sorted(agg.get("per_tier", {}).items(),
                          key=lambda kv: -kv[1].get("output", 0)):
        tier_cost = sum(model_cost(m, mc) for m, mc in agg.get("per_model", {}).items()
                        if model_tier(m) == tier)
        out_share = (c.get("output", 0) / total_output) if total_output else None
        cost_share = (tier_cost / total_priced_cost) if total_priced_cost else None
        L.append(f"| {tier} | {fmt_int(c.get('output',0))} | {fmt_pct(out_share)} | "
                 f"${tier_cost:,.2f} | {fmt_pct(cost_share)} |")
    L.append("")
    return L

def _billing_bucket_table(agg):
    """Per billing bucket — the answer to "is this Claude or Gemini money?". NEVER summed."""
    L = []
    buckets = agg.get("per_bucket", {})
    L.append("## Cost by billing bucket (provider) — never sum across these\n")
    L.append("| Billing bucket | Turns | Output tok | Cache-read tok | Est. cost | Priced? |")
    L.append("|---|---:|---:|---:|---:|:--:|")
    for b, c in sorted(buckets.items(), key=lambda kv: -kv[1].get("output", 0)):
        bcost = sum(model_cost(m, mc) for m, mc in agg.get("per_model", {}).items()
                    if provider_bucket(m) == b)
        priced = b == "Anthropic (Claude)"
        cost_cell = f"${bcost:,.2f}" if priced else ("$0.00" if b == "non-billed (harness)" else "— (no card)")
        L.append(f"| {b} | {fmt_int(c.get('turns',0))} | {fmt_int(c.get('output',0))} | "
                 f"{fmt_int(c.get('cache_read',0))} | {cost_cell} | {'✓' if priced else ''} |")
    L.append("")
    return L

def _per_model_table(agg):
    """Per-model tiers — so a blended cost can be read as opus-heavy vs haiku-heavy."""
    L = []
    L.append("## Per-model breakdown (token-exact; cost Anthropic-assumed)\n")
    L.append("| Model | Bucket | Turns | Output tok | Cache-read tok | Est. cost |")
    L.append("|---|---|---:|---:|---:|---:|")
    for m, c in sorted(agg.get("per_model", {}).items(), key=lambda kv: -kv[1].get("output", 0)):
        mc = model_cost(m, c)
        cost_cell = f"${mc:,.2f}" if price_for(m) is not None else ("$0.00" if m in NONBILLED_MODELS else "— (no card)")
        L.append(f"| {m} | {provider_bucket(m)} | {fmt_int(c.get('turns',0))} | {fmt_int(c.get('output',0))} | "
                 f"{fmt_int(c.get('cache_read',0))} | {cost_cell} |")
    L.append("")
    return L

def _namespace_rollup(agg):
    """Sessions, output, opus share, cache reads, tool calls and cost per namespace."""
    L = []
    L.append("## Per-namespace rollup\n")
    L.append("| Namespace | Sessions | Output tok | Opus output share | Cache-read tok | Tool calls | Top model (by output) | Est. cost |")
    L.append("|---|---:|---:|---:|---:|---:|---|---:|")
    top_model = agg.get("per_namespace_top_model", {})
    opus_share = agg.get("per_namespace_opus_share", {})
    for ns, v in sorted(agg["per_namespace"].items(), key=lambda kv: -kv[1]["output"]):
        L.append(f"| {ns} | {v['sessions']} | {fmt_int(v['output'])} | {fmt_pct(opus_share.get(ns))} | "
                 f"{fmt_int(v['cache_read'])} | "
                 f"{fmt_int(v['tool_use'])} | {top_model.get(ns, '?')} | ${agg['per_namespace_cost'][ns]:,.2f} |")
    L.append("")
    return L

def _distributions(agg):
    """Per-session spread of tool calls, output, I:O ratio and cache-hit fraction."""
    L = []
    d = agg["dist"]
    L.append("## Distributions (per session)\n")
    L.append(f"- **Tool calls/session:** median {d['calls_per_session']['median']}, "
             f"mean {d['calls_per_session']['mean']}, p90 {d['calls_per_session']['p90']}, max {d['calls_per_session']['max']}")
    L.append(f"- **Output tokens/session:** median {fmt_int(d['output_tokens_per_session']['median'] or 0)}, "
             f"p90 {fmt_int(d['output_tokens_per_session']['p90'] or 0)}, max {fmt_int(d['output_tokens_per_session']['max'] or 0)}")
    L.append(f"- **I:O ratio/session:** median {d['io_ratio']['median']}, p90 {d['io_ratio']['p90']}")
    L.append(f"- **Cache-hit fraction/session:** median {d['cache_hit_frac']['median']}, "
             f"p10 {d['cache_hit_frac']['p10']}, p90 {d['cache_hit_frac']['p90']}")
    L.append(f"- **Read-only tool fraction/session:** median {d['read_only_frac']['median']}\n")
    return L

def _tool_mix_table(agg):
    """The 25 most-used client tools and whether each one is read-only."""
    L = []
    L.append("## Global tool mix\n")
    L.append("| Tool | Calls | Read-only? |")
    L.append("|---|---:|:--:|")
    for name, n in list(agg["tool_mix"].items())[:25]:
        L.append(f"| {name} | {fmt_int(n)} | {'✓' if name in READ_ONLY_TOOLS else ''} |")
    L.append("")
    return L

def _tool_error_table(beh):
    """Call and error counts for the busiest tools, plus every tool that errored at all."""
    L = []
    L.append("## Behavioral lens — stuck/churn detectors\n")
    pt = beh.get("per_tool", {})
    with_calls = sorted((t for t in pt if pt[t]["calls"]),
                        key=lambda t: -pt[t]["calls"])
    show = with_calls[:12] + [t for t in pt
                              if pt[t]["errors"] and t not in with_calls[:12]]
    L.append("| Tool | Calls | Errors | Error rate |")
    L.append("|---|---:|---:|---:|")
    for t in show:
        v = pt[t]
        L.append(f"| {t} | {fmt_int(v['calls'])} | {fmt_int(v['errors'])} | "
                 f"{fmt_pct(v['error_rate'])} |")
    return L

def _churn_counters(beh):
    """Timeout kills, no-TTY hangs, shell error rate, sleep polls and edit/write churn."""
    L = []
    churn = beh.get("edit_churn", {})
    L.append("")
    L.append(f"- **Timeout kills (shell result matched exit-143 / \"timed out\"):** "
             f"{fmt_int(beh.get('timeout_kills', 0))}")
    # #2365 d3 — the editor/pager-no-TTY wedge, keyed on the repo-guard's exact
    # INTERACTIVE_HANG emission (fixed at the guard for headless children).
    L.append(f"- **Interactive-editor/pager hangs (no-TTY wedge, INTERACTIVE_HANG):** "
             f"{fmt_int(beh.get('interactive_hangs', 0))}")
    # #2365 finding 2 — the raw shell error rate is a mix; the genuine rate strips
    # guard refusals (the floor working) and now-fixed hangs.
    se = beh.get("shell_errors") or {}
    if se.get("shell_calls"):
        cls = se.get("classes", {})
        breakdown = " · ".join(f"{k} {v}" for k, v in cls.items()) or "—"
        L.append(f"- **Shell error rate (Bash+PowerShell):** raw "
                 f"{fmt_pct(se.get('raw_rate'))} ({fmt_int(se.get('raw_errors', 0))}"
                 f"/{fmt_int(se['shell_calls'])}) → **genuine "
                 f"{fmt_pct(se.get('genuine_rate'))}** "
                 f"({fmt_int(se.get('genuine_errors', 0))}, after dropping guard "
                 f"refusals + fixed hangs)")
        L.append(f"  - **by cause:** {breakdown}")
    L.append(f"- **Foreground sleep-polls (`sleep`/`Start-Sleep` command prefix):** "
             f"{fmt_int(beh.get('sleep_polls', 0))}")
    nrc = beh.get("not_read_classes", {})
    L.append(f"- **Edit/Write churn (wasted mutation calls):** "
             f"{fmt_int(beh.get('wasted_mutation_calls', 0))}  "
             f"(not-read {fmt_int(churn.get('not_read', 0))} · "
             f"stale-read {fmt_int(churn.get('stale_read', 0))})")
    # #2375 d1 — only true-never-read is agent misbehavior; the other two are
    # a --resume read-state reset and a guard-caught duplicate write.
    L.append(f"  - **not-read sub-classes:** post-resume "
             f"{fmt_int(nrc.get('post_resume', 0))} · self-duplicate "
             f"{fmt_int(nrc.get('self_duplicate', 0))} · **true-never-read "
             f"{fmt_int(nrc.get('true_never_read', 0))}** (the real defect)")
    # #3942 — name the offenders for the one sub-class that IS misbehavior,
    # so a reader can jump straight to the session + file that never got Read.
    nr = beh.get("never_read_sessions") or []
    if nr:
        L.append("")
        L.append("| Session | NS | × | Never-read file(s) |")
        L.append("|---|---|---:|---|")
        for r in nr:
            files = ", ".join(p for p in (r.get("paths") or []) if p) or "—"
            files = files.replace("|", "\\|")[:100]
            L.append(f"| {r['session'][:8]} | {r['ns']} | {r['count']} | {files} |")
    return L

def _loop_detector_tables(beh):
    """Stall gaps plus the four repeat-work loops: verbatim retry, failure class, file rewrite, successful call."""
    L = []
    L.append(f"- **Sessions with a ≥{STALL_GAP_S//60}-min zero-record stall "
             f"(harness/API dead time):** {fmt_int(beh.get('stall_sessions', 0))}"
             + (f"  (longest gap {beh.get('max_gap_s', 0)/60:.0f} min)"
                if beh.get("stall_sessions") else ""))
    rep = beh.get("repeat_failure_sessions") or []
    L.append(f"- **Sessions with a VERBATIM retry loop "
             f"(≥{REPEAT_FAILURE_MIN}× same tool+args+error):** {len(rep)}"
             + (" — worst below" if rep else ""))
    if rep:
        L.append("")
        L.append("| Session | NS | Tool | × | Failure signature |")
        L.append("|---|---|---|---:|---|")
        for r in rep:
            sig = r["sig"][:80].replace("|", "\\|")
            L.append(f"| {r['session'][:8]} | {r['ns']} | {r['tool']} | "
                     f"{r['count']} | {sig} |")
    mass = beh.get("failure_mass_sessions") or []
    L.append(f"- **Sessions with a recurring failure CLASS "
             f"(≥{REPEAT_FAILURE_MIN}× same tool+error, args vary):** {len(mass)}"
             + (" — worst below" if mass else ""))
    if mass:
        L.append("")
        L.append("| Session | NS | Tool | × | Failure class |")
        L.append("|---|---|---|---:|---|")
        for r in mass:
            sig = r["sig"][:80].replace("|", "\\|")
            L.append(f"| {r['session'][:8]} | {r['ns']} | {r['tool']} | "
                     f"{r['count']} | {sig} |")
    fc = beh.get("file_churn_sessions") or []
    L.append(f"- **Sessions with a REWRITE loop (≥{FILE_CHURN_MIN} mutations "
             f"of one file, revisiting the same regions or reverting amid "
             f"region reuse):** {len(fc)}"
             + (" — worst below" if fc else ""))
    if fc:
        L.append("")
        L.append("| Session | NS | × | Regions | Reverts | File |")
        L.append("|---|---|---:|---:|---:|---|")
        for r in fc:
            fp = r["file"].replace("|", "\\|")
            L.append(f"| {r['session'][:8]} | {r['ns']} | {r['count']} | "
                     f"{r.get('distinct_regions', '—')} | {r.get('reverts', '—')} | {fp} |")
    sl = beh.get("success_loop_sessions") or []
    L.append(f"- **Sessions with a SUCCESSFUL-call loop (≥{SUCCESS_LOOP_MIN}× "
             f"identical successful Read/Glob/Grep/shell call — read-loop / "
             f"glob-storm / output-poll):** {len(sl)}"
             + (" — worst below" if sl else ""))
    if sl:
        L.append("")
        L.append("| Session | NS | Tool | × | Target |")
        L.append("|---|---|---|---:|---|")
        for r in sl:
            tgt = (r.get("target") or "")[:80].replace("|", "\\|")
            L.append(f"| {r['session'][:8]} | {r['ns']} | {r['tool']} | "
                     f"{r['count']} | {tgt} |")
    return L

def _cache_burst_tables(beh):
    """Suffix-cache invalidations and the long sessions whose cache-CREATE bursts they drive."""
    L = []
    # #3069 — the suffix-cache reset burst: per-turn cache_read snapping back to a
    # floor is a mid-session cache-CREATE re-write the read-share lens hides.
    sr = beh.get("suffix_resets", 0)
    floor = beh.get("suffix_reset_floor")
    L.append(f"- **Suffix-cache invalidations (per-turn cache_read snap-back "
             f">{fmt_int(SUFFIX_RESET_DROP_MIN)} tok — a mid-session cache-CREATE "
             f"burst):** {fmt_int(sr)}"
             + (f"  (modal reset floor ≈ {fmt_int(floor)} tok)" if floor else ""))
    bs = beh.get("burst_sessions") or []
    L.append(f"- **High-burst long sessions (≥{BURST_LONG_SESSION_MIN} turns with "
             f"≥1 suffix-cache reset — the cache-CREATE thrash the heaviest-by-output "
             f"table hides):** {len(bs)}"
             + (" — worst below" if bs else ""))
    if bs:
        L.append("")
        L.append("| Session | NS | Turns | Cache-create tok | cc-share | Resets | Reset floor | Est.$ |")
        L.append("|---|---|---:|---:|---:|---:|---:|---:|---:|")
        for r in bs:
            fl = fmt_int(r["reset_floor"]) if r.get("reset_floor") is not None else "—"
            L.append(f"| {r['session'][:8]} | {r['ns']} | {r['turns']} | "
                     f"{fmt_int(r['cache_create'])} | {fmt_pct(r.get('cc_share'))} | "
                     f"{r['suffix_resets']} | {fl} | ${r['cost_usd']:.2f} |")
    return L

def _parse_iso_instant(value):
    """Parse an ISO-8601 instant, accepting the common trailing-Z spelling."""
    parsed = datetime.datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise ValueError("timestamp must include a timezone offset")
    return parsed.timestamp()


def load_dos_hook_observations(workspace, since_days, since_instant=None):
    """Read the independent DOS hook ledger when this audit runs in its workspace."""
    path = os.path.join(workspace, ".dos", "metrics", "observations.jsonl")
    if not os.path.isfile(path):
        return None
    if since_instant is not None:
        cutoff = datetime.datetime.fromtimestamp(
            _parse_iso_instant(since_instant), datetime.timezone.utc
        )
    else:
        cutoff = (datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(days=since_days)
                  if since_days is not None else None)
    verbs = collections.defaultdict(lambda: {"outcomes": collections.Counter(),
                                             "latencies_ms": [], "exits": collections.Counter()})
    malformed = 0
    timestamp_buckets = collections.Counter()
    try:
        with open(path, encoding="utf-8") as stream:
            for line in stream:
                try:
                    row = json.loads(line)
                    ts = datetime.datetime.fromisoformat(str(row["ts"]).replace("Z", "+00:00"))
                except (ValueError, TypeError, KeyError, json.JSONDecodeError):
                    malformed += 1
                    continue
                if cutoff is not None and ts < cutoff:
                    continue
                verb = str(row.get("verb") or "unknown")
                timestamp_buckets[(verb, str(row["ts"])[:19])] += 1
                verbs[verb]["outcomes"][str(row.get("outcome") or "<empty>")] += 1
                verbs[verb]["exits"][str(row.get("exit", "unknown"))] += 1
                latency = row.get("latency_ms")
                if isinstance(latency, (int, float)) and latency >= 0:
                    verbs[verb]["latencies_ms"].append(float(latency))
    except OSError as exc:
        return {"path": path, "error": str(exc)}
    result = {"path": path, "malformed_rows": malformed, "verbs": {},
              "since": cutoff.isoformat().replace("+00:00", "Z") if cutoff else None,
              "exact_since": since_instant is not None}
    for verb, data in verbs.items():
        values = data["latencies_ms"]
        result["verbs"][verb] = {
            "count": sum(data["outcomes"].values()),
            "outcomes": dict(data["outcomes"].most_common()),
            "exits": dict(data["exits"].most_common()),
            "duration_ms": {"median": _pct(values, 50), "p90": _pct(values, 90),
                            "p99": _pct(values, 99), "max": max(values) if values else None,
                            "over_100ms": sum(value > 100 for value in values),
                            "over_500ms": sum(value > 500 for value in values),
                            "over_1000ms": sum(value > 1000 for value in values)},
            "same_second": {
                "buckets": sum(1 for (v, _), n in timestamp_buckets.items() if v == verb and n > 1),
                "excess_calls": sum(n - 1 for (v, _), n in timestamp_buckets.items()
                                    if v == verb and n > 1),
                "max_calls": max((n for (v, _), n in timestamp_buckets.items() if v == verb),
                                 default=0),
            },
        }
    return result


def _dos_hook_lens(agg):
    ledger = agg.get("dos_hook_ledger")
    if not ledger:
        return []
    L = ["## DOS hook ledger — independent execution witness\n"]
    if ledger.get("error"):
        return L + [f"- Ledger unreadable: `{ledger['error']}`\n"]
    L += [f"- Source: `{ledger['path']}` (independent of Claude transcript attachments)."]
    if ledger.get("since"):
        L.append(f"- Observation window starts at `{ledger['since']}` (inclusive).")
    L += ["| Verb | Calls | Outcomes | Exit codes | p90 | p99 | Max | >100ms | >500ms | >1s |",
          "|---|---:|---|---|---:|---:|---:|---:|---:|---:|"]
    for verb, row in sorted((ledger.get("verbs") or {}).items()):
        outcomes = ", ".join(f"{k}={v:,}" for k, v in row.get("outcomes", {}).items())
        exits = ", ".join(f"{k}={v:,}" for k, v in row.get("exits", {}).items())
        dur = row.get("duration_ms") or {}
        ms = lambda value: "—" if value is None else f"{value:,.1f} ms"
        L.append(f"| {verb} | {row.get('count', 0):,} | {outcomes} | {exits} | "
                 f"{ms(dur.get('p90'))} | {ms(dur.get('p99'))} | {ms(dur.get('max'))} | "
                 f"{dur.get('over_100ms', 0)} | {dur.get('over_500ms', 0)} | "
                 f"{dur.get('over_1000ms', 0)} |")
    post = (ledger.get("verbs") or {}).get("posttool")
    transcript_post = (agg.get("hooks", {}).get("events", {}).get("PostToolUse", {})
                       .get("total", 0))
    if post and not transcript_post:
        L.append(f"\n- **PostToolUse reconciliation:** Claude recorded zero outcome attachments, while "
                 f"the DOS ledger independently witnessed {post['count']:,} `posttool` calls. "
                 "This is an attachment-observability gap, not evidence that the hook did not run.")
    if post:
        same_second = post.get("same_second") or {}
        pressure = (f"{same_second.get('excess_calls', 0):,} rows are additional calls in an "
                    f"already-occupied one-second bucket (max {same_second.get('max_calls', 0):,}/s).")
        if ledger.get("exact_since"):
            L.append(f"- **PostToolUse exact-window pressure:** {post['count']:,} ledger rows; "
                     f"{pressure} Amplification ratio omitted because the transcript and exact "
                     "ledger lower bounds differ.")
        else:
            audited_calls = sum((agg.get("tool_mix") or {}).values())
            ratio = post["count"] / audited_calls if audited_calls else None
            ratio_text = "—" if ratio is None else f"{ratio:.2f}x"
            L.append(f"- **PostToolUse amplification signal:** {post['count']:,} ledger rows / "
                     f"{audited_calls:,} audited transcript tool calls = {ratio_text}; {pressure} "
                     "The sources have different coverage, so this is a collision/duplication lead, "
                     "not a one-to-one duplicate count.")
    if ledger.get("malformed_rows"):
        L.append(f"- Malformed ledger rows skipped: {ledger['malformed_rows']:,}")
    return L + [""]


def _stream_timestamp(row):
    value = row.get("_captured_at") or row.get("timestamp") or row.get("ts")
    if not value:
        return None
    try:
        return _parse_iso_instant(str(value))
    except (ValueError, TypeError):
        return None


def load_hook_streams(paths):
    """Pair persisted Claude stream-json hook lifecycle rows by hook ID."""
    if not paths:
        return None
    started = {}
    events = []
    malformed = 0
    unmatched_responses = 0
    for path in paths:
        try:
            stream = open(path, encoding="utf-8")
        except OSError as exc:
            return {"paths": paths, "error": f"{path}: {exc}"}
        with stream:
            for line in stream:
                try:
                    row = json.loads(line)
                except (ValueError, TypeError):
                    malformed += 1
                    continue
                subtype = row.get("subtype")
                hook_id = row.get("hook_id")
                if subtype == "hook_started" and hook_id:
                    started[hook_id] = row
                    continue
                if subtype != "hook_response" or not hook_id:
                    continue
                begin = started.pop(hook_id, None)
                if begin is None:
                    unmatched_responses += 1
                event = row.get("hook_event") or (begin or {}).get("hook_event") or "Unknown"
                outcome = str(row.get("outcome") or "unknown").lower()
                start_ts = _stream_timestamp(begin or {})
                end_ts = _stream_timestamp(row)
                duration_ms = ((end_ts - start_ts) * 1000
                               if start_ts is not None and end_ts is not None else None)
                events.append({
                    "hook_id": hook_id,
                    "session_id": row.get("session_id") or (begin or {}).get("session_id"),
                    "event": event,
                    "outcome": outcome,
                    "exit_code": row.get("exit_code"),
                    "duration_ms": duration_ms,
                    "stderr": row.get("stderr") or "",
                    "source": path,
                })
    return {"paths": paths, "events": events, "malformed_rows": malformed,
            "unmatched_starts": len(started), "unmatched_responses": unmatched_responses}


def _reconcile_hook_stream(agg, stream):
    if not stream or stream.get("error"):
        return stream
    transcript = collections.Counter()
    for session in agg.get("_sessions", []):
        session_id = session.get("session")
        for key, count in (session.get("hooks", {}).get("outcomes") or {}).items():
            event, _, outcome = key.partition("|")
            transcript[(session_id, event, outcome)] += count
    kept = []
    suppressed = 0
    for row in stream.get("events", []):
        key = (row.get("session_id"), row.get("event"), row.get("outcome"))
        if transcript[key]:
            transcript[key] -= 1
            suppressed += 1
        else:
            kept.append(row)
    result = dict(stream)
    result["events"] = kept
    result["suppressed_transcript_overlaps"] = suppressed
    return result


def _hook_stream_lens(stream):
    if not stream:
        return []
    L = ["## Captured Claude hook stream — persisted lifecycle witness\n"]
    if stream.get("error"):
        return L + [f"- Stream unreadable: `{stream['error']}`\n"]
    events = stream.get("events") or []
    grouped = collections.defaultdict(list)
    for row in events:
        grouped[row["event"]].append(row)
    L += [f"- Sources: {', '.join(f'`{p}`' for p in stream.get('paths', []))}.",
          "| Event | Responses | Outcomes | Exit codes | p90 | Max |",
          "|---|---:|---|---|---:|---:|"]
    for event, rows in sorted(grouped.items()):
        outcomes = collections.Counter(row["outcome"] for row in rows)
        exits = collections.Counter(str(row["exit_code"]) for row in rows)
        durations = [row["duration_ms"] for row in rows if row.get("duration_ms") is not None]
        fmt = lambda value: "—" if value is None else f"{value:,.1f} ms"
        L.append(f"| {event} | {len(rows):,} | "
                 f"{', '.join(f'{k}={v}' for k, v in outcomes.items())} | "
                 f"{', '.join(f'{k}={v}' for k, v in exits.items())} | "
                 f"{fmt(_pct(durations, 90))} | {fmt(max(durations) if durations else None)} |")
    L.append(f"- Transcript overlaps suppressed: {stream.get('suppressed_transcript_overlaps', 0):,}.")
    if stream.get("unmatched_starts") or stream.get("unmatched_responses"):
        L.append(f"- Unmatched lifecycle rows: starts {stream.get('unmatched_starts', 0):,}, "
                 f"responses {stream.get('unmatched_responses', 0):,}.")
    if stream.get("malformed_rows"):
        L.append(f"- Malformed stream rows skipped: {stream['malformed_rows']:,}.")
    return L + [""]


def _hook_lens(agg):
    hooks = agg.get("hooks") or {}
    events = hooks.get("events") or {}
    L = ["## Hook execution lens — outcomes and latency\n"]
    if not events: return L + ["- No transcript-native hook outcome attachments were observed.\n"]
    L += ["| Event | Records | Success | Context | Non-blocking error | Cancelled | p90 | Max |",
          "|---|---:|---:|---:|---:|---:|---:|"]
    for event, row in sorted(events.items()):
        outcomes = row.get("outcomes") or {}
        dur = row.get("duration_ms") or {}
        ms = lambda v: "—" if v is None else f"{v:,} ms"
        L.append(f"| {event} | {row.get('total', 0):,} | {outcomes.get('success', 0):,} | {outcomes.get('additional_context', 0):,} | {outcomes.get('non_blocking_error', 0):,} | {outcomes.get('cancelled', 0):,} | {ms(dur.get('p90'))} | {ms(dur.get('max'))} |")
    L += ["", f"- **Hook failures/cancellations:** {hooks.get('failure_total', 0):,}"]
    if hooks.get("failures"):
        L += ["", "| Event | Outcome | × | Sessions | Failure signature |", "|---|---|---:|---:|---|"]
        for row in hooks["failures"][:10]:
            sig = str(row.get("signature", "")).replace("|", "\\|")
            L.append(f"| {row.get('event')} | {row.get('outcome')} | {row.get('count', 0):,} | {row.get('sessions', 0):,} | {sig} |")
    return L + [""]


def _behavior_lens(agg):
    """The stuck/churn half of the picture the token lens cannot see (#2365)."""
    beh = agg.get("behavior") or {}
    if not beh:
        return []
    L = []
    L.extend(_tool_error_table(beh))
    L.extend(_churn_counters(beh))
    L.extend(_loop_detector_tables(beh))
    L.extend(_cache_burst_tables(beh))
    L.append("")
    return L

def _transcript_label(s):
    """Row label for one transcript. A subagent row is rendered as
    `<parent>→<agent>` so a folded fan-out is never mistaken for a session (#3226)."""
    sid = s["session"][:8]
    if s.get("kind") == "subagent":
        return f"{_attributed_session(s)[:8]}→{sid}"
    return sid

def _top_sessions_table(S):
    """The 15 transcripts that generated the most output, with their cost shape."""
    L = []
    L.append("## Top 15 transcripts by output tokens\n")
    L.append("| Session | NS | Turns | Tool calls | Output tok | I:O | Cache-hit | cc-share | Est.$ |")
    L.append("|---|---|---:|---:|---:|---:|---:|---:|---:|")
    for s in sorted(S, key=lambda x: -x["tokens"]["output"])[:15]:
        ns = _ns_of(s)
        io = f"{s['io_ratio']:.0f}" if s["io_ratio"] else "—"
        ch = f"{s['cache_hit_frac']*100:.0f}%" if s["cache_hit_frac"] is not None else "—"
        cc = f"{s['cc_share']*100:.0f}%" if s.get("cc_share") is not None else "—"
        L.append(f"| {_transcript_label(s)} | {ns} | {s['assistant_turns']} | {s['n_tool_use']} | "
                 f"{fmt_int(s['tokens']['output'])} | {io} | {ch} | {cc} | ${s['cost_usd']:.2f} |")
    L.append("")
    return L

def _subagent_fold_table(S):
    """#3226 reconciliation: the top-level / subagent split of the SAME totals rendered
    above, so `audit` == `audit --top-level-only` + this delta is checkable by eye and
    the delta can never be added back on top (that would double-count)."""
    subs = [s for s in S if s.get("kind") == "subagent"]
    if not subs:
        return []
    top = [s for s in S if s.get("kind") != "subagent"]
    rows = [("top-level sessions", top), ("subagent / workflow", subs),
            ("**TOTAL (= scope totals above)**", S)]
    L = []
    L.append("## Subagent / workflow fold — INCLUDED in every total above (#3226)\n")
    L.append(f"{len(subs)} subagent transcripts are folded into the scope totals, the "
             "behavioral lens and the per-namespace rollup, attributed to their parent "
             "session. This table is a RECONCILIATION of those totals, not an addition — "
             "adding it back on would double-count. Use `--top-level-only` for the "
             "top-level-only view (the `top-level sessions` row below).\n")
    L.append("| Slice | Transcripts | Output tok | Fresh input | Cache-read | Cache-create | Est. cost |")
    L.append("|---|---:|---:|---:|---:|---:|---:|")
    for name, group in rows:
        g = summarize_analyses(group)
        t = g["tokens"]
        L.append(f"| {name} | {g['count']} | {fmt_int(t.get('output', 0))} | "
                 f"{fmt_int(t.get('input', 0))} | {fmt_int(t.get('cache_read', 0))} | "
                 f"{fmt_int(t.get('cache_create', 0))} | ${g['cost_usd']:,.2f} |")
    L.append("")
    return L

def report_md(sessions, agg, ns_prefix=NS_INCLUDE_PREFIX, since_days=None,
              top_level_only=False, max_sessions=None, excluded_subagents=None):
    S = [s for s in sessions if "error" not in s]
    L = _report_header(S, agg, ns_prefix, since_days, top_level_only,
                       max_sessions, excluded_subagents)
    L.extend(_scope_totals(agg))
    L.extend(_subagent_fold_table(S))
    L.extend(_model_mix_table(agg))
    L.extend(_billing_bucket_table(agg))
    L.extend(_per_model_table(agg))
    L.extend(_namespace_rollup(agg))
    L.extend(_distributions(agg))
    L.extend(_tool_mix_table(agg))
    L.extend(_hook_lens(agg))
    L.extend(_hook_stream_lens(agg.get("hook_stream")))
    L.extend(_dos_hook_lens(agg))
    L.extend(_behavior_lens(agg))
    L.extend(_top_sessions_table(S))
    return "\n".join(L)

def cmd_discover(a):
    ss = discover(a.root or DEFAULT_ROOTS, a.since_days, "" if a.all else a.ns_prefix)
    print(f"{len(ss)} sessions")
    for s in ss[:a.limit]:
        mt = datetime.datetime.fromtimestamp(s["mtime"]).isoformat(timespec="seconds")
        print(f"  {mt}  {s['size']//1024:6d}KB  {s['ns']}/{os.path.basename(s['path'])}")

def summarize_analyses(results):
    totals = collections.Counter()
    cost = 0.0
    ok = 0
    for r in results:
        if "error" in r:
            continue
        ok += 1
        for k, v in r["tokens"].items():
            totals[k] += v
        cost += r["cost_usd"]
    return {"count": ok, "tokens": dict(totals), "cost_usd": cost}

def summarize_transcripts(records):
    results = []
    for rec in records:
        r = analyze(rec["path"])
        results.append(r)
    return summarize_analyses(results)

def cmd_audit(a):
    roots = a.root or DEFAULT_ROOTS
    ns_prefix = "" if a.all else a.ns_prefix
    # #3226 — subagent/workflow transcripts are folded into the rollup BY DEFAULT.
    # They were ~23% of billed spend sitting outside the headline totals, and every
    # delegated turn lives in one, so the behavioral detectors (retry loop, file churn,
    # success loop, never-read edit, cache burst) had no delegated volume to look at
    # at all. `--top-level-only` restores the pre-#3226 view and prints what it hides.
    top_level_only = getattr(a, "top_level_only", False)
    ss = discover(roots, a.since_days, ns_prefix, include_subagents=not top_level_only)
    sess_recs = [s for s in ss if s.get("kind", "session") != "subagent"]
    sub_recs = [s for s in ss if s.get("kind") == "subagent"]
    if a.max:
        # --max caps TOP-LEVEL sessions (that is what it has always meant and what the
        # scope line reports). Slicing the interleaved list instead would silently make
        # it mean "transcripts" and drop sessions the operator asked for. A subagent
        # rides along when its parent survived the cap, or when its parent is not in the
        # window at all — dropping those orphans would hide real spend.
        kept = sess_recs[:a.max]
        keep_ids = {_parent_session(r) for r in kept}
        in_scope = {_parent_session(r) for r in sess_recs}
        sub_recs = [r for r in sub_recs
                    if _parent_session(r) in keep_ids or _parent_session(r) not in in_scope]
        sess_recs = kept
    print(f"analyzing {len(sess_recs) + len(sub_recs)} transcripts "
          f"({len(sess_recs)} top-level + {len(sub_recs)} subagent) ...", file=sys.stderr)
    out = []
    for rec in sess_recs + sub_recs:
        r = analyze(rec["path"])
        r["kind"] = rec.get("kind", "session")
        r["ns"] = rec.get("ns")
        r["parent_session"] = _parent_session(rec)
        out.append(r)
    sess = [r for r in out if r.get("kind") != "subagent"]
    subs = [r for r in out if r.get("kind") == "subagent"]
    # `out` is `sess` + `subs` with no overlap (discover returns each path once), so the
    # witness holds by construction: this aggregate == the --top-level-only aggregate
    # plus exactly the subagent delta. _subagent_fold_table renders that reconciliation.
    agg = aggregate(out)
    agg["_sessions"] = out
    agg["hook_stream"] = _reconcile_hook_stream(agg, load_hook_streams(getattr(a, "hook_stream", [])))
    agg["dos_hook_ledger"] = load_dos_hook_observations(
        os.getcwd(), a.since_days, getattr(a, "dos_since", None)
    )
    excluded_subagents = None
    if top_level_only:
        subagent_records = [s for s in discover(roots, a.since_days, ns_prefix,
                                                include_subagents=True)
                            if s.get("kind") == "subagent"]
        if subagent_records:
            excluded_subagents = summarize_transcripts(subagent_records)
    md = report_md(out, agg, ns_prefix=ns_prefix, since_days=a.since_days,
                   top_level_only=top_level_only, max_sessions=a.max,
                   excluded_subagents=excluded_subagents)
    agg.pop("_sessions", None)
    if a.md:
        open(a.md, "w", encoding="utf-8").write(md)
        print(f"wrote {a.md}", file=sys.stderr)
    if a.json:
        slim = {"aggregate": agg,
                "top_level_only": top_level_only,
                "excluded_subagents": excluded_subagents,
                "subagent_summary": summarize_analyses(subs) if subs else None,
                "top_level_summary": summarize_analyses(sess) if subs else None,
                "sessions": [{k: v for k, v in s.items() if k != "prompts"} for s in out]}
        json.dump(slim, open(a.json, "w", encoding="utf-8"), indent=2)
        print(f"wrote {a.json}", file=sys.stderr)
    print(md)
    # record→view→gate (#2365 d3): fail the run when the no-TTY hang regresses past the
    # threshold, so the guard fix stays enforced in CI. Off unless --gate-hangs is given.
    gate = getattr(a, "gate_hangs", None)
    if gate is not None:
        hangs = (agg.get("behavior") or {}).get("interactive_hangs", 0)
        if hangs > gate:
            print(f"::gate:: INTERACTIVE_HANG regression — {hangs} hang(s) across "
                  f"{len(out)} transcripts exceeds --gate-hangs {gate}", file=sys.stderr)
            raise SystemExit(3)
        print(f"gate ok: {hangs} interactive hang(s) ≤ --gate-hangs {gate}",
              file=sys.stderr)
    # #3942 — gate the one not-read sub-class that IS agent misbehavior: an edit
    # of a file the session never Read. post_resume (harness read-state reset)
    # and self_duplicate (guard-caught dup write) are excluded by construction,
    # so this gate never fires on those. Off unless --gate-never-read is given.
    nr_gate = getattr(a, "gate_never_read", None)
    if nr_gate is not None:
        never = ((agg.get("behavior") or {}).get("not_read_classes")
                 or {}).get("true_never_read", 0)
        if never > nr_gate:
            print(f"::gate:: TRUE_NEVER_READ regression — {never} never-read edit(s) "
                  f"across {len(out)} transcripts exceeds --gate-never-read {nr_gate}",
                  file=sys.stderr)
            raise SystemExit(3)
        print(f"gate ok: {never} never-read edit(s) ≤ --gate-never-read {nr_gate}",
              file=sys.stderr)

def _iso_bucket(ts, mode):
    # ts like 2026-06-16T21:19:39.123Z
    try:
        d = datetime.datetime.fromisoformat(ts.replace("Z", "+00:00"))
    except Exception:
        return None
    if mode == "day":
        return d.strftime("%Y-%m-%d")
    return f"{d.isocalendar().year}-W{d.isocalendar().week:02d}"   # ISO week

def trend_scan(roots, ns_prefix, bucket, include_subagents, exclude_substr=None):
    """Stream every transcript, fold usage/tools into time buckets.
    Cheap substring pre-filter: only json.loads lines that can carry usage or a tool_use,
    so multi-MB tool_result lines (browser snapshots, big reads) are skipped without parsing."""
    files = discover(roots, since_days=None, ns_prefix=ns_prefix,
                     include_subagents=include_subagents)
    if exclude_substr:
        files = [f for f in files if not any(s in f["path"] for s in exclude_substr)]
    buckets = collections.defaultdict(lambda: {
        "files": 0, "assist_turns": 0,
        "tok": collections.Counter(), "tools": collections.Counter(),
        "models": collections.Counter(), "cost": 0.0,
        "tool_errors": collections.Counter(), "beh": collections.Counter()})
    # Errored tool_results are the only user lines the behavioral lens needs;
    # gating on the serialized is_error:true keeps the huge SUCCESSFUL
    # tool_result lines (browser snapshots, big reads) unparsed as before.
    err_markers = ('"is_error": true', '"is_error":true')
    n = 0
    for f in files:
        n += 1
        first_ts = None
        try:
            fh = open(f["path"], encoding="utf-8")
        except OSError:
            continue
        # find this file's bucket from its first timestamped line
        rows_assist = []
        rows_err = []
        with fh:
            for line in fh:
                if '"timestamp"' in line and first_ts is None:
                    try:
                        first_ts = json.loads(line).get("timestamp")
                    except Exception:
                        pass
                if '"usage"' in line or '"tool_use"' in line:
                    try:
                        r = json.loads(line)
                    except Exception:
                        continue
                    if r.get("type") == "assistant":
                        rows_assist.append(r)
                    if first_ts is None and r.get("timestamp"):
                        first_ts = r.get("timestamp")
                elif any(m in line for m in err_markers):
                    try:
                        r = json.loads(line)
                    except Exception:
                        continue
                    if r.get("type") == "user":
                        rows_err.append(r)
        b = _iso_bucket(first_ts or "", bucket)
        if b is None:
            continue
        B = buckets[b]
        B["files"] += 1
        seen_msg_ids = set()   # de-dup billed turns per file (see analyze())
        seen_blocks = set()    # de-dup tool_use by BLOCK identity (see analyze())
        lens = BehaviorLens()
        tooluse_names = {}
        tooluse_args = {}
        tooluse_paths = {}
        for r in rows_assist:
            msg = r.get("message", {}) or {}
            mid = msg.get("id")
            new_turn = True
            if mid is not None:
                if mid in seen_msg_ids:
                    new_turn = False
                else:
                    seen_msg_ids.add(mid)
            if new_turn:
                u = msg.get("usage", {}) or {}
                B["assist_turns"] += 1
                B["models"][msg.get("model", "?")] += 1
                inp = u.get("input_tokens", 0) or 0
                out = u.get("output_tokens", 0) or 0
                cr = u.get("cache_read_input_tokens", 0) or 0
                cc = u.get("cache_creation_input_tokens", 0) or 0
                B["tok"]["input"] += inp
                B["tok"]["output"] += out
                B["tok"]["cache_read"] += cr
                B["tok"]["cache_create"] += cc
                B["cost"] += cost_usd(msg.get("model"), inp, cc, cr, out)
                lens.see_turn_usage(cr)   # #3069: suffix-cache reset detection
            for blk in (msg.get("content") or []):
                if isinstance(blk, dict) and blk.get("type") == "tool_use":
                    key = blk.get("id") or (mid, blk.get("name"),
                                            json.dumps(blk.get("input", {}),
                                                       sort_keys=True, default=str))
                    if mid is not None:
                        if key in seen_blocks:
                            continue
                        seen_blocks.add(key)
                    name = blk.get("name", "?")
                    B["tools"][name] += 1
                    ak = hash(json.dumps(blk.get("input", {}), sort_keys=True, default=str))
                    if blk.get("id"):
                        tooluse_names[blk["id"]] = name
                        tooluse_args[blk["id"]] = ak
                        tooluse_paths[blk["id"]] = _tool_path(blk.get("input"))
                    lens.see_tool_use(name, blk.get("input"), ak)
        for r in rows_err:
            content = (r.get("message", {}) or {}).get("content")
            if not isinstance(content, list):
                continue
            for blk in content:
                if isinstance(blk, dict) and blk.get("type") == "tool_result" \
                        and blk.get("is_error"):
                    lens.see_tool_result(
                        tooluse_names.get(blk.get("tool_use_id"), "?"),
                        True, _txt_str(blk.get("content", "")),
                        args_key=tooluse_args.get(blk.get("tool_use_id")),
                        path=tooluse_paths.get(blk.get("tool_use_id")))
        s = lens.summary()
        B["tool_errors"].update(s["tool_errors"])
        B["beh"]["timeout_kills"] += s["timeout_kills"]
        B["beh"]["interactive_hangs"] += s.get("interactive_hangs", 0)  # #2365 d3
        B["beh"]["sleep_polls"] += s["sleep_polls"]
        B["beh"]["suffix_resets"] += s.get("suffix_resets", 0)   # #3069
        B["beh"]["edit_churn"] += sum(s["edit_churn"].values())
        if s["max_repeat_failure"] >= REPEAT_FAILURE_MIN:
            B["beh"]["repeat_failure_files"] += 1
        if s["failure_mass"]:
            B["beh"]["failure_mass_files"] += 1
        if s["file_churn"]:   # flagged rewrite-loops only, not raw edit counts
            B["beh"]["file_churn_files"] += 1
        if s["success_loops"]:   # #2375 d2: successful read-loop / glob-storm / poll
            B["beh"]["success_loop_files"] += 1
    return buckets, n

def cmd_trend(a):
    roots = a.root or DEFAULT_ROOTS
    nsp = "" if a.all else a.ns_prefix
    excl = ["Temp-agf", "Temp-bench"] if a.exclude_bench else None
    buckets, n = trend_scan(roots, nsp, a.bucket, a.include_subagents, excl)
    print(f"# Trend — {n} transcripts, bucket={a.bucket}, ns_prefix={nsp or '(all)'}\n")
    print(f"{'bucket':10} {'files':>6} {'turns':>7} {'out_tok':>12} {'cacheRead':>14} "
          f"{'cacheHit%':>9} {'I:O':>7} {'cost$':>10} {'err%':>5} {'t/o':>4} "
          f"{'hang':>4} {'slp':>4} {'chrn':>5}  top_model / top_tool")
    rows = []
    for b in sorted(buckets):
        B = buckets[b]
        t = B["tok"]
        tot_in = t["input"] + t["cache_read"] + t["cache_create"]
        io = tot_in / t["output"] if t["output"] else 0
        chf = t["cache_read"] / tot_in * 100 if tot_in else 0
        tm = B["models"].most_common(1)
        tt = B["tools"].most_common(1)
        tmn = (tm[0][0].replace("claude-", "")[:14]) if tm else "—"
        ttn = (tt[0][0].replace("mcp__playwright__", "pw:")[:18]) if tt else "—"
        n_calls = sum(B["tools"].values())
        n_errs = sum(B["tool_errors"].values())
        errp = n_errs / n_calls * 100 if n_calls else 0
        beh = B["beh"]
        print(f"{b:10} {B['files']:>6} {B['assist_turns']:>7} {t['output']:>12,} "
              f"{t['cache_read']:>14,} {chf:>8.1f}% {io:>7.0f} {B['cost']:>10,.0f} "
              f"{errp:>4.0f}% {beh['timeout_kills']:>4} {beh['interactive_hangs']:>4} "
              f"{beh['sleep_polls']:>4} {beh['edit_churn']:>5}  {tmn} / {ttn}")
        rows.append({"bucket": b, "files": B["files"], "turns": B["assist_turns"],
                     "tok": dict(t), "io_ratio": round(io, 1), "cache_hit_pct": round(chf, 1),
                     "cost_usd": round(B["cost"], 2),
                     "top_models": B["models"].most_common(5),
                     "top_tools": B["tools"].most_common(12),
                     "behavior": {"tool_errors": dict(B["tool_errors"].most_common()),
                                  "tool_error_pct": round(errp, 1),
                                  **{k: beh[k] for k in
                                     ("timeout_kills", "interactive_hangs",
                                      "sleep_polls", "edit_churn",
                                      "repeat_failure_files", "file_churn_files",
                                      "success_loop_files")}}})
    if a.json:
        json.dump(rows, open(a.json, "w", encoding="utf-8"), indent=2)
        print(f"\nwrote {a.json}", file=sys.stderr)

def cmd_deep(a):
    s = analyze(a.session)
    # analyze() returns {"path","error"} (not the success shape) when it cannot
    # open/parse the transcript — a missing file, a wrong --root, or a path that
    # lives under a non-default $CLAUDE_CONFIG_DIR. Guard here BEFORE dereferencing
    # the success keys (s['session'], s['tokens'], s['behavior']); otherwise the
    # operator gets a KeyError traceback instead of an honest "cannot read that".
    if "error" in s or "session" not in s:
        print(f"cannot read transcript {a.session!r}: {s.get('error', 'no session data')}",
              file=sys.stderr)
        print("(is the path right? transcripts live under $CLAUDE_CONFIG_DIR/projects, "
              "not always ~/.claude)", file=sys.stderr)
        raise SystemExit(2)
    print(f"# Trajectory: {s['session']}")
    print(f"records={s['n_records']} turns={s['assistant_turns']} tool_calls={s['n_tool_use']} "
          f"output_tok={fmt_int(s['tokens']['output'])} io={s['io_ratio'] and round(s['io_ratio'],1)} "
          f"cache_hit={s['cache_hit_frac'] and round(s['cache_hit_frac'],3)} cost=${s['cost_usd']:.2f}")
    print(f"tools={s['tools']}")
    b = s["behavior"]
    print(f"behavior: tool_errors={sum(b['tool_errors'].values())} {b['tool_errors']} "
          f"timeout_kills={b['timeout_kills']} interactive_hangs={b.get('interactive_hangs', 0)} "
          f"shell_error_classes={b.get('shell_error_classes', {})} "
          f"sleep_polls={b['sleep_polls']} "
          f"edit_churn={b['edit_churn']} not_read_classes={b.get('not_read_classes', {})} "
          f"max_repeat_failure={b['max_repeat_failure']} "
          f"max_file_churn={b['max_file_churn']} "
          f"max_success_loop={b.get('max_success_loop', 0)}")
    for r in b["repeat_failures"][:5]:
        print(f"  repeat ×{r['count']} [{r['tool']}] {r['sig'][:120]}")
    for r in b["file_churn"][:5]:
        print(f"  churn  ×{r['count']} {r['file']}")
    for r in b.get("success_loops", [])[:5]:
        print(f"  succ-loop ×{r['count']} [{r['tool']}] {r['target'][:120]}")
    print("\n## User asks (the trajectory), in order:")
    for i, (ts, txt) in enumerate(s["prompts"]):
        one = " ".join(txt.split())
        print(f"  [{i:2d}] {ts}  {one[:200]}")

def cmd_capture_hooks(a):
    """Run a stream-json command and persist receive timestamps on every JSON row."""
    command = list(a.command or [])
    if command and command[0] == "--":
        command = command[1:]
    if not command:
        raise SystemExit("capture-hooks requires a command after --")
    out_path = os.path.abspath(a.out)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8", newline="\n") as capture:
        proc = subprocess.Popen(command, stdout=subprocess.PIPE, text=True,
                                encoding="utf-8", errors="replace")
        assert proc.stdout is not None
        try:
            for line in proc.stdout:
                sys.stdout.write(line)
                sys.stdout.flush()
                try:
                    row = json.loads(line)
                except (ValueError, TypeError):
                    continue
                row["_captured_at"] = datetime.datetime.now(
                    datetime.timezone.utc).isoformat().replace("+00:00", "Z")
                capture.write(json.dumps(row, ensure_ascii=False) + "\n")
                capture.flush()
        finally:
            proc.stdout.close()
        return proc.wait()


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    p = argparse.ArgumentParser(description="Audit Claude Code session transcripts.")
    sub = p.add_subparsers(dest="cmd", required=True)
    capture = sub.add_parser("capture-hooks")
    capture.add_argument("--out", required=True, metavar="NDJSON")
    capture.add_argument("command", nargs=argparse.REMAINDER)
    for name in ("discover", "audit"):
        q = sub.add_parser(name)
        q.add_argument("--root", action="append")
        q.add_argument("--since-days", type=float, default=None)
        q.add_argument("--ns-prefix", default=NS_INCLUDE_PREFIX)
        q.add_argument("--all", action="store_true", help="include ALL namespaces (no prefix filter)")
        if name == "discover":
            q.add_argument("--limit", type=int, default=40)
        else:
            q.add_argument(
                "--dos-since", default=None, metavar="ISO8601",
                help="exact timezone-qualified lower bound for the DOS observation ledger; "
                     "overrides --since-days for that ledger",
            )
            q.add_argument("--max", type=int, default=None)
            q.add_argument(
                "--hook-stream", action="append", default=[], metavar="NDJSON",
                help="captured claude stream-json with --include-hook-events; repeatable",
            )
            q.add_argument("--json", default=None)
            q.add_argument("--md", default=None)
            # #3226 — folding subagent/workflow transcripts is now the DEFAULT, so the
            # opt-IN flag became a no-op alias kept for callers/scripts that pass it,
            # and --top-level-only is the opt-OUT that restores the pre-#3226 view.
            g = q.add_mutually_exclusive_group()
            g.add_argument("--top-level-only", action="store_true",
                           help="count ONLY top-level session transcripts, excluding "
                                "subagent/workflow spend from every total and from the "
                                "behavioral lens (the pre-#3226 view)")
            g.add_argument("--include-subagents", action="store_true",
                           help="no-op: subagent/workflow transcripts are folded in by "
                                "default since #3226 (kept for compatibility)")
            q.add_argument("--gate-hangs", type=int, default=None, metavar="N",
                           help="exit 3 if interactive-editor/pager hangs (#2365 d3) "
                                "across the scanned window exceed N — a CI regression "
                                "gate on the guard fix (e.g. --gate-hangs 0)")
            q.add_argument("--gate-never-read", type=int, default=None, metavar="N",
                           help="exit 3 if true-never-read edit churn (#3942, an edit "
                                "of a file the session never Read) across the scanned "
                                "window exceeds N — a CI gate on the one not-read "
                                "sub-class that is agent misbehavior (e.g. "
                                "--gate-never-read 0)")
    qt = sub.add_parser("trend")
    qt.add_argument("--root", action="append")
    qt.add_argument("--ns-prefix", default=NS_INCLUDE_PREFIX)
    qt.add_argument("--all", action="store_true")
    qt.add_argument("--bucket", choices=["day", "week"], default="week")
    qt.add_argument("--include-subagents", action="store_true")
    qt.add_argument("--exclude-bench", action="store_true",
                    help="drop Temp-agf*/Temp-bench* eval namespaces")
    qt.add_argument("--json", default=None)
    q = sub.add_parser("deep")
    q.add_argument("session")
    a = p.parse_args()
    handlers = {"discover": cmd_discover, "audit": cmd_audit, "trend": cmd_trend,
                "deep": cmd_deep, "capture-hooks": cmd_capture_hooks}
    raise SystemExit(handlers[a.cmd](a) or 0)

if __name__ == "__main__":
    main()
