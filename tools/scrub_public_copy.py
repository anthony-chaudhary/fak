#!/usr/bin/env python3
"""Public-copy scrubber for the fleet repo.

Produces a public-safe snapshot from a `git archive HEAD` export of the private
canonical repo. Run from the export dir (or pass --export-dir):

    git archive --format=tar HEAD | tar -x -C /tmp/fleet-public-export
    python3 tools/scrub_public_copy.py --export-dir /tmp/fleet-public-export
    cd /tmp/fleet-public-export && git init && git add -A && git commit

The KEEP / REDACT / PRIVATE-ONLY lists below are the canonical reference for
what is intentional vs must-be-redacted vs never-published. See
PUBLIC-SCRUB-POLICY.md for the human-readable companion. Any future audit
sweep should consult both before flagging a string as sensitive.

Idempotent: safe to re-run.
"""
from __future__ import annotations

import argparse
import fnmatch
import json
import os
import re
import shutil
import sys

# Paths (relative to EXPORT_DIR) to delete entirely. Operator-private evidence,
# operator-machine captures, or the committed memory mirror. These live ONLY on
# the private side (see PUBLIC-SCRUB-POLICY.md).
DELETE_PATHS = [
    ".claude/memory",
    "AGENTS.md",  # operator-machine workflow guidance; private side only
    "CLAUDE.md",  # Claude Code mirror of AGENTS.md; private side only (symmetric)
    "fak/experiments/fleet-nodes/anthony",
    "fak/experiments/qwen36/node-reports",  # operator-machine nvidia preflight evidence (real IP, real paths)
    # --- DGX lab-benchmark subsystem (operator's private lab infra: the example.lab
    # DGX node, Slack control bridge, handoff packets, tied to the private
    # Benchmark repo). Excluded from the PUBLIC COPY ONLY -- content stays in
    # the private canonical repo. See PUBLIC-SCRUB-POLICY.md PRIVATE-ONLY list. ---
    "fak/experiments/dgx",
    "docs/dgx-benchmark-methodology.md",
    "fak/QWEN36-DGX-STANDALONE-PREP.md",
    # The DGX Slack *control bridge* code subsystem. internal/dgxbridge speaks the
    # operator's private lab-DGX Slack control protocol and ships a REAL recorded
    # control session (testdata/real_transcript.jsonl: live Slack channel/user ids,
    # lab host, internal paths) plus stray committed .dos/ observation logs. Same
    # private class as fak/experiments/dgx + tools/*dgx*, but it arrived as in-tree
    # CODE so the path-glob rules above did not catch it. Self-contained: only
    # cmd/dgxbridge, cmd/dgxbench, cmd/slackgc import it and all are `package main`,
    # so deleting the cluster keeps the public module building.
    "fak/internal/dgxbridge",
    "fak/cmd/dgxbridge",
    "fak/cmd/dgxbench",
    "fak/cmd/slackgc",
]

DELETE_GLOBS = [
    "fak/experiments/qwen36/anthony-nvidia-*",
    "fak/experiments/qwen36/qwen36-agent-anthony*",
    "fak/experiments/qwen36/qwen36-surface-smoke-anthony.*",
    "fak/experiments/qwen36/qwen36-standalone-readiness-current.*",
    "fak/experiments/qwen36/QWEN36-DGX-STANDALONE-READINESS.md",
    "fak/experiments/qwen36/peer-access-probe-20260619-final.json",
    "fak/experiments/qwen36/qwen36-direct-port-probe-20260619*.json",
    # DGX lab tooling (Slack bridge / packets / handoff / endpoint-load) + the
    # dgx sweep profile. fnmatch '*' crosses '/', so this catches tools/*dgx*
    # AND tools/sweep_profiles/*dgx*.
    "tools/*dgx*",
    # --- by-machine RELOCATION guard ---------------------------------------
    # `tools/bench_migrate.py` copies the qwen36 operator-node corpus from
    # fak/experiments/qwen36/ into fak/experiments/benchmark/runs/by-machine/
    # <machine>/<ts>-qwen36/. Those copies are BYTE-IDENTICAL to the qwen36/
    # originals above (same operator IP 100.64.0.10 / FQDN / node_name), but
    # the qwen36/-anchored globs above do NOT match the new path -- so ~21
    # operator-node files were shipping to public. These patterns key on the
    # FILENAME signature and are depth-independent (fnmatch '*' crosses '/'),
    # so any future by-machine/<machine>/<ts>/ layout is also covered and a
    # later migration cannot silently reopen the hole. They delete ONLY the
    # operator-node capture signatures; the legitimate public bench data under
    # the same by-machine dirs (native-gguf-q8-*, local-ollama-*, amd-vulkan-*,
    # manifests -- none carry a needle) is intentionally KEPT (the policy's
    # `anthony` = public-owner's-own-machine keep rule).
    "fak/experiments/benchmark/runs/*anthony-nvidia-*",
    "fak/experiments/benchmark/runs/*qwen36-agent-anthony*",
    "fak/experiments/benchmark/runs/*qwen36-surface-smoke-anthony.*",
    "fak/experiments/benchmark/runs/*qwen36-standalone-readiness-current.*",
    "fak/experiments/benchmark/runs/*peer-access-probe-*",
    "fak/experiments/benchmark/runs/*qwen36-direct-port-probe-*",
    # --- DGX-machine catalog runs ------------------------------------------
    # `tools/bench_slack.py register_dgx_run` registers DGX results under
    # by-machine/dgx*/<ts>-dgx/ (machine_id "dgx" / "dgx-test"). The DGX lab
    # subsystem is private-only (above + PUBLIC-SCRUB-POLICY.md); the run *dirs*
    # must never ship. fnmatch '*' crosses '/', so `dgx*` matches every file
    # under any dgx* machine dir, depth-independently. (The catalog.json
    # AGGREGATE that references these runs is stripped separately by
    # _strip_private_machines_from_catalog -- deleting the dirs is not enough.)
    "fak/experiments/benchmark/runs/by-machine/dgx*",
]

# Literal string replacements applied to every text file (incl. .svg, .json,
# .md, .py, .go). Order matters: more specific patterns first.
REPLACEMENTS = [
    ("github.com/anthony-chaudhary/fak",
     "github.com/anthony-chaudhary/fak",
     "Go module path netrasystems/fak -> anthony-chaudhary/fak"),
    ("github.com/example/dos-preflake/go",
     "github.com/example/dos-preflake/go",
     "go.mod comment: netrasystems/dos-preflake -> example"),
    ("github.com/example/metrics-service",
     "github.com/example/metrics-service",
     "go.mod comment: netrasystems/metrics-service -> example"),
    # Operator email domain (the `netra` codename) -- arrived with the GCP-quota
    # work (gcp_quota_request.py DEFAULT_CONTACT, the gcp probe/quota-request
    # JSONs, the account-probe runbook). All occurrences are `<local>@example.com`,
    # so a domain rewrite clears the `netrasystems` audit needle in every one of
    # them while leaving the public-owner name (anthony[-.]chaudhary) intact.
    ("example.com", "example.com",
     "operator email domain example.com (netra codename) -> example.com"),
    # Operator's real GCP project id -- infra identity, same class as the lab
    # DGX/tailnet/hostname rewrites. Appears in the GCP benchmark docs/tooling
    # and the probe/quota-request evidence JSONs. Verified absent post-export via
    # EXPORT_AUDIT_NEEDLES below.
    ("example-gcp-project", "example-gcp-project",
     "operator GCP project id -> generic placeholder"),
    # Operator Slack workspace ids from the lab-DGX control bridge. The dgxbridge
    # cluster (with the real transcript carrying the bot/user ids) is deleted in
    # DELETE_PATHS; these rewrites catch the channel id that ALSO leaked into kept
    # files -- the DGX runbook docs (PLAN-*-dgx-*, GLM-*-DGX-*,
    # docs/dgx-python3-fix-*), a test mock, and the bench catalog. The channel
    # *name* `dgx-control` and the standard /var/lib/slack-control paths are left:
    # not reachable without the workspace + a bot token, so not a secret once the
    # ids + host are scrubbed.
    ("C0EXAMPLE00", "C0EXAMPLE00", "operator Slack channel id (#dgx-control)"),
    ("U0EXAMPLE00", "U0EXAMPLE00", "operator Slack bot/user id"),
    ("U0EXAMPLE01", "U0EXAMPLE01", "operator Slack user id"),
    ("100.64.0.10", "100.64.0.10",
     "operator Tailscale IP 100.64.0.10 -> generic test IP 100.64.0.10"),
    ("100.64.0.10", "100.64.0.10",
     "test fixture IP 100.64.0.10 -> generic"),
    ("100.64.0.10", "100.64.0.10",
     "operator Tailscale IP 100.64.0.10 (host node-desktop-b) -> generic 100.64.0.10"),
    ("user@100.64.0.10", "user@100.64.0.10", "SSH user antho -> user"),
    ("`user`", "`user`", "SSH user antho -> user"),
    ("= \"antho\"", "= \"user\"", "SSH user antho -> user"),
    ("/Users/USER/.cache/", "/Users/USER/.cache/", "macOS cache path"),
    ("/Users/USER/.local/", "/Users/USER/.local/", "macOS local path"),
    ("/Users/USER/Documents/GitHub/fleet",
     "/Users/USER/Documents/GitHub/fleet",
     "macOS repo working-copy path"),
    ("<benchmark-checkout>",
     "<benchmark-checkout>",
     "macOS sibling Benchmark repo path"),
    ("<benchmark-checkout>",
     "<benchmark-checkout>",
     "drop sibling Benchmark repo path"),
    ("C:\\Users\\USER\\", "C:\\Users\\USER\\", "Windows user path"),
    ("C:\\\\Users\\\\antho\\\\", "C:\\\\Users\\\\USER\\\\",
     "Windows user path (JSON-escaped)"),
    ("C:\\Users\\antho", "C:\\Users\\USER",
     "Windows user path (no trailing slash)"),
    (".claude-agent", ".claude-agent",
     "custom Claude config dir .claude-agent -> .claude-agent"),
    (".claude-agent", ".claude-agent",
     "custom Claude config dir .claude-agent -> .claude-agent"),
    ("node-watch", "node-watch", "test fixture label node-watch"),
    ("node-old-watch", "node-old-watch", "test fixture label node-old-watch"),
    ("node-new-watch", "node-new-watch", "test fixture label node-new-watch"),
    ("node-current-watch", "node-current-watch", "test fixture label node-current-watch"),
    ("node-packeted-watch", "node-packeted-watch", "test fixture label node-packeted-watch"),
    ("node-preflight-watch", "node-preflight-watch", "test fixture label node-preflight-watch"),
    ("node-qwen36-surfaces", "node-qwen36-surfaces", "test fixture label node-qwen36-surfaces"),
    ("node-qwen36-watch-live", "node-qwen36-watch-live", "test fixture label node-qwen36-watch-live"),
    ("node-agent", "node-agent", "test fixture/account label node-agent"),
    # Operator org suffix on Claude account dirs: `.claude-<tag>-netra` -> `<tag>`.
    # The `-netra` suffix is the operator org codename (= netrasystems); it survived
    # the `netrasystems` needle because that needle never matched the bare suffix.
    # Rewrite to a generic `-acct` so the account-tooling grammar stays intact
    # (fleet_accounts.account_tag strips the suffix either way) without shipping the
    # codename. Already applied to the public copy.
    ("-netra", "-acct", "operator org suffix on account dirs -netra -> -acct"),
]

# Tokens that appear in varying CASES (Windows hostnames are case-insensitive;
# a person's name may be capitalized) -> matched case-insensitively so e.g.
# `node-desktop-b` / `node-macos-a` cannot slip past the lowercase
# literal rules above. Applied after REPLACEMENTS; output is the generic label.
CASE_INSENSITIVE_REPLACEMENTS = [
    ("node-windows-a", "node-windows-a"),
    ("node-desktop-b", "node-desktop-b"),
    ("node-macos-a", "node-macos-a"),
    ("node-macos-a", "node-macos-a"),
    ("node-macos-a", "node-macos-a"),
    # Operator lab / DGX infra (arrived with the DGX benchmark work). Order
    # matters: full FQDN -> short host -> domain -> tailnet, then the SSH
    # password LAST so the hostname forms clear before the shared `<ssh-password>`
    # prefix is rewritten. The password is a real credential -- rotate it.
    ("dgx-a100.example.lab", "dgx-a100.example.lab"),
    ("dgx-a100", "dgx-a100"),
    ("example.lab", "example.lab"),
    ("tailnet.example.ts.net", "tailnet.example.ts.net"),
    ("tailnet", "tailnet"),
    ("<ssh-password>", "<ssh-password>"),
]

# Directories whose NAME carries an owner/personal identifier. Content
# replacement cannot touch path components, so these are renamed at export
# (bottom-up, after replacements). The operator's own machine label `anthony`
# is intentionally NOT here -- it is the project's public owner identity.
DIR_RENAME = {
    "node-macos-a": "node-macos-a",
}

# AUDIT_NEEDLES -- the PRE-COMMIT gate (audit_staged, via githooks/pre-commit).
# SECRETS only: things that must NEVER enter private history. NOTE: the Go
# module path `netrasystems/fak` is intentionally NOT here -- it is legitimate
# private working material (~144 import sites + go.mod), rewritten at export by
# REPLACEMENTS. Gating it at commit forces FLEET_ALLOW_LEAK=1 on every Go commit
# that adds an import, crying wolf and diluting the real secret signal. It lives
# in EXPORT_AUDIT_NEEDLES below. Add a genuinely-new secret to BOTH lists.
# (On a provisioned PUBLIC clone the EXPORT/identity tier is folded into this gate
# at runtime from the pulled sidecar -- see _effective_audit_needles / PLAN §6.)
AUDIT_NEEDLES = [
    "100.64.0.10",
    "100.64.0.10",
    "100.64.0.10",  # third operator Tailscale IP (host node-desktop-b)
    "/Users/anthony",
    "Users\\antho",
    "Users\\\\antho",
    "GitHub/Benchmark",
    "Documents/GitHub/Benchmark",
    "node-agent-netra",
    "node-windows-a",
    "node-desktop-b",
    ".claude-agent",
]

# EXPORT_AUDIT_NEEDLES -- the POST-EXPORT verification (must be ABSENT in the
# public copy once REPLACEMENTS has run). Superset of AUDIT_NEEDLES: adds the
# identity rewrites (module org) that are fine in private but must not survive
# export. If any of these survive REPLACEMENTS, the post-scrub audit fails and
# the operator investigates before publishing.
EXPORT_AUDIT_NEEDLES = AUDIT_NEEDLES + [
    "netrasystems",
    "-netra",   # operator org suffix on account dirs (.claude-<tag>-netra) -- the
                # bare-suffix form the `netrasystems` needle above misses; rewritten
                # to the generic `-acct` by REPLACEMENTS. Hyphen-anchored so it does
                # not fire on the legit `netrasystems` module path or "Netra Systems".
    "node-macos-a",
    "dgx-a100",  # operator lab DGX machine (infra) -- prefix of the SSH password below
    "example.lab",       # operator lab DNS domain
    "tailnet",    # operator Tailscale tailnet name
    "<ssh-password>",       # SSH credential committed in DGX content -- ROTATE + scrub history
    "example-gcp-project",  # operator GCP project id (infra) -- export-only rewrite
    "C0EXAMPLE00",   # operator Slack channel id (#dgx-control) -- export-only rewrite
    "U0EXAMPLE00",   # operator Slack bot/user id
    "U0EXAMPLE01",   # operator Slack user id
]

# Machine-id prefixes whose benchmark runs are PRIVATE and must never reach the
# public copy. The DGX lab subsystem (PUBLIC-SCRUB-POLICY.md) registers runs
# under machine_id "dgx"/"dgx-test" via bench_slack.register_dgx_run. The
# DELETE_GLOBS above drop the run *dirs*, but catalog.json is a committed
# AGGREGATE that still carries their metadata (run_id, model, timestamps) --
# _strip_private_machines_from_catalog() removes those from the export copy.
PRIVATE_MACHINE_PREFIXES = ("dgx",)

# Generic SECRET-shaped patterns matched by SHAPE, not by a known literal value.
# A live Slack bot (xoxb-) or user (xoxp-) token must never enter EITHER repo,
# and unlike the redaction needles there is no fixed string to list -- the only
# durable defense is the shape. The fixtures the policy intentionally KEEPS
# (`xoxb-test`, `xoxb-redacted`, the literal `xoxb-...` doc placeholder) do NOT
# match this triplet shape, so this adds detections without false positives.
# Checked at COMMIT (pre-commit gate, added lines) AND at EXPORT (post-scrub
# audit). SELF_REFERENTIAL files are exempt, as for the needles.
AUDIT_REGEXES = [
    (re.compile(r"xox[bp]-\d{8,}-\d{8,}-[A-Za-z0-9]{16,}"), "live Slack token (xoxb/xoxp)"),
]

# Pulled-from-private REAL needle file (gitignored: tools/_registry is ignored).
# The HARD-CUT model edits the public copy DIRECTLY instead of regenerating it
# from the private repo, so the public tree needs its OWN standing leak scan. But
# the committed EXPORT_AUDIT_NEEDLES above are DE-FANGED -- REPLACEMENTS rewrote
# the real high-sensitivity values (operator IPs, lab host, Slack ids, SSH
# password) to generic placeholders the policy intentionally KEEPS in public
# (e.g. 100.64.0.10, example.lab). Scanning the public tree against that list
# would both cry wolf on kept placeholders AND miss a freshly-pasted real secret.
# The effective tree scan therefore sources the REAL needles from this file,
# produced by `tools/pull_scan_needles.py` ("pull the private scan instructions").
PRIVATE_NEEDLES_REL = os.path.join("tools", "_registry", "scrub_needles.private.json")
PRIVATE_NEEDLES_SCHEMA = "fleet-scrub-needles/1"


def load_private_needles(root: str) -> dict | None:
    """Load the REAL operator needles pulled from the private repo, if present.

    Returns the parsed sidecar dict (``audit_needles`` / ``export_audit_needles``)
    or None when the file is absent or unreadable. Absent the file, the tree scan
    runs SHAPE-only (degraded but honest) -- see ``audit_tree``.
    """
    path = os.path.join(root, PRIVATE_NEEDLES_REL)
    if not os.path.isfile(path):
        return None
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError):
        return None
    return data if isinstance(data, dict) else None


def _strip_private_machines_from_catalog(export_dir: str):
    """Strip PRIVATE_MACHINE_PREFIXES runs/machines from the EXPORT catalog.json.

    catalog.json aggregates every machine's runs into one committed file, so the
    DELETE_GLOBS that drop dgx run *dirs* are not sufficient -- the dgx run
    *metadata* survives in the catalog and would ship to public. This rewrites
    the export copy only (the private canonical catalog is untouched). Returns
    the number of runs dropped, 0 if none, or None if the catalog is absent.
    """
    rel = os.path.join("fak", "experiments", "benchmark", "catalog.json")
    path = os.path.join(export_dir, rel)
    if not os.path.isfile(path):
        return None
    try:
        with open(path, encoding="utf-8") as f:
            cat = json.load(f)
    except (OSError, json.JSONDecodeError):
        return None

    def is_private(mid) -> bool:
        return any(str(mid).startswith(p) for p in PRIVATE_MACHINE_PREFIXES)

    runs = cat.get("runs", []) or []
    kept = [r for r in runs if not is_private(r.get("machine_id", ""))]
    dropped_ids = {r.get("run_id") for r in runs if is_private(r.get("machine_id", ""))}
    n_dropped = len(runs) - len(kept)
    if n_dropped == 0:
        return 0
    cat["runs"] = kept
    if isinstance(cat.get("machines"), dict):
        cat["machines"] = {m: v for m, v in cat["machines"].items() if not is_private(m)}
    idx = cat.get("index")
    if isinstance(idx, dict):
        for bucket in idx.values():
            if isinstance(bucket, dict):
                for key in list(bucket):
                    bucket[key] = [rid for rid in bucket[key] if rid not in dropped_ids]
                    if not bucket[key]:
                        del bucket[key]
    with open(path, "w", encoding="utf-8") as f:
        json.dump(cat, f, indent=2, sort_keys=True)
    return n_dropped

# Files that DEFINE the scrub (denylists, replacement rules, policy text). They
# are excluded from the post-scrub AUDIT only -- they necessarily name the
# needles to document/enforce them, so flagging them is noise. REPLACEMENTS
# still runs on them so any realistic leakage form inside them (a full operator
# home path, a real IP) is still scrubbed out; what survives is
# the low-sensitivity bare needle names a published denylist must name anyway.
SELF_REFERENTIAL = {
    "PUBLIC-SCRUB-POLICY.md",
    "tools/scrub_public_copy.py",
    "tools/githooks/pre-commit",
}

BINARY_EXT = {
    ".png", ".jpg", ".jpeg", ".gif", ".pdf", ".ico",
    ".zip", ".tgz", ".tar", ".gz",
    ".bin", ".exe", ".dll", ".so", ".dylib", ".a", ".o",
    ".gguf", ".safetensors", ".model",
    ".otf", ".ttf", ".woff", ".woff2",
    ".mp4", ".mov", ".mp3", ".wav",
}


def is_text(path: str) -> bool:
    _, ext = os.path.splitext(path)
    return ext.lower() not in BINARY_EXT


def read_text(path: str):
    """Read a text file as a searchable string regardless of encoding.

    Returns ``(text, encoding)`` where ``encoding`` is how to write the file
    back unchanged, or ``(None, None)`` if the file cannot be read at all.

    Cascades utf-8 (strict) -> utf-16 (BOM-aware) -> latin-1 (byte-preserving,
    never fails). This is critical: a non-UTF-8 text file (e.g. a UTF-16 JSON)
    makes a strict utf-8 read raise, and the old ``except: continue`` silently
    skipped BOTH scrubbing and the post-scrub audit -- an undetectable leak
    path. The cascade ensures every text file is scrubbed and audited.
    """
    try:
        with open(path, "rb") as f:
            raw = f.read()
    except OSError:
        return None, None
    for enc in ("utf-8", "utf-16", "latin-1"):
        try:
            return raw.decode(enc), enc
        except UnicodeDecodeError:
            continue
    return raw.decode("latin-1", errors="replace"), "latin-1"


def expand_glob(root: str, pattern: str):
    matches = []
    prefix_len = len(root) + 1
    for dirpath, _d, filenames in os.walk(root):
        for name in filenames:
            full = os.path.join(dirpath, name)
            rel = full[prefix_len:] if full.startswith(root + os.sep) else full
            if fnmatch.fnmatch(rel.replace(os.sep, "/"), pattern):
                matches.append(full)
    return matches


def _effective_audit_needles(root: str) -> list[str]:
    """AUDIT_NEEDLES folded with the pulled-private REAL needles, if present.

    The public copy's committed AUDIT_NEEDLES are DE-FANGED (the export scrub
    rewrote the real high-sensitivity values to kept placeholders). Under the
    HARD CUT the commit gate must still catch the REAL values, so it folds in the
    needles pulled from the private repo (``tools/pull_scan_needles.py``) when the
    gitignored sidecar is present. Absent the sidecar the gate is byte-identical
    to before.

    Two tiers fold in, BOTH gated on SIDECAR PRESENCE (PLAN-hard-cut §6):

      - SECRET tier (``audit_needles``): the real high-sensitivity values.
      - EXPORT/identity tier (``export_audit_needles``): module org and the other
        identity rewrites. These are intentionally NOT in the committed
        AUDIT_NEEDLES because the PRIVATE repo legitimately carries them (e.g. the
        ``netrasystems`` module path on ~144 import sites + go.mod) and rewrites
        them only at export -- gating them there would cry wolf on every Go import
        commit. But under the hard cut a PROVISIONED PUBLIC CLONE (the ONLY place
        the sidecar exists) has already had them rewritten away and has no export
        step left to catch a fresh paste, so an identity-tier value reappearing IS
        a leak and must block at COMMIT. Gating on sidecar presence keeps the
        private repo's gate (no sidecar) exactly secret-tier-only -- unchanged.
    """
    priv = load_private_needles(root)
    if not priv:
        return list(AUDIT_NEEDLES)
    extra = list(priv.get("audit_needles") or [])
    extra += list(priv.get("export_audit_needles") or [])  # §6: identity tier too
    seen = set(AUDIT_NEEDLES)
    out = list(AUDIT_NEEDLES)
    for n in extra:
        if n and n not in seen:
            seen.add(n)  # dedup across both tiers (export is a superset of secret)
            out.append(n)
    return out


def _scan_added_lines(diff_text: str, needles: list[str] | None = None):
    """Scan a unified diff's ADDED lines for redact needles.

    Shared by `audit_staged` (the pre-commit gate) and `audit_range` (the CI
    gate) so both apply byte-identical matching: same needle set, same
    case-insensitive substring test, same SELF_REFERENTIAL exemption. ``needles``
    defaults to AUDIT_NEEDLES; callers pass `_effective_audit_needles(root)` to
    add the pulled-private REAL needles. Returns a list of
    (file, new_line_no, needle, preview) hits.
    """
    needle_list = AUDIT_NEEDLES if needles is None else needles
    hits = []
    current_file = None
    new_line_no = 0
    self_ref = False  # current file DEFINES the scrub -> exempt (it must name needles)
    hunk_re = re.compile(r"^@@ -\d+(?:,\d+)? \+(\d+)")
    for line in diff_text.splitlines():
        if line.startswith("+++ b/"):
            current_file = line[6:]
            # The denylist / replacement-rule / policy files necessarily name the
            # needles to enforce them -- the SAME exemption the post-scrub audit
            # applies (see SELF_REFERENTIAL / the main() audit loop). Without this,
            # every edit to the denylist itself trips the gate and forces
            # FLEET_ALLOW_LEAK=1 on routine scrub maintenance.
            self_ref = (current_file or "").replace("\\", "/") in SELF_REFERENTIAL
        elif line.startswith("@@"):
            m = hunk_re.match(line)
            new_line_no = int(m.group(1)) if m else 0
        elif line.startswith("+") and not line.startswith("+++"):
            payload = line[1:]
            payload_l = payload.lower()
            if not self_ref:
                for needle in needle_list:
                    if needle.lower() in payload_l:
                        preview = payload.strip()[:80]
                        hits.append((current_file, new_line_no, needle, preview))
                for rx, label in AUDIT_REGEXES:
                    if rx.search(payload):
                        preview = payload.strip()[:80]
                        hits.append((current_file, new_line_no, label, preview))
            new_line_no += 1
    return hits


def _report_hits(hits, where: str) -> int:
    if not hits:
        return 0
    print(f"FOUND {len(hits)} redact-needle hit(s) in {where}:")
    for f, n, needle, preview in hits:
        print(f"  {f}:{n}  [{needle}]  {preview}")
    return 1


def audit_staged(root: str) -> int:
    """Scan staged additions for AUDIT_NEEDLES. For the pre-commit hook.

    Uses `git diff --cached -U0` so only ADDED lines are scanned (no false
    positives on already-reviewed surrounding context). Reports every hit with
    file + line + needle and exits 1 if any are found.
    """
    import subprocess

    result = subprocess.run(
        ["git", "-C", root, "diff", "--cached", "--no-color", "-U0"],
        capture_output=True, encoding="utf-8", errors="replace",
    )
    if result.returncode != 0:
        print(f"git diff --cached failed: {result.stderr.strip()}", file=sys.stderr)
        return 2
    return _report_hits(
        _scan_added_lines(result.stdout, _effective_audit_needles(root)), "staged content"
    )


def audit_range(root: str, rev_range: str) -> int:
    """Scan the ADDED lines of a commit range for AUDIT_NEEDLES. For CI.

    The local pre-commit gate (`audit_staged`) only fires when a clone has the
    hook armed (`install_trunk_guard.py`). An un-armed clone -- or any CI-only
    path -- has no leak gate. This is the machine backstop: run over exactly the
    commits a push would ADD to the trunk (`BASE..HEAD`), scanning only added
    lines so already-reviewed context never cries wolf. Exit 1 on any hit, 0 if
    clean, 2 if the range is unreadable -- mirroring `dos review`'s exit grammar
    so the two CI gates compose. Identity-tier rewrites (e.g. `netrasystems`) are
    NOT in the committed AUDIT_NEEDLES, so on the private repo (no sidecar) they
    are not gated -- they are legal in private history and gating them would cry
    wolf on every Go import commit. On a PROVISIONED PUBLIC CLONE (sidecar present)
    `_effective_audit_needles` folds the pulled identity tier in, so a fresh paste
    IS caught here too (PLAN-hard-cut §6).
    """
    import subprocess

    result = subprocess.run(
        ["git", "-C", root, "diff", "--no-color", "-U0", rev_range],
        capture_output=True, encoding="utf-8", errors="replace",
    )
    if result.returncode != 0:
        print(f"git diff {rev_range} failed: {result.stderr.strip()}", file=sys.stderr)
        return 2
    rc = _report_hits(
        _scan_added_lines(result.stdout, _effective_audit_needles(root)), f"range {rev_range}"
    )
    if rc == 0:
        print(f"leak-scan: {rev_range} clean (no AUDIT_NEEDLES in added lines)")
    return rc


def audit_tree(root: str, as_json: bool = False) -> int:
    """HARD-CUT backstop: scan every git-TRACKED text file in ``root`` for needles.

    The pre-commit gate (``audit_staged``) only sees NEW additions, and only on an
    armed clone. Once the public copy is edited DIRECTLY (not regenerated from the
    private repo), a needle can sit in a tracked file with no export step to catch
    it. This scans the WHOLE tracked tree.

    Effective needle sources (NEVER the committed EXPORT_AUDIT_NEEDLES, which are
    de-fanged placeholders -- see ``load_private_needles``):
      * ``AUDIT_REGEXES``      -- secret SHAPES (live Slack token); always sound.
      * pulled private needles -- the REAL operator values, present only once
        ``tools/pull_scan_needles.py`` has written the gitignored sidecar.

    Mode is reported honestly: ``full`` once the private needles are pulled,
    ``shape-only`` (degraded) until then. SELF_REFERENTIAL denylist files are
    exempt. Scans only ``git ls-files`` (the shippable tracked surface, skipping
    .git/_registry/.dos with no extra walk). Exit 0 clean, 1 on any hit, 2 if the
    tree is unreadable -- the same grammar as ``audit_range``.
    """
    import subprocess

    result = subprocess.run(
        ["git", "-C", root, "ls-files"],
        capture_output=True, encoding="utf-8", errors="replace",
    )
    if result.returncode != 0:
        msg = f"git ls-files failed in {root}: {result.stderr.strip()}"
        if as_json:
            print(json.dumps({"schema": "fleet-public-leak-scan/1", "ok": False,
                              "root": root, "error": msg}))
        else:
            print(msg, file=sys.stderr)
        return 2

    priv = load_private_needles(root)
    real_needles = sorted({n for n in (priv.get("export_audit_needles") or []) if n}) if priv else []
    mode = "full" if real_needles else "shape-only"

    misses = []
    scanned = 0
    for rel in result.stdout.splitlines():
        rel = rel.strip()
        if not rel or rel in SELF_REFERENTIAL:
            continue
        full = os.path.join(root, rel.replace("/", os.sep))
        if not is_text(full):
            continue
        content, _enc = read_text(full)
        if content is None:
            continue
        scanned += 1
        content_l = content.lower()
        for needle in real_needles:
            if needle.lower() in content_l:
                misses.append({"file": rel, "needle": needle, "kind": "needle"})
        for rx, label in AUDIT_REGEXES:
            if rx.search(content):
                misses.append({"file": rel, "needle": label, "kind": "shape"})

    ok = not misses
    note = "" if real_needles else " [shape-only: run tools/pull_scan_needles.py for a full scan]"
    if as_json:
        print(json.dumps({
            "schema": "fleet-public-leak-scan/1",
            "ok": ok,
            "mode": mode,
            "root": root,
            "scanned": scanned,
            "private_needles_loaded": bool(real_needles),
            "needle_count": len(real_needles),
            "miss_count": len(misses),
            "misses": misses[:50],
            "reason": ("clean" if ok else f"{len(misses)} redact-needle hit(s)") + note,
        }))
    else:
        tag = "full" if real_needles else "shape-only (run tools/pull_scan_needles.py for full)"
        if misses:
            print(f"FOUND {len(misses)} redact-needle hit(s) in tracked tree {root} [{tag}]:")
            for m in misses:
                print(f"  {m['file']}  [{m['needle']}]  ({m['kind']})")
        else:
            print(f"leak-scan: tracked tree clean ({scanned} text files, mode={tag})")
    return 0 if ok else 1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--export-dir",
        default=os.environ.get(
            "EXPORT_DIR",
            "/var/folders/z2/qgyzc8gx6kn1mkh6kfhlb6qw0000gp/T/opencode/fleet-public-export",
        ),
        help="dir of the git-archive export to scrub in place",
    )
    ap.add_argument(
        "--audit-staged",
        action="store_true",
        help="pre-commit mode: scan staged additions in --root for AUDIT_NEEDLES",
    )
    ap.add_argument(
        "--audit-range",
        metavar="BASE..HEAD",
        default=None,
        help="CI mode: scan ADDED lines of a commit range in --root for AUDIT_NEEDLES",
    )
    ap.add_argument(
        "--audit-tree",
        action="store_true",
        help="hard-cut backstop: scan every git-tracked text file in --root "
             "(shape regexes always; real needles when pulled via pull_scan_needles.py)",
    )
    ap.add_argument(
        "--json",
        action="store_true",
        help="emit machine-readable JSON (for the control-pane public-leak-scan loop)",
    )
    ap.add_argument(
        "--root",
        default=".",
        help="repo root for --audit-staged / --audit-range / --audit-tree (default: cwd)",
    )
    args = ap.parse_args()

    if args.audit_staged:
        return audit_staged(os.path.abspath(args.root))
    if args.audit_range:
        return audit_range(os.path.abspath(args.root), args.audit_range)
    if args.audit_tree:
        return audit_tree(os.path.abspath(args.root), as_json=args.json)

    export_dir = args.export_dir
    if not os.path.isdir(export_dir):
        print(f"ERROR: export dir not found: {export_dir}", file=sys.stderr)
        return 2

    deleted_files, deleted_dirs = [], []
    for rel in DELETE_PATHS:
        abs_path = os.path.join(export_dir, rel)
        if os.path.isdir(abs_path):
            n = sum(1 for _r, _d, fs in os.walk(abs_path) for _ in fs)
            shutil.rmtree(abs_path)
            deleted_dirs.append((rel, n))
        elif os.path.isfile(abs_path):
            os.remove(abs_path)
            deleted_files.append(rel)
    for pat in DELETE_GLOBS:
        for match in expand_glob(export_dir, pat):
            if os.path.isfile(match):
                deleted_files.append(os.path.relpath(match, export_dir))
                os.remove(match)

    touched = {}
    for dirpath, _d, filenames in os.walk(export_dir):
        for name in filenames:
            full = os.path.join(dirpath, name)
            if not is_text(full):
                continue
            original, enc = read_text(full)
            if original is None:
                continue
            changed = original
            file_touches = {}
            for needle, replacement, desc in REPLACEMENTS:
                if needle in changed:
                    n = changed.count(needle)
                    changed = changed.replace(needle, replacement)
                    file_touches[desc] = file_touches.get(desc, 0) + n
            for needle, replacement in CASE_INSENSITIVE_REPLACEMENTS:
                pat = re.compile(re.escape(needle), re.IGNORECASE)
                hits = pat.findall(changed)
                if hits:
                    changed = pat.sub(replacement, changed)
                    desc = f"{needle} -> {replacement} (case-insensitive)"
                    file_touches[desc] = file_touches.get(desc, 0) + len(hits)
            if changed != original:
                with open(full, "w", encoding=enc) as f:
                    f.write(changed)
                touched[os.path.relpath(full, export_dir)] = file_touches

    # Strip PRIVATE_MACHINE_PREFIXES (DGX) runs from the aggregate catalog.json.
    # DELETE_GLOBS already removed the run dirs; this removes their surviving
    # metadata from the committed catalog so DGX results stay private by default.
    catalog_dropped = _strip_private_machines_from_catalog(export_dir)

    # Rename owner-named directories bottom-up (deepest first) so a renamed
    # parent does not invalidate its children's paths. Content replacement
    # above already rewrote any in-file references to the new label.
    renamed_dirs = []
    all_dirs = []
    for dirpath, _d, _f in os.walk(export_dir):
        all_dirs.append(dirpath)
    for dirpath in sorted(all_dirs, key=lambda p: p.count(os.sep), reverse=True):
        name = os.path.basename(dirpath.rstrip(os.sep))
        if name in DIR_RENAME:
            new_path = os.path.join(os.path.dirname(dirpath), DIR_RENAME[name])
            if dirpath != new_path and not os.path.exists(new_path):
                os.rename(dirpath, new_path)
                renamed_dirs.append(
                    (os.path.relpath(dirpath, export_dir).replace(os.sep, "/"),
                     os.path.relpath(new_path, export_dir).replace(os.sep, "/"))
                )

    print("=" * 72)
    print("PUBLIC-COPY SCRUB SUMMARY")
    print("=" * 72)
    if renamed_dirs:
        print(f"\nRenamed {len(renamed_dirs)} owner-named director{'y' if len(renamed_dirs)==1 else 'ies'}:")
        for old, new in sorted(renamed_dirs):
            print(f"  - {old}  ->  {new}")
    print(f"\nDeleted {len(deleted_dirs)} director{'y' if len(deleted_dirs)==1 else 'ies'}:")
    for rel, n in sorted(deleted_dirs):
        print(f"  - {rel}  ({n} files)")
    print(f"\nDeleted {len(deleted_files)} files:")
    for rel in sorted(deleted_files):
        print(f"  - {rel}")
    if catalog_dropped:
        print(f"\nStripped {catalog_dropped} private-machine run(s) "
              f"({'/'.join(PRIVATE_MACHINE_PREFIXES)}*) from catalog.json")
    print(f"\nText replacements in {len(touched)} files:")
    by_desc = {}
    for _r, touches in touched.items():
        for desc, n in touches.items():
            by_desc[desc] = by_desc.get(desc, 0) + n
    for desc, total in sorted(by_desc.items()):
        print(f"  - {total:4d}x  {desc}")

    print("\n" + "=" * 72)
    print("POST-SCRUB AUDIT (any hit below is a MISS)")
    print("=" * 72)
    misses = 0
    for dirpath, _d, filenames in os.walk(export_dir):
        for name in filenames:
            full = os.path.join(dirpath, name)
            if not is_text(full):
                continue
            # Normalize to forward slashes: SELF_REFERENTIAL keys are POSIX-style,
            # but os.path.relpath yields OS-native separators (backslash on
            # Windows), which would otherwise defeat the exemption there.
            rel = os.path.relpath(full, export_dir).replace(os.sep, "/")
            if rel in SELF_REFERENTIAL:
                continue
            content, _enc = read_text(full)
            if content is None:
                continue
            # Match case-insensitively: hostnames (`node-desktop-b`) and names
            # (`node-macos-a`) vary by case; the audit must never miss a variant.
            content_l = content.lower()
            for needle in EXPORT_AUDIT_NEEDLES:
                if needle.lower() in content_l:
                    print(f"  MISS: {needle} in {rel}")
                    misses += 1
            for rx, label in AUDIT_REGEXES:
                if rx.search(content):
                    print(f"  MISS: {label} in {rel}")
                    misses += 1
    if misses == 0:
        print("  (clean)")
    return 0 if misses == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
