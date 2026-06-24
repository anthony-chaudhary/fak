#!/usr/bin/env python3
"""claims-salience-register — the first real consumer of the `dos salience` verdict.

`dos salience` (dos-kernel docs/391) is the "is this TRUE thing LIVE, or
true-but-PARKED out of the hotpath?" verdict — the prevent-silent-loss dual of
`retire`. The 2026-06-24 usefulness audit
(`docs/notes/DOS-SALIENCE-USEFULNESS-AUDIT-2026-06-24.md`) found the verb **sound but
LATENT**: it is built and correct, but *nothing routes on the verdict* — no picker,
assembler, reviewer, hook, or CI step reads a salience verdict, so its "never silently
lose a true thing" guarantee lives only in the type system, never in a wired consumer.

This tool is that wired consumer — the audit's smallest-first recommendation, built.
It treats fak's honesty ledger (`CLAIMS.md`) as the corpus of true-but-variably-LIVE
items and ROUTES on the salience verdict:

  * `[SHIPPED]`   — real code on the critical path  ⇒ evidence (reachable, default_on)
                    that classifies **LIVE** (the controls).
  * `[SIMULATED]` — the seam is real, the numbers are illustrative (no GPU / no live
                    engine on the box) ⇒ a host-declared park reason `SIMULATED` that
                    classifies **PARKED** — true at its stated scope, out of the
                    production hotpath, RETAINED + surfaced with a recovery line.
  * `[STUB]`      — plumbing present, behavior deferred ⇒ a host-declared park reason
                    `STUB` ⇒ **PARKED**, likewise recoverable.

The `[SIMULATED]`/`[STUB]` claims are exactly the lines a cleanup could silently drop
"as if they were false," when they are merely *not, today, on the hot path*. Run
through `dos.salience.partition` they become a typed, recoverable **parked register** —
the prevent-silent-loss guarantee made operational, not just type-checked.

How it ROUTES on the verdict (what makes this a consumer, not another latent surface):

  1. **No-loss invariant, asserted.** Every claim line parsed from `CLAIMS.md` becomes
     exactly one `SalienceEvidence`, and `partition` routes each into exactly one of
     {live, parked, indeterminate}: `partition.total == len(claims)`, with nothing
     dropped between the ledger and the fold. A parse/filter bug that lost a claim is
     caught here — the no-loss count is now a *watched number*, not a type-only fact.
  2. **Cross-checked against the ledger.** live == #[SHIPPED], parked == #[SIMULATED] +
     #[STUB], and the parked reason-class counts equal the per-tag counts in CLAIMS.md.
     So the register cannot drift from the honesty ledger or be gamed by editing it
     alone — add/remove a `[STUB]` claim and the register (and the gate) move with it.
  3. **Surfaced, recoverable register.** The PARKED bucket is printed (and committed to
     `docs/claims-salience-register.md`) with each row's typed `reason_class` and its
     `reactivation` line — the kernel's generic recovery line plus fak's specific
     "promote to [SHIPPED] when …" affordance. A parked truth sits in a file, with its
     reason and its path back — never silently lost.

It uses the REAL kernel verb: it `import`s `dos.salience` (the pure, I/O-free leaf;
its no-loss `partition` fold is import-only — the CLI is single-item) and calls
`partition` / `classify` / `reactivation_for` directly. Where the dos kernel is not
importable (e.g. hermetic CI), `--check` degrades to an advisory SKIP (exit 0) — it is
a real gate where dos is present (the dev trunk, the fleet hosts) and a no-op where it
is not, so it joins `make ci` without ever breaking a hermetic build.

Deterministic + read-only over `CLAIMS.md` (two clones at one commit render
identically); the only disk write is the generated register doc under `--markdown`.

Run from the repo ROOT::

    python tools/claims_salience_register.py                 # human-readable parked register
    python tools/claims_salience_register.py --json          # machine payload (control-pane / doctor)
    python tools/claims_salience_register.py --check         # gate: no-loss + cross-check (advisory-skip w/o dos)
    python tools/claims_salience_register.py --markdown docs/claims-salience-register.md   # regenerate the doc
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

SCHEMA = "fak-claims-salience/1"
CLAIMS_REL = "CLAIMS.md"
GENERATED_DOC_REL = "docs/claims-salience-register.md"
AUDIT_REL = "docs/notes/DOS-SALIENCE-USEFULNESS-AUDIT-2026-06-24.md"
SPEC_REF = "dos-kernel docs/391 (salience)"

# The three honesty-ledger tags, and the salience axis each lands on. SHIPPED is on the
# critical path (LIVE); SIMULATED/STUB are true-at-stated-scope but parked out of the
# production hotpath. This mirrors the CLAIMS.md legend + Makefile `claims-lint` rule:
# a claim line is `^- [TAG] …`; the legend lines themselves use `- `[TAG]`` (backtick),
# so the bracket-immediately-after-dash rule excludes them.
LIVE_TAG = "SHIPPED"
PARK_TAGS = ("SIMULATED", "STUB")
ALL_TAGS = (LIVE_TAG,) + PARK_TAGS
_CLAIM_RE = re.compile(r"^\s*-\s*\[(SHIPPED|SIMULATED|STUB)\]\s*(.*)$")
_SECTION_RE = re.compile(r"^##\s+(.*)$")

# fak's own reactivation affordance per host-declared park reason — the "how to pull
# this back onto the hotpath" line layered over the kernel's generic recovery string.
# This is the load-bearing distinction from `retire` (evict-to-archive, no re-entry):
# every parked claim carries a defined, cheap path back to [SHIPPED].
FAK_REACTIVATION = {
    "SIMULATED": (
        "promote to [SHIPPED] when the real path is exercised on live hardware / a live "
        "engine and the illustrative numbers become measured — the seam is already real"
    ),
    "STUB": (
        "promote to [SHIPPED] when the deferred behavior is built behind the present "
        "plumbing — the labeled no-op is the placeholder, not a missing seam"
    ),
}


def _slug(text: str, *, limit: int = 56) -> str:
    """A deterministic kebab label from a claim's lead text (markdown stripped).

    Used as the `SalienceEvidence.label` so a surfaced verdict / partition row
    self-joins back to the claim. Stable under re-run; uniqueness is enforced by the
    caller (a numeric suffix on collision), so order-determinism is preserved."""
    t = text.lower()
    t = re.sub(r"[`*_\[\]()]", " ", t)        # drop markdown punctuation, KEEP its text
    t = re.sub(r"[^a-z0-9]+", "-", t).strip("-")
    return t[:limit].strip("-") or "claim"


def parse_claims(text: str) -> list[dict[str, str]]:
    """Every tagged claim line in CLAIMS.md → {tag, text, section, label}, in order.

    The match rule is the Makefile `claims-lint` / product_scorecard rule exactly: a
    line `^- [SHIPPED|SIMULATED|STUB] …` (legend lines use a backtick and are excluded).
    Labels are made unique with a deterministic numeric suffix on collision."""
    claims: list[dict[str, str]] = []
    section = ""
    seen: dict[str, int] = {}
    for line in text.splitlines():
        sm = _SECTION_RE.match(line)
        if sm:
            section = sm.group(1).strip()
            continue
        m = _CLAIM_RE.match(line)
        if not m:
            continue
        tag, body = m.group(1), m.group(2).strip()
        base = _slug(body)
        n = seen.get(base, 0)
        seen[base] = n + 1
        label = base if n == 0 else f"{base}-{n + 1}"
        claims.append({"tag": tag, "text": body, "section": section, "label": label})
    return claims


def tag_counts(claims: list[dict[str, str]]) -> dict[str, int]:
    """Per-tag claim counts — the ledger truth the register is cross-checked against."""
    counts = {t: 0 for t in ALL_TAGS}
    for c in claims:
        counts[c["tag"]] = counts.get(c["tag"], 0) + 1
    return counts


# ---------------------------------------------------------------------------
# The dos-gated core — uses the REAL `dos.salience` verb (import-only `partition`).
# ---------------------------------------------------------------------------


def load_salience() -> tuple[Any, str]:
    """Import the real `dos.salience` leaf. Returns (module, origin_path) or (None, "").

    Pure leaf, no I/O — importing it is the established fak pattern for a stateless dos
    verb (stateful verbs shell to the `dos` CLI). The no-loss `partition` fold is
    import-only (the CLI is single-item), so this is the only way to sweep the corpus."""
    try:
        from dos import salience as sal  # type: ignore
    except Exception:
        return None, ""
    return sal, getattr(sal, "__file__", "") or ""


def _evidence_for(claim: dict[str, str], sal: Any) -> Any:
    """One claim → one `SalienceEvidence`. SHIPPED gets the LIVE bits (reachable +
    default_on); SIMULATED/STUB get a host-DECLARED park reason (the open extension
    point over the kernel's generic classes — preserving fak's own taxonomy)."""
    if claim["tag"] == LIVE_TAG:
        return sal.SalienceEvidence(label=claim["label"], reachable=True, default_on=True)
    return sal.SalienceEvidence(label=claim["label"], declared_reason=claim["tag"])


def classify_corpus(claims: list[dict[str, str]], sal: Any) -> dict[str, Any]:
    """Route the whole ledger through `dos.salience.partition` and join verdicts back.

    Returns {partition, rows, by_state}. `rows` is one record per claim joined to its
    verdict (state, reason_class, kernel + fak reactivation, reason). The PARKED rows
    are the recoverable register; nothing is dropped (the partition's no-loss fold)."""
    by_label = {c["label"]: c for c in claims}
    evidence = [_evidence_for(c, sal) for c in claims]
    part = sal.partition(evidence)  # GENERIC_SALIENCE_POLICY (park_declared armed)
    rows: list[dict[str, Any]] = []
    for bucket in (part.live, part.parked, part.indeterminate):
        for v in bucket:
            claim = by_label.get(v.label, {})
            rows.append({
                "label": v.label,
                "tag": claim.get("tag", ""),
                "section": claim.get("section", ""),
                "text": claim.get("text", ""),
                "state": v.state.value,
                "reason_class": v.reason_class,
                "reactivation": v.reactivation,                       # kernel's generic line
                "fak_reactivation": FAK_REACTIVATION.get(v.reason_class, ""),
                "reason": v.reason,
            })
    rows.sort(key=lambda r: (r["state"], r["reason_class"], r["label"]))
    return {
        "partition": part,
        "rows": rows,
        "by_state": {"LIVE": len(part.live), "PARKED": len(part.parked),
                     "INDETERMINATE": len(part.indeterminate)},
    }


def verify_invariants(claims: list[dict[str, str]], result: dict[str, Any]) -> list[str]:
    """The checks that make 'never silently lose' a watched number, not a type fact.

    (1) NO-LOSS: partition.total == #claims (nothing dropped ledger→fold).
    (2) CROSS-CHECK vs the ledger: live==#SHIPPED, parked==#SIMULATED+#STUB,
        indeterminate==0, and each parked reason_class count == its CLAIMS tag count.
    (3) RECOVERABILITY: every PARKED row carries a non-empty reactivation line (kernel
        AND fak) — a parked thing that cannot be recovered is just a slow drop."""
    violations: list[str] = []
    counts = tag_counts(claims)
    part = result["partition"]
    by_state = result["by_state"]
    n = len(claims)

    if part.total != n:
        violations.append(f"no-loss VIOLATED: {n} claims parsed but partition.total={part.total} "
                          f"({n - part.total} silently dropped)")
    if by_state["LIVE"] != counts[LIVE_TAG]:
        violations.append(f"live={by_state['LIVE']} but #[{LIVE_TAG}]={counts[LIVE_TAG]}")
    parked_expected = counts["SIMULATED"] + counts["STUB"]
    if by_state["PARKED"] != parked_expected:
        violations.append(f"parked={by_state['PARKED']} but #[SIMULATED]+#[STUB]={parked_expected}")
    if by_state["INDETERMINATE"] != 0:
        violations.append(f"indeterminate={by_state['INDETERMINATE']} (every ledger claim carries "
                          f"a tag, so none should abstain)")

    parked_rows = [r for r in result["rows"] if r["state"] == "PARKED"]
    for tag in PARK_TAGS:
        got = sum(1 for r in parked_rows if r["reason_class"] == tag)
        if got != counts[tag]:
            violations.append(f"reason_class {tag}: register has {got} but #[{tag}]={counts[tag]}")
    for r in parked_rows:
        if not r["reactivation"] or not r["fak_reactivation"]:
            violations.append(f"parked {r['label']!r} lacks a reactivation line (not recoverable)")
    return violations


# ---------------------------------------------------------------------------
# Payload + renderers.
# ---------------------------------------------------------------------------


def build_payload(workspace: str, dos_origin: str, claims: list[dict[str, str]],
                  result: dict[str, Any] | None, violations: list[str],
                  skipped: bool) -> dict[str, Any]:
    counts = tag_counts(claims)
    payload: dict[str, Any] = {
        "schema": SCHEMA,
        "workspace": workspace,
        "spec": SPEC_REF,
        "dos_origin": dos_origin,
        "dos_available": not skipped,
        "claims_total": len(claims),
        "tag_counts": counts,
    }
    if skipped or result is None:
        payload["skipped"] = True
        payload["reason"] = "dos kernel not importable — salience register skipped (advisory)"
        return payload
    parked = [r for r in result["rows"] if r["state"] == "PARKED"]
    payload.update({
        "skipped": False,
        "by_state": result["by_state"],
        "no_loss_ok": result["partition"].total == len(claims),
        "by_reason_class": {t: sum(1 for r in parked if r["reason_class"] == t) for t in PARK_TAGS},
        "violations": violations,
        "ok": not violations,
        "parked": [{"label": r["label"], "tag": r["tag"], "reason_class": r["reason_class"],
                    "section": r["section"], "fak_reactivation": r["fak_reactivation"]}
                   for r in parked],
    })
    return payload


def render(payload: dict[str, Any], result: dict[str, Any] | None) -> str:
    lines: list[str] = []
    if payload.get("skipped"):
        lines.append("claims-salience-register: SKIPPED (advisory)")
        lines.append(f"  {payload.get('reason')}")
        lines.append(f"  ledger: {payload['claims_total']} claims "
                     f"({_fmt_counts(payload['tag_counts'])})")
        return "\n".join(lines)

    by = payload["by_state"]
    ok = payload["ok"]
    lines.append(f"claims-salience-register: {'OK' if ok else 'VIOLATIONS'} "
                 f"— routed {payload['claims_total']} CLAIMS.md claims through `dos salience`")
    lines.append(f"  using dos.salience from {payload['dos_origin']}")
    lines.append(f"  LIVE {by['LIVE']}  ·  PARKED {by['PARKED']}  ·  INDETERMINATE "
                 f"{by['INDETERMINATE']}  ·  no-loss {'✓' if payload['no_loss_ok'] else '✗'} "
                 f"({payload['claims_total']} in → {by['LIVE'] + by['PARKED'] + by['INDETERMINATE']} "
                 f"classified, 0 dropped)")
    lines.append("")
    lines.append("  PARKED register — true-but-not-on-the-hotpath, RETAINED + recoverable:")
    parked_rows = [r for r in (result["rows"] if result else []) if r["state"] == "PARKED"]
    for tag in PARK_TAGS:
        group = [r for r in parked_rows if r["reason_class"] == tag]
        if not group:
            continue
        lines.append(f"    [{tag}] ({len(group)}) — {FAK_REACTIVATION.get(tag, '')}")
        for r in group:
            lines.append(f"      · {r['label']}  ({r['section']})")
    if payload["violations"]:
        lines.append("")
        lines.append("  VIOLATIONS:")
        for v in payload["violations"]:
            lines.append(f"    ✗ {v}")
    return "\n".join(lines)


def _fmt_counts(counts: dict[str, int]) -> str:
    return ", ".join(f"{c}×[{t}]" for t, c in counts.items() if c)


def check_summary(payload: dict[str, Any]) -> str:
    """The concise one/two-line gate line for `--check` (the full register is for the
    default human render). States the verdict, the no-loss proof, and any violations."""
    if payload.get("skipped"):
        return ("claims-salience-register: SKIP (dos kernel not importable; advisory) — "
                f"ledger {payload['claims_total']} claims ({_fmt_counts(payload['tag_counts'])})")
    by = payload["by_state"]
    head = (f"claims-salience-register: {'OK' if payload['ok'] else 'VIOLATIONS'} — "
            f"{payload['claims_total']} claims routed through `dos salience` "
            f"(LIVE {by['LIVE']} · PARKED {by['PARKED']} · no-loss "
            f"{'✓' if payload['no_loss_ok'] else '✗'}); {by['PARKED']} parked truths "
            "surfaced & recoverable")
    if payload["violations"]:
        return head + "\n" + "\n".join(f"  ✗ {v}" for v in payload["violations"])
    return head


def render_markdown(payload: dict[str, Any], result: dict[str, Any]) -> str:
    """The committed surfaced register — the parked truths sitting in a file, recoverable.

    Generated, deterministic; regenerate with `--markdown`. This file IS the
    prevent-silent-loss guarantee made concrete: each parked claim is here with its
    typed reason and its path back to [SHIPPED]."""
    by = payload["by_state"]
    counts = payload["tag_counts"]
    out: list[str] = []
    out.append("---")
    out.append("title: \"CLAIMS salience register — fak's true-but-parked claims, recoverable\"")
    out.append("description: \"Generated by tools/claims_salience_register.py — every "
               "[SIMULATED]/[STUB] claim in CLAIMS.md routed through the dos-kernel "
               "salience verdict (docs/391) into a typed, recoverable parked register so a "
               "true-but-not-yet-LIVE claim is never silently dropped. Do not edit by hand.\"")
    out.append("---")
    out.append("")
    out.append("# CLAIMS salience register")
    out.append("")
    out.append(f"Generated by `tools/claims_salience_register.py` — the first wired consumer of "
               f"the `dos salience` verdict ({SPEC_REF}). It routes every claim in "
               f"[`CLAIMS.md`](../CLAIMS.md) through `dos.salience.partition`: `[SHIPPED]` "
               f"classifies **LIVE**, while `[SIMULATED]`/`[STUB]` — true at their stated "
               f"scope but off the production hotpath — classify **PARKED** and land in the "
               f"recoverable register below. See the "
               f"[usefulness audit](notes/DOS-SALIENCE-USEFULNESS-AUDIT-2026-06-24.md).")
    out.append("")
    out.append("**The contract:** `PARKED ≠ dropped`. Every parked claim is RETAINED here "
               "with its typed reason and a re-activation line — never silently lost on a "
               "cleanup. The no-loss invariant is asserted on every run: "
               f"`{payload['claims_total']}` claims in → "
               f"`{by['LIVE'] + by['PARKED'] + by['INDETERMINATE']}` classified, **0 dropped**.")
    out.append("")
    out.append(f"- **LIVE** (`[SHIPPED]`, on the critical path): **{by['LIVE']}**")
    out.append(f"- **PARKED** (`[SIMULATED]`+`[STUB]`, true-but-not-hotpath): **{by['PARKED']}**")
    out.append(f"- **INDETERMINATE**: **{by['INDETERMINATE']}**")
    out.append("")
    parked_rows = [r for r in result["rows"] if r["state"] == "PARKED"]
    for tag in PARK_TAGS:
        group = [r for r in parked_rows if r["reason_class"] == tag]
        if not group:
            continue
        out.append(f"## PARKED · `{tag}` ({len(group)})")
        out.append("")
        out.append(f"_Reactivation: {FAK_REACTIVATION.get(tag, '')}_")
        out.append("")
        out.append("| claim | section |")
        out.append("|---|---|")
        for r in group:
            text = r["text"].replace("|", "\\|")
            text = (text[:140] + "…") if len(text) > 140 else text
            section = r["section"].replace("|", "\\|") or "—"
            out.append(f"| {text} | {section} |")
        out.append("")
    out.append("---")
    out.append(f"_Cross-checked against the ledger: LIVE == {counts[LIVE_TAG]}×`[SHIPPED]`, "
               f"PARKED == {counts['SIMULATED']}×`[SIMULATED]` + {counts['STUB']}×`[STUB]`. "
               "Regenerate with `python tools/claims_salience_register.py --markdown "
               f"{GENERATED_DOC_REL}`._")
    out.append("")
    return "\n".join(out)


# ---------------------------------------------------------------------------
# Driver.
# ---------------------------------------------------------------------------


def collect(workspace: Path) -> tuple[dict[str, Any], dict[str, Any] | None, list[dict[str, str]]]:
    """Parse the ledger, classify it via the real dos verb (or note the skip), and
    fold the payload. Returns (payload, result_or_None, claims)."""
    root = workspace.resolve()
    claims_path = root / CLAIMS_REL
    text = claims_path.read_text(encoding="utf-8") if claims_path.exists() else ""
    claims = parse_claims(text)
    sal, origin = load_salience()
    if sal is None:
        payload = build_payload(str(root), "", claims, None, [], skipped=True)
        return payload, None, claims
    result = classify_corpus(claims, sal)
    violations = verify_invariants(claims, result)
    payload = build_payload(str(root), origin, claims, result, violations, skipped=False)
    return payload, result, claims


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--workspace", default=None, help="repo root (default: the parent of tools/)")
    ap.add_argument("--json", action="store_true", help="machine payload (control-pane / doctor)")
    ap.add_argument("--check", action="store_true",
                    help="gate: exit 1 on a no-loss/cross-check violation; advisory-skip (exit 0) without dos")
    ap.add_argument("--markdown", metavar="PATH", default=None,
                    help="regenerate the committed register doc at PATH")
    args = ap.parse_args(argv)

    root = Path(args.workspace) if args.workspace else Path(__file__).resolve().parent.parent
    payload, result, _claims = collect(root)

    if args.markdown:
        if payload.get("skipped"):
            print("claims-salience-register: dos kernel not importable — cannot regenerate the "
                  "register doc (it needs the real verdict). Skipped.", file=sys.stderr)
            return 0
        out = render_markdown(payload, result or {})
        Path(args.markdown).write_text(out, encoding="utf-8")
        print(f"claims-salience-register: wrote {args.markdown} "
              f"({payload['by_state']['PARKED']} parked claims surfaced)")
        return 0

    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    elif args.check:
        print(check_summary(payload))           # concise gate line (advisory-skip-aware)
    else:
        print(render(payload, result))          # the full surfaced register

    if args.check and not payload.get("skipped") and payload["violations"]:
        return 1                                # a real no-loss / cross-check violation
    return 0                                     # OK, or advisory SKIP where dos is absent


if __name__ == "__main__":
    raise SystemExit(main())
