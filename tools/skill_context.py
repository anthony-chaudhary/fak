#!/usr/bin/env python3
"""skill_context.py — skill output as a content-addressed, witnessed memory view (#433).

A skill is already a procedural memory capsule, but its *output* is not yet a
verified virtual memory view. Today a second agent either re-reads the whole
``SKILL.md`` + helper context, or trusts the producing agent's prose handoff.
This tool turns one skill run into a ``SkillContextRecord``: a compact,
digest-keyed projection a consumer can trust *only after re-running the witness*,
never from the producer's prose.

The record is schema-compatible with ``internal/contextq``'s ``MemoryViewRecord``
(view_id / view_type / source_digests / producer / scope / taint / coverage /
faithfulness_probe / labels) and adds the skill-specific keying the issue names:
skill digest, input-tree digest, policy digest, allowed tools, projection,
claims, and a witness command + verdict.

The trust contract is the same fail-closed shape as the Go view path: a consumer
``--check`` REJECTS a record the moment the skill bytes, the referenced input
digests, the policy digest, or the witness verdict no longer match — so a stale
or tampered record can never be paged in as if fresh.

    python tools/skill_context.py --list                 # skills that can emit a record
    python tools/skill_context.py --emit quality-score   # emit a record + witness summary
    python tools/skill_context.py --emit quality-score --out rec.json
    python tools/skill_context.py --check rec.json       # consumer check: exit 1 on any drift

Read-only except for the ``--out`` file it is asked to write. Model-free and
deterministic: every digest and the witness verdict are recomputed from the
bytes on disk, so the same tree always yields the same record.
"""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import re
import sys
from pathlib import Path

SCHEMA = "fak/skill-context-record@1"
VIEW_TYPE = "skill_context"

# Tools that mutate the tree — the same allowlist skill_frontmatter_lint.py uses
# to decide whether a skill declares a read-only boundary.
MUTATING_TOOLS = ("Write", "Edit", "NotebookEdit")


def repo_root() -> Path:
    # tools/skill_context.py -> repo root is the parent of tools/.
    return Path(__file__).resolve().parent.parent


def sha256_bytes(b: bytes) -> str:
    return "sha256:" + hashlib.sha256(b).hexdigest()


def sha256_file(p: Path) -> str:
    return sha256_bytes(p.read_bytes())


def short(digest: str) -> str:
    h = digest.split(":", 1)[-1]
    return h[:12]


# --------------------------------------------------------------------------- #
# Frontmatter + helper discovery
# --------------------------------------------------------------------------- #
def parse_frontmatter(text: str) -> dict:
    """Parse the leading ``---`` YAML-ish block into a flat dict. Only the simple
    ``key: value`` shape the skill pack uses is supported; that is all the witness
    and the policy digest need."""
    if not text.startswith("---"):
        return {}
    end = text.find("\n---", 3)
    if end == -1:
        return {}
    block = text[3:end]
    fm: dict[str, str] = {}
    for line in block.splitlines():
        line = line.rstrip()
        if not line or line.lstrip().startswith("#") or ":" not in line:
            continue
        if line[0] in " \t":  # nested scalar (e.g. under metadata:) — skip
            continue
        key, _, val = line.partition(":")
        val = val.strip()
        if val and val[0] in "\"'" and val[-1] == val[0]:
            val = val[1:-1]
        fm[key.strip()] = val
    return fm


def allowed_tools(fm: dict) -> list[str]:
    raw = fm.get("allowed-tools", "")
    return [t.strip() for t in raw.split(",") if t.strip()]


_HELPER_RE = re.compile(r"tools/([A-Za-z0-9_./-]+\.py)")


def discover_helpers(text: str, root: Path) -> list[Path]:
    """Helper scripts a SKILL.md references by ``tools/<name>.py`` path and that
    actually exist on disk. Deterministic and deduplicated."""
    seen: dict[str, Path] = {}
    for m in _HELPER_RE.finditer(text):
        rel = "tools/" + m.group(1)
        p = (root / rel).resolve()
        if p.is_file() and rel not in seen:
            seen[rel] = p
    return [seen[k] for k in sorted(seen)]


# --------------------------------------------------------------------------- #
# Digests
# --------------------------------------------------------------------------- #
def _file_entries(root: Path, paths: list[Path]) -> list[dict]:
    out = []
    for p in paths:
        rel = p.resolve().relative_to(root).as_posix()
        out.append({"path": rel, "digest": sha256_file(p)})
    return out


def tree_digest(entries: list[dict]) -> str:
    """A single digest over a set of (path, digest) entries, order-independent."""
    lines = sorted(f"{e['path']}={e['digest']}" for e in entries)
    return sha256_bytes("\n".join(lines).encode("utf-8"))


def policy_digest(allowed: list[str], policy_entries: list[dict]) -> str:
    """Digest the policy posture: the allowed-tools boundary plus any declared
    policy files. Changing either flips the digest, so a consumer check rejects."""
    payload = "allowed-tools=" + ",".join(allowed) + "\n" + "\n".join(
        sorted(f"{e['path']}={e['digest']}" for e in policy_entries)
    )
    return sha256_bytes(payload.encode("utf-8"))


# --------------------------------------------------------------------------- #
# Witness — model-free, deterministic, tamper-sensitive
# --------------------------------------------------------------------------- #
def run_witness(skill_name: str, skill_md: Path) -> tuple[str, str]:
    """Validate a skill artifact and return (verdict, summary). The verdict is
    ``pass`` only when the skill structurally checks out: the SKILL.md exists, its
    frontmatter declares name + description, and the declared name matches the
    directory. Breaking the bytes (rename, gut the frontmatter) flips it to
    ``fail`` — the consumer check then rejects the record."""
    if not skill_md.is_file():
        return "fail", f"{skill_md} missing"
    text = skill_md.read_text(encoding="utf-8", errors="replace")
    fm = parse_frontmatter(text)
    problems = []
    if not fm.get("name"):
        problems.append("no name in frontmatter")
    elif fm["name"] != skill_name:
        problems.append(f"name {fm['name']!r} != dir {skill_name!r}")
    if not fm.get("description"):
        problems.append("no description in frontmatter")
    if problems:
        return "fail", "; ".join(problems)
    return "pass", f"skill {skill_name!r}: frontmatter ok, name matches dir"


# --------------------------------------------------------------------------- #
# Projection — what a consumer reads INSTEAD of the whole SKILL.md
# --------------------------------------------------------------------------- #
def build_projection(skill_name: str, fm: dict, helpers: list[dict]) -> str:
    desc = fm.get("description", "")
    first = desc.split(". ")[0].strip()
    if first and not first.endswith("."):
        first += "."
    lines = [
        f"skill: {skill_name}",
        f"summary: {first}",
        "helpers: " + (", ".join(e["path"] for e in helpers) or "(none)"),
        "allowed-tools: " + (", ".join(allowed_tools(fm)) or "(unrestricted)"),
    ]
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Record
# --------------------------------------------------------------------------- #
@dataclasses.dataclass
class SkillContextRecord:
    schema: str
    view_id: str
    view_type: str
    producer: str
    skill: str
    skill_digest: str
    input_tree_digest: str
    policy_digest: str
    allowed_tools: list
    inputs: list          # [{path, digest}] referenced helpers + declared inputs
    policy_files: list     # [{path, digest}]
    projection: str
    claims: list
    witness_command: str
    witness_verdict: str
    # MemoryViewRecord-compatible axes:
    source_digests: list
    scope: str
    taint: str
    coverage: float
    faithfulness_probe: float
    labels: dict

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)


def skill_dir(root: Path, skill_name: str) -> Path:
    return root / ".claude" / "skills" / skill_name


def build_record(root: Path, skill_name: str, *, producer: str = "skill-context",
                 extra_inputs: list[Path] | None = None,
                 policy_paths: list[Path] | None = None) -> SkillContextRecord:
    sd = skill_dir(root, skill_name)
    skill_md = sd / "SKILL.md"
    if not skill_md.is_file():
        raise FileNotFoundError(f"no SKILL.md for skill {skill_name!r} at {skill_md}")
    text = skill_md.read_text(encoding="utf-8", errors="replace")
    fm = parse_frontmatter(text)

    skill_dg = sha256_file(skill_md)
    helpers = discover_helpers(text, root)
    input_paths = list(helpers) + list(extra_inputs or [])
    input_entries = _file_entries(root, input_paths)
    pol_entries = _file_entries(root, list(policy_paths or []))
    allowed = allowed_tools(fm)

    in_tree_dg = tree_digest(input_entries)
    pol_dg = policy_digest(allowed, pol_entries)

    verdict, summary = run_witness(skill_name, skill_md)
    projection = build_projection(skill_name, fm, input_entries)

    src_bytes = float(len(text.encode("utf-8")))
    coverage = round(len(projection.encode("utf-8")) / src_bytes, 4) if src_bytes else 0.0

    read_only = bool(allowed) and not any(t in allowed for t in MUTATING_TOOLS)
    claims = [
        f"skill {skill_name!r} pinned at {short(skill_dg)}",
        f"{len(input_entries)} input(s), tree digest {short(in_tree_dg)}",
        f"policy {short(pol_dg)} ({'read-only' if read_only else 'mutating-or-unrestricted'})",
        f"witness {verdict}: {summary}",
    ]
    return SkillContextRecord(
        schema=SCHEMA,
        view_id=f"skill-context-{skill_name}-{short(skill_dg)}",
        view_type=VIEW_TYPE,
        producer=producer,
        skill=skill_name,
        skill_digest=skill_dg,
        input_tree_digest=in_tree_dg,
        policy_digest=pol_dg,
        allowed_tools=allowed,
        inputs=input_entries,
        policy_files=pol_entries,
        projection=projection,
        claims=claims,
        witness_command=f"python tools/skill_context.py --check <record> --skill {skill_name}",
        witness_verdict=verdict,
        source_digests=[skill_dg] + [e["digest"] for e in input_entries],
        scope="skill",
        taint="benign",
        coverage=coverage,
        # Faithful by construction: the projection's substantive tokens are
        # verbatim frontmatter / on-disk helper paths, with only fixed structural
        # scaffolding added. No model in the loop -> no hallucination surface.
        faithfulness_probe=1.0,
        labels={"skill": skill_name, "view_type": VIEW_TYPE},
    )


def verify_record(record: dict, root: Path) -> tuple[bool, list[str]]:
    """Consumer check. Recompute every keyed digest and re-run the witness from
    the bytes on disk; reject (return False + reasons) the moment any of the skill
    bytes, the input-tree digest, the policy digest, or the witness verdict no
    longer matches the record. This is the trust gate: a stale or tampered record
    fails closed and is never paged in as fresh."""
    reasons: list[str] = []
    skill_name = record.get("skill", "")
    sd = skill_dir(root, skill_name)
    skill_md = sd / "SKILL.md"

    if record.get("schema") != SCHEMA:
        reasons.append(f"schema {record.get('schema')!r} != {SCHEMA!r}")
    if not skill_md.is_file():
        reasons.append(f"skill {skill_name!r} no longer on disk")
        return False, reasons

    if sha256_file(skill_md) != record.get("skill_digest"):
        reasons.append("skill_digest changed (SKILL.md bytes differ)")

    # Recompute the input-tree digest from the record's own input list, reading
    # the current bytes of each named path.
    cur_inputs = []
    for e in record.get("inputs", []):
        p = root / e["path"]
        if not p.is_file():
            reasons.append(f"input {e['path']} missing")
            continue
        cur = sha256_file(p)
        cur_inputs.append({"path": e["path"], "digest": cur})
        if cur != e.get("digest"):
            reasons.append(f"input {e['path']} digest changed")
    if tree_digest(cur_inputs) != record.get("input_tree_digest"):
        reasons.append("input_tree_digest changed")

    cur_pol = []
    for e in record.get("policy_files", []):
        p = root / e["path"]
        if not p.is_file():
            reasons.append(f"policy file {e['path']} missing")
            continue
        cur_pol.append({"path": e["path"], "digest": sha256_file(p)})
    if policy_digest(record.get("allowed_tools", []), cur_pol) != record.get("policy_digest"):
        reasons.append("policy_digest changed")

    verdict, summary = run_witness(skill_name, skill_md)
    if verdict != record.get("witness_verdict"):
        reasons.append(f"witness verdict changed: {record.get('witness_verdict')} -> {verdict} ({summary})")
    if verdict != "pass":
        reasons.append(f"witness does not pass: {summary}")

    return (not reasons), reasons


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def list_skills(root: Path) -> list[str]:
    base = root / ".claude" / "skills"
    if not base.is_dir():
        return []
    return sorted(p.parent.name for p in base.glob("*/SKILL.md"))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--list", action="store_true", help="list skills that can emit a record")
    g.add_argument("--emit", metavar="SKILL", help="emit a SkillContextRecord for SKILL")
    g.add_argument("--check", metavar="RECORD", help="consumer check: verify a record JSON")
    ap.add_argument("--out", metavar="FILE", help="write the emitted record JSON to FILE")
    ap.add_argument("--input", action="append", default=[], metavar="PATH",
                    help="extra declared input file to pin (repeatable)")
    ap.add_argument("--policy", action="append", default=[], metavar="PATH",
                    help="policy file to pin into the policy digest (repeatable)")
    ap.add_argument("--workspace", default="", help="workspace root (default: repo root)")
    args = ap.parse_args(argv)

    root = Path(args.workspace).resolve() if args.workspace else repo_root()

    if args.emit:
        try:
            rec = build_record(
                root, args.emit,
                extra_inputs=[Path(p) if Path(p).is_absolute() else root / p for p in args.input],
                policy_paths=[Path(p) if Path(p).is_absolute() else root / p for p in args.policy],
            )
        except FileNotFoundError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1
        blob = json.dumps(rec.to_dict(), indent=2)
        if args.out:
            Path(args.out).write_text(blob + "\n", encoding="utf-8")
            print(f"wrote {args.out}", file=sys.stderr)
        else:
            print(blob)
        print(f"witness: {rec.witness_verdict} — {rec.claims[-1]}", file=sys.stderr)
        print(f"projection: {rec.coverage:.1%} of SKILL.md bytes", file=sys.stderr)
        return 0 if rec.witness_verdict == "pass" else 1

    if args.check:
        record = json.loads(Path(args.check).read_text(encoding="utf-8"))
        ok, reasons = verify_record(record, root)
        if ok:
            print(f"OK: record for skill {record.get('skill')!r} verifies "
                  f"(skill+inputs+policy+witness all match)")
            return 0
        print(f"REJECT: record for skill {record.get('skill')!r} no longer trustworthy:")
        for r in reasons:
            print(f"  - {r}")
        return 1

    skills = list_skills(root)
    print(f"{len(skills)} skill(s) under .claude/skills/:")
    for s in skills:
        print(f"  {s}")
    print("\nemit a record:  python tools/skill_context.py --emit <skill>")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
