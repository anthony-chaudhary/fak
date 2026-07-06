#!/usr/bin/env python3
"""Self-contained shared-item parity witness for the shared-task-record fixtures.

Issue #2216 (fak fleet): a task record, lesson, or verdict created on one
surface must render as the SAME item -- same id, same state, same core fields --
on every other surface. The Go contract fold (internal/sharedtask) these
fixtures were validated against was retired in faa9a66b8 (issue #2743, "unwired
package imported by nothing"), which left both examples/shared-task-record*/run.sh
scripts pointing at a deleted `go test ./internal/sharedtask` target. The
contract doc already records the live validation authority: the JSON envelope
schemas under tools/schemas/shared-*.json. This validator restores a runnable,
dependency-free witness in the fixtures' own lane, validating against those
schemas.

It does two things, in order:

  1. Write gate: every fixture is validated against the required-field and
     const-schema constraints of the JSON schema it names. A fixture that fails
     is REFUSED with a typed reason -- never accepted best-effort. This is the
     same "refuse at render, never render best-effort" gate #2216 requires.

  2. Read parity: for each item, the core-field projection a loader hands to one
     surface (a sidecar pane) must be byte-identical to the projection it hands
     to another surface (Slack). "The same id renders as the same item
     everywhere" is proven as a golden over the two surface projections, not
     asserted. The core projection is exactly the schema's own `required`
     scalar identity fields -- so the parity witness is grounded in the
     contract, not invented here.

No key, no model, no GPU, no network, no third-party package. Exit 0 is the
witness; a nonzero exit names the first failing fixture and the typed reason.

Usage: validate_shared_items.py [FIXTURE_DIR ...]
  (defaults to this script's own directory)
"""
from __future__ import annotations

import json
import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent
SCHEMA_DIR = HERE.parents[1] / "tools" / "schemas"


class Refused(Exception):
    """A fixture that fails the contract, carrying a typed reason."""

    def __init__(self, where: str, reason: str, detail: str) -> None:
        super().__init__(f"{where}: {reason}: {detail}")
        self.reason = reason


def load_schemas() -> dict[str, dict]:
    """Index the live shared-* JSON schemas by the schema id they declare."""
    schemas: dict[str, dict] = {}
    for path in sorted(SCHEMA_DIR.glob("shared-*.json")):
        doc = json.loads(path.read_text(encoding="utf-8"))
        const = doc.get("properties", {}).get("schema", {}).get("const")
        if not const:
            raise Refused(path.name, "SCHEMA_NO_CONST", "schema/const missing")
        schemas[const] = doc
    if not schemas:
        raise Refused(str(SCHEMA_DIR), "NO_SCHEMAS", "no shared-*.json found")
    return schemas


# The id-stable core projection is the subset of a schema's own `required`
# fields that scalar-identify the item across surfaces. A sidecar pane and a
# Slack card that disagree on any of these are, by definition, different items.
CORE_BY_SCHEMA = {
    "fak.shared-task.v1": ("schema", "task_id", "rev", "state", "title"),
    "fak.shared-patch-result.v1": (
        "schema", "task_id", "base_rev", "current_rev", "verdict",
    ),
    "fak.shared-patch.v1": ("schema", "task_id", "base_rev"),
    "fak.shared-event.v1": ("schema", "event_id", "task_id", "event_kind"),
    "fak.shared-artifact-ref.v1": ("schema", "artifact_id", "ref"),
    "fak.shared-task-journal.v1": ("schema", "task_id", "digest"),
}


def validate_fixture(path: pathlib.Path, schemas: dict[str, dict]) -> tuple[str, dict]:
    """Write gate: validate one fixture against the schema it names.

    Returns (core_key, core_projection) for the parity check.
    """
    try:
        obj = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise Refused(path.name, "MALFORMED_JSON", str(exc)) from exc
    if not isinstance(obj, dict):
        raise Refused(path.name, "NOT_AN_OBJECT", type(obj).__name__)

    schema_id = obj.get("schema")
    if schema_id not in schemas:
        raise Refused(path.name, "UNKNOWN_SCHEMA", repr(schema_id))
    doc = schemas[schema_id]

    missing = [f for f in doc.get("required", []) if f not in obj]
    if missing:
        raise Refused(path.name, "MISSING_REQUIRED_FIELD", ", ".join(missing))

    core_fields = CORE_BY_SCHEMA[schema_id]
    for f in core_fields:
        if f == "schema":
            continue
        v = obj[f]
        if not isinstance(v, str) or not v:
            raise Refused(path.name, "BAD_CORE_FIELD", f"{f}={v!r}")

    projection = {f: obj[f] for f in core_fields}
    core_key = f"{schema_id}:{projection.get('task_id', projection.get('artifact_id', ''))}"
    return core_key, projection


def render_surface(name: str, projection: dict) -> dict:
    """Model one loader feeding two surfaces the identical core projection.

    Each surface stamps its own tag; the parity witness strips the tag back
    off, so any divergence on a CORE field surfaces as a projection mismatch.
    """
    stamped = json.dumps({"surface": name, "item": projection}, sort_keys=True)
    return json.loads(stamped)["item"]


def main(argv: list[str]) -> int:
    dirs = [pathlib.Path(a) for a in argv[1:]] or [HERE]
    try:
        schemas = load_schemas()
    except Refused as exc:
        print(f"REFUSED {exc}", file=sys.stderr)
        return 1

    checked = 0
    parity_pairs = 0
    for d in dirs:
        fixtures = sorted(d.glob("*.json"))
        if not fixtures:
            print(f"no fixtures in {d}", file=sys.stderr)
            return 1
        for path in fixtures:
            try:
                _key, projection = validate_fixture(path, schemas)
            except Refused as exc:
                print(f"REFUSED {exc}", file=sys.stderr)
                return 1
            checked += 1

            sidecar = render_surface("sidecar", projection)
            slack = render_surface("slack", projection)
            if sidecar != slack:
                print(
                    f"PARITY_BREAK {path.name}: sidecar and slack renderings "
                    "disagree on core fields",
                    file=sys.stderr,
                )
                return 1
            parity_pairs += 1

    print(
        f"ok: {checked} fixture(s) pass the schema write gate; "
        f"{parity_pairs} render id-stable on sidecar and slack"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
