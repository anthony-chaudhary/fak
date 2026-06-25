#!/usr/bin/env python3
"""Validate the shared task record contract examples and fixtures."""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DOC = ROOT / "docs" / "shared-task-record-contract.md"
SCHEMA_DIR = ROOT / "tools" / "schemas"

SCHEMAS = {
    "fak.shared-task.v1": "shared-task.v1.json",
    "fak.shared-event.v1": "shared-event.v1.json",
    "fak.shared-patch.v1": "shared-patch.v1.json",
    "fak.shared-patch-result.v1": "shared-patch-result.v1.json",
    "fak.shared-artifact-ref.v1": "shared-artifact-ref.v1.json",
    "fak.shared-task-journal.v1": "shared-task-journal.v1.json",
}


class ValidationError(ValueError):
    pass


def load_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def load_schema(schema_name: str, schema_dir: Path = SCHEMA_DIR) -> dict[str, Any]:
    rel = SCHEMAS.get(schema_name)
    if not rel:
        raise ValidationError(f"unknown schema {schema_name!r}")
    schema = load_json(schema_dir / rel)
    if not isinstance(schema, dict):
        raise ValidationError(f"{schema_name}: schema is not an object")
    return schema


def resolve_ref(schema: dict[str, Any], root: dict[str, Any]) -> dict[str, Any]:
    ref = schema.get("$ref")
    if not ref:
        return schema
    node: Any = root
    for part in ref.removeprefix("#/").split("/"):
        node = node[part]
    if not isinstance(node, dict):
        raise ValidationError(f"ref {ref!r} does not resolve to an object")
    return node


def validate(instance: Any, schema: dict[str, Any], root: dict[str, Any] | None = None, path: str = "$") -> None:
    root = root or schema
    schema = resolve_ref(schema, root)
    typ = schema.get("type")
    if typ == "object" and not isinstance(instance, dict):
        raise ValidationError(f"{path}: want object")
    if typ == "array" and not isinstance(instance, list):
        raise ValidationError(f"{path}: want array")
    if typ == "string" and not isinstance(instance, str):
        raise ValidationError(f"{path}: want string")
    if typ == "integer" and not (isinstance(instance, int) and not isinstance(instance, bool)):
        raise ValidationError(f"{path}: want integer")
    if "const" in schema and instance != schema["const"]:
        raise ValidationError(f"{path}: want {schema['const']!r}, got {instance!r}")
    if "enum" in schema and instance not in schema["enum"]:
        raise ValidationError(f"{path}: {instance!r} not in enum")
    if isinstance(instance, str) and "pattern" in schema and not re.search(schema["pattern"], instance):
        raise ValidationError(f"{path}: {instance!r} does not match {schema['pattern']!r}")
    if isinstance(instance, str) and len(instance) < schema.get("minLength", 0):
        raise ValidationError(f"{path}: string too short")
    if isinstance(instance, int) and "minimum" in schema and instance < schema["minimum"]:
        raise ValidationError(f"{path}: below minimum")
    if isinstance(instance, dict):
        missing = [key for key in schema.get("required", []) if key not in instance]
        if missing:
            raise ValidationError(f"{path}: missing {missing!r}")
        for key, value in instance.items():
            props = schema.get("properties", {})
            if key in props:
                validate(value, props[key], root, f"{path}.{key}")
    if isinstance(instance, list):
        if len(instance) < schema.get("minItems", 0):
            raise ValidationError(f"{path}: too few items")
        item_schema = schema.get("items")
        if item_schema:
            for i, item in enumerate(instance):
                validate(item, item_schema, root, f"{path}[{i}]")


def validate_envelope(envelope: dict[str, Any], schema_dir: Path = SCHEMA_DIR) -> str:
    schema_name = envelope.get("schema")
    if not isinstance(schema_name, str):
        raise ValidationError("envelope missing string schema")
    validate(envelope, load_schema(schema_name, schema_dir))
    return schema_name


def json_files_under(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    if not path.is_dir():
        raise ValidationError(f"{path}: not a file or directory")
    return sorted(p for p in path.rglob("*.json") if p.is_file())


def validate_files(paths: list[Path], schema_dir: Path = SCHEMA_DIR) -> dict[str, int]:
    counts: dict[str, int] = {}
    for path in paths:
        value = load_json(path)
        if not isinstance(value, dict):
            raise ValidationError(f"{path}: want object")
        schema_name = validate_envelope(value, schema_dir)
        counts[schema_name] = counts.get(schema_name, 0) + 1
    return counts


def doc_examples(doc: Path = DEFAULT_DOC) -> list[dict[str, Any]]:
    text = doc.read_text(encoding="utf-8")
    out: list[dict[str, Any]] = []
    for i, block in enumerate(re.findall(r"```json\n(.*?)\n```", text, flags=re.S), 1):
        value = json.loads(block)
        if not isinstance(value, dict):
            raise ValidationError(f"JSON example {i} is not an object")
        out.append(value)
    return out


def validate_doc(doc: Path = DEFAULT_DOC, schema_dir: Path = SCHEMA_DIR) -> dict[str, int]:
    counts: dict[str, int] = {}
    for example in doc_examples(doc):
        schema_name = validate_envelope(example, schema_dir)
        counts[schema_name] = counts.get(schema_name, 0) + 1
    return counts


def load_validated(path: Path, schema_dir: Path = SCHEMA_DIR) -> list[dict[str, Any]]:
    values: list[dict[str, Any]] = []
    for file in json_files_under(path):
        value = load_json(file)
        if not isinstance(value, dict):
            raise ValidationError(f"{file}: want object")
        validate_envelope(value, schema_dir)
        values.append(value)
    if not values:
        raise ValidationError(f"{path}: no JSON files")
    return values


def validate_sequence(path: Path, schema_dir: Path = SCHEMA_DIR) -> dict[str, int]:
    values = load_validated(path, schema_dir)
    counts: dict[str, int] = {}
    by_schema: dict[str, list[dict[str, Any]]] = {}
    for value in values:
        schema = value["schema"]
        counts[schema] = counts.get(schema, 0) + 1
        by_schema.setdefault(schema, []).append(value)
    tasks = by_schema.get("fak.shared-task.v1", [])
    journals = by_schema.get("fak.shared-task-journal.v1", [])
    if not tasks:
        raise ValidationError("sequence: missing task record")
    if not journals:
        raise ValidationError("sequence: missing materialized journal")
    task_id = tasks[0]["task_id"]
    for journal in journals:
        if journal["task_id"] != task_id or journal["initial"]["task_id"] != task_id:
            raise ValidationError("sequence: journal task mismatch")
        if not journal["entries"]:
            raise ValidationError("sequence: journal has no accepted-event snapshots")
    accepted = [r for r in by_schema.get("fak.shared-patch-result.v1", []) if r["verdict"] == "accepted"]
    if not accepted:
        raise ValidationError("sequence: no accepted patch result")
    if not any(any(op["op"] == "replace" and op["path"] == "/title" for op in p["ops"]) for p in by_schema.get("fak.shared-patch.v1", [])):
        raise ValidationError("sequence: missing title replacement patch")
    return counts


def validate_verdicts(path: Path, schema_dir: Path = SCHEMA_DIR) -> dict[str, int]:
    values = load_validated(path, schema_dir)
    counts = validate_files(json_files_under(path), schema_dir)
    results = [v for v in values if v.get("schema") == "fak.shared-patch-result.v1"]
    verdicts = {r["verdict"] for r in results}
    for want in {"needs_approval", "denied", "quarantined"}:
        if want not in verdicts:
            raise ValidationError(f"verdicts: missing {want}")
    for result in results:
        if not result.get("reason"):
            raise ValidationError(f"verdicts: {result['verdict']} missing reason")
        if result["current_rev"] != result["base_rev"]:
            raise ValidationError(f"verdicts: {result['verdict']} advanced revision")
    return counts


def one(by_schema: dict[str, list[dict[str, Any]]], schema_name: str) -> dict[str, Any]:
    values = by_schema.get(schema_name, [])
    if len(values) != 1:
        raise ValidationError(f"want exactly one {schema_name}, got {len(values)}")
    return values[0]


def fmt_counts(counts: dict[str, int]) -> str:
    return ", ".join(f"{k}={v}" for k, v in sorted(counts.items()))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p_doc = sub.add_parser("validate-doc")
    p_doc.add_argument("path", type=Path, nargs="?", default=DEFAULT_DOC)
    p_file = sub.add_parser("validate")
    p_file.add_argument("path", type=Path)
    p_dir = sub.add_parser("validate-dir")
    p_dir.add_argument("path", type=Path)
    p_sequence = sub.add_parser("validate-sequence")
    p_sequence.add_argument("path", type=Path)
    p_verdicts = sub.add_parser("validate-verdicts")
    p_verdicts.add_argument("path", type=Path)
    args = ap.parse_args(argv)
    try:
        if args.cmd == "validate-doc":
            counts = validate_doc(args.path)
            print(f"{args.path}: OK ({fmt_counts(counts)})")
        elif args.cmd == "validate":
            counts = validate_files([args.path])
            print(f"{args.path}: OK ({fmt_counts(counts)})")
        elif args.cmd == "validate-dir":
            counts = validate_files(json_files_under(args.path))
            print(f"{args.path}: OK ({fmt_counts(counts)})")
        elif args.cmd == "validate-sequence":
            counts = validate_sequence(args.path)
            print(f"{args.path}: OK shared sequence ({fmt_counts(counts)})")
        elif args.cmd == "validate-verdicts":
            counts = validate_verdicts(args.path)
            print(f"{args.path}: OK collaboration verdicts ({fmt_counts(counts)})")
    except (OSError, ValidationError, json.JSONDecodeError) as exc:
        print(f"shared-task-contract: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
