#!/usr/bin/env python3
"""Tests for the product scorecard — durable / real / useful-today.

Three things are exercised. (1) The pure helpers: grade, section normalization +
tolerant match, first-command parsing, and the verdict the evidence IMPLIES through
the surface gate (a benchmark can't be a durable-product, a subsystem is real-not-easy).
(2) Each KPI's defect trigger — the overclaim catch vs CLAIMS.md (tag membership in
the section), the unrunnable-command catch, the unwitnessed/undiscoverable catch, and
the verdict-mismatch catch — plus coverage of the concept catalog. (3) The disk shell
(merge a data DIRECTORY, parse the CLAIMS catalog) and the fold to the composite.

Closes with the load-bearing live smoke: the REAL committed data directory must fold
to ZERO product-debt AND 100% coverage — the proof the shipped product map is complete,
honest, and cross-checked against the tree.

Run: `python tools/product_scorecard_test.py`  (exit 0 = all pass),
or `python -m pytest tools/product_scorecard_test.py -q`.
"""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import product_scorecard as ps  # noqa: E402


def row(**over) -> dict:
    """A minimal well-formed durable-product row (shipped product surface, offline)."""
    r = {
        "id": "r1", "concept": "C", "category": "security", "surface": "product",
        "what_you_get": "a thing", "audience": "developer", "maturity": "shipped",
        "claims_section": "Adjudication", "claims_tag": "SHIPPED",
        "first_command": "go run ./cmd/fak preflight --tool t --args \"{}\"",
        "first_command_verb": "preflight", "needs_gpu": False, "needs_key": False,
        "witness_path": "internal/adjudicator", "witness": "TestX", "entry_doc": "docs/cli-reference.md",
        "verdict": "durable-product", "gaps": [], "durability_note": "n",
    }
    r.update(over)
    return r


def tree(**over) -> dict:
    """Synthetic tree-facts: a section→tags map, the concept catalog, cmd dirs,
    documented verbs, and an existence set."""
    present = {"internal/adjudicator", "internal/foo", "docs/cli-reference.md", "GETTING-STARTED.md"}
    t = {
        "section_tags": {"adjudication": {"SHIPPED"},
                         "pre-flight ladder + grammar rung": {"SHIPPED", "STUB"}},
        "catalog": [{"section": "Adjudication", "norm": "adjudication"},
                    {"section": "Tool vDSO", "norm": "tool vdso"}],
        "cmd_dirs": {"fak", "fanbench"},
        "doc_verbs": {"preflight", "serve", "agent", "bench"},
        "exists": lambda p: p in present,
    }
    t.update(over)
    return t


def categories() -> set[str]:
    return {"security", "performance", "memory", "model", "tooling", "platform"}


# --- pure helpers ----------------------------------------------------------

def test_grade_letter() -> None:
    assert ps.grade_letter(100) == "A" and ps.grade_letter(85) == "B"
    assert ps.grade_letter(59) == "F"


def test_norm_section_cuts_at_separators() -> None:
    assert ps.norm_section("## Gateway (`fak serve`)") == "gateway"
    assert ps.norm_section("Answer-shape: the witness") == "answer-shape"
    assert ps.norm_section("Fan-out benchmark (fanbench — N)") == "fan-out benchmark"
    # a plain hyphen is NOT a separator (don't truncate 'write-time').
    assert ps.norm_section("S7 write-time durability gate (x)") == "s7 write-time durability gate"


def test_section_match_equal_and_containment() -> None:
    assert ps.section_match("Adjudication", "adjudication") is True
    assert ps.section_match("In-kernel model", "in-kernel model (the model)".split("(")[0].strip().lower()) is True
    assert ps.section_match("Gateway", "tool vdso") is False
    # trivial short overlaps are guarded.
    assert ps.section_match("a", "abc") is False


def test_parse_command() -> None:
    assert ps.parse_command("go run ./cmd/fak preflight --tool t") == ("fak", "preflight")
    assert ps.parse_command("go run ./cmd/fanbench -profile research") == ("fanbench", None)
    assert ps.parse_command("go test ./internal/model") == (None, None)
    assert ps.parse_command("") == (None, None)


def test_expected_verdict_surface_gate() -> None:
    assert ps.expected_verdict(row())[0] == "durable-product"
    assert ps.expected_verdict(row(needs_key=True))[0] == "usable-today"
    assert ps.expected_verdict(row(surface="benchmark"))[0] == "usable-today"
    assert ps.expected_verdict(row(surface="subsystem"))[0] == "real-not-easy"
    assert ps.expected_verdict(row(surface="seam"))[0] == "real-not-easy"
    # a product surface with no command can't be a durable-product.
    assert ps.expected_verdict(row(first_command=""))[0] == "real-not-easy"
    assert ps.expected_verdict(row(maturity="stub"))[0] == "honest-stub"
    assert ps.expected_verdict(row(maturity="simulated"))[0] == "honest-stub"
    assert ps.expected_verdict(row(maturity="concept"))[0] == "concept-only"


# --- per-KPI defect triggers -----------------------------------------------

def test_well_formed_catches_missing_surface_and_dups() -> None:
    k = ps.kpi_well_formed([row(), row()], categories())  # duplicate id r1
    assert any("duplicate id" in d for d in k["defects"])
    k2 = ps.kpi_well_formed([row(surface="nonsense")], categories())
    assert any("surface" in d for d in k2["defects"])
    k3 = ps.kpi_well_formed([{"id": "x"}], categories())  # missing fields
    assert len(k3["defects"]) > 5


def test_claim_honest_catches_overclaim() -> None:
    st = {"adjudication": {"SHIPPED"}}
    # maturity stub but tag SHIPPED -> disagreement.
    k = ps.kpi_claim_honest([row(maturity="stub", verdict="honest-stub", claims_tag="SHIPPED")], st)
    assert any("disagrees" in d for d in k["defects"])
    # claims_tag STUB but the section only carries SHIPPED -> overclaim vs ledger.
    k2 = ps.kpi_claim_honest([row(maturity="stub", verdict="honest-stub", claims_tag="STUB")], st)
    assert any("carries only" in d for d in k2["defects"])
    # honest case: SHIPPED in a SHIPPED+STUB section is fine (membership, not lead).
    k3 = ps.kpi_claim_honest([row(claims_section="Pre-flight ladder + grammar rung")],
                             {"pre-flight ladder + grammar rung": {"SHIPPED", "STUB"}})
    assert k3["defects"] == []


def test_command_resolves_catches_unrunnable() -> None:
    t = tree()
    # a go-test command has no ./cmd/<dir>.
    k = ps.kpi_command_resolves([row(first_command="go test ./internal/x", first_command_verb="test")],
                                t["cmd_dirs"], t["doc_verbs"])
    assert any("no `./cmd" in d for d in k["defects"])
    # a cmd dir that doesn't exist.
    k2 = ps.kpi_command_resolves([row(first_command="go run ./cmd/ghost x")],
                                 t["cmd_dirs"], t["doc_verbs"])
    assert any("does not exist" in d for d in k2["defects"])
    # an undocumented fak verb.
    k3 = ps.kpi_command_resolves([row(first_command="go run ./cmd/fak frobnicate")],
                                 t["cmd_dirs"], t["doc_verbs"])
    assert any("not documented" in d for d in k3["defects"])
    # verb set with empty command.
    k4 = ps.kpi_command_resolves([row(first_command="", first_command_verb="serve")],
                                 t["cmd_dirs"], t["doc_verbs"])
    assert any("empty" in d for d in k4["defects"])
    # the happy case resolves clean.
    assert ps.kpi_command_resolves([row()], t["cmd_dirs"], t["doc_verbs"])["defects"] == []


def test_witnessed_and_discoverable() -> None:
    exists = lambda p: p in {"internal/adjudicator", "docs/cli-reference.md"}
    assert ps.kpi_witnessed([row()], exists)["defects"] == []
    assert ps.kpi_witnessed([row(witness_path="internal/ghost")], exists)["defects"]
    assert ps.kpi_witnessed([row(witness_path="")], exists)["defects"]
    # a concept maturity is exempt from witnessed/discoverable.
    assert ps.kpi_witnessed([row(maturity="concept", verdict="concept-only")], exists)["defects"] == []
    assert ps.kpi_discoverable([row(entry_doc="docs/ghost.md")], exists)["defects"]
    assert ps.kpi_discoverable([row(maturity="concept", verdict="concept-only", entry_doc="")], exists)["defects"] == []


def test_verdict_consistency_catches_overclaim() -> None:
    # declares durable-product on a subsystem -> mismatch.
    k = ps.kpi_verdict_consistency([row(surface="subsystem", verdict="durable-product")])
    assert k["defects"] and "implies 'real-not-easy'" in k["defects"][0]
    assert ps.kpi_verdict_consistency([row()])["defects"] == []


# --- coverage --------------------------------------------------------------

def test_coverage_report() -> None:
    cat = [{"section": "Adjudication", "norm": "adjudication"},
           {"section": "Tool vDSO", "norm": "tool vdso"}]
    cov = ps.coverage_report(cat, [row(claims_section="Adjudication")])
    assert cov["covered"] == 1 and cov["coverage_debt"] == 1
    assert cov["uncovered"][0]["norm"] == "tool vdso"


# --- CLAIMS catalog parsing -------------------------------------------------

def test_parse_claims_catalog_excludes_denylist_and_collects_tags() -> None:
    text = (
        "## The product\n- [SHIPPED] a\n- [STUB] b\n"
        "## What fak is NOT\n- [STUB] c\n"
        "## Prior-art posture\n- [SHIPPED] d\n"
    )
    catalog, tags = ps.parse_claims_catalog(text)
    norms = {c["norm"] for c in catalog}
    assert "the product" in norms
    assert "what fak is not" not in norms and "prior-art posture" not in norms
    assert tags["the product"] == {"SHIPPED", "STUB"}


# --- disk shell + fold ------------------------------------------------------

def test_load_data_dir_merges_modular_files() -> None:
    with tempfile.TemporaryDirectory() as td:
        d = Path(td)
        (d / "_meta.json").write_text(json.dumps({
            "meta": {"as_of": "2026-06-24", "fak_version": "t"},
            "categories": [{"id": "security", "name": "Sec"}],
        }), encoding="utf-8")
        (d / "rows-security.json").write_text(json.dumps({"rows": [row()]}), encoding="utf-8")
        data, err = ps.load_data_dir(d)
        assert err == "" and data is not None
        assert len(data["rows"]) == 1 and data["rows"][0]["_source_file"] == "rows-security.json"


def _data(rows: list[dict]) -> dict:
    return {"meta": {"as_of": "2026-06-24", "fak_version": "t"},
            "categories": [{"id": c, "name": c} for c in categories()],
            "rows": rows}


def test_build_payload_zero_debt_full_coverage_is_ok() -> None:
    t = tree(catalog=[{"section": "Adjudication", "norm": "adjudication"}])
    p = ps.build_payload(workspace=".", data=_data([row()]), tree=t)
    assert p["ok"] is True and p["verdict"] == "OK"
    assert p["corpus"]["product_debt"] == 0 and p["corpus"]["grade"] == "A"
    assert p["corpus"]["coverage"]["coverage_pct"] == 100.0


def test_build_payload_coverage_gap_drives_action() -> None:
    p = ps.build_payload(workspace=".", data=_data([row()]), tree=tree())  # 2-section catalog, 1 positioned
    assert p["ok"] is False and p["finding"] == "coverage_debt"
    assert p["corpus"]["coverage_debt"] == 1 and p["corpus"]["honesty_defects"] == 0


def test_build_payload_honesty_defect_drives_action() -> None:
    t = tree(catalog=[{"section": "Adjudication", "norm": "adjudication"}])
    bad = [row(surface="subsystem", verdict="durable-product")]  # verdict overclaim
    p = ps.build_payload(workspace=".", data=_data(bad), tree=t)
    assert p["ok"] is False and p["finding"] == "product_debt"
    assert p["corpus"]["honesty_defects"] >= 1


def test_build_payload_error_on_no_data() -> None:
    p = ps.build_payload(workspace=".", data=None, tree=tree(), error="missing data")
    assert p["ok"] is False and p["verdict"] == "AUDIT_ERROR"


# --- renderers don't crash + produce the doc folder -------------------------

def test_renderers_and_doc_folder() -> None:
    t = tree(catalog=[{"section": "Adjudication", "norm": "adjudication"}])
    p = ps.build_payload(workspace=".", data=_data([row()]), tree=t)
    assert "product-scorecard:" in ps.render(p)
    assert "backlog" in ps.render_critical(p)
    assert "backlog" in ps.render_gaps(p)
    files = ps.render_doc_folder(p, stamp="2026-06-24")
    assert "README.md" in files and "Product scorecard" in files["README.md"]
    # compare renders a ratio line.
    assert "product-debt:" in ps.render_compare(p, p)


# --- the load-bearing live smoke: the shipped map is complete + honest ------

def test_live_real_data_is_complete_and_honest() -> None:
    root = ps.repo_root()
    path = root / ps.DATA_DIR_REL
    if not path.exists():
        return  # tolerant: not in the repo tree
    p = ps.collect(root)
    assert p["schema"] == ps.SCHEMA, p
    c = p["corpus"]
    # The committed product map must carry ZERO product-debt (honest + cross-checked)...
    assert c["product_debt"] == 0, p["reason"]
    # ...and have positioned every CLAIMS.md concept section (100% coverage).
    assert c["coverage"]["coverage_debt"] == 0, p["reason"]
    assert p["ok"] is True and c["grade"] == "A"
    # the honest spread must be shown: real durable products AND honest real-not-easy gaps.
    assert c["standing"]["durable-product"] >= 5, "real product surfaces a person uses today"
    assert c["standing"]["real-not-easy"] >= 1, "the subsystems that are real but not a product surface"
    # every durable-product verdict must be backed by a `product` surface (the surface gate held).
    for r in c["leaderboard"]:
        if r["verdict"] == "durable-product":
            assert r["surface"] == "product", f"{r['id']} is durable-product but surface={r['surface']}"
            assert r["offline"], f"{r['id']} is durable-product but its first command is not offline"


def main() -> int:
    failures: list[str] = []

    def check(name: str, fn) -> None:
        try:
            fn()
        except AssertionError as exc:
            failures.append(f"{name}: {exc}")
        except Exception as exc:  # noqa: BLE001
            failures.append(f"{name}: unexpected {type(exc).__name__}: {exc}")

    tests = {n: f for n, f in globals().items()
             if n.startswith("test_") and callable(f)}
    for name, fn in tests.items():
        check(name, fn)

    if failures:
        print(f"FAIL ({len(failures)}/{len(tests)}):")
        for f in failures:
            print("  -", f)
        return 1
    print(f"ok ({len(tests)} tests)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
