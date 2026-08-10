"""Tests for the lightgap scorecard.

Three kinds of test live here, and the split is deliberate:

1. **Arithmetic** -- the rapidity model is the whole claim, so its identities are
   pinned directly: zero at the alternative, divergence at the ceiling, additivity
   of the tax, and the direction-agnostic beta that makes lower-is-better metrics
   work with no special case. If any of these break, every number on the board is
   wrong in a way no data fix can repair.
2. **Data integrity** -- every cell must trace to a real ceiling, a real
   alternative, a declared provenance, and a cost basis. A cell with a confident
   number and no source is worse than a missing cell, because the missing one
   shows up as UNCOVERED debt and the sourceless one does not.
3. **Honesty invariants** -- the caps that stop an authored estimate from
   reporting as a category-defining win, and the rule that an unmeasured material
   facet must surface as debt rather than silently vanishing from the weighted
   picture.

The tests do NOT pin the debt integer or any particular w_net. Those are supposed
to move as measurements land; pinning them would make the card's own improvement
red the suite.
"""

from __future__ import annotations

import json
import math
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

import lightgap_scorecard as lg  # noqa: E402

HERE = Path(__file__).resolve().parent
SCRIPT = HERE / "lightgap_scorecard.py"


@pytest.fixture(scope="module")
def sc() -> lg.Scorecard:
    return lg.Scorecard()


# -- 1. the arithmetic ----------------------------------------------------


def test_rapidity_is_zero_at_the_alternative() -> None:
    """beta = 0 means "exactly the next-best option", and that must score 0.

    This is the anchor that makes a positive score mean something: it has to be
    EARNED against what the buyer already has, not against nothing.
    """
    assert lg.rapidity(0.0) == 0.0


def test_rapidity_diverges_toward_the_ceiling() -> None:
    """Closing the last of the gap to physics costs unboundedly more than the first.

    A linear score would say 0.9 -> 0.99 is a tenth of the trip. artanh says it is
    roughly as far as everything before it, which is the point: a system near the
    limit has a moat, and the score should show it.
    """
    steps = [lg.rapidity(b) for b in (0.5, 0.9, 0.99, 0.999)]
    assert steps == sorted(steps)
    first_half = steps[0]
    last_tenth = steps[3] - steps[2]
    assert last_tenth > first_half, "the final 0.9% must cost more than the first 50%"


def test_rapidity_is_signed_and_antisymmetric() -> None:
    """Being as far below the alternative as above it scores the exact negative."""
    for b in (0.1, 0.5, 0.9):
        assert lg.rapidity(-b) == pytest.approx(-lg.rapidity(b))


def test_rapidity_is_clamped_to_a_finite_horizon() -> None:
    """JSON cannot hold an infinity; the horizon keeps the payload serializable."""
    assert lg.rapidity(1.0) == pytest.approx(lg.HORIZON)
    assert lg.rapidity(-1.0) == pytest.approx(-lg.HORIZON)
    assert lg.rapidity(50.0) == pytest.approx(lg.HORIZON)
    assert math.isfinite(lg.rapidity(1.0))


def test_adopt_iff_beta_exceeds_load(sc: lg.Scorecard) -> None:
    """The one-line decision rule, checked against every scored cell.

    w_net = artanh(beta) - artanh(load), and artanh is strictly increasing, so
    w_net > 0 exactly when beta > load. This is what lets the tax subtract on the
    same scale as the win instead of being a separate hand-waved discount.
    """
    for c in sc.cells:
        assert (c.w_net > 0) == (c.beta > c.load), (
            f"{c.segment}/{c.facet}: beta={c.beta:.4f} load={c.load:.4f} "
            f"w_net={c.w_net:.4f} -- the sign of the net score disagrees with the rule"
        )


def test_tax_is_additive_on_the_rapidity_scale(sc: lg.Scorecard) -> None:
    for c in sc.cells:
        assert c.w_net == pytest.approx(c.w - c.tau, abs=1e-9)


def test_beta_is_direction_agnostic() -> None:
    """Lower-is-better axes need no special case.

    For ASR or latency the ceiling sits BELOW the alternative, so (F - N) and
    (c - N) both flip sign and the ratio is already correct. Verified against a
    higher-is-better twin with the same relative position.
    """
    # higher-is-better: alternative 0.4, ceiling 1.0, fak 0.7 -> half the gap
    up = (0.7 - 0.4) / (1.0 - 0.4)
    # lower-is-better: alternative 0.6, ceiling 0.0, fak 0.3 -> also half the gap
    down = (0.3 - 0.6) / (0.0 - 0.6)
    assert up == pytest.approx(0.5)
    assert down == pytest.approx(0.5)
    assert up == pytest.approx(down)


def test_pure_tax_can_never_score_positive(sc: lg.Scorecard) -> None:
    """When the incumbent IS the ceiling, mediation can only take a cut.

    A pure_tax cell that came back positive would mean fak beat the thing it is
    running on top of, which is not a result -- it is a bug in the anchor.
    """
    tax_cells = [c for c in sc.cells if c.mode == "pure_tax"]
    assert tax_cells, "the board should contain pure_tax cells (gateway/engine shapes)"
    for c in tax_cells:
        assert c.beta <= 0, f"{c.segment}/{c.facet} claims to beat its own ceiling"


def test_parity_at_ceiling_scores_only_the_cost_difference(sc: lg.Scorecard) -> None:
    """Both options at the definitional floor -> the axis offers no reason to switch.

    Everything positive in such a cell comes from differential adoption cost, and
    that distinction matters: it is a procurement claim, not a security claim.
    """
    par = [c for c in sc.cells if c.mode == "parity_at_ceiling"]
    assert par, "injection-control cells should use the parity anchor"
    for c in par:
        assert c.beta == 0.0
        assert c.w == 0.0
        assert c.w_net == pytest.approx(-c.tau, abs=1e-9)


def test_cap_value_lands_just_under_the_next_rung() -> None:
    """A capped cell reports at the top of its allowed band, not at the band floor.

    Capping to the floor would throw away the difference between a barely-CRUISE
    and a would-have-been-NEAR-C cell; capping to just under the next rung keeps
    the ordering while refusing the unearned label.
    """
    assert lg.cap_value("CRUISE") == pytest.approx(lg.RUNG["RELATIVISTIC"] - lg.EPS)
    assert lg.cap_value("RELATIVISTIC") == pytest.approx(lg.RUNG["NEAR-C"] - lg.EPS)
    assert lg.cap_value(None) == float("inf")
    assert lg.verdict_for(lg.cap_value("CRUISE"))[0] == "CRUISE"
    assert lg.verdict_for(lg.cap_value("RELATIVISTIC"))[0] == "RELATIVISTIC"


def test_verdict_ladder_is_monotone_and_total() -> None:
    floors = [floor for _, floor, _ in lg.LADDER]
    assert floors == sorted(floors, reverse=True), "ladder must descend"
    assert floors[-1] == float("-inf"), "the ladder must be total (no unscorable w)"
    for name, floor, _ in lg.LADDER:
        if math.isfinite(floor):
            assert lg.verdict_for(floor)[0] == name
            assert lg.verdict_for(floor - lg.EPS)[0] != name


# -- 2. data integrity ----------------------------------------------------


def test_every_cell_traces_to_a_source(sc: lg.Scorecard) -> None:
    """A confident number with no provenance is the failure mode this card exists to prevent."""
    for c in sc.cells:
        assert c.fak_source.strip(), f"{c.segment}/{c.facet}: fak value has no source"
        assert c.alt_source.strip(), f"{c.segment}/{c.facet}: alternative value has no source"
        assert c.provenance in lg.PROVENANCE, (
            f"{c.segment}/{c.facet}: provenance {c.provenance!r} is outside the vocabulary"
        )


def test_every_alternative_is_declared(sc: lg.Scorecard) -> None:
    for c in sc.cells:
        assert c.alt_name, f"{c.segment}/{c.facet} names no alternative"
    for aid, alt in sc.alternatives.items():
        assert alt.get("class") in {"sota", "tuned", "floor", "naive"}, (
            f"alternative {aid} has no class -- the class is what says whether it is "
            "a real option or a strawman"
        )


def test_no_naive_alternative_is_scored_against(sc: lg.Scorecard) -> None:
    """The strawman may exist as a zero point; it may never BE the comparison.

    naive-reprefill is in the alternatives file because the headline 60.3x figure
    is measured against it, and the card's whole thesis is that scoring against a
    strawman is dishonest. So it is present, and no cell may use it.
    """
    assert any(a.get("class") == "naive" for a in sc.alternatives.values()), (
        "the strawman should still be catalogued, so the card can say what it is NOT scoring against"
    )
    for c in sc.cells:
        cls = sc.alternatives.get(c.alt_id, {}).get("class")
        assert cls != "naive", (
            f"{c.segment}/{c.facet} scores against a naive baseline -- "
            "that is the strawman comparison this card was built to replace"
        )


def test_every_cost_carries_a_basis(sc: lg.Scorecard) -> None:
    """The adoption tax is half the model, so its inputs get the same rigour as the win."""
    for path in sorted(Path(lg.DATA_DIR).glob("cells-*.json")):
        doc = json.loads(path.read_text(encoding="utf-8"))
        for row in doc.get("cells", []):
            where = f"{row['segment']}/{row['facet']}"
            cost = row.get("cost", {})
            assert "hours" in cost, f"{where}: no adoption cost"
            basis = cost.get("basis") or []
            assert basis and all(b.strip() for b in basis), (
                f"{where}: adoption cost has no stated basis"
            )


def test_every_ceiling_has_a_derivation_and_a_kind(sc: lg.Scorecard) -> None:
    """An undefended ceiling turns the denominator into a wish."""
    for cid, c in sc.ceilings.items():
        assert c.get("derivation", "").strip(), f"ceiling {cid} is asserted, not derived"
        assert c["kind"] in lg.CEILING_KINDS, f"ceiling {cid} has kind {c['kind']!r}"


def test_every_facet_has_a_ceiling_and_a_band(sc: lg.Scorecard) -> None:
    for fid, f in sc.facets.items():
        assert fid in sc.ceilings, f"facet {fid} has no ceiling"
        assert f["band"] in sc.bands, f"facet {fid} sits in unknown band {f['band']!r}"


def test_segment_weights_sum_to_one(sc: lg.Scorecard) -> None:
    """Weights are a buyer's attention budget; a segment that sums to 1.3 is claiming more."""
    for sid, seg in sc.segments.items():
        total = sum(seg.get("weights", {}).values())
        assert total == pytest.approx(1.0, abs=1e-6), f"{sid} weights sum to {total}"


def test_every_segment_declares_a_tolerance_and_a_bar(sc: lg.Scorecard) -> None:
    for sid, seg in sc.segments.items():
        assert seg.get("tolerance_hours", 0) > 0, f"{sid} has no adoption tolerance"
        assert "switch_bar" in seg, f"{sid} declares no bar for what is worth switching for"
        assert seg.get("next_best_summary", "").strip(), (
            f"{sid} does not say what this buyer would use instead"
        )


def test_no_structural_defects(sc: lg.Scorecard) -> None:
    """UNCOVERED is honest debt. Everything else is a broken card.

    An UNKNOWN_FACET or DEGENERATE_CEILING means the model is misconfigured, and a
    misconfigured model produces numbers that look fine and mean nothing.
    """
    structural = [d for d in sc.defects if d.code != "UNCOVERED"]
    assert not structural, "; ".join(f"{d.code} at {d.where}: {d.detail}" for d in structural)


# -- 3. honesty invariants ------------------------------------------------


def test_authored_estimates_cannot_report_as_category_defining(sc: lg.Scorecard) -> None:
    """The cap that keeps a MODELED corpus from reading like a measurement."""
    for c in sc.cells:
        if c.provenance in ("MODELED", "PROJECTED"):
            assert c.w_eff <= lg.cap_value("CRUISE") + 1e-9, (
                f"{c.segment}/{c.facet} is {c.provenance} but reports {c.w_eff:+.2f}"
            )
        if c.provenance == "OBSERVED":
            assert c.w_eff <= lg.cap_value("RELATIVISTIC") + 1e-9, (
                f"{c.segment}/{c.facet} is OBSERVED but reports {c.w_eff:+.2f}"
            )


def test_capped_cells_still_report_the_raw_score(sc: lg.Scorecard) -> None:
    """w_eff is what decisions use; w_net stays visible so the cap is auditable."""
    for c in sc.cells:
        assert c.w_eff <= c.w_net + 1e-9
        if c.cap:
            assert c.cap_reason.strip(), f"{c.segment}/{c.facet} is capped with no stated reason"


def test_unmeasured_material_facets_become_debt(sc: lg.Scorecard) -> None:
    """A material facet nobody measured must show up as debt, not as an absence.

    Silently dropping it would let a segment's weighted picture look complete
    while half of what the buyer cares about was never compared to anything.
    """
    uncovered = {(d.where.split("/")[0], d.where.split("/")[1])
                 for d in sc.defects if d.code == "UNCOVERED"}
    scored = {(c.segment, c.facet) for c in sc.cells}
    for sid, seg in sc.segments.items():
        for fid, weight in seg.get("weights", {}).items():
            if weight > 0:
                assert (sid, fid) in scored or (sid, fid) in uncovered, (
                    f"{sid}/{fid} is weighted {weight} but is neither scored nor booked as debt"
                )
    assert sc.debt == len(uncovered)


def test_every_uncovered_cell_names_the_experiment(sc: lg.Scorecard) -> None:
    """Debt without a next step is a complaint. Each gap has to name what would close it."""
    for d in sc.defects:
        if d.code == "UNCOVERED":
            assert d.detail.strip(), f"{d.where}: no reason given"
            assert d.next_action.strip(), f"{d.where}: no experiment named to close the gap"


def test_a_segment_can_be_undecidable(sc: lg.Scorecard) -> None:
    """The card must be able to refuse a verdict, not just grade one.

    If enough of a buyer's weighted attention was never measured, the honest
    answer is "we do not know", and that has to be a reachable outcome.
    """
    reachable = {"BLOCKED", "UNDECIDABLE", "ADOPT", "ADOPT-WITH-SCARS", "PILOT-ONLY",
                 "HOLD", "UNSCORED"}
    got = {sc.segment_profile(s)["verdict"] for s in sc.segments}
    assert got <= reachable, f"unexpected verdict(s): {got - reachable}"
    assert "UNDECIDABLE" in got or "BLOCKED" in got, (
        "no segment refuses or blocks -- a card that only ever says ADOPT is marketing"
    )


def test_the_board_is_not_uniformly_positive(sc: lg.Scorecard) -> None:
    """The 'spherical' requirement, mechanized.

    A scorecard whose own subject wins every cell is measuring the wrong things.
    This asserts the board has real negative territory, not that any particular
    cell is negative.
    """
    hull = sc.hull()
    assert hull["negative_cells"] > 0, "every cell is positive -- the axes are too kind"
    assert hull["eccentricity"] > 1.0, (
        "peak and dent are within one nat -- the card is not resolving the difference "
        "between what fak is good at and what it is bad at"
    )


def test_no_overall_score_is_emitted(sc: lg.Scorecard) -> None:
    """There is deliberately no mean.

    Averaging across segments would let a strong platform-team result paper over a
    BLOCKED local-first result, which is precisely the aggregation this card was
    built to refuse.
    """
    payload = sc.payload()
    banned = {"score", "overall", "total", "mean", "average", "grade"}
    assert not (banned & set(payload)), f"payload emits an aggregate: {banned & set(payload)}"


# -- the CLI contract -----------------------------------------------------


def _run(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(SCRIPT), *args],
        capture_output=True, text=True, cwd=str(HERE.parent), timeout=180,
    )


def test_json_payload_is_serializable_and_finite(sc: lg.Scorecard) -> None:
    text = json.dumps(sc.payload())          # raises on inf/nan
    reloaded = json.loads(text)
    assert reloaded["schema"] == "fak-lightgap-scorecard/1"
    assert isinstance(reloaded["lightgap_debt"], int)


def test_json_mode_exits_clean_and_carries_the_debt_key() -> None:
    """The control-pane fold finds the debt integer at the top level via find_int."""
    p = _run("--json")
    assert p.returncode == 0, p.stderr
    payload = json.loads(p.stdout)
    assert isinstance(payload["lightgap_debt"], int)


def test_check_mode_reds_on_debt() -> None:
    """The honesty gate has to fail while comparisons are missing, or it is decoration."""
    p = _run("--check")
    assert p.returncode == (1 if lg.Scorecard().debt else 0)
    assert "lightgap_debt" in p.stdout


def test_default_render_is_ascii() -> None:
    """The terminal surface must survive a cp1252 console.

    The markdown output and the module docstring keep Unicode on purpose; only
    what goes to a terminal is restricted.
    """
    p = _run()
    assert p.returncode == 0, p.stderr
    p.stdout.encode("ascii")  # raises UnicodeEncodeError on a stray em-dash


def test_determinism() -> None:
    """Same tree, same board -- the card is a pure read of committed data."""
    a = _run("--json")
    b = _run("--json")
    assert a.stdout == b.stdout
