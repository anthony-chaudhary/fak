"""Hermetic tests for self_improve.py — the dos-self-improve keep/revert logic (#388).

These cover the pure decision surface (no WSL / no `go` / no `dos` needed): the
AND-of-three keep-bit, the witness-honesty floor (#125), the candidate generators,
and the path translation. The live Go/`dos` witnesses are exercised by the seed run
itself (tools/self_improve.runs/), not here.
"""
import importlib.util
import os

_HERE = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location("self_improve", os.path.join(_HERE, "self_improve.py"))
si = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(si)


_SHIPPED = {"shipped": True, "source": "grep-subject", "rung": "trailer", "sha": "ad8da14"}


def test_keep_requires_all_three_and_kernel():
    keep, reasons = si.decide(True, True, _SHIPPED, "KEEP")
    assert keep is True, reasons
    assert any("suite green" in r for r in reasons)


def test_red_suite_reverts_even_with_kernel_keep():
    # A regression must REVERT regardless of any claimed gain — the non-negotiable floor.
    keep, _ = si.decide(False, True, _SHIPPED, "KEEP")
    assert keep is False


def test_red_architest_reverts():
    keep, _ = si.decide(True, False, _SHIPPED, "KEEP")
    assert keep is False


def test_kernel_revert_blocks_keep():
    # Even with all witnesses green, the kernel's REVERT verdict is fail-safe authoritative.
    keep, _ = si.decide(True, True, _SHIPPED, "REVERT")
    assert keep is False


def test_witness_honesty_source_none_is_not_confirmation():
    # #125: a source=none verdict is NOT a confirmation, so it can never reach KEEP.
    assert si.verify_is_clean({"shipped": True, "source": "none"}) is False
    keep, reasons = si.decide(True, True, {"shipped": True, "source": "none"}, "KEEP")
    assert keep is False
    assert any("NOT a confirmation" in r for r in reasons)


def test_witness_honesty_subject_only_rung_rejected():
    assert si.verify_is_clean({"shipped": True, "source": "grep-subject", "rung": "subject-only"}) is False


def test_witness_honesty_unshipped_rejected():
    assert si.verify_is_clean({"shipped": False, "source": "none"}) is False


def test_verify_clean_accepts_grep_subject_trailer():
    assert si.verify_is_clean(_SHIPPED) is True


def test_good_candidate_is_a_passing_additive_test():
    c = si.good_candidate("benchids")
    assert c["kind"] == "good"
    assert c["relpath"].endswith("_test.go")
    assert "package benchids" in c["content"]
    assert "t.Fatal" not in c["content"] or "out of [0," in c["content"]  # only the guarded-bound fatal
    assert "(fak benchids)" in c["subject"]


def test_bad_candidate_fails_on_purpose():
    c = si.bad_candidate("benchids")
    assert c["kind"] == "bad"
    assert "t.Fatal(" in c["content"]  # a deliberate regression the witness must catch
    assert "(fak benchids)" in c["subject"]


def test_to_wsl_path_translates_drive():
    assert si.to_wsl_path("C:\\work\\fak") == "/mnt/c/work/fak"
