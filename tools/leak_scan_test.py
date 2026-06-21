#!/usr/bin/env python3
"""Tests for leak_scan.py — the static leak-candidate scanner heuristics.

Run: python -m pytest tools/leak_scan_test.py   (or: python tools/leak_scan_test.py)
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import leak_scan as L


def klasses(findings, klass):
    return [f for f in findings if f[1] == klass]


def lines(src):
    return src.splitlines()


# --- ticker / timer ------------------------------------------------------------------

def test_ticker_without_stop_is_high():
    src = lines("""
func run() {
    ticker := time.NewTicker(d)
    for { <-ticker.C }
}
""")
    out = L.scan_file("x.go", src)
    tt = klasses(out, "ticker_timer_no_stop")
    assert tt and tt[0][2] == "high"


def test_ticker_with_defer_stop_is_clean():
    src = lines("""
func run() {
    ticker := time.NewTicker(d)
    defer ticker.Stop()
    for { <-ticker.C }
}
""")
    out = L.scan_file("x.go", src)
    assert not klasses(out, "ticker_timer_no_stop")


# --- context cancel ------------------------------------------------------------------

def test_ctx_cancel_unreleased_is_high():
    src = lines("""
func run(ctx context.Context) {
    ctx, cancel := context.WithTimeout(ctx, time.Second)
    use(ctx)
}
""")
    out = L.scan_file("x.go", src)
    cc = klasses(out, "ctx_cancel_unreleased")
    assert cc and cc[0][2] == "high"


def test_ctx_cancel_deferred_is_clean():
    src = lines("""
func run(ctx context.Context) {
    ctx, cancel := context.WithTimeout(ctx, time.Second)
    defer cancel()
    use(ctx)
}
""")
    out = L.scan_file("x.go", src)
    assert not klasses(out, "ctx_cancel_unreleased")


# --- goroutine spawn -----------------------------------------------------------------

def test_goroutine_spawn_flagged():
    out = L.scan_file("x.go", lines("func f() {\n    go func() { work() }()\n}"))
    assert klasses(out, "goroutine_spawn")


# --- unbounded map (the ctxmmu/normgate shape) ---------------------------------------

def _pkg(src):
    fl = {"p/a.go": lines(src)}
    return L.package_unbounded_maps(["p/a.go"], fl)


def test_runtime_written_map_no_delete_is_high():
    # the ctxmmu/normgate shape: a `held` map written on Admit(), never deleted.
    src = """
type G struct { held map[string]int }
func (g *G) Admit(r *R) {
    g.held[id] = 1
}
"""
    out = _pkg(src)
    um = [f for f in out if f[3] == "high"]
    assert um and "held" in um[0][4]


def test_map_with_delete_is_not_flagged():
    src = """
type G struct { held map[string]int }
func (g *G) Admit(r *R) { g.held[id] = 1 }
func (g *G) drop(id string) { delete(g.held, id) }
"""
    out = _pkg(src)
    assert not [f for f in out if "held" in f[4]]


def test_construction_written_map_is_low():
    # a vocab loaded once at construction is fixed-size, not a leak → demoted to low.
    src = """
type T struct { tokenToID map[string]int }
func loadVocab() *T {
    t := &T{tokenToID: map[string]int{}}
    t.tokenToID[tok] = id
    return t
}
"""
    out = _pkg(src)
    flagged = [f for f in out if "tokenToID" in f[4]]
    assert flagged and flagged[0][3] == "low"


def test_cap_tell_downgrades_runtime_map():
    # a package that mentions evict/maxHeld is presumed bounded → runtime write is med, not high.
    src = """
const maxHeld = 8192
type G struct { held map[string]int }
func (g *G) Admit(r *R) { g.held[id] = 1; g.evict() }
func (g *G) evict() {}
"""
    out = _pkg(src)
    flagged = [f for f in out if "held" in f[4]]
    assert flagged and flagged[0][3] == "med"


# --- partition balancing -------------------------------------------------------------

def test_partition_balances_lanes():
    by_pkg = {"a": 100, "b": 1, "c": 1, "d": 1}
    bins = L.partition(by_pkg, 2)
    # the heavy package must be alone in its lane; the three light ones share the other.
    weights = sorted(b["weight"] for b in bins)
    assert weights == [3, 100]


def test_end_to_end_run_shape():
    rep = L.run(["tools"], include_tests=False, lanes=2, min_signal="low")
    # smoke: the scanner returns a well-formed report over a non-Go tree (0 findings).
    assert set(["ok", "findings", "audit_partition", "by_signal"]).issubset(rep.keys())


if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main([os.path.abspath(__file__), "-q"]))
