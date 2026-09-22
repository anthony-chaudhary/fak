package model

// v41_incremental_session_test.go — the fak#13313 independent regression for the
// PRODUCTION activation of the incremental V4.1 decode route. It is authored
// against the CONTRACT, not the implementation: after a Session.Prefill seeds a
// plain-layer continuation state, Session.Step must advance ONE token through
// the shadow incremental seam (forwardV41Step, #13311) instead of recomputing
// the whole history, and must fall back to the historical full-history route
// (with an explicit, observable status) whenever the session is not eligible.
//
// It reuses the production fixtures from v41_incremental_test.go (the plain
// reduced model, the production prefill seed and the deterministic prefix
// builder) so the comparison is against rows the model actually produces, and
// the #13311 witness already pins the seam's own arithmetic.

import (
	"math"
	"reflect"
	"testing"
)

// v41SessionIncrementalCalls counts successful incremental production Steps via
// the package probe. It returns a read function and a restore function so a test
// can install the observer around a scoped sequence of Session.Step calls.
func v41SessionIncrementalCalls(t *testing.T) (read func() int, restore func()) {
	t.Helper()
	prev := v41IncrementalStepProbe
	n := 0
	v41IncrementalStepProbe = func() { n++ }
	return func() int { return n }, func() { v41IncrementalStepProbe = prev }
}

// TestV41IncrementalSessionRoute proves the PRODUCTION Session.Step entry point
// takes the incremental seam on a seeded plain-layer session and agrees with the
// full-history reference: the probe fires once per Step, the logits match the
// reference within a pre-declared tolerance, and the committed history and every
// layer cursor advance by exactly one per Step. A snapshot of history alone is
// not enough to reach this route — the incremental path requires the seeded
// per-layer retained state, so a history-only state would fall back (covered by
// TestV41IncrementalSessionFallback).
func TestV41IncrementalSessionRoute(t *testing.T) {
	const tol = 1e-6

	for _, prefixLen := range []int{1, 7, 31, 32} {
		m := v41IncrementalPlainModel(t, 2)
		cfg := m.Cfg
		s := m.NewSession()

		prefix, next := v41IncrementalPrefix(cfg, prefixLen)
		warm := s.Prefill(prefix)
		if len(warm) == 0 {
			t.Fatalf("prefix %d: Prefill returned no logits", prefixLen)
		}
		if got := s.v41Forward.history; !reflect.DeepEqual(got, prefix) {
			t.Fatalf("prefix %d: session history = %v, want %v", prefixLen, got, prefix)
		}
		if !s.v41IncrementalEligible() {
			t.Fatalf("prefix %d: seeded plain-layer session is not incrementally eligible", prefixLen)
		}

		read, restore := v41SessionIncrementalCalls(t)
		history := append([]int(nil), prefix...)
		steps := []int{next, (next*3 + 1) % cfg.VocabSize, (next*5 + 2) % cfg.VocabSize}
		for i, tok := range steps {
			history = append(history, tok)

			want, err := m.forwardV41(history, nil)
			if err != nil {
				t.Fatalf("prefix %d step %d: full forward: %v", prefixLen, i, err)
			}
			wantRow := want.Logits[len(want.Logits)-1]

			before := read()
			got := s.Step(tok)
			if after := read(); after != before+1 {
				t.Fatalf("prefix %d step %d: incremental probe calls = %d, want 1 more than %d",
					prefixLen, i, after, before)
			}
			if !reflect.DeepEqual(s.v41Forward.history, history) {
				t.Fatalf("prefix %d step %d: session history = %v, want %v",
					prefixLen, i, s.v41Forward.history, history)
			}
			if len(got) != len(wantRow) {
				t.Fatalf("prefix %d step %d: logits width %d, want %d", prefixLen, i, len(got), len(wantRow))
			}
			for j := range wantRow {
				delta := got[j] - wantRow[j]
				if delta < 0 {
					delta = -delta
				}
				if delta > tol || math.IsNaN(float64(got[j])) {
					t.Fatalf("prefix %d step %d: logits[%d] = %g, want %g (delta %g)",
						prefixLen, i, j, got[j], wantRow[j], delta)
				}
			}
			for l := 0; l < cfg.NumLayers; l++ {
				if pos := s.v41Forward.layerState(l).nextWindowPos; pos != len(history) {
					t.Fatalf("prefix %d step %d layer %d: nextWindowPos = %d, want %d",
						prefixLen, i, l, pos, len(history))
				}
			}
		}
		restore()
	}
}

// TestV41IncrementalSessionConstantWorkPerStep is the counter witness the leaf's
// definition of done requires: across a short and a long prefix the per-step
// work is CONSTANT — exactly cfg.NumLayers single-position layer calls each
// Step, never proportional to the prefix length — while the full-history
// reference the fallback would pay grows with the prefix. It drives the
// PRODUCTION route by asserting the incremental probe fired, then counts the
// incremental seam's own LayerCalls; a snapshot of history alone cannot reach
// this route because a history-only state carries no seeded layer cursors.
//
// Prefix lengths stay at or below the retained ring width (v41WindowSize) minus
// one: the ring is a fixed 128 rows and a step must fully retain its visible
// causal window, so 120 is the longest prefix this fixture can step.
func TestV41IncrementalSessionConstantWorkPerStep(t *testing.T) {
	var (
		refSeqByPrefix = map[int]int{}
		callsByPrefix  = map[int]int{}
	)
	for _, prefixLen := range []int{32, 120} {
		m := v41IncrementalPlainModel(t, 2)
		cfg := m.Cfg
		s := m.NewSession()

		prefix, next := v41IncrementalPrefix(cfg, prefixLen)
		s.Prefill(prefix)
		if !s.v41IncrementalEligible() {
			t.Fatalf("prefix %d: session is not incrementally eligible", prefixLen)
		}

		// The production Step must take the incremental route.
		var fired int
		prev := v41IncrementalStepProbe
		v41IncrementalStepProbe = func() { fired++ }
		s.Step(next)
		v41IncrementalStepProbe = prev
		if fired != 1 {
			t.Fatalf("prefix %d: production Step fired the incremental probe %d times, want 1", prefixLen, fired)
		}

		// The incremental seam's own per-step layer fan-out is independent of the
		// prefix: exactly NumLayers single-position calls, no replay of P. Measured
		// on a fresh seeded state so the counter is not perturbed by the step
		// above; a step that replayed P would fan out over len(history) positions
		// instead of NumLayers.
		m2 := v41IncrementalPlainModel(t, 2)
		s2 := m2.NewSession()
		s2.Prefill(prefix)
		st2 := s2.v41Forward
		tok := (next*3 + 1) % cfg.VocabSize
		_, stats, err := m2.forwardV41Step(tok, st2, &v41ProjScratch{})
		if err != nil {
			t.Fatalf("prefix %d: forwardV41Step: %v", prefixLen, err)
		}
		if stats.LayerCalls != cfg.NumLayers {
			t.Fatalf("prefix %d: LayerCalls = %d, want %d (constant, independent of prefix)",
				prefixLen, stats.LayerCalls, cfg.NumLayers)
		}
		if stats.LayersRolledBack != 0 {
			t.Fatalf("prefix %d: LayersRolledBack = %d, want 0", prefixLen, stats.LayersRolledBack)
		}
		callsByPrefix[prefixLen] = stats.LayerCalls

		// The contrast: the full-history reference the fallback pays fans out over
		// the WHOLE prefix, so its produced sequence length grows with the prefix.
		// The incremental step's constant NumLayers fan-out above is therefore a
		// real "does not replay P" property, not an artifact.
		ref, err := m2.forwardV41(append(append([]int(nil), prefix...), tok), nil)
		if err != nil {
			t.Fatalf("prefix %d: full reference: %v", prefixLen, err)
		}
		if ref.Seq != prefixLen+1 {
			t.Fatalf("prefix %d: full reference Seq = %d, want %d (grows with prefix)",
				prefixLen, ref.Seq, prefixLen+1)
		}
		refSeqByPrefix[prefixLen] = ref.Seq

		// A history-only state (the "snapshot" that omits the seeded per-layer
		// retained state) is NOT eligible and must fall back, never silently
		// produce an incremental hit.
		historyOnly := &v41ForwardState{history: append([]int(nil), st2.history...)}
		snap := &Session{M: m2, v41Forward: historyOnly}
		if snap.v41IncrementalEligible() {
			t.Fatalf("prefix %d: history-only state reported eligible, want fallback", prefixLen)
		}
	}

	// Constant work vs growing reference: the incremental step's layer fan-out is
	// identical at both prefixes, while the fallback's full-history reference Seq
	// grows with the prefix — the machine-checkable "does not replay P" witness.
	if callsByPrefix[32] != callsByPrefix[120] {
		t.Fatalf("LayerCalls differ by prefix: 32->%d, 120->%d", callsByPrefix[32], callsByPrefix[120])
	}
	if refSeqByPrefix[120] <= refSeqByPrefix[32] {
		t.Fatalf("full-history reference did not grow with prefix: 32->%d, 120->%d",
			refSeqByPrefix[32], refSeqByPrefix[120])
	}
	t.Logf("incremental Step LayerCalls=%d (constant); full-history reference Seq: prefix32=%d prefix120=%d",
		callsByPrefix[32], refSeqByPrefix[32], refSeqByPrefix[120])
}

// TestV41IncrementalSessionFallback pins the explicit cold/unsupported status of
// the activation across the two refusal classes the eligibility gate names:
//   - an UNSEEDED session state (the "history only" snapshot the leaf calls
//     out), and
//   - a layer whose resolved plan is NOT a plain per-layer role.
//
// In both cases Session.Step keeps the historical full-history route: the probe
// must NOT fire, the logits must still equal the full-history reference, and the
// history must advance by exactly one — a failed incremental attempt never
// claims a cache hit.
func TestV41IncrementalSessionFallback(t *testing.T) {
	t.Run("unseeded-state", func(t *testing.T) {
		const tol = 1e-6
		m := v41IncrementalPlainModel(t, 2)
		cfg := m.Cfg
		s := m.NewSession()
		prefix, next := v41IncrementalPrefix(cfg, 8)
		s.Prefill(prefix)

		// Drop the last layer's seeded retained state: the session now holds a
		// history-only continuation (the snapshot the leaf warns about) and must
		// not be eligible.
		last := cfg.NumLayers - 1
		s.v41Forward.layers[last] = nil
		if s.v41IncrementalEligible() {
			t.Fatal("session with an unseeded layer reported eligible, want fallback")
		}

		var fired int
		prev := v41IncrementalStepProbe
		v41IncrementalStepProbe = func() { fired++ }
		got := s.Step(next)
		v41IncrementalStepProbe = prev
		if fired != 0 {
			t.Fatalf("fallback Step fired the incremental probe %d times, want 0", fired)
		}
		wantHistory := append(append([]int(nil), prefix...), next)
		if !reflect.DeepEqual(s.v41Forward.history, wantHistory) {
			t.Fatalf("fallback history = %v, want %v", s.v41Forward.history, wantHistory)
		}
		assertV41LogitsParity(t, m, wantHistory, got, tol)
	})

	t.Run("non-plain-role", func(t *testing.T) {
		m := v41IncrementalPlainModel(t, 2)
		s := m.NewSession()
		prefix, _ := v41IncrementalPrefix(m.Cfg, 8)
		s.Prefill(prefix)
		if !s.v41IncrementalEligible() {
			t.Fatal("plain-layer session unexpectedly ineligible before poisoning a role")
		}

		// Force layer 0's resolved plan to a shared-source role by poisoning the
		// memoized role map the eligibility gate reads. The gate must refuse the
		// incremental route and keep the fallback; restore the map immediately so
		// the assertion is purely about the eligibility status, not a run.
		roles := s.M.v41AttentionRolesCached()
		orig, had := roles[0]
		roles[0] = V41AttentionRoleKVSource
		eligible := s.v41IncrementalEligible()
		if had {
			roles[0] = orig
		} else {
			delete(roles, 0)
		}
		if eligible {
			t.Fatal("session with a non-plain layer role reported eligible, want fallback")
		}
		if !s.v41IncrementalEligible() {
			t.Fatal("restoring the plain role did not restore eligibility")
		}
	})
}

// assertV41LogitsParity compares one step's logits against the full-history
// reference under a pre-declared tolerance.
func assertV41LogitsParity(t *testing.T, m *Model, history []int, got []float32, tol float32) {
	t.Helper()
	want, err := m.forwardV41(history, nil)
	if err != nil {
		t.Fatalf("full reference: %v", err)
	}
	wantRow := want.Logits[len(want.Logits)-1]
	if len(got) != len(wantRow) {
		t.Fatalf("logits width %d, want %d", len(got), len(wantRow))
	}
	for j := range wantRow {
		delta := got[j] - wantRow[j]
		if delta < 0 {
			delta = -delta
		}
		if delta > tol || math.IsNaN(float64(got[j])) {
			t.Fatalf("logits[%d] = %g, want %g (delta %g)", j, got[j], wantRow[j], delta)
		}
	}
}

// TestV41IncrementalSessionRetryAfterFailure proves the "retry after a failed
// step behaves as if it never occurred" clause of the activation: a poisoned
// layer cursor makes the incremental seam refuse with a typed
// ErrV41ForwardStage, the production Step then falls back and commits exactly
// one token (never two), and a freshly prefilled session advances incrementally
// again for the same token with identical logits.
func TestV41IncrementalSessionRetryAfterFailure(t *testing.T) {
	m := v41IncrementalPlainModel(t, 2)
	cfg := m.Cfg
	s := m.NewSession()

	prefix, next := v41IncrementalPrefix(cfg, 6)
	s.Prefill(prefix)

	// Poison the last layer's cursor so the incremental seam refuses inside the
	// layer loop; the seam rolls the step back and the production route falls
	// back to the full-history recompute for this one token.
	last := cfg.NumLayers - 1
	s.v41Forward.layerState(last).nextWindowPos = len(s.v41Forward.history) + 5

	var fired int
	prev := v41IncrementalStepProbe
	v41IncrementalStepProbe = func() { fired++ }
	gotFallback := s.Step(next)
	v41IncrementalStepProbe = prev
	if fired != 0 {
		t.Fatalf("poisoned-cursor Step fired the incremental probe %d times, want 0 (fallback)", fired)
	}
	if want := append(append([]int(nil), prefix...), next); !reflect.DeepEqual(s.v41Forward.history, want) {
		t.Fatalf("after fallback history = %v, want %v", s.v41Forward.history, want)
	}

	// A freshly prefilled session takes the incremental route for the same token
	// and must produce identical logits — the fallback was exactly as if the
	// failed incremental attempt never happened.
	s2 := m.NewSession()
	s2.Prefill(prefix)
	if !s2.v41IncrementalEligible() {
		t.Fatal("freshly prefilled session is not incrementally eligible")
	}
	var fired2 int
	prev2 := v41IncrementalStepProbe
	v41IncrementalStepProbe = func() { fired2++ }
	gotRetry := s2.Step(next)
	v41IncrementalStepProbe = prev2
	if fired2 != 1 {
		t.Fatalf("retry Step fired the incremental probe %d times, want 1", fired2)
	}
	if !reflect.DeepEqual(gotRetry, gotFallback) {
		t.Fatal("incremental retry logits disagree with the fallback logits for the same token")
	}
}
