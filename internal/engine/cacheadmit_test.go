package engine

import "testing"

// These tests are the repo witness for cache-frontier item 49 / issue #1567: an
// unsupported or unknown ACTIVE cache path fails closed with a NAMED reason, and never
// falls through to an optimistic claim. They fail before AdmitActiveCache exists (the
// symbol is absent) and pass after.

// measuredProvenances are the planes whose evidence is a measured fact. A forecast is
// deliberately excluded — it is never a measured fact (see ProvenanceForecast).
func measuredProvenances() []CacheProvenance {
	return []CacheProvenance{ProvenanceKernel, ProvenanceProvider, ProvenanceContext}
}

// activeVerdicts returns the ACTIVE members of the closed verdict vocabulary, read from
// the vocabulary itself so a new active class cannot be added without these tests seeing
// it.
func activeVerdicts() []CacheVerdict {
	var out []CacheVerdict
	for _, v := range CacheVerdicts() {
		if v.Active() {
			out = append(out, v)
		}
	}
	return out
}

// TestAdmitActiveCacheUnknownCapabilityFailsClosed is the issue's headline target:
// "Unknown engine/provider capabilities do not silently fall back to optimistic claims."
// An unestablished capability is refused with the named reason for EVERY active op.
func TestAdmitActiveCacheUnknownCapabilityFailsClosed(t *testing.T) {
	for _, op := range activeVerdicts() {
		cc := CacheCapability{
			Engine:          "mystery-engine",
			Verdict:         CacheUnknown,
			Provenance:      ProvenanceKernel,
			ColdPathCorrect: true, // even a witnessed cold path cannot rescue an unknown capability
		}
		got := AdmitActiveCache(cc, op)
		if got != RefusalCapabilityUnknown {
			t.Errorf("unknown capability, op %q: refusal = %q, want %q", op, got, RefusalCapabilityUnknown)
		}
		if !got.Refused() {
			t.Errorf("unknown capability, op %q: Refused() = false, want true (must not be an optimistic admit)", op)
		}
	}
}

// TestAdmitActiveCacheUnsupportedOperation: an ESTABLISHED capability for a different
// active class does not license the requested one.
func TestAdmitActiveCacheUnsupportedOperation(t *testing.T) {
	for _, have := range activeVerdicts() {
		for _, op := range activeVerdicts() {
			if have == op {
				continue
			}
			cc := CacheCapability{
				Engine: "one-trick-engine", Verdict: have,
				Provenance: ProvenanceProvider, ColdPathCorrect: true,
			}
			got := AdmitActiveCache(cc, op)
			if got != RefusalOperationUnsupported {
				t.Errorf("capability %q asked for op %q: refusal = %q, want %q",
					have, op, got, RefusalOperationUnsupported)
			}
		}
	}
}

// TestAdmitActiveCacheColdPathUnwitnessed: cold-path correctness stays explicit for any
// active behavior — an unwitnessed cold path is refused by name.
func TestAdmitActiveCacheColdPathUnwitnessed(t *testing.T) {
	for _, op := range activeVerdicts() {
		cc := CacheCapability{
			Engine: "unwitnessed-engine", Verdict: op,
			Provenance: ProvenanceKernel, ColdPathCorrect: false,
		}
		if got := AdmitActiveCache(cc, op); got != RefusalColdPathUnwitnessed {
			t.Errorf("op %q without cold-path witness: refusal = %q, want %q",
				op, got, RefusalColdPathUnwitnessed)
		}
	}
}

// TestAdmitActiveCacheMalformedCapability: an out-of-vocabulary verdict or provenance is
// trusted for nothing, and is named as malformed rather than silently admitted.
func TestAdmitActiveCacheMalformedCapability(t *testing.T) {
	cases := []CacheCapability{
		{Verdict: "totally-bogus", Provenance: ProvenanceKernel, ColdPathCorrect: true},
		{Verdict: CacheActiveWarm, Provenance: "totally-bogus", ColdPathCorrect: true},
		{Verdict: "totally-bogus", Provenance: "totally-bogus", ColdPathCorrect: true},
	}
	for _, cc := range cases {
		if got := AdmitActiveCache(cc, CacheActiveWarm); got != RefusalCapabilityMalformed {
			t.Errorf("malformed capability %+v: refusal = %q, want %q",
				cc, got, RefusalCapabilityMalformed)
		}
	}
}

// TestAdmitActiveCacheForecastProvenanceRefused: a forecast is never a measured fact, so
// it cannot license an ACTIVE act. Provenance stays separate from the verdict — the
// capability is not rewritten, the ACT is refused.
func TestAdmitActiveCacheForecastProvenanceRefused(t *testing.T) {
	for _, op := range activeVerdicts() {
		cc := CacheCapability{
			Engine: "forecast-engine", Verdict: op,
			Provenance: ProvenanceForecast, ColdPathCorrect: true,
		}
		if got := AdmitActiveCache(cc, op); got != RefusalForecastProvenance {
			t.Errorf("op %q backed only by a forecast: refusal = %q, want %q",
				op, got, RefusalForecastProvenance)
		}
	}
}

// TestAdmitActiveCacheNonActiveOperation: the guard admits only ACTIVE operations; a
// passive/paged/unknown "operation" is named as caller misuse.
func TestAdmitActiveCacheNonActiveOperation(t *testing.T) {
	for _, op := range CacheVerdicts() {
		if op.Active() {
			continue
		}
		cc := CacheCapability{
			Engine: "any-engine", Verdict: CacheActiveWarm,
			Provenance: ProvenanceKernel, ColdPathCorrect: true,
		}
		if got := AdmitActiveCache(cc, op); got != RefusalOperationNotActive {
			t.Errorf("non-active op %q: refusal = %q, want %q", op, got, RefusalOperationNotActive)
		}
	}
}

// TestAdmitActiveCacheSupportedProceeds: the guard is not a blanket deny — a well-formed,
// established, cold-path-witnessed, measured-provenance capability still admits its own
// active operation.
func TestAdmitActiveCacheSupportedProceeds(t *testing.T) {
	for _, op := range activeVerdicts() {
		for _, p := range measuredProvenances() {
			cc := CacheCapability{
				Engine: "good-engine", Verdict: op,
				Provenance: p, Evidence: "unit-test", ColdPathCorrect: true,
			}
			got := AdmitActiveCache(cc, op)
			if got != RefusalNone {
				t.Errorf("supported op %q (provenance %q): refusal = %q, want admit (%q)",
					op, p, got, RefusalNone)
			}
			if got.Refused() {
				t.Errorf("supported op %q (provenance %q): Refused() = true, want false", op, p)
			}
		}
	}
}

// TestAdmitActiveCacheNeverOptimistic is the exhaustive matrix witness. Over every
// (verdict x provenance x cold-path x op) combination it asserts the admission decision
// is EXACTLY equivalent to the conjunction of the honesty conditions. Both directions
// matter: no combination is silently admitted without every condition holding (no
// optimistic claim), and no fully-honest combination is spuriously refused. Every
// returned reason is a member of the closed vocabulary.
func TestAdmitActiveCacheNeverOptimistic(t *testing.T) {
	combos := 0
	admits := 0
	for _, v := range CacheVerdicts() {
		for _, p := range CacheProvenances() {
			for _, cold := range []bool{false, true} {
				for _, op := range CacheVerdicts() {
					combos++
					cc := CacheCapability{
						Engine: "matrix-engine", Verdict: v, Provenance: p, ColdPathCorrect: cold,
					}
					got := AdmitActiveCache(cc, op)

					if !got.Valid() {
						t.Fatalf("refusal %q is outside the closed vocabulary (cc=%+v op=%q)", got, cc, op)
					}

					// The honesty conjunction the admission MUST be equivalent to.
					honest := op.Active() &&
						cc.Valid() &&
						v != CacheUnknown &&
						v == op &&
						cold &&
						p != ProvenanceForecast

					if honest && got != RefusalNone {
						t.Errorf("honest combo spuriously refused %q: cc=%+v op=%q", got, cc, op)
					}
					if !honest && got == RefusalNone {
						t.Errorf("OPTIMISTIC ADMIT of a dishonest combo: cc=%+v op=%q", cc, op)
					}
					if got == RefusalNone {
						admits++
					}
				}
			}
		}
	}
	// Sanity: the matrix actually exercised both outcomes, so the equivalence above is
	// not vacuously true.
	if combos == 0 || admits == 0 || admits == combos {
		t.Fatalf("degenerate matrix: combos=%d admits=%d", combos, admits)
	}
	// 3 active verdicts x 3 measured provenances x cold=true x matching op.
	if want := 9; admits != want {
		t.Errorf("admitted %d combos, want %d", admits, want)
	}
}

// TestCacheRefusalClosedVocabulary pins the enum: exactly the listed members are Valid,
// RefusalNone is the ONLY non-refusal, and a value read off a wire that is not in the
// vocabulary is treated as REFUSED (fail closed), never as an admit.
func TestCacheRefusalClosedVocabulary(t *testing.T) {
	for _, r := range CacheRefusals() {
		if !r.Valid() {
			t.Errorf("vocabulary member %q reports Valid() == false", r)
		}
	}
	for _, r := range CacheRefusals() {
		if (r == RefusalNone) == r.Refused() {
			t.Errorf("member %q: Refused() = %v, inconsistent with RefusalNone identity", r, r.Refused())
		}
	}
	for _, bogus := range []CacheRefusal{"bogus", "admit", "ok", "none"} {
		if bogus.Valid() {
			t.Errorf("out-of-vocabulary refusal %q reports Valid() == true", bogus)
		}
		if !bogus.Refused() {
			t.Errorf("out-of-vocabulary refusal %q must fail closed (Refused() == true)", bogus)
		}
	}
	// The zero value of the type is the admit sentinel, so a forgotten assignment cannot
	// silently become a refusal-shaped admit of some other name.
	var zero CacheRefusal
	if zero != RefusalNone || zero.Refused() {
		t.Errorf("zero value = %q, Refused() = %v; want RefusalNone and false", zero, zero.Refused())
	}
}
