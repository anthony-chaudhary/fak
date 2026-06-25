package callavoid_test

// scope_flip_test.go — the acceptance-bullet-#3 witness for issue #821: it connects
// the vDSO invalidation GRANULARITY (internal/vdso) to callavoid.ProveMemo's effective
// MUTATION RATE and shows the resulting REFUTE -> PROVE flip.
//
// Bullets #1 (scoped invalidation) and #2 (a write to the SAME entity still
// invalidates) are already witnessed inside package vdso (scope_test.go,
// pathscope_test.go). What was missing — and what this test adds — is the END-TO-END
// link: under the OLD global world-version, an unrelated write strands a stable read,
// so its effective mutation rate is ~1 and the economics gate REFUTES memoization; the
// narrowed per-path (Resource) scope leaves the stable read warm, so m ~= 0 and the
// SAME gate PROVES it. No production code is exercised beyond the two shipped APIs.
//
// Determinism. Nothing here uses time or randomness. The two derived mutation rates are
// the literal EXTREMES 0.0 and 1.0 (traced below), so the ProveMemo flip has the widest
// possible margin and is robust to any reasonable cost choice — there is no statistical
// "small deviation" to worry about; the m values are exact by construction.

import (
	"context"
	"fmt"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/abi"
	"github.com/anthony-chaudhary/fak/internal/callavoid"
	"github.com/anthony-chaudhary/fak/internal/vdso"

	// Mirror the vdso unit tests: register a blob resolver so the fill/lookup path
	// behaves identically to scope_test.go. (Our payloads/args are inline refs, so a
	// resolver is not strictly required — casDigest(inline)==false makes Pin/Unpin
	// no-ops — but matching the idiom keeps this witness parity-clean.)
	_ "github.com/anthony-chaudhary/fak/internal/blob"
)

// roRead builds a read-only, idempotent file Read call for one path. The hint pair is
// what routes it through the vDSO tier-1/tier-2 read path (vdso.Lookup re-checks, never
// trusts, the hints), and file_path is the Claude Code harness key the per-path scope
// (pathscope.go) keys its "files:<path>" generation on.
func roRead(path string) *abi.ToolCall {
	return &abi.ToolCall{
		Tool: "Read",
		Args: abi.Ref{Kind: abi.RefInline, Inline: []byte(`{"file_path":"` + path + `"}`)},
		Meta: map[string]string{"readOnlyHint": "true", "idempotentHint": "true"},
	}
}

// wrEdit builds a write-shaped Edit of one path. "Edit" is write-shaped by NAME
// (vdso.IsWriteShaped), so no hints are needed: Emit treats its completion as a
// mutation and bumps exactly "files:<path>" under Resource (or the root under Global).
func wrEdit(path string) *abi.ToolCall {
	return &abi.ToolCall{
		Tool: "Edit",
		Args: abi.Ref{Kind: abi.RefInline, Inline: []byte(`{"file_path":"` + path + `","old_string":"x","new_string":"y"}`)},
	}
}

// complete wraps a call + an OK inline result into the EvComplete event vdso.Emit
// consumes (to fill the tier-2 cache for a read, or to bump the world for a write).
func complete(c *abi.ToolCall, payload string) abi.Event {
	return abi.Event{
		Kind: abi.EvComplete,
		Call: c,
		Result: &abi.Result{
			Call:    c,
			Status:  abi.StatusOK,
			Payload: abi.Ref{Kind: abi.RefInline, Inline: []byte(payload)},
		},
	}
}

func lookupHits(v *vdso.VDSO, c *abi.ToolCall) bool {
	_, ok := v.Lookup(context.Background(), c)
	return ok
}

// runStableReadWorkload fills a STABLE read of stablePath, then drives `reuses`
// reuse-attempts of that read, each preceded by an Edit to a fresh UNRELATED path. It
// returns how many reuse-attempts still hit. The only difference between the Global and
// Resource runs is SetGranularity — same reads, same writes, same order — so the hit
// count isolates exactly the effect of narrowing the world-version to the written scope.
func runStableReadWorkload(t *testing.T, g vdso.Granularity, stablePath string, reuses int) (hits int) {
	t.Helper()
	v := vdso.New(256)
	v.SetGranularity(g)

	stable := roRead(stablePath)
	v.Emit(complete(stable, `package stable`))
	// Sanity: with no intervening write, the freshly-filled entry must hit, or the
	// workload below would measure a fill bug rather than an invalidation-scope effect.
	if !lookupHits(v, stable) {
		t.Fatalf("[%s] stable read did not hit right after fill — cannot measure invalidation scope", g)
	}

	for i := 0; i < reuses; i++ {
		// One write to an UNRELATED path (different from the stable path and from each
		// other). Under Global this bumps the root epoch (worldVer++) and strands EVERY
		// entry; under Resource it bumps only "files:/work/other_i.go" and leaves the
		// stable entry's "files:<stablePath>" generation untouched.
		other := fmt.Sprintf("/work/other_%d.go", i)
		v.Emit(complete(wrEdit(other), `{"ok":true}`))
		if lookupHits(v, stable) {
			hits++
		}
	}
	return hits
}

// TestScopeFlip_RefuteToProveAcrossGranularity is the issue-#821 acceptance witness.
//
// Mechanism trace (no run needed to know the m values — they are exact):
//
//	Global   : the FIRST unrelated Edit does worldVer 0->1, so every later Lookup of the
//	           stable read recomputes key "...:<worldVer>" and misses the "...:0" entry.
//	           All `reuses` reuse-attempts miss  => m_global = reuses/reuses = 1.0.
//	Resource : each Edit bumps only nodes["files:/work/other_i.go"]; the stable read binds
//	           ["*","files","files:/work/stable.go"], whose epochs all stay 0, so every
//	           reuse-attempt hits => m_resource = 0/reuses = 0.0. (reuses << nodeCap=8192,
//	           so no node-table trim flushes the root.)
//
// ProveMemo flip (threshold from avoid.go: PROVEN iff saved>0, else REFUTED; the
// per-reuse net gain is D = 1 - m - v - m·c, and saved = (k-1)·D - c):
//
//	m_global = 1.0, v=c=0.02, k=accesses:
//	   D = 1 - 1 - 0.02 - 1·0.02 = -0.04 <= 0  => REFUTED ("never memoize this class").
//	   (D<=0 for ANY non-negative v,c when m=1, so REFUTE has no cost-tuning fragility.)
//	m_resource = 0.0, v=c=0.02, k=accesses:
//	   D = 1 - 0 - 0.02 - 0 = 0.98 ;  saved = (k-1)·0.98 - 0.02 > 0 for every k>=2.
//	   With k=25, saved = 24·0.98 - 0.02 = 23.5  => PROVEN, with a wide margin.
func TestScopeFlip_RefuteToProveAcrossGranularity(t *testing.T) {
	const (
		stablePath = "/work/stable.go"
		reuses     = 24         // reuse-attempts of the stable read (the denominator of m)
		accesses   = reuses + 1 // total stable-read proposals in the window: 1 fill + reuses
		validate   = 0.02       // v: a world-version / generation check is cheap, never free
		capture    = 0.02       // c: storing a fingerprint+result in the LRU is cheap
	)

	// --- Drive the two real vDSO instances and derive the effective mutation rates. ---
	hitsGlobal := runStableReadWorkload(t, vdso.Global, stablePath, reuses)
	hitsResource := runStableReadWorkload(t, vdso.Resource, stablePath, reuses)

	strandedGlobal := reuses - hitsGlobal
	strandedResource := reuses - hitsResource
	mGlobal := float64(strandedGlobal) / float64(reuses)
	mResource := float64(strandedResource) / float64(reuses)

	// The mechanism itself, asserted directly: Global over-invalidates (every unrelated
	// write strands the stable read), Resource narrows the world-version to the written
	// path and leaves it warm. This is the on-the-cache half of bullets #1/#2, now
	// measured through the SAME workload that feeds the economics gate.
	if hitsGlobal != 0 {
		t.Fatalf("Global: stable read hit %d/%d times after unrelated writes; want 0 (the global world-version strands every entry)", hitsGlobal, reuses)
	}
	if hitsResource != reuses {
		t.Fatalf("Resource: stable read hit %d/%d times after unrelated writes; want all %d (per-path scope spares it)", hitsResource, reuses, reuses)
	}
	if mGlobal != 1.0 {
		t.Fatalf("m_global = %v, want 1.0 (high effective mutation rate under global invalidation)", mGlobal)
	}
	if mResource != 0.0 {
		t.Fatalf("m_resource = %v, want 0.0 (no effective mutation under per-path invalidation)", mResource)
	}

	// --- Feed each derived mutation rate into the SAME economics gate and assert the flip. ---
	refuted := callavoid.ProveMemo(callavoid.MemoInput{
		Accesses:     accesses,
		ValidateCost: validate,
		MutationRate: mGlobal,
		CaptureCost:  capture,
	})
	if refuted.Status != callavoid.ProofRefuted {
		t.Fatalf("global-scope mutation rate %v should REFUTE memoization, got %s (%s)", mGlobal, refuted.Status, refuted.Reason)
	}
	if refuted.PerReuseNetGain > 0 {
		t.Errorf("global-scope per-reuse net gain = %v, want <= 0 (mutation overhead swamps the saved execution)", refuted.PerReuseNetGain)
	}

	proven := callavoid.ProveMemo(callavoid.MemoInput{
		Accesses:     accesses,
		ValidateCost: validate,
		MutationRate: mResource,
		CaptureCost:  capture,
	})
	if proven.Status != callavoid.ProofProven {
		t.Fatalf("resource-scope mutation rate %v should PROVE memoization, got %s (%s)", mResource, proven.Status, proven.Reason)
	}
	if proven.SavedCost <= 0 {
		t.Errorf("resource-scope saved cost = %v, want > 0 (the narrowed scope makes avoidance pay)", proven.SavedCost)
	}

	// The headline: only the invalidation granularity changed, yet the verdict flipped.
	if !(refuted.Status == callavoid.ProofRefuted && proven.Status == callavoid.ProofProven) {
		t.Fatalf("expected REFUTE(global)->PROVE(resource) flip; got %s -> %s", refuted.Status, proven.Status)
	}
}

// TestScopeFlip_SoundnessSamePathStillInvalidates is the bullet-#5 soundness guard on
// the SAME workload: narrowing the scope must not mean "never invalidate". Under
// Resource, a write to the STABLE path itself still strands its read (a hit equals a
// fresh call), so the per-path scope earns its m~=0 honestly rather than by being inert.
func TestScopeFlip_SoundnessSamePathStillInvalidates(t *testing.T) {
	const stablePath = "/work/stable.go"

	v := vdso.New(256)
	v.SetGranularity(vdso.Resource)

	stable := roRead(stablePath)
	v.Emit(complete(stable, `package stable`))

	// Unrelated writes leave it warm (the precision half)...
	v.Emit(complete(wrEdit("/work/unrelated_a.go"), `{"ok":true}`))
	v.Emit(complete(wrEdit("/work/unrelated_b.go"), `{"ok":true}`))
	if !lookupHits(v, stable) {
		t.Fatalf("stable read missed after writes to UNRELATED paths — per-path scope must spare it")
	}

	// ...but a write to the SAME path strands it (the soundness half).
	v.Emit(complete(wrEdit(stablePath), `{"ok":true}`))
	if lookupHits(v, stable) {
		t.Fatalf("stable read STILL hit after a write to its own path — the scope is inert, not invalidating (soundness violated)")
	}
}
