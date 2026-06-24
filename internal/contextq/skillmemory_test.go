package contextq

import (
	"testing"

	"github.com/anthony-chaudhary/fak/internal/abi"
	"github.com/anthony-chaudhary/fak/internal/cachemeta"
)

// TestSkillProcedureMemoryHitOnReInvocation is the central witness for #513: a
// SkillContextRecord's procedural-memory view, stored once under its invocation
// digest, is served as a HIT from the SAME ViewCache on a re-invocation with an
// identical digest — re-rendering nothing. The economic proof is the build closure:
// it runs exactly once for the cold build and is NEVER called again on the warm HIT.
func TestSkillProcedureMemoryHitOnReInvocation(t *testing.T) {
	cache := NewViewCache()
	rec := SkillContextRecord{
		SkillName:        "code-review",
		Version:          "1.4.0",
		InvocationDigest: "sha256:deadbeefcafe0001",
		Producer:         "skillrunner",
		Scope:            abi.ScopeAgent,
	}

	builds := 0
	body := []byte("## procedure: code-review\nstep 1 ... step 2 ...")
	build := func() []byte {
		builds++
		return append([]byte(nil), body...)
	}

	// Invocation 1 — cold: no view cached for this digest yet -> FAULT, build runs.
	cold := rec.Resolve(cache, build)
	if cold.Verdict.Kind != MaterializationFault {
		t.Fatalf("cold invocation: kind=%q, want FAULT (%+v)", cold.Verdict.Kind, cold.Verdict)
	}
	if !cold.Built {
		t.Fatal("cold invocation: build closure should have run")
	}
	if builds != 1 {
		t.Fatalf("cold invocation: build ran %d times, want 1", builds)
	}
	if string(cold.Payload) != string(body) {
		t.Fatalf("cold payload = %q, want %q", cold.Payload, body)
	}

	// The cold view must lower into valid memory-view cache metadata, keyed (in the
	// producer slot) by the invocation digest, faithful and source-scoped.
	if cold.View.ViewType != ViewProcedure {
		t.Fatalf("cold view type = %q, want %q", cold.View.ViewType, ViewProcedure)
	}
	if cold.View.CacheEntry.Plane != cachemeta.PlaneMemoryView ||
		cold.View.CacheEntry.ID.MediaType != cachemeta.MediaMemoryView {
		t.Fatalf("procedural view did not lower into memory-view cache metadata: %+v", cold.View.CacheEntry)
	}
	if cold.View.CacheEntry.ID.Digest != rec.InvocationDigest {
		t.Fatalf("synthesized entry digest = %q, want the invocation digest %q",
			cold.View.CacheEntry.ID.Digest, rec.InvocationDigest)
	}
	if cold.View.FaithfulnessProbe != 1.0 {
		t.Fatalf("procedural view must be faithful (1.0), got %f", cold.View.FaithfulnessProbe)
	}
	if cold.View.Scope != abi.ScopeAgent {
		t.Fatalf("scope did not carry through: %v", cold.View.Scope)
	}
	if cold.View.Labels["invocation_digest"] != rec.InvocationDigest {
		t.Fatalf("invocation digest not stamped in labels: %+v", cold.View.Labels)
	}

	// Invocation 2 — warm, IDENTICAL digest: HIT served from the cache, build NEVER
	// runs again. This is the proof the goal demands.
	warm := rec.Resolve(cache, build)
	if warm.Verdict.Kind != MaterializationHit {
		t.Fatalf("re-invocation: kind=%q, want HIT (%+v)", warm.Verdict.Kind, warm.Verdict)
	}
	if warm.Verdict.Reason != "skill_procedure_cache_hit" {
		t.Fatalf("re-invocation reason = %q, want skill_procedure_cache_hit", warm.Verdict.Reason)
	}
	if warm.Built {
		t.Fatal("re-invocation: build closure must NOT run on a HIT")
	}
	if builds != 1 {
		t.Fatalf("re-invocation re-rendered: build ran %d times, want still 1", builds)
	}
	if string(warm.Payload) != string(body) {
		t.Fatalf("HIT payload = %q, want the cached body %q", warm.Payload, body)
	}
	if warm.View.ViewID != cold.View.ViewID {
		t.Fatalf("HIT served a different view: %q vs cold %q", warm.View.ViewID, cold.View.ViewID)
	}

	// A different invocation digest must NOT alias the cached view — it is a fresh
	// cold build. This proves the cache key IS the invocation digest (no hashing /
	// no silent cross-invocation reuse).
	other := rec
	other.InvocationDigest = "sha256:00001111ffffeeee"
	miss := other.Resolve(cache, build)
	if miss.Verdict.Kind != MaterializationFault {
		t.Fatalf("distinct digest: kind=%q, want FAULT (a fresh build)", miss.Verdict.Kind)
	}
	if builds != 2 {
		t.Fatalf("distinct digest: build ran %d times total, want 2", builds)
	}

	// And the original digest is still a HIT after the unrelated build (the slots are
	// independent).
	again := rec.Resolve(cache, build)
	if again.Verdict.Kind != MaterializationHit {
		t.Fatalf("original digest after a sibling build: kind=%q, want HIT", again.Verdict.Kind)
	}
	if builds != 2 {
		t.Fatalf("original-digest HIT re-rendered: build ran %d times, want still 2", builds)
	}
}

// TestSkillProcedureVersionBumpReKeys proves a skill version bump re-keys the
// procedural memory: the same skill+digest at a new version is a distinct slot (a
// cold build), never a stale HIT against the old version's view.
func TestSkillProcedureVersionBumpReKeys(t *testing.T) {
	cache := NewViewCache()
	rec := SkillContextRecord{
		SkillName:        "deep-research",
		Version:          "2.0.0",
		InvocationDigest: "sha256:abcabcabc",
		Producer:         "skillrunner",
		Scope:            abi.ScopeAgent,
	}
	builds := 0
	build := func() []byte { builds++; return []byte("body") }

	if v := rec.Resolve(cache, build).Verdict.Kind; v != MaterializationFault {
		t.Fatalf("v2.0.0 cold: kind=%q, want FAULT", v)
	}
	if v := rec.Resolve(cache, build).Verdict.Kind; v != MaterializationHit {
		t.Fatalf("v2.0.0 warm: kind=%q, want HIT", v)
	}

	bumped := rec
	bumped.Version = "2.1.0"
	if v := bumped.Resolve(cache, build).Verdict.Kind; v != MaterializationFault {
		t.Fatalf("v2.1.0 (bumped) should re-key to a cold FAULT, got %q", v)
	}
	if builds != 2 {
		t.Fatalf("build ran %d times, want 2 (one per distinct version)", builds)
	}
}

// TestSkillProcedureSuppliedCacheEntry confirms a caller-supplied CacheEntry is used
// verbatim (not overwritten by a synthesized one) and still resolves to a HIT.
func TestSkillProcedureSuppliedCacheEntry(t *testing.T) {
	cache := NewViewCache()
	supplied := cachemeta.FromMemoryView(cachemeta.MemoryView{
		ViewID:            "prelowered",
		ViewType:          string(ViewProcedure),
		Digest:            "sha256:supplied-entry-digest",
		Length:            7,
		Producer:          "skillrunner",
		Scope:             abi.ScopeFleet,
		Coverage:          1.0,
		FaithfulnessProbe: 1.0,
	})
	rec := SkillContextRecord{
		SkillName:        "industry-score",
		Version:          "1.0.0",
		InvocationDigest: "sha256:111222333",
		Producer:         "skillrunner",
		Scope:            abi.ScopeFleet,
		CacheEntry:       supplied,
	}
	build := func() []byte { return []byte("scored.") }

	cold := rec.Resolve(cache, build)
	if cold.Verdict.Kind != MaterializationFault {
		t.Fatalf("cold: kind=%q, want FAULT", cold.Verdict.Kind)
	}
	if cold.View.CacheEntry.ID.Digest != "sha256:supplied-entry-digest" {
		t.Fatalf("supplied cache entry was not used verbatim: %+v", cold.View.CacheEntry.ID)
	}
	warm := rec.Resolve(cache, build)
	if warm.Verdict.Kind != MaterializationHit {
		t.Fatalf("warm: kind=%q, want HIT", warm.Verdict.Kind)
	}
	if warm.View.CacheEntry.ID.Digest != "sha256:supplied-entry-digest" {
		t.Fatalf("HIT lost the supplied cache entry: %+v", warm.View.CacheEntry.ID)
	}
}
