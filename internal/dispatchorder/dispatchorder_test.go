package dispatchorder

import (
	"encoding/json"
	"fmt"
	"strings"
	"testing"
)

// base is a fixed "now" the tests reason against; recency/cooldown are offsets from it.
const base = 1_000_000

// allZeroGolden is the pinned JSON of Plan over the TestAllZeroPriorityByteIdenticalResult input
// with every Priority zero — the additive-no-regression witness (#3222). The absence of any
// "priority" key (omitempty on the zero value) is itself the proof that a zero-priority candidate
// serializes exactly as it did before the field existed. Regenerate ONLY when an intentional
// ordering change lands (never to paper over a priority regression).
const allZeroGolden = `
{
  "order": [
    {
      "id": "solo-new",
      "key": "S",
      "created_unix": 0,
      "updated_unix": 999950,
      "last_attempt_unix": 0,
      "live": false,
      "disposition": "keep",
      "reason": "freshest",
      "recency": 999950,
      "rank": 0
    },
    {
      "id": "a",
      "key": "K",
      "created_unix": 0,
      "updated_unix": 999900,
      "last_attempt_unix": 0,
      "live": false,
      "disposition": "keep",
      "reason": "freshest",
      "recency": 999900,
      "rank": 1
    },
    {
      "id": "solo-old",
      "key": "T",
      "created_unix": 0,
      "updated_unix": 999600,
      "last_attempt_unix": 0,
      "live": false,
      "disposition": "keep",
      "reason": "freshest",
      "recency": 999600,
      "rank": 2
    },
    {
      "id": "running",
      "key": "R",
      "created_unix": 0,
      "updated_unix": 999990,
      "last_attempt_unix": 0,
      "live": true,
      "disposition": "live",
      "reason": "worker_live",
      "recency": 999990,
      "rank": -1
    },
    {
      "id": "cooling",
      "key": "C",
      "created_unix": 0,
      "updated_unix": 999970,
      "last_attempt_unix": 999940,
      "live": false,
      "disposition": "cooling",
      "reason": "cooldown",
      "cooling_since": 999940,
      "cooling_until": 1000540,
      "recency": 999970,
      "rank": -1
    },
    {
      "id": "b",
      "key": "K",
      "created_unix": 0,
      "updated_unix": 999800,
      "last_attempt_unix": 0,
      "live": false,
      "disposition": "superseded",
      "reason": "superseded_by_fresher",
      "superseded_by": "a",
      "recency": 999800,
      "rank": -1
    }
  ],
  "keep": [
    "solo-new",
    "a",
    "solo-old"
  ],
  "keep_count": 3,
  "superseded_count": 1,
  "live_count": 1,
  "cooling_count": 1,
  "collision_count": 0,
  "generation_held_count": 0,
  "blocked_count": 0,
  "collisions_avoided": 0,
  "lanes_utilized": 3,
  "serialization_wasted": 0,
  "safe_concurrency": 3
}`

// dispoOf returns the disposition the planner gave the unit with id, or "" if absent.
func dispoOf(r Result, id string) Disposition {
	for _, x := range r.Order {
		if x.ID == id {
			return x.Disposition
		}
	}
	return ""
}

// TestSupersedeKeepsFreshest is the headline scenario: 25 tasks spawned for the SAME target
// (one shared key) collapse to the single freshest unit — that one is kept and picked, the
// other 24 are superseded (not eligible, not re-attempted), and the pick is the freshest.
func TestSupersedeKeepsFreshest(t *testing.T) {
	var cands []Candidate
	for i := 0; i < 25; i++ {
		cands = append(cands, Candidate{
			ID:          fmt.Sprintf("task-%02d", i),
			Key:         "X",
			UpdatedUnix: int64(base - 10_000 + i*100), // task-24 is the most recently updated
		})
	}
	r := Plan(Input{Candidates: cands, NowUnix: base})

	if r.KeepCount != 1 || r.SupersededCount != 24 {
		t.Fatalf("counts = keep %d superseded %d, want 1/24", r.KeepCount, r.SupersededCount)
	}
	if r.Pick() != "task-24" {
		t.Errorf("pick = %q, want task-24 (the freshest)", r.Pick())
	}
	if dispoOf(r, "task-24") != DispKeep {
		t.Errorf("freshest task-24 disposition = %q, want keep", dispoOf(r, "task-24"))
	}
	if dispoOf(r, "task-00") != DispSuperseded {
		t.Errorf("stale task-00 disposition = %q, want superseded", dispoOf(r, "task-00"))
	}
	// Every superseded unit names the winner.
	for _, x := range r.Order {
		if x.Disposition == DispSuperseded && x.SupersededBy != "task-24" {
			t.Errorf("%s superseded_by = %q, want task-24", x.ID, x.SupersededBy)
		}
	}
}

// TestDistinctKeysAllKeptFreshestFirst: units with DIFFERENT keys are all distinct targets, so
// all are kept and ordered freshest-first (by recency, not by id).
func TestDistinctKeysAllKeptFreshestFirst(t *testing.T) {
	r := Plan(Input{NowUnix: base, Candidates: []Candidate{
		{ID: "a", Key: "A", UpdatedUnix: base - 300},
		{ID: "b", Key: "B", UpdatedUnix: base - 100}, // freshest
		{ID: "c", Key: "C", UpdatedUnix: base - 200},
	}})
	if r.KeepCount != 3 {
		t.Fatalf("keep = %d, want 3 (distinct keys never supersede)", r.KeepCount)
	}
	want := []string{"b", "c", "a"} // freshest-first
	for i, id := range want {
		if r.Keep[i] != id {
			t.Errorf("keep[%d] = %q, want %q (freshest-first order)", i, r.Keep[i], id)
		}
	}
}

// TestPreferOldestOrdersDistinctKeysOldestFirst: with PreferOldest, distinct kept units are
// ordered OLDEST-first by creation, even when the oldest ticket has the FRESHEST update — the
// backlog-draining policy that makes the dispatcher pick the longest-waiting ticket first.
func TestPreferOldestOrdersDistinctKeysOldestFirst(t *testing.T) {
	r := Plan(Input{NowUnix: base, PreferOldest: true, Candidates: []Candidate{
		{ID: "new", Key: "A", CreatedUnix: base - 100, UpdatedUnix: base - 10},
		{ID: "old", Key: "B", CreatedUnix: base - 900, UpdatedUnix: base - 5}, // oldest created, freshest update
		{ID: "mid", Key: "C", CreatedUnix: base - 500, UpdatedUnix: base - 50},
	}})
	if r.KeepCount != 3 {
		t.Fatalf("keep = %d, want 3 (distinct keys never supersede)", r.KeepCount)
	}
	want := []string{"old", "mid", "new"} // oldest-created first, regardless of update recency
	for i, id := range want {
		if r.Keep[i] != id {
			t.Errorf("keep[%d] = %q, want %q (oldest-first)", i, r.Keep[i], id)
		}
	}
	if r.Pick() != "old" {
		t.Errorf("pick = %q, want old (oldest ticket, even though it has the freshest update)", r.Pick())
	}
}

// TestPreferOldestStillKeepsFreshestWithinKey: PreferOldest changes only the ORDER of distinct
// survivors; within ONE supersede key the freshest duplicate still wins (the others are
// superseded), so the de-dup contract is preserved.
func TestPreferOldestStillKeepsFreshestWithinKey(t *testing.T) {
	r := Plan(Input{NowUnix: base, PreferOldest: true, Candidates: []Candidate{
		{ID: "dup-old", Key: "X", CreatedUnix: base - 900, UpdatedUnix: base - 500},
		{ID: "dup-new", Key: "X", CreatedUnix: base - 100, UpdatedUnix: base - 50}, // freshest update of target X
	}})
	if r.Pick() != "dup-new" {
		t.Errorf("pick = %q, want dup-new (freshest within the key, even under PreferOldest)", r.Pick())
	}
	if dispoOf(r, "dup-old") != DispSuperseded {
		t.Errorf("dup-old disposition = %q, want superseded", dispoOf(r, "dup-old"))
	}
}

func TestCollisionPricedFanoutSerializesExclusiveOverlapBeforeLaunch(t *testing.T) {
	r := Plan(Input{NowUnix: base, Candidates: []Candidate{
		{ID: "gateway-old", Key: "A", Lane: "gateway", Tree: []string{"internal/gateway/**"}, UpdatedUnix: base - 300},
		{ID: "gateway-fresh", Key: "B", Lane: "gateway", Tree: []string{"internal/gateway/http.go"}, UpdatedUnix: base - 100},
		{ID: "docs", Key: "C", Lane: "docs", Tree: []string{"docs/**"}, UpdatedUnix: base - 200},
	}})

	if r.KeepCount != 2 || r.CollisionCount != 1 {
		t.Fatalf("counts = keep %d collision %d, want 2/1", r.KeepCount, r.CollisionCount)
	}
	if dispoOf(r, "gateway-fresh") != DispKeep || dispoOf(r, "docs") != DispKeep {
		t.Fatalf("safe set dispositions: gateway-fresh=%q docs=%q, want keep/keep",
			dispoOf(r, "gateway-fresh"), dispoOf(r, "docs"))
	}
	if dispoOf(r, "gateway-old") != DispCollisionRisk {
		t.Fatalf("gateway-old disposition = %q, want collision_risk", dispoOf(r, "gateway-old"))
	}
	if r.CollisionsAvoided != 1 || r.LanesUtilized != 2 || r.SerializationWasted != 1 || r.SafeConcurrency != 2 {
		t.Fatalf("S0 counts = avoided %d lanes %d wasted %d safe %d, want 1/2/1/2",
			r.CollisionsAvoided, r.LanesUtilized, r.SerializationWasted, r.SafeConcurrency)
	}
	if len(r.Collisions) != 1 || r.Collisions[0].Reason != ReasonCollisionRisk {
		t.Fatalf("collisions = %+v, want one %s edge", r.Collisions, ReasonCollisionRisk)
	}
	if len(r.Repartition) != 1 {
		t.Fatalf("repartition advice = %+v, want one row", r.Repartition)
	}
	adv := r.Repartition[0]
	if adv.Candidate != "gateway-old" || adv.Action != "narrow_to_issue_paths" || adv.Reason != ReasonCollisionRisk {
		t.Fatalf("repartition advice = %+v, want gateway-old narrow_to_issue_paths %s", adv, ReasonCollisionRisk)
	}
	if len(adv.CollidesWith) != 1 || adv.CollidesWith[0] != "gateway-fresh" {
		t.Fatalf("repartition peers = %#v, want gateway-fresh", adv.CollidesWith)
	}
}

func TestCollisionPricedFanoutAllowsSharedSharedOverlap(t *testing.T) {
	r := Plan(Input{NowUnix: base, Candidates: []Candidate{
		{ID: "reader-a", Key: "A", Lane: "docs", Tree: []string{"docs/**"}, Mode: "shared", UpdatedUnix: base - 10},
		{ID: "reader-b", Key: "B", Lane: "docs", Tree: []string{"docs/notes/**"}, Mode: "shared", UpdatedUnix: base - 20},
	}})
	if r.KeepCount != 2 || r.CollisionCount != 0 || len(r.Collisions) != 0 {
		t.Fatalf("shared/shared overlap counts = keep %d collision %d edges %d, want 2/0/0",
			r.KeepCount, r.CollisionCount, len(r.Collisions))
	}
}

func TestCollisionPricedFanoutUnknownTreeCollidesConservatively(t *testing.T) {
	r := Plan(Input{NowUnix: base, Candidates: []Candidate{
		{ID: "unknown", Key: "A", Lane: "gateway", UpdatedUnix: base - 10},
		{ID: "known", Key: "B", Lane: "docs", Tree: []string{"docs/**"}, UpdatedUnix: base - 20},
	}})
	if r.KeepCount != 1 || r.CollisionCount != 1 || r.CollisionsAvoided != 1 {
		t.Fatalf("unknown-tree pricing = keep %d collision %d avoided %d, want 1/1/1",
			r.KeepCount, r.CollisionCount, r.CollisionsAvoided)
	}
	if dispoOf(r, "unknown") != DispKeep || dispoOf(r, "known") != DispCollisionRisk {
		t.Fatalf("dispositions unknown=%q known=%q, want keep/collision_risk",
			dispoOf(r, "unknown"), dispoOf(r, "known"))
	}
	if len(r.Repartition) != 1 || r.Repartition[0].Action != "peer_declare_tree_scope" {
		t.Fatalf("known-tree repartition = %+v, want peer-scope advice for serialized known candidate", r.Repartition)
	}
}

// TestReadOnlyCandidateNeverCollidesOnFilePlane is the tri-state twin of the
// empty-tree conservative-collide floor (#3898): a candidate that declares
// ReadOnly writes NOTHING (a provably empty WRITE footprint), which is distinct
// from an ABSENT tree (unknown blast radius). Where the same-lane, overlapping
// pair serializes without it, the read-only reader is admitted alongside the
// writer it overlaps — it cannot clobber and cannot be clobbered.
func TestReadOnlyCandidateNeverCollidesOnFilePlane(t *testing.T) {
	plan := func(readOnly bool) Result {
		return Plan(Input{NowUnix: base, Candidates: []Candidate{
			{ID: "reader", Key: "A", Lane: "gateway", Tree: []string{"internal/gateway/**"}, ReadOnly: readOnly, UpdatedUnix: base - 10},
			{ID: "writer", Key: "B", Lane: "gateway", Tree: []string{"internal/gateway/http.go"}, UpdatedUnix: base - 20},
		}})
	}
	// Floor: without the opt-out, a same-lane overlapping pair serializes before launch.
	if r := plan(false); r.CollisionCount != 1 {
		t.Fatalf("same-lane overlap must collide without ReadOnly, got collision %d", r.CollisionCount)
	}
	// Opt-out: a read-only reader writes nothing, so both are admitted.
	r := plan(true)
	if r.KeepCount != 2 || r.CollisionCount != 0 || len(r.Collisions) != 0 {
		t.Fatalf("read-only reader must not collide = keep %d collision %d edges %d, want 2/0/0",
			r.KeepCount, r.CollisionCount, len(r.Collisions))
	}
	if dispoOf(r, "reader") != DispKeep || dispoOf(r, "writer") != DispKeep {
		t.Fatalf("both must keep, got reader=%q writer=%q", dispoOf(r, "reader"), dispoOf(r, "writer"))
	}
}

// TestPreferOldestCollisionAdmissionAdmitsOlder is the #3594 headline: under PreferOldest, when a
// collision forces one of two overlapping workers to serialize, the OLDER unit is admitted and the
// FRESHER one is priced collision_risk — the exact inverse of the default freshest-first admission
// (TestCollisionPricedFanoutSerializesExclusiveOverlapBeforeLaunch). Without this, PreferOldest
// reorders Keep only for the admission step to re-starve the longest-waiting unit it rescued.
func TestPreferOldestCollisionAdmissionAdmitsOlder(t *testing.T) {
	r := Plan(Input{NowUnix: base, PreferOldest: true, Candidates: []Candidate{
		{ID: "gateway-old", Key: "A", Lane: "gateway", Tree: []string{"internal/gateway/**"}, CreatedUnix: base - 900, UpdatedUnix: base - 300},
		{ID: "gateway-fresh", Key: "B", Lane: "gateway", Tree: []string{"internal/gateway/http.go"}, CreatedUnix: base - 100, UpdatedUnix: base - 100},
		{ID: "docs", Key: "C", Lane: "docs", Tree: []string{"docs/**"}, CreatedUnix: base - 500, UpdatedUnix: base - 200},
	}})

	if r.KeepCount != 2 || r.CollisionCount != 1 {
		t.Fatalf("counts = keep %d collision %d, want 2/1", r.KeepCount, r.CollisionCount)
	}
	if dispoOf(r, "gateway-old") != DispKeep || dispoOf(r, "docs") != DispKeep {
		t.Fatalf("safe set dispositions: gateway-old=%q docs=%q, want keep/keep",
			dispoOf(r, "gateway-old"), dispoOf(r, "docs"))
	}
	if dispoOf(r, "gateway-fresh") != DispCollisionRisk {
		t.Fatalf("gateway-fresh disposition = %q, want collision_risk (the fresher unit serializes under PreferOldest)",
			dispoOf(r, "gateway-fresh"))
	}
	want := []string{"gateway-old", "docs"} // oldest-created first
	for i, id := range want {
		if r.Keep[i] != id {
			t.Errorf("keep[%d] = %q, want %q (oldest-first admission)", i, r.Keep[i], id)
		}
	}
}

// TestMaxSafeSetTieBreaksTowardAdmissionOrder pins the property applyCollisionPrice relies on
// (#3594): when two same-size safe subsets compete ({old} vs {fresh} across one collision edge),
// maxSafeSet admits the candidate EARLIER in cands — the caller's admission sort (freshest-first
// by default, oldest-first under PreferOldest) is the tie-break, not a hardcoded recency rule.
func TestMaxSafeSetTieBreaksTowardAdmissionOrder(t *testing.T) {
	old := Ranked{Candidate: Candidate{ID: "old", CreatedUnix: base - 900, UpdatedUnix: base - 900}}
	fresh := Ranked{Candidate: Candidate{ID: "fresh", CreatedUnix: base - 10, UpdatedUnix: base - 10}}
	edge := []Collision{{A: "old", B: "fresh", Reason: ReasonCollisionRisk}}

	def := maxSafeSet([]Ranked{fresh, old}, edge) // freshest-first admission order (the default)
	if !def["fresh"] || def["old"] {
		t.Errorf("freshest-first order: safe = %v, want fresh admitted / old serialized", def)
	}
	aged := maxSafeSet([]Ranked{old, fresh}, edge) // oldest-first admission order (PreferOldest)
	if !aged["old"] || aged["fresh"] {
		t.Errorf("oldest-first order: safe = %v, want old admitted / fresh serialized", aged)
	}
}

// defaultCollisionAdmissionGolden pins the DEFAULT (no PreferOldest) collision admission
// byte-for-byte (#3594): freshest-first — the fresher gateway worker is kept, the older
// serialized — exactly as before the PreferOldest-aware admission landed. The CreatedUnix values
// are chosen so an accidental oldest-first default would flip the safe set and break this golden.
// Regenerate ONLY when an intentional default-admission change lands.
const defaultCollisionAdmissionGolden = `
{
  "order": [
    {
      "id": "gateway-fresh",
      "key": "B",
      "lane": "gateway",
      "tree": [
        "internal/gateway/http.go"
      ],
      "created_unix": 999900,
      "updated_unix": 999900,
      "last_attempt_unix": 0,
      "live": false,
      "disposition": "keep",
      "reason": "freshest",
      "recency": 999900,
      "rank": 0
    },
    {
      "id": "docs",
      "key": "C",
      "lane": "docs",
      "tree": [
        "docs/**"
      ],
      "created_unix": 999500,
      "updated_unix": 999800,
      "last_attempt_unix": 0,
      "live": false,
      "disposition": "keep",
      "reason": "freshest",
      "recency": 999800,
      "rank": 1
    },
    {
      "id": "gateway-old",
      "key": "A",
      "lane": "gateway",
      "tree": [
        "internal/gateway/**"
      ],
      "created_unix": 999100,
      "updated_unix": 999700,
      "last_attempt_unix": 0,
      "live": false,
      "disposition": "collision_risk",
      "reason": "COLLISION_RISK",
      "collides_with": [
        "gateway-fresh"
      ],
      "recency": 999700,
      "rank": -1
    }
  ],
  "keep": [
    "gateway-fresh",
    "docs"
  ],
  "keep_count": 2,
  "superseded_count": 0,
  "live_count": 0,
  "cooling_count": 0,
  "collision_count": 1,
  "generation_held_count": 0,
  "blocked_count": 0,
  "collisions": [
    {
      "a": "gateway-old",
      "b": "gateway-fresh",
      "reason": "COLLISION_RISK",
      "lane": [
        "gateway",
        "gateway"
      ]
    }
  ],
  "repartition": [
    {
      "candidate": "gateway-old",
      "lane": "gateway",
      "collides_with": [
        "gateway-fresh"
      ],
      "current_tree": [
        "internal/gateway"
      ],
      "action": "narrow_to_issue_paths",
      "reason": "COLLISION_RISK",
      "detail": "replace the broad lane tree with path-confirmed issue paths, then re-price"
    }
  ],
  "collisions_avoided": 1,
  "lanes_utilized": 2,
  "serialization_wasted": 1,
  "safe_concurrency": 2
}`

// TestDefaultCollisionAdmissionByteIdentical: the same candidate set as the PreferOldest headline
// test, with the flag OFF, serializes byte-identically to the pinned pre-#3594 default — the
// additive-no-regression witness for the admission change.
func TestDefaultCollisionAdmissionByteIdentical(t *testing.T) {
	in := Input{NowUnix: base, Candidates: []Candidate{
		{ID: "gateway-old", Key: "A", Lane: "gateway", Tree: []string{"internal/gateway/**"}, CreatedUnix: base - 900, UpdatedUnix: base - 300},
		{ID: "gateway-fresh", Key: "B", Lane: "gateway", Tree: []string{"internal/gateway/http.go"}, CreatedUnix: base - 100, UpdatedUnix: base - 100},
		{ID: "docs", Key: "C", Lane: "docs", Tree: []string{"docs/**"}, CreatedUnix: base - 500, UpdatedUnix: base - 200},
	}}
	got, err := json.MarshalIndent(Plan(in), "", "  ")
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}
	gotStr := strings.TrimSpace(string(got))
	want := strings.TrimSpace(strings.ReplaceAll(defaultCollisionAdmissionGolden, "\r\n", "\n"))
	if gotStr != want {
		t.Fatalf("default collision admission drifted from the pinned golden.\n--- got ---\n%s\n--- want ---\n%s",
			gotStr, want)
	}
}

func TestGenerationDefaultWindowHoldsLaterHorizons(t *testing.T) {
	tests := []struct {
		generation string
		want       Disposition
	}{
		{"gen/now", DispKeep},
		{"gen/next", DispKeep},
		{"gen/second-next", DispGenerationHeld},
		{"gen/future", DispGenerationHeld},
	}
	for _, tt := range tests {
		t.Run(tt.generation, func(t *testing.T) {
			r := Plan(Input{NowUnix: base, Candidates: []Candidate{
				{ID: tt.generation, Key: tt.generation, Generation: tt.generation, UpdatedUnix: base - 10},
			}})
			if got := dispoOf(r, tt.generation); got != tt.want {
				t.Fatalf("disposition = %q, want %q", got, tt.want)
			}
			if tt.want == DispKeep && r.Pick() != tt.generation {
				t.Fatalf("pick = %q, want %q", r.Pick(), tt.generation)
			}
			if tt.want == DispGenerationHeld && r.Pick() != "" {
				t.Fatalf("pick = %q, want no pick for held %s", r.Pick(), tt.generation)
			}
		})
	}
}

func TestGenerationExplicitWindowAdmitsRequestedHorizon(t *testing.T) {
	r := Plan(Input{
		NowUnix:    base,
		Generation: "future",
		Candidates: []Candidate{
			{ID: "now", Key: "now", Generation: "gen/now", UpdatedUnix: base - 10},
			{ID: "future", Key: "future", Generation: "gen/future", UpdatedUnix: base - 20},
		},
	})
	if r.Pick() != "future" {
		t.Fatalf("pick = %q, want future under explicit future window", r.Pick())
	}
	if dispoOf(r, "now") != DispGenerationHeld {
		t.Fatalf("now disposition = %q, want generation-held under explicit future window", dispoOf(r, "now"))
	}
	if dispoOf(r, "future") != DispKeep {
		t.Fatalf("future disposition = %q, want keep", dispoOf(r, "future"))
	}
}

func TestTreesOverlapUsesPrefixGeometry(t *testing.T) {
	if TreesOverlap([]string{"internal/gateway/http.go"}, []string{"internal/gateway/mcp.go"}) {
		t.Fatalf("sibling files should be disjoint")
	}
	if !TreesOverlap([]string{"internal/gateway/**"}, []string{"internal/gateway/http.go"}) {
		t.Fatalf("parent tree should overlap child file")
	}
	if !TreesOverlap(nil, []string{"docs/**"}) {
		t.Fatalf("unknown tree should collide conservatively")
	}
}

// TestEmptyKeyNeverSuperseded: an empty Key opts a unit out of collapse — two empty-key units
// are both kept even though they would otherwise look like duplicates.
func TestEmptyKeyNeverSuperseded(t *testing.T) {
	r := Plan(Input{NowUnix: base, Candidates: []Candidate{
		{ID: "u1", Key: "", UpdatedUnix: base - 50},
		{ID: "u2", Key: "", UpdatedUnix: base - 60},
	}})
	if r.KeepCount != 2 || r.SupersededCount != 0 {
		t.Errorf("empty-key units: keep %d superseded %d, want 2/0", r.KeepCount, r.SupersededCount)
	}
}

// TestLiveWinnerYieldsNoKeep: when the freshest unit for a key is already running, the group
// yields NO keep this tick — the older duplicates are superseded (not run), and the live one is
// reported live, so the dispatcher does not start a stale duplicate behind a running fresh one.
func TestLiveWinnerYieldsNoKeep(t *testing.T) {
	r := Plan(Input{NowUnix: base, Candidates: []Candidate{
		{ID: "fresh", Key: "X", UpdatedUnix: base - 100, Live: true},
		{ID: "stale", Key: "X", UpdatedUnix: base - 500},
	}})
	if r.KeepCount != 0 {
		t.Fatalf("keep = %d, want 0 (freshest is live)", r.KeepCount)
	}
	if dispoOf(r, "fresh") != DispLive {
		t.Errorf("fresh disposition = %q, want live", dispoOf(r, "fresh"))
	}
	if dispoOf(r, "stale") != DispSuperseded {
		t.Errorf("stale disposition = %q, want superseded", dispoOf(r, "stale"))
	}
	if r.Pick() != "" {
		t.Errorf("pick = %q, want empty (nothing dispatchable)", r.Pick())
	}
}

// TestCooldownHoldsFreshestNoFallback: the freshest unit attempted within the cooldown window is
// held (cooling) — and the planner does NOT fall back to an older duplicate, so the group still
// yields no keep. This is the deliberate v1 posture (freshest-or-wait).
func TestCooldownHoldsFreshestNoFallback(t *testing.T) {
	r := Plan(Input{NowUnix: base, CooldownSeconds: 600, Candidates: []Candidate{
		{ID: "fresh", Key: "X", UpdatedUnix: base - 100, LastAttemptUnix: base - 60}, // attempted 60s ago < 600s
		{ID: "stale", Key: "X", UpdatedUnix: base - 900},
	}})
	if dispoOf(r, "fresh") != DispCooling {
		t.Fatalf("fresh disposition = %q, want cooling", dispoOf(r, "fresh"))
	}
	if dispoOf(r, "stale") != DispSuperseded {
		t.Errorf("stale disposition = %q, want superseded (no fallback to older dup)", dispoOf(r, "stale"))
	}
	if r.Pick() != "" || r.KeepCount != 0 {
		t.Errorf("pick=%q keep=%d, want empty/0 (freshest cooling, no fallback)", r.Pick(), r.KeepCount)
	}
}

// TestCooldownExpiredKeeps: once the cooldown window passes, the freshest unit is keepable again.
func TestCooldownExpiredKeeps(t *testing.T) {
	r := Plan(Input{NowUnix: base, CooldownSeconds: 600, Candidates: []Candidate{
		{ID: "fresh", Key: "X", UpdatedUnix: base - 100, LastAttemptUnix: base - 700}, // 700s ago > 600s
	}})
	if r.Pick() != "fresh" {
		t.Errorf("pick = %q, want fresh (cooldown expired)", r.Pick())
	}
}

// TestUpdatedBeatsCreatedAndId: recency is UPDATED time, not creation and not id — a unit
// created later but updated earlier loses to one updated more recently. This is the fix for the
// old "freshest = largest issue number" behavior.
func TestUpdatedBeatsCreatedAndId(t *testing.T) {
	r := Plan(Input{NowUnix: base, Candidates: []Candidate{
		{ID: "zzz-new", Key: "X", CreatedUnix: base - 10, UpdatedUnix: base - 900}, // newest+highest id, but stale update
		{ID: "aaa-old", Key: "X", CreatedUnix: base - 999, UpdatedUnix: base - 50}, // oldest+lowest id, freshest update
	}})
	if r.Pick() != "aaa-old" {
		t.Errorf("pick = %q, want aaa-old (most recently UPDATED, not highest id/newest created)", r.Pick())
	}
}

// TestDeterministicAndTotal: identical inputs give identical results, and the empty input yields
// a defined empty result.
func TestDeterministicAndTotal(t *testing.T) {
	in := Input{NowUnix: base, Candidates: []Candidate{
		{ID: "a", Key: "K", UpdatedUnix: base - 1},
		{ID: "b", Key: "K", UpdatedUnix: base - 2},
		{ID: "c", Key: "", UpdatedUnix: base - 3},
	}}
	a, b := Plan(in), Plan(in)
	if fmt.Sprint(a) != fmt.Sprint(b) {
		t.Errorf("Plan is not deterministic:\n%v\n%v", a, b)
	}
	empty := Plan(Input{NowUnix: base})
	if len(empty.Order) != 0 || empty.Pick() != "" {
		t.Errorf("empty input = %+v, want empty/no-pick", empty)
	}
}

// TestPriorityLeadsRecencyOlderP0First is the headline priority proof (#3222): a priority/P0
// unit that is OLDER (staler update) than a no-priority unit ranks FIRST in Keep — declared
// priority leads recency. Distinct keys so neither supersedes the other; the older P0 wins the
// pick purely on Priority.
func TestPriorityLeadsRecencyOlderP0First(t *testing.T) {
	r := Plan(Input{NowUnix: base, Candidates: []Candidate{
		{ID: "fresh-noprio", Key: "A", UpdatedUnix: base - 10},              // freshest, no priority
		{ID: "old-p0", Key: "B", Priority: 1000, UpdatedUnix: base - 5_000}, // much older, but P0
	}})
	if r.KeepCount != 2 {
		t.Fatalf("keep = %d, want 2 (distinct keys never supersede)", r.KeepCount)
	}
	if r.Pick() != "old-p0" {
		t.Fatalf("pick = %q, want old-p0 (P0 leads recency even though it is older)", r.Pick())
	}
	want := []string{"old-p0", "fresh-noprio"}
	for i, id := range want {
		if r.Keep[i] != id {
			t.Errorf("keep[%d] = %q, want %q (priority-desc first)", i, r.Keep[i], id)
		}
	}
	if got := dispoOf(r, "old-p0"); got != DispKeep {
		t.Errorf("old-p0 disposition = %q, want keep", got)
	}
}

// TestPriorityTierOrderThenRecency: P0 > P1 > P2 > unlabeled, and within a tier the freshest
// leads — the full ordering table. All distinct keys so every unit is kept.
func TestPriorityTierOrderThenRecency(t *testing.T) {
	r := Plan(Input{NowUnix: base, Candidates: []Candidate{
		{ID: "p2", Key: "A", Priority: 150, UpdatedUnix: base - 10},
		{ID: "none-fresh", Key: "B", UpdatedUnix: base - 1},
		{ID: "p0", Key: "C", Priority: 1000, UpdatedUnix: base - 9_000},
		{ID: "p1", Key: "D", Priority: 400, UpdatedUnix: base - 20},
		{ID: "none-stale", Key: "E", UpdatedUnix: base - 500},
	}})
	want := []string{"p0", "p1", "p2", "none-fresh", "none-stale"}
	if len(r.Keep) != len(want) {
		t.Fatalf("keep = %v, want %v", r.Keep, want)
	}
	for i, id := range want {
		if r.Keep[i] != id {
			t.Errorf("keep[%d] = %q, want %q", i, r.Keep[i], id)
		}
	}
}

// TestPriorityBeatsPreferOldest: Priority still leads even under PreferOldest — the oldest-first
// fallback only applies once priority ties. A P0 unit is picked ahead of an older unlabeled one.
func TestPriorityBeatsPreferOldest(t *testing.T) {
	r := Plan(Input{NowUnix: base, PreferOldest: true, Candidates: []Candidate{
		{ID: "old-noprio", Key: "A", CreatedUnix: base - 9_000, UpdatedUnix: base - 9_000},
		{ID: "new-p0", Key: "B", Priority: 1000, CreatedUnix: base - 10, UpdatedUnix: base - 10},
	}})
	if r.Pick() != "new-p0" {
		t.Fatalf("pick = %q, want new-p0 (P0 leads even under prefer-oldest)", r.Pick())
	}
}

// TestAllZeroPriorityByteIdenticalResult is the additive-no-regression proof (#3222): a
// representative candidate set with EVERY Priority left at zero produces a Result whose JSON is
// byte-identical to the pinned pre-priority golden. If the priority tiebreak ever perturbs an
// all-zero input, the golden breaks. Covers supersede, live, cooling, and distinct-key ordering
// in one fold.
func TestAllZeroPriorityByteIdenticalResult(t *testing.T) {
	in := Input{NowUnix: base, CooldownSeconds: 600, Candidates: []Candidate{
		{ID: "b", Key: "K", UpdatedUnix: base - 200},
		{ID: "a", Key: "K", UpdatedUnix: base - 100}, // freshest of key K
		{ID: "solo-new", Key: "S", UpdatedUnix: base - 50},
		{ID: "solo-old", Key: "T", UpdatedUnix: base - 400},
		{ID: "running", Key: "R", UpdatedUnix: base - 10, Live: true},
		{ID: "cooling", Key: "C", UpdatedUnix: base - 30, LastAttemptUnix: base - 60},
	}}
	got, err := json.MarshalIndent(Plan(in), "", "  ")
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}
	// Normalize surrounding whitespace and any CRLF the raw literal may carry on a
	// Windows checkout so the comparison is purely about JSON content/ordering.
	gotStr := strings.TrimSpace(string(got))
	want := strings.TrimSpace(strings.ReplaceAll(allZeroGolden, "\r\n", "\n"))
	if gotStr != want {
		t.Fatalf("all-zero-priority Result drifted from the pinned golden.\n--- got ---\n%s\n--- want ---\n%s",
			gotStr, want)
	}
}

// TestCoolingVerdictCarriesWindow (#3715): a DispCooling verdict declares its cooldown window —
// CoolingSince is the last attempt, CoolingUntil is when the unit re-enters the pool — under the
// exact JSON keys internal/dispatchaging.Candidate reads (cooling_since / cooling_until), so an
// aging caller building Candidates from these dispositions inherits the pause window verbatim.
// Every other disposition carries a zero window (omitempty: the keys are absent), the additive
// no-regression direction.
func TestCoolingVerdictCarriesWindow(t *testing.T) {
	r := Plan(Input{NowUnix: base, CooldownSeconds: 600, Candidates: []Candidate{
		{ID: "cooling", Key: "C", UpdatedUnix: base - 30, LastAttemptUnix: base - 60},
		{ID: "kept", Key: "K", UpdatedUnix: base - 100},
	}})
	var cool, kept *Ranked
	for i := range r.Order {
		switch r.Order[i].ID {
		case "cooling":
			cool = &r.Order[i]
		case "kept":
			kept = &r.Order[i]
		}
	}
	if cool == nil || cool.Disposition != DispCooling {
		t.Fatalf("cooling unit not ranked DispCooling: %+v", cool)
	}
	if cool.CoolingSince != base-60 || cool.CoolingUntil != base-60+600 {
		t.Errorf("cooling window = [%d, %d], want [%d, %d] (last attempt -> attempt+cooldown)",
			cool.CoolingSince, cool.CoolingUntil, base-60, base-60+600)
	}
	if kept == nil || kept.CoolingSince != 0 || kept.CoolingUntil != 0 {
		t.Errorf("kept unit carries a cooling window %+v, want zero (additive no-regression)", kept)
	}
	// Bind the JSON seam: the window serializes under dispatchaging.Candidate's key names.
	b, err := json.Marshal(cool)
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}
	for _, key := range []string{`"cooling_since":999940`, `"cooling_until":1000540`} {
		if !strings.Contains(string(b), key) {
			t.Errorf("cooling row JSON %s missing %s (the dispatchaging seam)", b, key)
		}
	}
}

// TestNegativeCooldownDisables: a negative cooldown disables the hold — an attempted freshest is
// still keepable.
func TestNegativeCooldownDisables(t *testing.T) {
	r := Plan(Input{NowUnix: base, CooldownSeconds: -1, Candidates: []Candidate{
		{ID: "fresh", Key: "X", UpdatedUnix: base - 10, LastAttemptUnix: base - 1},
	}})
	if r.Pick() != "fresh" {
		t.Errorf("pick = %q, want fresh (cooldown disabled)", r.Pick())
	}
}

// TestBlockedByOpenPrereqSoftHold is the headline prerequisite gate (#3224): B declares A as a
// prerequisite (BlockedBy) and both are open this tick, so B is soft-held (DispBlocked, excluded
// from Keep but still present in Order with the reason) while its prerequisite A is kept. Once A
// leaves the candidate set (closed), B is dispatchable again — a hold, never a permanent ban.
func TestBlockedByOpenPrereqSoftHold(t *testing.T) {
	r := Plan(Input{NowUnix: base, Candidates: []Candidate{
		{ID: "A", Key: "A", UpdatedUnix: base - 100},
		{ID: "B", Key: "B", BlockedBy: []string{"A"}, UpdatedUnix: base - 50},
	}})
	if dispoOf(r, "A") != DispKeep {
		t.Fatalf("A disposition = %q, want keep (the open prerequisite is still dispatchable)", dispoOf(r, "A"))
	}
	if dispoOf(r, "B") != DispBlocked {
		t.Fatalf("B disposition = %q, want blocked (prerequisite A is still open)", dispoOf(r, "B"))
	}
	if r.BlockedCount != 1 || r.KeepCount != 1 {
		t.Fatalf("counts = blocked %d keep %d, want 1/1", r.BlockedCount, r.KeepCount)
	}
	// B is excluded from Keep but still legible in Order with the reason and the open prerequisite.
	for _, id := range r.Keep {
		if id == "B" {
			t.Fatalf("keep = %v, want B excluded (soft-held on its open prerequisite)", r.Keep)
		}
	}
	var b Ranked
	for _, x := range r.Order {
		if x.ID == "B" {
			b = x
		}
	}
	if b.Reason != ReasonBlockedByOpenPrereq {
		t.Fatalf("B reason = %q, want %q", b.Reason, ReasonBlockedByOpenPrereq)
	}
	if len(b.BlockedByOpen) != 1 || b.BlockedByOpen[0] != "A" {
		t.Fatalf("B blocked_by_open = %#v, want [A]", b.BlockedByOpen)
	}
	if r.Pick() != "A" {
		t.Fatalf("pick = %q, want A (B is soft-held behind its prerequisite)", r.Pick())
	}

	// Once the prerequisite A closes (leaves the set), B unblocks and is dispatchable.
	r2 := Plan(Input{NowUnix: base, Candidates: []Candidate{
		{ID: "B", Key: "B", BlockedBy: []string{"A"}, UpdatedUnix: base - 50},
	}})
	if dispoOf(r2, "B") != DispKeep {
		t.Fatalf("B disposition with A absent = %q, want keep (prerequisite closed)", dispoOf(r2, "B"))
	}
	if r2.Pick() != "B" {
		t.Fatalf("pick with A absent = %q, want B", r2.Pick())
	}
}

// TestBlockedByMutualCycleLowestIDDispatches: an A<->B mutual block (each names the other as a
// prerequisite) breaks deterministically toward the LOWEST ID — A dispatches, B is held — so a
// dependency cycle resolves to one dispatchable unit instead of hanging or freezing both.
func TestBlockedByMutualCycleLowestIDDispatches(t *testing.T) {
	r := Plan(Input{NowUnix: base, Candidates: []Candidate{
		{ID: "B", Key: "B", BlockedBy: []string{"A"}, UpdatedUnix: base - 10},
		{ID: "A", Key: "A", BlockedBy: []string{"B"}, UpdatedUnix: base - 20},
	}})
	if dispoOf(r, "A") != DispKeep {
		t.Fatalf("A disposition = %q, want keep (lowest ID wins the cycle)", dispoOf(r, "A"))
	}
	if dispoOf(r, "B") != DispBlocked {
		t.Fatalf("B disposition = %q, want blocked (higher ID yields in the cycle)", dispoOf(r, "B"))
	}
	if r.Pick() != "A" {
		t.Fatalf("pick = %q, want A (lowest ID dispatches, no hang)", r.Pick())
	}
	if r.KeepCount != 1 || r.BlockedCount != 1 {
		t.Fatalf("counts = keep %d blocked %d, want 1/1 (cycle resolves, never deadlocks)", r.KeepCount, r.BlockedCount)
	}
}

// TestBlockedByAbsentPrereqFailOpen: a BlockedBy id that is NOT in the candidate set is an
// already-closed prerequisite and does NOT block (fail-open) — the unit is dispatchable.
func TestBlockedByAbsentPrereqFailOpen(t *testing.T) {
	r := Plan(Input{NowUnix: base, Candidates: []Candidate{
		{ID: "B", Key: "B", BlockedBy: []string{"999-closed"}, UpdatedUnix: base - 50},
	}})
	if dispoOf(r, "B") != DispKeep {
		t.Fatalf("B disposition = %q, want keep (absent prerequisite is closed, fail-open)", dispoOf(r, "B"))
	}
	if r.BlockedCount != 0 || r.Pick() != "B" {
		t.Fatalf("blocked=%d pick=%q, want 0/B (fail-open on an absent prerequisite)", r.BlockedCount, r.Pick())
	}
}

// strSet is a tiny order-independent membership helper for asserting Region/OverlapRegion contents.
func strSet(xs []string) map[string]bool {
	m := make(map[string]bool, len(xs))
	for _, x := range xs {
		m[x] = true
	}
	return m
}

// TestComputeExclusiveOverlapSerializes is the compute-plane twin of the file-tree collision test
// (#3268): two exclusive claims on overlapping ranges of the SAME class serialize before launch —
// the fresher is kept, the older is priced collision_risk — while a claim in a DIFFERENT class
// never contends. The collision edge carries the compute Region, and the advice is compute-flavored.
func TestComputeExclusiveOverlapSerializes(t *testing.T) {
	r := Plan(Input{NowUnix: base, Candidates: []Candidate{
		{ID: "dev-fresh", Key: "A", Compute: &ComputeClaim{Class: "device", Range: "0-3"}, UpdatedUnix: base - 100},
		{ID: "dev-old", Key: "B", Compute: &ComputeClaim{Class: "device", Range: "2-5"}, UpdatedUnix: base - 300},
		{ID: "pool", Key: "C", Compute: &ComputeClaim{Class: "phase-pool", Range: "0-3"}, UpdatedUnix: base - 200},
	}})

	if r.KeepCount != 2 || r.CollisionCount != 1 {
		t.Fatalf("counts = keep %d collision %d, want 2/1", r.KeepCount, r.CollisionCount)
	}
	if dispoOf(r, "dev-fresh") != DispKeep || dispoOf(r, "pool") != DispKeep {
		t.Fatalf("safe set: dev-fresh=%q pool=%q, want keep/keep (different class never contends)",
			dispoOf(r, "dev-fresh"), dispoOf(r, "pool"))
	}
	if dispoOf(r, "dev-old") != DispCollisionRisk {
		t.Fatalf("dev-old disposition = %q, want collision_risk", dispoOf(r, "dev-old"))
	}
	if len(r.Collisions) != 1 || r.Collisions[0].Reason != ReasonCollisionRisk {
		t.Fatalf("collisions = %+v, want one %s edge", r.Collisions, ReasonCollisionRisk)
	}
	region := strSet(r.Collisions[0].Region)
	if len(r.Collisions[0].Region) != 2 || !region["device:0-3"] || !region["device:2-5"] {
		t.Fatalf("collision Region = %#v, want {device:0-3, device:2-5}", r.Collisions[0].Region)
	}
	if len(r.Collisions[0].Tree) != 0 {
		t.Fatalf("compute collision leaked a file Tree = %#v, want none", r.Collisions[0].Tree)
	}
	if len(r.Repartition) != 1 {
		t.Fatalf("repartition = %+v, want one row", r.Repartition)
	}
	adv := r.Repartition[0]
	if adv.Candidate != "dev-old" || adv.Action != "narrow_compute_range" || adv.Reason != ReasonCollisionRisk {
		t.Fatalf("advice = %+v, want dev-old narrow_compute_range %s", adv, ReasonCollisionRisk)
	}
	if len(adv.CurrentRegion) != 1 || adv.CurrentRegion[0] != "device:2-5" {
		t.Fatalf("advice CurrentRegion = %#v, want [device:2-5]", adv.CurrentRegion)
	}
	if ov := strSet(adv.OverlapRegion); !ov["device:0-3"] || !ov["device:2-5"] {
		t.Fatalf("advice OverlapRegion = %#v, want to include device:0-3 and device:2-5", adv.OverlapRegion)
	}
	if len(adv.CurrentTree) != 0 || len(adv.OverlapTree) != 0 {
		t.Fatalf("compute advice leaked file scope: CurrentTree=%#v OverlapTree=%#v", adv.CurrentTree, adv.OverlapTree)
	}
}

// TestComputeDisjointAndCrossClassAllConcurrent: within one class, disjoint ranges run together;
// across classes, even identical ranges never contend. No collisions, all kept.
func TestComputeDisjointAndCrossClassAllConcurrent(t *testing.T) {
	r := Plan(Input{NowUnix: base, Candidates: []Candidate{
		{ID: "dev-lo", Key: "A", Compute: &ComputeClaim{Class: "device", Range: "0-3"}, UpdatedUnix: base - 10},
		{ID: "dev-hi", Key: "B", Compute: &ComputeClaim{Class: "device", Range: "4-7"}, UpdatedUnix: base - 20},
		{ID: "gpu-lo", Key: "C", Compute: &ComputeClaim{Class: "gpu", Range: "0-3"}, UpdatedUnix: base - 30},
	}})
	if r.KeepCount != 3 || r.CollisionCount != 0 || len(r.Collisions) != 0 {
		t.Fatalf("disjoint+cross-class = keep %d collision %d edges %d, want 3/0/0",
			r.KeepCount, r.CollisionCount, len(r.Collisions))
	}
}

// TestComputeSharedSharedOverlapAllowed: the compute lock discipline mirrors the file one — two
// shared claims on the SAME overlapping region may run concurrently.
func TestComputeSharedSharedOverlapAllowed(t *testing.T) {
	r := Plan(Input{NowUnix: base, Candidates: []Candidate{
		{ID: "reader-a", Key: "A", Compute: &ComputeClaim{Class: "device", Range: "0-3", Mode: "shared"}, UpdatedUnix: base - 10},
		{ID: "reader-b", Key: "B", Compute: &ComputeClaim{Class: "device", Range: "0-3", Mode: "shared"}, UpdatedUnix: base - 20},
	}})
	if r.KeepCount != 2 || r.CollisionCount != 0 || len(r.Collisions) != 0 {
		t.Fatalf("shared/shared compute = keep %d collision %d edges %d, want 2/0/0",
			r.KeepCount, r.CollisionCount, len(r.Collisions))
	}
}

// TestComputeUnknownRangeCollidesConservatively: an empty Range is unknown blast radius that
// collides with any same-class claim — the compute twin of an empty file tree — and the serialized
// unknown-range candidate is advised to declare a region before launch.
func TestComputeUnknownRangeCollidesConservatively(t *testing.T) {
	r := Plan(Input{NowUnix: base, Candidates: []Candidate{
		{ID: "known-fresh", Key: "A", Compute: &ComputeClaim{Class: "device", Range: "0"}, UpdatedUnix: base - 10},
		{ID: "unknown-old", Key: "B", Compute: &ComputeClaim{Class: "device"}, UpdatedUnix: base - 30},
	}})
	if r.KeepCount != 1 || r.CollisionCount != 1 {
		t.Fatalf("unknown-range = keep %d collision %d, want 1/1", r.KeepCount, r.CollisionCount)
	}
	if dispoOf(r, "known-fresh") != DispKeep || dispoOf(r, "unknown-old") != DispCollisionRisk {
		t.Fatalf("dispositions known-fresh=%q unknown-old=%q, want keep/collision_risk",
			dispoOf(r, "known-fresh"), dispoOf(r, "unknown-old"))
	}
	if len(r.Repartition) != 1 || r.Repartition[0].Action != "declare_compute_range" {
		t.Fatalf("advice = %+v, want one declare_compute_range row", r.Repartition)
	}
	if len(r.Repartition[0].CurrentRegion) != 0 {
		t.Fatalf("unknown-range advice CurrentRegion = %#v, want empty", r.Repartition[0].CurrentRegion)
	}
}

// TestComputeAndFileAxesAreIndependent is the load-bearing #3268 guarantee: a file-only worker and
// a compute-only worker never contend, because neither participates on the other's axis. A file
// claim and a compute claim address disjoint resource spaces.
func TestComputeAndFileAxesAreIndependent(t *testing.T) {
	r := Plan(Input{NowUnix: base, Candidates: []Candidate{
		{ID: "filer", Key: "A", Lane: "gateway", Tree: []string{"internal/gateway/**"}, UpdatedUnix: base - 10},
		{ID: "computer", Key: "B", Compute: &ComputeClaim{Class: "device", Range: "0-3"}, UpdatedUnix: base - 20},
	}})
	if r.KeepCount != 2 || r.CollisionCount != 0 || len(r.Collisions) != 0 {
		t.Fatalf("file vs compute = keep %d collision %d edges %d, want 2/0/0 (independent axes)",
			r.KeepCount, r.CollisionCount, len(r.Collisions))
	}
}

// TestComputeBothAxesSerializeIndependently: a candidate may carry BOTH a file tree and a compute
// region. Two such candidates that overlap on EITHER axis serialize — here they are file-disjoint
// but compute-overlapping, so the collision is priced on the compute plane.
func TestComputeBothAxesSerializeIndependently(t *testing.T) {
	r := Plan(Input{NowUnix: base, Candidates: []Candidate{
		{ID: "w-fresh", Key: "A", Tree: []string{"internal/a/**"}, Compute: &ComputeClaim{Class: "device", Range: "0-3"}, UpdatedUnix: base - 10},
		{ID: "w-old", Key: "B", Tree: []string{"internal/b/**"}, Compute: &ComputeClaim{Class: "device", Range: "1-2"}, UpdatedUnix: base - 20},
	}})
	if r.KeepCount != 1 || r.CollisionCount != 1 {
		t.Fatalf("both-axes = keep %d collision %d, want 1/1", r.KeepCount, r.CollisionCount)
	}
	if dispoOf(r, "w-fresh") != DispKeep || dispoOf(r, "w-old") != DispCollisionRisk {
		t.Fatalf("dispositions w-fresh=%q w-old=%q, want keep/collision_risk",
			dispoOf(r, "w-fresh"), dispoOf(r, "w-old"))
	}
	if len(r.Collisions) != 1 || len(r.Collisions[0].Region) != 2 {
		t.Fatalf("collision = %+v, want one compute-region edge", r.Collisions)
	}
}

// TestRangesOverlapGrammar pins the accepted Range shapes and the conservative fallback.
func TestRangesOverlapGrammar(t *testing.T) {
	tests := []struct {
		a, b string
		want bool
	}{
		{"0-3", "2-5", true},     // inclusive intervals overlap
		{"0-3", "4-7", false},    // adjacent, disjoint
		{"4..7", "5", true},      // dotted interval contains a point
		{"4+4", "7", true},       // start+len span [4,7] contains 7
		{"4+4", "8", false},      // [4,7] excludes 8
		{"[4,4)", "4", true},     // KV half-open span [4,7] contains 4
		{"0,2,4", "4-9", true},   // union member 4 hits [4,9]
		{"0,2,4", "5-9", false},  // union misses [5,9]
		{"0-3", "0-3", true},     // identical
		{"", "0-3", true},        // empty is unknown blast radius: conservative collide
		{"garbage", "0-3", true}, // unparseable is conservative collide
	}
	for _, tt := range tests {
		if got := rangesOverlap(tt.a, tt.b); got != tt.want {
			t.Errorf("rangesOverlap(%q, %q) = %v, want %v", tt.a, tt.b, got, tt.want)
		}
		if got := rangesOverlap(tt.b, tt.a); got != tt.want {
			t.Errorf("rangesOverlap(%q, %q) [swapped] = %v, want %v", tt.b, tt.a, got, tt.want)
		}
	}
}

func TestPlanFinishesAttemptedWorkBeforeOpeningFreshWork(t *testing.T) {
	in := Input{
		NowUnix:         10_000,
		CooldownSeconds: 100,
		FinishFirst:     true,
		Candidates: []Candidate{
			{ID: "fresh", Key: "fresh", UpdatedUnix: 9_900, Priority: 1},
			{ID: "wip", Key: "wip", UpdatedUnix: 1_000, LastAttemptUnix: 9_000, Priority: 1},
		},
	}

	got := Plan(in)
	if pick := got.Pick(); pick != "wip" {
		t.Fatalf("pick = %q, want attempted WIP before a newer unstarted issue; order=%+v", pick, got.Order)
	}
	if got.KeepCount != 2 {
		t.Fatalf("keep_count = %d, want both safe candidates retained", got.KeepCount)
	}
}

func TestPlanFinishFirstPreservesPriorityAndCooldown(t *testing.T) {
	in := Input{
		NowUnix:         10_000,
		CooldownSeconds: 100,
		FinishFirst:     true,
		Candidates: []Candidate{
			{ID: "urgent-fresh", Key: "urgent-fresh", UpdatedUnix: 9_900, Priority: 2},
			{ID: "wip", Key: "wip", UpdatedUnix: 1_000, LastAttemptUnix: 9_000, Priority: 1},
			{ID: "cooling", Key: "cooling", UpdatedUnix: 9_950, LastAttemptUnix: 9_950, Priority: 3},
		},
	}

	got := Plan(in)
	if pick := got.Pick(); pick != "urgent-fresh" {
		t.Fatalf("pick = %q, want explicit higher priority to lead finish-first; order=%+v", pick, got.Order)
	}
	for _, row := range got.Order {
		if row.ID == "cooling" && row.Disposition != DispCooling {
			t.Fatalf("cooling disposition = %q, want %q", row.Disposition, DispCooling)
		}
	}
}
