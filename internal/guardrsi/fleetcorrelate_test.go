package guardrsi

import (
	"encoding/json"
	"reflect"
	"testing"
)

// TestCorrelateThreeDistinctTracesPromoteOneCandidate is the first Done-condition
// case (#2714): 3 distinct traces emitting the same FailureHash within the window
// return exactly one candidate, carrying the reason class and the union of the
// three declared trees as the signature tree.
func TestCorrelateThreeDistinctTracesPromoteOneCandidate(t *testing.T) {
	const now = 10_000
	const hash = "sha256:deadbeef"
	obs := []FleetObservation{
		{TraceID: "trace-a", FailureHash: hash, Reason: "BUILD", TreeGlobs: []string{"internal/foo/**"}, TS: now - 10},
		{TraceID: "trace-b", FailureHash: hash, Reason: "BUILD", TreeGlobs: []string{"internal/foo/bar.go"}, TS: now - 5},
		{TraceID: "trace-c", FailureHash: hash, Reason: "BUILD", TreeGlobs: []string{"internal/baz/**"}, TS: now - 1},
	}
	got := Correlate(obs, DefaultCorrelateK, DefaultCorrelateWindowSecs, now)
	if len(got) != 1 {
		t.Fatalf("want exactly 1 candidate, got %d: %+v", len(got), got)
	}
	c := got[0]
	if c.FailureHash != hash {
		t.Errorf("FailureHash = %q, want %q", c.FailureHash, hash)
	}
	if c.DistinctTraces != 3 {
		t.Errorf("DistinctTraces = %d, want 3", c.DistinctTraces)
	}
	if c.ReasonClass != "BUILD" {
		t.Errorf("ReasonClass = %q, want BUILD", c.ReasonClass)
	}
	// Signature tree is the deduped, sorted union of every correlated trace's trees.
	wantTrees := []string{"internal/baz/**", "internal/foo/**", "internal/foo/bar.go"}
	if !reflect.DeepEqual(c.TreeGlobs, wantTrees) {
		t.Errorf("TreeGlobs = %v, want %v", c.TreeGlobs, wantTrees)
	}
	wantTraces := []string{"trace-a", "trace-b", "trace-c"}
	if !reflect.DeepEqual(c.TraceIDs, wantTraces) {
		t.Errorf("TraceIDs = %v, want %v", c.TraceIDs, wantTraces)
	}
}

// TestCorrelateThreeRepeatsOneTraceNoCandidate is the second Done-condition case:
// 3 emissions from a SINGLE trace are a per-trace livelock, not a fleet-wide shared
// cause, so Correlate returns nothing (distinct-trace count is 1, below K).
func TestCorrelateThreeRepeatsOneTraceNoCandidate(t *testing.T) {
	const now = 10_000
	const hash = "sha256:deadbeef"
	obs := []FleetObservation{
		{TraceID: "trace-a", FailureHash: hash, Reason: "BUILD", TreeGlobs: []string{"internal/foo/**"}, TS: now - 9},
		{TraceID: "trace-a", FailureHash: hash, Reason: "BUILD", TreeGlobs: []string{"internal/foo/**"}, TS: now - 6},
		{TraceID: "trace-a", FailureHash: hash, Reason: "BUILD", TreeGlobs: []string{"internal/foo/**"}, TS: now - 3},
	}
	got := Correlate(obs, DefaultCorrelateK, DefaultCorrelateWindowSecs, now)
	if len(got) != 0 {
		t.Fatalf("want 0 candidates for 3 repeats from one trace, got %d: %+v", len(got), got)
	}
}

// TestCorrelateObservationsOlderThanWindowDoNotCount is the third Done-condition
// case: an observation older than the window does not count toward K. Two in-window
// traces plus a stale third stays below K; sliding that third trace inside the
// window promotes the candidate.
func TestCorrelateObservationsOlderThanWindowDoNotCount(t *testing.T) {
	const now = 10_000
	const window = 100
	const hash = "sha256:deadbeef"
	obs := []FleetObservation{
		{TraceID: "trace-a", FailureHash: hash, Reason: "BUILD", TreeGlobs: []string{"internal/foo/**"}, TS: now - 10},
		{TraceID: "trace-b", FailureHash: hash, Reason: "BUILD", TreeGlobs: []string{"internal/foo/**"}, TS: now - 20},
		// Third distinct trace, but its only observation is older than the window.
		{TraceID: "trace-c", FailureHash: hash, Reason: "BUILD", TreeGlobs: []string{"internal/foo/**"}, TS: now - window - 1},
	}
	if got := Correlate(obs, DefaultCorrelateK, window, now); len(got) != 0 {
		t.Fatalf("stale third trace must not reach K=3; got %d candidate(s): %+v", len(got), got)
	}
	// Bring the same third trace inside the window and it promotes to one candidate.
	obs[2].TS = now - window + 1
	got := Correlate(obs, DefaultCorrelateK, window, now)
	if len(got) != 1 || got[0].DistinctTraces != 3 {
		t.Fatalf("in-window third trace should promote to 1 candidate with 3 distinct traces, got %+v", got)
	}
}

// TestCorrelateCandidateJSONWitness captures a full candidate as JSON — the witness
// artifact the issue (#2714) asks the commit body to cite. It pins the schema/event
// tags, the field order, and the derived union/first/last so a downstream reader of
// the record can rely on the exact wire shape.
func TestCorrelateCandidateJSONWitness(t *testing.T) {
	const now = 1_000_000
	const hash = "sha256:aad21e11"
	obs := []FleetObservation{
		{TraceID: "t1", FailureHash: hash, Reason: "BUILD", TreeGlobs: []string{"internal/foo/**"}, TS: now - 30},
		{TraceID: "t2", FailureHash: hash, Reason: "BUILD", TreeGlobs: []string{"internal/foo/x.go"}, TS: now - 20},
		{TraceID: "t3", FailureHash: hash, Reason: "BUILD", TreeGlobs: []string{"internal/foo/**"}, TS: now - 10},
	}
	got := Correlate(obs, 3, 900, now)
	if len(got) != 1 {
		t.Fatalf("want 1 candidate, got %d: %+v", len(got), got)
	}
	b, err := json.Marshal(got[0])
	if err != nil {
		t.Fatalf("marshal candidate: %v", err)
	}
	const want = `{"schema":"guardrsi.fleet-correlate/1","event":"FLEET_KNOWN_BAD_CANDIDATE","failure_hash":"sha256:aad21e11","reason_class":"BUILD","tree_globs":["internal/foo/**","internal/foo/x.go"],"trace_ids":["t1","t2","t3"],"distinct_traces":3,"window_secs":900,"first_seen_unix":999970,"last_seen_unix":999990}`
	if string(b) != want {
		t.Fatalf("candidate JSON witness mismatch:\n got: %s\nwant: %s", b, want)
	}
}

// TestCorrelateBelowThresholdNoCandidate guards the boundary: two distinct traces
// (K-1) never promote, so a merely-shared-by-two failure is not over-reported.
func TestCorrelateBelowThresholdNoCandidate(t *testing.T) {
	const now = 10_000
	const hash = "sha256:deadbeef"
	obs := []FleetObservation{
		{TraceID: "trace-a", FailureHash: hash, Reason: "BUILD", TreeGlobs: []string{"internal/foo/**"}, TS: now - 8},
		{TraceID: "trace-b", FailureHash: hash, Reason: "BUILD", TreeGlobs: []string{"internal/foo/**"}, TS: now - 4},
	}
	if got := Correlate(obs, DefaultCorrelateK, DefaultCorrelateWindowSecs, now); len(got) != 0 {
		t.Fatalf("two distinct traces (K-1) must not promote, got %d: %+v", len(got), got)
	}
}

// TestCorrelateDistinctHashesSortedAndIndependent confirms two independent shared
// causes each promote on their own K and are returned deterministically sorted by
// FailureHash.
func TestCorrelateDistinctHashesSortedAndIndependent(t *testing.T) {
	const now = 10_000
	mk := func(trace, hash string, ts int64) FleetObservation {
		return FleetObservation{TraceID: trace, FailureHash: hash, Reason: "BUILD", TreeGlobs: []string{"internal/foo/**"}, TS: ts}
	}
	obs := []FleetObservation{
		mk("a", "sha256:bbb", now-9), mk("b", "sha256:bbb", now-8), mk("c", "sha256:bbb", now-7),
		mk("a", "sha256:aaa", now-6), mk("b", "sha256:aaa", now-5), mk("c", "sha256:aaa", now-4),
	}
	got := Correlate(obs, DefaultCorrelateK, DefaultCorrelateWindowSecs, now)
	if len(got) != 2 {
		t.Fatalf("want 2 candidates (one per shared hash), got %d: %+v", len(got), got)
	}
	if got[0].FailureHash != "sha256:aaa" || got[1].FailureHash != "sha256:bbb" {
		t.Errorf("candidates not sorted by FailureHash: %q, %q", got[0].FailureHash, got[1].FailureHash)
	}
}

// TestCorrelateSkipsAnonymousAndHashless confirms observations lacking a TraceID or
// FailureHash are dropped (they cannot be correlated) rather than counted.
func TestCorrelateSkipsAnonymousAndHashless(t *testing.T) {
	const now = 10_000
	const hash = "sha256:deadbeef"
	obs := []FleetObservation{
		{TraceID: "trace-a", FailureHash: hash, Reason: "BUILD", TreeGlobs: []string{"internal/foo/**"}, TS: now - 9},
		{TraceID: "", FailureHash: hash, Reason: "BUILD", TreeGlobs: []string{"internal/foo/**"}, TS: now - 8},
		{TraceID: "trace-b", FailureHash: "", Reason: "BUILD", TreeGlobs: []string{"internal/foo/**"}, TS: now - 7},
		{TraceID: "trace-c", FailureHash: hash, Reason: "BUILD", TreeGlobs: []string{"internal/foo/**"}, TS: now - 6},
	}
	// Only trace-a and trace-c are valid & share the hash: 2 distinct < K, so none.
	if got := Correlate(obs, DefaultCorrelateK, DefaultCorrelateWindowSecs, now); len(got) != 0 {
		t.Fatalf("anonymous/hash-less observations must not count toward K, got %d: %+v", len(got), got)
	}
}
