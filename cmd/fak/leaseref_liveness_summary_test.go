package main

// leaseref_liveness_summary_test.go is the CLI-side acceptance for #5485: the aggregate
// (internal/leaseref.SummarizeLiveness) is actually REACHABLE from `fak leaseref liveness`,
// and reaching it does not break the array contract the verb already publishes.
//
// The defect these tests close is not a wrong number — every row `liveness` emits was
// already correct. It is that a fleet in which NOTHING publishes the liveness input yields
// a complete, well-formed array of correctly-computed rows, every one of them an ABSENCE of
// evidence, and nothing in the output separates that (a wiring defect in the observer, which
// makes every verdict downstream uninformative) from a healthy read. So the load-bearing
// assertion is a COMPARISON: two live sets of the same size, one classified on published
// descriptors and one with nothing published at all, must be told apart from the output.

import (
	"bytes"
	"context"
	"encoding/json"
	"os/exec"
	"strings"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/leaseref"
)

// leaserefTempRepo makes a real git repo for the liveness verb to read, skipping when git
// is unavailable (the same guard the sibling end-to-end tests use).
func leaserefTempRepo(t *testing.T) string {
	t.Helper()
	if _, err := exec.LookPath("git"); err != nil {
		t.Skip("git not on PATH")
	}
	dir := t.TempDir()
	for _, args := range [][]string{
		{"init", "-q"},
		{"config", "user.email", "t@example.com"},
		{"config", "user.name", "t"},
	} {
		c := exec.Command("git", args...)
		c.Dir = dir
		if out, err := c.CombinedOutput(); err != nil {
			t.Fatalf("git %v: %v\n%s", args, err, out)
		}
	}
	return dir
}

func leaserefSeed(t *testing.T, dir string, recs []leaseref.Record, descs []leaseref.SessionDescriptor) {
	t.Helper()
	store := leaseref.NewInDir(dir)
	ctx := context.Background()
	for _, r := range recs {
		if _, err := store.Acquire(ctx, r); err != nil {
			t.Fatalf("Acquire %s: %v", r.ID, err)
		}
	}
	for _, d := range descs {
		if _, err := store.PublishSession(ctx, d); err != nil {
			t.Fatalf("PublishSession %s: %v", d.ID, err)
		}
	}
}

// leaserefLivenessReportOut is the decoded --summary envelope. It is decoded into map/any
// rather than leaserefLivenessReport so the test asserts the JSON KEYS an external consumer
// actually sees (`liveness_coverage` and friends), not just that a Go struct round-trips.
type leaserefLivenessReportOut struct {
	Schema  string         `json:"schema"`
	Summary map[string]any `json:"summary"`
	Leases  []struct {
		ID           string `json:"id"`
		Liveness     string `json:"liveness"`
		EvidenceKind string `json:"evidence_kind"`
	} `json:"leases"`
}

func leaserefRunLiveness(t *testing.T, argv ...string) (stdout, stderr string, code int) {
	t.Helper()
	var out, errb bytes.Buffer
	code = runLeaseref(&out, &errb, append([]string{"liveness"}, argv...))
	return out.String(), errb.String(), code
}

// TestLeaserefLivenessSummaryEmitsCoverage is the direct wiring witness: --summary puts the
// aggregate on stdout under the keys internal/leaseref declares, INCLUDING liveness_coverage
// — the field whose absence made the producer's fix inert at the CLI.
func TestLeaserefLivenessSummaryEmitsCoverage(t *testing.T) {
	dir := leaserefTempRepo(t)
	now := time.Now().Unix()
	leaserefSeed(t, dir,
		[]leaseref.Record{
			{ID: "lane-live", TreeGlobs: []string{"a/**"}, Holder: "A", SessionID: "sess-live", AcquiredAt: now, TTLSeconds: 3600},
			{ID: "lane-unbound", TreeGlobs: []string{"b/**"}, Holder: "B", AcquiredAt: now, TTLSeconds: 3600},
		},
		[]leaseref.SessionDescriptor{
			{ID: "sess-live", Host: "h1", PCBState: "RUNNING", UpdatedAt: now, TTLSecs: 1800},
		})

	out, errb, code := leaserefRunLiveness(t, "--summary", "--dir", dir)
	if code != 0 {
		t.Fatalf("leaseref liveness --summary exit=%d stderr=%q", code, errb)
	}

	// The raw text must carry the key, not merely a Go field that happens to marshal.
	if !strings.Contains(out, `"liveness_coverage"`) {
		t.Fatalf("--summary output does not carry the liveness_coverage key:\n%s", out)
	}
	var rep leaserefLivenessReportOut
	if err := json.Unmarshal([]byte(out), &rep); err != nil {
		t.Fatalf("--summary JSON unmarshal: %v\nout=%s", err, out)
	}
	if rep.Schema != leaserefLivenessSchema {
		t.Fatalf("schema = %q, want %q", rep.Schema, leaserefLivenessSchema)
	}
	if len(rep.Leases) != 2 {
		t.Fatalf("leases = %d rows, want 2: %+v", len(rep.Leases), rep.Leases)
	}
	// Every summary key the aggregate declares must be present — a partial envelope is the
	// same silent-hole failure in miniature.
	for _, k := range []string{"total", "by_class", "by_evidence_kind", "positive_evidence", "liveness_coverage"} {
		if _, ok := rep.Summary[k]; !ok {
			t.Fatalf("summary is missing key %q: %v", k, rep.Summary)
		}
	}
	if got := rep.Summary["total"]; got != float64(2) {
		t.Fatalf("summary.total = %v, want 2", got)
	}
	// One row rests on a read descriptor, one on an absence: 1/2.
	if got := rep.Summary["liveness_coverage"]; got != float64(0.5) {
		t.Fatalf("summary.liveness_coverage = %v, want 0.5", got)
	}
	if got := rep.Summary["positive_evidence"]; got != float64(1) {
		t.Fatalf("summary.positive_evidence = %v, want 1", got)
	}
	// The histograms are zero-filled over their closed vocabularies, so a reader never has
	// to disambiguate "key absent" from "count zero".
	byKind, ok := rep.Summary["by_evidence_kind"].(map[string]any)
	if !ok {
		t.Fatalf("summary.by_evidence_kind is not an object: %v", rep.Summary["by_evidence_kind"])
	}
	for _, k := range []string{
		leaseref.EvidenceNoBinding, leaseref.EvidenceSelfSession, leaseref.EvidenceNoDescriptor,
		leaseref.EvidenceTerminalStopped, leaseref.EvidenceHeartbeatLapsed, leaseref.EvidenceHeartbeating,
	} {
		if _, ok := byKind[k]; !ok {
			t.Fatalf("by_evidence_kind is missing the zero-filled key %q: %v", k, byKind)
		}
	}
	if byKind[leaseref.EvidenceHeartbeating] != float64(1) || byKind[leaseref.EvidenceNoBinding] != float64(1) {
		t.Fatalf("by_evidence_kind = %v, want heartbeating=1 no-session-binding=1", byKind)
	}
}

// TestLeaserefLivenessUnwiredFleetIsDistinguishable is the motivating case, stated as the
// comparison it actually is. Two live sets of the SAME SIZE:
//
//	unwired — nothing published: no acquirer bound a session, and the one that did has no
//	          descriptor. Every row peer-unknown, every row an absence of evidence.
//	wired   — descriptors published: rows classified on inputs that were read.
//
// Before this wiring both emitted a complete, well-formed array and the CLI said nothing
// that separated them as READS. The assertions below are what now separates them, on BOTH
// output channels.
func TestLeaserefLivenessUnwiredFleetIsDistinguishable(t *testing.T) {
	now := time.Now().Unix()

	unwiredDir := leaserefTempRepo(t)
	leaserefSeed(t, unwiredDir,
		[]leaseref.Record{
			// No session_id at all: the ACQUIRER never bound one.
			{ID: "lane-u1", TreeGlobs: []string{"a/**"}, Holder: "A", AcquiredAt: now, TTLSeconds: 3600},
			{ID: "lane-u2", TreeGlobs: []string{"b/**"}, Holder: "B", AcquiredAt: now, TTLSeconds: 3600},
			// Bound, but nothing ever published the descriptor: the PUBLISHER is down.
			{ID: "lane-u3", TreeGlobs: []string{"c/**"}, Holder: "C", SessionID: "sess-ghost", AcquiredAt: now, TTLSeconds: 3600},
		},
		nil)

	wiredDir := leaserefTempRepo(t)
	leaserefSeed(t, wiredDir,
		[]leaseref.Record{
			{ID: "lane-w1", TreeGlobs: []string{"a/**"}, Holder: "A", SessionID: "sess-hb", AcquiredAt: now, TTLSeconds: 3600},
			{ID: "lane-w2", TreeGlobs: []string{"b/**"}, Holder: "B", SessionID: "sess-lapsed", AcquiredAt: now, TTLSeconds: 3600},
			{ID: "lane-w3", TreeGlobs: []string{"c/**"}, Holder: "C", SessionID: "sess-stopped", AcquiredAt: now, TTLSeconds: 3600},
		},
		[]leaseref.SessionDescriptor{
			{ID: "sess-hb", Host: "h1", PCBState: "RUNNING", UpdatedAt: now, TTLSecs: 1800},
			{ID: "sess-lapsed", Host: "h2", PCBState: "RUNNING", UpdatedAt: now - 7200, TTLSecs: 60},
			{ID: "sess-stopped", Host: "h3", PCBState: "STOPPED", UpdatedAt: now, TTLSecs: 1800},
		})

	// --- The premise: as BARE ARRAYS both reads are complete and well-formed. ---
	for name, dir := range map[string]string{"unwired": unwiredDir, "wired": wiredDir} {
		out, errb, code := leaserefRunLiveness(t, "--dir", dir)
		if code != 0 {
			t.Fatalf("%s: liveness exit=%d stderr=%q", name, code, errb)
		}
		var rows []leaseref.ClassifiedLease
		if err := json.Unmarshal([]byte(out), &rows); err != nil {
			t.Fatalf("%s: array unmarshal: %v\nout=%s", name, err, out)
		}
		if len(rows) != 3 {
			t.Fatalf("%s: %d rows, want 3 — the premise (a COMPLETE array either way) does not hold", name, len(rows))
		}
		for _, r := range rows {
			if r.Liveness == "" || r.Evidence == "" || r.EvidenceKind == "" {
				t.Fatalf("%s: row %s is not fully classified: %+v", name, r.ID, r)
			}
		}
	}

	// --- Channel 1 (--summary): liveness_coverage separates them. ---
	unwiredSum := leaserefSummaryOf(t, unwiredDir)
	wiredSum := leaserefSummaryOf(t, wiredDir)
	if unwiredSum["liveness_coverage"] != float64(0) {
		t.Fatalf("unwired liveness_coverage = %v, want 0.0", unwiredSum["liveness_coverage"])
	}
	if wiredSum["liveness_coverage"] != float64(1) {
		t.Fatalf("wired liveness_coverage = %v, want 1.0", wiredSum["liveness_coverage"])
	}
	if unwiredSum["liveness_coverage"] == wiredSum["liveness_coverage"] {
		t.Fatalf("coverage does not distinguish an unwired feed from a classified one: both %v",
			unwiredSum["liveness_coverage"])
	}
	if unwiredSum["total"] != wiredSum["total"] {
		t.Fatalf("the comparison is not like-for-like: totals %v vs %v", unwiredSum["total"], wiredSum["total"])
	}
	// by_evidence_kind names WHICH remedy the unwired fleet needs — two acquirers that never
	// bound, one publisher that never published. Different owners, so the distinction must
	// survive into the output, not collapse into a single "unknown" count.
	unwiredKinds := unwiredSum["by_evidence_kind"].(map[string]any)
	if unwiredKinds[leaseref.EvidenceNoBinding] != float64(2) {
		t.Fatalf("unwired by_evidence_kind[%s] = %v, want 2", leaseref.EvidenceNoBinding, unwiredKinds[leaseref.EvidenceNoBinding])
	}
	if unwiredKinds[leaseref.EvidenceNoDescriptor] != float64(1) {
		t.Fatalf("unwired by_evidence_kind[%s] = %v, want 1", leaseref.EvidenceNoDescriptor, unwiredKinds[leaseref.EvidenceNoDescriptor])
	}

	// --- Channel 2 (the DEFAULT array run): the stderr banner separates them too, so the
	// signal is not missable by a caller that never learns --summary exists. ---
	_, unwiredErr, _ := leaserefRunLiveness(t, "--dir", unwiredDir)
	_, wiredErr, _ := leaserefRunLiveness(t, "--dir", wiredDir)
	if !strings.Contains(unwiredErr, "liveness_coverage=0.00") {
		t.Fatalf("unwired stderr banner does not carry coverage 0.00: %q", unwiredErr)
	}
	if !strings.Contains(unwiredErr, "WIRING DEFECT IN THIS OBSERVER") {
		t.Fatalf("unwired stderr banner does not name the wiring defect: %q", unwiredErr)
	}
	if !strings.Contains(unwiredErr, leaseref.EvidenceNoBinding) || !strings.Contains(unwiredErr, leaseref.EvidenceNoDescriptor) {
		t.Fatalf("unwired stderr banner does not name both remedies: %q", unwiredErr)
	}
	if !strings.Contains(wiredErr, "liveness_coverage=1.00") {
		t.Fatalf("wired stderr banner does not carry coverage 1.00: %q", wiredErr)
	}
	if strings.Contains(wiredErr, "WIRING DEFECT") {
		t.Fatalf("wired stderr banner falsely warns of a wiring defect: %q", wiredErr)
	}
}

func leaserefSummaryOf(t *testing.T, dir string) map[string]any {
	t.Helper()
	out, errb, code := leaserefRunLiveness(t, "--summary", "--dir", dir)
	if code != 0 {
		t.Fatalf("liveness --summary exit=%d stderr=%q", code, errb)
	}
	var rep leaserefLivenessReportOut
	if err := json.Unmarshal([]byte(out), &rep); err != nil {
		t.Fatalf("--summary unmarshal: %v\nout=%s", err, out)
	}
	return rep.Summary
}

// TestLeaserefLivenessDefaultStaysABareArray is the COMPATIBILITY guard on the shape choice.
// tools/issue_dispatch.py's lease_ref_busy_lanes runs this exact verb with no flags,
// json.loads its stdout, and on any non-list returns an EMPTY busy-lane set with
// "unexpected leaseref liveness shape" — which fails OPEN, letting a wave plan onto lanes a
// live peer holds. Promoting the default to a top-level object would do that silently, so
// the default staying a JSON ARRAY is an assertion, not an implementation detail.
func TestLeaserefLivenessDefaultStaysABareArray(t *testing.T) {
	dir := leaserefTempRepo(t)
	now := time.Now().Unix()
	leaserefSeed(t, dir,
		[]leaseref.Record{
			{ID: "resolve-tools", TreeGlobs: []string{"tools/**"}, Holder: "node/sess", SessionID: "sess-hb", AcquiredAt: now, TTLSeconds: 3600},
		},
		[]leaseref.SessionDescriptor{
			{ID: "sess-hb", Host: "h1", PCBState: "RUNNING", UpdatedAt: now, TTLSecs: 1800},
		})

	out, errb, code := leaserefRunLiveness(t, "--dir", dir)
	if code != 0 {
		t.Fatalf("liveness exit=%d stderr=%q", code, errb)
	}
	if trimmed := strings.TrimSpace(out); !strings.HasPrefix(trimmed, "[") {
		t.Fatalf("default stdout is no longer a top-level JSON array (issue_dispatch.py fails OPEN on that): %s", out)
	}
	// Decode the way the Python gate does: a list of objects it reads `id`/`reclaimable` off.
	var generic []map[string]any
	if err := json.Unmarshal([]byte(out), &generic); err != nil {
		t.Fatalf("default stdout does not decode as a list of objects: %v\nout=%s", err, out)
	}
	if len(generic) != 1 {
		t.Fatalf("rows = %d, want 1", len(generic))
	}
	for _, k := range []string{"id", "holder", "session_id", "liveness", "reclaimable", "evidence"} {
		if _, ok := generic[0][k]; !ok {
			t.Fatalf("row lost the pre-existing key %q: %v", k, generic[0])
		}
	}
	// The new per-row field rides along additively.
	if generic[0]["evidence_kind"] != leaseref.EvidenceHeartbeating {
		t.Fatalf("row evidence_kind = %v, want %q", generic[0]["evidence_kind"], leaseref.EvidenceHeartbeating)
	}
	// The aggregate must NOT have leaked onto stdout on the default path.
	if strings.Contains(out, "liveness_coverage") {
		t.Fatalf("default stdout carries the summary — that is the breaking shape: %s", out)
	}
	if !strings.Contains(errb, "liveness_coverage=1.00") {
		t.Fatalf("default run did not put coverage on stderr: %q", errb)
	}
}

// TestLeaserefCoverageBannerEmptyLiveSet pins the one reading the producer explicitly warns
// about: an EMPTY live set also reports coverage 0.0, and that zero is an undefined ratio,
// not the wiring alarm. A banner that cried defect here would train operators to ignore it.
func TestLeaserefCoverageBannerEmptyLiveSet(t *testing.T) {
	banner := leaserefCoverageBanner(leaseref.SummarizeLiveness(nil))
	if strings.Contains(banner, "WIRING DEFECT") {
		t.Fatalf("empty live set falsely warns of a wiring defect: %q", banner)
	}
	if !strings.Contains(banner, "undefined") || !strings.Contains(banner, "0 live lease(s)") {
		t.Fatalf("empty-set banner does not say the ratio is undefined over zero rows: %q", banner)
	}

	// And the alarm DOES fire once there is a real live set with no evidence behind it.
	unwired := leaseref.SummarizeLiveness([]leaseref.ClassifiedLease{
		{Liveness: leaseref.LivenessPeerUnknown, EvidenceKind: leaseref.EvidenceNoBinding},
		{Liveness: leaseref.LivenessPeerUnknown, EvidenceKind: leaseref.EvidenceNoDescriptor},
	})
	alarm := leaserefCoverageBanner(unwired)
	if !strings.Contains(alarm, "WIRING DEFECT IN THIS OBSERVER") {
		t.Fatalf("unwired non-empty set does not raise the alarm: %q", alarm)
	}
	if !strings.Contains(alarm, "liveness_coverage=0.00") {
		t.Fatalf("alarm banner lacks the coverage number: %q", alarm)
	}
}
