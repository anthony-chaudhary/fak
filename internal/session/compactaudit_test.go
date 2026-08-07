package session

// compactaudit_test.go — failure-class proof for the compaction-health miner (#4763).
//
// Each test pins one class the audit exists to tell apart. The headline one is
// TestCompactAuditLongAppendOnlyLogReadsBounded / ...HealthyRepeatedFire: they encode
// the actual reported defect — a byte-large, cumulatively-huge session being read as
// "not compacting" when resident context was repeatedly reset.

import (
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

func scanFixture(t *testing.T, name string) CompactSessionReport {
	t.Helper()
	p := filepath.Join("testdata", "compactaudit", name)
	f, err := os.Open(p)
	if err != nil {
		t.Fatalf("open fixture: %v", err)
	}
	defer f.Close()
	fi, err := f.Stat()
	if err != nil {
		t.Fatalf("stat fixture: %v", err)
	}
	rep, err := ScanCompactRollout(f, p, fi.Size())
	if err != nil {
		t.Fatalf("scan %s: %v", name, err)
	}
	return rep
}

func hasAnomaly(list []string, want string) bool {
	for _, a := range list {
		if a == want {
			return true
		}
	}
	return false
}

// A healthy session fires repeatedly and holds. The compacted/context_compacted PAIR
// must count as ONE fire, and the zero-token row Codex writes between the pair must not
// be bound as the post-fire witness.
func TestCompactAuditHealthyRepeatedFireFiresAndHolds(t *testing.T) {
	rep := scanFixture(t, "healthy-repeated-fire.jsonl")

	if rep.FireCount != 2 {
		t.Fatalf("fire count = %d, want 2 (the compacted/context_compacted pair counts once)", rep.FireCount)
	}
	if rep.PairedEvents != 2 {
		t.Errorf("paired events = %d, want 2 (one deduped twin per fire)", rep.PairedEvents)
	}
	if rep.DuplicateFires != 0 {
		t.Errorf("duplicate fires = %d, want 0 (a pair is not a duplicate)", rep.DuplicateFires)
	}
	if rep.Verdict != VerdictFiredAndHeld {
		t.Errorf("verdict = %q (anomalies %v), want %q", rep.Verdict, rep.Anomalies, VerdictFiredAndHeld)
	}

	f0 := rep.Fires[0]
	if f0.PreTokens != 242067 || f0.PostTokens != 25039 {
		t.Errorf("fire0 pre/post = %d/%d, want 242067/25039 (the zero row between the pair must be skipped)",
			f0.PreTokens, f0.PostTokens)
	}
	if f0.Shed != 242067-25039 {
		t.Errorf("fire0 shed = %d, want %d", f0.Shed, 242067-25039)
	}
	if f0.Confidence != CompactConfidenceHigh || f0.Reason != CompactReasonOK {
		t.Errorf("fire0 confidence = %s/%s, want %s/%s", f0.Confidence, f0.Reason, CompactConfidenceHigh, CompactReasonOK)
	}
	if len(f0.Anomalies) != 0 {
		t.Errorf("fire0 anomalies = %v, want none", f0.Anomalies)
	}

	// The distinction the whole issue turns on: cumulative tokens dwarf the window
	// while resident context never exceeds it.
	if rep.CumulativeInputTokens <= rep.ContextWindow {
		t.Errorf("cumulative %d should exceed the window %d in this fixture", rep.CumulativeInputTokens, rep.ContextWindow)
	}
	if rep.PeakResidentTokens > rep.ContextWindow {
		t.Errorf("peak resident %d exceeds the window %d — resident must be bounded", rep.PeakResidentTokens, rep.ContextWindow)
	}
	if rep.ContextWindow != 258400 {
		t.Errorf("context window = %d, want 258400", rep.ContextWindow)
	}
}

// The reported defect, pinned: a long append-only rollout with a huge cumulative token
// count and NO fires is healthy when resident context stayed bounded. It must not read
// as a compaction failure.
func TestCompactAuditLongAppendOnlyLogReadsBounded(t *testing.T) {
	rep := scanFixture(t, "long-append-only-bounded.jsonl")

	if rep.FireCount != 0 {
		t.Fatalf("fire count = %d, want 0", rep.FireCount)
	}
	if rep.Verdict != VerdictNoFireBounded {
		t.Errorf("verdict = %q, want %q — a big append-only log with bounded resident context is not a failure",
			rep.Verdict, VerdictNoFireBounded)
	}
	if hasAnomaly(rep.Anomalies, AnomalyNoFireAboveCeiling) {
		t.Errorf("anomalies = %v, must not flag NO_FIRE_ABOVE_CEILING below the ceiling", rep.Anomalies)
	}
	if rep.CumulativeInputTokens < 3*rep.ContextWindow {
		t.Errorf("cumulative %d should be many windows' worth in this fixture", rep.CumulativeInputTokens)
	}
	if rep.PeakResidentTokens >= int(CompactCeilingApproachFraction*float64(rep.ContextWindow)) {
		t.Errorf("peak resident %d should stay well under the ceiling", rep.PeakResidentTokens)
	}
	if rep.ToolCalls == 0 || rep.Turns == 0 {
		t.Errorf("turns/tool calls = %d/%d, want both > 0", rep.Turns, rep.ToolCalls)
	}
}

// Two events of the SAME kind inside the pair window are a genuine duplicate — unlike
// the compacted/context_compacted twin, which is the expected pair.
func TestCompactAuditDuplicateFireEventFlagged(t *testing.T) {
	rep := scanFixture(t, "paired-duplicate.jsonl")

	if rep.FireCount != 1 {
		t.Fatalf("fire count = %d, want 1 (the repeat is a duplicate, not a second fire)", rep.FireCount)
	}
	if rep.DuplicateFires != 1 {
		t.Errorf("duplicate fires = %d, want 1", rep.DuplicateFires)
	}
	if !hasAnomaly(rep.Fires[0].Anomalies, AnomalyDuplicateFireEvent) {
		t.Errorf("fire0 anomalies = %v, want %s", rep.Fires[0].Anomalies, AnomalyDuplicateFireEvent)
	}
	if rep.Verdict != VerdictFiredWithAnomaly {
		t.Errorf("verdict = %q, want %q", rep.Verdict, VerdictFiredWithAnomaly)
	}
}

// A fire whose next non-zero resident sample is not lower did not reduce context. With
// an adjacent witness it is reported at high confidence.
func TestCompactAuditIneffectiveFireFlagged(t *testing.T) {
	rep := scanFixture(t, "ineffective-fire.jsonl")

	if rep.FireCount != 1 {
		t.Fatalf("fire count = %d, want 1", rep.FireCount)
	}
	f := rep.Fires[0]
	if !hasAnomaly(f.Anomalies, AnomalyIneffectiveFire) {
		t.Errorf("anomalies = %v, want %s", f.Anomalies, AnomalyIneffectiveFire)
	}
	if f.PostTokens < f.PreTokens {
		t.Errorf("fixture should not shed: pre %d post %d", f.PreTokens, f.PostTokens)
	}
	if f.Confidence != CompactConfidenceHigh {
		t.Errorf("confidence = %q, want %q — the witness is adjacent, so this is a real ineffective fire",
			f.Confidence, CompactConfidenceHigh)
	}
	if rep.Verdict != VerdictFiredWithAnomaly {
		t.Errorf("verdict = %q, want %q", rep.Verdict, VerdictFiredWithAnomaly)
	}
}

// A fire with no post-fire witness is UNKNOWN, not failed: typed confidence + reason
// rather than a silent assertion.
func TestCompactAuditMissingPostWitnessIsTypedNotAsserted(t *testing.T) {
	rep := scanFixture(t, "missing-token-sample.jsonl")

	if rep.FireCount != 1 {
		t.Fatalf("fire count = %d, want 1", rep.FireCount)
	}
	f := rep.Fires[0]
	if !hasAnomaly(f.Anomalies, AnomalyMissingPostWitness) {
		t.Errorf("anomalies = %v, want %s", f.Anomalies, AnomalyMissingPostWitness)
	}
	if f.Confidence != CompactConfidenceNone || f.Reason != CompactReasonTelemetryMissing {
		t.Errorf("confidence = %s/%s, want %s/%s", f.Confidence, f.Reason,
			CompactConfidenceNone, CompactReasonTelemetryMissing)
	}
	if hasAnomaly(f.Anomalies, AnomalyIneffectiveFire) {
		t.Errorf("a missing witness must not be reported as an ineffective fire: %v", f.Anomalies)
	}
	if f.PreTokens != 240000 {
		t.Errorf("pre = %d, want 240000 (the pre witness is present)", f.PreTokens)
	}
}

// Reaching the ceiling and never firing is the real failure the audit must catch.
func TestCompactAuditNoFireAboveCeilingFlagged(t *testing.T) {
	rollout := strings.Join([]string{
		`{"timestamp":"2026-06-18T07:00:00.000Z","type":"session_meta","payload":{"id":"sess-ceiling","cwd":"C:\\work\\fak"}}`,
		`{"timestamp":"2026-06-18T07:00:01.000Z","type":"event_msg","payload":{"type":"task_started"}}`,
		`{"timestamp":"2026-06-18T07:00:30.000Z","type":"event_msg","payload":{"type":"token_count","info":{"total_token_usage":{"input_tokens":250000},"last_token_usage":{"input_tokens":250000},"model_context_window":258400}}}`,
		"",
	}, "\n")

	rep, err := ScanCompactRollout(strings.NewReader(rollout), "mem", int64(len(rollout)))
	if err != nil {
		t.Fatalf("scan: %v", err)
	}
	if rep.Verdict != VerdictNoFireAtCeiling {
		t.Errorf("verdict = %q, want %q", rep.Verdict, VerdictNoFireAtCeiling)
	}
	if !hasAnomaly(rep.Anomalies, AnomalyNoFireAboveCeiling) {
		t.Errorf("anomalies = %v, want %s", rep.Anomalies, AnomalyNoFireAboveCeiling)
	}
}

// A fire that ran to the ceiling before firing is LATE.
func TestCompactAuditLateFireFlagged(t *testing.T) {
	rollout := strings.Join([]string{
		`{"timestamp":"2026-06-18T08:00:00.000Z","type":"session_meta","payload":{"id":"sess-late","cwd":"C:\\work\\fak"}}`,
		`{"timestamp":"2026-06-18T08:00:01.000Z","type":"event_msg","payload":{"type":"task_started"}}`,
		`{"timestamp":"2026-06-18T08:00:30.000Z","type":"event_msg","payload":{"type":"token_count","info":{"total_token_usage":{"input_tokens":256000},"last_token_usage":{"input_tokens":256000},"model_context_window":258400}}}`,
		`{"timestamp":"2026-06-18T08:00:35.000Z","type":"compacted","payload":{"message":""}}`,
		`{"timestamp":"2026-06-18T08:00:35.005Z","type":"event_msg","payload":{"type":"context_compacted"}}`,
		`{"timestamp":"2026-06-18T08:00:40.000Z","type":"event_msg","payload":{"type":"token_count","info":{"total_token_usage":{"input_tokens":286000},"last_token_usage":{"input_tokens":30000},"model_context_window":258400}}}`,
		"",
	}, "\n")

	rep, err := ScanCompactRollout(strings.NewReader(rollout), "mem", int64(len(rollout)))
	if err != nil {
		t.Fatalf("scan: %v", err)
	}
	if rep.FireCount != 1 {
		t.Fatalf("fire count = %d, want 1", rep.FireCount)
	}
	if !hasAnomaly(rep.Fires[0].Anomalies, AnomalyLateFire) {
		t.Errorf("anomalies = %v, want %s (ceiling ratio %.3f)",
			rep.Fires[0].Anomalies, AnomalyLateFire, rep.Fires[0].CeilingRatio)
	}
}

// Firing repeatedly yet never coming off the ceiling is the oversized-single-item WEDGE:
// compaction keeps cutting but the window walk can never seat a kept window under budget (a
// large image or paste it cannot drop), so resident stays pinned at the top. Two fires that both
// leave post-fire resident above the late-fire ceiling must land WEDGED_AT_CEILING — distinct from
// a single LATE_FIRE and from NO_FIRE_ABOVE_CEILING.
func TestCompactAuditWedgedAtCeilingFlagged(t *testing.T) {
	// window 258400; late-fire ceiling = 0.95*258400 ≈ 245480. Both fires leave post ≈ 250000,
	// still above the ceiling — the session fires and never gets headroom.
	rollout := strings.Join([]string{
		`{"timestamp":"2026-06-18T09:00:00.000Z","type":"session_meta","payload":{"id":"sess-wedge","cwd":"C:\\work\\fak"}}`,
		`{"timestamp":"2026-06-18T09:00:01.000Z","type":"event_msg","payload":{"type":"task_started"}}`,
		`{"timestamp":"2026-06-18T09:00:10.000Z","type":"event_msg","payload":{"type":"token_count","info":{"total_token_usage":{"input_tokens":255000},"last_token_usage":{"input_tokens":255000},"model_context_window":258400}}}`,
		`{"timestamp":"2026-06-18T09:00:15.000Z","type":"compacted","payload":{"message":""}}`,
		`{"timestamp":"2026-06-18T09:00:15.005Z","type":"event_msg","payload":{"type":"context_compacted"}}`,
		`{"timestamp":"2026-06-18T09:00:20.000Z","type":"event_msg","payload":{"type":"token_count","info":{"total_token_usage":{"input_tokens":510000},"last_token_usage":{"input_tokens":250000},"model_context_window":258400}}}`,
		`{"timestamp":"2026-06-18T09:00:25.000Z","type":"event_msg","payload":{"type":"task_started"}}`,
		`{"timestamp":"2026-06-18T09:00:30.000Z","type":"compacted","payload":{"message":""}}`,
		`{"timestamp":"2026-06-18T09:00:30.005Z","type":"event_msg","payload":{"type":"context_compacted"}}`,
		`{"timestamp":"2026-06-18T09:00:35.000Z","type":"event_msg","payload":{"type":"token_count","info":{"total_token_usage":{"input_tokens":760000},"last_token_usage":{"input_tokens":251000},"model_context_window":258400}}}`,
		"",
	}, "\n")

	rep, err := ScanCompactRollout(strings.NewReader(rollout), "mem", int64(len(rollout)))
	if err != nil {
		t.Fatalf("scan: %v", err)
	}
	if rep.FireCount != 2 {
		t.Fatalf("fire count = %d, want 2", rep.FireCount)
	}
	if rep.Verdict != VerdictWedgedAtCeiling {
		t.Errorf("verdict = %q, want %q", rep.Verdict, VerdictWedgedAtCeiling)
	}
	if !hasAnomaly(rep.Anomalies, AnomalyWedgedAtCeiling) {
		t.Errorf("anomalies = %v, want %s", rep.Anomalies, AnomalyWedgedAtCeiling)
	}
}

// A healthy repeated fire that DOES come off the ceiling must NOT be read as wedged — the wedge
// signal keys on post-fire resident staying at the top, not on the fire count.
func TestCompactAuditHealthyRepeatedFireNotWedged(t *testing.T) {
	rep := scanFixture(t, "healthy-repeated-fire.jsonl")
	if rep.Verdict == VerdictWedgedAtCeiling {
		t.Errorf("healthy repeated-fire session read as wedged: %v", rep.Anomalies)
	}
	if hasAnomaly(rep.Anomalies, AnomalyWedgedAtCeiling) {
		t.Errorf("healthy session flagged WEDGED_AT_CEILING: %v", rep.Anomalies)
	}
}

// The audit must never surface prompt or tool-output bodies — not in any report field,
// not in the JSON form. The fixtures carry sentinel bodies for exactly this check.
func TestCompactAuditNeverEmitsPromptOrToolBodies(t *testing.T) {
	for _, name := range []string{
		"healthy-repeated-fire.jsonl",
		"paired-duplicate.jsonl",
		"ineffective-fire.jsonl",
		"missing-token-sample.jsonl",
		"long-append-only-bounded.jsonl",
		// The wedge fixture is the only one carrying an inline base64 image body, the
		// surface that wedges compaction in the first place (#5168).
		"image-wedge.jsonl",
	} {
		rep := scanFixture(t, name)
		blob, err := json.Marshal(rep)
		if err != nil {
			t.Fatalf("marshal %s: %v", name, err)
		}
		for _, sentinel := range []string{"MUST_NOT_LEAK", "replacement_history", "BOUNDED_TOOL_OUTPUT"} {
			if strings.Contains(string(blob), sentinel) {
				t.Errorf("%s: report leaked %q — the audit must stay body-blind", name, sentinel)
			}
		}
	}
}

// A `compacted` row carrying a megabyte of replacement_history overruns the head bound.
// The scanner must still (a) recognize the fire, (b) recover its timestamp so the
// context_compacted twin dedupes against it instead of double-counting, and (c) keep
// parsing the rows after it.
func TestCompactAuditOverlongFireRowStillPairsAndScans(t *testing.T) {
	body := strings.Repeat("LEAKY_PROMPT_BODY ", 20000) // ~360 KB, well past the head bound
	rollout := strings.Join([]string{
		`{"timestamp":"2026-06-18T09:00:00.000Z","type":"session_meta","payload":{"id":"sess-huge","cwd":"C:\\work\\fak"}}`,
		`{"timestamp":"2026-06-18T09:00:01.000Z","type":"event_msg","payload":{"type":"task_started"}}`,
		`{"timestamp":"2026-06-18T09:00:30.000Z","type":"event_msg","payload":{"type":"token_count","info":{"total_token_usage":{"input_tokens":240000},"last_token_usage":{"input_tokens":240000},"model_context_window":258400}}}`,
		`{"timestamp":"2026-06-18T09:00:35.642Z","type":"compacted","payload":{"message":"","replacement_history":[{"text":"` + body + `"}]}}`,
		`{"timestamp":"2026-06-18T09:00:35.645Z","type":"event_msg","payload":{"type":"token_count","info":{"total_token_usage":{"input_tokens":240000},"last_token_usage":{"input_tokens":0},"model_context_window":258400}}}`,
		`{"timestamp":"2026-06-18T09:00:35.647Z","type":"event_msg","payload":{"type":"context_compacted"}}`,
		`{"timestamp":"2026-06-18T09:00:40.000Z","type":"event_msg","payload":{"type":"token_count","info":{"total_token_usage":{"input_tokens":265000},"last_token_usage":{"input_tokens":25000},"model_context_window":258400}}}`,
		"",
	}, "\n")

	rep, err := ScanCompactRollout(strings.NewReader(rollout), "mem", int64(len(rollout)))
	if err != nil {
		t.Fatalf("scan: %v", err)
	}
	if rep.FireCount != 1 {
		t.Fatalf("fire count = %d, want 1 — an over-long compacted row must still pair with its twin", rep.FireCount)
	}
	if rep.PairedEvents != 1 {
		t.Errorf("paired events = %d, want 1", rep.PairedEvents)
	}
	f := rep.Fires[0]
	if f.At.IsZero() {
		t.Error("fire timestamp lost on the over-long row — pair dedup would break on real rollouts")
	}
	if f.PreTokens != 240000 || f.PostTokens != 25000 {
		t.Errorf("pre/post = %d/%d, want 240000/25000 (rows after the huge one must still parse)", f.PreTokens, f.PostTokens)
	}
	blob, _ := json.Marshal(rep)
	if strings.Contains(string(blob), "LEAKY_PROMPT_BODY") {
		t.Error("over-long row body leaked into the report")
	}
}

// A fire that refills to ~pre-fire within a few turns bought no real headroom.
func TestCompactAuditFastReboundFlagged(t *testing.T) {
	rollout := strings.Join([]string{
		`{"timestamp":"2026-06-18T10:00:00.000Z","type":"session_meta","payload":{"id":"sess-rebound","cwd":"C:\\work\\fak"}}`,
		`{"timestamp":"2026-06-18T10:00:01.000Z","type":"event_msg","payload":{"type":"task_started"}}`,
		`{"timestamp":"2026-06-18T10:00:30.000Z","type":"event_msg","payload":{"type":"token_count","info":{"total_token_usage":{"input_tokens":200000},"last_token_usage":{"input_tokens":200000},"model_context_window":258400}}}`,
		`{"timestamp":"2026-06-18T10:00:35.000Z","type":"compacted","payload":{"message":""}}`,
		`{"timestamp":"2026-06-18T10:00:35.005Z","type":"event_msg","payload":{"type":"context_compacted"}}`,
		`{"timestamp":"2026-06-18T10:00:40.000Z","type":"event_msg","payload":{"type":"token_count","info":{"total_token_usage":{"input_tokens":220000},"last_token_usage":{"input_tokens":20000},"model_context_window":258400}}}`,
		`{"timestamp":"2026-06-18T10:01:00.000Z","type":"event_msg","payload":{"type":"task_started"}}`,
		`{"timestamp":"2026-06-18T10:01:30.000Z","type":"event_msg","payload":{"type":"token_count","info":{"total_token_usage":{"input_tokens":410000},"last_token_usage":{"input_tokens":190000},"model_context_window":258400}}}`,
		"",
	}, "\n")

	rep, err := ScanCompactRollout(strings.NewReader(rollout), "mem", int64(len(rollout)))
	if err != nil {
		t.Fatalf("scan: %v", err)
	}
	f := rep.Fires[0]
	if !hasAnomaly(f.Anomalies, AnomalyFastRebound) {
		t.Errorf("anomalies = %v, want %s (rebound turns %d)", f.Anomalies, AnomalyFastRebound, f.ReboundTurns)
	}
	if f.ReboundTurns != 1 {
		t.Errorf("rebound turns = %d, want 1", f.ReboundTurns)
	}
}

func TestAggregateCompactReportsUsesMedians(t *testing.T) {
	reports := []CompactSessionReport{
		scanFixture(t, "healthy-repeated-fire.jsonl"),
		scanFixture(t, "ineffective-fire.jsonl"),
		scanFixture(t, "long-append-only-bounded.jsonl"),
	}
	agg := AggregateCompactReports(reports)

	if agg.Sessions != 3 {
		t.Errorf("sessions = %d, want 3", agg.Sessions)
	}
	if agg.Fires != 3 {
		t.Errorf("fires = %d, want 3 (2 healthy + 1 ineffective)", agg.Fires)
	}
	if agg.CompactedSessions != 2 {
		t.Errorf("compacted sessions = %d, want 2", agg.CompactedSessions)
	}
	if agg.MeasuredFires != 3 {
		t.Errorf("measured fires = %d, want 3", agg.MeasuredFires)
	}
	// pre samples: 242067, 240000, 240000 -> median 240000
	if agg.MedianPreTokens != 240000 {
		t.Errorf("median pre = %d, want 240000", agg.MedianPreTokens)
	}
	if agg.VerdictCounts[VerdictNoFireBounded] != 1 {
		t.Errorf("verdict counts = %v, want one %s", agg.VerdictCounts, VerdictNoFireBounded)
	}
	if agg.AnomalyCounts[AnomalyIneffectiveFire] != 1 {
		t.Errorf("anomaly counts = %v, want one %s", agg.AnomalyCounts, AnomalyIneffectiveFire)
	}
}

// The checked-in artifact must carry no filesystem paths or session cwd.
func TestScrubCompactResultDropsPrivatePaths(t *testing.T) {
	res := CompactAuditResult{
		Root:     `C:\Users\someone\.codex\sessions`,
		Sessions: []CompactSessionReport{scanFixture(t, "healthy-repeated-fire.jsonl")},
	}
	if res.Sessions[0].Cwd == "" || res.Sessions[0].Path == "" {
		t.Fatal("fixture should carry a path and cwd before scrubbing")
	}
	scrubbed := ScrubCompactResult(res)

	if scrubbed.Root != "" {
		t.Errorf("root = %q, want empty", scrubbed.Root)
	}
	for _, s := range scrubbed.Sessions {
		if s.Path != "" || s.Cwd != "" {
			t.Errorf("session kept path=%q cwd=%q, want both empty", s.Path, s.Cwd)
		}
		if s.SessionID == "" {
			t.Error("scrub dropped the session id — the artifact needs it to be reproducible")
		}
		if s.FireCount == 0 {
			t.Error("scrub dropped the witnesses")
		}
	}
	blob, _ := json.Marshal(scrubbed)
	for _, needle := range []string{`work\fak`, ".codex", "Users"} {
		if strings.Contains(string(blob), needle) {
			t.Errorf("scrubbed artifact still contains %q", needle)
		}
	}
}

// The human report must name the three quantities apart and say "fired and held" —
// the DoD's explicit anti-misread requirement.
func TestRenderCompactAuditSeparatesBytesFromResident(t *testing.T) {
	res := CompactAuditResult{Sessions: []CompactSessionReport{scanFixture(t, "healthy-repeated-fire.jsonl")}}
	res.Aggregate = AggregateCompactReports(res.Sessions)

	var sb strings.Builder
	RenderCompactAudit(&sb, res, 5)
	out := sb.String()

	for _, want := range []string{
		"append-only",
		"resident context",
		"compaction fired and held",
		"HEALTHY",
	} {
		if !strings.Contains(out, want) {
			t.Errorf("render missing %q\n---\n%s", want, out)
		}
	}
	if strings.Contains(out, "MUST_NOT_LEAK") {
		t.Error("render leaked a prompt body")
	}
}

func TestAuditCompactCorpusWalksAndFilters(t *testing.T) {
	res, err := AuditCompactCorpus(CompactAuditOptions{Root: filepath.Join("testdata", "compactaudit")})
	if err != nil {
		t.Fatalf("audit corpus: %v", err)
	}
	if res.Aggregate.Sessions != 6 {
		t.Errorf("sessions = %d, want 6 fixtures", res.Aggregate.Sessions)
	}
	if res.Aggregate.Fires != 7 {
		t.Errorf("fires = %d, want 7 (2 healthy + 1 dup-session + 1 ineffective + 1 missing-post + 2 image-wedge)", res.Aggregate.Fires)
	}

	// The cwd filter keeps only this repo's sessions.
	none, err := AuditCompactCorpus(CompactAuditOptions{
		Root: filepath.Join("testdata", "compactaudit"),
		Cwd:  "no-such-workspace",
	})
	if err != nil {
		t.Fatalf("audit corpus (filtered): %v", err)
	}
	if none.Aggregate.Sessions != 0 {
		t.Errorf("filtered sessions = %d, want 0", none.Aggregate.Sessions)
	}

	// A future --since drops everything.
	future, err := AuditCompactCorpus(CompactAuditOptions{
		Root:  filepath.Join("testdata", "compactaudit"),
		Since: time.Now().Add(24 * time.Hour),
	})
	if err != nil {
		t.Fatalf("audit corpus (since): %v", err)
	}
	if future.Aggregate.Sessions != 0 {
		t.Errorf("since-filtered sessions = %d, want 0", future.Aggregate.Sessions)
	}
}
