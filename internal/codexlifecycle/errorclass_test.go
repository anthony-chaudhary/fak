package codexlifecycle

// #10668 tests: unknown-payload census, abort-cause classification, and
// provider error classes extracted from terminal task_complete free text.
// The committed fixture (testdata/errorclass/issue-10668) carries the two
// audited shapes — a clean user interrupt vs a torn-tail process death, and
// status-class mentions inside terminal free text — with conformance.json
// pinning the exact counts and a privacy probe string that must NEVER appear
// in any report.

import (
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

var errorClassFixtureDir = filepath.Join("testdata", "errorclass", "issue-10668")

// FIXTURE — the census counts unrecognized event_msg payload types by name
// without erroring, while the known lifecycle events keep flowing to the
// existing consumers unchanged.
func TestReadRolloutCensus_CountsUnknownPayloadTypes(t *testing.T) {
	body := strings.Join([]string{
		meta("u1", "fak", "0.144.4", `C:\work\fak`),
		started("2026-09-01T08:00:01.000Z", "A"),
		`{"timestamp":"2026-09-01T08:00:02.000Z","type":"event_msg","payload":{"type":"item_completed","turn_id":"A"}}`,
		`{"timestamp":"2026-09-01T08:00:03.000Z","type":"event_msg","payload":{"type":"thread_settings_applied"}}`,
		complete("2026-09-01T08:00:30.000Z", "A"),
		`{"timestamp":"2026-09-01T08:00:31.000Z","type":"event_msg","payload":{"type":"brand_new_error_event","status_class":"5xx"}}`,
	}, "\n") + "\n"
	meta, events, census, err := ReadRolloutCensus(strings.NewReader(body))
	if err != nil {
		t.Fatalf("ReadRolloutCensus: %v", err)
	}
	if meta.RolloutID == "" || len(events) != 2 {
		t.Errorf("backward-compat path degraded: meta=%+v events=%d, want non-empty meta and 2 lifecycle events", meta, len(events))
	}
	for _, want := range []struct {
		kind string
		n    int
	}{{KindStarted, 1}, {KindComplete, 1}, {"item_completed", 1}, {"thread_settings_applied", 1}, {"brand_new_error_event", 1}} {
		if got := census.PayloadTypes[want.kind]; got != want.n {
			t.Errorf("PayloadTypes[%q] = %d, want %d (full map: %v)", want.kind, got, want.n, census.PayloadTypes)
		}
	}
	if census.TornTail {
		t.Error("torn_tail = true on an intact file")
	}
	// Unknown = the uninterpreted subset; the three lifecycle kinds it DID
	// interpret must not appear there.
	if len(census.Unknown) != 3 {
		t.Errorf("unknown = %v, want exactly the 3 uninterpreted types", census.Unknown)
	}
	if census.Unknown[KindStarted] != 0 || census.Unknown[KindComplete] != 0 {
		t.Errorf("interpreted kinds leaked into the unknown census: %v", census.Unknown)
	}
	if census.Unknown["brand_new_error_event"] != 1 {
		t.Error("a future typed error event must be observed and counted, not dropped")
	}
}

// The OLD signature keeps working unchanged: consumers that ignore unknown
// types keep working (the #10668 backward-compatibility requirement).
func TestReadRollout_BackwardCompatNoCensus(t *testing.T) {
	body := started("2026-09-01T08:00:01.000Z", "A") + "\n" +
		`{"timestamp":"2026-09-01T08:00:02.000Z","type":"event_msg","payload":{"type":"item_completed","turn_id":"A"}}` + "\n" +
		complete("2026-09-01T08:00:30.000Z", "A") + "\n"
	events, err := ParseRollout(strings.NewReader(body))
	if err != nil {
		t.Fatalf("ParseRollout: %v", err)
	}
	if len(events) != 2 || events[0].Kind != KindStarted || events[1].Kind != KindComplete {
		t.Errorf("events = %+v, want the 2 lifecycle events only", events)
	}
}

// The analytics reader carries the same census plus torn-tail detection: a
// final line that fails to parse is writer-death evidence, not a read error.
func TestReadAnalyticsRolloutCensus_CountsUnknownAndTornTail(t *testing.T) {
	body := strings.Join([]string{
		`{"timestamp":"2026-09-01T08:00:02.000Z","type":"event_msg","payload":{"type":"token_count","info":{"last_token_usage":{"input_tokens":5000}}}}`,
		`{"timestamp":"2026-09-01T08:00:03.000Z","type":"event_msg","payload":{"type":"item_completed","turn_id":"A"}}`,
		`{"timestamp":"2026-09-01T08:0`,
	}, "\n")
	_, records, census, err := ReadAnalyticsRolloutCensus(strings.NewReader(body))
	if err != nil {
		t.Fatalf("ReadAnalyticsRolloutCensus must survive a torn tail: %v", err)
	}
	if len(records) != 1 || records[0].Kind != kindTokens {
		t.Errorf("records = %+v, want just the token_count", records)
	}
	if !census.TornTail {
		t.Error("torn_tail = false, want true (final line does not parse)")
	}
	if census.PayloadTypes[kindTokens] != 1 || census.Unknown["item_completed"] != 1 {
		t.Errorf("census = %+v, want token_count known and item_completed unknown", census.PayloadTypes)
	}
}

// Abort-cause classification table: the recorded `interrupted` is preserved
// as one enum member; only torn-tail evidence upgrades it to a process-death
// tail; any other recorded reason outranks the tail heuristic; an empty
// reason stays honestly unrecorded.
func TestClassifyAbort(t *testing.T) {
	for _, tc := range []struct {
		reason     string
		atTornTail bool
		want       AbortClass
	}{
		{"interrupted", false, AbortInterrupted},
		{"interrupted", true, AbortProcessDeathTail},
		{"", false, AbortUnrecorded},
		{"", true, AbortProcessDeathTail},
		{"shutdown", false, AbortOtherRecorded},
		{"shutdown", true, AbortOtherRecorded},
		{"  INTERRUPTED  ", false, AbortInterrupted},
	} {
		if got := ClassifyAbort(tc.reason, tc.atTornTail); got != tc.want {
			t.Errorf("ClassifyAbort(%q, %t) = %q, want %q", tc.reason, tc.atTornTail, got, tc.want)
		}
	}
}

// Provider error classes: code-anchored status references classify to the
// closed 5xx/429/400 vocabulary; bare numbers and prose like "rate limits"
// stay UNCLASSIFIED (conservative — the corpus is full of agent prose that
// *reports* on 502s, and a bare-number match would poison the counts).
func TestClassifyStatusClasses(t *testing.T) {
	for _, tc := range []struct {
		text string
		want []string
	}{
		{"stream ended: HTTP 502 Bad Gateway after 3 attempts, then HTTP 400 on retry", []string{ErrClassHTTP5xx, ErrClassHTTP400}},
		{"hit HTTP 429 before succeeding", []string{ErrClassHTTP429}},
		{"unexpected status 503 Service Unavailable", []string{ErrClassHTTP5xx}},
		{"status code: 429", []string{ErrClassHTTP429}},
		{"upstream 5xx flapped", []string{ErrClassHTTP5xx}},
		{"400 Bad Request on the final attempt", []string{ErrClassHTTP400}},
		{"502 tokens used", nil},
		{"avoid rate limits", nil},
		{"retries: 5023", nil},
		{"", nil},
	} {
		got := ClassifyStatusClasses(tc.text)
		if len(got) != len(tc.want) {
			t.Errorf("ClassifyStatusClasses(%q) = %v, want %v", tc.text, got, tc.want)
			continue
		}
		for i := range got {
			if got[i] != tc.want[i] {
				t.Errorf("ClassifyStatusClasses(%q) = %v, want %v", tc.text, got, tc.want)
			}
		}
	}
}

// The classes are a closed content-free vocabulary: whatever the text says,
// the result can only be the three class tokens.
func TestClassifyStatusClasses_NeverLeaksContent(t *testing.T) {
	text := "stream ended: HTTP 502 Bad Gateway after 3 attempts with SECRET-CONTENT-XYZ inside"
	for _, cl := range ClassifyStatusClasses(text) {
		switch cl {
		case ErrClassHTTP5xx, ErrClassHTTP429, ErrClassHTTP400:
		default:
			t.Errorf("class token %q is outside the closed vocabulary", cl)
		}
	}
	if strings.Contains(strings.Join(ClassifyStatusClasses(text), ","), "SECRET") {
		t.Error("message content leaked into class tokens")
	}
}

type confErrorClass struct {
	Schema string `json:"schema"`
	Expect struct {
		Sessions             int            `json:"sessions"`
		Unreadable           int            `json:"unreadable"`
		TornTailRollouts     int            `json:"torn_tail_rollouts"`
		PayloadTypes         map[string]int `json:"payload_types"`
		UnknownPayloadTypes  map[string]int `json:"unknown_payload_types"`
		UnknownPayloadTotal  int            `json:"unknown_payload_total"`
		RolloutsWithUnknown  int            `json:"rollouts_with_unknown"`
		TurnAborted          int            `json:"turn_aborted"`
		AbortClasses         map[string]int `json:"abort_classes"`
		TaskComplete         int            `json:"task_complete"`
		TerminalsWithClass   int            `json:"terminals_with_class"`
		ProviderErrorClasses map[string]int `json:"provider_error_classes"`
	} `json:"expect"`
	PrivacyProbe string `json:"privacy_probe"`
}

// The fixture reproduces the audited shapes end to end and matches the
// conformance values exactly.
func TestScanErrorClasses_FixtureConformance(t *testing.T) {
	rep, err := ScanErrorClasses(errorClassFixtureDir, ScanOptions{})
	if err != nil {
		t.Fatalf("ScanErrorClasses: %v", err)
	}
	blob, err := os.ReadFile(filepath.Join(errorClassFixtureDir, "conformance.json"))
	if err != nil {
		t.Fatalf("read conformance: %v", err)
	}
	var conf confErrorClass
	if err := json.Unmarshal(blob, &conf); err != nil {
		t.Fatalf("decode conformance: %v", err)
	}
	if rep.Schema != conf.Schema {
		t.Errorf("schema = %q, want %q", rep.Schema, conf.Schema)
	}
	if rep.Sessions != conf.Expect.Sessions || rep.Unreadable != conf.Expect.Unreadable || rep.TornTailRollouts != conf.Expect.TornTailRollouts {
		t.Errorf("sessions/unreadable/torn = %d/%d/%d, want %d/%d/%d",
			rep.Sessions, rep.Unreadable, rep.TornTailRollouts, conf.Expect.Sessions, conf.Expect.Unreadable, conf.Expect.TornTailRollouts)
	}
	if len(rep.PayloadTypes) != len(conf.Expect.PayloadTypes) {
		t.Errorf("payload_types = %v, want %v", rep.PayloadTypes, conf.Expect.PayloadTypes)
	}
	for k, want := range conf.Expect.PayloadTypes {
		if got := rep.PayloadTypes[k]; got != want {
			t.Errorf("payload_types[%q] = %d, want %d", k, got, want)
		}
	}
	if len(rep.UnknownPayloadTypes) != len(conf.Expect.UnknownPayloadTypes) || rep.UnknownPayloadTotal != conf.Expect.UnknownPayloadTotal || rep.RolloutsWithUnknown != conf.Expect.RolloutsWithUnknown {
		t.Errorf("unknown = %v total %d rollouts %d, want %v total %d rollouts %d",
			rep.UnknownPayloadTypes, rep.UnknownPayloadTotal, rep.RolloutsWithUnknown,
			conf.Expect.UnknownPayloadTypes, conf.Expect.UnknownPayloadTotal, conf.Expect.RolloutsWithUnknown)
	}
	if rep.TurnAborted != conf.Expect.TurnAborted {
		t.Errorf("turn_aborted = %d, want %d", rep.TurnAborted, conf.Expect.TurnAborted)
	}
	for _, class := range []AbortClass{AbortInterrupted, AbortProcessDeathTail, AbortUnrecorded, AbortOtherRecorded} {
		if got := rep.AbortClasses[string(class)]; got != conf.Expect.AbortClasses[string(class)] {
			t.Errorf("abort_classes[%q] = %d, want %d", class, got, conf.Expect.AbortClasses[string(class)])
		}
	}
	if rep.TaskComplete != conf.Expect.TaskComplete || rep.TerminalsWithClass != conf.Expect.TerminalsWithClass {
		t.Errorf("task_complete/with_class = %d/%d, want %d/%d",
			rep.TaskComplete, rep.TerminalsWithClass, conf.Expect.TaskComplete, conf.Expect.TerminalsWithClass)
	}
	for k, want := range conf.Expect.ProviderErrorClasses {
		if got := rep.ProviderErrorClasses[k]; got != want {
			t.Errorf("provider_error_classes[%q] = %d, want %d", k, got, want)
		}
	}

	// THE PRIVACY PROBE: the fixture's terminal free text must never surface
	// in any rendering of the report.
	out, err := json.Marshal(rep)
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}
	if conf.PrivacyProbe != "" && strings.Contains(string(out), conf.PrivacyProbe) {
		t.Errorf("report leaks message content: %q found in %s", conf.PrivacyProbe, out)
	}
}

// A rollout made entirely of unknown payload types plus garbage lines counts,
// never errors — the #10668 requirement that new upstream event types are
// observed and counted, not dropped and not fatal.
func TestScanErrorClasses_UnknownOnlyRolloutNeverErrors(t *testing.T) {
	dir := t.TempDir()
	body := strings.Join([]string{
		meta("u1", "fak", "0.144.4", `C:\work\fak`),
		`{"timestamp":"2026-09-01T08:00:01.000Z","type":"event_msg","payload":{"type":"mystery_a"}}`,
		`{"timestamp":"2026-09-01T08:00:02.000Z","type":"event_msg","payload":{"type":"mystery_a"}}`,
		`{"timestamp":"2026-09-01T08:00:03.000Z","type":"event_msg","payload":{"type":"mystery_b"}}`,
		"garbage line",
	}, "\n") + "\n"
	if err := os.WriteFile(filepath.Join(dir, "u.jsonl"), []byte(body), 0o600); err != nil {
		t.Fatalf("write: %v", err)
	}
	rep, err := ScanErrorClasses(dir, ScanOptions{})
	if err != nil {
		t.Fatalf("ScanErrorClasses must not error on unknown-only rollouts: %v", err)
	}
	if rep.Sessions != 1 {
		t.Errorf("sessions = %d, want 1", rep.Sessions)
	}
	if rep.UnknownPayloadTypes["mystery_a"] != 2 || rep.UnknownPayloadTypes["mystery_b"] != 1 {
		t.Errorf("unknown = %v, want mystery_a=2 mystery_b=1", rep.UnknownPayloadTypes)
	}
	if rep.UnknownPayloadTotal != 3 || rep.RolloutsWithUnknown != 1 {
		t.Errorf("total/rollouts = %d/%d, want 3/1", rep.UnknownPayloadTotal, rep.RolloutsWithUnknown)
	}
	if rep.TurnAborted != 0 || rep.TaskComplete != 0 {
		t.Errorf("no terminal records exist here: %+v", rep)
	}
}
