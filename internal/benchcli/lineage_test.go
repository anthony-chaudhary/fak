package benchcli

import (
	"encoding/json"
	"runtime"
	"strings"
	"testing"
)

// A spliced lineage block must be the FIRST member, valid JSON, carry all four
// axes, and leave the caller's original members byte-for-byte intact.
func TestLineageIntoSplicesFirstMember(t *testing.T) {
	lin := Lineage{
		Schema: LineageSchema, AppVersion: "0.0.0", UTC: "2026-06-29T00:00:00Z",
		GitCommit: "deadbee", GoVersion: "go1.23.1", Node: "node-test",
	}
	report := map[string]any{"peak_tok_per_sec": 12.5, "model": "smollm2"}
	b, _ := json.MarshalIndent(report, "", "  ")

	got := lin.Into(b)

	// Still valid JSON, with both the report keys and the lineage block.
	var round map[string]json.RawMessage
	if err := json.Unmarshal(got, &round); err != nil {
		t.Fatalf("stamped report is not valid JSON: %v\n%s", err, got)
	}
	rawLin, ok := round["lineage"]
	if !ok {
		t.Fatalf("no lineage member after Into:\n%s", got)
	}
	if _, ok := round["model"]; !ok {
		t.Fatalf("original member 'model' lost after Into:\n%s", got)
	}
	var back Lineage
	if err := json.Unmarshal(rawLin, &back); err != nil {
		t.Fatalf("lineage block does not decode: %v", err)
	}
	if back != lin {
		t.Fatalf("lineage round-trip = %+v, want %+v", back, lin)
	}
	// lineage must be the first member (the run before its key has no other key).
	li := strings.Index(string(got), `"lineage"`)
	mi := strings.Index(string(got), `"model"`)
	pi := strings.Index(string(got), `"peak_tok_per_sec"`)
	if li > mi || li > pi {
		t.Fatalf("lineage is not the first member (lineage@%d model@%d peak@%d)", li, mi, pi)
	}
}

// Into is a no-op on a non-object, on a compact object with no newline to align
// to, and on a report that already carries lineage (idempotent — no double stamp).
func TestLineageIntoNoOpCases(t *testing.T) {
	lin := Stamp()
	cases := map[string][]byte{
		"array":   []byte("[\n  1,\n  2\n]"),
		"scalar":  []byte("42"),
		"compact": []byte(`{"a":1}`),
		"empty":   []byte(""),
	}
	for name, in := range cases {
		if got := lin.Into(in); string(got) != string(in) {
			t.Errorf("%s: Into mutated a non-spliceable input:\n%s", name, got)
		}
	}
	// Idempotent: a second stamp must not add a second lineage block.
	b, _ := json.MarshalIndent(map[string]any{"a": 1}, "", "  ")
	once := lin.Into(b)
	twice := lin.Into(once)
	if string(once) != string(twice) {
		t.Fatalf("Into is not idempotent:\nonce:\n%s\ntwice:\n%s", once, twice)
	}
	if n := strings.Count(string(twice), `"lineage"`); n != 1 {
		t.Fatalf("double-stamp produced %d lineage members, want 1", n)
	}
}

// An empty object must not gain a trailing comma (which would be invalid JSON).
func TestLineageIntoEmptyObject(t *testing.T) {
	b, _ := json.MarshalIndent(map[string]any{}, "", "  ")
	got := Lineage{Schema: LineageSchema}.Into(b)
	if !json.Valid(got) {
		t.Fatalf("stamped empty object is invalid JSON:\n%s", got)
	}
}

// Stamp fills every axis, honours the deterministic env overrides, and never
// leaves a field blank (fail-soft to "unknown" or a live value).
func TestStampFieldsAndOverrides(t *testing.T) {
	t.Setenv("FAK_BENCH_UTC", "2099-01-02T03:04:05Z")
	t.Setenv("FAK_BENCH_COMMIT", "cafef00d")
	t.Setenv("FAK_BENCH_NODE", "pinned-node")

	lin := Stamp()
	if lin.Schema != LineageSchema {
		t.Errorf("schema = %q, want %q", lin.Schema, LineageSchema)
	}
	if lin.UTC != "2099-01-02T03:04:05Z" {
		t.Errorf("utc override not honoured: %q", lin.UTC)
	}
	if lin.GitCommit != "cafef00d" {
		t.Errorf("commit override not honoured: %q", lin.GitCommit)
	}
	if lin.Node != "pinned-node" {
		t.Errorf("node override not honoured: %q", lin.Node)
	}
	if lin.GoVersion != runtime.Version() {
		t.Errorf("go_version = %q, want %q", lin.GoVersion, runtime.Version())
	}
	if strings.TrimSpace(lin.AppVersion) == "" {
		t.Errorf("app_version is blank; appversion.Current() must always resolve")
	}
}

// MarshalReport accepts both a value and pre-marshalled bytes, stamping either.
func TestMarshalReportValueAndBytes(t *testing.T) {
	t.Setenv("FAK_BENCH_COMMIT", "abc1234")
	fromValue, err := MarshalReport(map[string]any{"k": "v"})
	if err != nil {
		t.Fatalf("MarshalReport(value): %v", err)
	}
	if !strings.Contains(string(fromValue), `"lineage"`) || !strings.Contains(string(fromValue), `"abc1234"`) {
		t.Fatalf("value form missing lineage:\n%s", fromValue)
	}
	pre, _ := json.MarshalIndent(map[string]any{"k": "v"}, "", "  ")
	fromBytes, err := MarshalReport(pre)
	if err != nil {
		t.Fatalf("MarshalReport(bytes): %v", err)
	}
	if !strings.Contains(string(fromBytes), `"lineage"`) {
		t.Fatalf("bytes form missing lineage:\n%s", fromBytes)
	}
}
