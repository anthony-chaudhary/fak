package disambiguation

import (
	"encoding/json"
	"os"
	"strings"
	"testing"
)

func loadEntryFixture(t *testing.T) []byte {
	t.Helper()
	data, err := os.ReadFile("testdata/entry-v1.json")
	if err != nil {
		t.Fatal(err)
	}
	return data
}

func TestSchemaSelfTest(t *testing.T) {
	if err := SelfTest(); err != nil {
		t.Fatal(err)
	}
}

func TestParseEntryAcceptsCompleteV1Record(t *testing.T) {
	entry, err := ParseEntry(loadEntryFixture(t))
	if err != nil {
		t.Fatal(err)
	}
	if entry.Schema != EntrySchemaVersion {
		t.Fatalf("schema = %q, want %q", entry.Schema, EntrySchemaVersion)
	}
	if entry.Identity.CanonicalTerm != "agent kernel" {
		t.Fatalf("canonical term = %q", entry.Identity.CanonicalTerm)
	}
	if entry.Freshness.Verdict != FreshnessFresh {
		t.Fatalf("freshness verdict = %q", entry.Freshness.Verdict)
	}
}

func TestParseEntryRejectsMissingRequiredGroups(t *testing.T) {
	var complete map[string]any
	if err := json.Unmarshal(loadEntryFixture(t), &complete); err != nil {
		t.Fatal(err)
	}
	tests := []struct {
		field string
		want  string
	}{
		{field: "identity", want: "identity"},
		{field: "owner", want: "owner"},
		{field: "contrasts", want: "contrast"},
		{field: "sources", want: "source"},
		{field: "freshness", want: "freshness"},
	}
	for _, tt := range tests {
		t.Run(tt.field, func(t *testing.T) {
			candidate := cloneObject(t, complete)
			delete(candidate, tt.field)
			data, err := json.Marshal(candidate)
			if err != nil {
				t.Fatal(err)
			}
			_, err = ParseEntry(data)
			if err == nil || !strings.Contains(err.Error(), tt.want) {
				t.Fatalf("ParseEntry error = %v, want containing %q", err, tt.want)
			}
		})
	}
}

func TestParseEntryRejectsVersionAndShapeDrift(t *testing.T) {
	var complete map[string]any
	if err := json.Unmarshal(loadEntryFixture(t), &complete); err != nil {
		t.Fatal(err)
	}

	next := cloneObject(t, complete)
	next["schema"] = "fak-disambiguation-entry/2"
	nextJSON, err := json.Marshal(next)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := ParseEntry(nextJSON); err == nil || !strings.Contains(err.Error(), "unsupported schema") {
		t.Fatalf("next schema error = %v, want unsupported schema", err)
	}

	unknown := cloneObject(t, complete)
	unknown["future_field"] = true
	unknownJSON, err := json.Marshal(unknown)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := ParseEntry(unknownJSON); err == nil || !strings.Contains(err.Error(), "unknown field") {
		t.Fatalf("unknown field error = %v, want unknown field", err)
	}

	nestedUnknown := cloneObject(t, complete)
	identity := nestedUnknown["identity"].(map[string]any)
	identity["future_field"] = true
	nestedUnknownJSON, err := json.Marshal(nestedUnknown)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := ParseEntry(nestedUnknownJSON); err == nil || !strings.Contains(err.Error(), "unknown field") {
		t.Fatalf("nested unknown field error = %v, want unknown field", err)
	}

	trailing := append(loadEntryFixture(t), []byte("\n{}")...)
	if _, err := ParseEntry(trailing); err == nil || !strings.Contains(err.Error(), "trailing JSON") {
		t.Fatalf("trailing JSON error = %v, want trailing JSON", err)
	}
}

func TestValidateRejectsIncompleteV1Fields(t *testing.T) {
	base, err := ParseEntry(loadEntryFixture(t))
	if err != nil {
		t.Fatal(err)
	}
	tests := []struct {
		name string
		edit func(*Entry)
		want string
	}{
		{"aliases field", func(e *Entry) { e.Identity.Aliases = nil }, "identity.aliases"},
		{"definition", func(e *Entry) { e.Definition = "" }, "definition"},
		{"contrast explanation", func(e *Entry) { e.Contrasts[0].Explanation = "" }, "contrast"},
		{"scope", func(e *Entry) { e.Scope.Kind = "" }, "scope.kind"},
		{"owner lane", func(e *Entry) { e.Owner.Lane = "" }, "owner.lane"},
		{"source revision", func(e *Entry) { e.Sources[0].Revision = "" }, "source"},
		{"freshness reason", func(e *Entry) { e.Freshness.ReasonCode = "" }, "freshness.reason_code"},
		{"freshness time", func(e *Entry) { e.Freshness.CheckedAt = "today" }, "RFC3339"},
		{"lifecycle class", func(e *Entry) { e.Lifecycle.Class = "active" }, "lifecycle.class"},
		{"rollout", func(e *Entry) { e.Lifecycle.Rollout = "enabled" }, "lifecycle.rollout"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			entry := base
			entry.Identity.Aliases = append([]string(nil), base.Identity.Aliases...)
			entry.Contrasts = append([]Contrast(nil), base.Contrasts...)
			entry.Sources = append([]SourceWitness(nil), base.Sources...)
			tt.edit(&entry)
			if err := entry.Validate(); err == nil || !strings.Contains(err.Error(), tt.want) {
				t.Fatalf("Validate error = %v, want containing %q", err, tt.want)
			}
		})
	}
}

func TestFreshnessVerdictVocabulary(t *testing.T) {
	base, err := ParseEntry(loadEntryFixture(t))
	if err != nil {
		t.Fatal(err)
	}
	for _, verdict := range []FreshnessVerdict{
		FreshnessFresh,
		FreshnessStale,
		FreshnessUnknown,
		FreshnessInvalid,
	} {
		entry := base
		entry.Freshness.Verdict = verdict
		if err := entry.Validate(); err != nil {
			t.Fatalf("verdict %q rejected: %v", verdict, err)
		}
	}
	entry := base
	entry.Freshness.Verdict = "available"
	if err := entry.Validate(); err == nil || !strings.Contains(err.Error(), "freshness.verdict") {
		t.Fatalf("unknown freshness verdict error = %v", err)
	}
}

func cloneObject(t *testing.T, src map[string]any) map[string]any {
	t.Helper()
	data, err := json.Marshal(src)
	if err != nil {
		t.Fatal(err)
	}
	var dst map[string]any
	if err := json.Unmarshal(data, &dst); err != nil {
		t.Fatal(err)
	}
	return dst
}
