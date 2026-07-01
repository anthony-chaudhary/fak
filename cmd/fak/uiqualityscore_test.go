package main

import (
	"bytes"
	"encoding/json"
	"strings"
	"testing"
)

// TestUIQualityScoreJSONShape smoke-tests the verb end to end against the live
// render tree: it must emit the control-pane shape the unified scorecard folds
// (schema + corpus.ui_quality_debt + corpus.grade), so a registration drift is
// caught here rather than silently un-folded.
func TestUIQualityScoreJSONShape(t *testing.T) {
	var out, errBuf bytes.Buffer
	// repoRoot() resolves the live tree; exit code reflects debt (0 = clean).
	_ = runUIQualityScore(&out, &errBuf, []string{"--json"})
	if errBuf.Len() != 0 {
		t.Fatalf("unexpected stderr: %s", errBuf.String())
	}
	var payload struct {
		Schema string `json:"schema"`
		Corpus struct {
			Debt  *int   `json:"ui_quality_debt"`
			Grade string `json:"grade"`
		} `json:"corpus"`
		KPIs []struct {
			Key string `json:"key"`
		} `json:"kpis"`
	}
	if err := json.Unmarshal(out.Bytes(), &payload); err != nil {
		t.Fatalf("verb did not emit valid JSON: %v\n%s", err, out.String())
	}
	if !strings.HasPrefix(payload.Schema, "fak-ui-quality-scorecard/") {
		t.Fatalf("schema = %q, want fak-ui-quality-scorecard/*", payload.Schema)
	}
	if payload.Corpus.Debt == nil {
		t.Fatal("corpus.ui_quality_debt missing — the control pane folds on this key")
	}
	if payload.Corpus.Grade == "" {
		t.Fatal("corpus.grade missing")
	}
	if len(payload.KPIs) == 0 {
		t.Fatal("no KPIs emitted")
	}
}

// TestUIQualityScoreHumanRuns confirms the default human render does not error and
// names the surface it grades.
func TestUIQualityScoreHumanRuns(t *testing.T) {
	var out, errBuf bytes.Buffer
	_ = runUIQualityScore(&out, &errBuf, nil)
	if !strings.Contains(out.String(), "ui quality scorecard") || !strings.Contains(out.String(), "ui_quality_debt") {
		t.Fatalf("human output missing header:\n%s", out.String())
	}
}

// TestUIQualityScoreRejectsExtraArg pins the arg-validation contract shared by the
// scorecard verbs.
func TestUIQualityScoreRejectsExtraArg(t *testing.T) {
	var out, errBuf bytes.Buffer
	if code := runUIQualityScore(&out, &errBuf, []string{"unexpected"}); code != 2 {
		t.Fatalf("extra arg should exit 2, got %d", code)
	}
}
