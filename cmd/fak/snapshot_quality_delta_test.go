package main

import (
	"testing"

	"github.com/anthony-chaudhary/fak/internal/session"
	"github.com/anthony-chaudhary/fak/internal/sessionimage"
)

// TestSnapshotInfoRendersQualityDelta is the #1979 inspect-render witness (QA-dogfood
// spine #1961/QD-019): the done condition names two effects — dump/restore PRESERVES the
// latest quality deltas (proven in internal/sessionimage) and `fak snapshot info` RENDERS
// them. This drives the CLI's inspect header builder against a real image carrying a
// quality.json part and asserts the deltas surface in the rendered output, so an operator
// inspecting a restored image reads "did quality move, against what evidence?" from the
// integrity-checked part rather than re-running a scorecard.
func TestSnapshotInfoRendersQualityDelta(t *testing.T) {
	dir := t.TempDir()

	deltas := []sessionimage.QualityDelta{
		{CardKey: "code_quality", Before: 42, After: 37, Evidence: "docs/nightrun/quality.jsonl"},
		{CardKey: "milestone_scorecard", Before: 0.71, After: 0.83, Evidence: "fak milestone report --json"},
	}
	if _, err := sessionimage.DumpDir(dir, sessionimage.Input{
		SessionID: "sess-qd-render",
		Drive:     session.DefaultState("sess-qd-render"),
		Quality:   deltas,
		Now:       1,
	}); err != nil {
		t.Fatalf("DumpDir: %v", err)
	}
	img, err := sessionimage.LoadDir(dir)
	if err != nil {
		t.Fatalf("LoadDir: %v", err)
	}

	info := sessionImageInfo(img)
	if _, ok := info["quality_deltas_error"]; ok {
		t.Fatalf("inspect reported a quality-delta decode error: %v", info["quality_deltas_error"])
	}
	rendered, ok := info["quality_deltas"].([]sessionimage.QualityDelta)
	if !ok {
		t.Fatalf("inspect output did not render quality_deltas; got %T (%v)", info["quality_deltas"], info["quality_deltas"])
	}
	if len(rendered) != 2 {
		t.Fatalf("want 2 rendered deltas, got %d: %+v", len(rendered), rendered)
	}
	// Deterministically sorted by CardKey on write.
	if rendered[0].CardKey != "code_quality" || rendered[0].After != 37 || rendered[0].Evidence != "docs/nightrun/quality.jsonl" {
		t.Fatalf("first rendered delta wrong: %+v", rendered[0])
	}
	if rendered[1].CardKey != "milestone_scorecard" || rendered[1].After != 0.83 {
		t.Fatalf("second rendered delta wrong: %+v", rendered[1])
	}
}

// TestSnapshotInfoOmitsQualityDeltaWhenAbsent pins that an image no scorecard ran against
// renders no quality_deltas key at all — inspect never invents a phantom empty block.
func TestSnapshotInfoOmitsQualityDeltaWhenAbsent(t *testing.T) {
	dir := t.TempDir()
	if _, err := sessionimage.DumpDir(dir, sessionimage.Input{
		SessionID: "sess-qd-none",
		Drive:     session.DefaultState("sess-qd-none"),
		Now:       1,
	}); err != nil {
		t.Fatalf("DumpDir: %v", err)
	}
	img, err := sessionimage.LoadDir(dir)
	if err != nil {
		t.Fatalf("LoadDir: %v", err)
	}
	info := sessionImageInfo(img)
	if _, ok := info["quality_deltas"]; ok {
		t.Fatalf("inspect rendered a quality_deltas block for an image that carried none: %v", info["quality_deltas"])
	}
	if _, ok := info["quality_deltas_error"]; ok {
		t.Fatalf("inspect reported a spurious quality-delta error: %v", info["quality_deltas_error"])
	}
}
