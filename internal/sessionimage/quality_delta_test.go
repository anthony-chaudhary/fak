package sessionimage

import (
	"reflect"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/session"
)

// TestQualityDeltaSurvivesDumpRestore is the #1979 witness (QA-dogfood spine
// #1961/QD-019): a session that observed quality-score movement mid-run stamps the latest
// per-card delta into the image (quality.json), and after the image is dumped and
// restored those deltas read back byte-for-byte through Image.QualityDeltas — no re-run
// scorecard, no narrated claim. The part rides the same integrity boundary as every other
// sibling (LoadDir/verifyParts hashes it), so a truncated/tampered offload fails closed.
func TestQualityDeltaSurvivesDumpRestore(t *testing.T) {
	dir := t.TempDir()

	deltas := []QualityDelta{
		{CardKey: "milestone_scorecard", Before: 0.71, After: 0.83, Evidence: "fak milestone report --json@abc123"},
		{CardKey: "code_quality", Before: 42, After: 37, Evidence: "docs/nightrun/quality.jsonl"},
	}
	in := Input{
		SessionID: "sess-qd",
		Drive:     session.DefaultState("sess-qd"),
		Quality:   deltas,
		Now:       1,
	}
	if _, err := DumpDir(dir, in); err != nil {
		t.Fatalf("DumpDir: %v", err)
	}

	img, err := LoadDir(dir)
	if err != nil {
		t.Fatalf("LoadDir: %v", err)
	}
	got, err := img.QualityDeltas()
	if err != nil {
		t.Fatalf("QualityDeltas: %v", err)
	}
	// Persisted deduped + sorted by CardKey, so the restored order is deterministic
	// regardless of the caller's input order.
	want := []QualityDelta{
		{CardKey: "code_quality", Before: 42, After: 37, Evidence: "docs/nightrun/quality.jsonl"},
		{CardKey: "milestone_scorecard", Before: 0.71, After: 0.83, Evidence: "fak milestone report --json@abc123"},
	}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("quality deltas changed across dump/restore:\n got=%+v\n want=%+v", got, want)
	}

	// The part is integrity-indexed like every other sibling.
	var listed bool
	for _, p := range img.Meta.Parts {
		if p.Name == QualityFile {
			listed = true
		}
	}
	if !listed {
		t.Fatalf("%s not in the image integrity index: %+v", QualityFile, img.Meta.Parts)
	}
}

// TestQualityDeltaLatestWinsPerCard pins the latest-only contract: a fresh delta for a
// CardKey already recorded REPLACES the earlier one, so the part carries the LATEST delta
// per card (the done condition names "the latest session quality deltas"), never a growing
// history.
func TestQualityDeltaLatestWinsPerCard(t *testing.T) {
	dir := t.TempDir()
	in := Input{
		SessionID: "sess-qd2",
		Drive:     session.DefaultState("sess-qd2"),
		Quality: []QualityDelta{
			{CardKey: "code_quality", Before: 10, After: 20, Evidence: "early"},
			{CardKey: "code_quality", Before: 20, After: 31, Evidence: "later"},
		},
		Now: 1,
	}
	if _, err := DumpDir(dir, in); err != nil {
		t.Fatalf("DumpDir: %v", err)
	}
	img, err := LoadDir(dir)
	if err != nil {
		t.Fatalf("LoadDir: %v", err)
	}
	got, err := img.QualityDeltas()
	if err != nil {
		t.Fatalf("QualityDeltas: %v", err)
	}
	if len(got) != 1 {
		t.Fatalf("want 1 latest delta per card, got %d: %+v", len(got), got)
	}
	if got[0].After != 31 || got[0].Evidence != "later" {
		t.Fatalf("latest-wins lost: got %+v", got[0])
	}
}

// TestQualityDeltaAbsentStaysNil anchors wire compatibility: a session no scorecard ran
// against carries no quality.json, and QualityDeltas restores nil — never a phantom
// empty-but-present part.
func TestQualityDeltaAbsentStaysNil(t *testing.T) {
	dir := t.TempDir()
	if _, err := DumpDir(dir, Input{SessionID: "sess-noqd", Drive: session.DefaultState("sess-noqd"), Now: 1}); err != nil {
		t.Fatalf("DumpDir: %v", err)
	}
	img, err := LoadDir(dir)
	if err != nil {
		t.Fatalf("LoadDir: %v", err)
	}
	got, err := img.QualityDeltas()
	if err != nil {
		t.Fatalf("QualityDeltas: %v", err)
	}
	if got != nil {
		t.Fatalf("absent quality deltas restored non-nil: %+v", got)
	}
	for _, p := range img.Meta.Parts {
		if p.Name == QualityFile {
			t.Fatalf("%s listed for a session that carried no deltas", QualityFile)
		}
	}
}
