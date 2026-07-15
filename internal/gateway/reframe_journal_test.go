package gateway

import (
	"bufio"
	"encoding/json"
	"os"
	"path/filepath"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/negframe"
)

func TestReframeJournalGatewayTreatmentAndControl(t *testing.T) {
	path := filepath.Join(t.TempDir(), "j.jsonl")
	text := "Do not forget to recover."
	if got := journalReframePass(path, "t", "treatment", text, time.UnixMilli(1)); got == text {
		t.Fatalf("treatment did not reframe: %q", got)
	}
	if got := journalReframePass(path, "c", "control", text, time.UnixMilli(2)); got != text {
		t.Fatalf("control changed text: %q", got)
	}
	f, err := os.Open(path)
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()
	var rows []negframe.ReframeJournalRow
	s := bufio.NewScanner(f)
	for s.Scan() {
		var r negframe.ReframeJournalRow
		if err := json.Unmarshal(s.Bytes(), &r); err != nil {
			t.Fatal(err)
		}
		rows = append(rows, r)
	}
	if len(rows) != 2 || rows[0].Arm != "treatment" || rows[0].Applied != 1 || rows[1].Arm != "control" || rows[1].Applied != 0 {
		t.Fatalf("rows=%+v", rows)
	}
}

func TestReframeJournalFragmentsPreserveOpaqueAndArm(t *testing.T) {
	path := filepath.Join(t.TempDir(), "j.jsonl")
	fragments := []negframe.Fragment{negframe.Fak("Do not forget to recover: "), negframe.Opaque("Do not mutate USER")}
	got := journalReframeFragments(path, "t", "treatment", fragments, time.UnixMilli(3))
	if got != "remember to recover: Do not mutate USER" {
		t.Fatalf("treatment=%q", got)
	}
	got = journalReframeFragments(path, "c", "control", fragments, time.UnixMilli(4))
	if got != "Do not forget to recover: Do not mutate USER" {
		t.Fatalf("control=%q", got)
	}
}
