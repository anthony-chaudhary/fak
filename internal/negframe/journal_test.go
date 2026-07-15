package negframe

import (
	"bufio"
	"encoding/json"
	"os"
	"path/filepath"
	"testing"
	"time"
)

func TestReframeJournalRecordsTreatmentAndControlExactly(t *testing.T) {
	text := "Do not forget to recover. Do not delete `LOCK`."
	treatment := ReframePass(text)
	control := ReframeResult{Text: text, ResidualNegatives: len(Classify("runtime", text))}
	path := filepath.Join(t.TempDir(), "journal.jsonl")
	for _, row := range []ReframeJournalRow{
		NewReframeJournalRow("t", "treatment", treatment, time.UnixMilli(10)),
		NewReframeJournalRow("c", "control", control, time.UnixMilli(11)),
	} {
		if err := AppendReframeJournal(path, row, 8); err != nil {
			t.Fatal(err)
		}
	}
	rows := readJournalRows(t, path)
	if len(rows) != 2 {
		t.Fatalf("rows=%d", len(rows))
	}
	if rows[0].Arm != "treatment" || rows[0].Applied != treatment.Applied || rows[0].VerbatimFallback != treatment.VerbatimFallback || rows[0].ResidualNegatives != treatment.ResidualNegatives {
		t.Fatalf("treatment=%+v result=%+v", rows[0], treatment)
	}
	if rows[1].Arm != "control" || rows[1].Applied != 0 || rows[1].ResidualNegatives != control.ResidualNegatives {
		t.Fatalf("control=%+v", rows[1])
	}
}

func TestReframeJournalBoundedAndContentFree(t *testing.T) {
	path := filepath.Join(t.TempDir(), "j.jsonl")
	for i := 0; i < 5; i++ {
		if err := AppendReframeJournal(path, NewReframeJournalRow("trace", "treatment", ReframeResult{Applied: i}, time.UnixMilli(int64(i))), 3); err != nil {
			t.Fatal(err)
		}
	}
	rows := readJournalRows(t, path)
	if len(rows) != 3 || rows[0].Applied != 2 || rows[2].Applied != 4 {
		t.Fatalf("bounded rows=%+v", rows)
	}
	b, _ := os.ReadFile(path)
	if string(b) == "" || contains(string(b), "Do not") {
		t.Fatalf("journal leaked prose: %q", b)
	}
}

func readJournalRows(t *testing.T, path string) []ReframeJournalRow {
	t.Helper()
	f, err := os.Open(path)
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()
	var rows []ReframeJournalRow
	s := bufio.NewScanner(f)
	for s.Scan() {
		var row ReframeJournalRow
		if err := json.Unmarshal(s.Bytes(), &row); err != nil {
			t.Fatal(err)
		}
		rows = append(rows, row)
	}
	return rows
}
func contains(s, sub string) bool {
	for i := 0; i+len(sub) <= len(s); i++ {
		if s[i:i+len(sub)] == sub {
			return true
		}
	}
	return false
}
