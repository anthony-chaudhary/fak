package headroom

import (
	"bytes"
	"strings"
	"testing"
)

func diagnosticClassifier(line []byte) recordLineKind {
	text := string(line)
	switch {
	case strings.HasPrefix(text, "ERROR ") || strings.HasPrefix(text, "panic:"):
		return recordError
	case strings.HasPrefix(text, "PASS "):
		return recordRoutine
	case strings.HasPrefix(text, " ") || strings.HasPrefix(text, "\t") || strings.HasPrefix(text, "at ") || text == "":
		return recordContinuation
	default:
		return recordUnknown
	}
}

func TestGroupDistillRecordsPreservesBytesAcrossShapes(t *testing.T) {
	tests := map[string]string{
		"compiler source caret": "ERROR x.go:2: bad\r\n  value()\r\n  ^^^^^\r\nPASS next\r\n",
		"stack trace":           "panic: sentinel\nat one.go:1\nat two.go:2\nunknown\n",
		"blank boundary":        "ERROR first\n  detail\n\nPASS next\n",
		"interleaved":           "PASS one\nERROR two\n  detail\nPASS three\n",
		"no final newline":      "ERROR last\n  detail",
		"orphan continuation":   "  malformed first\nunknown\n",
	}
	for name, input := range tests {
		t.Run(name, func(t *testing.T) {
			records := groupDistillRecords([]byte(input), 1024, diagnosticClassifier)
			if got := joinDistillRecords(records); !bytes.Equal(got, []byte(input)) {
				t.Fatalf("round trip:\n got %q\nwant %q", got, input)
			}
		})
	}
}

func TestGroupDistillRecordsKeepsErrorContinuationsAtomic(t *testing.T) {
	records := groupDistillRecords([]byte("PASS one\nERROR bad\n  source\n  ^ caret\nPASS two\n"), 1024, diagnosticClassifier)
	if len(records) != 3 {
		t.Fatalf("records = %#v", records)
	}
	if records[1].Kind != recordError || string(records[1].Bytes) != "ERROR bad\n  source\n  ^ caret\n" {
		t.Fatalf("error record = %#v", records[1])
	}
	var kept []byte
	for _, record := range records {
		if record.Kind != recordRoutine || record.ForceKeep {
			kept = append(kept, record.Bytes...)
		}
	}
	if string(kept) != "ERROR bad\n  source\n  ^ caret\n" {
		t.Fatalf("filtered = %q", kept)
	}
}

func TestGroupDistillRecordsOverflowIsBoundedAndForceKept(t *testing.T) {
	input := "ERROR huge\n" + strings.Repeat("  continuation\n", 20) + "PASS done\n"
	records := groupDistillRecords([]byte(input), 40, diagnosticClassifier)
	if !bytes.Equal(joinDistillRecords(records), []byte(input)) {
		t.Fatal("overflow changed bytes")
	}
	forceKept := 0
	for _, record := range records {
		if record.ForceKeep {
			forceKept++
			if len(record.Bytes) > len("  continuation\n") {
				t.Fatalf("overflow chunk grew without bound: %d", len(record.Bytes))
			}
		}
	}
	if forceKept == 0 {
		t.Fatal("expected force-kept overflow chunks")
	}
}

func TestGroupDistillRecordsMutationRoundTrip(t *testing.T) {
	base := []byte("PASS routine\nERROR sentinel\n  detail\nunknown")
	mutants := [][]byte{
		append([]byte("prefix\r\n"), base...),
		append(append([]byte(nil), base...), '\n'),
		bytes.ReplaceAll(base, []byte("\n"), []byte("\r\n")),
		append(append([]byte(nil), base...), bytes.Repeat([]byte("\n  more"), 100)...),
	}
	for i, mutant := range mutants {
		if got := joinDistillRecords(groupDistillRecords(mutant, 64, diagnosticClassifier)); !bytes.Equal(got, mutant) {
			t.Fatalf("mutant %d did not round trip", i)
		}
	}
}
