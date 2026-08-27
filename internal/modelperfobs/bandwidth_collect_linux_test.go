//go:build linux

package modelperfobs

import "testing"

func TestParseProcSelfStatFaultsHandlesSpacesInComm(t *testing.T) {
	line := "123 (fak worker) R 1 2 3 4 5 6 17 8 19 10 0 0"
	minor, major, ok := parseProcSelfStatFaults(line)
	if !ok || minor != 17 || major != 19 {
		t.Fatalf("minor=%d major=%d ok=%v", minor, major, ok)
	}
}

func TestParseProcSelfStatFaultsRejectsMalformed(t *testing.T) {
	if _, _, ok := parseProcSelfStatFaults("123 malformed"); ok {
		t.Fatal("expected malformed row rejection")
	}
}
