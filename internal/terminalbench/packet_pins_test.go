package terminalbench

import (
	"crypto/sha256"
	"encoding/hex"
	"os"
	"path/filepath"
	"regexp"
	"testing"
)

const submissionPacketPath = "docs/benchmarks/TERMINAL-BENCH-2.1-SUBMISSION-PACKET.md"

var packetPinRow = regexp.MustCompile("(?m)^\\| `([^`]+)` \\|.*\\| `([0-9a-f]{64})` \\|$")

// TestSubmissionPacketHashPinsMatchDisk gates the #898/#902 submission-packet
// index: every artifact the packet hash-pins must match its on-disk bytes, so
// regenerating a contract or preflight artifact cannot silently strand the
// "reproducible from this file alone" claim.
func TestSubmissionPacketHashPinsMatchDisk(t *testing.T) {
	root := filepath.Join("..", "..")
	packet, err := os.ReadFile(filepath.Join(root, filepath.FromSlash(submissionPacketPath)))
	if err != nil {
		t.Fatal(err)
	}
	rows := packetPinRow.FindAllStringSubmatch(string(packet), -1)
	if len(rows) < 8 {
		t.Fatalf("parsed only %d hash-pinned rows from %s; the pin table changed shape", len(rows), submissionPacketPath)
	}
	for _, row := range rows {
		artifactPath, pinned := row[1], row[2]
		raw, err := os.ReadFile(filepath.Join(root, filepath.FromSlash(artifactPath)))
		if err != nil {
			t.Errorf("pinned artifact missing: %v", err)
			continue
		}
		sum := sha256.Sum256(raw)
		if got := hex.EncodeToString(sum[:]); got != pinned {
			t.Errorf("%s: pinned sha256 %s but on-disk bytes hash to %s — refresh the packet index", artifactPath, pinned, got)
		}
	}
}
