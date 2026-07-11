package terminalbench

import (
	"bytes"
	"crypto/sha256"
	"encoding/hex"
	"os"
	"path/filepath"
	"regexp"
	"testing"
)

const submissionPacketPath = "docs/benchmarks/TERMINAL-BENCH-2.1-SUBMISSION-PACKET.md"

var packetPinRow = regexp.MustCompile("(?m)^\\| `([^`]+)` \\|.*\\| `([0-9a-f]{64})` \\|$")

// packetArtifactSHA256 hashes the repository's canonical LF representation.
// Git stores text artifacts with LF line endings, but a Windows checkout may
// materialize them as CRLF. Checkout representation must not invalidate a pin;
// any actual content change still changes the digest.
func packetArtifactSHA256(raw []byte) string {
	canonical := bytes.ReplaceAll(raw, []byte("\r\n"), []byte("\n"))
	sum := sha256.Sum256(canonical)
	return hex.EncodeToString(sum[:])
}

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
	packet = bytes.ReplaceAll(packet, []byte("\r\n"), []byte("\n"))
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
		if got := packetArtifactSHA256(raw); got != pinned {
			t.Errorf("%s: pinned sha256 %s but canonical content hashes to %s — refresh the packet index", artifactPath, pinned, got)
		}
	}
}

func TestPacketArtifactSHA256CanonicalizesOnlyLineEndings(t *testing.T) {
	lf := []byte("alpha\nbeta\n")
	crlf := []byte("alpha\r\nbeta\r\n")
	changed := []byte("alpha\ngamma\n")

	want := packetArtifactSHA256(lf)
	if got := packetArtifactSHA256(crlf); got != want {
		t.Fatalf("CRLF checkout hash = %s, want canonical LF hash %s", got, want)
	}
	if got := packetArtifactSHA256(changed); got == want {
		t.Fatalf("real content change retained hash %s", got)
	}
}
