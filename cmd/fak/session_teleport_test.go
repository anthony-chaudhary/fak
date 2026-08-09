package main

import (
	"bytes"
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/gateway"
	"github.com/anthony-chaudhary/fak/internal/sessionledger"
)

// seedTeleportLedger writes a small hash-chained session into a fresh ledger
// directory and returns the directory and the trace.
func seedTeleportLedger(t *testing.T) (dir, trace string) {
	t.Helper()
	dir = t.TempDir()
	l, err := sessionledger.Open(dir)
	if err != nil {
		t.Fatalf("open ledger: %v", err)
	}
	trace = "sess:cli-teleport"
	for _, e := range []struct{ kind, content string }{
		{sessionledger.KindEstablish, `{"surface":"cli","conversation":"c1"}`},
		{"turn_start", `{"messages":2}`},
		{"turn_complete", `{"output_tokens":91}`},
	} {
		if _, err := l.Append(trace, e.kind, []byte(e.content)); err != nil {
			t.Fatalf("seed %s: %v", e.kind, err)
		}
	}
	return dir, trace
}

// TestSessionTeleportCLIRoundTrip drives `fak session export` on host A's ledger
// directory and `fak session import` on host B's, through the real argv surface.
func TestSessionTeleportCLIRoundTrip(t *testing.T) {
	hostA, trace := seedTeleportLedger(t)
	hostB := t.TempDir()
	bundlePath := filepath.Join(t.TempDir(), "bundle.json")

	var out, errOut bytes.Buffer
	if rc := runSession(&out, &errOut, []string{
		"export", trace, "--ledger-dir", hostA, "--out", bundlePath,
		"--turns", "3", "--tokens", "5000", "--taint", "tainted",
	}); rc != 0 {
		t.Fatalf("export rc=%d stderr=%s", rc, errOut.String())
	}
	if !strings.Contains(out.String(), "exported "+trace) {
		t.Fatalf("export stdout = %q", out.String())
	}

	raw, err := os.ReadFile(bundlePath)
	if err != nil {
		t.Fatalf("read bundle: %v", err)
	}
	var bundle gateway.TeleportBundle
	if err := json.Unmarshal(raw, &bundle); err != nil {
		t.Fatalf("decode bundle: %v", err)
	}
	if bundle.TraceID != trace || len(bundle.Entries) != 3 {
		t.Fatalf("bundle = %q / %d entries", bundle.TraceID, len(bundle.Entries))
	}

	// Import on host B, reading the bundle from stdin to exercise that path.
	out.Reset()
	errOut.Reset()
	if rc := runTeleportImport(bytes.NewReader(raw), &out, &errOut, []string{"--ledger-dir", hostB}); rc != 0 {
		t.Fatalf("import rc=%d stderr=%s", rc, errOut.String())
	}
	if !strings.Contains(out.String(), "imported "+trace) ||
		!strings.Contains(out.String(), "3 re-derived") ||
		!strings.Contains(out.String(), "taint=tainted") {
		t.Fatalf("import stdout = %q", out.String())
	}

	// Host B now holds the same head as host A.
	lb, err := sessionledger.Open(hostB)
	if err != nil {
		t.Fatalf("reopen host B: %v", err)
	}
	if lb.Head(trace) != bundle.Head {
		t.Fatalf("host B head = %s, want %s", lb.Head(trace), bundle.Head)
	}

	// A tampered bundle is refused, and host B's ledger is left alone.
	bundle.Arm.Budget.TokensLeft = 1 << 30
	bad, err := json.Marshal(bundle)
	if err != nil {
		t.Fatalf("marshal tampered: %v", err)
	}
	hostC := t.TempDir()
	out.Reset()
	errOut.Reset()
	if rc := runTeleportImport(bytes.NewReader(bad), &out, &errOut, []string{"--ledger-dir", hostC}); rc == 0 {
		t.Fatalf("import accepted a tampered bundle")
	}
	if !strings.Contains(errOut.String(), "refused") {
		t.Fatalf("import stderr = %q, want a refusal", errOut.String())
	}
	lc, err := sessionledger.Open(hostC)
	if err != nil {
		t.Fatalf("open host C: %v", err)
	}
	if lc.Head(trace) != "" {
		t.Fatalf("a refused import still wrote to the receiving host")
	}
}

// TestSessionTeleportCLIFork pins issue #2419's CLI acceptance: `fak session fork
// <trace>` prints the new trace id and the shared-prefix hash.
func TestSessionTeleportCLIFork(t *testing.T) {
	dir, trace := seedTeleportLedger(t)
	l, err := sessionledger.Open(dir)
	if err != nil {
		t.Fatalf("open: %v", err)
	}
	head := string(l.Head(trace))

	var out, errOut bytes.Buffer
	if rc := runSession(&out, &errOut, []string{"fork", trace, "--ledger-dir", dir}); rc != 0 {
		t.Fatalf("fork rc=%d stderr=%s", rc, errOut.String())
	}
	got := out.String()
	if !strings.Contains(got, "forked "+trace+" -> ") {
		t.Fatalf("fork stdout = %q, want the new trace id", got)
	}
	if !strings.Contains(got, "shared prefix: "+head) {
		t.Fatalf("fork stdout = %q, want shared prefix %s", got, head)
	}

	// The minted id is a real, usable trace on the ledger.
	var forkID string
	for _, line := range strings.Split(got, "\n") {
		if _, rest, ok := strings.Cut(line, " -> "); ok {
			forkID = strings.TrimSpace(rest)
		}
	}
	l2, err := sessionledger.Open(dir)
	if err != nil {
		t.Fatalf("reopen: %v", err)
	}
	if forkID == "" || string(l2.Head(forkID)) != head {
		t.Fatalf("fork %q head = %q, want %s", forkID, l2.Head(forkID), head)
	}
}

// TestSessionForkVerbRouting pins the shape-based split between the two forks that
// share this verb, so a later edit cannot silently re-route one onto the other.
func TestSessionForkVerbRouting(t *testing.T) {
	cases := []struct {
		name   string
		argv   []string
		ledger bool
	}{
		{"bare trace is the ledger fork", []string{"sess:x"}, true},
		{"trace with --to is the ledger fork", []string{"sess:x", "--to", "sess:y"}, true},
		{"--out names the image fork", []string{"dir", "--out", "d2", "--checkpoint", "d3"}, false},
		{"single-dash -out names the image fork", []string{"dir", "-out", "d2"}, false},
		{"--out=value names the image fork", []string{"dir", "--out=d2"}, false},
		{"--checkpoint alone names the image fork", []string{"dir", "--checkpoint", "d3"}, false},
		{"no positional is not the ledger fork", []string{"--out", "d2"}, false},
		{"empty is not the ledger fork", nil, false},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			if got := teleportIsLedgerFork(tc.argv); got != tc.ledger {
				t.Fatalf("teleportIsLedgerFork(%q) = %v, want %v", tc.argv, got, tc.ledger)
			}
		})
	}

	// The image fork's own contract is unchanged: a bare directory with --out but no
	// --checkpoint is still its usage error, not a ledger fork.
	var out, errOut bytes.Buffer
	if rc := runSession(&out, &errOut, []string{"fork", "somedir", "--out", t.TempDir()}); rc != 2 {
		t.Fatalf("image fork without --checkpoint rc=%d, want 2; stderr=%s", rc, errOut.String())
	}
	if !strings.Contains(errOut.String(), "--checkpoint") {
		t.Fatalf("stderr = %q, want the image fork's own usage error", errOut.String())
	}
}
