// relay_handoff_test.go — the witness for rung C6 (#1875): `go test ./cmd/fak -run
// RelayHandoff` asserts the offline baton writer produces a valid, DETERMINISTIC baton
// from stated leg state, that the bytes round-trip through the resume reader, and that a
// baton missing its two load-bearing pointers (relay id, start sha) is refused as a usage
// error rather than written. The determinism assertion is the issue's Done condition made
// executable: the same flags must yield byte-identical baton bytes across runs.
package main

import (
	"bytes"
	"io"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/relay"
)

// baseHandoffArgs is a complete, valid handoff flag set reused across the subtests so a
// determinism assertion compares like against like. It exercises every repeatable flag
// (artifact, held-region, open-question, do-not-rederive) so the projected slice order is
// under test too.
func baseHandoffArgs(extra ...string) []string {
	return append([]string{
		"--relay-id", "RID-2026-07-06-handoff-witness",
		"--leg", "2",
		"--parent-trace", "trace-leg2",
		"--objective", "Ship the offline baton writer (#1875).",
		"--done-when", "go test ./cmd/fak -run RelayHandoff is green.",
		"--next-action", "resume: re-render the prompt and run the witness.",
		"--start-sha", "0123456789abcdef0123456789abcdef01234567",
		"--ledger-ref", "intent:RID-2026-07-06-handoff-witness/leg2",
		"--tombstone-reason", "RELAY_ROTATED",
		"--tombstone-note", "context ceiling",
		"--artifact", "commit:0123456789abcdef0123456789abcdef01234567",
		"--artifact", "issue:#1875",
		"--held-region", "cmd/fak/**",
		"--open-question", "does the successor re-anchor on the same sha?",
		"--do-not-rederive", "issue:#1874",
	}, extra...)
}

// TestRelayHandoffWritesValidBaton drives runRelayHandoff to stdout and asserts the emitted
// bytes parse back into a well-formed baton carrying exactly the stated leg state — the
// core "writes a valid baton" half of the Done condition.
func TestRelayHandoffWritesValidBaton(t *testing.T) {
	var stdout, stderr bytes.Buffer
	if code := runRelayHandoff(&stdout, &stderr, baseHandoffArgs()); code != 0 {
		t.Fatalf("exit = %d, want 0 (stderr: %s)", code, stderr.String())
	}
	b, err := relay.Parse(stdout.Bytes())
	if err != nil {
		t.Fatalf("emitted bytes did not parse as a baton: %v\n%s", err, stdout.String())
	}
	if b.Schema != relay.Schema {
		t.Fatalf("schema = %q, want %s", b.Schema, relay.Schema)
	}
	if b.RelayID != "RID-2026-07-06-handoff-witness" {
		t.Fatalf("relay_id = %q", b.RelayID)
	}
	if b.Leg != 2 {
		t.Fatalf("leg = %d, want 2", b.Leg)
	}
	if b.ProgressCursor.StartSHA != "0123456789abcdef0123456789abcdef01234567" {
		t.Fatalf("start_sha = %q", b.ProgressCursor.StartSHA)
	}
	// The tombstone's observed sha defaults to the progress anchor when not separately
	// stated — the closing leg stood at its start sha.
	if b.Tombstone.AtSHA != b.ProgressCursor.StartSHA {
		t.Fatalf("tombstone at_sha = %q, want default to start_sha %q", b.Tombstone.AtSHA, b.ProgressCursor.StartSHA)
	}
	if b.Tombstone.Reason != "RELAY_ROTATED" {
		t.Fatalf("tombstone reason = %q", b.Tombstone.Reason)
	}
	// The objective pin's digest is computed, not stated: a non-empty digest proves the
	// verb ran NewObjectivePin rather than hand-setting the pin.
	if strings.TrimSpace(b.Objective.Digest) == "" {
		t.Fatalf("objective digest empty — pin was not computed via NewObjectivePin")
	}
	if got := []string{"commit", "issue"}; len(b.Artifacts) != 2 || b.Artifacts[0].Kind != got[0] || b.Artifacts[1].Kind != got[1] {
		t.Fatalf("artifacts = %+v, want commit then issue in flag order", b.Artifacts)
	}
	if len(b.ProgressCursor.HeldRegion) != 1 || b.ProgressCursor.HeldRegion[0] != "cmd/fak/**" {
		t.Fatalf("held_region = %+v", b.ProgressCursor.HeldRegion)
	}
}

// TestRelayHandoffDeterministic is the load-bearing witness: the same flags must produce
// byte-identical baton bytes across two independent runs (the relay's determinism
// contract — same input, same wire, so a rotation is reproducible and auditable).
func TestRelayHandoffDeterministic(t *testing.T) {
	run := func() []byte {
		var stdout, stderr bytes.Buffer
		if code := runRelayHandoff(&stdout, &stderr, baseHandoffArgs()); code != 0 {
			t.Fatalf("exit = %d, want 0 (stderr: %s)", code, stderr.String())
		}
		return append([]byte(nil), stdout.Bytes()...)
	}
	a, b := run(), run()
	if !bytes.Equal(a, b) {
		t.Fatalf("baton bytes not deterministic across runs:\n a=%s\n b=%s", a, b)
	}
}

// TestRelayHandoffRoundTripsThroughResume writes a baton to --out and reads it back with
// runRelayResume --json, proving the write half and the read half agree on the canonical
// wire form (relay.Marshal(relay.Parse(x)) == x).
func TestRelayHandoffRoundTripsThroughResume(t *testing.T) {
	dir := t.TempDir()
	out := filepath.Join(dir, "sub", "leg2.baton.json") // nested so the mkdir path is exercised

	var wStdout, wStderr bytes.Buffer
	if code := runRelayHandoff(&wStdout, &wStderr, baseHandoffArgs("--out", out)); code != 0 {
		t.Fatalf("handoff exit = %d, want 0 (stderr: %s)", code, wStderr.String())
	}
	if !strings.Contains(wStdout.String(), out) {
		t.Fatalf("handoff did not echo the written path; stdout=%q", wStdout.String())
	}
	onDisk, err := os.ReadFile(out)
	if err != nil {
		t.Fatalf("baton file not written: %v", err)
	}

	// resume --json re-marshals the parsed file; its output (minus the trailing newline
	// resume prints) must equal the canonical bytes on disk (minus the newline handoff
	// appended when writing a file).
	var rStdout, rStderr bytes.Buffer
	if code := runRelayResume(strings.NewReader(""), &rStdout, &rStderr, []string{"--baton", out, "--json"}); code != 0 {
		t.Fatalf("resume exit = %d, want 0 (stderr: %s)", code, rStderr.String())
	}
	got := strings.TrimRight(rStdout.String(), "\n")
	want := strings.TrimRight(string(onDisk), "\n")
	if got != want {
		t.Fatalf("resume --json did not round-trip the written baton:\n on-disk: %s\n resume:  %s", want, got)
	}
}

// TestRelayHandoffPipeRoundTrip proves the pipe composition the usage advertises:
// handoff to stdout, parsed back, re-marshaled — byte-identical.
func TestRelayHandoffPipeRoundTrip(t *testing.T) {
	var stdout, stderr bytes.Buffer
	if code := runRelayHandoff(&stdout, &stderr, baseHandoffArgs()); code != 0 {
		t.Fatalf("exit = %d, want 0 (stderr: %s)", code, stderr.String())
	}
	piped := strings.TrimRight(stdout.String(), "\n")

	var rStdout, rStderr bytes.Buffer
	if code := runRelayResume(strings.NewReader(piped), &rStdout, &rStderr, []string{"--baton", "-", "--json"}); code != 0 {
		t.Fatalf("resume exit = %d, want 0 (stderr: %s)", code, rStderr.String())
	}
	if got := strings.TrimRight(rStdout.String(), "\n"); got != piped {
		t.Fatalf("pipe did not round-trip:\n in:  %s\n out: %s", piped, got)
	}
}

// TestRelayHandoffRejectsMissingRequired asserts the two load-bearing pointers are
// enforced: a baton with no relay id or no start sha is a usage error (exit 2), never a
// written file — the successor keys the read on the relay id and re-verifies the sha, so
// neither may be absent.
func TestRelayHandoffRejectsMissingRequired(t *testing.T) {
	cases := []struct {
		name string
		argv []string
		want string
	}{
		{"no relay-id", []string{"--start-sha", "deadbeef"}, "--relay-id is required"},
		{"no start-sha", []string{"--relay-id", "RID-x"}, "--start-sha is required"},
		{"negative leg", []string{"--relay-id", "RID-x", "--start-sha", "deadbeef", "--leg", "-1"}, "--leg must be >= 0"},
		{"bad artifact", []string{"--relay-id", "RID-x", "--start-sha", "deadbeef", "--artifact", "noColon"}, "kind:ref"},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			var stdout, stderr bytes.Buffer
			code := runRelayHandoff(&stdout, &stderr, tc.argv)
			if code != 2 {
				t.Fatalf("exit = %d, want 2 (stdout=%q stderr=%q)", code, stdout.String(), stderr.String())
			}
			if stdout.Len() != 0 {
				t.Fatalf("wrote to stdout on a usage error: %q", stdout.String())
			}
			if !strings.Contains(stderr.String(), tc.want) {
				t.Fatalf("stderr = %q, want it to contain %q", stderr.String(), tc.want)
			}
		})
	}
}

// TestRelayHandoffDispatchViaRunRelay is the registration witness for rung C6 (#1875):
// `fak relay handoff` must be reachable through the `fak relay` dispatcher, not only via
// the runRelayHandoff shell. It drives runRelay with the "handoff" verb and asserts the
// emitted bytes are the same valid baton the direct shell writes, and that the verb is
// named in both the unknown-subcommand hint and the usage text.
func TestRelayHandoffDispatchViaRunRelay(t *testing.T) {
	var stdout, stderr bytes.Buffer
	argv := append([]string{"handoff"}, baseHandoffArgs()...)
	if code := runRelay(strings.NewReader(""), &stdout, &stderr, argv); code != 0 {
		t.Fatalf("runRelay handoff exit = %d, want 0 (stderr: %s)", code, stderr.String())
	}
	b, err := relay.Parse(stdout.Bytes())
	if err != nil {
		t.Fatalf("dispatched handoff did not emit a parseable baton: %v\n%s", err, stdout.String())
	}
	if b.Schema != relay.Schema || b.RelayID != "RID-2026-07-06-handoff-witness" {
		t.Fatalf("dispatched baton = schema %q relay_id %q", b.Schema, b.RelayID)
	}

	// The dispatched bytes must be byte-identical to the direct shell's — dispatch adds
	// no wrapping, so the determinism contract holds through the verb too.
	var direct bytes.Buffer
	if code := runRelayHandoff(&direct, io.Discard, baseHandoffArgs()); code != 0 {
		t.Fatalf("direct runRelayHandoff exit = %d, want 0", code)
	}
	if !bytes.Equal(stdout.Bytes(), direct.Bytes()) {
		t.Fatalf("dispatched bytes differ from direct shell bytes:\n dispatch: %s\n direct:   %s", stdout.Bytes(), direct.Bytes())
	}

	// The verb is a first-class citizen of the dispatcher's surface: the unknown-verb
	// hint and the usage text both name it.
	var uStdout, uStderr bytes.Buffer
	if code := runRelay(strings.NewReader(""), &uStdout, &uStderr, []string{"nonesuch"}); code != 2 {
		t.Fatalf("unknown verb exit = %d, want 2", code)
	}
	if !strings.Contains(uStderr.String(), "handoff") {
		t.Fatalf("unknown-verb hint does not name handoff: %q", uStderr.String())
	}
	var hStdout, hStderr bytes.Buffer
	if code := runRelay(strings.NewReader(""), &hStdout, &hStderr, []string{"help"}); code != 0 {
		t.Fatalf("help exit = %d, want 0 (stderr: %s)", code, hStderr.String())
	}
	if !strings.Contains(hStdout.String(), "fak relay handoff") {
		t.Fatalf("usage does not document the handoff verb:\n%s", hStdout.String())
	}
}
