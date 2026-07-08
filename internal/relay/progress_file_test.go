package relay

import (
	"errors"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"testing"
)

// D3 follow-on (issue #1879) witnesses for the file-backed LedgerReader: the production
// wiring reads verified progress FROM a durable ledger file, fails closed on an
// unreachable/absent ledger, and contains a peer-supplied ledger_ref under the ledger
// root (run: `go test ./internal/relay -run FileLedger`).

const fileLedgerJSONL = `{"ref":"0123456789abcdef0123456789abcdef01234567","note":"committed schema"}
{"ref":"#1877","note":"D1 reload verifier landed"}
{"note":"no ref — dropped"}`

// TestFileLedgerReaderProjectsThroughProbe drives the injected-probe path: the reader
// projects exactly what ParseLedgerProgress yields for the bytes the probe returns, so a
// FileLedgerReader over a hermetic probe reads the same steps the D3 contract test pins.
func TestFileLedgerReaderProjectsThroughProbe(t *testing.T) {
	reader := NewFileLedgerReader(func(ref string) ([]byte, error) {
		if ref != "ledger.jsonl" {
			return nil, errors.New("no ledger at " + ref)
		}
		return []byte(fileLedgerJSONL), nil
	})
	got := ReadVerifiedProgress(ProgressCursor{StartSHA: "HEAD", LedgerRef: "ledger.jsonl"}, reader)
	if got.Verdict != ProgressVerified {
		t.Fatalf("verdict = %q, want verified (reason=%s)", got.Verdict, got.Reason)
	}
	want := ParseLedgerProgress(fileLedgerJSONL)
	if !reflect.DeepEqual(got.Steps, want) {
		t.Errorf("steps did not match the ledger projection:\n got  = %+v\n want = %+v", got.Steps, want)
	}
}

// TestFileLedgerReaderFailsClosedOnProbeError pins that a probe error (an unreachable or
// refused ledger) propagates so ReadVerifiedProgress reports ProgressUnknown with no
// steps — never a trusted "zero progress" from an absence the reader could not check.
func TestFileLedgerReaderFailsClosedOnProbeError(t *testing.T) {
	reader := NewFileLedgerReader(func(string) ([]byte, error) {
		return nil, errors.New("ledger store unreachable")
	})
	got := ReadVerifiedProgress(ProgressCursor{LedgerRef: ".dos/runs/x.jsonl"}, reader)
	if got.Verdict != ProgressUnknown || len(got.Steps) != 0 {
		t.Errorf("probe error must fail closed: verdict=%q steps=%v (reason=%s)", got.Verdict, got.Steps, got.Reason)
	}
	// A reader built with no probe also fails closed rather than panicking.
	if _, err := (FileLedgerReader{}).ReadProgress("anything"); err == nil {
		t.Error("nil-probe reader must return an error, not a nil-deref")
	}
}

// TestOSFileLedgerReadsRealFile is the end-to-end production wiring: a real ledger file
// under a temp ledger root reads back through OSFileLedger -> FileLedgerReader ->
// ReadVerifiedProgress as verified steps, exactly the file's rows in order.
func TestOSFileLedgerReadsRealFile(t *testing.T) {
	dir := t.TempDir()
	ref := filepath.Join(".dos", "runs", "relay-demo.jsonl")
	full := filepath.Join(dir, ref)
	if err := os.MkdirAll(filepath.Dir(full), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(full, []byte(fileLedgerJSONL), 0o644); err != nil {
		t.Fatal(err)
	}
	reader := NewFileLedgerReader(OSFileLedger(dir))
	got := ReadVerifiedProgress(ProgressCursor{StartSHA: "HEAD", LedgerRef: ref}, reader)
	if got.Verdict != ProgressVerified {
		t.Fatalf("verdict = %q, want verified (reason=%s)", got.Verdict, got.Reason)
	}
	if !reflect.DeepEqual(got.Steps, ParseLedgerProgress(fileLedgerJSONL)) {
		t.Errorf("steps = %+v, want the file's rows", got.Steps)
	}
}

// TestOSFileLedgerMissingFileIsUnknown pins the fail-closed edge for the production
// wiring: a cursor naming a ledger that does not exist on this store reads as
// ProgressUnknown (the ledger could not be verified), never verified-zero.
func TestOSFileLedgerMissingFileIsUnknown(t *testing.T) {
	reader := NewFileLedgerReader(OSFileLedger(t.TempDir()))
	got := ReadVerifiedProgress(ProgressCursor{LedgerRef: "never-written.jsonl"}, reader)
	if got.Verdict != ProgressUnknown || len(got.Steps) != 0 {
		t.Errorf("missing ledger must be unknown: verdict=%q steps=%v (reason=%s)", got.Verdict, got.Steps, got.Reason)
	}
}

// TestOSFileLedgerEmptyLedgerIsVerifiedEmpty distinguishes an existing-but-rowless
// ledger (read succeeded, nothing recorded yet) from a missing one: the former is the
// honest verified-empty answer, the latter is unknown. A false "verified" here would be
// the same self-report the read exists to keep out, so this boundary is load-bearing.
func TestOSFileLedgerEmptyLedgerIsVerifiedEmpty(t *testing.T) {
	dir := t.TempDir()
	if err := os.WriteFile(filepath.Join(dir, "empty.jsonl"), []byte("\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	reader := NewFileLedgerReader(OSFileLedger(dir))
	got := ReadVerifiedProgress(ProgressCursor{StartSHA: "HEAD", LedgerRef: "empty.jsonl"}, reader)
	if got.Verdict != ProgressVerified {
		t.Fatalf("existing rowless ledger should read verified: verdict=%q (reason=%s)", got.Verdict, got.Reason)
	}
	if len(got.Steps) != 0 {
		t.Errorf("rowless ledger should have no steps, got %+v", got.Steps)
	}
}

// TestOSFileLedgerContainsTraversal is the A2A transport-attack witness (issue #87): a
// peer-supplied ledger_ref that escapes the ledger root — via "../" or an absolute path
// — is refused (fail closed to ProgressUnknown) and never reads the out-of-root file,
// even when such a file really exists.
func TestOSFileLedgerContainsTraversal(t *testing.T) {
	root := t.TempDir()
	// A secret ledger the peer must NOT be able to read, one level above the root.
	secret := filepath.Join(filepath.Dir(root), "secret.jsonl")
	if err := os.WriteFile(secret, []byte(fileLedgerJSONL), 0o644); err != nil {
		t.Fatal(err)
	}
	defer os.Remove(secret)

	probe := OSFileLedger(root)
	for _, ref := range []string{
		"../secret.jsonl",
		filepath.Join("..", "secret.jsonl"),
		"a/../../secret.jsonl",
		secret, // absolute
		"",
	} {
		if _, err := probe(ref); err == nil {
			t.Errorf("ref %q escaped containment (should be refused)", ref)
		}
	}
	// And end-to-end: a traversal ledger_ref fails closed to unknown, not verified.
	reader := NewFileLedgerReader(probe)
	got := ReadVerifiedProgress(ProgressCursor{LedgerRef: "../secret.jsonl"}, reader)
	if got.Verdict != ProgressUnknown || len(got.Steps) != 0 {
		t.Errorf("traversal ref must fail closed: verdict=%q steps=%v", got.Verdict, got.Steps)
	}
	// A legitimate in-root ledger with the same basename still reads fine (containment
	// refuses the climb, not the name).
	if err := os.WriteFile(filepath.Join(root, "secret.jsonl"), []byte(fileLedgerJSONL), 0o644); err != nil {
		t.Fatal(err)
	}
	if b, err := probe("secret.jsonl"); err != nil || !strings.Contains(string(b), "#1877") {
		t.Errorf("in-root ledger should read: err=%v", err)
	}
}
