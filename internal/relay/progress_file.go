// Rung D3 follow-on (issue #1879): the file-backed LedgerReader production wiring.
// progress.go builds the ledger-verified progress READ (ReadVerifiedProgress over an
// injected LedgerReader) and states plainly that "wiring a file-backed LedgerReader
// into the live reload/driver path is a later rung ... a file-backed reader that reuses
// ParseLedgerProgress over the durable ledger is the production wiring." This file is
// that reader: it turns a cursor's ledger_ref into progress read FROM a durable
// run/intent ledger file, so a successor (or a peer over the A2A edge, gateway
// a2a_progress.go) can read the predecessor's progress as the ledger actually records
// it — never a self-report.
//
// Same discipline as resolve.go's CommitResolver: the store read is INJECTED (a probe
// returning the ledger bytes) so the reader is unit-testable without disk, and
// OSFileLedger provides the production wiring rooted at a ledger directory — exactly
// mirroring NewCommitResolver(exists) + GitCommitExists(dir).
//
// Fail-closed, and contained. The progress.go contract is that a reader error means the
// ledger is unreachable (-> ProgressUnknown, never a false "zero progress"), while a
// successful read of a rowless ledger is the honest empty answer (ProgressVerified with
// no steps). This reader honors both: a missing or unreadable file surfaces as an error
// (unknown), an existing-but-empty ledger reads as verified-empty. And because a
// carried baton's ledger_ref crosses an agent boundary, OSFileLedger CONTAINS the ref
// under the ledger root — an absolute path or a "../" traversal is refused, not read
// (the A2A inter-agent transport attack surface, issue #87), so a peer-supplied cursor
// can never read a file outside the ledger directory.
package relay

import (
	"errors"
	"os"
	"path/filepath"
	"strings"
)

// FileLedgerReader implements LedgerReader by reading the run/intent ledger FILE a
// cursor's ledger_ref names and projecting its JSONL through ParseLedgerProgress. The
// file read is injected (read) so the reader is hermetic in tests, like CommitResolver's
// injected exists probe; OSFileLedger is the production wiring.
type FileLedgerReader struct {
	// read returns the raw bytes of the ledger named by ledgerRef. A non-nil error means
	// the ledger is unreachable/absent/refused and maps to ProgressUnknown (fail closed);
	// a nil error with empty bytes is a real, rowless ledger (verified-empty).
	read func(ledgerRef string) ([]byte, error)
}

// NewFileLedgerReader builds a FileLedgerReader over an injected ledger-bytes probe.
func NewFileLedgerReader(read func(ledgerRef string) ([]byte, error)) FileLedgerReader {
	return FileLedgerReader{read: read}
}

// ReadProgress reads the ledger bytes for ledgerRef through the injected probe and
// projects them with ParseLedgerProgress (the same projection progress.go and the
// hermetic test reader use, so the production path and the contract test agree). It is
// fail-closed to match the LedgerReader contract: a probe error propagates unchanged so
// ReadVerifiedProgress reports ProgressUnknown rather than a trusted absence; a
// successful read returns exactly the ledger's rows (empty when the ledger has none).
func (r FileLedgerReader) ReadProgress(ledgerRef string) ([]ProgressStep, error) {
	if r.read == nil {
		return nil, errors.New("file ledger reader has no read probe configured")
	}
	b, err := r.read(ledgerRef)
	if err != nil {
		return nil, err
	}
	return ParseLedgerProgress(string(b)), nil
}

// OSFileLedger returns a ledger-bytes probe backed by the ledger directory rooted at
// dir. A cursor's ledger_ref is a repo-relative path (".dos/runs/relay-demo.jsonl" in
// the D3 tests); the probe contains it under dir and reads the file. Both a refused ref
// (absolute or escaping dir) and a missing/unreadable file surface as an error, which
// ReadProgress maps to the fail-closed ProgressUnknown — an out-of-tree or absent ledger
// is never reported as verified-zero-progress.
func OSFileLedger(dir string) func(ledgerRef string) ([]byte, error) {
	return func(ledgerRef string) ([]byte, error) {
		p, err := containedLedgerPath(dir, ledgerRef)
		if err != nil {
			return nil, err
		}
		return os.ReadFile(p)
	}
}

// containedLedgerPath resolves a repo-relative ledger_ref under root and PROVES the
// result stays within root before returning it. An empty ref names no file; an absolute
// ref, or one whose cleaned form climbs above root via "..", is refused with an error
// (never silently clamped) so the caller fails closed. The check is lexical over
// cleaned, separator-normalized paths (filepath.Join cleans, filepath.Rel reports the
// climb), so it is deterministic and touches no filesystem.
func containedLedgerPath(root, ref string) (string, error) {
	if ref == "" {
		return "", errors.New("empty ledger_ref names no file")
	}
	if filepath.IsAbs(ref) {
		return "", errors.New("ledger_ref must be repo-relative, not absolute: " + ref)
	}
	joined := filepath.Join(root, ref)
	rel, err := filepath.Rel(root, joined)
	if err != nil {
		return "", err
	}
	if rel == ".." || strings.HasPrefix(rel, ".."+string(filepath.Separator)) {
		return "", errors.New("ledger_ref escapes the ledger root: " + ref)
	}
	return joined, nil
}
