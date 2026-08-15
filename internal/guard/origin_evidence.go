package guard

// origin_evidence.go is the LOCATION side of the guarded child's origin evidence: the
// durable paths `fak guard` hands the process task manager the moment it seeds the
// child (#1965), before any producer has appended its first row.
//
// What these paths satisfy is a LOCATION contract, not a content one —
// taskmgr.PathWitness reads a "path" EvidenceRef back with a bare os.Stat
// (internal/taskmgr/evidence.go) and nothing in the tree parses the bytes. So each
// location is proven with a 0-byte placeholder. The policy location used to be the one
// exception: it re-materialized the embedded capability floor (a 34,702-byte
// compile-time constant already inside the binary) into a fresh
// .fak/guard-origin/<trace-id>-policy.json on EVERY launch, leaving an unbounded pile
// of byte-identical copies no reader ever opened (#6093). It now proves its location
// the same way its transcript, budget-envelope and Stop-ledger siblings always did.

import (
	"os"
	"path/filepath"
	"strings"
)

// PolicyOriginEvidencePath returns the durable location carrying the guarded child's
// policy origin evidence for traceID under the repo root.
//
// An operator-supplied policy file (`fak guard --policy`) wins and is returned
// untouched: it is already a durable location the operator owns. Otherwise the child
// ran on the embedded floor and the location is derived from the trace id alone —
// deliberately WITHOUT copying the floor's bytes into it (#6093). Returns "" when
// there is no location to hand over.
func PolicyOriginEvidencePath(root, traceID, explicitPath string) string {
	if explicit := strings.TrimSpace(explicitPath); explicit != "" {
		return explicit
	}
	root, traceID = strings.TrimSpace(root), strings.TrimSpace(traceID)
	if root == "" || traceID == "" {
		return ""
	}
	return filepath.Join(root, ".fak", "guard-origin", traceID+"-policy.json")
}

// EnsureOriginEvidence proves an origin-evidence location exists, returning its
// absolute path and whether it can be handed over as evidence.
//
// When the producer has not appended its first row yet it creates an EMPTY file, so
// PathWitness verifies the location rather than marking a valid pre-launch task
// refused. An existing file is left exactly as it is: this never truncates a
// producer's rows, and it never writes content of its own.
func EnsureOriginEvidence(path string) (string, bool) {
	path = strings.TrimSpace(path)
	if path == "" {
		return "", false
	}
	if abs, err := filepath.Abs(path); err == nil {
		path = abs
	}
	if _, err := os.Stat(path); os.IsNotExist(err) {
		if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
			return "", false
		}
		if err := os.WriteFile(path, nil, 0o600); err != nil {
			return "", false
		}
	}
	return path, true
}
