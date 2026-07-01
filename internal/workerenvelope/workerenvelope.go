// Package workerenvelope defines a small machine-readable envelope for the
// RESULT a dispatch worker hands back when it finishes a GitHub issue. It
// exists so a fleet supervisor can fold hundreds of worker returns per hour
// WITHOUT trusting a free-text "done" string: every field the supervisor needs
// to decide "did this actually ship?" is typed, and Validate enforces the
// witness-gated dispatch contract (no self-reported completion without a
// pointer to the evidence).
//
// # Schema
//
// A Result carries exactly what the supervisor must see:
//
//   - Status    — the worker's own claim about the outcome, one of the three
//     StatusShipped / StatusBlocked / StatusNotYet values. Any other string is
//     a validation error.
//   - Issue     — the GitHub issue number the worker was assigned. Must be > 0.
//   - CommitSHA — the commit the work landed in. Required (40-hex or a >=7-char
//     short prefix) when Status is StatusShipped; may be empty otherwise.
//   - TestsRun  — the tests the worker actually ran (e.g. the `go test ./...`
//     package paths). Advisory: recorded so the fold can see coverage, never
//     required.
//   - Blocker   — a short reason the work did NOT ship. Required (non-empty)
//     when Status is StatusBlocked or StatusNotYet; must be empty when
//     StatusShipped (a shipped result has no blocker).
//   - Witness   — a POINTER to the evidence: a commit ref, a test path, or a
//     log path. Required (non-empty) when Status is StatusShipped so the claim
//     can be checked. Advisory otherwise.
//
// # Witness-gated contract
//
// The whole point of the envelope is that Validate refuses a bare "I shipped
// it": a StatusShipped result MUST carry both a well-formed CommitSHA AND a
// non-empty Witness, and a StatusBlocked / StatusNotYet result MUST name a
// Blocker. This mirrors the dispatch rule that a worker's PASS is not
// trustworthy until re-verified against origin/main — the envelope forces the
// pointer that makes re-verification possible.
//
// Everything here is stdlib-only (encoding/json), imports nothing internal,
// exposes no CLI surface, and is off the hot path.
package workerenvelope

import (
	"encoding/json"
	"fmt"
	"strings"
)

// Status is the worker's own claim about how the assigned issue ended.
type Status string

const (
	// StatusShipped means the work landed in a commit. Requires a well-formed
	// CommitSHA and a non-empty Witness; forbids a Blocker.
	StatusShipped Status = "shipped"
	// StatusBlocked means the worker could not proceed. Requires a Blocker.
	StatusBlocked Status = "blocked"
	// StatusNotYet means the work is incomplete but not hard-blocked (an open
	// follow-on, missing wiring). Requires a Blocker naming what is missing.
	StatusNotYet Status = "not_yet"
)

// valid reports whether s is one of the three recognized statuses.
func (s Status) valid() bool {
	switch s {
	case StatusShipped, StatusBlocked, StatusNotYet:
		return true
	default:
		return false
	}
}

// Result is the machine-readable envelope a dispatch worker returns for one
// GitHub issue. See the package doc for the field-by-field contract.
type Result struct {
	Status    Status   `json:"status"`
	Issue     int      `json:"issue"`
	CommitSHA string   `json:"commit_sha,omitempty"`
	TestsRun  []string `json:"tests_run,omitempty"`
	Blocker   string   `json:"blocker,omitempty"`
	Witness   string   `json:"witness,omitempty"`
}

// looksLikeSHA reports whether s is a plausible git commit SHA: a hex string of
// at least 7 characters (a short SHA) and at most 40 (a full SHA-1). It does
// not verify the SHA exists — only its shape, which is enough to catch a
// truncated field, a subject line pasted into the wrong slot, or a branch name.
func looksLikeSHA(s string) bool {
	if len(s) < 7 || len(s) > 40 {
		return false
	}
	for _, r := range s {
		switch {
		case r >= '0' && r <= '9':
		case r >= 'a' && r <= 'f':
		case r >= 'A' && r <= 'F':
		default:
			return false
		}
	}
	return true
}

// Validate enforces the witness-gated dispatch contract. It returns a
// descriptive error naming the first field that violates the contract for the
// declared Status, or nil if the envelope is well-formed.
//
// Checks that apply to every status:
//
//   - Status must be one of the three recognized values.
//   - Issue must be > 0.
//   - CommitSHA, when present, must look like a git SHA.
//
// Status-specific checks:
//
//   - StatusShipped: CommitSHA and Witness must both be non-empty (and the SHA
//     well-formed); Blocker must be empty.
//   - StatusBlocked / StatusNotYet: Blocker must be non-empty.
func (r Result) Validate() error {
	if !r.Status.valid() {
		return fmt.Errorf("workerenvelope: invalid status %q (want shipped|blocked|not_yet)", r.Status)
	}
	if r.Issue <= 0 {
		return fmt.Errorf("workerenvelope: issue must be > 0, got %d", r.Issue)
	}
	// A present CommitSHA must be well-shaped regardless of status: a garbage
	// SHA on a blocked result is still a bug worth catching.
	if r.CommitSHA != "" && !looksLikeSHA(r.CommitSHA) {
		return fmt.Errorf("workerenvelope: commit_sha %q is not a 7-40 char hex sha", r.CommitSHA)
	}

	switch r.Status {
	case StatusShipped:
		if strings.TrimSpace(r.CommitSHA) == "" {
			return fmt.Errorf("workerenvelope: shipped result requires a commit_sha")
		}
		if !looksLikeSHA(r.CommitSHA) {
			return fmt.Errorf("workerenvelope: shipped result commit_sha %q is not a 7-40 char hex sha", r.CommitSHA)
		}
		if strings.TrimSpace(r.Witness) == "" {
			return fmt.Errorf("workerenvelope: shipped result requires a witness (commit ref / test path / log path)")
		}
		if strings.TrimSpace(r.Blocker) != "" {
			return fmt.Errorf("workerenvelope: shipped result must not carry a blocker (got %q)", r.Blocker)
		}
	case StatusBlocked, StatusNotYet:
		if strings.TrimSpace(r.Blocker) == "" {
			return fmt.Errorf("workerenvelope: %s result requires a blocker naming what is missing", r.Status)
		}
	}
	return nil
}

// Parse decodes a Result from JSON and validates it. It returns the decoded
// Result together with the first error from JSON decoding or Validate; on any
// error the returned Result is the partially-decoded value and should not be
// trusted.
func Parse(data []byte) (Result, error) {
	var r Result
	if err := json.Unmarshal(data, &r); err != nil {
		return r, fmt.Errorf("workerenvelope: decode: %w", err)
	}
	if err := r.Validate(); err != nil {
		return r, err
	}
	return r, nil
}
