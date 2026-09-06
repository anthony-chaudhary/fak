// Package workerenvelope defines a machine-readable result envelope returned
// by dispatch workers upon completing assigned tasks, enforcing evidence pointers
// for shipped outcomes and mandatory blocker descriptions for blocked work.
package workerenvelope

import (
	"encoding/json"
	"fmt"
	"strings"
)

// Status represents the reported terminal outcome of an assigned issue.
type Status string

const (
	// StatusShipped indicates the work landed in a commit with verified witness evidence.
	StatusShipped Status = "shipped"
	// StatusBlocked indicates the worker encountered a blocking condition and could not proceed.
	StatusBlocked Status = "blocked"
	// StatusNotYet indicates the work is incomplete and requires follow-up.
	StatusNotYet Status = "not_yet"
)

// valid reports whether s is one of the recognized statuses.
func (s Status) valid() bool {
	switch s {
	case StatusShipped, StatusBlocked, StatusNotYet:
		return true
	default:
		return false
	}
}

// Result is the machine-readable envelope returned by a dispatch worker.
type Result struct {
	Status    Status   `json:"status"`
	Issue     int      `json:"issue"`
	CommitSHA string   `json:"commit_sha,omitempty"`
	TestsRun  []string `json:"tests_run,omitempty"`
	Blocker   string   `json:"blocker,omitempty"`
	Witness   string   `json:"witness,omitempty"`
}

// looksLikeSHA reports whether s is a valid 7-64 character hexadecimal commit SHA.
func looksLikeSHA(s string) bool {
	if len(s) < 7 || len(s) > 64 {
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

// Validate verifies that the envelope conforms to dispatch requirements:
// status must be known, issue must be positive, and any present commit SHA
// must be 7-64 hex characters. Shipped results require a commit SHA and witness
// with no blocker; blocked or not_yet results require a blocker.
func (r Result) Validate() error {
	if !r.Status.valid() {
		return fmt.Errorf("workerenvelope: invalid status %q (want shipped|blocked|not_yet)", r.Status)
	}
	if r.Issue <= 0 {
		return fmt.Errorf("workerenvelope: issue must be > 0, got %d", r.Issue)
	}
	if r.CommitSHA != "" && !looksLikeSHA(r.CommitSHA) {
		return fmt.Errorf("workerenvelope: commit_sha %q is not a 7-64 char hex sha", r.CommitSHA)
	}

	switch r.Status {
	case StatusShipped:
		if strings.TrimSpace(r.CommitSHA) == "" {
			return fmt.Errorf("workerenvelope: shipped result requires a commit_sha")
		}
		if !looksLikeSHA(r.CommitSHA) {
			return fmt.Errorf("workerenvelope: shipped result commit_sha %q is not a 7-64 char hex sha", r.CommitSHA)
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

// Parse decodes JSON data into a Result and validates its fields.
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
