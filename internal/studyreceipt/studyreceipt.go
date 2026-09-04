// Package studyreceipt implements verifiable study receipt tracking, validation,
// and tamper-evident storage for empirical study artifacts.
package studyreceipt

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"regexp"
	"sort"
	"strings"
	"time"
)

// Schema is the canonical schema identifier for study receipts.
const Schema = "fak.study-receipt/1"

var (
	idRegex     = regexp.MustCompile(`^[a-z0-9][a-z0-9_-]{2,63}$`)
	digestRegex = regexp.MustCompile(`^sha256:[0-9a-f]{64}$`)
)

// Common validation errors.
var (
	ErrInvalidSchema    = errors.New("studyreceipt: invalid schema")
	ErrInvalidID        = errors.New("studyreceipt: invalid receipt or study ID")
	ErrMissingField     = errors.New("studyreceipt: missing required field")
	ErrInvalidDigest    = errors.New("studyreceipt: invalid sha256 digest format")
	ErrInvalidTiming    = errors.New("studyreceipt: invalid start/stop timestamps or elapsed duration")
	ErrInvalidOutcome   = errors.New("studyreceipt: invalid outcome verdict")
	ErrCorruptedReceipt = errors.New("studyreceipt: receipt content does not match digest")
	ErrDuplicateReceipt = errors.New("studyreceipt: receipt already exists in store")
)

// Outcome represents the final determination of a study execution.
type Outcome string

const (
	OutcomeSuccess Outcome = "success"
	OutcomeFailure Outcome = "failure"
	OutcomeAbstain Outcome = "abstain"
)

// SourceRef binds an external or internal source input to its immutable cryptographic digest.
type SourceRef struct {
	Name     string `json:"name"`
	URL      string `json:"url,omitempty"`
	Revision string `json:"revision"`
	Digest   string `json:"digest"`
}

// Observation represents a captured empirical data point.
type Observation struct {
	ID         string `json:"id"`
	RecordedAt string `json:"recorded_at"`
	Metric     string `json:"metric"`
	Value      string `json:"value"`
	Witness    string `json:"witness,omitempty"`
}

// Decision represents an actionable decision derived from study observations.
type Decision struct {
	ID          string   `json:"id"`
	Candidate   string   `json:"candidate"`
	Disposition string   `json:"disposition"`
	Rationale   string   `json:"rationale"`
	Evidence    []string `json:"evidence"`
}

// Receipt is an immutable, verifiable record of a completed study execution.
type Receipt struct {
	Schema       string        `json:"schema"`
	ID           string        `json:"id"`
	StudyID      string        `json:"study_id"`
	Track        string        `json:"track"`
	Participant  string        `json:"participant"`
	Environment  string        `json:"environment"`
	StartedAt    string        `json:"started_at"`
	CompletedAt  string        `json:"completed_at"`
	ElapsedSec   float64       `json:"elapsed_seconds"`
	Outcome      Outcome       `json:"outcome"`
	Artifacts    []string      `json:"artifacts,omitempty"`
	Sources      []SourceRef   `json:"sources"`
	Observations []Observation `json:"observations,omitempty"`
	Decisions    []Decision    `json:"decisions,omitempty"`
	Digest       string        `json:"digest,omitempty"`
}

// Digest computes the canonical SHA-256 digest of the receipt payload excluding the Digest field.
func Digest(r Receipt) (string, error) {
	// Create a copy with Digest cleared for deterministic hashing.
	c := r
	c.Digest = ""
	b, err := json.Marshal(c)
	if err != nil {
		return "", fmt.Errorf("marshal receipt: %w", err)
	}
	h := sha256.Sum256(b)
	return "sha256:" + hex.EncodeToString(h[:]), nil
}

// Validate verifies the structural and logical validity of a study receipt.
func Validate(r Receipt) error {
	if r.Schema != Schema {
		return fmt.Errorf("%w: expected %q, got %q", ErrInvalidSchema, Schema, r.Schema)
	}
	if !idRegex.MatchString(r.ID) {
		return fmt.Errorf("%w: ID %q does not match pattern %s", ErrInvalidID, r.ID, idRegex.String())
	}
	if !idRegex.MatchString(r.StudyID) {
		return fmt.Errorf("%w: StudyID %q does not match pattern %s", ErrInvalidID, r.StudyID, idRegex.String())
	}
	if strings.TrimSpace(r.Track) == "" {
		return fmt.Errorf("%w: track is required", ErrMissingField)
	}
	if strings.TrimSpace(r.Participant) == "" {
		return fmt.Errorf("%w: participant is required", ErrMissingField)
	}
	if strings.TrimSpace(r.Environment) == "" {
		return fmt.Errorf("%w: environment is required", ErrMissingField)
	}

	start, err := time.Parse(time.RFC3339Nano, r.StartedAt)
	if err != nil {
		return fmt.Errorf("%w: started_at must be RFC3339: %v", ErrInvalidTiming, err)
	}
	end, err := time.Parse(time.RFC3339Nano, r.CompletedAt)
	if err != nil {
		return fmt.Errorf("%w: completed_at must be RFC3339: %v", ErrInvalidTiming, err)
	}
	if end.Before(start) {
		return fmt.Errorf("%w: completed_at is before started_at", ErrInvalidTiming)
	}
	if r.ElapsedSec < 0 {
		return fmt.Errorf("%w: elapsed_seconds must be non-negative", ErrInvalidTiming)
	}

	switch r.Outcome {
	case OutcomeSuccess, OutcomeFailure, OutcomeAbstain:
	default:
		return fmt.Errorf("%w: unexpected outcome %q", ErrInvalidOutcome, r.Outcome)
	}

	if len(r.Sources) == 0 {
		return fmt.Errorf("%w: at least one source reference is required", ErrMissingField)
	}
	for i, s := range r.Sources {
		if strings.TrimSpace(s.Name) == "" || strings.TrimSpace(s.Revision) == "" {
			return fmt.Errorf("%w: source[%d] requires name and revision", ErrMissingField, i)
		}
		if !digestRegex.MatchString(s.Digest) {
			return fmt.Errorf("%w: source[%d] digest %q", ErrInvalidDigest, i, s.Digest)
		}
	}

	for i, obs := range r.Observations {
		if strings.TrimSpace(obs.ID) == "" || strings.TrimSpace(obs.Metric) == "" {
			return fmt.Errorf("%w: observation[%d] requires id and metric", ErrMissingField, i)
		}
	}

	for i, dec := range r.Decisions {
		if strings.TrimSpace(dec.ID) == "" || strings.TrimSpace(dec.Candidate) == "" {
			return fmt.Errorf("%w: decision[%d] requires id and candidate", ErrMissingField, i)
		}
		if len(dec.Evidence) == 0 {
			return fmt.Errorf("%w: decision[%d] requires evidence references", ErrMissingField, i)
		}
	}

	if r.Digest != "" {
		expected, err := Digest(r)
		if err != nil {
			return err
		}
		if r.Digest != expected {
			return fmt.Errorf("%w: receipt claims %s, computed %s", ErrCorruptedReceipt, r.Digest, expected)
		}
	}

	return nil
}

// Store provides persistence and retrieval operations for verified receipts.
type Store struct {
	dir string
}

// OpenStore opens or creates a study receipt store in the given directory.
func OpenStore(dir string) (*Store, error) {
	if strings.TrimSpace(dir) == "" {
		return nil, errors.New("studyreceipt: store directory cannot be empty")
	}
	if err := os.MkdirAll(dir, 0700); err != nil {
		return nil, fmt.Errorf("studyreceipt: create store dir: %w", err)
	}
	return &Store{dir: dir}, nil
}

// Put validates, seals with digest, and durably writes a receipt.
func (s *Store) Put(r Receipt) (Receipt, error) {
	if err := Validate(r); err != nil {
		return Receipt{}, err
	}
	d, err := Digest(r)
	if err != nil {
		return Receipt{}, err
	}
	r.Digest = d

	path := filepath.Join(s.dir, r.ID+".json")
	if _, err := os.Stat(path); err == nil {
		return Receipt{}, fmt.Errorf("%w: %s", ErrDuplicateReceipt, r.ID)
	}

	b, err := json.MarshalIndent(r, "", "  ")
	if err != nil {
		return Receipt{}, fmt.Errorf("marshal receipt: %w", err)
	}

	tmpPath := path + ".tmp"
	if err := os.WriteFile(tmpPath, b, 0600); err != nil {
		return Receipt{}, fmt.Errorf("write temp receipt: %w", err)
	}
	if err := os.Rename(tmpPath, path); err != nil {
		_ = os.Remove(tmpPath)
		return Receipt{}, fmt.Errorf("commit receipt file: %w", err)
	}
	return r, nil
}

// Get reads and validates a receipt from the store by ID.
func (s *Store) Get(id string) (Receipt, error) {
	if !idRegex.MatchString(id) {
		return Receipt{}, fmt.Errorf("%w: %q", ErrInvalidID, id)
	}
	path := filepath.Join(s.dir, id+".json")
	b, err := os.ReadFile(path)
	if err != nil {
		return Receipt{}, err
	}
	var r Receipt
	if err := json.Unmarshal(b, &r); err != nil {
		return Receipt{}, fmt.Errorf("unmarshal receipt: %w", err)
	}
	if err := Validate(r); err != nil {
		return Receipt{}, fmt.Errorf("validate stored receipt: %w", err)
	}
	return r, nil
}

// List returns all verified receipts matching the optional studyID filter.
func (s *Store) List(studyID string) ([]Receipt, error) {
	entries, err := os.ReadDir(s.dir)
	if err != nil {
		return nil, fmt.Errorf("read store dir: %w", err)
	}
	var out []Receipt
	for _, entry := range entries {
		if entry.IsDir() || !strings.HasSuffix(entry.Name(), ".json") {
			continue
		}
		id := strings.TrimSuffix(entry.Name(), ".json")
		r, err := s.Get(id)
		if err != nil {
			// Skip corrupted or unvalidated non-receipt JSON files.
			continue
		}
		if studyID == "" || r.StudyID == studyID {
			out = append(out, r)
		}
	}
	sort.Slice(out, func(i, j int) bool {
		return out[i].ID < out[j].ID
	})
	return out, nil
}
