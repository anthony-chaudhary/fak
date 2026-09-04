// Package borrowprovenance records and re-verifies exact external source bytes.
// Invariant: Source identity and checksums are immutable once pinned.
// Guard: Any digest drift or invalid schema causes immediate verification rejection.
package borrowprovenance

import (
	"crypto/sha256"
	"encoding/hex"
	"errors"
	"fmt"
	"strings"
)

// Schema identifies the stable JSON record contract for borrow provenance.
// Invariant: Schema matches the frozen identifier string "fak/borrow-provenance/v1".
// Guard: Any mismatched schema string fails validation immediately.
const Schema = "fak/borrow-provenance/v1"

// Record pins the exact upstream bytes used for a licensed borrow.
// Invariant: SourceURL and SourceRef must be non-empty trimmed strings.
// Invariant: SourceSHA256 must be a valid 64-character lowercase hex SHA-256 checksum.
// Guard: Unvalidated or drifted records must not be accepted for provenance tracking.
type Record struct {
	// Schema names the serialization format identifier.
	Schema string `json:"schema"`
	// SourceURL identifies the canonical repository or publication source.
	SourceURL string `json:"source_url"`
	// SourceRef identifies the immutable commit SHA, tag, or version revision.
	SourceRef string `json:"source_ref"`
	// SourcePath optionally records the relative path within the upstream source tree.
	SourcePath string `json:"source_path,omitempty"`
	// SourceSHA256 holds the lowercase 64-character hex digest of the unmutated source bytes.
	SourceSHA256 string `json:"source_sha256"`
	// License identifies the declared upstream license terms (e.g., Apache-2.0, MIT).
	License string `json:"license,omitempty"`
	// Transformation documents any modifications applied to the borrowed content.
	Transformation string `json:"transformation,omitempty"`
}

// Verification exposes both expected and actual hashes so drift is independently inspectable.
// Invariant: Match is true if and only if ExpectedSHA256 exactly equals ActualSHA256.
// Guard: Drifted digests yield Match=false without mutating the reference record.
type Verification struct {
	// Match reports whether the recomputed digest exactly matches ExpectedSHA256.
	Match bool `json:"match"`
	// ExpectedSHA256 is the trusted reference checksum from the pin record.
	ExpectedSHA256 string `json:"expected_sha256"`
	// ActualSHA256 is the checksum recomputed from the inspected source bytes.
	ActualSHA256 string `json:"actual_sha256"`
}

// Pin creates a durable record for exact source bytes.
// Invariant: Pin yields a valid Record whose SourceSHA256 matches Digest(source).
// Guard: Returns an error if sourceURL or sourceRef are blank or if validation fails.
func Pin(sourceURL, sourceRef, sourcePath, license, transformation string, source []byte) (Record, error) {
	record := Record{
		Schema:         Schema,
		SourceURL:      strings.TrimSpace(sourceURL),
		SourceRef:      strings.TrimSpace(sourceRef),
		SourcePath:     strings.TrimSpace(sourcePath),
		License:        strings.TrimSpace(license),
		Transformation: strings.TrimSpace(transformation),
		SourceSHA256:   Digest(source),
	}
	if err := record.Validate(); err != nil {
		return Record{}, err
	}
	return record, nil
}

// Digest returns the lowercase SHA-256 hex string of exact source bytes.
// Invariant: The returned digest is deterministically 64 lowercase hexadecimal characters.
// Guard: Empty source byte slices produce the well-known SHA-256 empty digest.
func Digest(source []byte) string {
	sum := sha256.Sum256(source)
	return hex.EncodeToString(sum[:])
}

// Validate checks that a record contains an immutable source identity and valid checksum.
// Invariant: Requires Schema == Schema, non-empty SourceURL and SourceRef, and a 64-char lowercase hex digest.
// Guard: Fail-closed; returns an error if any invariant is violated.
func (r Record) Validate() error {
	if r.Schema != Schema {
		return fmt.Errorf("borrow provenance: schema %q, want %q", r.Schema, Schema)
	}
	if strings.TrimSpace(r.SourceURL) == "" || strings.TrimSpace(r.SourceRef) == "" {
		return errors.New("borrow provenance: source_url and source_ref are required")
	}
	if len(r.SourceSHA256) != sha256.Size*2 {
		return errors.New("borrow provenance: source_sha256 must be 64 lowercase hexadecimal characters")
	}
	decoded, err := hex.DecodeString(r.SourceSHA256)
	if err != nil || hex.EncodeToString(decoded) != r.SourceSHA256 {
		return errors.New("borrow provenance: source_sha256 must be 64 lowercase hexadecimal characters")
	}
	return nil
}

// Verify recomputes the digest without mutating or silently refreshing the pin.
// Invariant: Verification.Match is true if and only if Digest(source) == record.SourceSHA256.
// Guard: Fail-closed; invalid records return an error before inspection proceeds.
func Verify(record Record, source []byte) (Verification, error) {
	if err := record.Validate(); err != nil {
		return Verification{}, err
	}
	actual := Digest(source)
	return Verification{
		Match:          actual == record.SourceSHA256,
		ExpectedSHA256: record.SourceSHA256,
		ActualSHA256:   actual,
	}, nil
}
