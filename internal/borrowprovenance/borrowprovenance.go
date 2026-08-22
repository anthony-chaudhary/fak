// Package borrowprovenance records and re-verifies exact external source bytes.
package borrowprovenance

import (
	"crypto/sha256"
	"encoding/hex"
	"errors"
	"fmt"
	"strings"
)

// Schema identifies the stable JSON record contract.
const Schema = "fak/borrow-provenance/v1"

// Record pins the exact upstream bytes used for a licensed borrow.
type Record struct {
	Schema         string `json:"schema"`
	SourceURL      string `json:"source_url"`
	SourceRef      string `json:"source_ref"`
	SourcePath     string `json:"source_path,omitempty"`
	SourceSHA256   string `json:"source_sha256"`
	License        string `json:"license,omitempty"`
	Transformation string `json:"transformation,omitempty"`
}

// Verification exposes both hashes so a drift failure is independently inspectable.
type Verification struct {
	Match          bool   `json:"match"`
	ExpectedSHA256 string `json:"expected_sha256"`
	ActualSHA256   string `json:"actual_sha256"`
}

// Pin creates a durable record for exact source bytes.
func Pin(sourceURL, sourceRef, sourcePath, license, transformation string, source []byte) (Record, error) {
	record := Record{Schema: Schema, SourceURL: strings.TrimSpace(sourceURL), SourceRef: strings.TrimSpace(sourceRef), SourcePath: strings.TrimSpace(sourcePath), License: strings.TrimSpace(license), Transformation: strings.TrimSpace(transformation), SourceSHA256: Digest(source)}
	if err := record.Validate(); err != nil {
		return Record{}, err
	}
	return record, nil
}

// Digest returns the lowercase SHA-256 of exact source bytes.
func Digest(source []byte) string {
	sum := sha256.Sum256(source)
	return hex.EncodeToString(sum[:])
}

// Validate checks that a record contains an immutable source identity.
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
func Verify(record Record, source []byte) (Verification, error) {
	if err := record.Validate(); err != nil {
		return Verification{}, err
	}
	actual := Digest(source)
	return Verification{Match: actual == record.SourceSHA256, ExpectedSHA256: record.SourceSHA256, ActualSHA256: actual}, nil
}
