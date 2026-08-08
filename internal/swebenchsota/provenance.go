package swebenchsota

import (
	"crypto/sha256"
	"encoding/hex"
	"errors"
	"fmt"
	"strings"
)

// AssetMode records how a borrowed asset relates to its cited source.
type AssetMode string

const (
	AssetModeVerbatim  AssetMode = "verbatim"
	AssetModeExtracted AssetMode = "extracted"
	AssetModeDerived   AssetMode = "derived"
)

// BorrowedAsset is the provenance envelope for an excerpt admitted by a
// field-borrow workflow. ExcerptSHA256 is the lowercase SHA-256 of Excerpt.
type BorrowedAsset struct {
	AssetMode     AssetMode `json:"asset_mode"`
	UpstreamSHA   string    `json:"upstream_sha,omitempty"`
	DerivedFrom   []string  `json:"derived_from,omitempty"`
	Excerpt       string    `json:"excerpt"`
	ExcerptSHA256 string    `json:"excerpt_sha256"`
}

// HashExcerpt returns the canonical digest stored in ExcerptSHA256.
func HashExcerpt(excerpt string) string {
	sum := sha256.Sum256([]byte(excerpt))
	return hex.EncodeToString(sum[:])
}

// Validate refuses unknown provenance modes, incomplete mode-specific
// provenance, and excerpts whose stored digest does not match their content.
func (a BorrowedAsset) Validate() error {
	if strings.TrimSpace(a.Excerpt) == "" {
		return errors.New("excerpt is required")
	}
	if a.ExcerptSHA256 == "" {
		return errors.New("excerpt_sha256 is required")
	}
	if _, err := hex.DecodeString(a.ExcerptSHA256); err != nil || len(a.ExcerptSHA256) != sha256.Size*2 {
		return errors.New("excerpt_sha256 must be a 64-character hexadecimal SHA-256")
	}
	if got := HashExcerpt(a.Excerpt); a.ExcerptSHA256 != got {
		return fmt.Errorf("excerpt_sha256 mismatch: got %s, want %s", a.ExcerptSHA256, got)
	}

	switch a.AssetMode {
	case AssetModeVerbatim, AssetModeExtracted:
		if strings.TrimSpace(a.UpstreamSHA) == "" {
			return fmt.Errorf("asset_mode %q requires upstream_sha", a.AssetMode)
		}
		if len(a.DerivedFrom) != 0 {
			return fmt.Errorf("asset_mode %q must not set derived_from", a.AssetMode)
		}
	case AssetModeDerived:
		if len(a.DerivedFrom) == 0 {
			return errors.New("asset_mode \"derived\" requires derived_from source IDs")
		}
		seen := make(map[string]struct{}, len(a.DerivedFrom))
		for _, sourceID := range a.DerivedFrom {
			sourceID = strings.TrimSpace(sourceID)
			if sourceID == "" {
				return errors.New("derived_from source IDs must not be empty")
			}
			if _, ok := seen[sourceID]; ok {
				return fmt.Errorf("derived_from source ID %q is duplicated", sourceID)
			}
			seen[sourceID] = struct{}{}
		}
	default:
		return fmt.Errorf("unknown asset_mode %q (want verbatim, extracted, or derived)", a.AssetMode)
	}
	return nil
}
