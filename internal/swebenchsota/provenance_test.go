package swebenchsota

import (
	"strings"
	"testing"
)

func TestBorrowedAssetValidateModes(t *testing.T) {
	excerpt := "a bounded upstream excerpt"
	digest := HashExcerpt(excerpt)
	for _, asset := range []BorrowedAsset{
		{AssetMode: AssetModeVerbatim, UpstreamSHA: "upstream-commit", Excerpt: excerpt, ExcerptSHA256: digest},
		{AssetMode: AssetModeExtracted, UpstreamSHA: "upstream-commit", Excerpt: excerpt, ExcerptSHA256: digest},
		{AssetMode: AssetModeDerived, DerivedFrom: []string{"source:one", "source:two"}, Excerpt: excerpt, ExcerptSHA256: digest},
	} {
		if err := asset.Validate(); err != nil {
			t.Fatalf("Validate(%q): %v", asset.AssetMode, err)
		}
	}
}

func TestBorrowedAssetValidateRefusesIncompleteProvenance(t *testing.T) {
	excerpt := "evidence"
	digest := HashExcerpt(excerpt)
	tests := []struct {
		name  string
		asset BorrowedAsset
		want  string
	}{
		{"unknown mode", BorrowedAsset{AssetMode: "copied", Excerpt: excerpt, ExcerptSHA256: digest}, "unknown asset_mode"},
		{"verbatim source", BorrowedAsset{AssetMode: AssetModeVerbatim, Excerpt: excerpt, ExcerptSHA256: digest}, "requires upstream_sha"},
		{"extracted source", BorrowedAsset{AssetMode: AssetModeExtracted, Excerpt: excerpt, ExcerptSHA256: digest}, "requires upstream_sha"},
		{"derived source", BorrowedAsset{AssetMode: AssetModeDerived, Excerpt: excerpt, ExcerptSHA256: digest}, "requires derived_from"},
		{"empty source ID", BorrowedAsset{AssetMode: AssetModeDerived, DerivedFrom: []string{" "}, Excerpt: excerpt, ExcerptSHA256: digest}, "must not be empty"},
		{"duplicate source ID", BorrowedAsset{AssetMode: AssetModeDerived, DerivedFrom: []string{"source:one", " source:one "}, Excerpt: excerpt, ExcerptSHA256: digest}, "duplicated"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			err := tt.asset.Validate()
			if err == nil || !strings.Contains(err.Error(), tt.want) {
				t.Fatalf("Validate() error = %v, want substring %q", err, tt.want)
			}
		})
	}
}

func TestBorrowedAssetValidateRecomputesExcerptHash(t *testing.T) {
	asset := BorrowedAsset{
		AssetMode:   AssetModeVerbatim,
		UpstreamSHA: "upstream-commit",
		Excerpt:     "original excerpt",
	}
	asset.ExcerptSHA256 = HashExcerpt(asset.Excerpt)
	asset.Excerpt = "tampered excerpt"
	if err := asset.Validate(); err == nil || !strings.Contains(err.Error(), "mismatch") {
		t.Fatalf("Validate() error = %v, want excerpt hash mismatch", err)
	}
}
