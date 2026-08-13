package main

import (
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/seoaeoscore"
)

func TestValidateSEO(t *testing.T) {
	base := seoaeoscore.Corpus{OverallScore: 85.2, SEODebt: 414, DiscoveryOrphans: 46}
	if err := validateSEO(base, 85, 414, 46); err != nil {
		t.Fatalf("current floor: %v", err)
	}
	for _, tc := range []struct {
		name   string
		corpus seoaeoscore.Corpus
		want   string
	}{
		{"score regression", seoaeoscore.Corpus{OverallScore: 84.9}, "below"},
		{"debt regression", seoaeoscore.Corpus{OverallScore: 90, SEODebt: 415}, "debt"},
		{"orphan regression", seoaeoscore.Corpus{OverallScore: 90, DiscoveryOrphans: 47}, "orphans"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			err := validateSEO(tc.corpus, 85, 414, 46)
			if err == nil || !strings.Contains(err.Error(), tc.want) {
				t.Fatalf("error = %v, want %q", err, tc.want)
			}
		})
	}
}
