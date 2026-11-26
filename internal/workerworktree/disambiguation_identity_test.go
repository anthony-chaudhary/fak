package workerworktree

import (
	"testing"

	"github.com/anthony-chaudhary/fak/internal/conceptcatalog/scorecard"
)

func TestDisambiguationIdentityCoversCompiledAnalyzerSources(t *testing.T) {
	if disambiguationAnalyzerSources != scorecard.SourceDigest() || disambiguationAnalyzerSources == "" {
		t.Fatalf("analyzer source identity %q does not name the compiled scorecard %q", disambiguationAnalyzerSources, scorecard.SourceDigest())
	}
	old := disambiguationAnalyzerSources
	t.Cleanup(func() { disambiguationAnalyzerSources = old })

	archive := []byte("same archive bytes")
	tree, arch, prepared := disambiguationCacheKey("tree-a"), disambiguationArchiveCacheKey(archive), preparedDisambiguationAnalyzerIdentity()
	disambiguationAnalyzerSources = old + "+edited"
	if disambiguationCacheKey("tree-a") == tree {
		t.Fatal("tree-scoped witness identity ignores the analyzer sources")
	}
	if disambiguationArchiveCacheKey(archive) == arch {
		t.Fatal("archive-scoped witness identity ignores the analyzer sources")
	}
	if preparedDisambiguationAnalyzerIdentity() == prepared {
		t.Fatal("prepared-land analyzer identity ignores the analyzer sources")
	}
	disambiguationAnalyzerSources = old
	if disambiguationCacheKey("tree-a") != tree || disambiguationArchiveCacheKey(archive) != arch {
		t.Fatal("witness identity is not stable for unchanged analyzer sources")
	}
}
