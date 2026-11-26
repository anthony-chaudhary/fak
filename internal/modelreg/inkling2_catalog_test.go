package modelreg

import (
	"crypto/sha1"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/hfhub"
)

// The Inkling-2 aliases pin an exact repo revision and file: Small is shard 1
// of the six-shard UD-Q2_K_XL set, Mini the Mainline_Support UD-Q3_K_XL file
// whose sha256 the Mini baseline receipt recorded.
func TestInkling2CatalogAliasesResolveToPinnedGGUF(t *testing.T) {
	withCacheRoot(t)
	want := map[string]hfhub.Ref{
		"inkling2-small": {
			Repo:     "unsloth/Inkling-2-Small-GGUF",
			Revision: "42593c63926d23380ea03d643ab2cde742427608",
			File:     "UD-Q2_K_XL/Inkling-2-Small-UD-Q2_K_XL-00001-of-00006.gguf",
		},
		"inkling2-mini": {
			Repo:     "unsloth/Inkling-2-Mini-GGUF",
			Revision: "8078d149286eec7193795413b9e0643da6d11fa1",
			File:     "Mainline_Support/Inkling-2-Mini-UD-Q3_K_XL.gguf_file",
		},
	}
	for alias, ref := range want {
		got, expanded := Resolve(alias)
		if !expanded {
			t.Fatalf("Resolve(%q) did not expand; got %q", alias, got)
		}
		parsed, err := hfhub.ParseURI(got)
		if err != nil {
			t.Fatalf("Resolve(%q) = %q: %v", alias, got, err)
		}
		if parsed != ref {
			t.Fatalf("Resolve(%q) = %+v, want %+v", alias, parsed, ref)
		}
		if !isFullGitCommitSHA(parsed.Revision) {
			t.Fatalf("Resolve(%q) revision %q is not a full commit sha", alias, parsed.Revision)
		}
	}
}

func TestInkling2CatalogAliasesAreNotCoding(t *testing.T) {
	for _, alias := range []string{"inkling2-small", "inkling2-mini"} {
		if IsCoding(alias) {
			t.Fatalf("%s flagged as a coding alias", alias)
		}
	}
}

// isFullGitCommitSHA reports whether rev is a complete lowercase-hex commit id,
// the pin form every catalog revision must take.
func isFullGitCommitSHA(rev string) bool {
	return len(rev) == sha1.Size*2 && strings.Trim(rev, "0123456789abcdef") == ""
}
