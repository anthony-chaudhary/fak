package managedocs

import (
	"os"
	"path/filepath"
	"regexp"
	"strings"
	"testing"
)

func TestClaimsIndexHasOnePagePerFormerClaim(t *testing.T) {
	root := filepath.Clean(filepath.Join("..", ".."))
	b, err := os.ReadFile(filepath.Join(root, "CLAIMS.md"))
	if err != nil {
		t.Fatal(err)
	}
	links := regexp.MustCompile(`\(docs/claims/([^)]+\.md)\)`).FindAllStringSubmatch(string(b), -1)
	if len(links) < 30 {
		t.Fatalf("claims pages=%d, want at least 30", len(links))
	}
	for _, m := range links {
		p := filepath.Join(root, "docs", "claims", filepath.FromSlash(m[1]))
		body, err := os.ReadFile(p)
		if err != nil {
			t.Errorf("%s: %v", p, err)
			continue
		}
		if !strings.Contains(string(body), "[← Claims index](../../CLAIMS.md)") {
			t.Errorf("%s lacks index backlink", p)
		}
	}
}
