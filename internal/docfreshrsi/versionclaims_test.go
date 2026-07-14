package docfreshrsi

import "testing"

func TestScanVersionClaims(t *testing.T) {
	text := `# Install
The latest Codex release is available.

CUDA 13 or newer is supported. As of 2026-07-14.

The current provider is documented here.
Freshness: docs/provider.md@1a2b3c4

## Historical behavior
The latest release required Python 3.9.

## Live again
This feature shipped in v2.4.

` + "```" + `text
The latest release requires Node 22.
` + "```" + `
`
	got := ScanVersionClaims("docs/install.md", text)
	if len(got) != 2 {
		t.Fatalf("findings=%d want 2: %+v", len(got), got)
	}
	if got[0].Line != 2 || got[0].Signature != "current/latest" {
		t.Fatalf("first=%+v", got[0])
	}
	if got[1].Signature != "release assertion" {
		t.Fatalf("second=%+v", got[1])
	}
}

func TestScanVersionClaimsPointerForms(t *testing.T) {
	clean := []string{
		"Latest runtime. Source: https://example.com/releases",
		"Go 1.26 or newer. Last verified: 2026-07-14",
		"Current API. source: upstream@1a2b3c4",
		"CUDA 13+. Verified 2026-07-14",
	}
	for _, line := range clean {
		if got := ScanVersionClaims("x.md", line); len(got) != 0 {
			t.Fatalf("pointed claim flagged %q: %+v", line, got)
		}
	}
}
