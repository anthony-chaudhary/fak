package headlesslint

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// TestScanClosing exercises both arms of the closing-shape fold: a trailing prose wall
// is refused, while bulleted, short, empty, and escaped closings pass clean. The cases
// double as the fold's re-derivable corpus (leftovers.go's fixture is Scan-specific, so
// closing cases live here rather than in fixture.go).
func TestScanClosing(t *testing.T) {
	const proseWall = "Shipped the retry fix and pushed abc123.\n\n" +
		"Overall this took longer than expected because the backoff logic interacts with the existing rate limiter in a subtle way. I ended up refactoring the limiter to expose its window, which touched three files. The tests all pass now and the build is green, so this should be good to go once someone glances at the limiter change."

	const bulleted = "Summary:\n\n" +
		"- Shipped the retry fix, pushed abc123\n" +
		"- Tests pass (go test ./internal/retry)\n" +
		"- Next: wire the dashboard panel (#4821)"

	const numbered = "Done:\n\n1. Refactored the limiter\n2. Added the jitter test\n3. Next: land the docs"

	// Prose in the MIDDLE but a bulleted final block — clean, only the last block counts.
	const proseThenBullets = "I reworked the whole retry path from scratch because the old one conflated backoff and rate limiting, which made the jitter case untestable in isolation and hid the real bug.\n\n" +
		"- Fixed the conflation, pushed abc123\n- Next: dashboard panel (#4821)"

	// Bullets earlier but a trailing prose block — refused, the operator's eye lands on prose.
	const bulletsThenProse = "- Fixed the conflation, pushed abc123\n- Tests pass\n\n" +
		"One more thing worth calling out before you look at this: the limiter change is subtle and the window it now exposes is shared mutable state, so if anything else starts reading it concurrently we will need a lock, which is not there today and would be a real correctness bug under load."

	cases := []struct {
		name     string
		summary  string
		override bool
		refused  bool
	}{
		{"prose-wall-refused", proseWall, false, true},
		{"prose-wall-overridden", proseWall, true, false},
		{"bulleted-clean", bulleted, false, false},
		{"numbered-clean", numbered, false, false},
		{"short-two-sentence-closer-clean", "Done. Pushed abc123.", false, false},
		{"short-semicolon-closer-clean", "Nothing left; pushed abc123.", false, false},
		{"honest-three-sentence-completion-clean", "Implemented the parser and committed as abc123. Tests pass (go test ./internal/parser). Pushed to main.", false, false},
		{"empty-clean", "", false, false},
		{"blank-only-clean", "\n\n   \n", false, false},
		{"prose-then-bullets-clean", proseThenBullets, false, false},
		{"bullets-then-prose-refused", bulletsThenProse, false, true},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			rep := ScanClosing(tc.summary, tc.override)
			if rep.Refused() != tc.refused {
				t.Fatalf("ScanClosing(%q, override=%v) Refused()=%v, want %v (verdict=%q, hit=%+v)",
					tc.name, tc.override, rep.Refused(), tc.refused, rep.Verdict, rep.Hit)
			}
			if rep.Schema != ClosingSchema {
				t.Errorf("Schema=%q, want %q", rep.Schema, ClosingSchema)
			}
			if rep.Doctrine != ClosingDoctrine {
				t.Errorf("Doctrine not stamped; got %q", rep.Doctrine)
			}
			if tc.refused {
				if rep.Verdict != ClosingProseWall {
					t.Errorf("verdict=%q, want %q", rep.Verdict, ClosingProseWall)
				}
				if rep.Hit == nil {
					t.Errorf("refused report has no Hit")
				}
				if strings.TrimSpace(rep.Resolve) == "" {
					t.Errorf("refused report has no Resolve remediation")
				}
			}
			if tc.override && rep.Overridden != true {
				t.Errorf("Overridden not recorded on an escaped report")
			}
		})
	}
}

// TestClosingDoctrineBindsAgentsMd welds ClosingDoctrine to AGENTS.md: the enforcing
// const must be present verbatim in the doctrine file, so rewording one without the
// other reds the build. Mirrors TestLeftoversDoctrineBindsAgentsMd.
func TestClosingDoctrineBindsAgentsMd(t *testing.T) {
	path := filepath.Join("..", "..", "AGENTS.md")
	b, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("read %s: %v", path, err)
	}
	if !strings.Contains(string(b), ClosingDoctrine) {
		t.Fatalf("AGENTS.md does not contain ClosingDoctrine verbatim:\n%q", ClosingDoctrine)
	}
}
