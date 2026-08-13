package fieldborrow_test

import (
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"testing"
)

func repoRoot(t *testing.T) string {
	t.Helper()
	_, file, _, ok := runtime.Caller(0)
	if !ok {
		t.Fatal("runtime.Caller failed")
	}
	return filepath.Clean(filepath.Join(filepath.Dir(file), "..", "..", ".."))
}

func readRepoFile(t *testing.T, rel string) string {
	t.Helper()
	b, err := os.ReadFile(filepath.Join(repoRoot(t), filepath.FromSlash(rel)))
	if err != nil {
		t.Fatalf("read %s: %v", rel, err)
	}
	return string(b)
}

func requirePhrases(t *testing.T, rel string, phrases ...string) {
	t.Helper()
	body := readRepoFile(t, rel)
	for _, phrase := range phrases {
		if !strings.Contains(body, phrase) {
			t.Errorf("%s does not carry contract phrase %q", rel, phrase)
		}
	}
}

// TestInspiredByContract pins the six operator-requested improvements in the authoritative
// field-borrow skill: proactive priority, lawful direct copying, exhaustive breadth, dated
// observations, broad source mining, and spirit-level exploration of unshipped ideas.
func TestInspiredByContract(t *testing.T) {
	requirePhrases(t, ".claude/skills/field-borrow/SKILL.md",
		"High-priority default \"inspired by\" workflow",
		"Do not wait for an explicit",
		"DIRECT-PORT",
		"Do **not** default to inspire-only",
		"all relevant source classes were",
		"observed_at",
		"source_event_at",
		"open",
		"closed** issues",
		"closed, and open PRs",
		"Spirit extensions",
		"source fact -> inferred principle -> fak opportunity -> disconfirming check",
		"partial prototype",
	)
}

// TestStudyAndScoutCannotNarrowInspiredBy prevents the common regression where the core skill
// is strong but its default callers still request a README/code-only, inspire-only pass.
func TestStudyAndScoutCannotNarrowInspiredBy(t *testing.T) {
	requirePhrases(t, ".claude/skills/study-repo/SKILL.md",
		"Invoke proactively",
		"open and closed issues",
		"closed, and open PRs",
		"observed_at",
		"DIRECT-PORT (preferred when compatible)",
		"spirit",
		"upstream prototype may inspire exploration",
	)
	requirePhrases(t, ".claude/goal-prompts/scout-and-study-witnessed.md",
		"/study-repo and /field-borrow",
		"full evidence surface",
		"open+closed issues",
		"DIRECT-PORT or ADAPT",
		"spirit-level",
	)
}

// TestBoundedSupersetContract pins the portfolio-level behavior: lead with the strongest
// reasonable default, preserve useful cohort-specific alternatives behind modular seams,
// and explicitly bound coverage by evidence, scope, support economics, and freshness.
func TestBoundedSupersetContract(t *testing.T) {
	requirePhrases(t, ".claude/skills/field-borrow/SKILL.md",
		"bounded capability superset",
		"DEFAULT",
		"OPTIONAL-MODULE",
		"RECIPE",
		"WATCH",
		"EXCLUDE",
		"reasonable-coverage budget",
		"Default frontier",
		"Coverage frontier",
		"boiling the ocean",
	)
	requirePhrases(t, ".claude/skills/study-repo/SKILL.md",
		"bounded-superset portfolio rule",
		"technique that loses globally",
		"modular leaves/adapters/integrations",
		"moment-in-time findings",
	)
	requirePhrases(t, ".claude/skills/industry-score/SKILL.md",
		"default and the reasonable coverage envelope",
		"Default frontier",
		"Coverage frontier",
		"feature pile",
		"support-uneconomic",
	)
	requirePhrases(t, ".claude/skills/sota-check/SKILL.md",
		"best default plus bounded coverage",
		"globally second-place approach",
		"boil-the-ocean accumulation",
	)
	requirePhrases(t, ".claude/skills/scout-loop/SKILL.md",
		"bounded-superset",
		"credible user/job/constraint cohort",
		"scope and support economics",
	)
}
