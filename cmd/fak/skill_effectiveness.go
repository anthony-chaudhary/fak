package main

// skill_effectiveness.go is the THIN shell over internal/skilleffectiveness -- the pure
// core that probes `.claude/skills/*/SKILL.md` for the affordances that make a skill
// discoverable, triggerable, and affordable, plus the queried loader's own health. The
// shell owns flags and exit codes only; every number comes from the package, and the
// rendering comes from the shared pkg/scorecard kernel via emitScorecard, exactly like
// the conflation/checkpoint/unwired cards (the propagation-family convention).

import (
	"flag"
	"fmt"
	"io"
	"os"

	"github.com/anthony-chaudhary/fak/internal/skilleffectiveness"
)

func cmdSkillEffectivenessScorecard(argv []string) {
	os.Exit(runSkillEffectivenessScorecard(os.Stdout, os.Stderr, argv))
}

func runSkillEffectivenessScorecard(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("fak skill-effectiveness-scorecard", flag.ContinueOnError)
	fs.SetOutput(stderr)
	workspace := fs.String("workspace", "", "workspace root (default: repo root)")
	asJSON := fs.Bool("json", false, "emit control-pane JSON")
	asMarkdown := fs.Bool("markdown", false, "emit the committed snapshot body")
	comparePath := fs.String("compare", "", "compare against a prior --json payload")
	if !parseFlags(fs, argv) {
		return 2
	}
	if fs.NArg() != 0 {
		fmt.Fprintf(stderr, "fak skill-effectiveness-scorecard: unexpected argument %q\n", fs.Arg(0))
		return 2
	}
	root := *workspace
	if root == "" {
		root = repoRoot()
	}
	payload := skilleffectiveness.Build(root)

	return emitScorecard(stdout, stderr, "fak skill-effectiveness-scorecard", skilleffectiveness.DebtKey,
		payload, *comparePath, *asJSON, *asMarkdown, skilleffectiveness.MarkdownDoc(payload))
}

// collectSkillEffectivenessScorecard renders Build's kernel payload in the loose map shape
// the cmd-level regression tests read (cmd/fak/skill_test.go, cmd/fak/skill_budget_test.go).
// The corpus map is passed through by reference rather than JSON round-tripped, so its
// counters stay `int` and a caller can compare them directly.
func collectSkillEffectivenessScorecard(root string) map[string]any {
	p := skilleffectiveness.Build(root)
	return map[string]any{
		"schema":      p.Schema,
		"ok":          p.OK,
		"verdict":     p.Verdict,
		"finding":     p.Finding,
		"reason":      p.Reason,
		"next_action": p.NextAction,
		"corpus":      p.Corpus,
	}
}

// collectLoaderDebt is the cmd-level view of the queried-loader dimension: the debt total
// plus its three components (un-queryable cards, cards that failed to page, out-of-sync rows).
func collectLoaderDebt(root string) (loaderDebt, queryable, pages, inSync int) {
	l := skilleffectiveness.ScanLoader(root)
	return l.Debt(), l.Queryable, l.Pages, l.InSync
}
