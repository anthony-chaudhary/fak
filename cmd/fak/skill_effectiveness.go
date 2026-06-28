package main

import (
	"flag"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strings"
)

// scorecardCmdSetup runs the shared front-half of a scorecard subcommand: it parses
// the common --json/--markdown flags, collects the payload, and on the --json path
// emits it and signals the caller to stop. It returns the payload p, its corpus map
// c, whether --markdown was requested, and done=true when the --json branch already
// rendered (the caller returns immediately). On a flag parse error it exits(2), the
// same as the inline form it replaces.
func scorecardCmdSetup(name string, argv []string, collect func(string) map[string]any) (p, c map[string]any, asMarkdown, done bool) {
	fs := flag.NewFlagSet(name, flag.ContinueOnError)
	asJSON := fs.Bool("json", false, "emit machine-readable scorecard JSON")
	md := fs.Bool("markdown", false, "emit markdown")
	fs.SetOutput(io.Discard)
	if err := fs.Parse(argv); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(2)
	}
	p = collect(repoRoot())
	if *asJSON {
		_ = writeIndentedJSONNoEscape(os.Stdout, p)
		return p, nil, false, true
	}
	return p, p["corpus"].(map[string]any), *md, false
}

func cmdSkillEffectivenessScorecard(argv []string) {
	p, c, asMarkdown, done := scorecardCmdSetup("fak skill-effectiveness-scorecard", argv, collectSkillEffectivenessScorecard)
	if done {
		return
	}
	if asMarkdown {
		fmt.Printf("# fak skill-effectiveness scorecard\n\n**skill_debt: %v** across **%v** skills.\n", c["skill_debt"], c["skills"])
		return
	}
	fmt.Printf("skill-effectiveness-scorecard: %s (%s)\n  skill_debt: %v   skills: %v\n", p["verdict"], p["finding"], c["skill_debt"], c["skills"])
}

func collectSkillEffectivenessScorecard(root string) map[string]any {
	matches, _ := filepath.Glob(filepath.Join(root, ".claude", "skills", "*", "SKILL.md"))
	debt := 0
	for _, path := range matches {
		b, err := os.ReadFile(path)
		if err != nil {
			debt++
			continue
		}
		text := string(b)
		if !strings.Contains(text, "description:") {
			debt++
		}
		if !strings.Contains(strings.ToLower(text), "use when") && !strings.Contains(strings.ToLower(text), "use to") {
			debt++
		}
	}
	score := 100
	grade := "A"
	ok, verdict, finding := true, "OK", "skills_effective"
	reason := "all discovered skills carry the minimal trigger affordances"
	next := "rerun after changing .claude/skills"
	if debt > 0 {
		ok, verdict, finding = false, "ACTION", "skill_debt"
		score, grade = 85, "B"
		reason = fmt.Sprintf("%d skill affordance debt unit(s)", debt)
		next = "add missing front-matter descriptions or trigger clauses"
	}
	return map[string]any{
		"schema":      "fak-skill-effectiveness-scorecard/1",
		"ok":          ok,
		"verdict":     verdict,
		"finding":     finding,
		"reason":      reason,
		"next_action": next,
		"corpus": map[string]any{
			"skill_debt": debt,
			"skills":     len(matches),
			"score":      score,
			"grade":      grade,
		},
	}
}
