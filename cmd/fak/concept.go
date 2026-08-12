package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/conceptcatalog"
	"github.com/anthony-chaudhary/fak/internal/windowgate"
)

func runConcept(stdout, stderr io.Writer, args []string) int {
	if len(args) == 0 {
		fmt.Fprintln(stderr, "usage: fak concept <position|classify|validate|freshness|admission> [flags]")
		return 2
	}
	root, err := conceptRoot()
	if err != nil {
		fmt.Fprintln(stderr, "fak concept:", err)
		return 1
	}
	c, err := conceptcatalog.Load(root)
	if err != nil {
		fmt.Fprintln(stderr, "fak concept:", err)
		return 1
	}
	switch args[0] {
	case "position":
		return runConceptPosition(stdout, stderr, c, args[1:])
	case "classify":
		return runConceptClassify(stdout, stderr, c, args[1:])
	case "freshness":
		fs := flag.NewFlagSet("concept freshness", flag.ContinueOnError)
		fs.SetOutput(stderr)
		check := fs.Bool("check", true, "check tracked generated artifacts")
		jsonOut := fs.Bool("json", false, "emit JSON")
		if fs.Parse(args[1:]) != nil {
			return 2
		}
		if !*check {
			fmt.Fprintln(stderr, "fak concept freshness: --check=false is unsupported; use: "+conceptcatalog.RegenerateCommand)
			return 2
		}
		// The (result, error) pair is folded by the renderer, never read in order
		// here: encoding the result first printed `{"fresh":true}` on stdout for a
		// check that had just failed (#5962).
		res, e := conceptcatalog.CheckFresh(root)
		return conceptcatalog.RenderFreshness(stdout, stderr, "fak concept freshness", "", res, e, *jsonOut)
	case "generate":
		cmd := exec.Command("python", filepath.Join(root, "tools", "concept_disambiguation_scorecard.py"), "--workspace", root, "--markdown-dir", filepath.Join(root, "docs", "concept-disambiguation-scorecard"))
		cmd.Dir = root
		windowgate.ConfigureBackgroundCommand(cmd)
		cmd.Stdout = stdout
		cmd.Stderr = stderr
		cmd.Env = os.Environ()
		if e := cmd.Run(); e != nil {
			// Exit 1 is an honest ACTION snapshot, not a generator failure - but only
			// if EVERY tracked artifact was actually written.
			for _, tracked := range []string{conceptcatalog.GeneratedReadme, conceptcatalog.GeneratedIndex} {
				if _, statErr := os.Stat(filepath.Join(root, filepath.FromSlash(tracked))); statErr != nil {
					return 1
				}
			}
		}
		return 0
	case "validate":
		ds := conceptcatalog.Validate(c)
		if len(ds) > 0 {
			_ = json.NewEncoder(stdout).Encode(map[string]any{"ok": false, "diagnostics": ds})
			return 1
		}
		fmt.Fprintln(stdout, "concept catalog valid")
		return 0
	default:
		fmt.Fprintf(stderr, "fak concept: unknown subcommand %q\n", args[0])
		return 2
	}
}
func conceptRoot() (string, error) {
	out, err := runGitOutput("rev-parse", "--show-toplevel")
	if err != nil {
		return "", err
	}
	return strings.TrimSpace(out), nil
}
func runConceptPosition(out, errw io.Writer, c conceptcatalog.Catalog, args []string) int {
	fs := flag.NewFlagSet("concept position", flag.ContinueOnError)
	fs.SetOutput(errw)
	var r conceptcatalog.PositionRequest
	var refs, aliases string
	var dry, jsonOut bool
	fs.StringVar(&r.ID, "id", "", "stable row ID")
	fs.StringVar(&r.Canonical, "canonical", "", "canonical name")
	fs.StringVar(&r.Family, "family", "", "family ID")
	fs.StringVar(&r.Definition, "definition", "", "definition")
	fs.StringVar(&r.Distinction, "distinction", "", "boundary from siblings")
	fs.StringVar(&r.Kind, "kind", "symbol", "concept kind")
	fs.StringVar(&r.Grounding, "grounding", "", "grounding token")
	fs.StringVar(&r.GroundingKind, "grounding-kind", "symbol", "grounding kind")
	fs.StringVar(&r.Glossary, "glossary", "docs/fak/concept-glossary.md", "glossary path")
	fs.StringVar(&r.RowFile, "row-file", "", "narrow rows-*.json target")
	fs.StringVar(&refs, "distinct-from", "", "comma-separated row IDs")
	fs.StringVar(&aliases, "aliases", "", "comma-separated aliases")
	fs.BoolVar(&dry, "dry-run", false, "show plan without writing")
	fs.BoolVar(&jsonOut, "json", false, "emit JSON plan")
	if fs.Parse(args) != nil {
		return 2
	}
	r.DistinctFrom = conceptCSV(refs)
	r.Aliases = conceptCSV(aliases)
	grounded, e := conceptcatalog.ProductionCorpus(filepath.Dir(filepath.Dir(c.Dir)), r.Grounding)
	if e != nil {
		fmt.Fprintln(errw, "fak concept position:", e)
		return 1
	}
	if !grounded {
		fmt.Fprintf(errw, "fak concept position: grounding %q does not appear in the production corpus (tests/build-tag-only text does not count)\n", r.Grounding)
		return 1
	}
	p, e := conceptcatalog.PlanPosition(c, r)
	if e != nil {
		fmt.Fprintln(errw, "fak concept position:", e)
		return 1
	}
	return emitConceptPlan(out, errw, p, dry, jsonOut)
}
func runConceptClassify(out, errw io.Writer, c conceptcatalog.Catalog, args []string) int {
	fs := flag.NewFlagSet("concept classify", flag.ContinueOnError)
	fs.SetOutput(errw)
	var r conceptcatalog.ClassifyRequest
	var dry, jsonOut bool
	fs.StringVar(&r.Family, "family", "", "family ID")
	fs.StringVar(&r.Token, "token", "", "token to classify")
	fs.StringVar(&r.Category, "category", "", "incidental|false-positive|test-only|build-tag-only")
	fs.StringVar(&r.Reason, "reason", "", "explicit classification reason")
	fs.BoolVar(&dry, "dry-run", false, "show plan without writing")
	fs.BoolVar(&jsonOut, "json", false, "emit JSON plan")
	if fs.Parse(args) != nil {
		return 2
	}
	p, e := conceptcatalog.PlanClassify(c, r)
	if e != nil {
		fmt.Fprintln(errw, "fak concept classify:", e)
		return 1
	}
	return emitConceptPlan(out, errw, p, dry, jsonOut)
}
func emitConceptPlan(out, errw io.Writer, p conceptcatalog.Plan, dry, jsonOut bool) int {
	if !dry {
		if e := conceptcatalog.Apply(p); e != nil {
			fmt.Fprintln(errw, "fak concept:", e)
			return 1
		}
	}
	if jsonOut {
		_ = json.NewEncoder(out).Encode(map[string]any{"ok": true, "dry_run": dry, "mode": p.Mode, "family": p.Family, "before_family_count": p.BeforeFamilyCount, "after_family_count": p.AfterFamilyCount, "files": relFiles(p.Files)})
		return 0
	}
	fmt.Fprintf(out, "%s %s: family %d -> %d\n", map[bool]string{true: "PLAN", false: "APPLIED"}[dry], p.Mode, p.BeforeFamilyCount, p.AfterFamilyCount)
	for _, f := range relFiles(p.Files) {
		fmt.Fprintln(out, " -", f)
	}
	return 0
}
func conceptCSV(s string) []string {
	var out []string
	for _, x := range strings.Split(s, ",") {
		if x = strings.TrimSpace(x); x != "" {
			out = append(out, x)
		}
	}
	return out
}
func relFiles(in []string) []string {
	root, _ := conceptRoot()
	out := make([]string, 0, len(in))
	for _, p := range in {
		if r, e := filepath.Rel(root, p); e == nil && !strings.HasPrefix(r, "..") {
			p = r
		}
		out = append(out, filepath.ToSlash(p))
	}
	return out
}
func runGitOutput(args ...string) (string, error) {
	cmd := exec.Command("git", args...)
	windowgate.ConfigureBackgroundCommand(cmd)
	b, e := cmd.Output()
	return string(b), e
}
