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
	"time"

	"github.com/anthony-chaudhary/fak/internal/conceptcatalog"
	"github.com/anthony-chaudhary/fak/internal/hooks"
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
	case "admission":
		return runConceptAdmission(stdout, stderr, root, args[1:])
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
	var dry, jsonOut, stage bool
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
	fs.BoolVar(&stage, "stage", false, "stage exactly the corpus files written so index-aware admission sees the remedy; use only in an isolated checkout")
	if fs.Parse(args) != nil {
		return 2
	}
	r.DistinctFrom = conceptCSV(refs)
	r.Aliases = conceptCSV(aliases)
	// Plan first: it is pure field validation, while the grounding check walks the
	// whole tree. Ordered the other way, a request that was going to be refused for a
	// missing field still paid the full walk before anyone looked at the field.
	p, e := conceptcatalog.PlanPosition(c, r)
	if e != nil {
		fmt.Fprintln(errw, "fak concept position:", e)
		return 1
	}
	grounded, e := conceptcatalog.ProductionCorpus(filepath.Dir(filepath.Dir(c.Dir)), r.Grounding)
	if e != nil {
		fmt.Fprintln(errw, "fak concept position:", e)
		return 1
	}
	if !grounded {
		fmt.Fprintf(errw, "fak concept position: grounding %q does not appear in the production corpus (tests/build-tag-only text does not count)\n", r.Grounding)
		return 1
	}
	return emitConceptPlan(out, errw, filepath.Dir(filepath.Dir(c.Dir)), p, dry, jsonOut, stage)
}
func runConceptGenerate(out, errw io.Writer, c conceptcatalog.Catalog, args []string) int {
	fs := flag.NewFlagSet("concept generate", flag.ContinueOnError)
	fs.SetOutput(errw)
	jsonOut := fs.Bool("json", false, "emit JSON result")
	if fs.Parse(args) != nil {
		return 2
	}
	if fs.NArg() != 0 {
		fmt.Fprintln(errw, "fak concept generate: unexpected positional arguments")
		return 2
	}
	root := filepath.Dir(filepath.Dir(c.Dir))
	files, err := conceptcatalog.Regenerate(root)
	if err != nil {
		fmt.Fprintln(errw, "fak concept generate:", err)
		return 1
	}
	if *jsonOut {
		_ = json.NewEncoder(out).Encode(map[string]any{"ok": true, "files": relFiles(files)})
		return 0
	}
	fmt.Fprintln(out, "GENERATED concept scorecard")
	for _, file := range relFiles(files) {
		fmt.Fprintln(out, " -", file)
	}
	return 0
}

func runConceptAdmission(out, errw io.Writer, root string, args []string) int {
	fs := flag.NewFlagSet("concept admission", flag.ContinueOnError)
	fs.SetOutput(errw)
	var pathsFlag string
	fs.StringVar(&pathsFlag, "paths", "", "repo-relative paths to evaluate for concept admission (comma-separated)")
	fs.StringVar(&pathsFlag, "path", "", "alias for --paths")
	jsonOut := fs.Bool("json", false, "emit all staged findings as JSON")
	if fs.Parse(args) != nil {
		return 2
	}
	var targetPaths []string
	if pathsFlag != "" {
		for _, p := range strings.Split(pathsFlag, ",") {
			if trimmed := strings.TrimSpace(p); trimmed != "" {
				targetPaths = append(targetPaths, trimmed)
			}
		}
	}
	for _, p := range fs.Args() {
		if trimmed := strings.TrimSpace(p); trimmed != "" {
			targetPaths = append(targetPaths, trimmed)
		}
	}

	var d *hooks.StagedDiff
	var err error
	if len(targetPaths) > 0 {
		d, err = hooks.ReadPathsDiff(root, targetPaths)
	} else {
		d, err = hooks.ReadStagedDiff(root)
	}
	if err != nil {
		fmt.Fprintln(errw, "fak concept admission:", err)
		return 1
	}
	findings, err := hooks.CheckConceptAdmission(d)
	if err != nil {
		fmt.Fprintln(errw, "fak concept admission:", err)
		return 1
	}
	if len(targetPaths) > 0 {
		filter := make(map[string]bool, len(targetPaths))
		for _, p := range targetPaths {
			filter[filepath.ToSlash(p)] = true
		}
		var filtered []hooks.Finding
		for _, f := range findings {
			if filter[filepath.ToSlash(f.File)] {
				filtered = append(filtered, f)
			}
		}
		findings = filtered
	}
	if *jsonOut {
		_ = json.NewEncoder(out).Encode(struct {
			Schema   string          `json:"schema"`
			OK       bool            `json:"ok"`
			Findings []hooks.Finding `json:"findings"`
		}{"fak.concept_admission.v1", len(findings) == 0, findings})
		return 0
	}
	for _, f := range findings {
		fmt.Fprintf(out, "CONCEPT_ADMISSION %s:%d: %s\n", f.File, f.Line, f.Detail)
	}
	return 0
}

type classifyRowsFlag []conceptcatalog.ClassifyRequest

func (f *classifyRowsFlag) String() string { return "" }
func (f *classifyRowsFlag) Set(value string) error {
	var row conceptcatalog.ClassifyRequest
	if err := json.Unmarshal([]byte(value), &row); err != nil {
		return fmt.Errorf("--row must be a JSON classification object: %w", err)
	}
	*f = append(*f, row)
	return nil
}

type conceptClassifyReceipt struct {
	Schema            string                          `json:"schema"`
	OK                bool                            `json:"ok"`
	DryRun            bool                            `json:"dry_run"`
	Staged            bool                            `json:"staged"`
	Idempotent        bool                            `json:"idempotent"`
	BeforeFamilyCount int                             `json:"before_family_count"`
	AfterFamilyCount  int                             `json:"after_family_count"`
	Rows              []conceptcatalog.ClassifyResult `json:"rows"`
	Files             []string                        `json:"files"`
	Timings           conceptcatalog.PhaseTimings     `json:"phase_timings"`
}

func runConceptClassify(out, errw io.Writer, c conceptcatalog.Catalog, args []string) int {
	fs := flag.NewFlagSet("concept classify", flag.ContinueOnError)
	fs.SetOutput(errw)
	var r conceptcatalog.ClassifyRequest
	var rows classifyRowsFlag
	var dry, jsonOut, stage bool
	fs.StringVar(&r.Family, "family", "", "family ID")
	fs.StringVar(&r.Token, "token", "", "token to classify")
	fs.StringVar(&r.Category, "category", "", "incidental|false-positive|test-only|build-tag-only")
	fs.StringVar(&r.Reason, "reason", "", "explicit classification reason")
	fs.Var(&rows, "row", `repeatable JSON row: {"family":"...","token":"...","category":"...","reason":"..."}`)
	fs.BoolVar(&dry, "dry-run", false, "show plan without writing")
	fs.BoolVar(&jsonOut, "json", false, "emit JSON plan")
	fs.BoolVar(&stage, "stage", false, "stage exactly the corpus files written so index-aware admission sees the remedy; use only in an isolated checkout")
	if fs.Parse(args) != nil {
		return 2
	}
	if len(rows) > 0 {
		if r.Family != "" || r.Token != "" || r.Category != "" || r.Reason != "" {
			fmt.Fprintln(errw, "fak concept classify: --row cannot be combined with scalar classification flags")
			return 2
		}
	} else {
		rows = append(rows, r)
	}
	p, e := conceptcatalog.PlanClassifyMany(c, rows)
	if e != nil {
		fmt.Fprintln(errw, "fak concept classify:", e)
		return 1
	}
	return emitConceptPlan(out, errw, filepath.Dir(filepath.Dir(c.Dir)), p, dry, jsonOut, stage)
}
func emitConceptPlan(out, errw io.Writer, root string, p conceptcatalog.Plan, dry, jsonOut, stage bool) int {
	if dry && stage {
		fmt.Fprintln(errw, "fak concept: --stage cannot be combined with --dry-run")
		return 2
	}
	if !dry {
		if e := conceptcatalog.Apply(p); e != nil {
			fmt.Fprintln(errw, "fak concept:", e)
			return 1
		}
	}
	if stage {
		filesToStage := p.Files
		if len(filesToStage) == 0 {
			filesToStage = []string{filepath.Join(root, "tools", "concept_disambiguation_scorecard.data", "_meta.json")}
		}
		if e := stageConceptFiles(root, filesToStage); e != nil {
			fmt.Fprintln(errw, "fak concept: stage remedy:", e)
			return 1
		}
	}
	if jsonOut {
		_ = json.NewEncoder(out).Encode(conceptClassifyReceipt{Schema: "fak.concept_classify_receipt.v1", OK: true, DryRun: dry, Staged: stage, Idempotent: len(p.Files) == 0, BeforeFamilyCount: p.BeforeFamilyCount, AfterFamilyCount: p.AfterFamilyCount, Rows: p.Classifications, Files: relFiles(p.Files), Timings: p.Timings})
		return 0
	}
	fmt.Fprintf(out, "%s %s: family %d -> %d\n", map[bool]string{true: "PLAN", false: "APPLIED"}[dry], p.Mode, p.BeforeFamilyCount, p.AfterFamilyCount)
	for _, f := range relFiles(p.Files) {
		fmt.Fprintln(out, " -", f)
	}
	return 0
}
func stageConceptFiles(root string, files []string) error {
	args := []string{"add", "--"}
	for _, file := range files {
		rel, err := filepath.Rel(root, file)
		if err != nil || rel == ".." || strings.HasPrefix(rel, ".."+string(os.PathSeparator)) {
			return fmt.Errorf("refuse path outside repository: %s", file)
		}
		args = append(args, rel)
	}
	if len(args) == 2 {
		return nil
	}
	var lastErr error
	for attempt := 0; attempt < 60; attempt++ {
		cmd := exec.Command("git", args...)
		cmd.Dir = root
		windowgate.ConfigureBackgroundCommand(cmd)
		output, err := cmd.CombinedOutput()
		if err == nil {
			return nil
		}
		outStr := strings.TrimSpace(string(output))
		lastErr = fmt.Errorf("git add: %w: %s", err, outStr)
		if !strings.Contains(outStr, "index.lock") {
			return lastErr
		}
		time.Sleep(500 * time.Millisecond)
	}
	return lastErr
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
