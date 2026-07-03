package main

import (
	"flag"
	"fmt"
	"io"
	"os"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/renameconcept"
)

func cmdRenameConcept(argv []string) { os.Exit(runRenameConcept(os.Stdout, os.Stderr, argv)) }

// runRenameConcept plans (default) or applies (--apply) a concept rename across
// the tree: expands the old/new spelling into case-form variants, inventories
// every touched site, and triages MECHANICAL (rewritten) vs HOLDOUT (history /
// binary artifacts / irregular casings — listed, never touched). Exit codes:
// 0 plan built / applied clean, 1 apply error or mechanical residual, 2 usage.
func runRenameConcept(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("fak rename-concept", flag.ContinueOnError)
	fs.SetOutput(stderr)
	workspace := fs.String("workspace", "", "workspace root (default: repo root)")
	from := fs.String("from", "", "current concept spelling; mark word boundaries with '-'/'_'/humps (e.g. dgx-bridge)")
	to := fs.String("to", "", "replacement spelling in the same word form")
	apply := fs.Bool("apply", false, "apply the mechanical share (default: dry-run plan). NOTE: rewrites the shared working tree — plan first")
	includeHistorical := fs.Bool("include-historical", false, "also rewrite history-shaped holdouts (dated notes, JSONL/CSV records)")
	asJSON := fs.Bool("json", false, "emit the plan/apply report as JSON")
	if !parseFlags(fs, argv) {
		return 2
	}
	if args := fs.Args(); *from == "" && *to == "" && len(args) == 2 {
		*from, *to = args[0], args[1]
	} else if len(args) != 0 {
		fmt.Fprintf(stderr, "fak rename-concept: unexpected argument %q (use --from/--to or exactly two positionals)\n", args[0])
		return 2
	}
	if *from == "" || *to == "" {
		fmt.Fprintln(stderr, "fak rename-concept: both --from and --to are required (e.g. fak rename-concept --from dgxbridge --to slackbridge)")
		return 2
	}
	root := *workspace
	if root == "" {
		root = repoRoot()
	}
	opts := renameconcept.Options{Root: root, From: *from, To: *to, IncludeHistorical: *includeHistorical}

	if *apply {
		res, err := renameconcept.Apply(opts)
		if err != nil {
			fmt.Fprintf(stderr, "fak rename-concept: %v\n", err)
			return 2
		}
		if *asJSON {
			if err := writeIndentedJSONNoEscape(stdout, res); err != nil {
				fmt.Fprintf(stderr, "fak rename-concept: encode json: %v\n", err)
				return 1
			}
		} else {
			renderApply(stdout, res)
		}
		if res.OK {
			return 0
		}
		return 1
	}

	plan, err := renameconcept.BuildPlan(opts)
	if err != nil {
		fmt.Fprintf(stderr, "fak rename-concept: %v\n", err)
		return 2
	}
	if *asJSON {
		if err := writeIndentedJSONNoEscape(stdout, plan); err != nil {
			fmt.Fprintf(stderr, "fak rename-concept: encode json: %v\n", err)
			return 1
		}
		return 0
	}
	renderPlan(stdout, plan)
	return 0
}

func renderPlan(w io.Writer, p renameconcept.Plan) {
	fmt.Fprintf(w, "rename-concept %s -> %s  (%s)\n", p.From, p.To, p.Finding)
	fmt.Fprintf(w, "  scanned %d file(s): %d exact match(es), %d irregular-case hit(s)\n",
		p.FilesScanned, p.TotalMatches, p.IrregularMatches)
	var olds []string
	for _, v := range p.Variants {
		olds = append(olds, v.Old)
	}
	fmt.Fprintf(w, "  variants: %s\n", strings.Join(olds, " "))
	if len(p.Renames) > 0 {
		fmt.Fprintf(w, "\n  PATH RENAMES (%d, deepest-first):\n", len(p.Renames))
		for _, r := range p.Renames {
			fmt.Fprintf(w, "    %s -> %s\n", r.From, r.To)
		}
	}
	if len(p.Mechanical) > 0 {
		fmt.Fprintf(w, "\n  MECHANICAL (%d — --apply rewrites these):\n", len(p.Mechanical))
		for _, s := range p.Mechanical {
			fmt.Fprintf(w, "    %-8s %s  (%d match(es)%s)\n", s.Kind, s.Path, s.Matches, irregularSuffix(s.Irregular))
		}
	}
	if len(p.Holdouts) > 0 {
		fmt.Fprintf(w, "\n  HOLDOUTS (%d — listed, never touched):\n", len(p.Holdouts))
		for _, s := range p.Holdouts {
			fmt.Fprintf(w, "    %-8s %s  (%d match(es)%s) — %s\n", s.Kind, s.Path, s.Matches, irregularSuffix(s.Irregular), s.Hold)
		}
	}
	fmt.Fprintf(w, "\n  NEXT: %s\n", p.NextAction)
}

// irregularSuffix annotates a site line with the case-insensitive hits no
// variant covers — the share --apply will NOT rewrite.
func irregularSuffix(n int) string {
	if n == 0 {
		return ""
	}
	return fmt.Sprintf(", %d irregular stay(s)", n)
}

func renderApply(w io.Writer, res renameconcept.ApplyResult) {
	fmt.Fprintf(w, "rename-concept apply: %d file(s) rewritten, %d path(s) renamed, %d holdout(s) skipped\n",
		len(res.RewrittenFiles), len(res.RenamedPaths), len(res.SkippedHolds))
	for _, r := range res.RenamedPaths {
		fmt.Fprintf(w, "    renamed  %s -> %s\n", r.From, r.To)
	}
	for _, f := range res.RewrittenFiles {
		fmt.Fprintf(w, "    rewrote  %s\n", f)
	}
	for _, s := range res.SkippedHolds {
		fmt.Fprintf(w, "    held     %s — %s\n", s.Path, s.Hold)
	}
	for _, e := range res.Errors {
		fmt.Fprintf(w, "    ERROR    %s\n", e)
	}
	if res.Residual != nil {
		fmt.Fprintf(w, "  residual: %d mechanical site(s), %d holdout(s), %d irregular hit(s)\n",
			len(res.Residual.Mechanical), len(res.Residual.Holdouts), res.Residual.IrregularMatches)
	}
	fmt.Fprintf(w, "  NEXT: %s\n", res.NextAction)
}
