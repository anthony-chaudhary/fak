package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"os"
	"text/tabwriter"

	"github.com/anthony-chaudhary/fak/internal/ctxplans"
)

//fak:ctxplan verb=ctxplans enters="nothing live — an offline scan over cmd/fak/*.go and .claude/skills/*/SKILL.md in the specified workspace root" pages="nothing into a model window — renders a human-readable table or JSON report of context-touching surfaces and their context-plan declarations" warms="nothing — pure static filesystem lint; warms no prompt cache or KV"

func cmdCtxPlans(argv []string) {
	os.Exit(runCtxPlans(os.Stdout, os.Stderr, argv))
}

func runCtxPlans(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("ctxplans", flag.ContinueOnError)
	fs.SetOutput(stderr)
	asJSON := fs.Bool("json", false, "emit the report as JSON")
	root := fs.String("root", ".", "workspace root directory to scan")
	fs.Usage = func() {
		fmt.Fprintln(stderr, "usage: fak ctxplans [--json] [--root PATH]")
		fmt.Fprintln(stderr, "  Advisory lint reporting context-touching surfaces and plan declarations.")
	}
	if code, done := parseFlagsRejectArgs(fs, argv, stderr); done {
		return code
	}

	rep, err := ctxplans.Scan(*root)
	if err != nil {
		fmt.Fprintf(stderr, "fak ctxplans: %v\n", err)
		return 1
	}

	if *asJSON {
		enc := json.NewEncoder(stdout)
		enc.SetIndent("", "  ")
		if err := enc.Encode(rep); err != nil {
			fmt.Fprintf(stderr, "fak ctxplans: encode json: %v\n", err)
			return 1
		}
		return 0
	}

	tw := tabwriter.NewWriter(stdout, 0, 0, 2, ' ', 0)
	fmt.Fprintf(tw, "KIND\tSURFACE\tCONTEXT PLAN\tPROVENANCE\n")
	for _, s := range rep.Surfaces {
		status := "UNDECLARED"
		prov := ""
		if s.Declared {
			status = "declared"
			prov = fmt.Sprintf("%s:%d", s.File, s.Line)
		}
		fmt.Fprintf(tw, "%s\t%s\t%s\t%s\n", s.Kind, s.Name, status, prov)
	}
	_ = tw.Flush()
	fmt.Fprintf(stdout, "\ndeclared verbs: %d (floor 10)   undeclared-surface debt: %d (advisory)\n",
		rep.DeclaredVerbs, rep.Debt)
	return 0
}
