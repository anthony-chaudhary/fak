package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"os"
	"os/exec"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/dogfoodissues"
	"github.com/anthony-chaudhary/fak/internal/guardroute"
	"github.com/anthony-chaudhary/fak/internal/guardrsi"
	"github.com/anthony-chaudhary/fak/pkg/scorecard"
)

func cmdGuardVerdictRSI(argv []string) { os.Exit(runGuardVerdictRSI(os.Stdout, os.Stderr, argv)) }

func runGuardVerdictRSI(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("fak guard-verdict-rsi", flag.ContinueOnError)
	fs.SetOutput(stderr)
	checkPath := fs.String("check", "", "honesty-gate an emitted iteration JSON")
	if err := fs.Parse(argv); err != nil {
		return 2
	}
	if *checkPath != "" {
		if fs.NArg() != 0 {
			fmt.Fprintf(stderr, "fak guard-verdict-rsi: unexpected argument %q\n", fs.Arg(0))
			return 2
		}
		return runGuardVerdictRSICheck(stdout, stderr, *checkPath)
	}
	args := fs.Args()
	cmd := "run"
	if len(args) > 0 {
		cmd = args[0]
		args = args[1:]
	}
	switch cmd {
	case "fold":
		return runGuardVerdictRSIFold(stdout, stderr, args)
	case "run":
		return runGuardVerdictRSIRun(stdout, stderr, args)
	case "route":
		return runGuardVerdictRSIRoute(stdout, stderr, args)
	case "recovery":
		return runGuardVerdictRSIRecovery(stdout, stderr, args)
	case "-h", "--help", "help":
		guardVerdictRSIUsage(stdout)
		return 0
	default:
		fmt.Fprintf(stderr, "fak guard-verdict-rsi: unknown command %q\n", cmd)
		guardVerdictRSIUsage(stderr)
		return 2
	}
}

func runGuardVerdictRSIFold(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("fak guard-verdict-rsi fold", flag.ContinueOnError)
	fs.SetOutput(stderr)
	audit := fs.String("audit", "", "explicit guard-audit.jsonl to fold")
	workspace := fs.String("workspace", "", "workspace root (default: repo root)")
	asJSON := fs.Bool("json", false, "emit JSON")
	if err := fs.Parse(argv); err != nil {
		return 2
	}
	if fs.NArg() != 0 {
		fmt.Fprintf(stderr, "fak guard-verdict-rsi fold: unexpected argument %q\n", fs.Arg(0))
		return 2
	}
	root := *workspace
	if root == "" {
		root = repoRoot()
	}
	payload := guardrsi.BuildFold(root, *audit)
	if *asJSON {
		return encodeGuardRSIJSON(stdout, stderr, "fak guard-verdict-rsi fold", payload)
	}
	fmt.Fprintf(stdout, "guard-verdict-rsi fold: rows %d  quality %.3g  by_verdict %v\n",
		payload.Fold.TotalRows, payload.VerdictQuality, payload.Fold.ByVerdict)
	if len(payload.JournalPaths) == 0 {
		fmt.Fprintf(stdout, "  (no journal: %s)\n", guardrsi.DiagnoseAuditGap(root))
	}
	return 0
}

func runGuardVerdictRSIRun(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("fak guard-verdict-rsi run", flag.ContinueOnError)
	fs.SetOutput(stderr)
	audit := fs.String("audit", "", "explicit guard-audit.jsonl to fold")
	workspace := fs.String("workspace", "", "workspace root (default: repo root)")
	witnessJSON := fs.String("witness", "", "JSON witness object not authored by the loop, e.g. {\"ok\":true}")
	outPath := fs.String("out", "", "write iteration JSON to this file")
	asJSON := fs.Bool("json", false, "emit JSON")
	if err := fs.Parse(argv); err != nil {
		return 2
	}
	if fs.NArg() != 0 {
		fmt.Fprintf(stderr, "fak guard-verdict-rsi run: unexpected argument %q\n", fs.Arg(0))
		return 2
	}
	root := *workspace
	if root == "" {
		root = repoRoot()
	}
	var witness map[string]any
	if *witnessJSON != "" {
		if err := json.Unmarshal([]byte(*witnessJSON), &witness); err != nil {
			fmt.Fprintf(stderr, "fak guard-verdict-rsi run: parse --witness: %v\n", err)
			return 2
		}
	}
	it := guardrsi.RunIteration(root, *audit, witness)
	if *outPath != "" {
		b, _ := json.MarshalIndent(it, "", "  ")
		if err := os.WriteFile(*outPath, append(b, '\n'), 0o644); err != nil {
			fmt.Fprintf(stderr, "fak guard-verdict-rsi run: write --out: %v\n", err)
			return 1
		}
	}
	if *asJSON {
		return encodeGuardRSIJSON(stdout, stderr, "fak guard-verdict-rsi run", it)
	}
	fmt.Fprintln(stdout, guardrsi.RenderIteration(it))
	return 0
}

// runGuardVerdictRSIRecovery is the refusal-recovery telemetry rung (#2143): for
// every refusal reason token, how often the SAME refused call (tool + args-digest,
// within one session journal) later cleared vs looped, worst token first — the
// measured answer to "which closed-vocabulary fix hint is not landing".
func runGuardVerdictRSIRecovery(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("fak guard-verdict-rsi recovery", flag.ContinueOnError)
	fs.SetOutput(stderr)
	audit := fs.String("audit", "", "explicit guard-audit.jsonl to fold")
	workspace := fs.String("workspace", "", "workspace root (default: repo root)")
	asJSON := fs.Bool("json", false, "emit JSON")
	if err := fs.Parse(argv); err != nil {
		return 2
	}
	if fs.NArg() != 0 {
		fmt.Fprintf(stderr, "fak guard-verdict-rsi recovery: unexpected argument %q\n", fs.Arg(0))
		return 2
	}
	root := *workspace
	if root == "" {
		root = repoRoot()
	}
	rep := guardrsi.BuildRecovery(root, *audit)
	if *asJSON {
		return encodeGuardRSIJSON(stdout, stderr, "fak guard-verdict-rsi recovery", rep)
	}
	fmt.Fprintln(stdout, guardrsi.RenderRecovery(rep))
	if len(rep.JournalPaths) == 0 {
		fmt.Fprintf(stdout, "  (no journal: %s)\n", guardrsi.DiagnoseAuditGap(root))
	}
	return 0
}

// runGuardVerdictRSIRoute is the closure rung: it reviews the session journal,
// decides (purely) whether the worst bucket is route-worthy, and -- when it is --
// materializes the finding through the two EXISTING idempotent sinks:
// tools/findings_route.py (the pickable queue row, always) and internal/dogfoodissues
// (a deduped gh issue, for an honesty-hole). Issue filing is ON by default per the
// operator decision; --no-issues skips it (queue-only, for a host without gh auth)
// and --dry-run plans the issue without touching gh. Fail-open throughout: a sink
// failure is reported, never fatal -- the queue row still gets written if it can.
func runGuardVerdictRSIRoute(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("fak guard-verdict-rsi route", flag.ContinueOnError)
	fs.SetOutput(stderr)
	audit := fs.String("audit", "", "explicit guard-audit.jsonl to fold")
	workspace := fs.String("workspace", "", "workspace root (default: repo root)")
	source := fs.String("source", "guard-verdict-rsi", "originating session/run id recorded on the routed row")
	threshold := fs.Int("threshold", guardroute.DefaultReasonThreshold, "min recurrence of a denial reason before it routes")
	noIssues := fs.Bool("no-issues", false, "skip the gh-issue half (queue-only; for a host without gh)")
	dryRun := fs.Bool("dry-run", false, "plan the gh issue without touching gh (queue row still written)")
	repo := fs.String("repo", "", "gh repo for the issue half (default: current repo)")
	asJSON := fs.Bool("json", false, "emit the control-pane envelope as JSON")
	if err := fs.Parse(argv); err != nil {
		return 2
	}
	if fs.NArg() != 0 {
		fmt.Fprintf(stderr, "fak guard-verdict-rsi route: unexpected argument %q\n", fs.Arg(0))
		return 2
	}
	root := *workspace
	if root == "" {
		root = repoRoot()
	}

	fold := guardrsi.BuildFold(root, *audit)
	decision := guardroute.Decide(fold.Fold, fold.WorstBucket, *threshold)

	var routed, issue map[string]any
	if decision.Route {
		routed = routeQueueRow(stderr, root, decision, *source)
		if decision.FileIssue && !*noIssues {
			issue = routeFileIssue(stderr, decision, fold.JournalPaths, *repo, *dryRun)
		}
	}

	env := guardroute.Fold(decision, routed, issue)
	if *asJSON {
		return encodeGuardRSIJSON(stdout, stderr, "fak guard-verdict-rsi route", env)
	}
	fmt.Fprintf(stdout, "guard-verdict-rsi route: %s (%s)\n", env.Verdict, env.Finding)
	fmt.Fprintf(stdout, "  %s\n", env.Reason)
	if routed != nil {
		fmt.Fprintf(stdout, "  queue: action=%v n=%v sev=%v\n", routed["action"], routed["n"], routed["sev"])
	}
	if issue != nil {
		fmt.Fprintf(stdout, "  issue: %v\n", issue["summary"])
	}
	fmt.Fprintf(stdout, "  -> %s\n", env.NextAction)
	return 0
}

// routeQueueRow shells tools/findings_route.py to append the pickable queue row.
// Returns the parsed verdict dict, or a fail-open error stub -- never nil-on-routed
// so the envelope always records that routing was attempted.
func routeQueueRow(stderr io.Writer, root string, d guardroute.RouteDecision, source string) map[string]any {
	args := append([]string{"tools/findings_route.py"}, guardroute.RouteArgv(d, source)...)
	args = append(args, "--json")
	out, err := runPythonTool(root, args)
	if err != nil {
		fmt.Fprintf(stderr, "guard-verdict-rsi route: findings_route failed (fail-open, finding unrecorded): %v\n", err)
		return map[string]any{"action": "error", "detail": err.Error()}
	}
	var verdict map[string]any
	if jerr := json.Unmarshal(out, &verdict); jerr != nil {
		fmt.Fprintf(stderr, "guard-verdict-rsi route: findings_route output not JSON (fail-open): %v\n", jerr)
		return map[string]any{"action": "error", "detail": "non-JSON findings_route output"}
	}
	return verdict
}

// routeFileIssue runs the dogfoodissues create-vs-update sync for an honesty-hole.
// dryRun plans without touching gh. Fail-open: a gh/auth failure is summarized, not
// fatal.
func routeFileIssue(stderr io.Writer, d guardroute.RouteDecision, journals []string, repo string, dryRun bool) map[string]any {
	evidence := ""
	if len(journals) > 0 {
		evidence = journals[0]
	}
	item := guardroute.ToActionItem(d, evidence)
	plan := dogfoodissues.BuildPlan([]dogfoodissues.ActionItem{item}, nil)
	if dryRun {
		return map[string]any{"mode": "dry-run", "planned": len(plan), "summary": fmt.Sprintf("dry-run: would create/update 1 issue (key=%s)", item.Key)}
	}
	existing, err := dogfoodissues.FetchExistingIssues(repo, 300)
	if err != nil {
		fmt.Fprintf(stderr, "guard-verdict-rsi route: gh issue list failed (fail-open, queue row stands): %v\n", err)
		return map[string]any{"mode": "live", "ok": false, "summary": "gh unavailable -- queue row written, issue skipped"}
	}
	plan = dogfoodissues.BuildPlan([]dogfoodissues.ActionItem{item}, existing)
	synced := dogfoodissues.Sync(plan, repo, []string{"guard-rsi", "dogfood"}, nil)
	ok := true
	for _, row := range synced {
		if !row.OK {
			ok = false
		}
	}
	action := "created"
	if len(plan) > 0 && plan[0].Number != nil {
		action = "updated"
	}
	return map[string]any{"mode": "live", "ok": ok, "summary": fmt.Sprintf("%s issue (key=%s)", action, item.Key)}
}

// runPythonTool shells a repo python tool, trying FAK_PYTHON, then python3/python,
// matching how the rest of cmd/fak shells to Python across OSes (see steering.go).
func runPythonTool(root string, args []string) ([]byte, error) {
	interps := []string{}
	if p := strings.TrimSpace(os.Getenv("FAK_PYTHON")); p != "" {
		interps = append(interps, p)
	}
	interps = append(interps, "python3", "python")
	var lastErr error
	for _, py := range interps {
		cmd := exec.Command(py, args...)
		cmd.Dir = root
		cmd.Stderr = os.Stderr
		out, err := cmd.Output()
		if err == nil {
			return out, nil
		}
		lastErr = err
	}
	return nil, fmt.Errorf("python tool (tried %s): %w", strings.Join(interps, ", "), lastErr)
}

func runGuardVerdictRSICheck(stdout, stderr io.Writer, path string) int {
	b, err := os.ReadFile(path)
	if err != nil {
		fmt.Fprintf(stderr, "fak guard-verdict-rsi --check: read: %v\n", err)
		return 2
	}
	var it guardrsi.Iteration
	if err := json.Unmarshal(b, &it); err != nil {
		fmt.Fprintf(stderr, "fak guard-verdict-rsi --check: parse: %v\n", err)
		return 2
	}
	violations := guardrsi.CheckIteration(it)
	if len(violations) > 0 {
		fmt.Fprintln(stdout, "guard-verdict-rsi --check: FAIL")
		for _, v := range violations {
			fmt.Fprintf(stdout, "  - %s\n", v)
		}
		return 1
	}
	fmt.Fprintln(stdout, "guard-verdict-rsi --check: OK (iteration is honest)")
	return 0
}

func cmdGuardRSIScorecard(argv []string) { os.Exit(runGuardRSIScorecard(os.Stdout, os.Stderr, argv)) }

func runGuardRSIScorecard(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("fak guard-rsi-scorecard", flag.ContinueOnError)
	fs.SetOutput(stderr)
	workspace := fs.String("workspace", "", "workspace root (default: repo root)")
	asJSON := fs.Bool("json", false, "emit control-pane JSON")
	asMarkdown := fs.Bool("markdown", false, "emit scorecard markdown")
	comparePath := fs.String("compare", "", "compare against a prior --json payload")
	if err := fs.Parse(argv); err != nil {
		return 2
	}
	if fs.NArg() != 0 {
		fmt.Fprintf(stderr, "fak guard-rsi-scorecard: unexpected argument %q\n", fs.Arg(0))
		return 2
	}
	root := *workspace
	if root == "" {
		root = repoRoot()
	}
	payload := guardrsi.BuildScorecard(root)
	if *comparePath != "" {
		base, ok := readCompareBase(stderr, "fak guard-rsi-scorecard", *comparePath)
		if !ok {
			return 2
		}
		fmt.Fprintln(stdout, scorecard.Compare(payload, base, guardrsi.DebtKey))
		if payload.OK {
			return 0
		}
		return 1
	}
	if *asJSON {
		_ = encodeGuardRSIJSON(stdout, stderr, "fak guard-rsi-scorecard", payload)
	} else if *asMarkdown {
		fmt.Fprint(stdout, scorecard.Markdown(payload, scorecard.MarkdownDoc{
			Title:       "fak guard RSI loop scorecard",
			Description: "How mature and realized the RSI loop(s) for fak guard are, scored from the tree plus the real decision journal.",
			Heading:     "fak guard RSI loop scorecard",
			DebtKey:     guardrsi.DebtKey,
			HeaderExtra: fmt.Sprintf(" - maturity value %v - realized value %v - %v real journal row(s)",
				payload.Corpus["maturity_value"], payload.Corpus["realized_value"], payload.Corpus["audit_rows"]),
		}))
	} else {
		fmt.Fprintln(stdout, scorecard.Render(payload, guardrsi.DebtKey))
	}
	if payload.OK {
		return 0
	}
	return 1
}

func encodeGuardRSIJSON(stdout, stderr io.Writer, label string, v any) int {
	return encodeJSONOrFail(stdout, stderr, v, label)
}

func guardVerdictRSIUsage(w io.Writer) {
	fmt.Fprint(w, `fak guard-verdict-rsi - RSI loop over the real guard decision journal

  fak guard-verdict-rsi fold     [--audit FILE] [--json]
  fak guard-verdict-rsi run      [--audit FILE] [--witness JSON] [--json] [--out FILE]
  fak guard-verdict-rsi route    [--source ID] [--threshold N] [--no-issues] [--dry-run] [--repo R] [--json]
  fak guard-verdict-rsi recovery [--audit FILE] [--json]
  fak guard-verdict-rsi --check ITER.json

  recovery measures, per refusal reason token, how often the refused call (same
  tool + args-digest, within one session journal) later cleared vs looped — the
  refusal-recovery success telemetry that surfaces the fix hint that is not landing.

  route closes the loop: it reviews the session journal and, when the worst bucket is
  a real finding, routes a pickable findings-queue row (always) plus a deduped gh issue
  for an honesty-hole (P1/P0; on by default, --no-issues for queue-only). Fail-open.
`)
}
