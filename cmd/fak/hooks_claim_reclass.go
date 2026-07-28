package main

import (
	"flag"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/hooks"
)

// hooks_claim_reclass.go — `fak hooks claim-reclass`: the forward-only cure rung for an
// already-landed CLAIM_UNWITNESSED residual (#5434).
//
// The push-seam claim-honesty gate reviews the whole `origin/<trunk>..HEAD` range and refuses on
// any commit whose subject claims something its diff cannot witness. Its stated remedy — amend the
// subject — is unavailable on this shared trunk, where a rebase rewrites peers' commits, so a
// single mistyped subject wedges every later commit and `FLEET_ALLOW_RESIDUAL=1` becomes the only
// exit. This rung supplies the missing remedy: it reads the review's OWN residual list, resolves
// each named commit's real subject and diff from git, and clears the range only when every
// residual carries a verified forward-only reclassification in docs/claim-reclass.txt.
//
// The rung is RELAX-ONLY by construction. The shell hook consults it exclusively after the review
// has already refused, and its verdict can only turn that standing refusal into an allow — it can
// never introduce a refusal for a push the review passed, so it cannot wedge a lane that is green.
//
// Exit contract, mirroring the sibling push rungs:
//
//	0  every reported residual is cured by a verified reclassification — allow the push
//	1  at least one residual is uncured — the standing refusal stands
//	2  could-not-run (no review text, unreadable repo, unparseable review output) — the standing
//	   refusal also stands, because a rung that cannot read its evidence must never clear a claim
func runHooksClaimReclass(stdout, stderr io.Writer, stdin io.Reader, argv []string) int {
	fs := flag.NewFlagSet("hooks claim-reclass", flag.ContinueOnError)
	fs.SetOutput(stderr)
	root := fs.String("root", "", "repo root (default: git toplevel from cwd)")
	reviewFile := fs.String("review-file", "", "file holding the claim-honesty review output (default: stdin)")
	ledger := fs.String("ledger", "", "reclassification ledger path (default: <root>/"+hooks.ReclassFile+")")
	asJSON := fs.Bool("json", false, "emit the gate result as JSON")
	if !parseFlags(fs, argv) {
		return 2
	}
	r := resolveRoot(*root)
	if r == "" {
		fmt.Fprintln(stderr, "fak hooks claim-reclass: not in a git repo (or git unavailable); cure rung skipped")
		return 2
	}

	review, err := readClaimReview(*reviewFile, stdin)
	if err != nil {
		fmt.Fprintf(stderr, "fak hooks claim-reclass: could not read the review output: %v; cure rung skipped\n", err)
		return 2
	}
	residuals := hooks.ParseResidualCommits(review)
	if len(residuals) == 0 {
		fmt.Fprintln(stderr, "fak hooks claim-reclass: no residual commit ids could be read from the review output; cure rung skipped (the standing refusal stands)")
		return 2
	}

	ledgerPath := strings.TrimSpace(*ledger)
	if ledgerPath == "" {
		ledgerPath = filepath.Join(r, filepath.FromSlash(hooks.ReclassFile))
	}
	var records []hooks.Reclass
	var problems []string
	if raw, rerr := os.ReadFile(ledgerPath); rerr == nil {
		records, problems = hooks.ParseReclassRecords(string(raw))
	}

	res := hooks.ClearResiduals(residuals, records, func(id string) (hooks.CommitFacts, bool) {
		return claimCommitFacts(r, id)
	})

	if *asJSON {
		if encErr := writeIndentedJSON(stdout, map[string]any{"result": res, "ledger": ledgerPath, "problems": problems}); encErr != nil {
			fmt.Fprintf(stderr, "fak hooks claim-reclass: %v\n", encErr)
			return 2
		}
	} else {
		renderClaimReclass(stdout, r, res, ledgerPath, problems)
	}
	if res.OK {
		return 0
	}
	return 1
}

// readClaimReview returns the review text from --review-file, else from stdin.
func readClaimReview(file string, stdin io.Reader) (string, error) {
	if strings.TrimSpace(file) != "" {
		b, err := os.ReadFile(file)
		if err != nil {
			return "", err
		}
		return string(b), nil
	}
	if stdin == nil {
		return "", fmt.Errorf("no --review-file and no stdin")
	}
	b, err := io.ReadAll(io.LimitReader(stdin, 4<<20))
	if err != nil {
		return "", err
	}
	return string(b), nil
}

// claimCommitFacts resolves one reported residual id to the subject and changed-path set git
// itself recorded. Everything the verdict trusts comes from here, never from the ledger.
func claimCommitFacts(root, id string) (hooks.CommitFacts, bool) {
	head, err := gitOut(root, "show", "-s", "--format=%H%n%s", id)
	if err != nil {
		return hooks.CommitFacts{}, false
	}
	lines := strings.Split(strings.ReplaceAll(head, "\r\n", "\n"), "\n")
	if len(lines) < 2 || strings.TrimSpace(lines[0]) == "" {
		return hooks.CommitFacts{}, false
	}
	facts := hooks.CommitFacts{SHA: strings.TrimSpace(lines[0]), Subject: strings.TrimSpace(lines[1])}
	names, err := gitOut(root, "show", "--name-only", "--pretty=format:", "--no-renames", id)
	if err != nil {
		return hooks.CommitFacts{}, false
	}
	for _, ln := range strings.Split(strings.ReplaceAll(names, "\r\n", "\n"), "\n") {
		if p := strings.TrimSpace(ln); p != "" {
			facts.Paths = append(facts.Paths, p)
		}
	}
	return facts, true
}

// renderClaimReclass prints the verdict and — when a residual is still uncured — the exact record
// that would cure it. Handing back the text that resolves the refusal is what makes the cure
// reachable instead of theoretical.
func renderClaimReclass(w io.Writer, root string, res hooks.ReclassGateResult, ledgerPath string, problems []string) {
	fmt.Fprintf(w, "claim-reclass: %d residual(s) reported, %d cleared, %d uncured (ledger: %s)\n",
		len(res.Residuals), len(res.Cleared), len(res.Uncured), ledgerPath)
	for _, p := range problems {
		fmt.Fprintf(w, "  ledger problem: %s\n", p)
	}
	for _, v := range res.Verdicts {
		mark := "REFUSED"
		if v.Accepted {
			mark = "cleared"
		}
		fmt.Fprintf(w, "  %s %s: %s\n", v.Commit, mark, v.Reason)
	}
	if res.OK {
		return
	}
	fmt.Fprintf(w, "\n  cure (forward-only, no rewrite): append to %s and commit it, then re-push.\n", hooks.ReclassFile)
	fmt.Fprintln(w, "  A record may only DEMOTE a claim to a type the commit's own diff already witnesses;")
	fmt.Fprintln(w, "  it can never restate an unwitnessed code-effect claim, so it cannot launder a subject.")
	for _, id := range res.Uncured {
		facts, ok := claimCommitFacts(root, id)
		if !ok {
			continue
		}
		fmt.Fprintf(w, "\n  # %s — landed subject: %s\n", id, facts.Subject)
		for _, ln := range strings.Split(strings.TrimRight(hooks.ReclassTemplate(facts), "\n"), "\n") {
			fmt.Fprintf(w, "  %s\n", ln)
		}
	}
}
