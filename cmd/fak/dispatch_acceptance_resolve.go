package main

// dispatch_acceptance_resolve.go — `fak dispatch acceptance-resolve`, the I/O shell
// for acceptance-SYMBOL closure resolution (#5435).
//
// A fleet wave dispatched against the P0/P1 view spent 2 of 7 worker slots on issues
// that had already shipped, because the only binding between a landing and an issue
// was the issue number in the commit subject — and the highest-value stale-opens do
// not carry it (a subject names a different issue, or names none, or the acceptance
// item lands inside a peer's unrelated commit). This verb makes the binding the
// auditor actually used: read what the acceptance NAMES, then look for it on the
// trunk.
//
//	# resolve one issue's acceptance against origin/main
//	fak dispatch acceptance-resolve --issue 5435
//	fak dispatch acceptance-resolve --issue 5435 --json
//	# resolve a body you already have (no gh, fully offline)
//	fak dispatch acceptance-resolve --body-file issue.md --ref origin/main
//	# just the caller-count half, for one symbol
//	fak dispatch acceptance-resolve --symbol ResolveHostSpill
//
// All classification lives in the pure internal/closureaudit fold (acceptance.go);
// this file is the git/gh wire and the render. Presence is probed against a REF
// (default origin/main), never the working tree: a peer's uncommitted lane must not
// read as landed, and a local HEAD lags the shared trunk.
//
// The same resolver is folded into the closure-audit backlog view — see
// `fak dispatch closure-audit --resolve-acceptance` in dispatch_closure_audit.go —
// so the view that dispatch reads stops reporting a shipped issue as open.

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"io"
	"os"
	"os/exec"
	"regexp"
	"sort"
	"strconv"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/closureaudit"
)

// acceptanceResolveSchema is the machine payload tag for this verb.
const acceptanceResolveSchema = "fak-acceptance-resolve/1"

// acceptanceIssueBody is the gh seam (overridable in tests so the verb is exercised
// without network): it returns one issue's Markdown body.
var acceptanceIssueBody = acceptanceIssueBodyGH

// acceptanceResolveOut is the --json envelope.
type acceptanceResolveOut struct {
	Schema     string                    `json:"schema"`
	Issue      int                       `json:"issue,omitempty"`
	Symbol     string                    `json:"symbol,omitempty"`
	Ref        string                    `json:"ref"`
	Workspace  string                    `json:"workspace,omitempty"`
	Resolution *closureaudit.Resolution  `json:"resolution,omitempty"`
	Callers    *closureaudit.CallerCount `json:"callers,omitempty"`
}

func runDispatchAcceptanceResolve(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("dispatch acceptance-resolve", flag.ContinueOnError)
	fs.SetOutput(stderr)
	workspace := fs.String("workspace", "", "workspace root (default: repo root)")
	ref := fs.String("ref", closureaudit.DefaultRef, "git ref every presence probe resolves against")
	issue := fs.Int("issue", 0, "resolve this GitHub issue's acceptance (reads the body via gh)")
	bodyFile := fs.String("body-file", "", "resolve this file's Markdown body instead of fetching an issue")
	symbol := fs.String("symbol", "", "run only the caller-count check for this symbol")
	asJSON := fs.Bool("json", false, "emit the "+acceptanceResolveSchema+" payload")
	if !parseFlags(fs, argv) {
		return 2
	}
	if fs.NArg() != 0 {
		fmt.Fprintf(stderr, "fak dispatch acceptance-resolve: unexpected argument %q\n", fs.Arg(0))
		return 2
	}
	chosen := 0
	for _, on := range []bool{*issue > 0, strings.TrimSpace(*bodyFile) != "", strings.TrimSpace(*symbol) != ""} {
		if on {
			chosen++
		}
	}
	if chosen != 1 {
		fmt.Fprintln(stderr, "fak dispatch acceptance-resolve: choose exactly one of --issue, --body-file, or --symbol")
		return 2
	}

	root := *workspace
	if root == "" {
		root = repoRoot()
	}
	trunk := newAcceptanceTrunk(root, *ref)

	if sym := strings.TrimSpace(*symbol); sym != "" {
		cc, err := trunk.callerCount(sym)
		if err != nil {
			fmt.Fprintf(stderr, "fak dispatch acceptance-resolve: caller-count %s on %s: %v\n", sym, trunk.ref, err)
			return 1
		}
		if *asJSON {
			return encodeJSONOrFail(stdout, stderr, acceptanceResolveOut{
				Schema: acceptanceResolveSchema, Symbol: sym, Ref: trunk.ref, Workspace: root, Callers: &cc,
			}, "fak dispatch acceptance-resolve")
		}
		fmt.Fprint(stdout, acceptanceCallersLine(trunk.ref, cc))
		return 0
	}

	var body string
	if *issue > 0 {
		b, err := acceptanceIssueBody(root, *issue)
		if err != nil {
			fmt.Fprintf(stderr, "fak dispatch acceptance-resolve: read issue #%d: %v\n", *issue, err)
			return 1
		}
		body = b
	} else {
		raw, err := os.ReadFile(*bodyFile)
		if err != nil {
			fmt.Fprintf(stderr, "fak dispatch acceptance-resolve: read %q: %v\n", *bodyFile, err)
			return 1
		}
		body = string(raw)
	}

	res := resolveAcceptanceBody(trunk, body)
	if *asJSON {
		return encodeJSONOrFail(stdout, stderr, acceptanceResolveOut{
			Schema: acceptanceResolveSchema, Issue: *issue, Ref: trunk.ref, Workspace: root, Resolution: &res,
		}, "fak dispatch acceptance-resolve")
	}
	fmt.Fprint(stdout, acceptanceResolutionBlock(*issue, res))
	return 0
}

// resolveAcceptanceBody is the whole decision for one body: extract the needles,
// probe each on the trunk ref, fold. Pure decision, impure probes — the split the
// rest of the dispatch family uses.
func resolveAcceptanceBody(trunk *acceptanceTrunk, body string) closureaudit.Resolution {
	acc := closureaudit.ExtractAcceptance(body)
	presence := make([]closureaudit.Presence, 0, len(acc.Needles))
	for _, n := range acc.Needles {
		presence = append(presence, trunk.probe(n))
	}
	return closureaudit.Resolve(trunk.ref, acc, presence)
}

// resolveAcceptanceForReport is the closure-audit backlog view's wiring (#5435):
// it acceptance-resolves the issues that are still OPEN on GitHub — exactly the
// population a dispatch wave draws from, and exactly where a subject-grep audit
// wastes worker slots. Newest-first, capped at limit because each issue costs one
// gh read plus a handful of git probes. Issues the subject grep already bound to a
// witnessed close need no second opinion and are skipped.
func resolveAcceptanceForReport(root, ref string, rep *closureaudit.Report, limit int) {
	if rep == nil || limit <= 0 {
		return
	}
	var open []int
	for _, g := range rep.Issues {
		if g.Bucket == closureaudit.Open || g.Bucket == closureaudit.OpenWitnessed {
			open = append(open, g.Number)
		}
	}
	sort.Sort(sort.Reverse(sort.IntSlice(open)))
	// A cap that silently drops the tail would reproduce this verb's own defect one
	// level up: the unprobed issues would render indistinguishably from issues that
	// were probed and found still open. Keep them in the population and mark them
	// UNKNOWN(NOT_RESOLVED) instead, so the cap is visible in the counts and in each
	// row's own reason.
	var capped []int
	if len(open) > limit {
		capped = open[limit:]
		open = open[:limit]
	}
	trunk := newAcceptanceTrunk(root, ref)
	out := make(map[int]closureaudit.Resolution, len(open)+len(capped))
	for _, n := range capped {
		out[n] = closureaudit.Resolution{
			Verdict: closureaudit.AcceptanceUnknown, Ref: ref,
			Reason: fmt.Sprintf("UNKNOWN (%s): the --acceptance-limit of %d was reached before this issue was probed — it was NOT examined, which is not evidence either way",
				closureaudit.ReasonNotResolved, limit),
			Remaining: fmt.Sprintf("re-run with --acceptance-limit above %d to probe this issue",
				limit),
		}
	}
	for _, n := range open {
		body, err := acceptanceIssueBody(root, n)
		if err != nil {
			// A body we could not read is UNKNOWN, never "still open" — the whole
			// point of this verb is that absent evidence must not read as a verdict.
			out[n] = closureaudit.Resolution{
				Verdict: closureaudit.AcceptanceUnknown, Ref: trunk.ref,
				Reason:    "UNKNOWN (" + closureaudit.ReasonProbeFailed + "): could not read the issue body: " + err.Error(),
				Remaining: "re-run once gh can read the issue",
			}
			continue
		}
		out[n] = resolveAcceptanceBody(trunk, body)
	}
	closureaudit.AttachResolutions(rep, out)
}

// closureAuditAcceptanceBlock is the acceptance block the closure-audit card grows
// when --resolve-acceptance ran. Empty string when it did not, so the historical
// card is byte-identical without the flag.
func closureAuditAcceptanceBlock(rep closureaudit.Report) string {
	if len(rep.AcceptanceCounts) == 0 {
		return ""
	}
	var b strings.Builder
	c := rep.AcceptanceCounts
	fmt.Fprintf(&b, "acceptance-symbol resolution: shipped=%d partial=%d open=%d unknown=%d\n",
		c[closureaudit.AcceptanceShipped], c[closureaudit.AcceptancePartial],
		c[closureaudit.AcceptanceOpen], c[closureaudit.AcceptanceUnknown])
	if stale := closureaudit.StaleOpens(rep); len(stale) > 0 {
		nums := make([]string, 0, len(stale))
		for _, n := range stale {
			nums = append(nums, "#"+strconv.Itoa(n))
		}
		fmt.Fprintf(&b, "  STALE_OPEN (acceptance present on trunk, no subject-grep binding): %s\n", strings.Join(nums, " "))
	}
	shown := 0
	for _, g := range rep.Issues {
		if g.Acceptance == nil || g.Acceptance.Verdict != closureaudit.AcceptancePartial || g.Acceptance.Remaining == "" {
			continue
		}
		if shown >= 15 {
			break
		}
		shown++
		fmt.Fprintf(&b, "  #%-5d PARTIAL — %s\n", g.Number, g.Acceptance.Remaining)
	}
	return b.String()
}

// acceptanceTrunk probes one git ref. The tracked-path listing is loaded at most
// once per run because a path needle would otherwise re-walk the whole tree.
type acceptanceTrunk struct {
	root       string
	ref        string
	tree       []string
	treeErr    error
	treeLoaded bool
}

// runBufferedCommand captures both output streams while leaving command construction,
// process posture, timeouts, exit-code interpretation, and error wording to the caller.
func runBufferedCommand(cmd *exec.Cmd) (string, string, error) {
	var stdout, stderr bytes.Buffer
	cmd.Stdout = &stdout
	cmd.Stderr = &stderr
	err := cmd.Run()
	return stdout.String(), stderr.String(), err
}

func newAcceptanceTrunk(root, ref string) *acceptanceTrunk {
	if strings.TrimSpace(ref) == "" {
		ref = closureaudit.DefaultRef
	}
	return &acceptanceTrunk{root: root, ref: ref}
}

// probe answers "is this needle present on the ref, and where?".
func (t *acceptanceTrunk) probe(n closureaudit.Needle) closureaudit.Presence {
	p := closureaudit.Presence{Needle: n.Grep}
	if n.Kind == closureaudit.NeedlePath {
		paths, err := t.paths()
		if err != nil {
			p.Err = err.Error()
			return p
		}
		want := strings.TrimSuffix(n.Grep, "/")
		for _, f := range paths {
			if f == want || strings.HasPrefix(f, want+"/") || strings.HasSuffix(f, "/"+want) {
				p.Files = append(p.Files, f)
			}
		}
		p.Found = len(p.Files) > 0
		return p
	}

	// A Go symbol lives in Go files; a refusal token can be declared anywhere
	// (dos.toml, a policy JSON), so its search is not pathspec-restricted.
	var pathspec []string
	if n.Kind == closureaudit.NeedleSymbol {
		pathspec = []string{"*.go"}
	}
	files, err := t.grepFixed(n.Grep, pathspec)
	if err != nil {
		p.Err = err.Error()
		return p
	}
	p.Files = files
	p.Found = len(files) > 0
	if n.Kind == closureaudit.NeedleSymbol && p.Found {
		def, err := t.declarationOf(n.Grep)
		if err != nil {
			p.Err = err.Error()
			return p
		}
		p.DefFile = def
	}
	return p
}

// callerCount is the caller-count half on its own: find the declaration, find every
// referent, partition. This is what a reader runs against their OWN new symbol to
// check they did not just ship another caller-less primitive.
func (t *acceptanceTrunk) callerCount(symbol string) (closureaudit.CallerCount, error) {
	def, err := t.declarationOf(symbol)
	if err != nil {
		return closureaudit.CallerCount{}, err
	}
	files, err := t.grepFixed(symbol, []string{"*.go"})
	if err != nil {
		return closureaudit.CallerCount{}, err
	}
	return closureaudit.CountCallers(symbol, def, files), nil
}

// paths returns every tracked path on the ref (repo-relative, forward slashes).
func (t *acceptanceTrunk) paths() ([]string, error) {
	if t.treeLoaded {
		return t.tree, t.treeErr
	}
	t.treeLoaded = true
	ctx, cancel := context.WithTimeout(context.Background(), 60*time.Second)
	defer cancel()
	cmd := exec.CommandContext(ctx, "git", "ls-tree", "-r", "--name-only", t.ref)
	cmd.Dir = t.root
	configureDispatchHelperCommand(cmd)
	out, errOut, err := runBufferedCommand(cmd)
	if err != nil {
		t.treeErr = fmt.Errorf("git ls-tree %s: %w: %s", t.ref, err, strings.TrimSpace(errOut))
		return nil, t.treeErr
	}
	t.tree = splitAcceptanceLines(out)
	return t.tree, nil
}

// grepFixed returns the tracked files on the ref containing needle as a literal.
// `git grep` exits 1 with no output when nothing matched, which is an ANSWER, not a
// failure; anything above 1 is a real error and is surfaced so the fold reports
// UNKNOWN(PROBE_FAILED) rather than silently reading as absent.
func (t *acceptanceTrunk) grepFixed(needle string, pathspec []string) ([]string, error) {
	args := []string{"grep", "-l", "-I", "--fixed-strings", "-e", needle, t.ref}
	if len(pathspec) > 0 {
		args = append(args, "--")
		args = append(args, pathspec...)
	}
	return t.gitLines(args)
}

var acceptanceDeclSanitizeRE = regexp.MustCompile(`[^A-Za-z0-9_]`)

// declarationOf returns the file on the ref that DECLARES symbol as a Go
// func/type/var/const, or "" when nothing on the ref declares it (which is not an
// error — a refusal token or a symbol only referenced has no declaration site).
// Only assignment forms are accepted inside grouped blocks, so a struct FIELD named
// like the symbol can never be mistaken for its declaration.
func (t *acceptanceTrunk) declarationOf(symbol string) (string, error) {
	sym := acceptanceDeclSanitizeRE.ReplaceAllString(symbol, "")
	if sym == "" {
		return "", nil
	}
	pattern := `^(func|type|var|const)[ \t]+(\([^)]*\)[ \t]+)?` + sym + `([ \t(\[{=]|$)` +
		`|^[ \t]+` + sym + `[ \t]*=`
	files, err := t.gitLines([]string{"grep", "-l", "-I", "-E", "-e", pattern, t.ref, "--", "*.go"})
	if err != nil {
		return "", err
	}
	for _, f := range files {
		if !strings.HasSuffix(f, "_test.go") {
			return f, nil
		}
	}
	if len(files) > 0 {
		return files[0], nil
	}
	return "", nil
}

// gitLines runs a `git grep`-shaped command and returns its `<ref>:<path>` output
// with the ref prefix stripped, sorted and deduped.
func (t *acceptanceTrunk) gitLines(args []string) ([]string, error) {
	ctx, cancel := context.WithTimeout(context.Background(), 60*time.Second)
	defer cancel()
	cmd := exec.CommandContext(ctx, "git", args...)
	cmd.Dir = t.root
	configureDispatchHelperCommand(cmd)
	out, errOut, err := runBufferedCommand(cmd)
	if err != nil {
		var ee *exec.ExitError
		if !errors.As(err, &ee) || ee.ExitCode() != 1 {
			return nil, fmt.Errorf("git %s: %w: %s", strings.Join(args[:2], " "), err, strings.TrimSpace(errOut))
		}
		return nil, nil // exit 1 = no match
	}
	prefix := t.ref + ":"
	seen := map[string]bool{}
	var files []string
	for _, ln := range splitAcceptanceLines(out) {
		ln = strings.TrimPrefix(ln, prefix)
		if ln == "" || seen[ln] {
			continue
		}
		seen[ln] = true
		files = append(files, ln)
	}
	sort.Strings(files)
	return files, nil
}

func splitAcceptanceLines(raw string) []string {
	var out []string
	for _, ln := range strings.Split(strings.ReplaceAll(raw, "\r\n", "\n"), "\n") {
		if ln = strings.TrimSpace(ln); ln != "" {
			out = append(out, strings.ReplaceAll(ln, "\\", "/"))
		}
	}
	return out
}

func acceptanceIssueBodyGH(root string, number int) (string, error) {
	ctx, cancel := context.WithTimeout(context.Background(), 90*time.Second)
	defer cancel()
	cmd := exec.CommandContext(ctx, "gh", "issue", "view", strconv.Itoa(number), "--json", "body")
	cmd.Dir = root
	configureDispatchHelperCommand(cmd)
	out, err := cmd.Output()
	if err != nil {
		return "", fmt.Errorf("gh issue view %d: %w", number, err)
	}
	var rec struct {
		Body string `json:"body"`
	}
	if err := json.Unmarshal(out, &rec); err != nil {
		return "", fmt.Errorf("gh issue view %d: parse json: %w", number, err)
	}
	return rec.Body, nil
}

func acceptanceResolutionBlock(issue int, res closureaudit.Resolution) string {
	var b strings.Builder
	label := "acceptance-resolve"
	if issue > 0 {
		label = fmt.Sprintf("acceptance-resolve #%d", issue)
	}
	fmt.Fprintf(&b, "%s @ %s: %s\n", label, res.Ref, res.Verdict)
	fmt.Fprintf(&b, "  reason: %s\n", res.Reason)
	if res.Remaining != "" {
		fmt.Fprintf(&b, "  remaining: %s\n", res.Remaining)
	}
	byNeedle := map[string]closureaudit.Presence{}
	for _, p := range res.Presence {
		byNeedle[p.Needle] = p
	}
	for _, n := range res.Acceptance.Needles {
		state := "absent"
		if p, ok := byNeedle[n.Grep]; ok {
			switch {
			case p.Err != "":
				state = "probe-failed"
			case p.Found:
				state = fmt.Sprintf("present in %d file(s)", len(p.Files))
			}
		} else {
			state = "unprobed"
		}
		fmt.Fprintf(&b, "  needle %-40s %-7s %s\n", n.Text, n.Kind, state)
	}
	for _, d := range res.Acceptance.Declined {
		fmt.Fprintf(&b, "  declined %-38s %s\n", d.Text, d.Reason)
	}
	for _, cc := range res.Callers {
		b.WriteString("  " + strings.TrimPrefix(acceptanceCallersLine(res.Ref, cc), "callers "))
	}
	return b.String()
}

func acceptanceCallersLine(ref string, cc closureaudit.CallerCount) string {
	state := "WIRED"
	if cc.Unwired {
		state = "UNWIRED"
	}
	def := cc.DefFile
	if def == "" {
		def = "(no declaration on " + ref + ")"
	}
	return fmt.Sprintf("callers %s @ %s: %s  production=%d own_package=%d tests=%d architest=%d  def=%s\n",
		cc.Needle, ref, state, cc.Production, cc.OwnPackage, cc.Tests, cc.Architest, def)
}
