package devcmd

import (
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"os"
	"sort"
	"strconv"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/issuecohort"
	"github.com/anthony-chaudhary/fak/internal/issuecontract"
	"github.com/anthony-chaudhary/fak/internal/issuecontractrepair"
)

// decomposeSchema is the stable tag stamped on the machine-readable plan/result.
const decomposeSchema = "fak.issue-decompose.v1"

// runIssueDecompose is the governed consumer that closes the last seam in the
// dispatch dependency pipeline: it turns a detected epic (a routeable but
// non-leaf / oversized issue — the exact rows issuecohort pulls into its
// Subdivide queue and issue repair marks defer-to-phase3) into FILED, LINKED
// child leaves.
//
// The point is the end-to-end goal: every open issue must be selectable by a
// background dispatch loop, and the dependency chain must be visible upfront.
// An epic is neither — there is nothing atomic to dispatch, and a loop that
// picked it would stall. Decompose splits it into child leaves (each its own
// dispatch unit) and then edits the PARENT body to declare `Blocked by #child…`
// — the exact grammar dispatchtick.CandidateBlockedBy parses — so the parent is
// held (holdOpenPrereqForRoute → BLOCKED_BY_OPEN_PREREQ) until its children
// close, the children are dispatchable roots now, and `fak dispatch graph`
// renders the whole chain. The prereq hold is fail-open and cycle-safe, so a
// parent that is later closed self-clears with no persisted state.
//
// Who authors the child CONTENT is the division of labor: with --from-plan the
// caller (typically an agent the loop spawned to read the epic) supplies real
// child leaves and --live files them; without a plan, decompose emits SCAFFOLD
// stubs — the work order an agent then fills — which are dry-run only unless
// --allow-stubs is set (filing empty stubs would just flood the tracker with
// non-dispatchable rows). Governance mirrors `fak-dev issue repair`: dry-run by
// default, --live to arm, and a --max-create blast-radius fuse because filing
// new issues is the highest-consequence write in the pipeline.
func runIssueDecompose(stdout, stderr io.Writer, argv []string) int {
	return runIssueDecomposeWith(stdout, stderr, argv, nil, nil)
}

// issueDraftInjector lets tests hand decompose the loaded open-issue set
// directly (nil = load from --from-issues or, failing that, fetch live via gh),
// so a test asserts on the built gh argv without a network round-trip. The
// runner is the same injectable seam runIssueCreateWith/runIssueEditWith use.
func runIssueDecomposeWith(stdout, stderr io.Writer, argv []string, injected []issuecontract.IssueDraft, runner issueCreateRunner) int {
	fs := flag.NewFlagSet("issue decompose", flag.ContinueOnError)
	fs.SetOutput(stderr)
	fromIssues := fs.String("from-issues", "", "GitHub issue JSON (gh issue list --json number,title,body,labels); default: fetch open issues live")
	fromPlan := fs.String("from-plan", "", "decompose-plan JSON [{\"parent\":N,\"children\":[{\"title\",\"body\",\"labels\"}]}] — real child leaves authored upstream")
	issueSel := fs.String("issue", "", "restrict to these parent issue numbers (comma-separated)")
	live := fs.Bool("live", false, "file child issues and link parents (default: dry-run plan only)")
	allowStubs := fs.Bool("allow-stubs", false, "permit --live filing of scaffold stub children (default: scaffolds are dry-run only)")
	maxCreate := fs.Int("max-create", 20, "blast-radius fuse: refuse --live if total child issues to create exceeds this")
	parentBaseline := fs.Float64("parent-baseline-points", 0, "parent production-scope baseline points (required with --live)")
	completionStandard := fs.String("completion-standard", "production", "generated child maturity (default production)")
	targetEnvelope := fs.String("target-envelope", "", "production target operating envelope")
	witnessedEnvelope := fs.String("witnessed-envelope", "", "currently witnessed operating envelope")
	repo := fs.String("repo", "", "owner/name override passed to gh")
	root := fs.String("C", "", "workspace root for the live open-issue fetch (default: cwd)")
	asJSON := fs.Bool("json", false, "emit the machine-readable plan/result")
	if !parseFlags(fs, argv) {
		return 2
	}
	if fs.NArg() != 0 {
		fmt.Fprintf(stderr, "fak-dev issue decompose: unexpected argument %q\n", fs.Arg(0))
		return 2
	}

	// Load the parent issue set. Parents are the source of truth for the
	// blocked-by edit (gh replaces the whole body, so we need the current one);
	// --from-plan only overlays child CONTENT keyed by parent number.
	loaded := injected
	if loaded == nil {
		var err error
		loaded, err = loadDecomposeIssues(*fromIssues, *root)
		if err != nil {
			fmt.Fprintf(stderr, "fak-dev issue decompose: %v\n", err)
			return 2
		}
	}

	planByParent, err := loadDecomposePlan(*fromPlan)
	if err != nil {
		fmt.Fprintf(stderr, "fak-dev issue decompose: %v\n", err)
		return 2
	}

	sel := parseIssueNumberSet(*issueSel)
	rows, byNumber := buildDecomposeRows(loaded, planByParent, sel)

	result := decomposeResult{
		Schema:     decomposeSchema,
		Live:       *live,
		DryRun:     !*live,
		MaxCreate:  *maxCreate,
		AllowStubs: *allowStubs,
		Rows:       rows,
	}

	// Blast-radius fuse: count every child that --live would actually file and
	// refuse the whole run before touching gh if it exceeds the cap.
	if *live {
		fileable := 0
		for _, r := range rows {
			if decomposeRowFileable(r, *allowStubs) {
				fileable += len(r.Children)
			}
		}
		if fileable > 0 && *parentBaseline <= 0 {
			fmt.Fprintln(stderr, "fak-dev issue decompose: --live child creation requires --parent-baseline-points")
			return 2
		}
		if fileable > *maxCreate {
			fmt.Fprintf(stderr, "fak-dev issue decompose: --live would create %d child issues, over --max-create=%d; narrow with --issue or raise --max-create\n", fileable, *maxCreate)
			return 2
		}
	}

	anyFail := applyDecompose(&result, byNumber, *live, *allowStubs, *repo, runner, stderr, *parentBaseline, *completionStandard, *targetEnvelope, *witnessedEnvelope)
	tallyDecompose(&result)

	exit := 0
	if anyFail {
		exit = 1
	}
	if *asJSON {
		if code := encodeJSONOrFail(stdout, stderr, result, "fak-dev issue decompose"); code != 0 {
			return code
		}
		return exit
	}
	renderDecompose(stdout, result)
	return exit
}

// --- input loading ---------------------------------------------------------

func loadDecomposeIssues(fromIssues, root string) ([]issuecontract.IssueDraft, error) {
	if strings.TrimSpace(fromIssues) != "" {
		b, err := os.ReadFile(fromIssues)
		if err != nil {
			return nil, fmt.Errorf("read --from-issues: %w", err)
		}
		issues, derr := decodeIssueContractIssues(b)
		if derr != nil {
			return nil, fmt.Errorf("decode --from-issues: %w", derr)
		}
		return issues, nil
	}
	issues, err := issuecontractrepair.FetchOpenIssues(resolveRoot(root), issuecontractrepair.DefaultCap)
	if err != nil {
		return nil, fmt.Errorf("fetch open issues (pass --from-issues to work offline): %w", err)
	}
	return issues, nil
}

// decomposePlanSpec is one entry in a --from-plan file: the real child leaves an
// upstream author decided the parent epic decomposes into.
type decomposePlanSpec struct {
	Parent   int                  `json:"parent"`
	Children []decomposeChildSpec `json:"children"`
}

type decomposeChildSpec struct {
	Title  string   `json:"title"`
	Body   string   `json:"body"`
	Labels []string `json:"labels,omitempty"`
}

func loadDecomposePlan(path string) (map[int][]decomposeChildSpec, error) {
	if strings.TrimSpace(path) == "" {
		return nil, nil
	}
	b, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("read --from-plan: %w", err)
	}
	var specs []decomposePlanSpec
	if err := json.Unmarshal(b, &specs); err != nil {
		return nil, fmt.Errorf("decode --from-plan (want [{\"parent\":N,\"children\":[…]}]): %w", err)
	}
	out := map[int][]decomposeChildSpec{}
	for _, s := range specs {
		if s.Parent <= 0 {
			return nil, fmt.Errorf("--from-plan: every spec needs a positive \"parent\" issue number")
		}
		out[s.Parent] = append(out[s.Parent], s.Children...)
	}
	return out, nil
}

// --- plan building ---------------------------------------------------------

type decomposeChildResult struct {
	Title   string   `json:"title"`
	Args    []string `json:"args"`
	Number  int      `json:"number,omitempty"`
	URL     string   `json:"url,omitempty"`
	Applied bool     `json:"applied"`
	Error   string   `json:"error,omitempty"`
}

type decomposeRow struct {
	Parent      int                    `json:"parent"`
	Title       string                 `json:"title,omitempty"`
	Disposition string                 `json:"disposition"`
	Reasons     []string               `json:"reasons,omitempty"`
	ChildBudget int                    `json:"child_budget"`
	Children    []decomposeChildResult `json:"children"`
	LinkArgs    []string               `json:"parent_link_args,omitempty"`
	Linked      bool                   `json:"parent_linked"`
	Error       string                 `json:"error,omitempty"`
}

type decomposeCounts struct {
	Epics           int `json:"epics"`
	ChildrenPlanned int `json:"children_planned"`
	ChildrenCreated int `json:"children_created"`
	ParentsLinked   int `json:"parents_linked"`
	Failed          int `json:"failed"`
}

type decomposeResult struct {
	Schema     string          `json:"schema"`
	Live       bool            `json:"live"`
	DryRun     bool            `json:"dry_run"`
	MaxCreate  int             `json:"max_create"`
	AllowStubs bool            `json:"allow_stubs"`
	Counts     decomposeCounts `json:"counts"`
	Rows       []decomposeRow  `json:"rows"`
}

// Disposition values.
const (
	dispositionDecompose = "decompose" // real child drafts supplied → fileable under --live
	dispositionScaffold  = "scaffold"  // stub children an agent must fill → dry-run unless --allow-stubs
	dispositionError     = "error"     // a --from-plan parent absent from the loaded issues → cannot link
)

// buildDecomposeRows folds the loaded issues + child overlay into ordered rows.
// A row is emitted when the issue is a decompose target (mirrors
// issuecohort.isSplitTarget so decompose and the cohort planner agree on what an
// epic is) OR the caller explicitly named it in --from-plan (operator override,
// the way `fak-dev issue repair --issue` honors an explicit selection). Loaded
// issues are walked in input order for determinism; plan-only parents that are
// not in the loaded set become error rows (sorted) because we cannot fetch their
// current body to link against.
func buildDecomposeRows(loaded []issuecontract.IssueDraft, planByParent map[int][]decomposeChildSpec, sel map[int]bool) ([]decomposeRow, map[int]issuecontract.IssueDraft) {
	byNumber := make(map[int]issuecontract.IssueDraft, len(loaded))
	for _, d := range loaded {
		if d.Number > 0 {
			byNumber[d.Number] = d
		}
	}

	var rows []decomposeRow
	seen := map[int]bool{}
	for _, d := range loaded {
		if d.Number <= 0 {
			continue
		}
		if len(sel) > 0 && !sel[d.Number] {
			continue
		}
		review := issuecontract.ReviewCandidate(issuecontract.CandidateFromIssueDraft(d), issuecontract.Options{})
		children, planned := planByParent[d.Number]
		if !isDecomposeTarget(review) && !planned {
			continue
		}
		seen[d.Number] = true
		rows = append(rows, buildDecomposeRow(d, review, children))
	}

	// --from-plan parents we could not find among the loaded issues: we have no
	// current body to append the blocked-by line to, so we refuse to guess.
	var orphans []int
	for parent := range planByParent {
		if seen[parent] {
			continue
		}
		if len(sel) > 0 && !sel[parent] {
			continue
		}
		orphans = append(orphans, parent)
	}
	sort.Ints(orphans)
	for _, parent := range orphans {
		rows = append(rows, decomposeRow{
			Parent:      parent,
			Disposition: dispositionError,
			Children:    []decomposeChildResult{},
			Error:       "parent not found in loaded issues; pass --from-issues covering it so its body can be linked",
		})
	}
	return rows, byNumber
}

func buildDecomposeRow(d issuecontract.IssueDraft, review issuecontract.Review, plan []decomposeChildSpec) decomposeRow {
	row := decomposeRow{
		Parent:      d.Number,
		Title:       strings.TrimSpace(d.Title),
		ChildBudget: decomposeChildBudget(review.ExpectedSteps),
		Reasons:     decomposeReasons(review),
	}
	specs := plan
	if len(specs) > 0 {
		row.Disposition = dispositionDecompose
	} else {
		row.Disposition = dispositionScaffold
		specs = scaffoldChildren(d, row.ChildBudget)
	}
	row.Children = make([]decomposeChildResult, 0, len(specs))
	for _, spec := range specs {
		row.Children = append(row.Children, decomposeChildResult{
			Title: strings.TrimSpace(spec.Title),
			Args:  decomposeCreateArgs(spec, d.Number),
		})
	}
	return row
}

// isDecomposeTarget no longer MIRRORS issuecohort's rule — it IS that rule. A review is
// an epic to split when it is flagged non-leaf or oversized, the two always-on structural
// gates issuecontract raises for a unit that must decompose before dispatch; a mirrored
// copy could have drifted a reason apart from the cohort planner that routes on it.
func isDecomposeTarget(review issuecontract.Review) bool { return issuecohort.IsSplitTarget(review) }

func decomposeReasons(review issuecontract.Review) []string {
	var out []string
	for _, r := range review.Reasons {
		if r == issuecontract.ReasonNotDispatchLeaf || r == issuecontract.ReasonOversizedSteps {
			out = append(out, r)
		}
	}
	if len(out) == 0 {
		// Only reachable via an explicit --from-plan override of an issue the
		// contract did not itself flag; record that provenance honestly.
		out = append(out, "OPERATOR_REQUESTED")
	}
	return out
}

// decomposeChildBudget mirrors issuecohort.childIssueBudget — ceil(steps / cap)
// — but floors an unknown / unsized epic at 2, since a "split" into a single
// child is a no-op and only a scaffold count needs a sensible default.
func decomposeChildBudget(steps int) int {
	if steps <= 0 {
		return 2
	}
	n := (steps + issuecontract.MaxDispatchExpectedSteps - 1) / issuecontract.MaxDispatchExpectedSteps
	if n < 2 {
		n = 2
	}
	return n
}

// scaffoldChildren emits budget stub leaves for an epic with no supplied plan.
// Each carries the contract field skeleton so a filled version is dispatch-shaped,
// plus a Parent context pointer so provenance survives.
func scaffoldChildren(d issuecontract.IssueDraft, budget int) []decomposeChildSpec {
	title := strings.TrimSpace(d.Title)
	specs := make([]decomposeChildSpec, 0, budget)
	for k := 1; k <= budget; k++ {
		specs = append(specs, decomposeChildSpec{
			Title: fmt.Sprintf("%s — part %d/%d", title, k, budget),
			Body:  scaffoldChildBody(d.Number, title),
		})
	}
	return specs
}

func scaffoldChildBody(parent int, parentTitle string) string {
	var b strings.Builder
	fmt.Fprintf(&b, "## Parent context\nDecomposed from #%d: %s\n\n", parent, parentTitle)
	b.WriteString("## In scope\n<!-- TODO: the one atomic slice this child owns -->\n\n")
	b.WriteString("## Out of scope\n<!-- TODO -->\n\n")
	b.WriteString("## Lane\n<!-- TODO -->\n\n")
	b.WriteString("## Likely files\n<!-- TODO -->\n\n")
	b.WriteString("## Expected steps\n<!-- TODO: <= 8 -->\n\n")
	b.WriteString("## Done condition / witness\n<!-- TODO -->\n")
	return b.String()
}

// decomposeCreateArgs builds the `gh issue create` argv for one child, matching
// runIssueCreateWith's shape (inline --body; one --label per label). The runner
// executes via exec with an arg slice, so the multi-line body needs no quoting.
func decomposeCreateArgs(spec decomposeChildSpec, parent int) []string {
	args := []string{"issue", "create", "--title", strings.TrimSpace(spec.Title), "--body", spec.Body}
	for _, l := range spec.Labels {
		if s := strings.TrimSpace(l); s != "" {
			args = append(args, "--label", s)
		}
	}
	return args
}

// --- apply -----------------------------------------------------------------

func decomposeRowFileable(r decomposeRow, allowStubs bool) bool {
	switch r.Disposition {
	case dispositionDecompose:
		return true
	case dispositionScaffold:
		return allowStubs
	default:
		return false
	}
}

// applyDecompose files children and links parents when live. It returns true if
// anything failed (a create error, an unparseable issue number, a link error, or
// an error row), which the caller turns into a non-zero exit. Parent linking is
// all-or-nothing per epic: we only append `Blocked by #…` once EVERY child of
// that parent was created and numbered, so a partially-filed epic is never left
// with a misleading dependency edge — the created children are reported so a
// rerun can be narrowed with --issue.
func applyDecompose(result *decomposeResult, byNumber map[int]issuecontract.IssueDraft, live, allowStubs bool, repo string, runner issueCreateRunner, stderr io.Writer, parentBaseline float64, standard, targetEnvelope, witnessedEnvelope string) bool {
	run := runner
	if run == nil {
		run = runTaskHandoffGH
	}
	anyFail := false
	skippedStubs := 0
	for i := range result.Rows {
		r := &result.Rows[i]
		if r.Disposition == dispositionError {
			anyFail = true
			continue
		}
		if !live {
			continue
		}
		if !decomposeRowFileable(*r, allowStubs) {
			skippedStubs++
			continue
		}
		if err := authorDecomposeChildren(r, parentBaseline, standard, targetEnvelope, witnessedEnvelope); err != nil {
			r.Error = err.Error()
			anyFail = true
			continue
		}
		created := decomposeFileChildren(r, repo, run)
		if r.Error != "" {
			anyFail = true
		}
		if len(created) == len(r.Children) && len(created) > 0 {
			body := byNumber[r.Parent].Body
			r.LinkArgs = decomposeParentLinkArgs(r.Parent, body, created, repo)
			if _, errOut, ok := run(r.LinkArgs); ok {
				r.Linked = true
			} else {
				r.Error = "children created but parent link failed: " + strings.TrimSpace(errOut)
				anyFail = true
			}
		}
	}
	if live && skippedStubs > 0 {
		fmt.Fprintf(stderr, "fak-dev issue decompose: %d scaffold epic(s) not filed — supply real children via --from-plan, or pass --allow-stubs to file the stubs\n", skippedStubs)
	}
	return anyFail
}

func decomposeArgValue(args []string, name string) string {
	for i := 0; i+1 < len(args); i++ {
		if args[i] == name {
			return args[i+1]
		}
	}
	return ""
}

func authorDecomposeChildren(r *decomposeRow, baseline float64, standard, target, witnessed string) error {
	for i := range r.Children {
		body := decomposeArgValue(r.Children[i].Args, "--body")
		points := float64(issuecontract.MaxDispatchExpectedSteps)
		if points > baseline {
			points = baseline
		}
		authored, err := issuecontract.AuthorBatchProjectWork(body, issuecontract.BatchProjectWork{ParentIssue: r.Parent, EstimatePoints: points, ParentBaseline: baseline, CompletionStandard: standard, TargetEnvelope: target, WitnessedEnvelope: witnessed})
		if err != nil {
			return err
		}
		for j := 0; j+1 < len(r.Children[i].Args); j++ {
			if r.Children[i].Args[j] == "--body" {
				r.Children[i].Args[j+1] = authored
				break
			}
		}
	}
	return nil
}

// decomposeFileChildren creates each child of a row and returns the numbers of
// the children that were created AND whose issue number parsed cleanly. It sets
// r.Error on the first failure (create error or unparseable number).
func decomposeFileChildren(r *decomposeRow, repo string, run issueCreateRunner) []int {
	var created []int
	for ci := range r.Children {
		ch := &r.Children[ci]
		args := ch.Args
		if strings.TrimSpace(repo) != "" {
			args = append(append([]string(nil), ch.Args...), "--repo", repo)
			ch.Args = args
		}
		out, errOut, ok := run(args)
		if !ok {
			ch.Error = strings.TrimSpace(errOut)
			if r.Error == "" {
				r.Error = "one or more child creates failed"
			}
			continue
		}
		ch.Applied = true
		ch.URL = strings.TrimSpace(out)
		if n, okn := parseCreatedIssueNumber(ch.URL); okn {
			ch.Number = n
			created = append(created, n)
		} else if r.Error == "" {
			r.Error = "child created but its issue number could not be parsed from gh output; parent not linked"
		}
	}
	return created
}

// decomposeParentLinkArgs renders the `gh issue edit` argv that appends a
// `Blocked by #child…` line to the parent's current body — the grammar
// dispatchtick.CandidateBlockedBy parses, so the dispatch tick holds the parent
// until the children close. gh replaces the whole body, so we preserve the
// existing one and append.
func decomposeParentLinkArgs(parent int, body string, childNumbers []int, repo string) []string {
	args := []string{"issue", "edit", strconv.Itoa(parent), "--body", appendBlockedBy(body, childNumbers)}
	if strings.TrimSpace(repo) != "" {
		args = append(args, "--repo", repo)
	}
	return args
}

func appendBlockedBy(body string, nums []int) string {
	refs := make([]string, len(nums))
	for i, n := range nums {
		refs[i] = "#" + strconv.Itoa(n)
	}
	line := "Blocked by " + strings.Join(refs, ", ")
	trimmed := strings.TrimRight(body, "\n")
	if trimmed == "" {
		return line + "\n<!-- fak decompose: parent held until child leaves close -->\n"
	}
	return trimmed + "\n\n" + line + "\n<!-- fak decompose: parent held until child leaves close -->\n"
}

// parseCreatedIssueNumber pulls the issue number out of `gh issue create`
// output, which is the created issue URL (…/issues/1234).
func parseCreatedIssueNumber(s string) (int, bool) {
	s = strings.TrimSpace(s)
	if s == "" {
		return 0, false
	}
	if i := strings.LastIndex(s, "/"); i >= 0 {
		s = s[i+1:]
	}
	s = strings.TrimSpace(s)
	n, err := strconv.Atoi(s)
	if err != nil || n <= 0 {
		return 0, false
	}
	return n, true
}

func tallyDecompose(result *decomposeResult) {
	for _, r := range result.Rows {
		if r.Disposition != dispositionError {
			result.Counts.Epics++
		}
		result.Counts.ChildrenPlanned += len(r.Children)
		for _, ch := range r.Children {
			if ch.Applied {
				result.Counts.ChildrenCreated++
			}
		}
		if r.Linked {
			result.Counts.ParentsLinked++
		}
		if r.Error != "" {
			result.Counts.Failed++
		}
	}
}

// --- render ----------------------------------------------------------------

func renderDecompose(w io.Writer, result decomposeResult) {
	mode := "dry-run (plan only)"
	if result.Live {
		mode = "live"
	}
	fmt.Fprintf(w, "fak-dev issue decompose — %s\n", mode)
	fmt.Fprintf(w, "epics=%d children_planned=%d children_created=%d parents_linked=%d failed=%d\n",
		result.Counts.Epics, result.Counts.ChildrenPlanned, result.Counts.ChildrenCreated, result.Counts.ParentsLinked, result.Counts.Failed)
	for _, r := range result.Rows {
		fmt.Fprintf(w, "\n#%d %s [%s]", r.Parent, r.Title, r.Disposition)
		if len(r.Reasons) > 0 {
			fmt.Fprintf(w, " reasons=%s", strings.Join(r.Reasons, ","))
		}
		fmt.Fprintf(w, " budget=%d\n", r.ChildBudget)
		if r.Error != "" {
			fmt.Fprintf(w, "  ! %s\n", r.Error)
		}
		for _, ch := range r.Children {
			status := "planned"
			if ch.Applied && ch.Number > 0 {
				status = fmt.Sprintf("created #%d", ch.Number)
			} else if ch.Applied {
				status = "created"
			} else if ch.Error != "" {
				status = "failed: " + ch.Error
			}
			fmt.Fprintf(w, "  - %s (%s)\n", ch.Title, status)
		}
		if r.Linked {
			fmt.Fprintf(w, "  → parent #%d linked: Blocked by children\n", r.Parent)
		}
	}
}
