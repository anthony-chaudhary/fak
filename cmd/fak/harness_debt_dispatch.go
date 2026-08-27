package main

// fak harness-debt-dispatch -- the harness-strength verdict to backlog bridge
// (#4414, epic #4396, under self-ablation #607 / open ablation registry #2828).
//
// It reads the sibling model-strength classifier's --json verdict payload -- each
// HARD harness scaffold graded LOAD_BEARING / REDUNDANT / HOBBLING against current
// model strength -- keeps only the debt scaffolds (REDUNDANT or HOBBLING), folds
// each into a stable per-scaffold dedup key, and plans at most --cap deletion
// issues. It adds no new judgment: it routes an already-produced verdict into the
// backlog, exactly like the learning-/mode-/propagation-/qa-process-/unwired-
// debt-dispatch siblings. Safe by default: dry-run mutates nothing (including the
// seen-cache); --live is required to call gh and record successfully filed keys.
//
// The verdict SOURCE -- the classifier under internal/ablate/ -- is the blocked-by
// sibling leaf and is NOT built here; this leaf only consumes its verdict. If the
// classifier's payload field names differ from the schema read below, adapt this
// reader (it already accepts the common aliases), never the family contract.

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"sort"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/dogfoodissues"
	"github.com/anthony-chaudhary/fak/internal/windowgate"
)

const (
	hddSchema     = "fak.harness-debt-dispatch.v1"
	hddSeenSchema = "fak.harness-debt-dispatch.seen.v1"
	hddCacheRel   = ".fak/harness-debt-dispatch/seen.json"
)

// Grades the model-strength classifier assigns each HARD scaffold. REDUNDANT and
// HOBBLING are BOTH debt (a stronger model carries the capability for free, or is
// actively slowed by the scaffold); LOAD_BEARING is the one grade that files
// nothing -- the scaffold still earns its keep.
const (
	hddGradeLoadBearing = "LOAD_BEARING"
	hddGradeRedundant   = "REDUNDANT"
	hddGradeHobbling    = "HOBBLING"
)

// hddGHTimeout bounds each real gh subprocess so a stalled network call or a
// blocking auth prompt cannot hang the caller indefinitely.
const hddGHTimeout = 60 * time.Second

var hddDefaultTriageLabels = []string{"needs-triage", "triage-only"}

// hddMergeLabels unions the always-on triage labels with any caller-supplied
// labels into one deduplicated, order-preserving slice: base labels first (in
// their declared order), then extras that were not already present. Membership is
// tested on the trimmed value so a whitespace-only or duplicate entry never
// produces a redundant `--label`; empty/blank entries are dropped entirely.
func hddMergeLabels(base, extra []string) []string {
	seen := make(map[string]struct{}, len(base)+len(extra))
	out := make([]string, 0, len(base)+len(extra))
	for _, label := range append(append([]string{}, base...), extra...) {
		key := strings.TrimSpace(label)
		if key == "" {
			continue
		}
		if _, ok := seen[key]; ok {
			continue
		}
		seen[key] = struct{}{}
		out = append(out, key)
	}
	return out
}

var hddMarkerRE = regexp.MustCompile(`<!--\s*fak-harness-debt-key:\s*([^>\s]+)\s*-->`)

// hddRawScaffold is the tolerant wire shape: the classifier's own field names win,
// but the common aliases are accepted so a field-name drift in the (blocked-by)
// verdict source adapts the reader, not the contract.
type hddRawScaffold struct {
	ID        string `json:"id"`
	Scaffold  string `json:"scaffold"`
	Name      string `json:"name"`
	Grade     string `json:"grade"`
	Verdict   string `json:"verdict"`
	Hardness  string `json:"hardness"`
	Hard      *bool  `json:"hard"`
	Rationale string `json:"rationale"`
	Reason    string `json:"reason"`
}

type hddRawVerdict struct {
	Schema    string           `json:"schema"`
	Scaffolds []hddRawScaffold `json:"scaffolds"`
	Verdicts  []hddRawScaffold `json:"verdicts"`
}

// hddScaffold is the normalized, deduplicated verdict row this leaf routes.
type hddScaffold struct {
	ID        string `json:"id"`
	Grade     string `json:"grade"`
	Hard      bool   `json:"hard"`
	Rationale string `json:"rationale,omitempty"`
}

type hddSeenRecord struct {
	FiledAt  string `json:"filed_at"`
	ID       string `json:"id"`
	Grade    string `json:"grade"`
	IssueURL string `json:"issue_url,omitempty"`
}

// hddSeenCache is the harness-debt surface's instance of the shared debt-dispatch seen cache
// (an alias, so the literals and field accesses below are unchanged).
type hddSeenCache = dogfoodissues.SeenCache[hddSeenRecord]

type hddIssue struct {
	Number int    `json:"number"`
	Title  string `json:"title"`
	Body   string `json:"body"`
	State  string `json:"state"`
	URL    string `json:"url"`
}

type hddPlanRow struct {
	Action    string `json:"action"`
	Key       string `json:"key"`
	Title     string `json:"title"`
	Body      string `json:"-"`
	ID        string `json:"id"`
	Grade     string `json:"grade"`
	Rationale string `json:"rationale,omitempty"`
}

type hddStats struct {
	TotalScaffolds   int `json:"total_scaffolds"`
	DebtScaffolds    int `json:"debt_scaffolds"`
	Planned          int `json:"planned"`
	SkippedSeen      int `json:"skipped_seen"`
	SkippedIssueBody int `json:"skipped_issue_body"`
	SkippedWithinRun int `json:"skipped_within_run"`
	SkippedCap       int `json:"skipped_cap"`
	Cap              int `json:"cap"`
}

type hddSyncRow struct {
	Key    string `json:"key"`
	OK     bool   `json:"ok"`
	Stdout string `json:"stdout,omitempty"`
	Stderr string `json:"stderr,omitempty"`
}

type hddResult struct {
	Schema  string       `json:"schema"`
	Mode    string       `json:"mode"`
	Verdict string       `json:"verdict"`
	Cache   string       `json:"cache"`
	Stats   hddStats     `json:"stats"`
	Planned []hddPlanRow `json:"planned"`
	Synced  []hddSyncRow `json:"synced"`
}

type hddRunner func(args []string) (stdout, stderr string, ok bool)

func cmdHarnessDebtDispatch(argv []string) {
	os.Exit(runHarnessDebtDispatch(os.Stdout, os.Stderr, argv))
}

func runHarnessDebtDispatch(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("harness-debt-dispatch", flag.ContinueOnError)
	fs.SetOutput(stderr)
	workspace := fs.String("workspace", "", "workspace root for the default cache (default: repo root)")
	verdict := fs.String("verdict", "", "the classifier's --json verdict payload (path)")
	cachePath := fs.String("cache", "", "seen-cache path (default: <workspace>/.fak/harness-debt-dispatch/seen.json)")
	capN := fs.Int("cap", 3, "maximum new deletion issues to file in one run")
	repo := fs.String("repo", "", "owner/repo for gh; default is the current repo")
	limit := fs.Int("limit", 300, "existing-issue scan limit for live/fetch modes")
	existingJSON := fs.String("existing-json", "", "fixture list of existing gh issues for dry-run dedup tests")
	fetchExisting := fs.Bool("fetch-existing", false, "dry-run but query gh to dedup against existing issue bodies")
	live := fs.Bool("live", false, "create GitHub issues with gh and update the seen-cache")
	asJSON := fs.Bool("json", false, "emit machine-readable plan/result")
	var labels stringList
	fs.Var(&labels, "label", "label to add to newly-created issues; repeatable")
	if !parseFlags(fs, argv) {
		return 2
	}
	if *capN < 0 {
		fmt.Fprintln(stderr, "fak harness-debt-dispatch: --cap must be >= 0")
		return 2
	}
	// Accept the verdict as a bare positional too, mirroring the siblings.
	if *verdict == "" && fs.NArg() == 1 {
		*verdict = fs.Arg(0)
	} else if fs.NArg() != 0 {
		fmt.Fprintf(stderr, "fak harness-debt-dispatch: unexpected argument %q\n", fs.Arg(0))
		return 2
	}
	if *verdict == "" {
		fmt.Fprintln(stderr, "fak harness-debt-dispatch: --verdict is required")
		return 2
	}

	root := *workspace
	if root == "" {
		root = repoRoot()
	} else if abs, err := filepath.Abs(root); err == nil {
		root = abs
	}
	if *cachePath == "" {
		*cachePath = filepath.Join(root, hddCacheRel)
	}
	verdictPath, err := filepath.Abs(*verdict)
	if err != nil {
		fmt.Fprintf(stderr, "harness-debt-dispatch: %v\n", err)
		return 2
	}
	cacheAbs, err := filepath.Abs(*cachePath)
	if err != nil {
		fmt.Fprintf(stderr, "harness-debt-dispatch: %v\n", err)
		return 2
	}

	// Fail closed on an absent/unparseable verdict rather than inventing scaffolds.
	scaffolds, err := hddLoadVerdict(verdictPath)
	if err != nil {
		fmt.Fprintf(stderr, "harness-debt-dispatch: %v\n", err)
		return 2
	}
	seen, err := hddLoadSeen(cacheAbs)
	if err != nil {
		fmt.Fprintf(stderr, "harness-debt-dispatch: load seen-cache: %v\n", err)
		return 2
	}

	var existing []hddIssue
	switch {
	case *existingJSON != "":
		b, err := os.ReadFile(*existingJSON)
		if err != nil {
			fmt.Fprintf(stderr, "harness-debt-dispatch: %v\n", err)
			return 2
		}
		if err := json.Unmarshal(b, &existing); err != nil {
			fmt.Fprintf(stderr, "harness-debt-dispatch: --existing-json must contain a JSON list: %v\n", err)
			return 2
		}
	case *live || *fetchExisting:
		existing, err = hddFetchExisting(*repo, *limit, nil)
		if err != nil {
			fmt.Fprintf(stderr, "harness-debt-dispatch: %v\n", err)
			return 2
		}
	}

	debt := hddSelectDebt(scaffolds)
	plan, stats := hddBuildPlan(debt, seen, existing, *capN, verdictPath)
	stats.TotalScaffolds = len(scaffolds)
	mode := "dry-run"
	if *live {
		mode = "live"
	}
	result := hddResult{
		Schema:  hddSchema,
		Mode:    mode,
		Verdict: verdictPath,
		Cache:   cacheAbs,
		Stats:   stats,
		Planned: plan,
		Synced:  []hddSyncRow{},
	}

	exit := 0
	if *live && len(plan) > 0 {
		result.Synced = hddSync(plan, *repo, []string(labels), nil)
		hddMarkSuccessful(&seen, plan, result.Synced, time.Now())
		for _, row := range result.Synced {
			if !row.OK {
				exit = 1
			}
		}
		if err := hddSaveSeen(cacheAbs, seen); err != nil {
			fmt.Fprintf(stderr, "harness-debt-dispatch: save seen-cache: %v\n", err)
			exit = 1
		}
	}

	if code := emitJSONOrPrintln(stdout, stderr, "harness-debt-dispatch", *asJSON, result, hddRender(result)); code != 0 {
		return code
	}
	return exit
}

// hddLoadVerdict reads the classifier payload and normalizes each row, tolerating
// the common field-name aliases. It fails closed (error) on an absent or
// unparseable payload -- a missing verdict must never be read as "no debt".
func hddLoadVerdict(path string) ([]hddScaffold, error) {
	b, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	var raw hddRawVerdict
	if err := json.Unmarshal(b, &raw); err != nil {
		return nil, fmt.Errorf("verdict payload must be a JSON object: %w", err)
	}
	rows := raw.Scaffolds
	if len(rows) == 0 {
		rows = raw.Verdicts
	}
	out := make([]hddScaffold, 0, len(rows))
	for _, r := range rows {
		id := firstNonEmpty(r.ID, r.Scaffold, r.Name)
		if id == "" {
			// No stable identity => no dedup key can be seeded; skip rather than
			// mint a colliding blank key.
			continue
		}
		grade := hddNormalizeGrade(firstNonEmpty(r.Grade, r.Verdict))
		hard := (r.Hard != nil && *r.Hard) || strings.EqualFold(strings.TrimSpace(r.Hardness), "HARD")
		out = append(out, hddScaffold{
			ID:        id,
			Grade:     grade,
			Hard:      hard,
			Rationale: firstNonEmpty(r.Rationale, r.Reason),
		})
	}
	return out, nil
}

// hddSelectDebt keeps only the debt scaffolds: HARD-graded and REDUNDANT or
// HOBBLING. LOAD_BEARING (and any soft/unknown grade) files nothing.
func hddSelectDebt(scaffolds []hddScaffold) []hddScaffold {
	out := make([]hddScaffold, 0, len(scaffolds))
	for _, s := range scaffolds {
		if !s.Hard {
			continue
		}
		if s.Grade == hddGradeRedundant || s.Grade == hddGradeHobbling {
			out = append(out, s)
		}
	}
	sort.SliceStable(out, func(i, j int) bool {
		if out[i].ID != out[j].ID {
			return out[i].ID < out[j].ID
		}
		return out[i].Grade < out[j].Grade
	})
	return out
}

func hddBuildPlan(debt []hddScaffold, seen hddSeenCache, existing []hddIssue, cap int, verdictPath string) ([]hddPlanRow, hddStats) {
	if cap < 0 {
		cap = 0
	}
	stats := hddStats{DebtScaffolds: len(debt), Cap: cap}
	existingKeys := hddExistingByKey(existing)
	within := map[string]bool{}
	plan := make([]hddPlanRow, 0, min(cap, len(debt)))
	for _, s := range debt {
		key := hddStableKey(s)
		if seen.Seen != nil {
			if _, ok := seen.Seen[key]; ok {
				stats.SkippedSeen++
				continue
			}
		}
		if _, ok := existingKeys[key]; ok {
			stats.SkippedIssueBody++
			continue
		}
		if within[key] {
			stats.SkippedWithinRun++
			continue
		}
		within[key] = true
		if len(plan) >= cap {
			stats.SkippedCap++
			continue
		}
		plan = append(plan, hddPlanRow{
			Action:    "create",
			Key:       key,
			Title:     hddIssueTitle(s),
			Body:      hddIssueBody(s, key, verdictPath),
			ID:        s.ID,
			Grade:     s.Grade,
			Rationale: s.Rationale,
		})
	}
	stats.Planned = len(plan)
	return plan, stats
}

// hddStableKey seeds the dedup key from the scaffold identity ALONE (not the
// grade): a scaffold already filed as debt is skipped on every later run even if
// a re-grade flips REDUNDANT<->HOBBLING, so the family's "one issue per scaffold,
// no re-filing" invariant holds.
func hddStableKey(s hddScaffold) string {
	sum := sha256.Sum256([]byte(s.ID))
	return "harness-debt/" + hddSlug(s.ID) + "/" + hex.EncodeToString(sum[:])[:16]
}

func hddIssueTitle(s hddScaffold) string {
	title := "harness-debt: retire " + s.ID + " [" + s.Grade + "]"
	if len(title) <= 120 {
		return title
	}
	return strings.TrimSpace(title[:117]) + "..."
}

func hddIssueBody(s hddScaffold, key, verdictPath string) string {
	var b strings.Builder
	fmt.Fprintf(&b, "<!-- fak-harness-debt-key: %s -->\n", key)
	b.WriteString("# Harness-debt: retire a graded scaffold\n\n")
	fmt.Fprintf(&b, "- Stable key: `%s`\n", key)
	fmt.Fprintf(&b, "- Scaffold: `%s`\n", s.ID)
	fmt.Fprintf(&b, "- Grade: `%s` (HARD)\n", s.Grade)
	fmt.Fprintf(&b, "- Verdict JSON: `%s`\n", verdictPath)
	b.WriteString("- dispatchability: `triage_only`\n")
	switch s.Grade {
	case hddGradeRedundant:
		b.WriteString("\nThis HARD scaffold is graded **REDUNDANT**: a stronger model now carries this")
		b.WriteString(" capability for free, so the scaffold is dead weight.\n")
	case hddGradeHobbling:
		b.WriteString("\nThis HARD scaffold is graded **HOBBLING**: a stronger model is actively slowed")
		b.WriteString(" by it, so the scaffold now costs capability.\n")
	}
	if strings.TrimSpace(s.Rationale) != "" {
		b.WriteString("\nClassifier rationale:\n\n```text\n")
		b.WriteString(strings.TrimSpace(s.Rationale))
		b.WriteString("\n```\n")
	}
	b.WriteString("\nSuggested next action:\n\n")
	b.WriteString("Delete this scaffold (build scaffolding with a deletion date), then re-run the")
	b.WriteString(" model-strength classifier and this dispatcher.\n\n")
	b.WriteString("Managed by `fak harness-debt-dispatch`; the HTML marker above is the dedup key.\n")
	return b.String()
}

// hddLoadSeen/hddSaveSeen are dogfoodissues.LoadSeen/SaveSeen under this surface's schema tag.
// The debt-dispatch siblings named in this file's header all persist the same key->record
// cache, and this leaf used to carry a byte-identical private copy of the read/write rules
// (missing file and empty file both mean "nothing filed yet"; the schema tag and map are
// defaulted on both sides). Only the schema tag and the record type are this surface's own.
func hddLoadSeen(path string) (hddSeenCache, error) {
	return dogfoodissues.LoadSeen[hddSeenRecord](path, hddSeenSchema)
}

func hddSaveSeen(path string, cache hddSeenCache) error {
	return dogfoodissues.SaveSeen(path, hddSeenSchema, cache)
}

func hddMarkSuccessful(cache *hddSeenCache, plan []hddPlanRow, synced []hddSyncRow, now time.Time) {
	if cache.Schema == "" {
		cache.Schema = hddSeenSchema
	}
	if cache.Seen == nil {
		cache.Seen = map[string]hddSeenRecord{}
	}
	byKey := map[string]hddPlanRow{}
	for _, row := range plan {
		byKey[row.Key] = row
	}
	for _, row := range synced {
		if !row.OK {
			continue
		}
		planned, ok := byKey[row.Key]
		if !ok {
			continue
		}
		cache.Seen[row.Key] = hddSeenRecord{
			FiledAt:  now.UTC().Format(time.RFC3339),
			ID:       planned.ID,
			Grade:    planned.Grade,
			IssueURL: hddFirstLine(row.Stdout),
		}
	}
}

func hddSync(plan []hddPlanRow, repo string, labels []string, runner hddRunner) []hddSyncRow {
	run := runner
	if run == nil {
		run = hddDefaultRunner
	}
	rows := make([]hddSyncRow, 0, len(plan))
	for _, row := range plan {
		args := []string{"issue", "create", "--title", row.Title, "--body", row.Body}
		for _, label := range hddMergeLabels(hddDefaultTriageLabels, labels) {
			if strings.TrimSpace(label) != "" {
				args = append(args, "--label", label)
			}
		}
		if repo != "" {
			args = append(args, "--repo", repo)
		}
		stdout, stderr, ok := run(args)
		rows = append(rows, hddSyncRow{
			Key:    row.Key,
			OK:     ok,
			Stdout: strings.TrimSpace(stdout),
			Stderr: strings.TrimSpace(stderr),
		})
	}
	return rows
}

func hddFetchExisting(repo string, limit int, runner hddRunner) ([]hddIssue, error) {
	run := runner
	if run == nil {
		run = hddDefaultRunner
	}
	args := []string{"issue", "list", "--state", "all", "--limit", fmt.Sprintf("%d", limit),
		"--json", "number,title,body,state,url"}
	if repo != "" {
		args = append(args, "--repo", repo)
	}
	stdout, stderr, ok := run(args)
	if !ok {
		return nil, fmt.Errorf("gh %s: %s", strings.Join(args, " "), strings.TrimSpace(stderr))
	}
	if strings.TrimSpace(stdout) == "" {
		return nil, nil
	}
	var issues []hddIssue
	if err := json.Unmarshal([]byte(stdout), &issues); err != nil {
		return nil, err
	}
	return issues, nil
}

func hddRender(r hddResult) string {
	lines := []string{
		fmt.Sprintf("harness-debt-dispatch: %s  planned=%d cap=%d debt=%d total=%d",
			r.Mode, len(r.Planned), r.Stats.Cap, r.Stats.DebtScaffolds, r.Stats.TotalScaffolds),
		fmt.Sprintf("  verdict: %s", r.Verdict),
		fmt.Sprintf("  seen-cache: %s", r.Cache),
	}
	if len(r.Planned) == 0 {
		lines = append(lines, "  no new harness-debt issues to file")
	} else {
		for _, row := range r.Planned {
			lines = append(lines, fmt.Sprintf("  [create] %s  scaffold=%s grade=%s",
				row.Title, row.ID, row.Grade))
		}
	}
	lines = append(lines, fmt.Sprintf("  dedup: seen=%d issue-body=%d within-run=%d cap-skipped=%d",
		r.Stats.SkippedSeen, r.Stats.SkippedIssueBody, r.Stats.SkippedWithinRun, r.Stats.SkippedCap))
	if r.Mode == "dry-run" {
		lines = append(lines, "  dry-run: pass --live to create issues and update the seen-cache")
	}
	return strings.Join(lines, "\n")
}

func hddDefaultRunner(args []string) (string, string, bool) {
	ctx, cancel := context.WithTimeout(context.Background(), hddGHTimeout)
	defer cancel()
	cmd := exec.CommandContext(ctx, "gh", args...)
	windowgate.ConfigureBackgroundCommand(cmd)
	cmd.WaitDelay = 10 * time.Second
	var out, errb strings.Builder
	cmd.Stdout = &out
	cmd.Stderr = &errb
	err := cmd.Run()
	if ctx.Err() == context.DeadlineExceeded {
		if errb.Len() > 0 {
			errb.WriteByte('\n')
		}
		errb.WriteString("gh timed out after " + hddGHTimeout.String())
		return out.String(), errb.String(), false
	}
	return out.String(), errb.String(), err == nil
}

func hddExistingByKey(issues []hddIssue) map[string]hddIssue {
	out := map[string]hddIssue{}
	for _, issue := range issues {
		if key := hddMarkerKey(issue.Body); key != "" {
			out[key] = issue
		}
	}
	return out
}

func hddMarkerKey(body string) string {
	m := hddMarkerRE.FindStringSubmatch(body)
	if m == nil {
		return ""
	}
	return strings.TrimSpace(m[1])
}

func hddNormalizeGrade(s string) string {
	s = strings.ToUpper(strings.TrimSpace(s))
	return strings.ReplaceAll(s, "-", "_")
}

func hddSlug(s string) string {
	s = strings.ToLower(s)
	var b strings.Builder
	lastDash := false
	for _, r := range s {
		if (r >= 'a' && r <= 'z') || (r >= '0' && r <= '9') {
			b.WriteRune(r)
			lastDash = false
			continue
		}
		if !lastDash {
			b.WriteByte('-')
			lastDash = true
		}
	}
	out := strings.Trim(b.String(), "-")
	if out == "" {
		return "scaffold"
	}
	return out
}

func hddFirstLine(s string) string {
	s = strings.TrimSpace(s)
	if i := strings.IndexByte(s, '\n'); i >= 0 {
		return strings.TrimSpace(s[:i])
	}
	return s
}
