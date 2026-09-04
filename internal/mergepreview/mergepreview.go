package mergepreview

import (
	"bytes"
	"context"
	"errors"
	"fmt"
	"os/exec"
	"sort"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/windowgate"
)

// Outcome classifies the simulated three-way merge relationship between HEAD and target.
type Outcome string

const (
	// OutcomeEmptyNetDiff indicates target adds no net changes beyond what already exists in HEAD.
	OutcomeEmptyNetDiff Outcome = "empty_net_diff"
	// OutcomeCleanMerge indicates the branches combine without conflicts but introduce tree modifications.
	OutcomeCleanMerge Outcome = "clean_merge"
	// OutcomeConflicts indicates overlapping conflicting modifications prevent automated three-way combination.
	OutcomeConflicts Outcome = "conflicts"
)

// Result describes the dry-run evaluation of merging target into HEAD.
type Result struct {
	Head            string   `json:"head"`
	Target          string   `json:"target"`
	MergeTree       string   `json:"merge_tree,omitempty"`
	Outcome         Outcome  `json:"outcome"`
	CachedDiffEmpty bool     `json:"cached_diff_empty"`
	ChangedFiles    []string `json:"changed_files,omitempty"`
	Conflicts       []string `json:"conflicts,omitempty"`
	Detail          string   `json:"detail"`
}

// Runner executes git subcommands in a specified working directory.
type Runner func(ctx context.Context, dir string, args ...string) (RunResult, error)

// RunResult captures standard streams and exit status from a git subcommand execution.
type RunResult struct {
	Stdout []byte
	Stderr []byte
	Code   int
}

// RealRunner executes git subcommands via os/exec with background process configuration.
func RealRunner(ctx context.Context, dir string, args ...string) (RunResult, error) {
	cmd := exec.CommandContext(ctx, "git", args...)
	cmd.Dir = dir
	var stdout, stderr bytes.Buffer
	cmd.Stdout = &stdout
	cmd.Stderr = &stderr
	windowgate.ConfigureBackgroundCommand(cmd)
	err := cmd.Run()
	code := 0
	if err != nil {
		var exit *exec.ExitError
		if errors.As(err, &exit) {
			code = exit.ExitCode()
			err = nil
		}
	}
	return RunResult{Stdout: stdout.Bytes(), Stderr: stderr.Bytes(), Code: code}, err
}

// Preview evaluates a three-way merge between HEAD and target without mutating the working tree or index.
func Preview(ctx context.Context, dir, target string, run Runner) (Result, error) {
	target = strings.TrimSpace(target)
	if target == "" {
		return Result{}, fmt.Errorf("merge preview: target ref is required")
	}
	if run == nil {
		run = RealRunner
	}
	head, err := revParse(ctx, dir, "HEAD", run)
	if err != nil {
		return Result{}, err
	}
	targetSHA, err := revParse(ctx, dir, target, run)
	if err != nil {
		return Result{}, err
	}

	merged, err := run(ctx, dir, "merge-tree", "--write-tree", "--name-only", "--no-messages", "-z", "HEAD", target)
	if err != nil {
		return Result{}, err
	}
	if merged.Code != 0 && merged.Code != 1 {
		return Result{}, gitError("merge-tree", merged)
	}
	fields := splitNUL(merged.Stdout)
	if len(fields) == 0 || !isHexOID(fields[0]) {
		return Result{}, fmt.Errorf("merge preview: could not read merge-tree object from git output")
	}
	res := Result{
		Head:      head,
		Target:    targetSHA,
		MergeTree: fields[0],
	}
	if merged.Code == 1 {
		res.Outcome = OutcomeConflicts
		res.Conflicts = uniqueSorted(fields[1:])
		res.Detail = fmt.Sprintf("%d conflict(s) predicted before touching the index/worktree", len(res.Conflicts))
		return res, nil
	}

	diff, err := run(ctx, dir, "diff", "--name-only", "-z", "HEAD", res.MergeTree)
	if err != nil {
		return Result{}, err
	}
	if diff.Code != 0 {
		return Result{}, gitError("diff", diff)
	}
	res.ChangedFiles = uniqueSorted(splitNUL(diff.Stdout))
	res.CachedDiffEmpty = len(res.ChangedFiles) == 0
	if res.CachedDiffEmpty {
		res.Outcome = OutcomeEmptyNetDiff
		res.Detail = "merge resolves cleanly and the cached diff against HEAD would be empty"
		return res, nil
	}
	res.Outcome = OutcomeCleanMerge
	res.Detail = fmt.Sprintf("merge resolves cleanly but would change %d file(s) relative to HEAD", len(res.ChangedFiles))
	return res, nil
}

// ApplyOutcome classifies what Apply did (or declined to do) with a diverged merge.
type ApplyOutcome string

const (
	// ApplyResolvedSuperset: the trivial two-agents-same-block case. The real 3-way merge
	// tree already equals HEAD (both sides added equivalent blocks; HEAD is the superset), so
	// Apply committed a textless `-s ours` merge and ASSERTED the resulting tree == HEAD.
	ApplyResolvedSuperset ApplyOutcome = "resolved_superset"
	// ApplyDeferredClean: the merge resolves without conflict but WOULD change the tree
	// relative to HEAD (target carries net-new content). Not the diff-empty superset case, so
	// Apply mutates nothing and leaves the plain `git merge` to the agent.
	ApplyDeferredClean ApplyOutcome = "deferred_clean"
	// ApplyDeferredConflict: a genuine divergent conflict. Apply mutates nothing and hands the
	// text resolution back to the agent — exactly the "a real semantic conflict still stops
	// for the agent" half of the witness.
	ApplyDeferredConflict ApplyOutcome = "deferred_conflict"
)

// ApplyResult summarizes the actions taken and generated merge commit details from Apply.
type ApplyResult struct {
	Preview      Result       `json:"preview"`
	ApplyOutcome ApplyOutcome `json:"apply_outcome"`
	MergeCommit  string       `json:"merge_commit,omitempty"`
	Detail       string       `json:"detail"`
}

// Apply auto-resolves the trivial superset case named in #2154. It previews the merge of
// target into HEAD with the same conflict-free 3-way merge-tree the dry-run uses, and ONLY when
// that merge tree already equals HEAD (OutcomeEmptyNetDiff — both sides added equivalent blocks,
// so HEAD is already the union) does it resolve textlessly: a `git merge -s ours` commit that
// records target as a second parent while keeping HEAD's tree byte-for-byte. Before returning it
// ASSERTS the committed tree equals the previewed superset tree (== HEAD); a mismatch is a hard
// error, never a silent commit. Every other outcome — a clean merge that changes the tree, or a
// genuine conflict — mutates nothing and defers to the agent's hand-merge. Apply must be given a
// message so the merge commit is attributable; an empty message is filled with a default.
func Apply(ctx context.Context, dir, target, message string, run Runner) (ApplyResult, error) {
	if run == nil {
		run = RealRunner
	}
	pre, err := Preview(ctx, dir, target, run)
	if err != nil {
		return ApplyResult{}, err
	}
	res := ApplyResult{Preview: pre}
	switch pre.Outcome {
	case OutcomeConflicts:
		res.ApplyOutcome = ApplyDeferredConflict
		res.Detail = fmt.Sprintf("%d genuine conflict(s) — hand-merge required; no auto-resolve", len(pre.Conflicts))
		return res, nil
	case OutcomeCleanMerge:
		res.ApplyOutcome = ApplyDeferredClean
		res.Detail = fmt.Sprintf("merge is clean but changes %d file(s) vs HEAD — not the diff-empty superset case; run a plain `git merge`", len(pre.ChangedFiles))
		return res, nil
	case OutcomeEmptyNetDiff:
		// The superset already equals HEAD. Resolve textlessly and assert.
	default:
		return ApplyResult{}, fmt.Errorf("merge apply: unexpected preview outcome %q", pre.Outcome)
	}

	// Record the expected superset tree (== HEAD tree, proven by the empty net diff) before we
	// touch anything, so the post-merge assertion compares against a value captured pre-mutation.
	expectedTree, err := revParseTree(ctx, dir, "HEAD", run)
	if err != nil {
		return ApplyResult{}, err
	}
	if message == "" {
		message = "Merge " + target + " (superset already in HEAD)"
	}
	// `-s ours` keeps HEAD's tree exactly and records target as a second parent. It is safe here
	// ONLY because Preview proved the real 3-way merge tree equals HEAD — so nothing target added
	// is being dropped. --no-ff/--no-edit make the commit deterministic; --signoff satisfies DCO.
	merge, err := run(ctx, dir, "merge", "-s", "ours", "--no-ff", "--no-edit", "--signoff", "-m", message, target)
	if err != nil {
		return ApplyResult{}, err
	}
	if merge.Code != 0 {
		return ApplyResult{}, gitError("merge -s ours", merge)
	}

	// ASSERT the merged tree equals the expected superset before trusting the commit. `-s ours`
	// cannot itself change the tree, but the assertion is the contract the issue names: a witness
	// that the textless resolve dropped nothing. A mismatch is reported, never swallowed.
	gotTree, err := revParseTree(ctx, dir, "HEAD", run)
	if err != nil {
		return ApplyResult{}, err
	}
	if gotTree != expectedTree {
		return ApplyResult{}, fmt.Errorf("merge apply: superset assertion failed — merged tree %s != expected HEAD tree %s", gotTree, expectedTree)
	}
	commit, err := revParse(ctx, dir, "HEAD", run)
	if err != nil {
		return ApplyResult{}, err
	}
	res.ApplyOutcome = ApplyResolvedSuperset
	res.MergeCommit = commit
	res.Detail = "resolved textlessly: merged tree == HEAD (git diff --cached empty vs HEAD); target recorded as a second parent"
	return res, nil
}

func revParse(ctx context.Context, dir, ref string, run Runner) (string, error) {
	out, err := run(ctx, dir, "rev-parse", "--verify", ref+"^{commit}")
	if err != nil {
		return "", err
	}
	if out.Code != 0 {
		return "", gitError("rev-parse "+ref, out)
	}
	return strings.TrimSpace(string(out.Stdout)), nil
}

// revParseTree resolves the tree OID a commit-ish points at (ref^{tree}). It is kept distinct
// from revParse, which forces ^{commit}; appending ^{tree} to a value already suffixed ^{commit}
// would fail with "expected commit type".
func revParseTree(ctx context.Context, dir, ref string, run Runner) (string, error) {
	out, err := run(ctx, dir, "rev-parse", "--verify", ref+"^{tree}")
	if err != nil {
		return "", err
	}
	if out.Code != 0 {
		return "", gitError("rev-parse "+ref+"^{tree}", out)
	}
	return strings.TrimSpace(string(out.Stdout)), nil
}

func gitError(op string, r RunResult) error {
	detail := strings.TrimSpace(string(r.Stderr))
	if detail == "" {
		detail = strings.TrimSpace(string(r.Stdout))
	}
	if detail == "" {
		detail = fmt.Sprintf("exit %d", r.Code)
	}
	return fmt.Errorf("merge preview: git %s failed: %s", op, detail)
}

func splitNUL(b []byte) []string {
	parts := bytes.Split(b, []byte{0})
	out := make([]string, 0, len(parts))
	for _, p := range parts {
		s := string(p)
		if s != "" {
			out = append(out, s)
		}
	}
	return out
}

func uniqueSorted(in []string) []string {
	seen := map[string]bool{}
	out := make([]string, 0, len(in))
	for _, item := range in {
		if item == "" || seen[item] {
			continue
		}
		seen[item] = true
		out = append(out, item)
	}
	sort.Strings(out)
	return out
}

func isHexOID(s string) bool {
	if len(s) < 40 {
		return false
	}
	for _, r := range s {
		if (r >= '0' && r <= '9') || (r >= 'a' && r <= 'f') || (r >= 'A' && r <= 'F') {
			continue
		}
		return false
	}
	return true
}
