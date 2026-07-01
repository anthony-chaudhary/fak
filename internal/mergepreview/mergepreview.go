package mergepreview

import (
	"bytes"
	"context"
	"errors"
	"fmt"
	"os/exec"
	"sort"
	"strings"
)

type Outcome string

const (
	OutcomeEmptyNetDiff Outcome = "empty_net_diff"
	OutcomeCleanMerge   Outcome = "clean_merge"
	OutcomeConflicts    Outcome = "conflicts"
)

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

type Runner func(ctx context.Context, dir string, args ...string) (RunResult, error)

type RunResult struct {
	Stdout []byte
	Stderr []byte
	Code   int
}

func RealRunner(ctx context.Context, dir string, args ...string) (RunResult, error) {
	cmd := exec.CommandContext(ctx, "git", args...)
	cmd.Dir = dir
	var stdout, stderr bytes.Buffer
	cmd.Stdout = &stdout
	cmd.Stderr = &stderr
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
