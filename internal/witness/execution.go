package witness

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/abi"
)

// ExecutionVerdict is the execution witness label space. It is deliberately
// separate from abi.WitnessOutcome so callers can distinguish test evidence from
// the existing git-evidence resolver verdicts.
type ExecutionVerdict string

const (
	ExecPass          ExecutionVerdict = "EXEC_PASS"
	ExecUnwitnessed   ExecutionVerdict = "EXEC_UNWITNESSED"
	ExecNotApplicable ExecutionVerdict = "EXEC_NOT_APPLICABLE"
)

// ExecutionSelector names one executable selector. Command is an argv vector run
// without a shell inside a detached scratch worktree.
type ExecutionSelector struct {
	ID      string   `json:"id,omitempty"`
	Command []string `json:"command"`
}

// ExecutionSpec is the JSON payload accepted by the exec:<json> witness claim.
// FailToPass selectors must fail at the parent and pass at Commit. PassToPass
// selectors must pass at both parent and Commit.
type ExecutionSpec struct {
	Commit     string              `json:"commit"`
	FailToPass []ExecutionSelector `json:"fail_to_pass,omitempty"`
	PassToPass []ExecutionSelector `json:"pass_to_pass,omitempty"`
}

// ExecutionEvidence records a single selector observation.
type ExecutionEvidence struct {
	Kind     string `json:"kind"`
	Ref      string `json:"ref"`
	Selector string `json:"selector"`
	ExitCode int    `json:"exit_code"`
	Outcome  string `json:"outcome"`
	Error    string `json:"error,omitempty"`
}

// ExecutionResult is the portable read-back from the execution witness rung.
type ExecutionResult struct {
	Verdict  ExecutionVerdict    `json:"verdict"`
	Reason   string              `json:"reason,omitempty"`
	Commit   string              `json:"commit,omitempty"`
	Parent   string              `json:"parent,omitempty"`
	Evidence []ExecutionEvidence `json:"evidence,omitempty"`
}

// WitnessOutcome maps the execution-specific verdict back to the kernel's
// three-way witness contract when the execution rung is used through exec:<json>.
func (r ExecutionResult) WitnessOutcome() abi.WitnessOutcome {
	switch r.Verdict {
	case ExecPass:
		return abi.WitnessConfirmed
	case ExecUnwitnessed:
		return abi.WitnessRefuted
	default:
		return abi.WitnessAbstain
	}
}

// ExecutionVerifier checks fail-to-pass/pass-to-pass selectors in detached git
// worktrees, leaving the caller's working clone files untouched.
type ExecutionVerifier struct {
	git Runner
	run CommandRunner
	dir string
}

// NewExecutionVerifier constructs the real execution witness for repoDir.
func NewExecutionVerifier(repoDir string) *ExecutionVerifier {
	return &ExecutionVerifier{git: gitRunner, run: commandRunner, dir: repoDir}
}

// NewExecutionVerifierWithRunners injects git and command runners for tests.
func NewExecutionVerifierWithRunners(git Runner, run CommandRunner, repoDir string) *ExecutionVerifier {
	if git == nil {
		git = gitRunner
	}
	if run == nil {
		run = commandRunner
	}
	return &ExecutionVerifier{git: git, run: run, dir: repoDir}
}

// Verify confirms that every FAIL_TO_PASS selector made a red->green transition
// and every PASS_TO_PASS selector stayed green. With no FAIL_TO_PASS selector,
// the execution rung is not applicable and abstains instead of blocking a
// docs-only or otherwise non-test-gated claim.
func (v *ExecutionVerifier) Verify(ctx context.Context, spec ExecutionSpec) ExecutionResult {
	spec.Commit = strings.TrimSpace(spec.Commit)
	if spec.Commit == "" {
		return ExecutionResult{Verdict: ExecUnwitnessed, Reason: "missing_commit"}
	}
	if len(spec.FailToPass) == 0 {
		return ExecutionResult{Verdict: ExecNotApplicable, Reason: "missing_fail_to_pass_selector", Commit: spec.Commit}
	}
	if bad := firstInvalidSelector(spec.FailToPass, spec.PassToPass); bad != "" {
		return ExecutionResult{Verdict: ExecUnwitnessed, Reason: "invalid_selector:" + bad, Commit: spec.Commit}
	}

	commit, ok := v.revParse(ctx, spec.Commit+"^{commit}")
	if !ok {
		return ExecutionResult{Verdict: ExecUnwitnessed, Reason: "commit_not_found", Commit: spec.Commit}
	}
	parent, ok := v.revParse(ctx, commit+"^")
	if !ok {
		return ExecutionResult{Verdict: ExecUnwitnessed, Reason: "parent_not_found", Commit: commit}
	}

	res := ExecutionResult{Verdict: ExecUnwitnessed, Commit: commit, Parent: parent}
	parentDir, cleanupParent, err := v.scratchWorktree(ctx, parent)
	if err != nil {
		res.Reason = "scratch_parent:" + err.Error()
		return res
	}
	defer cleanupParent()

	if ok := v.expectSelectors(ctx, parentDir, parent, "fail_to_pass", spec.FailToPass, false, &res); !ok {
		return res
	}
	if ok := v.expectSelectors(ctx, parentDir, parent, "pass_to_pass", spec.PassToPass, true, &res); !ok {
		return res
	}

	commitDir, cleanupCommit, err := v.scratchWorktree(ctx, commit)
	if err != nil {
		res.Reason = "scratch_commit:" + err.Error()
		return res
	}
	defer cleanupCommit()

	if ok := v.expectSelectors(ctx, commitDir, commit, "fail_to_pass", spec.FailToPass, true, &res); !ok {
		return res
	}
	if ok := v.expectSelectors(ctx, commitDir, commit, "pass_to_pass", spec.PassToPass, true, &res); !ok {
		return res
	}

	res.Verdict = ExecPass
	res.Reason = ""
	return res
}

func (r *Resolver) resolveExecution(ctx context.Context, raw string) abi.WitnessOutcome {
	var spec ExecutionSpec
	if err := json.Unmarshal([]byte(raw), &spec); err != nil {
		return abi.WitnessAbstain
	}
	git := r.run
	if git == nil {
		git = gitRunner
	}
	run := r.execRun
	if run == nil {
		run = commandRunner
	}
	return NewExecutionVerifierWithRunners(git, run, r.dir).Verify(ctx, spec).WitnessOutcome()
}

func (v *ExecutionVerifier) revParse(ctx context.Context, ref string) (string, bool) {
	out, code, err := v.gitRun()(ctx, v.dir, "rev-parse", "--verify", ref)
	if err != nil || code != 0 {
		return "", false
	}
	sha := strings.TrimSpace(out)
	if i := strings.IndexByte(sha, '\n'); i >= 0 {
		sha = strings.TrimSpace(sha[:i])
	}
	return sha, sha != ""
}

func (v *ExecutionVerifier) scratchWorktree(ctx context.Context, ref string) (string, func(), error) {
	dir, err := os.MkdirTemp("", "fak-exec-witness-*")
	if err != nil {
		return "", nil, err
	}
	if err := os.Remove(dir); err != nil {
		_ = os.RemoveAll(dir)
		return "", nil, err
	}
	_, code, runErr := v.gitRun()(ctx, v.dir, "worktree", "add", "--detach", "--quiet", dir, ref)
	if runErr != nil {
		_ = os.RemoveAll(dir)
		return "", nil, runErr
	}
	if code != 0 {
		_ = os.RemoveAll(dir)
		return "", nil, fmt.Errorf("git worktree add exited %d", code)
	}
	cleanup := func() {
		_, _, _ = v.gitRun()(context.Background(), v.dir, "worktree", "remove", "--force", dir)
		_ = os.RemoveAll(dir)
	}
	return dir, cleanup, nil
}

func (v *ExecutionVerifier) expectSelectors(ctx context.Context, dir, ref, kind string, selectors []ExecutionSelector, wantPass bool, res *ExecutionResult) bool {
	for _, sel := range selectors {
		ev := v.runSelector(ctx, dir, ref, kind, sel)
		res.Evidence = append(res.Evidence, ev)
		if ev.Error != "" {
			res.Reason = kind + "_error:" + ev.Selector
			return false
		}
		passed := ev.Outcome == "pass"
		if passed != wantPass {
			if kind == "fail_to_pass" && !wantPass {
				res.Reason = "fail_to_pass_green_at_parent:" + ev.Selector
			} else if kind == "fail_to_pass" {
				res.Reason = "fail_to_pass_still_red:" + ev.Selector
			} else {
				res.Reason = "pass_to_pass_regressed:" + ev.Selector
			}
			return false
		}
	}
	return true
}

func (v *ExecutionVerifier) runSelector(ctx context.Context, dir, ref, kind string, sel ExecutionSelector) ExecutionEvidence {
	out, code, err := v.commandRun()(ctx, dir, sel.Command...)
	ev := ExecutionEvidence{
		Kind:     kind,
		Ref:      ref,
		Selector: selectorID(sel),
		ExitCode: code,
		Outcome:  "fail",
	}
	if code == 0 {
		ev.Outcome = "pass"
	}
	if err != nil {
		ev.Outcome = "error"
		ev.Error = err.Error()
	} else if code == -1 && strings.TrimSpace(out) != "" {
		ev.Error = strings.TrimSpace(out)
	}
	return ev
}

func (v *ExecutionVerifier) gitRun() Runner {
	if v.git != nil {
		return v.git
	}
	return gitRunner
}

func (v *ExecutionVerifier) commandRun() CommandRunner {
	if v.run != nil {
		return v.run
	}
	return commandRunner
}

func firstInvalidSelector(groups ...[]ExecutionSelector) string {
	for _, group := range groups {
		for _, sel := range group {
			if len(sel.Command) == 0 || strings.TrimSpace(sel.Command[0]) == "" {
				return selectorID(sel)
			}
		}
	}
	return ""
}

func selectorID(sel ExecutionSelector) string {
	if strings.TrimSpace(sel.ID) != "" {
		return strings.TrimSpace(sel.ID)
	}
	return strings.Join(sel.Command, " ")
}
