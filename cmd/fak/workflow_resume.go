package main

// fak workflow resume — re-derive "done" from evidence before replaying a step (#2444).
//
//	fak workflow resume <run-dir> [--json] [--epoch LABEL] [--now-ms N] [--repo DIR]
//
// A run directory holds the workflow document (spec.json) and the append-only step
// journal (journal.jsonl). Resume folds the journal purely — the clock is injected, never
// read inside the fold — and then decides each step the hard way: a pure step must still
// hash to its journaled output under an unchanged inputs and epoch hash, and an effectful
// step must carry a claim the dos_verify ladder (internal/witness, the in-process
// registry-row / ancestry / ship-stamp-grep resolver) confirms right now. A step is never
// skipped because the journal narrates it as done.
//
// The counters printed are MEASURED, not intended: skipped counts the steps whose journaled
// output the engine actually replayed, executed counts the steps that actually reached the
// runner. A claimed commit that has since been reverted stops corroborating — its cache
// line is refused and the step, plus everything downstream of it, re-executes.

import (
	"context"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"github.com/anthony-chaudhary/fak/internal/interspersedflags"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"sort"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/abi"
	"github.com/anthony-chaudhary/fak/internal/witness"
	"github.com/anthony-chaudhary/fak/internal/workflow"
)

const (
	workflowSpecFile    = "spec.json"
	workflowJournalFile = "journal.jsonl"
	workflowResumeTool  = "fak-workflow-resume"
)

// workflowResumeReport is the machine form of a resume: flat rows, one per step, plus the
// measured counters. Depth stays at 2 so it satisfies the same structured-output law the
// steps themselves must.
type workflowResumeReport struct {
	Run      string                `json:"run"`
	Epoch    string                `json:"epoch"`
	Steps    []workflowResumeStep  `json:"steps"`
	Skipped  int                   `json:"skipped"`
	Executed int                   `json:"executed"`
	Failed   int                   `json:"failed"`
	Rows     int                   `json:"journal_rows"`
	Refusals []workflowStepRefusal `json:"refusals,omitempty"`
}

// workflowResumeStep is one step's settled disposition: the evidence source that let it be
// skipped, or the closed reason it had to run again.
type workflowResumeStep struct {
	Step     string `json:"step"`
	Action   string `json:"action"`
	Source   string `json:"source,omitempty"`
	Reason   string `json:"reason,omitempty"`
	Status   string `json:"status"`
	Attempts int    `json:"attempts,omitempty"`
	Err      string `json:"error,omitempty"`
}

// workflowStepRefusal is the typed refusal a step yields when this runner cannot perform
// its op — a flat object, never prose, so a caller folds it instead of reading it.
type workflowStepRefusal struct {
	Refusal string `json:"refusal"`
	Step    string `json:"step"`
	Op      string `json:"op"`
}

func (r workflowStepRefusal) Error() string {
	b, err := json.Marshal(r)
	if err != nil {
		return `{"refusal":"op_unregistered"}`
	}
	return string(b)
}

// workflowResume is the verb body. corr is the evidence oracle for effectful steps; nil
// means open the real git-backed dos_verify ladder rooted at --repo.
func workflowResume(stdout, stderr io.Writer, argv []string, corr workflow.Corroborate) int {
	fs := flag.NewFlagSet("workflow resume", flag.ContinueOnError)
	fs.SetOutput(stderr)
	asJSON := fs.Bool("json", false, "emit the resume report as JSON")
	epochLabel := fs.String("epoch", "", "epoch label folded into every step's cache key (a policy revision)")
	nowMS := fs.Int64("now-ms", 0, "injected fold clock in unix milliseconds (0 = read the wall clock once, here)")
	repo := fs.String("repo", ".", "repository the dos_verify ladder reads its evidence from")
	// Accept flags before OR after the run directory: Go's flag package stops at the
	// first non-flag argument, so interleave Parse with positional collection (the same
	// ergonomics `fak workflow lint` already has).
	pos, err := interspersedflags.Parse(fs, argv)
	if err != nil {
		return 2
	}
	if len(pos) != 1 {
		fmt.Fprintln(stderr, "usage: fak workflow resume <run-dir> [--json] [--epoch LABEL] [--now-ms N] [--repo DIR]")
		return 2
	}
	dir := pos[0]

	specDoc, err := os.ReadFile(filepath.Join(dir, workflowSpecFile))
	if err != nil {
		fmt.Fprintf(stderr, "fak workflow resume: %v\n", err)
		return 2
	}
	graph, err := workflow.CompileJSON(specDoc)
	if err != nil {
		fmt.Fprintf(stderr, "fak workflow resume: %v\n", err)
		return 2
	}
	journalPath := filepath.Join(dir, workflowJournalFile)
	rows, err := workflowReadJournalFile(journalPath)
	if err != nil {
		fmt.Fprintf(stderr, "fak workflow resume: %v\n", err)
		return 2
	}
	at := *nowMS
	if at == 0 {
		at = time.Now().UnixMilli()
	}
	state, err := workflow.Fold(rows, at)
	if err != nil {
		fmt.Fprintf(stderr, "fak workflow resume: %v\n", err)
		return 2
	}
	if corr == nil {
		corr = workflowClaimOracle(witness.NewWithRunner(workflowGitRunner, *repo))
	}

	ctx := context.Background()
	epoch := workflow.GraphEpoch(graph, *epochLabel)
	resumption := workflow.Resume(ctx, graph, state, epoch, corr)
	runner := workflow.NewResumeRunner(resumption, workflow.RunnerFunc(workflowRunStep))
	result := workflow.Execute(ctx, graph, runner, workflow.Options{ContinueOnError: true})

	report := workflowResumeReport{Run: resumption.Run, Epoch: epoch, Rows: state.Rows}
	fresh := make([]workflow.Entry, 0, len(graph.Nodes))
	hashes := make(map[string]string, len(graph.Nodes))
	ran := workflowSet(runner.Executed())
	replayed := workflowSet(runner.Skipped())
	for _, node := range graph.Nodes {
		v, _ := runner.Verdict(node.ID)
		nr := result.Nodes[node.ID]
		row := workflowResumeStep{
			Step: node.ID, Action: string(v.Disposition), Source: v.Source, Reason: v.Reason,
			Status: string(nr.Status), Attempts: nr.Attempts, Err: nr.Err,
		}
		switch {
		case replayed[node.ID]:
			row.Action = "skip"
			report.Skipped++
		case ran[node.ID]:
			row.Action = "exec"
			report.Executed++
		default:
			row.Action = "unreached" // the engine never got here: an earlier failure stopped it
		}
		if nr.Status == workflow.StatusFailed {
			report.Failed++
			if r, ok := workflowParseRefusal(nr.Err); ok {
				report.Refusals = append(report.Refusals, r)
			}
		}
		// A replayed step contributes the hash its journal row was keyed by — the same
		// value Resume folded — so a later resume compares like with like.
		if replayed[node.ID] {
			hashes[node.ID] = state.Steps[node.ID].OutputHash
		} else {
			hashes[node.ID] = workflow.HashOutput(nr.Output)
		}
		if ran[node.ID] && nr.Status == workflow.StatusSucceeded {
			fresh = append(fresh, workflow.Entry{
				Run: report.Run, Step: node.ID, Kind: workflow.StepPure,
				InputsHash: workflow.StepInputsHash(node, hashes), EpochHash: epoch,
				OutputHash: hashes[node.ID], Output: nr.Output, TSMS: at,
			})
		}
		report.Steps = append(report.Steps, row)
	}
	if err := workflowAppendJournal(journalPath, fresh); err != nil {
		fmt.Fprintf(stderr, "fak workflow resume: %v\n", err)
		return 2
	}

	if *asJSON {
		enc := json.NewEncoder(stdout)
		enc.SetIndent("", "  ")
		if err := enc.Encode(report); err != nil {
			fmt.Fprintf(stderr, "fak workflow resume: %v\n", err)
			return 1
		}
	} else {
		for _, s := range report.Steps {
			switch s.Action {
			case "skip":
				fmt.Fprintf(stdout, "  skip %-24s witness=%s\n", s.Step, s.Source)
			case "exec":
				fmt.Fprintf(stdout, "  exec %-24s reason=%s status=%s\n", s.Step, s.Reason, s.Status)
			default:
				fmt.Fprintf(stdout, "  %-4s %-24s reason=%s status=%s\n", s.Action, s.Step, s.Reason, s.Status)
			}
		}
		fmt.Fprintf(stdout, "run=%s skipped=%d executed=%d failed=%d\n",
			report.Run, report.Skipped, report.Executed, report.Failed)
	}
	if report.Failed > 0 {
		return 1
	}
	return 0
}

// workflowRunStep is the built-in op table a local resume can perform itself: the two pure
// ops the DSL's own patterns are written against. Anything else — an agent call, a lane
// dispatch — has no local executor yet and yields a typed refusal rather than a guess.
func workflowRunStep(_ context.Context, in workflow.RunInput) (string, error) {
	switch in.Node.Op {
	case "emit", "":
		return in.Node.Payload, nil
	case "join":
		keys := make([]string, 0, len(in.Deps))
		for k := range in.Deps {
			keys = append(keys, k)
		}
		sort.Strings(keys)
		vals := make([]string, 0, len(keys))
		for _, k := range keys {
			vals = append(vals, in.Deps[k])
		}
		return strings.Join(vals, "\n"), nil
	default:
		return "", workflowStepRefusal{Refusal: "op_unregistered", Step: in.Node.ID, Op: in.Node.Op}
	}
}

// workflowClaimOracle corroborates an effectful step's journaled claim through the
// in-process dos_verify ladder — the same evidence rungs (registry-shaped ancestry, a
// tracked path, a ship-stamp grep) the kernel's require-witness gate uses.
//
// The second rung is what makes a revert re-execute: `git revert` leaves the reverted
// commit an ancestor of HEAD and its object present, so a commit-shaped claim would keep
// corroborating forever off its own tombstone. Reading the revert out of history refuses
// that cache line, which is the difference between evidence and narration.
func workflowClaimOracle(res *witness.Resolver) workflow.Corroborate {
	return func(ctx context.Context, _, claim string) (string, bool) {
		call := &abi.ToolCall{Tool: workflowResumeTool}
		if res.Resolve(ctx, call, claim) != abi.WitnessConfirmed {
			return "", false
		}
		if ref, ok := workflowCommitClaimRef(claim); ok {
			if res.Resolve(ctx, call, "grep:This reverts commit "+ref) == abi.WitnessConfirmed {
				return "", false
			}
		}
		return "dos_verify:" + claim, true
	}
}

// workflowCommitClaimRef names the commit a claim is anchored on, for the claim shapes
// whose evidence survives the undoing of their own effect.
func workflowCommitClaimRef(claim string) (string, bool) {
	kind, arg, ok := strings.Cut(claim, ":")
	if !ok {
		return "", false
	}
	arg = strings.TrimSpace(arg)
	switch strings.ToLower(strings.TrimSpace(kind)) {
	case "ancestor", "commit":
		return arg, arg != ""
	}
	return "", false
}

func workflowGitRunner(ctx context.Context, dir string, args ...string) (string, int, error) {
	cmd := exec.CommandContext(ctx, "git", args...)
	cmd.Dir = dir
	configureDispatchHelperCommand(cmd)
	var out strings.Builder
	cmd.Stdout = &out
	err := cmd.Run()
	if err == nil {
		return out.String(), 0, nil
	}
	var exit *exec.ExitError
	if errors.As(err, &exit) {
		return out.String(), exit.ExitCode(), nil
	}
	return out.String(), -1, err
}

func workflowReadJournalFile(path string) ([]workflow.Entry, error) {
	f, err := os.Open(path)
	if errors.Is(err, os.ErrNotExist) {
		return nil, nil // a run with no journal yet: every step is unwitnessed
	}
	if err != nil {
		return nil, err
	}
	defer f.Close()
	return workflow.ReadJournal(f)
}

func workflowAppendJournal(path string, rows []workflow.Entry) error {
	if len(rows) == 0 {
		return nil
	}
	f, err := os.OpenFile(path, os.O_APPEND|os.O_CREATE|os.O_WRONLY, 0o644)
	if err != nil {
		return err
	}
	defer f.Close()
	for _, r := range rows {
		if err := workflow.AppendEntry(f, r); err != nil {
			return err
		}
	}
	return nil
}

func workflowParseRefusal(s string) (workflowStepRefusal, bool) {
	var r workflowStepRefusal
	if err := json.Unmarshal([]byte(s), &r); err != nil || r.Refusal == "" {
		return workflowStepRefusal{}, false
	}
	return r, true
}

func workflowSet(ids []string) map[string]bool {
	out := make(map[string]bool, len(ids))
	for _, id := range ids {
		out[id] = true
	}
	return out
}
