package nightrun

import (
	"context"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"runtime"
	"strings"
	"time"
)

// Run is one attempt the loop made on a Task: the OBSERVED outcome, the captured
// artifact (when any), an OBSERVED headline number (only when parsed â€” never
// fabricated), and timing. A dry-run Run executed nothing.
type Run struct {
	Task     Task          `json:"task"`
	Outcome  Outcome       `json:"outcome"`
	Artifact string        `json:"artifact,omitempty"`
	Number   string        `json:"number,omitempty"`
	Err      string        `json:"err,omitempty"`
	Duration time.Duration `json:"-"`
}

// RunSummary is the result of a RunLoop invocation â€” the sequence of attempts and
// why the loop stopped. It is the honest record of a "run it all night" session.
type RunSummary struct {
	Box        string `json:"box"`
	Applied    bool   `json:"applied"`
	Runs       []Run  `json:"runs"`
	StopReason string `json:"stop_reason"`
}

// Executor runs one Task's command, writes its combined output to artifactPath,
// and returns the OBSERVED outcome, a parsed headline number (empty when none),
// the wall duration, and any error. It is a seam so a test drives the loop
// without shelling out.
type Executor func(ctx context.Context, t Task, artifactPath string) (Outcome, string, time.Duration, error)

// RunOptions configures one RunLoop invocation. The read/append/executor seams
// default to the live disk + shell wiring when nil, so the cmd layer can pass
// nothing and a test can pass fakes.
type RunOptions struct {
	Root        string       // repo root, for resolving artifact + ledger paths
	Caps        Capabilities // the probed box
	Tasks       []Task       // the assembled backlog
	Now         time.Time    // injected clock (artifact stamp + ledger date)
	Apply       bool         // execute for real (else dry-run: print, write nothing)
	Loop        bool         // keep collecting until Max / queue exhausted
	Max         int          // cap on attempts (0 = unbounded within a loop; 1 implied when !Loop)
	ArtifactDir string       // base dir for captured output (default experiments/nightrun)
	LedgerPath  string       // ledger file (default <root>/DefaultLedgerRel)
	Executor    Executor     // nil => execTask via the live shell (go-run benches pre-built once)
	ReadLedger  func() []CollectRow
	AppendRow   func(CollectRow) error
}

// RunLoop drives the collection loop: rank â†’ pick the most important feasible
// task not yet attempted this session â†’ (dry-run: record the plan / apply:
// execute + append a ledger row) â†’ repeat. Each Task is attempted at most once
// per invocation (so a failing task cannot spin the loop), and Max bounds the
// total attempts. It returns the honest summary of what happened.
func RunLoop(ctx context.Context, opts RunOptions) (RunSummary, error) {
	if opts.Executor == nil {
		root := opts.Root
		cache := newGoRunCache()
		defer cache.cleanup()
		opts.Executor = func(ctx context.Context, t Task, artifactPath string) (Outcome, string, time.Duration, error) {
			return execTask(ctx, root, cache, t, artifactPath)
		}
	}
	if opts.LedgerPath == "" {
		opts.LedgerPath = filepath.Join(opts.Root, filepath.FromSlash(DefaultLedgerRel))
	}
	if opts.ArtifactDir == "" {
		opts.ArtifactDir = filepath.Join(opts.Root, "experiments", "nightrun")
	}
	if opts.ReadLedger == nil {
		opts.ReadLedger = func() []CollectRow { return ReadLedgerFile(opts.LedgerPath) }
	}
	if opts.AppendRow == nil {
		opts.AppendRow = func(r CollectRow) error { return AppendLedgerFile(opts.LedgerPath, r) }
	}

	summary := RunSummary{Box: opts.Caps.Box, Applied: opts.Apply}
	attempted := map[string]bool{}
	realAttempts := 0 // executed/shown tasks; a skip is free (does not count toward --max)

	for {
		if opts.Max > 0 && realAttempts >= opts.Max {
			summary.StopReason = fmt.Sprintf("reached --max %d", opts.Max)
			break
		}
		ranked := Rank(opts.Tasks, opts.Caps, opts.ReadLedger(), opts.Now)
		next, ok := pickFresh(ranked, attempted)
		if !ok {
			summary.StopReason = stopReason(ranked, realAttempts, opts.Apply)
			break
		}
		attempted[next.Task.ID] = true

		// A feasible-but-MANUAL task (its Run is a placeholder/prose recipe — a
		// curated witness that needs operator setup) is surfaced but never auto-run:
		// exec-ing the prose would just record a spurious failure every sweep. Record
		// it skipped (not a ledger datum, not a --max attempt) and pick the next.
		if !next.Task.autoRunnable() {
			summary.Runs = append(summary.Runs, Run{Task: next.Task, Outcome: OutcomeSkipped})
			continue
		}

		if !opts.Apply {
			summary.Runs = append(summary.Runs, Run{
				Task:     next.Task,
				Outcome:  OutcomeDryRun,
				Artifact: relArtifact(opts, next.Task),
			})
			realAttempts++
			if !opts.Loop {
				summary.StopReason = "dry-run (pass --apply to execute)"
				break
			}
			continue
		}

		artifact := absArtifact(opts, next.Task)
		outcome, number, dur, runErr := opts.Executor(ctx, next.Task, artifact)
		row := NewCollectRow(next.Task, opts.Caps.Box, outcome, toRel(opts.Root, artifact), number, dur, opts.Now)
		if err := opts.AppendRow(row); err != nil {
			return summary, fmt.Errorf("nightrun: append ledger: %w", err)
		}
		r := Run{Task: next.Task, Outcome: outcome, Artifact: toRel(opts.Root, artifact), Number: number, Duration: dur}
		if runErr != nil {
			r.Err = runErr.Error()
		}
		summary.Runs = append(summary.Runs, r)
		realAttempts++

		if !opts.Loop {
			summary.StopReason = "ran one task (pass --loop to continue)"
			break
		}
	}
	return summary, nil
}

// pickFresh returns the highest-priority feasible Scored whose Task has not been
// attempted this session.
func pickFresh(ranked []Scored, attempted map[string]bool) (Scored, bool) {
	for _, s := range ranked {
		if s.Feasible && !attempted[s.Task.ID] {
			return s, true
		}
	}
	return Scored{}, false
}

func stopReason(ranked []Scored, ran int, applied bool) string {
	anyFeasible := false
	for _, s := range ranked {
		if s.Feasible {
			anyFeasible = true
			break
		}
	}
	if !anyFeasible {
		return "no feasible task on this box (every candidate needs a missing capability)"
	}
	if ran == 0 {
		return "nothing to collect"
	}
	// Outcome-neutral: the feasible queue was exhausted, but the attempts may have
	// failed/timed out (and in a dry run nothing executed) â€” so never claim "collected"
	// here. Each attempt's true outcome is in its ledger row.
	verb := "planned"
	if applied {
		verb = "attempted"
	}
	return fmt.Sprintf("%s the whole feasible queue (%d task(s)) â€” nothing left", verb, ran)
}

// numberRE best-effort-extracts a headline number with a known unit from a run's
// output. It is intentionally narrow: an empty match yields no number (the row's
// Number stays blank) rather than a guessed value.
var numberRE = regexp.MustCompile(`([0-9]+(?:\.[0-9]+)?)\s*(tok/s|tokens/s|GB/s|ms/tok|it/s)`)

// DefaultExecutor runs t.Run through the platform shell, tees the combined output
// to artifactPath, and reports the OBSERVED outcome: a zero exit with captured
// output is collected; a non-zero exit is failed; exceeding the task's wall-clock
// budget (t.timeout()) is a timeout â€” the process is killed, the partial output is
// still captured, and the loop moves on (so one slow/hung task cannot stall an
// unattended --loop). It parses a headline number only if the output clearly
// contains one â€” otherwise the Number is left empty.
// DefaultExecutor is the back-compat seam (root derived as "" => no prebuild, plain go run).
// RunLoop wires execTask with the real repo root so go-run benches are pre-built once.
func DefaultExecutor(ctx context.Context, t Task, artifactPath string) (Outcome, string, time.Duration, error) {
	return execTask(ctx, "", nil, t, artifactPath)
}

// execTask runs t.Run through the platform shell, tees output to artifactPath, and reports
// the OBSERVED outcome. When root != "" and the Run is `go run ./cmd/<x>`, the package is
// pre-built once (shared build cache) and the Run is rewritten to the binary, so a --loop
// pass is not dominated by per-task `go run` compile time (#965).
func execTask(ctx context.Context, root string, cache *goRunCache, t Task, artifactPath string) (Outcome, string, time.Duration, error) {
	start := time.Now()
	// Peel a leading POSIX `VAR=val ...` env prefix off the command and apply it via
	// cmd.Env below: that prefix syntax is invalid under Windows `cmd /c`, and cmd.Env
	// is shell-agnostic (valid under both `sh -c` and `cmd /c`). Stripping it first also
	// lets a `FAK_X=1 go run ./cmd/<x>` task become prebuild-eligible.
	envPrefix, run := splitEnvPrefix(t.Run)
	if root != "" && cache != nil {
		run = cache.maybePrebuildRun(ctx, root, run)
	}
	shell, flag := "sh", "-c"
	if runtime.GOOS == "windows" {
		shell, flag = "cmd", "/c"
	}
	// Bound the attempt: a per-task deadline on top of any caller deadline. The
	// whole process GROUP is killed on expiry (configureProcGroup) â€” exec.Command
	// only kills the direct child, so a `go run ./cmd/<bench>` whose compiled binary
	// is the real worker (or any grandchild) would otherwise outlive the kill AND
	// keep the output pipe open, blocking CombinedOutput past the deadline. WaitDelay
	// is the portable backstop: it force-closes the pipes if a straggler lingers, so
	// the loop is guaranteed to move on.
	runCtx, cancel := context.WithTimeout(ctx, t.timeout())
	defer cancel()
	cmd := exec.CommandContext(runCtx, shell, flag, run)
	if len(envPrefix) > 0 {
		cmd.Env = append(os.Environ(), envPrefix...)
	}
	configureProcGroup(cmd)
	cmd.WaitDelay = 10 * time.Second
	out, err := cmd.CombinedOutput()
	dur := time.Since(start)

	// The disk-wiring seam owns its own directory: RunLoop stays seam-pure (no
	// filesystem side-effects of its own), so a fully-faked loop touches no disk.
	if derr := os.MkdirAll(filepath.Dir(artifactPath), 0o755); derr != nil {
		return OutcomeFailed, "", dur, fmt.Errorf("create artifact dir: %w", derr)
	}
	// Capture whatever output the command produced before it ended â€” a timed-out
	// run's partial log is still evidence of how far it got.
	if werr := os.WriteFile(artifactPath, out, 0o644); werr != nil {
		return OutcomeFailed, "", dur, fmt.Errorf("capture output: %w", werr)
	}
	// A deadline hit is reported as a timeout, not a generic failure: it is the
	// budget firing, not the command itself exiting non-zero.
	if runCtx.Err() == context.DeadlineExceeded {
		return OutcomeTimeout, "", dur, fmt.Errorf("task %q exceeded its %s budget", t.ID, t.timeout())
	}
	if err != nil {
		return OutcomeFailed, "", dur, err
	}
	return OutcomeCollected, parseNumber(string(out)), dur, nil
}

func parseNumber(out string) string {
	m := numberRE.FindStringSubmatch(out)
	if len(m) == 3 {
		return m[1] + " " + m[2]
	}
	return ""
}

// envAssignRE matches a leading `NAME=` token â€” the POSIX env-assignment shape.
var envAssignRE = regexp.MustCompile(`^[A-Za-z_][A-Za-z0-9_]*=`)

// splitEnvPrefix peels a leading `VAR=val ...` env-assignment prefix off a command,
// returning the assignments (NAME=VALUE, applied via cmd.Env) and the residual
// command. No prefix returns (nil, run) unchanged. Tokenisation is whitespace-split,
// matching the package's other simple command parsers (parseGoRun); it does not
// handle a quoted value with spaces, which none of the built-in tasks use.
func splitEnvPrefix(run string) ([]string, string) {
	fields := strings.Fields(run)
	var env []string
	i := 0
	for i < len(fields) && envAssignRE.MatchString(fields[i]) {
		env = append(env, fields[i])
		i++
	}
	if len(env) == 0 {
		return nil, run
	}
	return env, strings.Join(fields[i:], " ")
}

// ReadLedgerFile reads + parses the ledger file, tolerating a missing file (no
// rows). The first collection establishes the series.
func ReadLedgerFile(path string) []CollectRow {
	raw, err := os.ReadFile(path)
	if err != nil {
		return nil
	}
	return ParseLedger(string(raw))
}

// AppendLedgerFile appends one row to the ledger, creating the parent directory
// on first write. Append-only â€” it never rewrites an existing row.
func AppendLedgerFile(path string, row CollectRow) error {
	line, err := AppendLedgerLine(row)
	if err != nil {
		return err
	}
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		return err
	}
	f, err := os.OpenFile(path, os.O_CREATE|os.O_WRONLY|os.O_APPEND, 0o644)
	if err != nil {
		return err
	}
	defer f.Close()
	_, err = f.WriteString(line + "\n")
	return err
}

func absArtifact(opts RunOptions, t Task) string {
	stamp := opts.Now.UTC().Format("20060102T150405Z")
	box := sanitize(opts.Caps.Box)
	name := fmt.Sprintf("%s-%s.log", stamp, sanitize(t.ID))
	return filepath.Join(opts.ArtifactDir, box, name)
}

func relArtifact(opts RunOptions, t Task) string {
	return toRel(opts.Root, absArtifact(opts, t))
}

func toRel(root, p string) string {
	if rel, err := filepath.Rel(root, p); err == nil {
		return filepath.ToSlash(rel)
	}
	return filepath.ToSlash(p)
}

func sanitize(s string) string {
	s = strings.TrimSpace(s)
	var b strings.Builder
	for _, r := range s {
		switch {
		case r >= 'a' && r <= 'z', r >= 'A' && r <= 'Z', r >= '0' && r <= '9', r == '-', r == '_':
			b.WriteRune(r)
		default:
			b.WriteRune('-')
		}
	}
	if b.Len() == 0 {
		return "unknown"
	}
	return b.String()
}
