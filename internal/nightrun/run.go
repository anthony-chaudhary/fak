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
// artifact (when any), an OBSERVED headline number (only when parsed — never
// fabricated), and timing. A dry-run Run executed nothing.
type Run struct {
	Task     Task          `json:"task"`
	Outcome  Outcome       `json:"outcome"`
	Artifact string        `json:"artifact,omitempty"`
	Number   string        `json:"number,omitempty"`
	Err      string        `json:"err,omitempty"`
	Duration time.Duration `json:"-"`
}

// RunSummary is the result of a RunLoop invocation — the sequence of attempts and
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
	Executor    Executor     // nil => DefaultExecutor (shell)
	ReadLedger  func() []CollectRow
	AppendRow   func(CollectRow) error
}

// RunLoop drives the collection loop: rank → pick the most important feasible
// task not yet attempted this session → (dry-run: record the plan / apply:
// execute + append a ledger row) → repeat. Each Task is attempted at most once
// per invocation (so a failing task cannot spin the loop), and Max bounds the
// total attempts. It returns the honest summary of what happened.
func RunLoop(ctx context.Context, opts RunOptions) (RunSummary, error) {
	if opts.Executor == nil {
		opts.Executor = DefaultExecutor
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

	for {
		if opts.Max > 0 && len(summary.Runs) >= opts.Max {
			summary.StopReason = fmt.Sprintf("reached --max %d", opts.Max)
			break
		}
		ranked := Rank(opts.Tasks, opts.Caps, opts.ReadLedger(), opts.Now)
		next, ok := pickFresh(ranked, attempted)
		if !ok {
			summary.StopReason = stopReason(ranked, len(summary.Runs))
			break
		}
		attempted[next.Task.ID] = true

		if !opts.Apply {
			summary.Runs = append(summary.Runs, Run{
				Task:     next.Task,
				Outcome:  OutcomeDryRun,
				Artifact: relArtifact(opts, next.Task),
			})
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

func stopReason(ranked []Scored, ran int) string {
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
	return fmt.Sprintf("collected the whole feasible queue (%d task(s)) — nothing left", ran)
}

// numberRE best-effort-extracts a headline number with a known unit from a run's
// output. It is intentionally narrow: an empty match yields no number (the row's
// Number stays blank) rather than a guessed value.
var numberRE = regexp.MustCompile(`([0-9]+(?:\.[0-9]+)?)\s*(tok/s|tokens/s|GB/s|ms/tok|it/s)`)

// DefaultExecutor runs t.Run through the platform shell, tees the combined output
// to artifactPath, and reports the OBSERVED outcome: a zero exit with captured
// output is collected; a non-zero exit is failed. It parses a headline number
// only if the output clearly contains one — otherwise the Number is left empty.
func DefaultExecutor(ctx context.Context, t Task, artifactPath string) (Outcome, string, time.Duration, error) {
	start := time.Now()
	shell, flag := "sh", "-c"
	if runtime.GOOS == "windows" {
		shell, flag = "cmd", "/c"
	}
	cmd := exec.CommandContext(ctx, shell, flag, t.Run)
	out, err := cmd.CombinedOutput()
	dur := time.Since(start)

	// The disk-wiring seam owns its own directory: RunLoop stays seam-pure (no
	// filesystem side-effects of its own), so a fully-faked loop touches no disk.
	if derr := os.MkdirAll(filepath.Dir(artifactPath), 0o755); derr != nil {
		return OutcomeFailed, "", dur, fmt.Errorf("create artifact dir: %w", derr)
	}
	if werr := os.WriteFile(artifactPath, out, 0o644); werr != nil {
		return OutcomeFailed, "", dur, fmt.Errorf("capture output: %w", werr)
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
// on first write. Append-only — it never rewrites an existing row.
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
