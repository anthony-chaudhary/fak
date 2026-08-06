package main

import (
	"context"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"sort"
	"strings"
	"time"
)

// `fak bench gitspawn`  -  count GIT PROCESS SPAWNS per unit of work on the three hot
// paths (#5620).
//
// WHY A COUNT AND NOT A DURATION. Every claim in epic #5619 is a spawn-count claim: the
// cost it is about is paid *per process created*, not per millisecond elapsed. Wall-clock
// moves with host load and hides that cost, so this verb reports a count -- and, because a
// count with an implicit interval is not a measurement (the defect `internal/stallscan`
// shipped once and now carries `SpawnWindowSeconds` to avoid), every count here is emitted
// with the window it was counted over.
//
// HOW THE COUNTING WORKS -- AND WHY NOT BY POLLING. Sampling the process table cannot see a
// process that is born and dies between two samples; a PID-diff poll sampler measured a 20x
// undercount against a ground truth of 200 injected ~40ms spawns (it caught 10), and the
// bias grows as processes get shorter, which is exactly where the churn this epic targets
// lives. So this counts AT THE SOURCE instead: git's own Trace2 event stream. Setting
// GIT_TRACE2_EVENT to a directory makes every git process write its own event file, with a
// `start` event carrying its argv, before it does any work. The env var is inherited, so a
// hot path's git children are counted no matter which of the ~100 `exec.Command("git", ...)`
// sites in this tree issued them -- no wrapper to thread through, and no change to any
// existing path. It is event-driven and per-process, so a 1ms git is counted exactly like a
// 10s one.
//
// The verb self-calibrates before it measures: it injects a known number of short-lived git
// spawns and reports counted-vs-injected, so the undercount factor of the instrument is a
// measured number in the report rather than an assertion.
//
// ATTRIBUTION. Trace2 `sid` is hierarchical -- a git process spawned BY git carries its
// parent's sid plus its own, joined by '/'. A depth-0 sid therefore means "spawned by a
// non-git parent", i.e. by the hot path itself. That depth-0 count is the exec-seam number
// this issue asks for; git's own internal children are reported separately as
// `nested_spawns` and are out of scope for the headline count.
//
//	fak bench gitspawn                  measure all three hot paths, human-readable
//	fak bench gitspawn --json           the same as a fak-gitspawn-bench/1 report
//	fak bench gitspawn --out b.json     also write the report (this is how the fixture is made)
//	fak bench gitspawn --baseline f     print the delta against a committed baseline fixture
//	fak bench gitspawn --path stop-hook measure one named path only
const (
	// gitSpawnTraceEnv is git's own event-stream target. Pointed at a DIRECTORY it
	// yields one file per git process, which sidesteps concurrent-append loss.
	gitSpawnTraceEnv = "GIT_TRACE2_EVENT"

	// gitSpawnSchema versions the report so a later rung can tell a stale fixture from a
	// current one instead of silently subtracting from the wrong shape.
	gitSpawnSchema = "fak-gitspawn-bench/1"

	// gitSpawnMethod names the counting method in the report. The acceptance gate for
	// #5620 requires the method to be stated wherever the number is.
	gitSpawnMethod = "git-trace2-event"

	// gitSpawnDiscardSentinel is the file git drops when `trace2.maxFiles` is exceeded and
	// it STOPS writing per-process files. Its presence means the directory undercounted,
	// so the run is reported as invalid rather than as a small number.
	gitSpawnDiscardSentinel = "git-trace2-discard"

	// gitSpawnCalibrationN is the injected ground truth. 200 matches the sample size the
	// poll sampler was measured against, so the two numbers are directly comparable.
	gitSpawnCalibrationN = 200
)

// gitSpawnCmdCount is one git subcommand and how many times the unit of work spawned it.
// This is what turns a bare total into something actionable for the later rungs: it names
// which call is worth batching.
type gitSpawnCmdCount struct {
	Command string `json:"command"`
	Spawns  int    `json:"spawns"`
}

// gitSpawnPathReport is one hot path's measurement. Spawns and WindowSeconds travel
// together on purpose: a count without its window is not a measurement.
type gitSpawnPathReport struct {
	Path            string             `json:"path"`
	Unit            string             `json:"unit"`
	Entrypoint      string             `json:"entrypoint"`
	Spawns          int                `json:"spawns"`
	NestedSpawns    int                `json:"nested_spawns"`
	WindowSeconds   float64            `json:"window_seconds"`
	SpawnsPerSecond float64            `json:"spawns_per_second"`
	ExitCode        int                `json:"exit_code"`
	Note            string             `json:"note,omitempty"`
	TopCommands     []gitSpawnCmdCount `json:"top_commands,omitempty"`
}

// gitSpawnCalibration is the instrument measuring itself: Injected short-lived git spawns
// against Counted. UndercountX is Injected/Counted -- 1.0 means the instrument lost nothing.
type gitSpawnCalibration struct {
	Method        string  `json:"method"`
	Injected      int     `json:"injected"`
	Counted       int     `json:"counted"`
	UndercountX   float64 `json:"undercount_x"`
	WindowSeconds float64 `json:"window_seconds"`
}

// gitSpawnReport is the whole emission, and the on-disk shape of the baseline fixture.
type gitSpawnReport struct {
	Schema      string               `json:"schema"`
	Method      string               `json:"method"`
	MethodNote  string               `json:"method_note"`
	GOOS        string               `json:"goos"`
	GitVersion  string               `json:"git_version,omitempty"`
	Calibration gitSpawnCalibration  `json:"calibration"`
	Paths       []gitSpawnPathReport `json:"paths"`
}

// gitSpawnMethodNote is the standing caveat that ships with every number: what the count
// does and does not include. Stated here once so the fixture, the JSON, and the table can
// never disagree about it.
const gitSpawnMethodNote = "counted at the source from git's own Trace2 `start` event (one per git process, written before the process does any work); " +
	"NOT by polling the process table, which undercounts short-lived processes ~20x. " +
	"`spawns` counts git processes whose Trace2 sid has depth 0, i.e. those exec'd by the hot path itself; " +
	"git's own internal git children are reported separately as `nested_spawns` and are out of scope for the headline count."

// gitSpawnScenario is one unit of work to measure. Argv is run as a real subprocess so the
// number is the one a fleet actually pays, and so a wedged path can be bounded by a timeout
// instead of hanging the benchmark.
type gitSpawnScenario struct {
	Name       string
	Unit       string
	Entrypoint string
	Dir        string
	Argv       []string
}

func cmdGitSpawnBench(argv []string) { os.Exit(runGitSpawnBench(os.Stdout, os.Stderr, argv)) }

func runGitSpawnBench(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("bench gitspawn", flag.ContinueOnError)
	fs.SetOutput(stderr)
	verbFlagUsage(fs, "bench")
	asJSON := fs.Bool("json", false, "emit the fak-gitspawn-bench/1 report as JSON")
	out := fs.String("out", "", "also write the JSON report to this path (how the baseline fixture is produced)")
	baseline := fs.String("baseline", "", "compare against a committed baseline report and print the per-path delta")
	only := fs.String("path", "", "measure only this hot path (stop-hook|dispatch-tick|commit-gate); default: all three")
	fakBin := fs.String("fak", "", "fak binary to drive the hot paths with (default: this executable)")
	timeoutS := fs.Int("timeout", 180, "per-path wall-clock bound, seconds; a path that exceeds it is reported as timed out, not as a small count")
	calibrateN := fs.Int("calibrate", gitSpawnCalibrationN, "injected short-lived git spawns used to measure the instrument's own undercount (0 skips)")
	if code, ok := parseFlagsOrHelp(fs, argv); !ok {
		return code
	}

	bin := strings.TrimSpace(*fakBin)
	if bin == "" {
		self, err := os.Executable()
		if err != nil {
			fmt.Fprintf(stderr, "fak bench gitspawn: locate this executable: %v\n", err)
			return 1
		}
		bin = self
	}
	if _, err := exec.LookPath("git"); err != nil {
		fmt.Fprintf(stderr, "fak bench gitspawn: git not on PATH: %v\n", err)
		return 1
	}

	work, err := os.MkdirTemp("", "fak-gitspawn-")
	if err != nil {
		fmt.Fprintf(stderr, "fak bench gitspawn: scratch dir: %v\n", err)
		return 1
	}

	// The scratch repo is seeded BEFORE the instrument is armed, so the seeding's own git
	// spawns never land in any measured window. Measuring against a scratch repo (rather
	// than the live checkout) is what makes this runnable on any host: no fleet, no
	// network, no admin, no scheduled task -- and no side effect on a shared trunk.
	repo, err := seedGitSpawnRepo(work)
	if err != nil {
		fmt.Fprintf(stderr, "fak bench gitspawn: seed scratch repo: %v\n", err)
		return 1
	}

	rep := gitSpawnReport{
		Schema:     gitSpawnSchema,
		Method:     gitSpawnMethod,
		MethodNote: gitSpawnMethodNote,
		GOOS:       runtime.GOOS,
		GitVersion: gitSpawnGitVersion(),
	}

	if *calibrateN > 0 {
		cal, err := calibrateGitSpawnCounter(filepath.Join(work, "calibrate"), repo, *calibrateN)
		if err != nil {
			fmt.Fprintf(stderr, "fak bench gitspawn: calibrate: %v\n", err)
			return 1
		}
		rep.Calibration = cal
	}

	scenarios := gitSpawnScenarios(bin, repo)
	if sel := strings.TrimSpace(*only); sel != "" {
		kept := scenarios[:0]
		for _, sc := range scenarios {
			if sc.Name == sel {
				kept = append(kept, sc)
			}
		}
		if len(kept) == 0 {
			fmt.Fprintf(stderr, "fak bench gitspawn: unknown --path %q (want stop-hook|dispatch-tick|commit-gate)\n", sel)
			return 2
		}
		scenarios = kept
	}

	timeout := time.Duration(*timeoutS) * time.Second
	for _, sc := range scenarios {
		pr, err := measureGitSpawnScenario(context.Background(), sc, filepath.Join(work, "ev-"+sc.Name), timeout)
		if err != nil {
			fmt.Fprintf(stderr, "fak bench gitspawn: %s: %v\n", sc.Name, err)
			return 1
		}
		rep.Paths = append(rep.Paths, pr)
	}

	if p := strings.TrimSpace(*out); p != "" {
		if err := writeIndentedJSONFile(p, rep); err != nil {
			fmt.Fprintf(stderr, "fak bench gitspawn: write %s: %v\n", p, err)
			return 1
		}
	}
	if *asJSON {
		return encodeJSONOrFail(stdout, stderr, rep, "fak bench gitspawn")
	}
	renderGitSpawnReport(stdout, rep)
	if p := strings.TrimSpace(*baseline); p != "" {
		base, err := loadGitSpawnReport(p)
		if err != nil {
			fmt.Fprintf(stderr, "fak bench gitspawn: read baseline %s: %v\n", p, err)
			return 1
		}
		renderGitSpawnDelta(stdout, base, rep)
	}
	return 0
}

// gitSpawnScenarios names the three hot paths #5620 asks for, each as the exact in-tree
// entrypoint the count is attributed to. Each runs against the scratch repo, so none of
// them touches the caller's checkout.
func gitSpawnScenarios(fakBin, repo string) []gitSpawnScenario {
	return []gitSpawnScenario{{
		Name:       "stop-hook",
		Unit:       "one Stop-hook fire",
		Entrypoint: "cmd/fak/guard_stophook.go -> runWipAutoCheckpoint",
		Dir:        repo,
		// The Stop hook's git work is exactly this call (guard_stophook.go calls it with
		// --reason stop); -C/--session only aim it at the scratch repo.
		Argv: []string{fakBin, "wip", "autocheckpoint", "-C", repo, "--session", "gitspawn-bench", "--reason", "stop"},
	}, {
		Name:       "dispatch-tick",
		Unit:       "one dispatch tick",
		Entrypoint: "cmd/fak/dispatch_tick.go -> runDispatchTick",
		Dir:        repo,
		// No --live: a tick that evaluates and decides, which is the git-spawning part.
		// --no-refresh drops the python registry scan (not a git cost) so the number is
		// the tick's own git fan-out.
		Argv: []string{fakBin, "dispatch", "tick", "--workspace", repo, "--no-refresh", "--no-loop-ledger", "--json"},
	}, {
		Name:       "commit-gate",
		Unit:       "one `fak commit` gate run",
		Entrypoint: "cmd/fak/commit.go -> runCommit",
		Dir:        repo,
		Argv: []string{fakBin, "commit", "--dir", repo, "--path", gitSpawnCommitFile,
			"-m", "test(gitspawn): commit-gate spawn probe (fak bench)", "--no-build-check"},
	}}
}

const (
	// gitSpawnWipFile is left dirty so the Stop hook has a real delta to checkpoint --
	// a clean tree would short-circuit the path and measure nothing.
	gitSpawnWipFile = "gitspawn-wip.txt"
	// gitSpawnCommitFile is the one path the commit gate is pointed at.
	gitSpawnCommitFile = "gitspawn-commit.txt"
)

// seedGitSpawnRepo builds the scratch repo the scenarios run against: on `main`, with one
// commit behind it, hooks disabled (an inherited core.hooksPath would measure the host's
// hooks rather than the path), and both a dirty file and a to-be-committed file present.
func seedGitSpawnRepo(work string) (string, error) {
	repo := filepath.Join(work, "repo")
	hooks := filepath.Join(work, "empty-hooks")
	for _, d := range []string{repo, hooks} {
		if err := os.MkdirAll(d, 0o755); err != nil {
			return "", err
		}
	}
	git := func(args ...string) error {
		c := exec.Command("git", append([]string{"-C", repo}, args...)...)
		c.Env = append(os.Environ(),
			"GIT_AUTHOR_NAME=fak bench", "GIT_AUTHOR_EMAIL=bench@fak.invalid",
			"GIT_COMMITTER_NAME=fak bench", "GIT_COMMITTER_EMAIL=bench@fak.invalid",
			// Never let an ambient trace target of the CALLER's leak into seeding.
			gitSpawnTraceEnv+"=",
		)
		if out, err := c.CombinedOutput(); err != nil {
			return fmt.Errorf("git %s: %v: %s", strings.Join(args, " "), err, strings.TrimSpace(string(out)))
		}
		return nil
	}
	if err := git("init", "-q", "-b", "main"); err != nil {
		return "", err
	}
	for _, kv := range [][2]string{
		{"core.hooksPath", hooks},
		{"user.name", "fak bench"},
		{"user.email", "bench@fak.invalid"},
		{"commit.gpgsign", "false"},
	} {
		if err := git("config", kv[0], kv[1]); err != nil {
			return "", err
		}
	}
	if err := os.WriteFile(filepath.Join(repo, "README.md"), []byte("# gitspawn bench scratch\n"), 0o644); err != nil {
		return "", err
	}
	if err := git("add", "README.md"); err != nil {
		return "", err
	}
	if err := git("commit", "-q", "-m", "seed"); err != nil {
		return "", err
	}
	for _, f := range []string{gitSpawnWipFile, gitSpawnCommitFile} {
		if err := os.WriteFile(filepath.Join(repo, f), []byte("spawn bench\n"), 0o644); err != nil {
			return "", err
		}
	}
	return repo, nil
}

// measureGitSpawnScenario runs one unit of work with the instrument armed and returns its
// spawn count together with the window that count covers. The window is wall-clock around
// the subprocess: it is not the unit of measurement, it is what makes the count readable.
func measureGitSpawnScenario(ctx context.Context, sc gitSpawnScenario, evDir string, timeout time.Duration) (gitSpawnPathReport, error) {
	if err := os.MkdirAll(evDir, 0o755); err != nil {
		return gitSpawnPathReport{}, err
	}
	rctx, cancel := context.WithTimeout(ctx, timeout)
	defer cancel()

	c := exec.CommandContext(rctx, sc.Argv[0], sc.Argv[1:]...)
	c.Dir = sc.Dir
	c.Env = append(os.Environ(), gitSpawnTraceEnv+"="+evDir)
	c.Stdout, c.Stderr = io.Discard, io.Discard

	start := time.Now()
	err := c.Run()
	window := time.Since(start)

	pr := gitSpawnPathReport{
		Path:          sc.Name,
		Unit:          sc.Unit,
		Entrypoint:    sc.Entrypoint,
		WindowSeconds: window.Seconds(),
	}
	var ee *exec.ExitError
	switch {
	case err == nil:
		pr.ExitCode = 0
	case errors.As(err, &ee):
		pr.ExitCode = ee.ExitCode()
	default:
		// Could not run it at all -- report that instead of a zero that reads like a win.
		pr.ExitCode = -1
		pr.Note = "did not run: " + err.Error()
	}
	if rctx.Err() != nil {
		pr.Note = fmt.Sprintf("timed out after %s; the count below is a LOWER BOUND", timeout)
	}

	count, err := readGitSpawnEvents(evDir)
	if err != nil {
		return pr, err
	}
	pr.Spawns, pr.NestedSpawns, pr.TopCommands = count.Top, count.Nested, count.topCommands(5)
	if count.Discarded {
		pr.Note = strings.TrimSpace(pr.Note + " trace2.maxFiles was exceeded (git-trace2-discard present): this count UNDERCOUNTS; raise trace2.maxFiles or unset it.")
	}
	if window > 0 {
		pr.SpawnsPerSecond = float64(pr.Spawns) / window.Seconds()
	}
	return pr, nil
}

// gitSpawnCount is one event directory folded into counts.
type gitSpawnCount struct {
	Top       int
	Nested    int
	Commands  map[string]int
	Discarded bool
}

func (c gitSpawnCount) topCommands(n int) []gitSpawnCmdCount {
	out := make([]gitSpawnCmdCount, 0, len(c.Commands))
	for k, v := range c.Commands {
		out = append(out, gitSpawnCmdCount{Command: k, Spawns: v})
	}
	// Descending by count, then by name, so the table is stable across runs.
	sort.Slice(out, func(i, j int) bool {
		if out[i].Spawns != out[j].Spawns {
			return out[i].Spawns > out[j].Spawns
		}
		return out[i].Command < out[j].Command
	})
	if len(out) > n {
		out = out[:n]
	}
	return out
}

// gitSpawnStartEvent is the only Trace2 record this fold reads: one per git process.
type gitSpawnStartEvent struct {
	Event string   `json:"event"`
	Sid   string   `json:"sid"`
	Argv  []string `json:"argv"`
}

// readGitSpawnEvents folds a Trace2 event directory into a spawn count. Every git process
// writes its own file, so this is a per-process count and not a sample: a process that
// lived 1ms contributes exactly as much as one that lived 10s.
func readGitSpawnEvents(dir string) (gitSpawnCount, error) {
	out := gitSpawnCount{Commands: map[string]int{}}
	ents, err := os.ReadDir(dir)
	if err != nil {
		if os.IsNotExist(err) {
			return out, nil
		}
		return out, err
	}
	for _, e := range ents {
		if e.IsDir() {
			continue
		}
		if e.Name() == gitSpawnDiscardSentinel {
			out.Discarded = true
			continue
		}
		b, err := os.ReadFile(filepath.Join(dir, e.Name()))
		if err != nil {
			return out, err
		}
		for _, line := range strings.Split(string(b), "\n") {
			line = strings.TrimSpace(line)
			// Cheap prefilter: only `start` records carry argv, one per process.
			if line == "" || !strings.Contains(line, `"event":"start"`) {
				continue
			}
			var ev gitSpawnStartEvent
			if json.Unmarshal([]byte(line), &ev) != nil || ev.Event != "start" {
				continue
			}
			// Trace2 sid is hierarchical: depth 0 means a non-git parent exec'd it, which
			// is the exec seam this issue counts. Deeper sids are git's own children.
			if strings.Contains(ev.Sid, "/") {
				out.Nested++
				continue
			}
			out.Top++
			out.Commands[gitSpawnSubcommand(ev.Argv)]++
		}
	}
	return out, nil
}

// gitSpawnSubcommand reduces a git argv to the subcommand that identifies the call, so the
// per-command breakdown groups `git -C x rev-parse ...` with `git rev-parse ...`.
func gitSpawnSubcommand(argv []string) string {
	for i := 1; i < len(argv); i++ {
		a := argv[i]
		switch {
		case a == "-C" || a == "-c" || a == "--git-dir" || a == "--work-tree" || a == "--namespace":
			i++ // skip this flag's value
		case strings.HasPrefix(a, "-"):
		default:
			return a
		}
	}
	return "git"
}

// calibrateGitSpawnCounter measures the instrument against a known ground truth: n
// short-lived git processes, injected through the same armed directory the real paths use.
// The acceptance gate for #5620 requires that any method able to undercount report its
// measured undercount alongside its numbers -- this is that number, per run, per host.
func calibrateGitSpawnCounter(evDir, repo string, n int) (gitSpawnCalibration, error) {
	if err := os.MkdirAll(evDir, 0o755); err != nil {
		return gitSpawnCalibration{}, err
	}
	env := append(os.Environ(), gitSpawnTraceEnv+"="+evDir)
	start := time.Now()
	for i := 0; i < n; i++ {
		c := exec.Command("git", "-C", repo, "rev-parse", "--git-dir")
		c.Env = env
		c.Stdout, c.Stderr = io.Discard, io.Discard
		if err := c.Run(); err != nil {
			return gitSpawnCalibration{}, fmt.Errorf("inject spawn %d/%d: %w", i+1, n, err)
		}
	}
	window := time.Since(start)
	count, err := readGitSpawnEvents(evDir)
	if err != nil {
		return gitSpawnCalibration{}, err
	}
	cal := gitSpawnCalibration{
		Method:        gitSpawnMethod,
		Injected:      n,
		Counted:       count.Top,
		WindowSeconds: window.Seconds(),
	}
	if count.Top > 0 {
		cal.UndercountX = float64(n) / float64(count.Top)
	}
	return cal, nil
}

func gitSpawnGitVersion() string {
	out, err := exec.Command("git", "--version").Output()
	if err != nil {
		return ""
	}
	return strings.TrimSpace(strings.TrimPrefix(strings.TrimSpace(string(out)), "git version "))
}

func loadGitSpawnReport(path string) (gitSpawnReport, error) {
	var rep gitSpawnReport
	b, err := os.ReadFile(path)
	if err != nil {
		return rep, err
	}
	if err := json.Unmarshal(b, &rep); err != nil {
		return rep, err
	}
	if rep.Schema != gitSpawnSchema {
		return rep, fmt.Errorf("schema %q, want %q", rep.Schema, gitSpawnSchema)
	}
	return rep, nil
}

func renderGitSpawnReport(w io.Writer, rep gitSpawnReport) {
	fmt.Fprintf(w, "git process spawns per unit of work  (%s, git %s, %s)\n", rep.Method, rep.GitVersion, rep.GOOS)
	if rep.Calibration.Injected > 0 {
		fmt.Fprintf(w, "instrument calibration: counted %d of %d injected short-lived spawns over %.2fs -> measured undercount %.2fx\n",
			rep.Calibration.Counted, rep.Calibration.Injected, rep.Calibration.WindowSeconds, rep.Calibration.UndercountX)
	}
	fmt.Fprintln(w)
	fmt.Fprintf(w, "%-14s %8s %8s %10s %10s  %s\n", "path", "spawns", "nested", "window(s)", "spawns/s", "unit of work")
	for _, p := range rep.Paths {
		fmt.Fprintf(w, "%-14s %8d %8d %10.2f %10.1f  %s\n", p.Path, p.Spawns, p.NestedSpawns, p.WindowSeconds, p.SpawnsPerSecond, p.Unit)
	}
	fmt.Fprintln(w)
	for _, p := range rep.Paths {
		if len(p.TopCommands) == 0 && p.Note == "" {
			continue
		}
		parts := make([]string, 0, len(p.TopCommands))
		for _, c := range p.TopCommands {
			parts = append(parts, fmt.Sprintf("%s x%d", c.Command, c.Spawns))
		}
		fmt.Fprintf(w, "%-14s exit %d", p.Path, p.ExitCode)
		if len(parts) > 0 {
			fmt.Fprintf(w, "  top: %s", strings.Join(parts, ", "))
		}
		if p.Note != "" {
			fmt.Fprintf(w, "  NOTE: %s", p.Note)
		}
		fmt.Fprintln(w)
	}
	fmt.Fprintf(w, "\nmethod: %s\n", rep.MethodNote)
}

// renderGitSpawnDelta is what makes the committed fixture load-bearing: a later rung's
// improvement claim subtracts from a number it did not produce.
func renderGitSpawnDelta(w io.Writer, base, now gitSpawnReport) {
	byName := map[string]gitSpawnPathReport{}
	for _, p := range base.Paths {
		byName[p.Path] = p
	}
	fmt.Fprintf(w, "\ndelta vs baseline (%s, git %s)\n", base.GOOS, base.GitVersion)
	fmt.Fprintf(w, "%-14s %10s %10s %10s\n", "path", "baseline", "now", "delta")
	for _, p := range now.Paths {
		b, ok := byName[p.Path]
		if !ok {
			fmt.Fprintf(w, "%-14s %10s %10d %10s\n", p.Path, "-", p.Spawns, "new")
			continue
		}
		fmt.Fprintf(w, "%-14s %10d %10d %+10d\n", p.Path, b.Spawns, p.Spawns, p.Spawns-b.Spawns)
	}
}
