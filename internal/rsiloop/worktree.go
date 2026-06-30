package rsiloop

// worktree.go is the REAL measurement seam — the impls that make the loop true
// rather than hand-fed. Each candidate is applied to a fresh DETACHED git worktree
// off the baseline ref (so `main` is never touched), its KPI is read by actually
// running cmd/kpiprobe in that worktree, its suite-green bit comes from a real
// build+vet, and its truth-clean bit comes from inspecting the worktree's git
// status — the loop author supplies NONE of these. Like internal/shipgate, this is
// the RSI harness and uses os/exec; it is not the dispatch hot path, so the
// os/exec-absence proof does not apply here.
//
// The git repo root and the Go module dir are NOT the same here: the fak module is
// a subdir of the repo. A worktree off a ref checks out the WHOLE repo, so the
// module lives at <worktree>/<moduleRel> (e.g. <worktree>/fak). Every go/probe/edit
// runs against that module dir; git status runs at the worktree root.

import (
	"errors"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"strconv"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/shipgate"
	"github.com/anthony-chaudhary/fak/internal/windowgate"
)

// WorktreeConfig parameterizes the real harness.
type WorktreeConfig struct {
	Repo        string     // a path inside the working copy (the fak module root is fine; git finds the repo)
	BaselineRef string     // the ref the baseline + every candidate fork from (e.g. "main")
	Candidates  []int      // proposed DefaultCacheSize values, tried in order
	ProbePkg    string     // the KPI probe package path, e.g. "./cmd/kpiprobe"
	SuiteCmds   [][]string // suite-green gate; ALL must exit 0. Default: build + vet over SuitePkgs
	SuitePkgs   string     // package pattern for the default suite gate (default "./...")
	ScratchDir  string     // parent for ephemeral worktrees ("" => os.TempDir)
}

// tunableRewrite matches the single `DefaultCacheSize = <int>` literal the Proposer
// rewrites. Anchored on the const name so it can't match anything else.
var tunableRewrite = regexp.MustCompile(`(` + TunableConstName + `\s*=\s*)(-?\d+)`)

// TunableRelPath is the module-relative path of the tunable file (forward slashes).
const TunableRelPath = "internal/rsiloop/tunable.go"

const tunableRelPath = TunableRelPath

// NewWorktreeHarness wires a Harness to the real worktree/probe/suite/truth impls.
func NewWorktreeHarness(cfg WorktreeConfig) Harness {
	if cfg.ProbePkg == "" {
		cfg.ProbePkg = "./cmd/kpiprobe"
	}
	if cfg.BaselineRef == "" {
		cfg.BaselineRef = "main"
	}
	if cfg.SuitePkgs == "" {
		cfg.SuitePkgs = "./..."
	}
	if len(cfg.SuiteCmds) == 0 {
		// Windows-safe default: build + vet are a sound, native suite-green proxy
		// (`go test` binaries are blocked by OS app-control on this host — AGENTS.md).
		// A production run overrides this with the full WSL suite.
		cfg.SuiteCmds = [][]string{
			{"go", "build", cfg.SuitePkgs},
			{"go", "vet", cfg.SuitePkgs},
		}
	}

	// Pin the baseline SHA exactly ONCE, on the first resolve (BaselineMetric, which
	// the engine calls before any Measure). Every candidate then forks from this same
	// immutable SHA — so before/after are measured on the IDENTICAL tree and the
	// journal's baseline_ref is truthful for the whole run, even if `main` advances
	// mid-run on this live, auto-syncing shared trunk. Run() is sequential, so the
	// lazy pin needs no lock.
	var pinned string
	resolvePinned := func() (string, error) {
		if pinned != "" {
			return pinned, nil
		}
		sha, err := resolveRef(cfg.Repo, cfg.BaselineRef)
		if err != nil {
			return "", err
		}
		pinned = sha
		return sha, nil
	}

	return Harness{
		MetricName:      "lru_hit_rate",
		LowerBetter:     false,
		BaselineRefName: cfg.BaselineRef,
		BaselineMetric: func() (float64, string, error) {
			sha, err := resolvePinned()
			if err != nil {
				return 0, "", err
			}
			var metric float64
			err = withWorktree(cfg, sha, func(p wtPaths) error {
				m, perr := runProbe(p.module, cfg.ProbePkg)
				metric = m
				return perr
			})
			return metric, shortSHA(sha), err
		},
		Candidates: func() []Candidate {
			cs := make([]Candidate, 0, len(cfg.Candidates))
			for _, n := range cfg.Candidates {
				cs = append(cs, Candidate{Label: fmt.Sprintf("%s=%d", TunableConstName, n), Payload: n})
			}
			return cs
		},
		Measure: func(c Candidate) (Measurement, error) {
			size, ok := c.Payload.(int)
			if !ok {
				return Measurement{}, fmt.Errorf("candidate payload is %T, want int", c.Payload)
			}
			sha, err := resolvePinned() // the SAME pinned SHA the baseline forked from
			if err != nil {
				return Measurement{}, err
			}
			var meas Measurement
			err = withWorktree(cfg, sha, func(p wtPaths) error {
				if rerr := rewriteTunable(p.module, size); rerr != nil {
					return rerr
				}
				m, perr := runProbe(p.module, cfg.ProbePkg)
				if perr != nil {
					return perr
				}
				meas.Metric = m
				meas.Score = lruHitRateScorecard(size, m)
				green, detail := runSuite(p.module, cfg.SuiteCmds)
				meas.SuiteGreen = green
				if !green {
					meas.Note = detail // surface WHY the suite was red, not a silent false
				}
				// truth-clean: the ONLY change in the worktree must be the proposed
				// tunable edit. Anything else (stray artifact, unexpected diff) fails
				// closed — the local proxy for `dos verify` clean.
				meas.TruthClean = treeChangedOnly(p.root, filepath.ToSlash(filepath.Join(p.moduleRel, tunableRelPath)))
				return nil
			})
			return meas, err
		},
	}
}

func lruHitRateScorecard(size int, metric float64) *Scorecard {
	return &Scorecard{
		Name:  "lru_hit_rate",
		Value: metric,
		Grade: lruHitRateGrade(metric),
		Components: []ScoreComponent{
			{Name: "hit_rate", Value: metric, Unit: "ratio"},
			{Name: "cache_size", Value: float64(size), Unit: "entries"},
			{Name: "trace_len", Value: float64(TraceLen()), Unit: "accesses"},
			{Name: "working_set", Value: float64(workingSet), Unit: "entries"},
		},
	}
}

func lruHitRateGrade(metric float64) string {
	switch {
	case metric >= 0.8:
		return "high"
	case metric >= 0.5:
		return "medium"
	default:
		return "low"
	}
}

// wtPaths bundles the three locations a candidate run needs: the worktree root
// (where git status runs), the module dir within it (where go runs), and the
// module's path relative to the repo root (to name the expected changed file).
type wtPaths struct {
	root      string // the worktree root (a full checkout of the repo)
	module    string // root/moduleRel — where go.mod lives (e.g. .../wt/fak)
	moduleRel string // module path relative to the repo root (e.g. "fak" or ".")
}

// resolveRef returns the full SHA a ref points at.
func resolveRef(repo, ref string) (string, error) {
	out, err := exec.Command("git", "-C", repo, "rev-parse", ref).Output()
	if err != nil {
		return "", fmt.Errorf("rev-parse %s: %w", ref, err)
	}
	return strings.TrimSpace(string(out)), nil
}

func shortSHA(sha string) string {
	if len(sha) > 12 {
		return sha[:12]
	}
	return sha
}

// repoTopAndRel resolves the git repo root and the module's path relative to it.
func repoTopAndRel(repoArg string) (top, moduleRel string, err error) {
	out, rerr := exec.Command("git", "-C", repoArg, "rev-parse", "--show-toplevel").Output()
	if rerr != nil {
		return "", "", fmt.Errorf("rev-parse --show-toplevel: %w", rerr)
	}
	top = filepath.Clean(strings.TrimSpace(string(out)))
	abs, aerr := filepath.Abs(repoArg)
	if aerr != nil {
		return "", "", aerr
	}
	rel, rerr := filepath.Rel(top, abs)
	if rerr != nil {
		return "", "", rerr
	}
	if rel == "" || rel == "." {
		rel = "."
	}
	return top, filepath.ToSlash(rel), nil
}

// withWorktree creates a fresh detached worktree at ref, runs fn against it, and
// always tears it down. main is never modified.
func withWorktree(cfg WorktreeConfig, ref string, fn func(wtPaths) error) error {
	top, moduleRel, err := repoTopAndRel(cfg.Repo)
	if err != nil {
		return err
	}
	parent, err := os.MkdirTemp(cfg.ScratchDir, "rsiloop-")
	if err != nil {
		return err
	}
	defer os.RemoveAll(parent)
	wt := filepath.Join(parent, "wt")
	add := exec.Command("git", "-C", top, "worktree", "add", "--detach", wt, ref)
	windowgate.ConfigureBackgroundCommand(add)
	if out, err := add.CombinedOutput(); err != nil {
		return fmt.Errorf("worktree add %s: %v: %s", ref, err, out)
	}
	defer shipgate.RemoveWorktree(top, wt)
	module := wt
	if moduleRel != "." {
		module = filepath.Join(wt, filepath.FromSlash(moduleRel))
	}
	return fn(wtPaths{root: wt, module: module, moduleRel: moduleRel})
}

// runProbe runs the KPI probe in the module dir and parses its `KPI=<float>` line.
func runProbe(moduleDir, probePkg string) (float64, error) {
	cmd := exec.Command("go", "run", probePkg)
	windowgate.ConfigureBackgroundCommand(cmd)
	cmd.Dir = moduleDir
	out, err := cmd.CombinedOutput()
	if err != nil {
		return 0, fmt.Errorf("probe %s: %v: %s", probePkg, err, out)
	}
	return parseKPI(string(out))
}

// parseKPI extracts the float after the last `KPI=` token in the probe output.
func parseKPI(out string) (float64, error) {
	for _, line := range strings.Split(out, "\n") {
		line = strings.TrimSpace(line)
		if i := strings.Index(line, "KPI="); i >= 0 {
			return strconv.ParseFloat(strings.TrimSpace(line[i+len("KPI="):]), 64)
		}
	}
	return 0, fmt.Errorf("no KPI= line in probe output: %q", out)
}

// rewriteTunable edits the worktree's copy of tunable.go, setting DefaultCacheSize
// to size. It fails if the literal is not found (the rewrite contract is broken).
func rewriteTunable(moduleDir string, size int) error {
	path := filepath.Join(moduleDir, filepath.FromSlash(tunableRelPath))
	src, err := os.ReadFile(path)
	if err != nil {
		return err
	}
	if !tunableRewrite.Match(src) {
		return fmt.Errorf("tunable literal %q not found in %s", TunableConstName, path)
	}
	out := tunableRewrite.ReplaceAll(src, []byte("${1}"+strconv.Itoa(size)))
	return os.WriteFile(path, out, 0o644)
}

// runSuite runs every configured suite command in the module dir; green iff all
// exit 0. On a failure it returns a short diagnostic so a REVERT is explainable
// rather than a silent false: a command that could not START (binary missing,
// permission) is labeled distinctly from one that RAN and exited non-zero, and a
// tail of its combined output is included. (For the default `go build`/`go vet`
// suite, runProbe has already proven `go` is runnable; the distinction matters for a
// production override like the WSL test suite.)
func runSuite(moduleDir string, cmds [][]string) (bool, string) {
	for _, c := range cmds {
		if len(c) == 0 {
			continue
		}
		cmd := exec.Command(c[0], c[1:]...)
		windowgate.ConfigureBackgroundCommand(cmd)
		cmd.Dir = moduleDir
		out, err := cmd.CombinedOutput()
		if err == nil {
			continue
		}
		label := "suite cmd " + strings.Join(c, " ")
		var exitErr *exec.ExitError
		if errors.As(err, &exitErr) {
			label += " exited non-zero"
		} else {
			label += " could not start (" + err.Error() + ")"
		}
		return false, label + ": " + tail(string(out), 400)
	}
	return true, ""
}

// tail returns the last n bytes of s (whole string if shorter), for a compact
// diagnostic in the journal Note.
func tail(s string, n int) string {
	s = strings.TrimSpace(s)
	if len(s) <= n {
		return s
	}
	return "…" + s[len(s)-n:]
}

// treeChangedOnly reports whether the worktree's only modified path is `only`
// (a repo-root-relative, forward-slash path). A clean tree (no changes) also
// returns false — a candidate that changed nothing is not a real proposal. Build
// artifacts or unexpected edits fail closed.
func treeChangedOnly(wtRoot, only string) bool {
	out, err := exec.Command("git", "-C", wtRoot, "status", "--porcelain").Output()
	if err != nil {
		return false
	}
	changed := []string{}
	for _, line := range strings.Split(strings.TrimRight(string(out), "\n"), "\n") {
		if strings.TrimSpace(line) == "" {
			continue
		}
		// porcelain: "XY <path>"; take the path after the 3-char status prefix.
		if len(line) > 3 {
			changed = append(changed, filepath.ToSlash(strings.TrimSpace(line[3:])))
		}
	}
	return len(changed) == 1 && changed[0] == only
}
