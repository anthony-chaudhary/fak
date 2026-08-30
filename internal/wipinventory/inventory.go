package wipinventory

import (
	"bytes"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"sort"
	"strconv"
	"strings"
	"sync"
	"time"
)

const Schema = "fak-wip-inventory/1"
const sampleLimit = 20
const checkoutWorkerLimit = 16

const (
	workerRootEnv = "FLEET_WORKER_WORKTREE_ROOT"
	workerMarker  = "fak-worker-wt"
)

func defaultWorkerRoot() string {
	if override := strings.TrimSpace(os.Getenv(workerRootEnv)); override != "" {
		return override
	}
	base := os.Getenv("LOCALAPPDATA")
	if base == "" {
		base = os.TempDir()
	}
	return filepath.Join(base, "Fleet", "worker-worktrees")
}

func isWorkerWorktree(path string) bool {
	name := filepath.Base(filepath.Clean(path))
	return name == workerMarker || strings.HasPrefix(name, workerMarker+"-")
}

type Runner interface {
	Run(dir string, args ...string) ([]byte, error)
}
type GitRunner struct{}

func (GitRunner) Run(dir string, args ...string) ([]byte, error) {
	cmd := exec.Command("git", args...)
	cmd.Dir = dir
	configureDispatchHelperCommand(cmd)
	var stderr bytes.Buffer
	cmd.Stderr = &stderr
	out, err := cmd.Output()
	if err != nil {
		return nil, fmt.Errorf("git %s: %s", strings.Join(args, " "), strings.TrimSpace(stderr.String()))
	}
	return out, nil
}

type Population struct {
	Count                       int      `json:"count"`
	Known                       bool     `json:"known"`
	Samples                     []string `json:"samples,omitempty"`
	Error                       string   `json:"error,omitempty"`
	OldestPath                  string   `json:"oldest_path,omitempty"`
	OldestAgeSeconds            int64    `json:"oldest_age_seconds,omitempty"`
	OldestUnprotectedPath       string   `json:"oldest_unprotected_path,omitempty"`
	OldestUnprotectedAgeSeconds int64    `json:"oldest_unprotected_age_seconds,omitempty"`
	Protection                  string   `json:"protection,omitempty"`
	paths                       []agedPath
}

type agedPath struct {
	path       string
	ageSeconds int64
}
type Checkout struct {
	Path      string     `json:"path"`
	HEAD      string     `json:"head,omitempty"`
	Branch    string     `json:"branch,omitempty"`
	Tracked   Population `json:"tracked_changes"`
	Untracked Population `json:"untracked"`
}
type StaleWorker struct {
	Path   string `json:"path"`
	Kind   string `json:"kind"`
	Detail string `json:"detail,omitempty"`
}
type Checkpoint struct {
	Ref      string   `json:"ref"`
	SHA      string   `json:"sha"`
	Unix     int64    `json:"unix"`
	Changed  int      `json:"changed_paths"`
	Added    int      `json:"added_paths"`
	Known    bool     `json:"known"`
	Error    string   `json:"error,omitempty"`
	Paths    []string `json:"paths,omitempty"`
	allPaths []string
}
type IgnoreInputs struct {
	GitignoreHash string `json:"gitignore_hash,omitempty"`
	ExcludeHash   string `json:"exclude_hash,omitempty"`
	GlobalExclude string `json:"global_exclude,omitempty"`
	Sparse        bool   `json:"sparse_checkout"`
	HiddenIndex   int    `json:"skip_or_assume_unchanged"`
	Known         bool   `json:"known"`
	Error         string `json:"error,omitempty"`
}
type Report struct {
	Schema           string        `json:"schema"`
	ObservedAt       time.Time     `json:"observed_at"`
	Repository       string        `json:"repository"`
	HEAD             string        `json:"head,omitempty"`
	Main             Checkout      `json:"main"`
	Ignored          Population    `json:"ignored_generated"`
	Worktrees        []Checkout    `json:"registered_worker_worktrees"`
	StaleWorkers     []StaleWorker `json:"stale_worker_residue"`
	Checkpoints      []Checkpoint  `json:"wip_checkpoints"`
	CheckpointsKnown bool          `json:"wip_checkpoints_known"`
	IgnoreInputs     IgnoreInputs  `json:"ignore_visibility"`
	Errors           []string      `json:"errors,omitempty"`
}
type Options struct{ WorkerRoot string }

func Collect(root string, now time.Time, r Runner, opts ...Options) Report {
	abs, err := filepath.Abs(root)
	if err == nil {
		root = abs
	}
	rep := Report{Schema: Schema, ObservedAt: now.UTC(), Repository: filepath.ToSlash(root)}
	rep.HEAD = one(root, r, &rep, "rev-parse", "HEAD")
	rep.Main, err = checkout(root, rep.HEAD, "main", now, r)
	if err != nil {
		rep.Errors = append(rep.Errors, "checkout "+rep.Main.Path+": "+err.Error())
	}
	rep.Ignored = populationZ(root, r, "ls-files", "--others", "--ignored", "--exclude-standard", "-z")
	recordPopulationError(&rep, "ignored_generated", &rep.Ignored)
	workerRoot := defaultWorkerRoot()
	if len(opts) > 0 && opts[0].WorkerRoot != "" {
		workerRoot = opts[0].WorkerRoot
	}
	rep.Worktrees, rep.StaleWorkers = worktrees(root, workerRoot, r, &rep)
	rep.Checkpoints, rep.CheckpointsKnown = checkpoints(root, r, &rep)
	labelProtection(&rep)
	rep.IgnoreInputs = ignoreInputs(root, r)
	if !rep.IgnoreInputs.Known {
		rep.Errors = append(rep.Errors, "ignore_visibility: "+rep.IgnoreInputs.Error)
	}
	sort.Strings(rep.Errors)
	return rep
}
func one(root string, r Runner, rep *Report, args ...string) string {
	out, err := r.Run(root, args...)
	if err != nil {
		rep.Errors = append(rep.Errors, err.Error())
		return ""
	}
	return strings.TrimSpace(string(out))
}
func unknownPopulation(err error) Population { return Population{Error: err.Error()} }
func populationZ(root string, r Runner, args ...string) Population {
	out, err := r.Run(root, args...)
	if err != nil {
		return unknownPopulation(err)
	}
	p := Population{Known: true}
	for _, raw := range bytes.Split(out, []byte{0}) {
		if len(raw) > 0 {
			addSample(&p, string(raw))
		}
	}
	return p
}
func recordPopulationError(rep *Report, name string, p *Population) {
	if !p.Known {
		rep.Errors = append(rep.Errors, name+": "+p.Error)
	}
}
func checkout(path, head, branch string, now time.Time, r Runner) (Checkout, error) {
	c := Checkout{Path: filepath.ToSlash(path), HEAD: head, Branch: branch, Tracked: Population{Known: true}, Untracked: Population{Known: true}}
	out, err := r.Run(path, "status", "--porcelain=v1", "-z", "--untracked-files=all")
	if err != nil {
		c.Tracked = unknownPopulation(err)
		c.Untracked = unknownPopulation(err)
		return c, err
	}
	for _, raw := range bytes.Split(out, []byte{0}) {
		if len(raw) < 4 {
			continue
		}
		line := string(raw)
		name := line[3:]
		if strings.HasPrefix(line, "?? ") {
			addSample(&c.Untracked, name)
			observeAge(&c.Untracked, path, name, now)
		} else {
			addSample(&c.Tracked, name)
		}
	}
	return c, nil
}
func addSample(p *Population, path string) {
	p.Count++
	if len(p.Samples) < sampleLimit {
		p.Samples = append(p.Samples, filepath.ToSlash(path))
	}
}

func observeAge(p *Population, root, name string, now time.Time) {
	info, err := os.Stat(filepath.Join(root, filepath.FromSlash(name)))
	if err != nil {
		p.Known = false
		if p.Error == "" {
			p.Error = err.Error()
		}
		return
	}
	age := max(int64(now.Sub(info.ModTime()).Seconds()), 0)
	path := filepath.ToSlash(name)
	p.paths = append(p.paths, agedPath{path: path, ageSeconds: age})
	if p.OldestPath == "" || age > p.OldestAgeSeconds {
		p.OldestPath = path
		p.OldestAgeSeconds = age
	}
}

func worktrees(root, workerRoot string, r Runner, rep *Report) ([]Checkout, []StaleWorker) {
	out, err := r.Run(root, "worktree", "list", "--porcelain")
	if err != nil {
		rep.Errors = append(rep.Errors, "registered_worker_worktrees: "+err.Error())
		return nil, []StaleWorker{{Kind: "unknown", Detail: err.Error()}}
	}
	registered := map[string]bool{}
	var pending []checkoutSpec
	var stale []StaleWorker
	for _, block := range strings.Split(strings.TrimSpace(string(out)), "\n\n") {
		path, head, branch, prunable := "", "", "detached", ""
		for _, line := range strings.Split(block, "\n") {
			switch {
			case strings.HasPrefix(line, "worktree "):
				path = strings.TrimPrefix(line, "worktree ")
			case strings.HasPrefix(line, "HEAD "):
				head = strings.TrimPrefix(line, "HEAD ")
			case strings.HasPrefix(line, "branch "):
				branch = strings.TrimPrefix(line, "branch refs/heads/")
			case strings.HasPrefix(line, "prunable"):
				prunable = strings.TrimSpace(strings.TrimPrefix(line, "prunable"))
			}
		}
		if path == "" || samePath(path, root) || !isWorkerWorktree(path) {
			continue
		}
		registered[pathKey(path)] = true
		if prunable != "" {
			stale = append(stale, StaleWorker{Path: filepath.ToSlash(path), Kind: "registered-prunable", Detail: prunable})
			continue
		}
		if _, statErr := os.Stat(path); statErr != nil {
			stale = append(stale, StaleWorker{Path: filepath.ToSlash(path), Kind: "registered-missing", Detail: statErr.Error()})
			continue
		}
		pending = append(pending, checkoutSpec{path: path, head: head, branch: branch})
	}
	live, checkoutErrors := probeCheckouts(pending, rep.ObservedAt, r)
	rep.Errors = append(rep.Errors, checkoutErrors...)
	entries, readErr := os.ReadDir(workerRoot)
	if readErr != nil && !os.IsNotExist(readErr) {
		rep.Errors = append(rep.Errors, "stale_worker_residue: "+readErr.Error())
		stale = append(stale, StaleWorker{Path: filepath.ToSlash(workerRoot), Kind: "unknown", Detail: readErr.Error()})
	}
	for _, e := range entries {
		if !e.IsDir() || !isWorkerWorktree(e.Name()) {
			continue
		}
		path := filepath.Join(workerRoot, e.Name())
		if !registered[pathKey(path)] {
			stale = append(stale, StaleWorker{Path: filepath.ToSlash(path), Kind: "unregistered-directory"})
		}
	}
	sort.Slice(live, func(i, j int) bool { return live[i].Path < live[j].Path })
	sort.Slice(stale, func(i, j int) bool {
		if stale[i].Path == stale[j].Path {
			return stale[i].Kind < stale[j].Kind
		}
		return stale[i].Path < stale[j].Path
	})
	return live, stale
}

type checkoutSpec struct {
	path   string
	head   string
	branch string
}

type checkoutResult struct {
	checkout Checkout
	err      error
}

func probeCheckouts(specs []checkoutSpec, now time.Time, r Runner) ([]Checkout, []string) {
	if len(specs) == 0 {
		return nil, nil
	}

	// Git status can be slow on large worktree fleets. A fixed ceiling bounds the
	// process and filesystem pressure independently of the number of registrations.
	workerCount := min(checkoutWorkerLimit, len(specs))
	jobs := make(chan int)
	results := make([]checkoutResult, len(specs))
	var wg sync.WaitGroup
	for range workerCount {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for index := range jobs {
				spec := specs[index]
				c, err := checkout(spec.path, spec.head, spec.branch, now, r)
				results[index] = checkoutResult{checkout: c, err: err}
			}
		}()
	}
	for index := range specs {
		jobs <- index
	}
	close(jobs)
	wg.Wait()

	checkouts := make([]Checkout, 0, len(results))
	var errors []string
	for _, result := range results {
		checkouts = append(checkouts, result.checkout)
		if result.err != nil {
			errors = append(errors, "checkout "+result.checkout.Path+": "+result.err.Error())
		}
	}
	return checkouts, errors
}
func pathKey(path string) string {
	abs, _ := filepath.Abs(path)
	return strings.ToLower(filepath.Clean(abs))
}
func samePath(a, b string) bool { return pathKey(a) == pathKey(b) }
func checkpoints(root string, r Runner, rep *Report) ([]Checkpoint, bool) {
	out, err := r.Run(root, "for-each-ref", "--format=%(refname)%00%(objectname)%00%(creatordate:unix)", "refs/fak/wip")
	if err != nil {
		rep.Errors = append(rep.Errors, "wip_checkpoints: "+err.Error())
		return nil, false
	}
	var result []Checkpoint
	for _, line := range strings.Split(strings.TrimSpace(string(out)), "\n") {
		if line == "" {
			continue
		}
		parts := strings.Split(line, "\x00")
		if len(parts) != 3 {
			rep.Errors = append(rep.Errors, "wip_checkpoints: malformed for-each-ref row")
			continue
		}
		unix, _ := strconv.ParseInt(parts[2], 10, 64)
		cp := Checkpoint{Ref: parts[0], SHA: parts[1], Unix: unix, Known: true}
		diff, derr := r.Run(root, "diff-tree", "--root", "--no-commit-id", "--name-status", "-r", parts[1])
		if derr != nil {
			cp.Known = false
			cp.Error = derr.Error()
			rep.Errors = append(rep.Errors, "checkpoint "+cp.Ref+": "+derr.Error())
		} else {
			for _, row := range strings.Split(strings.TrimSpace(string(diff)), "\n") {
				if row == "" {
					continue
				}
				cp.Changed++
				parts := strings.SplitN(row, "\t", 2)
				if len(parts) == 2 {
					path := filepath.ToSlash(parts[1])
					cp.allPaths = append(cp.allPaths, path)
					if len(cp.Paths) < sampleLimit {
						cp.Paths = append(cp.Paths, path)
					}
				}
				if strings.HasPrefix(row, "A\t") {
					cp.Added++
				}
			}
		}
		result = append(result, cp)
	}
	sort.Slice(result, func(i, j int) bool { return result[i].Ref < result[j].Ref })
	return result, true
}
func labelProtection(rep *Report) {
	for i := range rep.Worktrees {
		rep.Worktrees[i].Tracked.Protection = "worker-worktree"
		rep.Worktrees[i].Untracked.Protection = "worker-worktree"
	}
	if rep.Main.Untracked.Count == 0 {
		return
	}
	protected := make(map[string]string)
	for _, cp := range rep.Checkpoints {
		for _, path := range cp.allPaths {
			if _, exists := protected[path]; !exists {
				protected[path] = "checkpoint:" + cp.Ref
			}
		}
	}
	protectedCount := 0
	var oneProtection string
	for _, source := range rep.Main.Untracked.paths {
		if protection := protected[source.path]; protection != "" {
			protectedCount++
			oneProtection = protection
			continue
		}
		if source.ageSeconds > rep.Main.Untracked.OldestUnprotectedAgeSeconds || rep.Main.Untracked.OldestUnprotectedPath == "" {
			rep.Main.Untracked.OldestUnprotectedPath = source.path
			rep.Main.Untracked.OldestUnprotectedAgeSeconds = source.ageSeconds
		}
	}
	switch {
	case protectedCount == 0:
		rep.Main.Untracked.Protection = "unprotected"
	case protectedCount == rep.Main.Untracked.Count:
		rep.Main.Untracked.Protection = oneProtection
	default:
		rep.Main.Untracked.Protection = "mixed"
	}
}

func ignoreInputs(root string, r Runner) IgnoreInputs {
	in := IgnoreInputs{Known: true}
	in.GitignoreHash = hashFile(filepath.Join(root, ".gitignore"))
	in.ExcludeHash = hashFile(gitPath(root, "info/exclude", r))
	if out, err := r.Run(root, "config", "--get", "core.excludesfile"); err == nil {
		in.GlobalExclude = strings.TrimSpace(string(out))
	}
	if out, err := r.Run(root, "config", "--bool", "--get", "core.sparseCheckout"); err == nil {
		in.Sparse = strings.TrimSpace(string(out)) == "true"
	}
	out, err := r.Run(root, "ls-files", "-v")
	if err != nil {
		in.Known = false
		in.Error = err.Error()
	} else {
		for _, line := range strings.Split(string(out), "\n") {
			if strings.HasPrefix(line, "S ") || strings.HasPrefix(line, "h ") {
				in.HiddenIndex++
			}
		}
	}
	return in
}
func gitPath(root, name string, r Runner) string {
	out, err := r.Run(root, "rev-parse", "--git-path", name)
	if err != nil {
		return ""
	}
	path := strings.TrimSpace(string(out))
	if !filepath.IsAbs(path) {
		path = filepath.Join(root, path)
	}
	return path
}
func hashFile(path string) string {
	if path == "" {
		return ""
	}
	b, err := os.ReadFile(path)
	if err != nil {
		return ""
	}
	sum := sha256.Sum256(b)
	return hex.EncodeToString(sum[:])
}
func (r Report) JSON() ([]byte, error) { return json.MarshalIndent(r, "", "  ") }
