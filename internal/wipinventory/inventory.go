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
	"time"

	"github.com/anthony-chaudhary/fak/internal/workerworktree"
)

const Schema = "fak-wip-inventory/1"
const sampleLimit = 20

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
	Count   int      `json:"count"`
	Known   bool     `json:"known"`
	Samples []string `json:"samples,omitempty"`
	Error   string   `json:"error,omitempty"`
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
	Ref     string `json:"ref"`
	SHA     string `json:"sha"`
	Unix    int64  `json:"unix"`
	Changed int    `json:"changed_paths"`
	Added   int    `json:"added_paths"`
	Known   bool   `json:"known"`
	Error   string `json:"error,omitempty"`
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
	rep.Main = checkout(root, rep.HEAD, "main", r, &rep)
	rep.Ignored = populationZ(root, r, "ls-files", "--others", "--ignored", "--exclude-standard", "-z")
	recordPopulationError(&rep, "ignored_generated", &rep.Ignored)
	workerRoot := workerworktree.DefaultRoot()
	if len(opts) > 0 && opts[0].WorkerRoot != "" {
		workerRoot = opts[0].WorkerRoot
	}
	rep.Worktrees, rep.StaleWorkers = worktrees(root, workerRoot, r, &rep)
	rep.Checkpoints, rep.CheckpointsKnown = checkpoints(root, r, &rep)
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
func checkout(path, head, branch string, r Runner, rep *Report) Checkout {
	c := Checkout{Path: filepath.ToSlash(path), HEAD: head, Branch: branch, Tracked: Population{Known: true}, Untracked: Population{Known: true}}
	out, err := r.Run(path, "status", "--porcelain=v1", "-z", "--untracked-files=all")
	if err != nil {
		c.Tracked = unknownPopulation(err)
		c.Untracked = unknownPopulation(err)
		rep.Errors = append(rep.Errors, "checkout "+c.Path+": "+err.Error())
		return c
	}
	for _, raw := range bytes.Split(out, []byte{0}) {
		if len(raw) < 4 {
			continue
		}
		line := string(raw)
		name := line[3:]
		if strings.HasPrefix(line, "?? ") {
			addSample(&c.Untracked, name)
		} else {
			addSample(&c.Tracked, name)
		}
	}
	return c
}
func addSample(p *Population, path string) {
	p.Count++
	if len(p.Samples) < sampleLimit {
		p.Samples = append(p.Samples, filepath.ToSlash(path))
	}
}

func worktrees(root, workerRoot string, r Runner, rep *Report) ([]Checkout, []StaleWorker) {
	out, err := r.Run(root, "worktree", "list", "--porcelain")
	if err != nil {
		rep.Errors = append(rep.Errors, "registered_worker_worktrees: "+err.Error())
		return nil, []StaleWorker{{Kind: "unknown", Detail: err.Error()}}
	}
	registered := map[string]bool{}
	var live []Checkout
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
		if path == "" || samePath(path, root) || !workerworktree.IsWorkerWorktree(path) {
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
		live = append(live, checkout(path, head, branch, r, rep))
	}
	entries, readErr := os.ReadDir(workerRoot)
	if readErr != nil && !os.IsNotExist(readErr) {
		rep.Errors = append(rep.Errors, "stale_worker_residue: "+readErr.Error())
		stale = append(stale, StaleWorker{Path: filepath.ToSlash(workerRoot), Kind: "unknown", Detail: readErr.Error()})
	}
	for _, e := range entries {
		if !e.IsDir() || !workerworktree.IsWorkerWorktree(e.Name()) {
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
