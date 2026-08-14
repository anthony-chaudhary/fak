package armbench

import (
	"bytes"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"sort"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/windowgate"
)

const (
	PonytailRevision        = "2ed6c52c9d7e5e56942508591085fd45dea277d3"
	PonytailCavemanRevision = "c72984e4392c7a154e55c11dbf445f01ce5c35d4"
)

var ponytailFiles = map[string]string{
	"benchmarks/agentic/README.md":   "1df410888cf4785e1180d12ca78c3ea419671990e1aeded14d329a4a655f5b3b",
	"benchmarks/agentic/complete.py": "2730deb3fd49439c38a8e8ba66c80f18d6b8ea33902247e393fae177f24352cf",
	"benchmarks/agentic/judge.py":    "c845548290c062dac5bef93ba3231e26b35be62a07fdebabec61833c4a20b6c0",
	"benchmarks/agentic/run.py":      "88c105299b38850c6d944a11faac7b0e18de2f5c9a3e5f4ae1f6b641d6853811",
	"benchmarks/agentic/tasks.py":    "68f473557695f69a036cd6fdb5d8e9ec51aef407183416545b29823c1e3190a2",
}

var PonytailTasks = []string{
	"todo-null", "safe-path", "critic-email", "rate-limit", "sql-user", "auth-token", "csv-sum", "cache",
	"reuse-slug", "reuse-money", "trace-transfer", "trace-amount", "open-dataclass", "open-decorators",
	"open-mandelbrot", "vibe-todo", "vibe-password", "vibe-shortener", "vibe-md2html", "vibe-csvstats",
	"vibe-langgraph", "vibe-restapi", "vibe-scraper", "vibe-logparse", "vibe-rename", "vibe-adventure",
	"vibe-jsonconf", "tmpl-fe-datepicker", "tmpl-fe-colorpicker", "tmpl-fe-command", "tmpl-fe-dropzone",
	"tmpl-fe-wizard", "tmpl-fe-rating", "tmpl-be-duplicate", "tmpl-be-search", "tmpl-be-count",
	"tmpl-be-archive", "tmpl-be-bulkdelete", "tmpl-be-csv",
}
var PonytailArms = []string{"baseline", "caveman", "ponytail"}

// PonytailPacket is both the no-spend launch packet and the live run receipt.
type PonytailPacket struct {
	Schema         string              `json:"schema"`
	Mode           string              `json:"mode"`
	Comparator     string              `json:"comparator"`
	Revision       string              `json:"revision"`
	Caveman        string              `json:"caveman,omitempty"`
	Checkout       string              `json:"checkout"`
	Files          map[string]string   `json:"files_sha256"`
	Tasks          []string            `json:"tasks"`
	Arms           []string            `json:"arms"`
	Models         map[string]string   `json:"models"`
	AgentModel     string              `json:"agent_model"`
	JudgeModel     string              `json:"judge_model"`
	TimeoutSeconds int                 `json:"timeout_seconds"`
	Trials         int                 `json:"trials"`
	Workers        int                 `json:"workers"`
	Account        string              `json:"account_identity,omitempty"`
	Counterbalance [][]string          `json:"counterbalanced_orders"`
	Commands       []string            `json:"commands"`
	StartedAt      string              `json:"started_at,omitempty"`
	FinishedAt     string              `json:"finished_at,omitempty"`
	Runs           []PonytailRun       `json:"runs,omitempty"`
	Report         []PonytailArmReport `json:"report_task_success_first,omitempty"`
}

type PonytailRun struct {
	Task       string   `json:"task"`
	Order      []string `json:"order"`
	OutputDir  string   `json:"output_dir"`
	ExitCode   int      `json:"exit_code"`
	DurationMS int64    `json:"duration_ms"`
	Stdout     string   `json:"stdout"`
	Stderr     string   `json:"stderr"`
}

type PonytailArmReport struct {
	Arm       string  `json:"arm"`
	Successes int     `json:"task_successes"`
	Safe      int     `json:"safe"`
	Cells     int     `json:"cells"`
	Tokens    int64   `json:"tokens"`
	CostUSD   float64 `json:"cost_usd"`
	LatencyMS int64   `json:"latency_ms"`
	Denials   int     `json:"permission_denials"`
	Failures  int     `json:"failures"`
	Retries   int     `json:"retries"`
}

type PonytailOptions struct {
	Checkout, Caveman, Out, Account, Python, Model string
	Trials                                         int
	Live                                           bool
}

func Ponytail(opts PonytailOptions) (PonytailPacket, error) {
	if opts.Checkout == "" {
		return PonytailPacket{}, errors.New("--checkout is required")
	}
	if opts.Python == "" {
		opts.Python = "python"
	}
	if opts.Model == "" {
		opts.Model = "haiku"
	}
	models := map[string]string{"haiku": "claude-haiku-4-5-20251001", "sonnet": "claude-sonnet-4-6", "opus": "claude-opus-4-8"}
	if _, ok := models[opts.Model]; !ok {
		return PonytailPacket{}, fmt.Errorf("unsupported model alias %q", opts.Model)
	}
	if opts.Trials == 0 {
		opts.Trials = 1
	}
	if opts.Trials < 1 {
		return PonytailPacket{}, errors.New("--trials must be positive")
	}
	checkout, err := filepath.Abs(opts.Checkout)
	if err != nil {
		return PonytailPacket{}, err
	}
	hashes, err := verifyPonytail(checkout)
	if err != nil {
		return PonytailPacket{}, err
	}
	orders := [][]string{{"baseline", "caveman", "ponytail"}, {"caveman", "ponytail", "baseline"}, {"ponytail", "baseline", "caveman"}}
	p := PonytailPacket{
		Schema: "fak-armbench-ponytail/1", Mode: "dry-run", Comparator: "DietrichGebert/ponytail@" + PonytailRevision,
		Revision: PonytailRevision, Checkout: checkout, Files: hashes, Tasks: append([]string(nil), PonytailTasks...),
		Arms: append([]string(nil), PonytailArms...), Models: models, AgentModel: opts.Model,
		JudgeModel: "claude-sonnet-4-6 (temperature 0; pinned judge.py and complete.py)", TimeoutSeconds: 300,
		Trials: opts.Trials, Workers: 1, Counterbalance: orders,
	}
	for i, task := range PonytailTasks {
		p.Commands = append(p.Commands, fmt.Sprintf("%s benchmarks/agentic/run.py --task %s --arms %s --model %s --runs %d --workers 1", opts.Python, task, strings.Join(orders[i%3], ","), opts.Model, opts.Trials))
	}
	if err := runSelftests(opts.Python, checkout); err != nil {
		return p, err
	}
	if !opts.Live {
		return p, nil
	}
	if opts.Caveman == "" {
		return p, errors.New("live mode requires --caveman pinned checkout")
	}
	caveman, err := filepath.Abs(opts.Caveman)
	if err != nil {
		return p, err
	}
	if err := verifyRevision(caveman, PonytailCavemanRevision); err != nil {
		return p, fmt.Errorf("caveman: %w", err)
	}
	if opts.Account == "" {
		return p, errors.New("live mode requires --account identity")
	}
	if opts.Out == "" {
		return p, errors.New("live mode requires --out")
	}
	accountDir, err := resolvePonytailAccount(opts.Account)
	if err != nil {
		return p, err
	}
	out, err := filepath.Abs(opts.Out)
	if err != nil {
		return p, err
	}
	if err := os.MkdirAll(out, 0o755); err != nil {
		return p, err
	}
	p.Mode, p.Account, p.Caveman, p.StartedAt = "live", opts.Account, "JuliusBrussee/caveman@"+PonytailCavemanRevision, time.Now().UTC().Format(time.RFC3339Nano)
	for i, task := range PonytailTasks {
		order := orders[i%3]
		args := []string{"benchmarks/agentic/run.py", "--task", task, "--arms", strings.Join(order, ","), "--model", opts.Model, "--runs", fmt.Sprint(opts.Trials), "--workers", "1"}
		before, err := runDirs(checkout)
		if err != nil {
			return p, err
		}
		start := time.Now()
		cmd := exec.Command(opts.Python, args...)
		windowgate.ConfigureBackgroundCommand(cmd)
		cmd.Dir = checkout
		cmd.Env = replaceEnv(replaceEnv(replaceEnv(os.Environ(), "CLAUDE_CONFIG_DIR", accountDir), "PONYTAIL_PLUGIN_DIR", checkout), "CAVEMAN_PLUGIN_DIR", caveman)
		var stdout, stderr bytes.Buffer
		cmd.Stdout, cmd.Stderr = io.MultiWriter(os.Stdout, &stdout), io.MultiWriter(os.Stderr, &stderr)
		runErr := cmd.Run()
		code := 0
		if runErr != nil {
			code = 1
			var ee *exec.ExitError
			if errors.As(runErr, &ee) {
				code = ee.ExitCode()
			}
		}
		after, err := runDirs(checkout)
		if err != nil {
			return p, err
		}
		created := newestDifference(before, after)
		dst := filepath.Join(out, fmt.Sprintf("%02d-%s", i+1, task))
		if created != "" {
			if err := copyTree(created, dst); err != nil {
				return p, err
			}
		}
		p.Runs = append(p.Runs, PonytailRun{Task: task, Order: append([]string(nil), order...), OutputDir: dst, ExitCode: code, DurationMS: time.Since(start).Milliseconds(), Stdout: stdout.String(), Stderr: stderr.String()})
		if runErr != nil {
			p.FinishedAt = time.Now().UTC().Format(time.RFC3339Nano)
			_ = writePonytailPacket(out, p)
			return p, fmt.Errorf("task %s live run failed: %w", task, runErr)
		}
	}
	p.FinishedAt = time.Now().UTC().Format(time.RFC3339Nano)
	p.Report, err = SummarizePonytailEvidence(out)
	if err != nil {
		return p, err
	}
	if err := writePonytailPacket(out, p); err != nil {
		return p, err
	}
	return p, nil
}

func verifyPonytail(checkout string) (map[string]string, error) {
	if err := verifyRevision(checkout, PonytailRevision); err != nil {
		return nil, err
	}
	got := map[string]string{}
	for rel, want := range ponytailFiles {
		b, err := os.ReadFile(filepath.Join(checkout, filepath.FromSlash(rel)))
		if err != nil {
			return nil, err
		}
		s := sha256.Sum256(b)
		h := hex.EncodeToString(s[:])
		if h != want {
			return nil, fmt.Errorf("%s sha256 %s, want %s", rel, h, want)
		}
		got[rel] = h
	}
	return got, nil
}
func verifyRevision(checkout, want string) error {
	cmd := exec.Command("git", "rev-parse", "HEAD")
	windowgate.ConfigureBackgroundCommand(cmd)
	cmd.Dir = checkout
	b, err := cmd.Output()
	if err != nil {
		return fmt.Errorf("read revision: %w", err)
	}
	if got := strings.TrimSpace(string(b)); got != want {
		return fmt.Errorf("revision %s, want %s", got, want)
	}
	return nil
}
func runSelftests(python, checkout string) error {
	cmd := exec.Command(python, "benchmarks/agentic/run.py", "--selftest")
	windowgate.ConfigureBackgroundCommand(cmd)
	cmd.Dir = checkout
	out, err := cmd.CombinedOutput()
	if err != nil {
		return fmt.Errorf("pinned upstream selftest: %w: %s", err, strings.TrimSpace(string(out)))
	}
	complete := exec.Command(python, "benchmarks/agentic/complete.py", "--selftest-offline")
	windowgate.ConfigureBackgroundCommand(complete)
	complete.Dir = checkout
	if out, err := complete.CombinedOutput(); err != nil {
		return fmt.Errorf("pinned completeness offline selftest: %w: %s", err, strings.TrimSpace(string(out)))
	}
	return nil
}
func resolvePonytailAccount(account string) (string, error) {
	exe, err := os.Executable()
	if err != nil {
		return "", err
	}
	cmd := exec.Command(exe, "fleet-accounts", "resolve", "--account", account, "--product", "claude", "--json")
	windowgate.ConfigureBackgroundCommand(cmd)
	b, err := cmd.Output()
	if err != nil {
		return "", fmt.Errorf("resolve account identity %q: %w", account, err)
	}
	var r struct {
		OK        bool   `json:"ok"`
		ConfigDir string `json:"config_dir"`
		CanServe  bool   `json:"can_serve"`
		Reason    string `json:"reason"`
	}
	if err := json.Unmarshal(b, &r); err != nil {
		return "", err
	}
	if !r.OK || !r.CanServe || r.ConfigDir == "" {
		return "", fmt.Errorf("account identity %q cannot serve: %s", account, r.Reason)
	}
	return r.ConfigDir, nil
}
func replaceEnv(env []string, key, val string) []string {
	p := strings.ToUpper(key) + "="
	out := make([]string, 0, len(env)+1)
	for _, v := range env {
		if !strings.HasPrefix(strings.ToUpper(v), p) {
			out = append(out, v)
		}
	}
	return append(out, key+"="+val)
}
func runDirs(checkout string) (map[string]bool, error) {
	root := filepath.Join(checkout, "benchmarks", "agentic", "runs")
	es, err := os.ReadDir(root)
	if os.IsNotExist(err) {
		return map[string]bool{}, nil
	}
	if err != nil {
		return nil, err
	}
	m := map[string]bool{}
	for _, e := range es {
		if e.IsDir() {
			m[filepath.Join(root, e.Name())] = true
		}
	}
	return m, nil
}
func newestDifference(before, after map[string]bool) string {
	var a []string
	for p := range after {
		if !before[p] {
			a = append(a, p)
		}
	}
	sort.Strings(a)
	if len(a) == 0 {
		return ""
	}
	return a[len(a)-1]
}
func copyTree(src, dst string) error {
	return filepath.Walk(src, func(path string, info os.FileInfo, err error) error {
		if err != nil {
			return err
		}
		rel, e := filepath.Rel(src, path)
		if e != nil {
			return e
		}
		target := filepath.Join(dst, rel)
		if info.IsDir() {
			return os.MkdirAll(target, info.Mode())
		}
		b, e := os.ReadFile(path)
		if e != nil {
			return e
		}
		return os.WriteFile(target, b, info.Mode())
	})
}
func writePonytailPacket(out string, p PonytailPacket) error {
	b, err := json.MarshalIndent(p, "", "  ")
	if err != nil {
		return err
	}
	return os.WriteFile(filepath.Join(out, "run-packet.json"), append(b, '\n'), 0o644)
}

// SummarizePonytailEvidence folds unchanged upstream results.json files. Upstream has no cell retry loop,
// so retries are explicitly zero; process errors and missing correctness become failures.
func SummarizePonytailEvidence(root string) ([]PonytailArmReport, error) {
	by := map[string]*PonytailArmReport{}
	for _, arm := range PonytailArms {
		by[arm] = &PonytailArmReport{Arm: arm}
	}
	err := filepath.Walk(root, func(path string, info os.FileInfo, err error) error {
		if err != nil {
			return err
		}
		if info.IsDir() || info.Name() != "results.json" {
			return nil
		}
		b, e := os.ReadFile(path)
		if e != nil {
			return e
		}
		var doc struct {
			Results []struct {
				Arm      string  `json:"arm"`
				Correct  int     `json:"correct"`
				Safe     int     `json:"safe"`
				Cost     float64 `json:"cost"`
				Duration int64   `json:"duration_ms"`
				Denials  int     `json:"denials"`
				In       int64   `json:"in_tokens"`
				Out      int64   `json:"out_tokens"`
				Cache    int64   `json:"cache_tokens"`
				Error    string  `json:"error"`
			} `json:"results"`
		}
		if e = json.Unmarshal(b, &doc); e != nil {
			return fmt.Errorf("%s: %w", path, e)
		}
		for _, x := range doc.Results {
			r := by[x.Arm]
			if r == nil {
				continue
			}
			r.Cells++
			r.Successes += x.Correct
			r.Safe += x.Safe
			r.CostUSD += x.Cost
			r.LatencyMS += x.Duration
			r.Denials += x.Denials
			r.Tokens += x.In + x.Out + x.Cache
			if x.Error != "" || x.Correct == 0 {
				r.Failures++
			}
		}
		return nil
	})
	if err != nil {
		return nil, err
	}
	out := make([]PonytailArmReport, 0, 3)
	for _, arm := range PonytailArms {
		out = append(out, *by[arm])
	}
	return out, nil
}
