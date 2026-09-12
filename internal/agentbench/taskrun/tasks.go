package taskrun

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"sort"
	"strings"
	"sync"
	"time"

	"github.com/anthony-chaudhary/fak/internal/agentbench/taskfixture"
)

const (
	maxTaskTurns = 12
	maxTaskTime  = 5 * time.Minute
	maxChildOut  = 64 << 10
)

type Options struct {
	Executable     string
	Endpoint       string
	Model          string
	OutDir         string
	Concurrency    int
	IncludeHeldout bool
	TaskLimit      int
	modelMeter     *modelConcurrencyMeter
}

// Receipt records only witnessed task outcomes. ObservedModelRequestConcurrency
// is nil until an external endpoint observer supplies that evidence.
type Receipt struct {
	Schema                          string        `json:"schema"`
	Endpoint                        string        `json:"endpoint"`
	Model                           string        `json:"model"`
	ConfiguredConcurrency           int           `json:"configured_concurrency"`
	ObservedModelRequestConcurrency *int          `json:"observed_model_request_concurrency,omitempty"`
	ConcurrencyQualified            bool          `json:"concurrency_qualified"`
	IncludeHeldout                  bool          `json:"include_heldout"`
	StartedAt                       time.Time     `json:"started_at"`
	Duration                        time.Duration `json:"duration"`
	Tasks                           []TaskReceipt `json:"tasks"`
}

type TaskReceipt struct {
	ID             string            `json:"id"`
	Family         string            `json:"family"`
	Passed         bool              `json:"passed"`
	Accepted       bool              `json:"accepted"`
	Error          string            `json:"error,omitempty"`
	Duration       time.Duration     `json:"duration"`
	PlannerCalls   int               `json:"planner_calls,omitempty"`
	Inspected      bool              `json:"inspected,omitempty"`
	Edited         bool              `json:"edited,omitempty"`
	TestSucceeded  bool              `json:"test_succeeded,omitempty"`
	ExternalPassed bool              `json:"external_passed,omitempty"`
	DeniedAttempts int               `json:"denied_attempts,omitempty"`
	DiffDigest     string            `json:"diff_digest,omitempty"`
	Artifact       string            `json:"artifact,omitempty"`
	Controls       ControlReceipt    `json:"controls"`
	Prehistory     PrehistoryReceipt `json:"prehistory,omitempty"`
	Model          ModelObservation  `json:"model_observation"`
}

type ControlReceipt struct {
	BrokenFails    bool   `json:"broken_fails"`
	KnownFixPasses bool   `json:"known_fix_passes"`
	BrokenDigest   string `json:"broken_digest,omitempty"`
	FixedDigest    string `json:"fixed_digest,omitempty"`
}

type childConfig struct {
	TaskID      string `json:"task_id"`
	Prompt      string `json:"prompt"`
	TrialRoot   string `json:"trial_root"`
	Endpoint    string `json:"endpoint,omitempty"`
	Model       string `json:"model,omitempty"`
	TestCommand string `json:"test_command,omitempty"`
	VisibleTest string `json:"visible_test,omitempty"`
}

type testChildConfig struct {
	CandidateRoot string `json:"candidate_root"`
	TargetFile    string `json:"target_file"`
	VisibleRoot   string `json:"visible_root"`
	OracleRoot    string `json:"oracle_root"`
}

// sandboxRunMu avoids multiplying cold local-toolchain compiles. Model children
// remain bounded by Options.Concurrency; this lock covers only local test controls.
var sandboxRunSlot = make(chan struct{}, 1)

func runControlledTests(ctx context.Context, candidate, target, visible, oracle string) (TestResult, error) {
	select {
	case sandboxRunSlot <- struct{}{}:
		defer func() { <-sandboxRunSlot }()
	case <-ctx.Done():
		return TestResult{ExitCode: -1}, ctx.Err()
	}
	return runCandidateTests(ctx, candidate, target, visible, oracle)
}

func Run(ctx context.Context, opts Options) (Receipt, error) {
	started := time.Now()
	r := Receipt{Schema: "fak.agentbench.taskrun.v1", Endpoint: opts.Endpoint, Model: opts.Model, ConfiguredConcurrency: opts.Concurrency, IncludeHeldout: opts.IncludeHeldout, StartedAt: started}
	if strings.TrimSpace(opts.Executable) == "" || strings.TrimSpace(opts.Endpoint) == "" || strings.TrimSpace(opts.Model) == "" || strings.TrimSpace(opts.OutDir) == "" {
		return r, errors.New("taskrun: executable, endpoint, model, and out dir are required")
	}
	executable, err := filepath.Abs(opts.Executable)
	if err != nil {
		return r, err
	}
	info, err := os.Stat(executable)
	if err != nil || !info.Mode().IsRegular() || info.Mode()&0111 == 0 {
		return r, errors.New("taskrun: executable must be an executable regular file")
	}
	opts.Executable = executable
	if opts.Concurrency <= 0 {
		return r, errors.New("taskrun: concurrency must be positive")
	}
	if err := SandboxAvailable(); err != nil {
		return r, err
	}
	if err := os.MkdirAll(opts.OutDir, 0700); err != nil {
		return r, err
	}
	if err := os.Chmod(opts.OutDir, 0700); err != nil {
		return r, err
	}
	cases := taskfixture.Cases(opts.IncludeHeldout)
	if opts.TaskLimit < 0 {
		return r, errors.New("taskrun: task limit must not be negative")
	}
	if opts.TaskLimit > 0 && opts.TaskLimit < len(cases) {
		cases = cases[:opts.TaskLimit]
	}
	r.Tasks = make([]TaskReceipt, len(cases))
	opts.modelMeter = &modelConcurrencyMeter{}
	sem := make(chan struct{}, opts.Concurrency)
	var wg sync.WaitGroup
	for i, fixture := range cases {
		i, fixture := i, fixture
		wg.Add(1)
		go func() {
			defer wg.Done()
			select {
			case sem <- struct{}{}:
			case <-ctx.Done():
				r.Tasks[i] = TaskReceipt{ID: fixture.ID, Family: fixture.Family, Error: ctx.Err().Error()}
				return
			}
			defer func() { <-sem }()
			r.Tasks[i] = runOne(ctx, opts, fixture)
		}()
	}
	wg.Wait()
	qualified := len(r.Tasks) > 0
	for _, task := range r.Tasks {
		if task.Model.EventsDigest == "" || task.Model.SuccessfulRequests == 0 || task.Model.ModelIdentityStatus != "matched" || task.Model.LimitExceeded {
			qualified = false
		}
	}
	if qualified {
		peak := opts.modelMeter.observedPeak()
		r.ObservedModelRequestConcurrency = &peak
		r.ConcurrencyQualified = true
	}
	r.Duration = time.Since(started)
	if err := writeJSON(filepath.Join(opts.OutDir, "receipt.json"), r); err != nil {
		return r, err
	}
	return r, nil
}

func runOne(parent context.Context, opts Options, f taskfixture.Fixture) (tr TaskReceipt) {
	started := time.Now()
	tr = TaskReceipt{ID: f.ID, Family: f.Family}
	ctx, cancel := context.WithTimeout(parent, maxTaskTime)
	defer cancel()
	taskDir := filepath.Join(opts.OutDir, safeName(f.ID))
	defer func() {
		tr.Duration = time.Since(started)
		if err := writeJSON(filepath.Join(taskDir, "task.json"), tr); err != nil {
			tr.Passed = false
			tr.Accepted = false
			message := "persist task receipt: " + err.Error()
			if tr.Error != "" {
				message += "; task error: " + tr.Error
			}
			tr.Error = message
		}
	}()
	trial, visible, oracle, known := filepath.Join(taskDir, "trial"), filepath.Join(taskDir, "visible"), filepath.Join(taskDir, "oracle"), filepath.Join(taskDir, "known-fix")
	for _, d := range []string{taskDir, trial, visible, oracle, known} {
		if err := os.MkdirAll(d, 0700); err != nil {
			tr.Error = err.Error()
			return tr
		}
		_ = os.Chmod(d, 0700)
	}
	if err := seedFixture(trial, f, f.BrokenSource, false); err != nil {
		tr.Error = err.Error()
		return tr
	}
	if err := seedFixture(known, f, f.FixedSource, false); err != nil {
		tr.Error = err.Error()
		return tr
	}
	if err := os.WriteFile(filepath.Join(visible, "visible_test.go"), []byte(f.VisibleTest), 0400); err != nil {
		tr.Error = err.Error()
		return tr
	}
	if err := os.WriteFile(filepath.Join(oracle, "oracle_test.go"), []byte(f.OracleTest), 0600); err != nil {
		tr.Error = err.Error()
		return tr
	}
	if f.Workflow.Kind == "shared_area_followup" {
		var prehistoryErr error
		tr.Prehistory, prehistoryErr = validateFrozenPrehistory(ctx, f, filepath.Join(taskDir, "prehistory.json"))
		if prehistoryErr != nil || !tr.Prehistory.Passed {
			tr.Error = "frozen prehistory verification failed"
			if prehistoryErr != nil {
				tr.Error += ": " + prehistoryErr.Error()
			}
			return tr
		}
	}
	tr.Controls.BrokenDigest = digestString(f.BrokenSource)
	tr.Controls.FixedDigest = digestString(f.FixedSource)
	broken, err := runControlledTests(ctx, trial, f.TargetFile, visible, oracle)
	if err != nil {
		tr.Error = "broken control: " + err.Error()
		return tr
	}
	tr.Controls.BrokenFails = broken.ExitCode != 0
	green, err := runControlledTests(ctx, known, f.TargetFile, visible, oracle)
	if err != nil {
		tr.Error = "known-fix control: " + err.Error()
		return tr
	}
	tr.Controls.KnownFixPasses = green.ExitCode == 0
	if !tr.Controls.BrokenFails || !tr.Controls.KnownFixPasses {
		tr.Error = "fixture controls did not prove red/green"
		return tr
	}
	observer, err := startModelObserver(ctx, opts.Endpoint, opts.Model, filepath.Join(taskDir, "model-events.jsonl"), maxTaskTurns)
	if err != nil {
		tr.Error = "model observer: " + err.Error()
		return tr
	}
	observer.meter = opts.modelMeter
	testCfgPath := filepath.Join(taskDir, "test-config.json")
	if err := writeJSON(testCfgPath, testChildConfig{CandidateRoot: trial, TargetFile: f.TargetFile, VisibleRoot: visible, OracleRoot: oracle}); err != nil {
		tr.Model, _ = observer.Close()
		tr.Error = err.Error()
		return tr
	}
	testCommand := strings.Join([]string{opts.Executable, "bench", "agent", "test-child", "--config", testCfgPath}, " ")
	childCfgPath := filepath.Join(taskDir, "child-config.json")
	if err := writeJSON(childCfgPath, childConfig{TaskID: f.ID, Prompt: f.Prompt, TrialRoot: trial, Endpoint: observer.Endpoint(), Model: opts.Model, TestCommand: testCommand, VisibleTest: f.VisibleTest}); err != nil {
		tr.Model, _ = observer.Close()
		tr.Error = err.Error()
		return tr
	}
	out, exit, err := runProcess(ctx, opts.Executable, "bench", "agent", "task-child", "--config", childCfgPath)
	observation, observerErr := observer.Close()
	tr.Model = observation
	_ = os.WriteFile(filepath.Join(taskDir, "child.json"), out, 0600)
	var cr ChildReceipt
	if decodeErr := json.NewDecoder(bytes.NewReader(out)).Decode(&cr); decodeErr != nil {
		tr.Error = "decode child receipt: " + decodeErr.Error()
		return tr
	}
	tr.PlannerCalls, tr.Inspected, tr.Edited, tr.TestSucceeded, tr.DeniedAttempts = cr.PlannerCalls, cr.Inspected, cr.Edited, cr.TestSucceeded, cr.DeniedAttempts
	if observerErr != nil {
		tr.Error = "model observer: " + observerErr.Error()
		return tr
	}
	if observation.LimitExceeded || observation.Requests == 0 || observation.ModelIdentityStatus != "matched" {
		tr.Error = "model request witness incomplete"
		return tr
	}
	if err != nil || exit != 0 {
		tr.Error = fmt.Sprintf("task child exit %d: %v", exit, err)
		return tr
	}
	if cr.PlannerCalls > maxTaskTurns || !cr.Inspected || !cr.Edited || !cr.TestSucceeded || cr.DeniedAttempts != 0 {
		tr.Error = "child behavior witness incomplete"
		return tr
	}
	changed, digest, err := validateCandidate(trial, f)
	if err != nil {
		tr.Error = err.Error()
		return tr
	}
	tr.DiffDigest = digest
	if !changed {
		tr.Error = "candidate source did not change"
		return tr
	}
	verifyRoot := filepath.Join(taskDir, "verify")
	if err := seedFixture(verifyRoot, f, mustRead(filepath.Join(trial, f.TargetFile)), true); err != nil {
		tr.Error = err.Error()
		return tr
	}
	if err := os.WriteFile(filepath.Join(verifyRoot, "oracle_test.go"), []byte(f.OracleTest), 0600); err != nil {
		tr.Error = err.Error()
		return tr
	}
	vr, err := runSandboxedTests(ctx, verifyRoot, oracle, false)
	if err != nil {
		tr.Error = "external verifier: " + err.Error()
		return tr
	}
	tr.ExternalPassed = vr.ExitCode == 0
	tr.Passed = tr.ExternalPassed
	tr.Accepted = tr.Passed
	if !tr.Passed {
		tr.Error = "external hidden verifier failed"
	}
	tr.Artifact = filepath.Join(taskDir, "task.json")
	return tr
}

func seedFixture(root string, f taskfixture.Fixture, source string, includeOracle bool) error {
	if err := os.MkdirAll(filepath.Dir(filepath.Join(root, f.TargetFile)), 0700); err != nil {
		return err
	}
	files := map[string]string{"go.mod": "module fixture\n\ngo 1.26\n", f.TargetFile: source}
	if includeOracle {
		files["oracle_test.go"] = f.OracleTest
	}
	for name, body := range files {
		if err := os.WriteFile(filepath.Join(root, name), []byte(body), 0600); err != nil {
			return err
		}
	}
	return nil
}

func validateCandidate(root string, f taskfixture.Fixture) (bool, string, error) {
	allowed := map[string]bool{"go.mod": true, f.TargetFile: true}
	var names []string
	err := filepath.WalkDir(root, func(path string, d os.DirEntry, err error) error {
		if err != nil {
			return err
		}
		if path == root {
			return nil
		}
		rel, e := filepath.Rel(root, path)
		if e != nil {
			return e
		}
		if d.Type()&os.ModeSymlink != 0 {
			return fmt.Errorf("candidate contains symlink %s", rel)
		}
		if d.IsDir() {
			return nil
		}
		if !allowed[rel] {
			return fmt.Errorf("candidate changed or added out-of-scope file %s", rel)
		}
		names = append(names, rel)
		return nil
	})
	if err != nil {
		return false, "", err
	}
	for _, name := range []string{"go.mod", f.TargetFile} {
		if !contains(names, name) {
			return false, "", fmt.Errorf("candidate removed %s", name)
		}
	}
	if mustRead(filepath.Join(root, "go.mod")) != "module fixture\n\ngo 1.26\n" {
		return false, "", errors.New("candidate changed go.mod")
	}
	source := mustRead(filepath.Join(root, f.TargetFile))
	if err := validateCandidateSource(f.BrokenSource, source); err != nil {
		return false, "", fmt.Errorf("candidate structural boundary: %w", err)
	}
	changed := source != f.BrokenSource
	return changed, digestString(source), nil
}

func runProcess(ctx context.Context, exe string, args ...string) ([]byte, int, error) {
	cmd := exec.Command(exe, args...)
	configureChildProcess(cmd)
	var b boundedWriter
	cmd.Stdout = &b
	cmd.Stderr = &b
	if err := cmd.Start(); err != nil {
		return nil, -1, err
	}
	done := make(chan error, 1)
	go func() { done <- cmd.Wait() }()
	select {
	case err := <-done:
		if err == nil {
			return b.Bytes(), 0, nil
		}
		var ee *exec.ExitError
		if errors.As(err, &ee) {
			return b.Bytes(), ee.ExitCode(), err
		}
		return b.Bytes(), -1, err
	case <-ctx.Done():
		killChildProcess(cmd)
		<-done
		return b.Bytes(), -1, ctx.Err()
	}
}

type boundedWriter struct {
	mu sync.Mutex
	b  bytes.Buffer
}

func (b *boundedWriter) Write(p []byte) (int, error) {
	b.mu.Lock()
	defer b.mu.Unlock()
	n := len(p)
	if left := maxChildOut - b.b.Len(); left > 0 {
		if len(p) > left {
			p = p[:left]
		}
		_, _ = b.b.Write(p)
	}
	return n, nil
}
func (b *boundedWriter) Bytes() []byte {
	b.mu.Lock()
	defer b.mu.Unlock()
	return append([]byte(nil), b.b.Bytes()...)
}
func writeJSON(path string, v any) error {
	b, e := json.MarshalIndent(v, "", "  ")
	if e != nil {
		return e
	}
	return os.WriteFile(path, append(b, '\n'), 0600)
}
func digestString(s string) string { h := sha256.Sum256([]byte(s)); return hex.EncodeToString(h[:]) }
func mustRead(path string) string  { b, _ := os.ReadFile(path); return string(b) }
func safeName(s string) string {
	return strings.Map(func(r rune) rune {
		if r >= 'a' && r <= 'z' || r >= 'A' && r <= 'Z' || r >= '0' && r <= '9' || r == '-' || r == '_' {
			return r
		}
		return '-'
	}, s)
}
func contains(xs []string, s string) bool {
	sort.Strings(xs)
	i := sort.SearchStrings(xs, s)
	return i < len(xs) && xs[i] == s
}

var _ io.Writer = (*boundedWriter)(nil)
var _ = runtime.GOOS
