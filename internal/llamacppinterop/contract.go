// Package llamacppinterop defines fak's versioned delegation seam for llama.cpp.
package llamacppinterop

import (
	"bufio"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net"
	"net/http"
	"os"
	"os/exec"
	"regexp"
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/anthony-chaudhary/fak/internal/quantmeta"
)

// Schema identifies the versioned llama.cpp interop contract schema.
const (
	Schema = "fak.llamacppinterop/1"
	// WitnessedQwen38MTPCommit is the llama.cpp runtime measured on the A100 default path.
	WitnessedQwen38MTPCommit = "8144f31"
)

// Outcome indicates whether delegation to llama.cpp is accepted, abstained, or refused.
type Outcome string

const (
	// OutcomeDelegate accepts delegation to the discovered or planned runtime.
	OutcomeDelegate Outcome = "delegate"
	// OutcomeAbstain skips delegation because prerequisites or hardware features are missing.
	OutcomeAbstain Outcome = "abstain"
	// OutcomeRefuse rejects delegation due to invalid configuration or failed probes.
	OutcomeRefuse Outcome = "refuse"
)

// Capability captures runtime attributes discovered from a llama.cpp binary.
type Capability struct {
	Binary   string `json:"binary"`
	Version  string `json:"version"`
	Commit   string `json:"commit,omitempty"`
	Server   bool   `json:"server"`
	DraftMTP bool   `json:"draft_mtp,omitempty"`
	CUDA     bool   `json:"cuda,omitempty"`
}

// Result carries the structured outcome, diagnostic reason, and planned execution argv.
type Result struct {
	Schema     string     `json:"schema"`
	Outcome    Outcome    `json:"outcome"`
	Reason     string     `json:"reason"`
	Capability Capability `json:"capability"`
	Argv       []string   `json:"argv,omitempty"`
}

// Runner executes command-line probes against a candidate binary.
type Runner interface {
	Output(context.Context, string, ...string) ([]byte, error)
}

// ExecRunner executes binary probes via os/exec commands.
type ExecRunner struct{}

// Output executes the binary with arguments and returns combined stdout and stderr output.
func (ExecRunner) Output(ctx context.Context, name string, args ...string) ([]byte, error) {
	return exec.CommandContext(ctx, name, args...).CombinedOutput()
}

var (
	versionRE = regexp.MustCompile(`(?i)(?:version|build)\s*[: ]\s*([0-9]+(?:\.[0-9]+){0,2}|[a-f0-9]{7,40})`)
	commitRE  = regexp.MustCompile(`(?i)commit\s+([a-f0-9]{7,40})`)
)

// Discover probes a candidate binary to extract version, commit, and supported features.
func Discover(ctx context.Context, r Runner, binary string) Result {
	if strings.TrimSpace(binary) == "" {
		return Result{Schema: Schema, Outcome: OutcomeRefuse, Reason: "llama.cpp binary is empty"}
	}
	b, err := r.Output(ctx, binary, "--version")
	if err != nil {
		return Result{Schema: Schema, Outcome: OutcomeRefuse, Reason: fmt.Sprintf("version probe failed: %v", err)}
	}
	m := versionRE.FindStringSubmatch(string(b))
	if len(m) < 2 {
		return Result{Schema: Schema, Outcome: OutcomeAbstain, Reason: "llama.cpp version is not parseable"}
	}
	cap := Capability{Binary: binary, Version: m[1], Server: strings.Contains(strings.ToLower(binary), "server")}
	if cm := commitRE.FindStringSubmatch(string(b)); len(cm) == 2 {
		cap.Commit = cm[1]
	}
	if cap.Server {
		if help, helpErr := r.Output(ctx, binary, "--help"); helpErr == nil {
			cap.DraftMTP = strings.Contains(string(help), "draft-mtp")
		}
		if devices, deviceErr := r.Output(ctx, binary, "--list-devices"); deviceErr == nil {
			cap.CUDA = strings.Contains(strings.ToLower(string(devices)), "cuda")
		}
	}
	return Result{Schema: Schema, Outcome: OutcomeDelegate, Reason: "llama.cpp capability discovered", Capability: cap}
}

// WitnessedQwen38MTP reports whether capability provenance matches the measured runtime.
func WitnessedQwen38MTP(cap Capability) bool {
	commit := strings.ToLower(strings.TrimSpace(cap.Commit))
	return commit != "" && (strings.HasPrefix(commit, WitnessedQwen38MTPCommit) || strings.HasPrefix(WitnessedQwen38MTPCommit, commit)) && cap.Server && cap.DraftMTP && cap.CUDA
}

// Plan validates model requirements and constructs command-line arguments for standard delegation.
func Plan(cap Capability, model string, d quantmeta.Descriptor) Result {
	if cap.Binary == "" || cap.Version == "" {
		return Result{Schema: Schema, Outcome: OutcomeRefuse, Reason: "unproven llama.cpp capability"}
	}
	if strings.TrimSpace(model) == "" {
		return Result{Schema: Schema, Outcome: OutcomeRefuse, Reason: "model path is empty"}
	}
	if d.Artifact == nil || !strings.EqualFold(d.Artifact.ContainerID, "gguf") {
		return Result{Schema: Schema, Outcome: OutcomeAbstain, Reason: "llama.cpp delegation requires a GGUF artifact", Capability: cap}
	}
	argv := []string{cap.Binary, "-m", model}
	if cap.Server {
		argv = append(argv, "--host", "127.0.0.1", "--port", "0")
	}
	return Result{Schema: Schema, Outcome: OutcomeDelegate, Reason: "delegate to versioned llama.cpp runtime", Capability: cap, Argv: argv}
}

// PlanQwen38MTP plans the measured Qwen3.8 CUDA path. The caller supplies a
// loopback port so fak remains the public control plane and llama-server never binds externally.
func PlanQwen38MTP(cap Capability, model string, d quantmeta.Descriptor, port, contextTokens int) Result {
	base := Plan(cap, model, d)
	if base.Outcome != OutcomeDelegate {
		return base
	}
	if !cap.Server || !cap.DraftMTP || !cap.CUDA {
		base.Outcome, base.Reason, base.Argv = OutcomeAbstain, "llama-server does not advertise draft-mtp on a CUDA device", nil
		return base
	}
	arch := ""
	if raw := d.Extra["gguf_architecture"]; len(raw) > 0 {
		_ = json.Unmarshal(raw, &arch)
	}
	if arch != "qwen35" {
		base.Outcome, base.Reason, base.Argv = OutcomeAbstain, "Qwen3.8 MTP delegation requires qwen35 GGUF metadata", nil
		return base
	}
	if port < 1 || contextTokens < 1 {
		base.Outcome, base.Reason, base.Argv = OutcomeRefuse, "invalid llama-server loopback port or context", nil
		return base
	}
	base.Reason = "delegate Qwen3.8 to versioned llama-server draft-mtp runtime"
	base.Argv = []string{cap.Binary, "-m", model, "--host", "127.0.0.1", "--port", strconv.Itoa(port), "-ngl", "99", "-c", strconv.Itoa(contextTokens), "-b", "4096", "-ub", "1024", "--spec-type", "draft-mtp", "--spec-draft-n-max", "3", "--spec-draft-p-min", "0.0"}
	return base
}

// Health holds the parsed status response from llama-server's /health endpoint.
type Health struct {
	Status string `json:"status"`
}

// CheckHealth queries a llama-server /health endpoint to verify server readiness.
func CheckHealth(ctx context.Context, c *http.Client, url string) Result {
	if c == nil {
		c = &http.Client{Timeout: 2 * time.Second}
	}
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, strings.TrimRight(url, "/")+"/health", nil)
	if err != nil {
		return Result{Schema: Schema, Outcome: OutcomeRefuse, Reason: err.Error()}
	}
	resp, err := c.Do(req)
	if err != nil {
		return Result{Schema: Schema, Outcome: OutcomeRefuse, Reason: "health request failed: " + err.Error()}
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return Result{Schema: Schema, Outcome: OutcomeRefuse, Reason: fmt.Sprintf("health returned HTTP %d", resp.StatusCode)}
	}
	var h Health
	if json.NewDecoder(resp.Body).Decode(&h) != nil || !(h.Status == "ok" || h.Status == "ready") {
		return Result{Schema: Schema, Outcome: OutcomeAbstain, Reason: "health response is not ready"}
	}
	return Result{Schema: Schema, Outcome: OutcomeDelegate, Reason: "llama.cpp server is healthy"}
}

// Process owns one loopback llama-server child and its bounded startup/shutdown lifecycle.
type Process struct {
	cmd      *exec.Cmd
	url      string
	log      io.ReadCloser
	done     chan error
	stopOnce sync.Once
}

// Start spawns a llama-server subprocess and waits for health readiness until the timeout.
func Start(ctx context.Context, argv []string, startupTimeout time.Duration) (*Process, error) {
	if len(argv) == 0 {
		return nil, fmt.Errorf("llama.cpp argv is empty")
	}
	if startupTimeout <= 0 {
		startupTimeout = 90 * time.Second
	}
	cmd := exec.Command(argv[0], argv[1:]...)
	logR, logW := io.Pipe()
	cmd.Stdout = logW
	cmd.Stderr = logW
	if err := cmd.Start(); err != nil {
		_ = logR.Close()
		return nil, err
	}
	p := &Process{cmd: cmd, url: loopbackURL(argv), log: logR, done: make(chan error, 1)}
	go func() {
		err := cmd.Wait()
		_ = logW.Close()
		p.done <- err
		close(p.done)
	}()
	deadline := time.NewTimer(startupTimeout)
	defer deadline.Stop()
	tick := time.NewTicker(100 * time.Millisecond)
	defer tick.Stop()
	client := &http.Client{Timeout: 500 * time.Millisecond}
	var lines strings.Builder
	scanDone := make(chan struct{})
	go func() {
		s := bufio.NewScanner(logR)
		for s.Scan() {
			if lines.Len() < 32<<10 {
				lines.WriteString(s.Text())
				lines.WriteByte('\n')
			}
		}
		close(scanDone)
	}()
	for {
		if CheckHealth(ctx, client, p.url).Outcome == OutcomeDelegate {
			return p, nil
		}
		select {
		case err := <-p.done:
			<-scanDone
			return nil, fmt.Errorf("llama-server exited before ready: %v: %s", err, strings.TrimSpace(lines.String()))
		case <-ctx.Done():
			_ = p.Stop()
			<-scanDone
			return nil, ctx.Err()
		case <-deadline.C:
			_ = p.Stop()
			<-scanDone
			return nil, fmt.Errorf("llama-server readiness timed out after %s: %s", startupTimeout, strings.TrimSpace(lines.String()))
		case <-tick.C:
		}
	}
}

// BaseURL returns the HTTP base URL for OpenAI-compatible completions.
func (p *Process) BaseURL() string { return p.url + "/v1" }

// PID returns the OS process ID of the running subprocess, or 0 if inactive.
func (p *Process) PID() int {
	if p == nil || p.cmd == nil || p.cmd.Process == nil {
		return 0
	}
	return p.cmd.Process.Pid
}

// Stop terminates the subprocess cleanly with SIGINT, escalating to SIGKILL on timeout.
func (p *Process) Stop() error {
	if p == nil {
		return nil
	}
	var stopErr error
	p.stopOnce.Do(func() {
		select {
		case <-p.done:
			return
		default:
		}
		if p.cmd != nil && p.cmd.Process != nil {
			if err := p.cmd.Process.Signal(os.Interrupt); err != nil && !errors.Is(err, os.ErrProcessDone) {
				stopErr = err
			}
		}
		select {
		case <-p.done:
		case <-time.After(5 * time.Second):
			if p.cmd != nil && p.cmd.Process != nil {
				if err := p.cmd.Process.Kill(); err != nil && !errors.Is(err, os.ErrProcessDone) {
					stopErr = err
				}
			}
		}
	})
	return stopErr
}
func loopbackURL(argv []string) string {
	port := ""
	for i := 0; i+1 < len(argv); i++ {
		if argv[i] == "--port" {
			port = argv[i+1]
		}
	}
	if port == "" {
		return ""
	}
	return "http://" + net.JoinHostPort("127.0.0.1", port)
}
