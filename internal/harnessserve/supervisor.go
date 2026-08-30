// Package harnessserve owns the bounded lifecycle of one adapter-provided local
// model runtime. It never downloads artifacts and never exposes a non-loopback
// listener.
package harnessserve

import (
	"bytes"
	"context"
	"crypto/rand"
	"crypto/sha256"
	"crypto/subtle"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/anthony-chaudhary/fak/internal/processalive"
)

const (
	RefusalInvalidPlan       = "INVALID_PLAN"
	RefusalAlreadyRunning    = "ALREADY_RUNNING"
	RefusalStartFailed       = "START_FAILED"
	RefusalReadinessTimeout  = "READINESS_TIMEOUT"
	RefusalProbeFailed       = "PROBE_FAILED"
	RefusalStaleOwnership    = "STALE_OWNERSHIP"
	RefusalStopTimeout       = "STOP_TIMEOUT"
	addressPlaceholder       = "{address}"
	ownershipTokenEnv        = "FAK_HARNESS_SERVE_OWNERSHIP_TOKEN"
	defaultMaximumOutputSize = 16 << 10
)

// Refusal is a stable fail-closed lifecycle error.
type Refusal struct {
	Code   string
	Detail string
	Err    error
}

func (r *Refusal) Error() string {
	if r.Err != nil {
		return fmt.Sprintf("harness serve %s: %s: %v", r.Code, r.Detail, r.Err)
	}
	return fmt.Sprintf("harness serve %s: %s", r.Code, r.Detail)
}

func (r *Refusal) Unwrap() error { return r.Err }

// Plan is the explicit adapter contract. Args must contain {address}; the
// supervisor replaces it with a freshly allocated IPv4 loopback address.
type Plan struct {
	Executable           string
	Args                 []string
	Env                  []string
	Dir                  string
	Model                string
	HealthPath           string
	CompletionPath       string
	GracefulShutdownPath string
	StartupTimeout       time.Duration
	ProbeTimeout         time.Duration
	PollInterval         time.Duration
	GracefulStopTimeout  time.Duration
	KillTimeout          time.Duration
}

// Ownership is the capability required to stop the process. PID alone is never
// sufficient: the random token and process-start identity must also match.
type Ownership struct {
	PID           int    `json:"pid"`
	Token         string `json:"token"`
	StartIdentity string `json:"start_identity"`
}

type ProbeReceipt struct {
	HTTPStatus       int `json:"http_status"`
	CompletionTokens int `json:"completion_tokens"`
}

// Receipt binds readiness to the owned process and the exact explicit argv.
type Receipt struct {
	Endpoint   string       `json:"endpoint"`
	Ownership  Ownership    `json:"ownership"`
	ArgvSHA256 string       `json:"argv_sha256"`
	StartedAt  time.Time    `json:"started_at"`
	ReadyAt    time.Time    `json:"ready_at"`
	Probe      ProbeReceipt `json:"probe"`
}

type StopReceipt struct {
	GracefulAttempted bool `json:"graceful_attempted"`
	Escalated         bool `json:"escalated"`
	AlreadyExited     bool `json:"already_exited"`
}

// Supervisor owns at most one child. It retains the os.Process handle rather
// than reopening an arbitrary PID when Stop is called.
type Supervisor struct {
	mu       sync.Mutex
	starting bool
	owned    *ownedProcess
}

type ownedProcess struct {
	cmd      *exec.Cmd
	plan     Plan
	endpoint string
	identity Ownership
	started  time.Time
	done     chan struct{}
	waitMu   sync.Mutex
	waitErr  error
	output   *boundedBuffer
}

// Launch starts, health-checks, and protocol-probes one local runtime.
func (s *Supervisor) Launch(ctx context.Context, plan Plan) (Receipt, error) {
	if err := validateSupervisorLaunchContract(plan); err != nil {
		return Receipt{}, err
	}
	s.mu.Lock()
	if s.starting || s.owned != nil {
		s.mu.Unlock()
		return Receipt{}, refuse(RefusalAlreadyRunning, "supervisor already owns a runtime", nil)
	}
	s.starting = true
	s.mu.Unlock()

	launched := false
	defer func() {
		if !launched {
			s.mu.Lock()
			s.starting = false
			s.mu.Unlock()
		}
	}()

	address, err := allocateLoopbackAddress()
	if err != nil {
		return Receipt{}, refuse(RefusalStartFailed, "allocate IPv4 loopback address", err)
	}
	argv, err := materializeArgs(plan.Args, address)
	if err != nil {
		return Receipt{}, err
	}
	token, err := randomToken()
	if err != nil {
		return Receipt{}, refuse(RefusalStartFailed, "create ownership token", err)
	}
	cmd := exec.Command(plan.Executable, argv...)
	cmd.Dir = plan.Dir
	cmd.Env = append([]string(nil), plan.Env...)
	cmd.Env = append(cmd.Env, ownershipTokenEnv+"="+token)
	output := &boundedBuffer{limit: defaultMaximumOutputSize}
	cmd.Stdout, cmd.Stderr = output, output
	started := time.Now().UTC()
	if err := cmd.Start(); err != nil {
		return Receipt{}, refuse(RefusalStartFailed, "start explicit runtime argv", err)
	}
	identity := Ownership{PID: cmd.Process.Pid, Token: token}
	identity.StartIdentity = captureStartIdentity(identity.PID, started)
	p := &ownedProcess{
		cmd: cmd, plan: plan, endpoint: "http://" + address, identity: identity,
		started: started, done: make(chan struct{}), output: output,
	}
	go func() {
		err := cmd.Wait()
		p.waitMu.Lock()
		p.waitErr = err
		p.waitMu.Unlock()
		close(p.done)
	}()
	s.mu.Lock()
	s.owned = p
	s.mu.Unlock()

	cleanup := func() {
		_, _ = stopProcess(context.Background(), p)
		s.mu.Lock()
		if s.owned == p {
			s.owned = nil
		}
		s.starting = false
		s.mu.Unlock()
	}
	readyAt, err := waitReady(ctx, p)
	if err != nil {
		cleanup()
		return Receipt{}, err
	}
	probe, err := runCompletionProbe(ctx, p)
	if err != nil {
		cleanup()
		return Receipt{}, err
	}
	s.mu.Lock()
	s.starting = false
	s.mu.Unlock()
	launched = true
	return Receipt{
		Endpoint: p.endpoint, Ownership: identity, ArgvSHA256: digestArgv(plan.Executable, argv),
		StartedAt: started, ReadyAt: readyAt, Probe: probe,
	}, nil
}

// Stop refuses stale ownership before any signal, request, or PID operation.
func (s *Supervisor) Stop(ctx context.Context, ownership Ownership) (StopReceipt, error) {
	s.mu.Lock()
	if s.starting {
		s.mu.Unlock()
		return StopReceipt{}, refuse(RefusalAlreadyRunning, "runtime launch is still being proven", nil)
	}
	p := s.owned
	if p == nil {
		s.mu.Unlock()
		return StopReceipt{}, refuse(RefusalStaleOwnership, "supervisor owns no matching runtime", nil)
	}
	if !sameOwnership(p.identity, ownership) {
		s.mu.Unlock()
		return StopReceipt{}, refuse(RefusalStaleOwnership, "PID, start identity, or ownership token does not match", nil)
	}
	select {
	case <-p.done:
		s.owned = nil
		s.mu.Unlock()
		return StopReceipt{AlreadyExited: true}, nil
	default:
	}
	if current, ok := currentStartIdentity(p.identity.PID); ok && current != p.identity.StartIdentity {
		s.mu.Unlock()
		return StopReceipt{}, refuse(RefusalStaleOwnership, "kernel process start identity changed; refusing PID reuse", nil)
	}
	s.mu.Unlock()

	receipt, err := stopProcess(ctx, p)
	if err == nil {
		s.mu.Lock()
		if s.owned == p {
			s.owned = nil
		}
		s.mu.Unlock()
	}
	return receipt, err
}

// validateSupervisorLaunchContract validates only one adapter-neutral child
// launch contract. It is distinct from repository/work dispatch planning: this
// check never selects work, schedules workers, or mutates a durable plan.
func validateSupervisorLaunchContract(plan Plan) error {
	if strings.TrimSpace(plan.Executable) == "" || !filepath.IsAbs(plan.Executable) {
		return refuse(RefusalInvalidPlan, "executable must be an absolute path", nil)
	}
	info, err := os.Stat(plan.Executable)
	if err != nil || info.IsDir() {
		return refuse(RefusalInvalidPlan, "executable must name a readable file", err)
	}
	if plan.Dir != "" && !filepath.IsAbs(plan.Dir) {
		return refuse(RefusalInvalidPlan, "working directory must be absolute when set", nil)
	}
	if plan.StartupTimeout <= 0 || plan.ProbeTimeout <= 0 || plan.PollInterval <= 0 || plan.GracefulStopTimeout <= 0 || plan.KillTimeout <= 0 {
		return refuse(RefusalInvalidPlan, "all lifecycle durations must be positive and bounded", nil)
	}
	if strings.TrimSpace(plan.Model) == "" {
		return refuse(RefusalInvalidPlan, "probe model is required", nil)
	}
	for _, path := range []string{plan.HealthPath, plan.CompletionPath} {
		if err := validateHTTPPath(path); err != nil {
			return err
		}
	}
	if plan.GracefulShutdownPath != "" {
		if err := validateHTTPPath(plan.GracefulShutdownPath); err != nil {
			return err
		}
	}
	foundAddress := false
	for _, arg := range plan.Args {
		if strings.ContainsAny(arg, "\x00\r\n") {
			return refuse(RefusalInvalidPlan, "argv contains a control character", nil)
		}
		foundAddress = foundAddress || strings.Contains(arg, addressPlaceholder)
	}
	if !foundAddress {
		return refuse(RefusalInvalidPlan, "argv must contain the explicit {address} placeholder", nil)
	}
	for _, env := range plan.Env {
		key, value, ok := strings.Cut(env, "=")
		if !ok || key == "" || strings.ContainsAny(key+value, "\x00\r\n") || strings.EqualFold(key, ownershipTokenEnv) {
			return refuse(RefusalInvalidPlan, "environment entries must be explicit KEY=VALUE rows and may not set the ownership token", nil)
		}
	}
	return nil
}

func validateHTTPPath(path string) error {
	if !strings.HasPrefix(path, "/") || strings.HasPrefix(path, "//") || strings.ContainsAny(path, "\x00\r\n\\") || strings.Contains(path, "://") {
		return refuse(RefusalInvalidPlan, "health, completion, and shutdown paths must be local absolute HTTP paths", nil)
	}
	return nil
}

func materializeArgs(args []string, address string) ([]string, error) {
	out := make([]string, len(args))
	for i, arg := range args {
		out[i] = strings.ReplaceAll(arg, addressPlaceholder, address)
		if strings.Contains(out[i], "{") || strings.Contains(out[i], "}") {
			return nil, refuse(RefusalInvalidPlan, "argv contains an unknown substitution", nil)
		}
	}
	return out, nil
}

func allocateLoopbackAddress() (string, error) {
	listener, err := net.Listen("tcp4", "127.0.0.1:0")
	if err != nil {
		return "", err
	}
	address := listener.Addr().String()
	if err := listener.Close(); err != nil {
		return "", err
	}
	host, _, err := net.SplitHostPort(address)
	if err != nil || !net.ParseIP(host).IsLoopback() {
		return "", errors.New("allocator returned a non-loopback address")
	}
	return address, nil
}

func waitReady(parent context.Context, p *ownedProcess) (time.Time, error) {
	ctx, cancel := context.WithTimeout(parent, p.plan.StartupTimeout)
	defer cancel()
	client := loopbackClient(p.plan.ProbeTimeout)
	ticker := time.NewTicker(p.plan.PollInterval)
	defer ticker.Stop()
	var last string
	for {
		req, _ := http.NewRequestWithContext(ctx, http.MethodGet, p.endpoint+p.plan.HealthPath, nil)
		resp, err := client.Do(req)
		if err == nil {
			body, readErr := io.ReadAll(io.LimitReader(resp.Body, 4<<10))
			_ = resp.Body.Close()
			var health struct {
				Status string `json:"status"`
			}
			decodeErr := json.Unmarshal(body, &health)
			if resp.StatusCode >= 200 && resp.StatusCode < 300 && decodeErr == nil && strings.EqualFold(health.Status, "ready") {
				return time.Now().UTC(), nil
			}
			if readErr != nil {
				last = readErr.Error()
			} else {
				last = fmt.Sprintf("status=%d body=%q", resp.StatusCode, strings.TrimSpace(string(body)))
			}
		} else {
			last = err.Error()
		}
		select {
		case <-p.done:
			return time.Time{}, refuse(RefusalStartFailed, "runtime exited before readiness: "+p.output.String(), p.waitError())
		case <-ctx.Done():
			return time.Time{}, refuse(RefusalReadinessTimeout, "health never became ready; last="+last, ctx.Err())
		case <-ticker.C:
		}
	}
}

func runCompletionProbe(parent context.Context, p *ownedProcess) (ProbeReceipt, error) {
	ctx, cancel := context.WithTimeout(parent, p.plan.ProbeTimeout)
	defer cancel()
	payload, _ := json.Marshal(map[string]any{
		"model": p.plan.Model, "prompt": "fak readiness probe", "max_tokens": 1, "temperature": 0,
	})
	req, _ := http.NewRequestWithContext(ctx, http.MethodPost, p.endpoint+p.plan.CompletionPath, bytes.NewReader(payload))
	req.Header.Set("Content-Type", "application/json")
	resp, err := loopbackClient(p.plan.ProbeTimeout).Do(req)
	if err != nil {
		return ProbeReceipt{}, refuse(RefusalProbeFailed, "one-token completion request failed", err)
	}
	defer resp.Body.Close()
	body, err := io.ReadAll(io.LimitReader(resp.Body, 16<<10))
	if err != nil {
		return ProbeReceipt{}, refuse(RefusalProbeFailed, "read completion response", err)
	}
	var decoded struct {
		Choices []struct {
			Text string `json:"text"`
		} `json:"choices"`
		Usage struct {
			CompletionTokens int `json:"completion_tokens"`
		} `json:"usage"`
	}
	if resp.StatusCode < 200 || resp.StatusCode >= 300 || json.Unmarshal(body, &decoded) != nil || len(decoded.Choices) != 1 || decoded.Choices[0].Text == "" || decoded.Usage.CompletionTokens < 0 || decoded.Usage.CompletionTokens > 1 {
		return ProbeReceipt{}, refuse(RefusalProbeFailed, fmt.Sprintf("invalid bounded completion response status=%d body=%q", resp.StatusCode, strings.TrimSpace(string(body))), nil)
	}
	return ProbeReceipt{HTTPStatus: resp.StatusCode, CompletionTokens: decoded.Usage.CompletionTokens}, nil
}

func stopProcess(parent context.Context, p *ownedProcess) (StopReceipt, error) {
	receipt := StopReceipt{GracefulAttempted: true}
	select {
	case <-p.done:
		receipt.AlreadyExited = true
		return receipt, nil
	default:
	}
	if p.plan.GracefulShutdownPath != "" {
		ctx, cancel := context.WithTimeout(parent, p.plan.GracefulStopTimeout)
		req, _ := http.NewRequestWithContext(ctx, http.MethodPost, p.endpoint+p.plan.GracefulShutdownPath, nil)
		req.Header.Set("X-Fak-Ownership-Token", p.identity.Token)
		resp, err := loopbackClient(p.plan.GracefulStopTimeout).Do(req)
		if err == nil {
			_, _ = io.Copy(io.Discard, io.LimitReader(resp.Body, 4<<10))
			_ = resp.Body.Close()
		}
		cancel()
	} else if p.cmd.Process != nil {
		_ = p.cmd.Process.Signal(os.Interrupt)
	}
	if waitForDone(parent, p.done, p.plan.GracefulStopTimeout) {
		return receipt, nil
	}
	receipt.Escalated = true
	if p.cmd.Process != nil {
		_ = p.cmd.Process.Kill()
	}
	if !waitForDone(context.Background(), p.done, p.plan.KillTimeout) {
		return receipt, refuse(RefusalStopTimeout, "runtime survived graceful stop and forced termination bounds", nil)
	}
	return receipt, nil
}

func waitForDone(parent context.Context, done <-chan struct{}, timeout time.Duration) bool {
	timer := time.NewTimer(timeout)
	defer timer.Stop()
	select {
	case <-done:
		return true
	case <-parent.Done():
		return false
	case <-timer.C:
		return false
	}
}

func loopbackClient(timeout time.Duration) *http.Client {
	return &http.Client{
		Timeout: timeout,
		CheckRedirect: func(*http.Request, []*http.Request) error {
			// A loopback runtime does not get to turn a readiness or shutdown
			// request into a hidden network mutation through a redirect.
			return http.ErrUseLastResponse
		},
		Transport: &http.Transport{
			Proxy:             nil,
			DialContext:       (&net.Dialer{Timeout: timeout}).DialContext,
			DisableKeepAlives: true,
		},
	}
}

func sameOwnership(a, b Ownership) bool {
	return a.PID == b.PID && a.StartIdentity == b.StartIdentity && len(a.Token) == len(b.Token) && subtle.ConstantTimeCompare([]byte(a.Token), []byte(b.Token)) == 1
}

func randomToken() (string, error) {
	var raw [24]byte
	if _, err := rand.Read(raw[:]); err != nil {
		return "", err
	}
	return hex.EncodeToString(raw[:]), nil
}

func digestArgv(executable string, argv []string) string {
	h := sha256.New()
	_, _ = io.WriteString(h, executable)
	for _, arg := range argv {
		_, _ = h.Write([]byte{0})
		_, _ = io.WriteString(h, arg)
	}
	return hex.EncodeToString(h.Sum(nil))
}

func captureStartIdentity(pid int, launched time.Time) string {
	if started, ok := processalive.StartTime(pid); ok {
		return "kernel:" + started.UTC().Format(time.RFC3339Nano)
	}
	if token, ok := linuxProcessStartToken(pid); ok {
		return "proc:" + token
	}
	return "launch:" + launched.UTC().Format(time.RFC3339Nano)
}

func currentStartIdentity(pid int) (string, bool) {
	if started, ok := processalive.StartTime(pid); ok {
		return "kernel:" + started.UTC().Format(time.RFC3339Nano), true
	}
	if token, ok := linuxProcessStartToken(pid); ok {
		return "proc:" + token, true
	}
	return "", false
}

func linuxProcessStartToken(pid int) (string, bool) {
	if runtime.GOOS != "linux" || pid <= 0 {
		return "", false
	}
	data, err := os.ReadFile(filepath.Join("/proc", strconv.Itoa(pid), "stat"))
	if err != nil {
		return "", false
	}
	end := bytes.LastIndexByte(data, ')')
	if end < 0 {
		return "", false
	}
	fields := strings.Fields(string(data[end+1:]))
	// After the parenthesized command, fields[0] is field 3 (state), making
	// fields[19] the kernel start-time tick (field 22).
	if len(fields) <= 19 {
		return "", false
	}
	return fields[19], true
}

func (p *ownedProcess) waitError() error {
	p.waitMu.Lock()
	defer p.waitMu.Unlock()
	return p.waitErr
}

func refuse(code, detail string, err error) error {
	return &Refusal{Code: code, Detail: detail, Err: err}
}

type boundedBuffer struct {
	mu    sync.Mutex
	limit int
	buf   bytes.Buffer
}

func (b *boundedBuffer) Write(p []byte) (int, error) {
	b.mu.Lock()
	defer b.mu.Unlock()
	want := len(p)
	if b.buf.Len() < b.limit {
		remaining := b.limit - b.buf.Len()
		if len(p) > remaining {
			p = p[:remaining]
		}
		_, _ = b.buf.Write(p)
	}
	return want, nil
}

func (b *boundedBuffer) String() string {
	b.mu.Lock()
	defer b.mu.Unlock()
	return strings.TrimSpace(b.buf.String())
}
