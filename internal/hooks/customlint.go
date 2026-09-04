package hooks

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"os"
	"os/exec"
	"runtime"
	"strings"
	"time"
)

const CustomLintSchema = "fak-custom-lint/1"

type LintDisposition string

const (
	LintAllow    LintDisposition = "allow"
	LintDeny     LintDisposition = "deny"
	LintAdvisory LintDisposition = "advisory"
)

type LintErrorMode string

const (
	LintErrorAllow LintErrorMode = "open"
	LintErrorDeny  LintErrorMode = "closed"
)

type LintRequest struct {
	Schema  string          `json:"schema"`
	Hook    string          `json:"hook"`
	Subject json.RawMessage `json:"subject"`
}

type LintLocation struct {
	Path   string `json:"path,omitempty"`
	Line   int    `json:"line,omitempty"`
	Column int    `json:"column,omitempty"`
}

type LintFinding struct {
	ID       string       `json:"id"`
	Severity string       `json:"severity"`
	Message  string       `json:"message"`
	Location LintLocation `json:"location,omitempty"`
	Evidence string       `json:"evidence,omitempty"`
}

type LintResponse struct {
	Schema      string          `json:"schema"`
	Disposition LintDisposition `json:"disposition"`
	Findings    []LintFinding   `json:"findings,omitempty"`
}

type CustomLintLimits struct {
	Timeout        time.Duration
	MaxInputBytes  int
	MaxOutputBytes int
	MaxStderrBytes int
	MaxFindings    int
}

type CustomLintSpec struct {
	Name    string
	Command []string
	OnError LintErrorMode
	Limits  CustomLintLimits
	Dir     string
	Env     []string
}

type CustomLintResult struct {
	Response  LintResponse
	ErrorKind string
	Error     string
	Stderr    string
	Duration  time.Duration
}

func (r CustomLintResult) Allowed() bool { return r.Response.Disposition != LintDeny }

func DefaultCustomLintLimits() CustomLintLimits {
	return CustomLintLimits{Timeout: 5 * time.Second, MaxInputBytes: 1 << 20, MaxOutputBytes: 1 << 20, MaxStderrBytes: 64 << 10, MaxFindings: 256}
}

func RunCustomLint(ctx context.Context, spec CustomLintSpec, req LintRequest) (result CustomLintResult) {
	started := time.Now()
	defer func() { result.Duration = time.Since(started) }()
	if err := validateLintSpec(spec); err != nil {
		return lintFailure(spec, result, "invalid_spec", err)
	}
	if req.Schema == "" {
		req.Schema = CustomLintSchema
	}
	if req.Schema != CustomLintSchema || strings.TrimSpace(req.Hook) == "" || !json.Valid(req.Subject) {
		return lintFailure(spec, result, "invalid_request", errors.New("request requires schema fak-custom-lint/1, hook, and valid JSON subject"))
	}
	payload, err := json.Marshal(req)
	if err != nil {
		return lintFailure(spec, result, "invalid_request", err)
	}
	limits := normalizedLintLimits(spec.Limits)
	if len(payload) > limits.MaxInputBytes {
		return lintFailure(spec, result, "input_overflow", fmt.Errorf("request is %d bytes (limit %d)", len(payload), limits.MaxInputBytes))
	}

	runCtx, cancel := context.WithTimeout(ctx, limits.Timeout)
	defer cancel()
	cmd := exec.CommandContext(runCtx, spec.Command[0], spec.Command[1:]...)
	cmd.WaitDelay = 250 * time.Millisecond
	cmd.Dir = spec.Dir
	cmd.Env = lintEnvironment(spec.Env)
	cmd.Stdin = bytes.NewReader(append(payload, '\n'))
	stdout := &boundedBuffer{limit: limits.MaxOutputBytes}
	stderr := &boundedBuffer{limit: limits.MaxStderrBytes}
	cmd.Stdout, cmd.Stderr = stdout, stderr
	err = cmd.Run()
	result.Stderr = stderr.String()
	if runCtx.Err() == context.DeadlineExceeded {
		return lintFailure(spec, result, "timeout", fmt.Errorf("linter exceeded %s", limits.Timeout))
	}
	if stdout.overflow || stderr.overflow {
		return lintFailure(spec, result, "output_overflow", fmt.Errorf("linter output exceeded stdout/stderr limit"))
	}
	if err != nil {
		return lintFailure(spec, result, "crash", err)
	}
	dec := json.NewDecoder(bytes.NewReader(stdout.Bytes()))
	dec.DisallowUnknownFields()
	if err := dec.Decode(&result.Response); err != nil {
		return lintFailure(spec, result, "malformed_output", err)
	}
	if err := ensureJSONEOF(dec); err != nil {
		return lintFailure(spec, result, "malformed_output", err)
	}
	if err := validateLintResponse(result.Response, limits.MaxFindings); err != nil {
		return lintFailure(spec, result, "invalid_response", err)
	}
	return result
}

func validateLintSpec(spec CustomLintSpec) error {
	if strings.TrimSpace(spec.Name) == "" || len(spec.Command) == 0 || strings.TrimSpace(spec.Command[0]) == "" {
		return errors.New("name and command are required")
	}
	if spec.OnError != LintErrorAllow && spec.OnError != LintErrorDeny {
		return errors.New("failure policy must be open or closed")
	}
	return nil
}

func normalizedLintLimits(in CustomLintLimits) CustomLintLimits {
	d := DefaultCustomLintLimits()
	if in.Timeout > 0 {
		d.Timeout = in.Timeout
	}
	if in.MaxInputBytes > 0 {
		d.MaxInputBytes = in.MaxInputBytes
	}
	if in.MaxOutputBytes > 0 {
		d.MaxOutputBytes = in.MaxOutputBytes
	}
	if in.MaxStderrBytes > 0 {
		d.MaxStderrBytes = in.MaxStderrBytes
	}
	if in.MaxFindings > 0 {
		d.MaxFindings = in.MaxFindings
	}
	return d
}

func validateLintResponse(resp LintResponse, maxFindings int) error {
	if resp.Schema != CustomLintSchema {
		return fmt.Errorf("response schema %q is not %q", resp.Schema, CustomLintSchema)
	}
	if resp.Disposition != LintAllow && resp.Disposition != LintDeny && resp.Disposition != LintAdvisory {
		return fmt.Errorf("unknown disposition %q", resp.Disposition)
	}
	if len(resp.Findings) > maxFindings {
		return fmt.Errorf("response has %d findings (limit %d)", len(resp.Findings), maxFindings)
	}
	for i, f := range resp.Findings {
		if strings.TrimSpace(f.ID) == "" || strings.TrimSpace(f.Severity) == "" || strings.TrimSpace(f.Message) == "" {
			return fmt.Errorf("finding %d requires id, severity, and message", i)
		}
		if f.Location.Line < 0 || f.Location.Column < 0 {
			return fmt.Errorf("finding %d has a negative location", i)
		}
	}
	return nil
}

func lintFailure(spec CustomLintSpec, r CustomLintResult, kind string, err error) CustomLintResult {
	r.ErrorKind, r.Error = kind, err.Error()
	r.Response = LintResponse{Schema: CustomLintSchema, Disposition: LintAdvisory, Findings: []LintFinding{{ID: "fak.custom-lint." + kind, Severity: "error", Message: spec.Name + ": " + err.Error()}}}
	if spec.OnError == LintErrorDeny {
		r.Response.Disposition = LintDeny
	}
	return r
}

func lintEnvironment(extra []string) []string {
	keys := []string{"PATH", "SystemRoot", "SYSTEMROOT", "ComSpec", "COMSPEC", "TMP", "TEMP", "TMPDIR", "HOME", "USERPROFILE"}
	out := make([]string, 0, len(keys)+len(extra)+1)
	for _, key := range keys {
		if value, ok := os.LookupEnv(key); ok {
			out = append(out, key+"="+value)
		}
	}
	out = append(out, "FAK_CUSTOM_LINT_SCHEMA="+CustomLintSchema)
	out = append(out, extra...)
	if runtime.GOOS == "windows" && !hasEnvKey(out, "SystemRoot") {
		if root := os.Getenv("windir"); root != "" {
			out = append(out, "SystemRoot="+root)
		}
	}
	return out
}

func hasEnvKey(env []string, key string) bool {
	for _, item := range env {
		if name, _, ok := strings.Cut(item, "="); ok && strings.EqualFold(name, key) {
			return true
		}
	}
	return false
}

func ensureJSONEOF(dec *json.Decoder) error {
	var extra any
	if err := dec.Decode(&extra); err == io.EOF {
		return nil
	}
	return errors.New("response must contain exactly one JSON value")
}

type boundedBuffer struct {
	buf      bytes.Buffer
	limit    int
	overflow bool
}

func (b *boundedBuffer) Write(p []byte) (int, error) {
	if b.limit <= 0 {
		b.overflow = true
		return len(p), nil
	}
	remaining := b.limit - b.buf.Len()
	if remaining > 0 {
		n := len(p)
		if n > remaining {
			n = remaining
		}
		_, _ = b.buf.Write(p[:n])
	}
	if len(p) > remaining {
		b.overflow = true
	}
	return len(p), nil
}
func (b *boundedBuffer) Bytes() []byte  { return b.buf.Bytes() }
func (b *boundedBuffer) String() string { return b.buf.String() }
