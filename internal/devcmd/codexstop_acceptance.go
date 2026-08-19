package devcmd

import (
	"bufio"
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"time"
)

const codexStopAcceptanceSchema = "fak/codex-stop-acceptance/v1"

type stopAcceptanceReport struct {
	Schema      string             `json:"schema"`
	GeneratedAt time.Time          `json:"generated_at"`
	CodexHome   string             `json:"codex_home"`
	Workspace   string             `json:"workspace"`
	CodexBinary string             `json:"codex_binary"`
	ThreadID    string             `json:"thread_id,omitempty"`
	TurnID      string             `json:"turn_id,omitempty"`
	Expectation string             `json:"expectation"`
	Runs        []stopLifecycleRow `json:"runs,omitempty"`
	Stop        lifecycleCounts    `json:"stop"`
	Verdict     string             `json:"verdict"`
	Reasons     []string           `json:"reasons,omitempty"`
	Detail      string             `json:"detail,omitempty"`
}

type appServerMessage struct {
	ID     int             `json:"id,omitempty"`
	Method string          `json:"method,omitempty"`
	Result json.RawMessage `json:"result,omitempty"`
	Error  *appServerError `json:"error,omitempty"`
	Params json.RawMessage `json:"params,omitempty"`
}

type appServerError struct {
	Code    int    `json:"code"`
	Message string `json:"message"`
}

type appServerTransport interface {
	Send(any) error
	Receive(context.Context) (appServerMessage, error)
	Close() error
}

type processAppServer struct {
	cmd      *exec.Cmd
	in       io.WriteCloser
	messages chan appServerMessage
	errors   chan error
}

func RunCodexStopAcceptance(stdout, stderr io.Writer, args []string) int {
	fs := flag.NewFlagSet("codex-stop-acceptance", flag.ContinueOnError)
	fs.SetOutput(io.Discard)
	home := fs.String("codex-home", os.Getenv("CODEX_HOME"), "active Codex home")
	workspace := fs.String("workspace", ".", "workspace used for the acceptance turn")
	binary := fs.String("codex-bin", "", "Codex executable (auto-detected by default)")
	prompt := fs.String("prompt", "Reply with exactly CODEX_STOP_ACCEPTANCE_OK. Do not call tools.", "acceptance turn prompt")
	expect := fs.String("expect", "completed", "expected Stop result: completed or blocked")
	timeout := fs.Duration("timeout", 2*time.Minute, "app-server acceptance timeout")
	jsonOut := fs.Bool("json", false, "emit JSON")
	if fs.Parse(args) != nil || fs.NArg() != 0 || (*expect != "completed" && *expect != "blocked") || *timeout <= 0 {
		fmt.Fprintln(stderr, "usage: fak-dev codex-stop-acceptance [--codex-home DIR] [--workspace DIR] [--expect completed|blocked] [--prompt TEXT] [--timeout 2m] [--json]")
		return 2
	}
	if *home == "" {
		h, err := os.UserHomeDir()
		if err != nil {
			return 1
		}
		*home = filepath.Join(h, ".codex")
	}
	if *binary == "" {
		*binary = discoverCodexBinary()
	}
	if *binary == "" {
		fmt.Fprintln(stderr, "codex-stop-acceptance: Codex executable not found")
		return 1
	}
	absHome, _ := filepath.Abs(*home)
	absWork, _ := filepath.Abs(*workspace)
	ctx, cancel := context.WithTimeout(context.Background(), *timeout)
	defer cancel()
	transport, err := startProcessAppServer(ctx, *binary, absHome)
	if err != nil {
		fmt.Fprintf(stderr, "codex-stop-acceptance: %v\n", err)
		return 1
	}
	defer transport.Close()
	r := runStopAcceptance(ctx, transport, absHome, absWork, *binary, *prompt, *expect)
	if *jsonOut {
		enc := json.NewEncoder(stdout)
		enc.SetIndent("", "  ")
		_ = enc.Encode(r)
	} else {
		writeStopAcceptance(stdout, r)
	}
	if r.Verdict != "PASS" {
		return 1
	}
	return 0
}

func startProcessAppServer(ctx context.Context, binary, home string) (*processAppServer, error) {
	cmd := exec.CommandContext(ctx, binary, "app-server", "--stdio")
	configureDispatchHelperCommand(cmd)
	cmd.Env = append(os.Environ(), "CODEX_HOME="+home)
	in, err := cmd.StdinPipe()
	if err != nil {
		return nil, err
	}
	out, err := cmd.StdoutPipe()
	if err != nil {
		return nil, err
	}
	var errbuf strings.Builder
	cmd.Stderr = &errbuf
	if err := cmd.Start(); err != nil {
		return nil, fmt.Errorf("start app-server: %w", err)
	}
	p := &processAppServer{cmd: cmd, in: in, messages: make(chan appServerMessage, 32), errors: make(chan error, 1)}
	go func() {
		s := bufio.NewScanner(out)
		s.Buffer(make([]byte, 64*1024), 4*1024*1024)
		for s.Scan() {
			var m appServerMessage
			if json.Unmarshal(s.Bytes(), &m) == nil {
				p.messages <- m
			}
		}
		if err := s.Err(); err != nil {
			p.errors <- err
		} else {
			p.errors <- fmt.Errorf("app-server closed stdout: %s", strings.TrimSpace(errbuf.String()))
		}
	}()
	return p, nil
}

func (p *processAppServer) Send(v any) error { return json.NewEncoder(p.in).Encode(v) }
func (p *processAppServer) Receive(ctx context.Context) (appServerMessage, error) {
	select {
	case m := <-p.messages:
		return m, nil
	case err := <-p.errors:
		return appServerMessage{}, err
	case <-ctx.Done():
		return appServerMessage{}, ctx.Err()
	}
}
func (p *processAppServer) Close() error {
	_ = p.in.Close()
	if p.cmd.Process != nil {
		_ = p.cmd.Process.Kill()
	}
	return nil
}

func runStopAcceptance(ctx context.Context, t appServerTransport, home, workspace, binary, prompt, expect string) stopAcceptanceReport {
	r := stopAcceptanceReport{Schema: codexStopAcceptanceSchema, GeneratedAt: time.Now().UTC(), CodexHome: home, Workspace: workspace, CodexBinary: binary, Expectation: expect, Verdict: "FAIL"}
	send := func(v any) bool {
		if err := t.Send(v); err != nil {
			r.Detail = err.Error()
			r.Reasons = append(r.Reasons, "PROTOCOL_SEND_FAILED")
			return false
		}
		return true
	}
	if !send(map[string]any{"method": "initialize", "id": 1, "params": map[string]any{"clientInfo": map[string]string{"name": "fak_stop_acceptance", "title": "FAK Stop acceptance", "version": "1"}, "capabilities": map[string]bool{"experimentalApi": true}}}) {
		return r
	}
	if !send(map[string]any{"method": "initialized", "params": map[string]any{}}) {
		return r
	}
	if !send(map[string]any{"method": "thread/start", "id": 2, "params": map[string]any{"cwd": workspace, "ephemeral": true}}) {
		return r
	}
	for r.ThreadID == "" {
		m, err := t.Receive(ctx)
		if err != nil {
			return acceptanceReceiveFailure(r, err)
		}
		if m.ID != 2 {
			continue
		}
		if m.Error != nil {
			r.Detail = m.Error.Message
			r.Reasons = append(r.Reasons, "THREAD_START_FAILED")
			return r
		}
		var result struct {
			Thread struct {
				ID string `json:"id"`
			} `json:"thread"`
			ThreadID string `json:"threadId"`
		}
		if json.Unmarshal(m.Result, &result) != nil {
			r.Reasons = append(r.Reasons, "THREAD_START_INVALID_JSON")
			return r
		}
		r.ThreadID = result.Thread.ID
		if r.ThreadID == "" {
			r.ThreadID = result.ThreadID
		}
	}
	if !send(map[string]any{"method": "turn/start", "id": 3, "params": map[string]any{"threadId": r.ThreadID, "input": []map[string]string{{"type": "text", "text": prompt}}}}) {
		return r
	}
	started := map[string]hookNotification{}
	completed := map[string]hookNotification{}
	turnDone := false
	for !turnDone {
		m, err := t.Receive(ctx)
		if err != nil {
			return acceptanceReceiveFailure(r, err)
		}
		if m.ID == 3 {
			if m.Error != nil {
				r.Detail = m.Error.Message
				r.Reasons = append(r.Reasons, "TURN_START_FAILED")
				return r
			}
			var x struct {
				Turn struct {
					ID string `json:"id"`
				} `json:"turn"`
				TurnID string `json:"turnId"`
			}
			_ = json.Unmarshal(m.Result, &x)
			r.TurnID = x.Turn.ID
			if r.TurnID == "" {
				r.TurnID = x.TurnID
			}
			continue
		}
		if m.Method == "turn/completed" {
			turnDone = true
			continue
		}
		if m.Method != "hook/started" && m.Method != "hook/completed" {
			continue
		}
		var n hookNotification
		n.Method = m.Method
		if json.Unmarshal(m.Params, &n.Params) != nil {
			r.Stop.InvalidJSON++
			continue
		}
		if n.Params.ThreadID != r.ThreadID || n.Params.Run.EventName != "stop" {
			continue
		}
		key := stopRunKey(n)
		if m.Method == "hook/started" {
			started[key] = n
		} else {
			completed[key] = n
		}
	}
	for key, n := range started {
		if _, ok := completed[key]; !ok {
			r.Stop.Denominator++
			r.Stop.Attempted++
			r.Stop.Unknown++
			r.Runs = append(r.Runs, stopRow(n, "unknown"))
		}
	}
	for _, n := range completed {
		r.Stop.Denominator++
		r.Stop.Attempted++
		status := strings.ToLower(n.Params.Run.Status)
		switch status {
		case "completed":
			r.Stop.Succeeded++
		case "blocked":
			r.Stop.Blocked++
		case "failed":
			if invalidHookOutput(n) {
				r.Stop.InvalidJSON++
			} else {
				r.Stop.Failed++
			}
		case "stopped":
			r.Stop.Skipped++
		default:
			r.Stop.Unknown++
		}
		r.Runs = append(r.Runs, stopRow(n, status))
	}
	if r.Stop.Denominator == 0 {
		r.Reasons = append(r.Reasons, "STOP_UNOBSERVED")
	}
	if r.Stop.InvalidJSON > 0 {
		r.Reasons = append(r.Reasons, "STOP_INVALID_JSON")
	}
	if r.Stop.Failed > 0 {
		r.Reasons = append(r.Reasons, "STOP_FAILED")
	}
	if r.Stop.Unknown > 0 {
		r.Reasons = append(r.Reasons, "STOP_UNKNOWN")
	}
	if expect == "completed" && r.Stop.Succeeded != r.Stop.Denominator {
		r.Reasons = append(r.Reasons, "COMPLETION_EXPECTATION_UNMET")
	}
	if expect == "blocked" && r.Stop.Blocked == 0 {
		r.Reasons = append(r.Reasons, "BLOCK_EXPECTATION_UNMET")
	}
	if len(r.Reasons) == 0 {
		r.Verdict = "PASS"
	}
	return r
}

func acceptanceReceiveFailure(r stopAcceptanceReport, err error) stopAcceptanceReport {
	r.Detail = err.Error()
	if err == context.DeadlineExceeded {
		r.Reasons = append(r.Reasons, "TIMEOUT")
	} else {
		r.Reasons = append(r.Reasons, "PROTOCOL_RECEIVE_FAILED")
	}
	return r
}
func writeStopAcceptance(w io.Writer, r stopAcceptanceReport) {
	fmt.Fprintf(w, "Codex Stop acceptance: %s\nthread=%s turn=%s expectation=%s stop=%d succeeded=%d blocked=%d failed=%d invalid-json=%d unknown=%d\n", r.Verdict, r.ThreadID, r.TurnID, r.Expectation, r.Stop.Denominator, r.Stop.Succeeded, r.Stop.Blocked, r.Stop.Failed, r.Stop.InvalidJSON, r.Stop.Unknown)
	if len(r.Reasons) > 0 {
		fmt.Fprintf(w, "reasons: %s\n", strings.Join(r.Reasons, ", "))
	}
	if r.Detail != "" {
		fmt.Fprintf(w, "detail: %s\n", r.Detail)
	}
}
