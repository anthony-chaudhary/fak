// Package codexsession projects the typed Codex app-server protocol into fak's
// public harness event protocol. It deliberately never interprets terminal text.
package codexsession

import (
	"bufio"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"os/exec"
	"path/filepath"
	"strings"
	"sync"
	"time"

	"github.com/anthony-chaudhary/fak/internal/harnessprotocol"
	"github.com/anthony-chaudhary/fak/internal/sessionctl"
	"github.com/anthony-chaudhary/fak/pkg/harnesskit"
)

// Engine identifies the execution engine descriptor used across harness runs.
const Engine = "codex"

// Config specifies runtime options, workspace root paths, event sink callbacks,
// and compatibility constraints for launching a Codex session adapter.
type Config struct {
	Command          string
	Args             []string
	Workspace        string
	Version          string
	RunID            string
	Sink             func(harnesskit.Envelope) error
	ApprovalPolicy   ApprovalPolicy
	ApprovalJournal  func(ApprovalJournalEntry)
	ApprovalTimeout  time.Duration
	Now              func() time.Time
	Session          *sessionctl.CodexSession
	StartMode        sessionctl.CodexStartMode
	InputLease       string
	Compatibility    *CompatibilityEnvelope
	TestedReceipt    *CompatibilityReceipt
	AuthorityMethods []string
}

// Adapter coordinates Codex app-server child process execution, translating
// bidirectional JSON-RPC notifications into structured fak harness protocol events.
type Adapter struct {
	cfg      Config
	mu       sync.Mutex
	stdin    io.WriteCloser
	threadID string
	turnID   string
	nextID   int64
	writeMu  sync.Mutex
	pending  map[string]pendingApproval
	resolved map[string]struct{}
	inputIDs map[string]struct{}
	epoch    uint64
	emit     func(harnesskit.EventType, string, string, any) error
}

type rpcMessage struct {
	ID     json.RawMessage `json:"id,omitempty"`
	Method string          `json:"method,omitempty"`
	Params json.RawMessage `json:"params,omitempty"`
	Result json.RawMessage `json:"result,omitempty"`
	Error  *struct {
		Code    int    `json:"code"`
		Message string `json:"message"`
	} `json:"error,omitempty"`
}

// New validates configuration parameters and constructs an initialized Codex
// app-server adapter ready for session execution.
// Precondition: cfg.Version must be non-empty and cfg.Sink must be a non-nil envelope dispatcher.
// Postcondition: returns initialized Adapter with normalized workspace root and active timeout bounds.
func New(cfg Config) (*Adapter, error) {
	if cfg.Command == "" {
		cfg.Command = "codex"
	}
	if len(cfg.Args) == 0 {
		cfg.Args = []string{"app-server"}
	}
	if cfg.Version == "" {
		return nil, errors.New("codexsession: Codex version must be recorded")
	}
	if cfg.RunID == "" || cfg.Sink == nil {
		return nil, errors.New("codexsession: run id and sink are required")
	}
	root, err := filepath.Abs(cfg.Workspace)
	if err != nil {
		return nil, err
	}
	cfg.Workspace = filepath.Clean(root)
	if cfg.ApprovalTimeout <= 0 {
		cfg.ApprovalTimeout = 2 * time.Minute
	}
	if cfg.Now == nil {
		cfg.Now = time.Now
	}
	return &Adapter{cfg: cfg}, nil
}

func (a *Adapter) checkCompatibility() error {
	if a.cfg.Compatibility != nil || a.cfg.TestedReceipt != nil {
		if a.cfg.Compatibility == nil || a.cfg.TestedReceipt == nil {
			return errors.New("codexsession: compatibility envelope and tested receipt must be configured together")
		}
		if err := CheckCompatibility(*a.cfg.Compatibility, *a.cfg.TestedReceipt, a.cfg.AuthorityMethods); err != nil {
			return err
		}
	}
	return nil
}

func (a *Adapter) handshake(stdin io.Writer, write func(int64, string, any) error, wait func(int64) (json.RawMessage, error)) error {
	a.nextID = 1
	if err := write(1, "initialize", map[string]any{"clientInfo": map[string]any{"name": "fak", "title": "fak native UI", "version": "1"}}); err != nil {
		return err
	}
	if _, err := wait(1); err != nil {
		return fmt.Errorf("initialize: %w", err)
	}
	return json.NewEncoder(stdin).Encode(map[string]any{"jsonrpc": "2.0", "method": "initialized"})
}

func (a *Adapter) startThread(execution sessionctl.CodexExecution, write func(int64, string, any) error, wait func(int64) (json.RawMessage, error)) (string, error) {
	threadMethod := "thread/start"
	threadParams := map[string]any{"cwd": a.cfg.Workspace, "ephemeral": a.cfg.Session == nil, "approvalPolicy": "untrusted", "sandbox": "workspace-write"}
	if a.cfg.Session != nil && execution.Mode != sessionctl.CodexNew {
		threadMethod = "thread/resume"
		if execution.Mode == sessionctl.CodexFork {
			threadMethod = "thread/fork"
		}
		threadParams["threadId"] = execution.ThreadID
	}
	if err := write(2, threadMethod, threadParams); err != nil {
		return "", err
	}
	raw, err := wait(2)
	if err != nil {
		if a.cfg.Session != nil && execution.Mode != sessionctl.CodexNew {
			reason := sessionctl.CodexThreadIncompatible
			if strings.Contains(strings.ToLower(err.Error()), "not found") || strings.Contains(strings.ToLower(err.Error()), "missing") {
				reason = sessionctl.CodexThreadMissing
			}
			return "", &sessionctl.CodexRecoveryError{Reason: reason, Choices: []sessionctl.CodexStartMode{sessionctl.CodexNew, sessionctl.CodexFork}, Detail: err.Error()}
		}
		return "", fmt.Errorf("%s: %w", threadMethod, err)
	}
	var ts struct {
		Thread struct {
			ID string `json:"id"`
		} `json:"thread"`
	}
	if err = json.Unmarshal(raw, &ts); err != nil || ts.Thread.ID == "" {
		return "", errors.New("thread/start returned no thread id")
	}
	a.mu.Lock()
	a.threadID = ts.Thread.ID
	a.mu.Unlock()
	if a.cfg.Session != nil {
		var err error
		if execution.Mode == sessionctl.CodexFork {
			err = a.cfg.Session.RecordFork(execution.Epoch, execution.ThreadID, ts.Thread.ID)
		} else {
			err = a.cfg.Session.RecordThread(execution.Epoch, ts.Thread.ID)
		}
		if err != nil {
			return "", err
		}
	}
	return ts.Thread.ID, nil
}

func (a *Adapter) startTurn(threadID string, text string, write func(int64, string, any) error, wait func(int64) (json.RawMessage, error)) error {
	if err := write(3, "turn/start", map[string]any{"threadId": threadID, "cwd": a.cfg.Workspace, "input": []any{map[string]any{"type": "text", "text": text, "textElements": []any{}}}}); err != nil {
		return err
	}
	raw, err := wait(3)
	if err != nil {
		return fmt.Errorf("turn/start: %w", err)
	}
	var tr struct {
		Turn struct {
			ID string `json:"id"`
		} `json:"turn"`
	}
	_ = json.Unmarshal(raw, &tr)
	a.mu.Lock()
	a.turnID = tr.Turn.ID
	a.mu.Unlock()
	return nil
}

// Run starts the Codex app-server subprocess, completes initial protocol handshakes,
// begins a turn, and streams projected harness events until turn completion.
// Precondition: ctx must be active and text prompt must contain the input instruction for the turn.
// Postcondition: turn completes or returns typed execution error, streaming projected harness events.
func (a *Adapter) Run(ctx context.Context, text string) error {
	if err := a.checkCompatibility(); err != nil {
		return err
	}
	var execution sessionctl.CodexExecution
	if a.cfg.Session != nil {
		mode := a.cfg.StartMode
		if mode == "" {
			mode = sessionctl.CodexNew
		}
		var err error
		execution, err = a.cfg.Session.Begin(mode, a.cfg.InputLease)
		if err != nil {
			return err
		}
		defer a.cfg.Session.Release(a.cfg.InputLease)
	}
	cmd := exec.CommandContext(ctx, a.cfg.Command, a.cfg.Args...)
	cmd.Dir = a.cfg.Workspace
	stdout, err := cmd.StdoutPipe()
	if err != nil {
		return err
	}
	stderr, err := cmd.StderrPipe()
	if err != nil {
		return err
	}
	stdin, err := cmd.StdinPipe()
	if err != nil {
		return err
	}
	a.mu.Lock()
	a.stdin = stdin
	a.pending = make(map[string]pendingApproval)
	a.resolved = make(map[string]struct{})
	a.inputIDs = make(map[string]struct{})
	a.epoch++
	a.mu.Unlock()
	if err := cmd.Start(); err != nil {
		return fmt.Errorf("start Codex app-server: %w", err)
	}
	stderrDone := make(chan []byte, 1)
	go func() { b, _ := io.ReadAll(io.LimitReader(stderr, 64<<10)); stderrDone <- b }()
	p := harnessprotocol.NewProducer(a.cfg.RunID, []byte("codexsession-local"))
	emit := func(t harnesskit.EventType, corr, cause string, payload any) error {
		e, err := p.Append(t, corr, cause, harnesskit.SensitivityPrivate, payload)
		if err != nil {
			return err
		}
		if a.cfg.Session != nil {
			_, duplicate, err := a.cfg.Session.Append(execution.Epoch, fmt.Sprintf("%d:%s", execution.Epoch, e.EventID), t == harnesskit.EventMessageDelta, e)
			if err != nil || duplicate {
				return err
			}
		}
		return a.cfg.Sink(e)
	}
	a.emit = emit
	runCause := "codex-session"
	if err := emit(harnesskit.EventRunStarted, a.cfg.RunID, runCause, harnesskit.RunPayload{Status: "running", Reason: fmt.Sprintf("engine=%s transport=stdio:// codex=%s workspace=%s", Engine, a.cfg.Version, a.cfg.Workspace)}); err != nil {
		return err
	}
	scan := bufio.NewScanner(stdout)
	scan.Buffer(make([]byte, 64<<10), 4<<20)
	write := func(id int64, method string, params any) error {
		return json.NewEncoder(stdin).Encode(map[string]any{"jsonrpc": "2.0", "id": id, "method": method, "params": params})
	}
	wait := func(id int64) (json.RawMessage, error) {
		for scan.Scan() {
			var m rpcMessage
			if err := json.Unmarshal(scan.Bytes(), &m); err != nil {
				return nil, fmt.Errorf("decode Codex protocol: %w", err)
			}
			if len(m.ID) > 0 {
				var got int64
				_ = json.Unmarshal(m.ID, &got)
				if got == id {
					if m.Error != nil {
						return nil, fmt.Errorf("Codex RPC %d: %s", m.Error.Code, m.Error.Message)
					}
					return m.Result, nil
				}
			}
			if m.Method != "" {
				if done, err := a.notification(emit, m); err != nil {
					return nil, err
				} else if done {
					return nil, io.EOF
				}
			}
		}
		if err := scan.Err(); err != nil {
			return nil, err
		}
		return nil, io.ErrUnexpectedEOF
	}
	if err := a.handshake(stdin, write, wait); err != nil {
		return err
	}
	threadID, err := a.startThread(execution, write, wait)
	if err != nil {
		return err
	}
	if err := a.startTurn(threadID, text, write, wait); err != nil {
		return err
	}
	for scan.Scan() {
		var m rpcMessage
		if err := json.Unmarshal(scan.Bytes(), &m); err != nil {
			return err
		}
		if len(m.ID) != 0 && strings.Contains(m.Method, "requestApproval") {
			if err := a.handleApprovalRequest(m); err != nil {
				a.failPending("adapter_crash")
				return err
			}
			continue
		}
		if len(m.ID) != 0 { // Additive server requests fail closed and stay visible.
			if err := a.handleApprovalRequest(m); err != nil {
				return err
			}
			continue
		}
		if done, err := a.notification(emit, m); err != nil {
			return err
		} else if done {
			_ = stdin.Close()
			return cmd.Wait()
		}
	}
	waitErr := cmd.Wait()
	a.failPending("disconnect")
	serr := <-stderrDone
	if ctx.Err() != nil {
		return ctx.Err()
	}
	if waitErr != nil {
		return fmt.Errorf("Codex app-server exited: %w: %s", waitErr, string(serr))
	}
	return io.ErrUnexpectedEOF
}

func (a *Adapter) notification(emit func(harnesskit.EventType, string, string, any) error, m rpcMessage) (bool, error) {
	var p struct {
		ThreadID string `json:"threadId"`
		TurnID   string `json:"turnId"`
		ItemID   string `json:"itemId"`
		Delta    string `json:"delta"`
		Item     struct {
			ID               string `json:"id"`
			Type             string `json:"type"`
			Command          string `json:"command"`
			Status           string `json:"status"`
			AggregatedOutput string `json:"aggregatedOutput"`
		} `json:"item"`
		Turn struct {
			ID     string `json:"id"`
			Status string `json:"status"`
			Error  *struct {
				Message string `json:"message"`
			} `json:"error"`
		} `json:"turn"`
		TokenUsage struct {
			InputTokens  int64 `json:"inputTokens"`
			OutputTokens int64 `json:"outputTokens"`
		} `json:"tokenUsage"`
	}
	if err := json.Unmarshal(m.Params, &p); err != nil {
		return false, err
	}
	cause := p.TurnID
	switch m.Method {
	case "item/agentMessage/delta":
		return false, emit(harnesskit.EventMessageDelta, p.ItemID, cause, harnesskit.MessagePayload{MessageID: p.ItemID, Role: "assistant", Text: p.Delta})
	case "item/started":
		if p.Item.Type == "agentMessage" {
			return false, emit(harnesskit.EventMessageStarted, p.Item.ID, cause, harnesskit.MessagePayload{MessageID: p.Item.ID, Role: "assistant"})
		}
		if p.Item.Type == "commandExecution" {
			return false, emit(harnesskit.EventToolStarted, p.Item.ID, cause, harnesskit.ToolPayload{CallID: p.Item.ID, Name: "command", Status: p.Item.Status, Summary: p.Item.Command})
		}
	case "item/completed":
		if p.Item.Type == "agentMessage" {
			return false, emit(harnesskit.EventMessageCompleted, p.Item.ID, cause, harnesskit.MessagePayload{MessageID: p.Item.ID, Role: "assistant"})
		}
		if p.Item.Type == "commandExecution" {
			return false, emit(harnesskit.EventToolCompleted, p.Item.ID, cause, harnesskit.ToolPayload{CallID: p.Item.ID, Name: "command", Status: p.Item.Status, Summary: p.Item.AggregatedOutput})
		}
	case "thread/tokenUsage/updated":
		return false, emit(harnesskit.EventUsage, p.ThreadID, cause, harnesskit.UsagePayload{InputTokens: p.TokenUsage.InputTokens, OutputTokens: p.TokenUsage.OutputTokens})
	case "turn/completed":
		if p.Turn.Status == "failed" {
			msg := "Codex turn failed"
			if p.Turn.Error != nil {
				msg = p.Turn.Error.Message
			}
			if err := emit(harnesskit.EventError, p.Turn.ID, p.Turn.ID, harnesskit.ErrorPayload{Code: "CODEX_TURN_FAILED", Message: msg}); err != nil {
				return false, err
			}
		}
		return true, emit(harnesskit.EventRunCompleted, a.cfg.RunID, p.Turn.ID, harnesskit.RunPayload{Status: p.Turn.Status, Reason: "engine=codex"})
	}
	return false, nil
}

// Interrupt transmits a cooperative cancellation request for the active turn
// through the underlying JSON-RPC protocol stream.
// Precondition: adapter must have an active child process, thread identifier, and turn identifier.
// Postcondition: sends cooperative turn interrupt frame over stdio RPC channel without killing process.
func (a *Adapter) Interrupt() error {
	a.mu.Lock()
	defer a.mu.Unlock()
	if a.stdin == nil || a.threadID == "" || a.turnID == "" {
		return errors.New("codexsession: no active turn")
	}
	a.nextID++
	return json.NewEncoder(a.stdin).Encode(map[string]any{"jsonrpc": "2.0", "id": 1000 + a.nextID, "method": "turn/interrupt", "params": map[string]any{"threadId": a.threadID, "turnId": a.turnID}})
}

func (a *Adapter) now() time.Time { return a.cfg.Now() }
