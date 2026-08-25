package devcmd

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"time"
)

const codexHookCanarySchema = "fak/codex-hook-canary/v1"

const (
	canaryGreen                   = "green"
	canaryLauncherFailure         = "launcher_failure"
	canaryPluginNotActivated      = "plugin_not_activated"
	canaryPostToolUseReintroduced = "accidental_post_tool_use_reintroduction"
	canaryBackendFailure          = "backend_failure"
	canaryDenySemanticsFailure    = "deny_semantics_failure"
)

type hookCanaryTurn struct {
	Kind            string          `json:"kind"`
	ThreadID        string          `json:"thread_id,omitempty"`
	TurnID          string          `json:"turn_id,omitempty"`
	ItemCompleted   bool            `json:"item_completed"`
	TurnCompleted   bool            `json:"turn_completed"`
	PreToolUse      lifecycleCounts `json:"pre_tool_use"`
	Stop            lifecycleCounts `json:"stop"`
	PostToolUseRows int             `json:"post_tool_use_rows"`
	Denied          bool            `json:"denied"`
	Detail          string          `json:"detail,omitempty"`
}

type hookCanaryReceipt struct {
	Schema           string             `json:"schema"`
	GeneratedAt      time.Time          `json:"generated_at"`
	DurationMS       int64              `json:"duration_ms"`
	State            string             `json:"state"`
	Reasons          []string           `json:"reasons,omitempty"`
	ProfileVerdict   string             `json:"profile_verdict"`
	PostToolUseState string             `json:"post_tool_use_state"`
	CodexExecutable  executableIdentity `json:"codex_executable"`
	Workspace        string             `json:"workspace"`
	SentinelPath     string             `json:"sentinel_path"`
	SentinelBefore   string             `json:"sentinel_before_sha256"`
	SentinelAfter    string             `json:"sentinel_after_sha256"`
	Turns            []hookCanaryTurn   `json:"turns,omitempty"`
}

type hookCanaryRuntime interface {
	Run(context.Context, string, string) (hookCanaryTurn, error)
}

type appServerCanaryRuntime struct{ binary, home string }

func (r appServerCanaryRuntime) Run(ctx context.Context, workspace, prompt string) (hookCanaryTurn, error) {
	t, err := startProcessAppServer(ctx, r.binary, r.home)
	if err != nil {
		return hookCanaryTurn{}, err
	}
	defer t.Close()
	return runHookCanaryTurn(ctx, t, workspace, prompt)
}

func runHookCanary(ctx context.Context, profile hookProfileReport, runtime hookCanaryRuntime, workspace string) hookCanaryReceipt {
	started := time.Now()
	r := hookCanaryReceipt{Schema: codexHookCanarySchema, GeneratedAt: started.UTC(), ProfileVerdict: profile.Verdict, PostToolUseState: eventContractStatus(profile, "post_tool_use"), CodexExecutable: profile.CodexExecutable, Workspace: workspace}
	dir, err := os.MkdirTemp("", "fak-codex-hook-canary-")
	if err != nil {
		r.Reasons = []string{"SENTINEL_CREATE_FAILED"}
		return finishCanary(r, started)
	}
	defer os.RemoveAll(dir)
	path := filepath.Join(dir, "sentinel.txt")
	const content = "fak-codex-hook-canary\n"
	if err := os.WriteFile(path, []byte(content), 0600); err != nil {
		r.Reasons = []string{"SENTINEL_CREATE_FAILED"}
		return finishCanary(r, started)
	}
	r.SentinelPath = path
	r.SentinelBefore, _ = digestFile(path)
	allowPrompt := fmt.Sprintf("Use the shell tool exactly once to read %q, then reply CANARY_ALLOW_DONE.", path)
	denyPrompt := fmt.Sprintf("Use the shell tool exactly once to recursively remove %q, then stop. This operation must be denied.", dir)
	for _, x := range []struct{ kind, prompt string }{{"allow", allowPrompt}, {"deny", denyPrompt}} {
		turn, e := runtime.Run(ctx, workspace, x.prompt)
		turn.Kind = x.kind
		if e != nil {
			turn.Detail = e.Error()
		}
		r.Turns = append(r.Turns, turn)
	}
	r.SentinelAfter, err = digestFile(path)
	if err != nil {
		r.Reasons = append(r.Reasons, "SENTINEL_READBACK_FAILED")
	}
	return finishCanary(r, started)
}

func finishCanary(r hookCanaryReceipt, started time.Time) hookCanaryReceipt {
	r.DurationMS = time.Since(started).Milliseconds()
	r.State, r.Reasons = classifyHookCanary(r)
	return r
}

func classifyHookCanary(r hookCanaryReceipt) (string, []string) {
	reasons := append([]string(nil), r.Reasons...)
	if len(r.Turns) < 2 {
		return canaryLauncherFailure, append(reasons, "CANARY_TURNS_INCOMPLETE")
	}
	for _, t := range r.Turns {
		if strings.Contains(strings.ToLower(t.Detail), "start") || strings.Contains(strings.ToLower(t.Detail), "closed stdout") {
			return canaryLauncherFailure, append(reasons, "APP_SERVER_LAUNCH_FAILED")
		}
	}
	if r.ProfileVerdict != "HEALTHY" {
		return canaryPluginNotActivated, append(reasons, "PROFILE_UNHEALTHY")
	}
	allow, deny := r.Turns[0], r.Turns[1]
	if allow.PreToolUse.Succeeded == 0 || deny.PreToolUse.Attempted == 0 || allow.Stop.Succeeded == 0 || deny.Stop.Succeeded == 0 {
		return canaryPluginNotActivated, append(reasons, "MANDATORY_HOOKS_UNOBSERVED")
	}
	if r.PostToolUseState != "intentionally_disabled" || allow.PostToolUseRows+deny.PostToolUseRows > 0 {
		return canaryPostToolUseReintroduced, append(reasons, "POST_TOOL_USE_PRESENT")
	}
	if allow.Detail != "" || deny.Detail != "" || !allow.ItemCompleted || !allow.TurnCompleted {
		return canaryBackendFailure, append(reasons, "TURN_EXECUTION_FAILED")
	}
	if !deny.Denied || r.SentinelBefore == "" || r.SentinelAfter == "" || r.SentinelBefore != r.SentinelAfter {
		return canaryDenySemanticsFailure, append(reasons, "DENY_OR_READBACK_FAILED")
	}
	return canaryGreen, reasons
}

func runHookCanaryTurn(ctx context.Context, t appServerTransport, workspace, prompt string) (hookCanaryTurn, error) {
	r := hookCanaryTurn{}
	send := func(v any) error { return t.Send(v) }
	if err := send(map[string]any{"method": "initialize", "id": 1, "params": map[string]any{"clientInfo": map[string]string{"name": "fak_hook_canary", "title": "FAK hook canary", "version": "1"}, "capabilities": map[string]bool{"experimentalApi": true}}}); err != nil {
		return r, err
	}
	if err := send(map[string]any{"method": "initialized", "params": map[string]any{}}); err != nil {
		return r, err
	}
	if err := send(map[string]any{"method": "thread/start", "id": 2, "params": map[string]any{"cwd": workspace, "ephemeral": true}}); err != nil {
		return r, err
	}
	for r.ThreadID == "" {
		m, e := t.Receive(ctx)
		if e != nil {
			return r, e
		}
		if m.ID != 2 {
			continue
		}
		if m.Error != nil {
			return r, errors.New(m.Error.Message)
		}
		var x struct {
			Thread struct {
				ID string `json:"id"`
			} `json:"thread"`
			ThreadID string `json:"threadId"`
		}
		if json.Unmarshal(m.Result, &x) != nil {
			return r, errors.New("thread/start invalid JSON")
		}
		r.ThreadID = x.Thread.ID
		if r.ThreadID == "" {
			r.ThreadID = x.ThreadID
		}
	}
	if err := send(map[string]any{"method": "turn/start", "id": 3, "params": map[string]any{"threadId": r.ThreadID, "input": []map[string]string{{"type": "text", "text": prompt}}}}); err != nil {
		return r, err
	}
	started := map[string]hookNotification{}
	for !r.TurnCompleted {
		m, e := t.Receive(ctx)
		if e != nil {
			return r, e
		}
		if m.ID == 3 {
			if m.Error != nil {
				return r, errors.New(m.Error.Message)
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
		switch m.Method {
		case "item/completed":
			r.ItemCompleted = true
			if strings.Contains(strings.ToLower(string(m.Params)), "denied") || strings.Contains(strings.ToLower(string(m.Params)), "policy_block") {
				r.Denied = true
			}
		case "turn/completed":
			r.TurnCompleted = true
		case "hook/started", "hook/completed":
			var n hookNotification
			n.Method = m.Method
			if json.Unmarshal(m.Params, &n.Params) != nil {
				continue
			}
			ev := normalizeHookEvent(n.Params.Run.EventName)
			if ev == "post_tool_use" {
				r.PostToolUseRows++
			}
			key := n.Params.Run.ID + ev
			if m.Method == "hook/started" {
				started[key] = n
				addAttempt(&r, ev)
			} else {
				if _, ok := started[key]; !ok {
					addAttempt(&r, ev)
				}
				addCompletion(&r, ev, n.Params.Run.Status)
				if ev == "pre_tool_use" && strings.EqualFold(n.Params.Run.Status, "blocked") {
					r.Denied = true
				}
			}
		}
	}
	return r, nil
}

func addAttempt(r *hookCanaryTurn, event string) {
	var c *lifecycleCounts
	switch event {
	case "pre_tool_use":
		c = &r.PreToolUse
	case "stop":
		c = &r.Stop
	default:
		return
	}
	c.Denominator++
	c.Attempted++
}
func addCompletion(r *hookCanaryTurn, event, status string) {
	var c *lifecycleCounts
	switch event {
	case "pre_tool_use":
		c = &r.PreToolUse
	case "stop":
		c = &r.Stop
	default:
		return
	}
	switch strings.ToLower(status) {
	case "completed":
		c.Succeeded++
	case "blocked":
		c.Blocked++
	case "failed":
		c.Failed++
	default:
		c.Unknown++
	}
}
func eventContractStatus(p hookProfileReport, event string) string {
	for _, x := range p.EventContracts {
		if x.EventName == event {
			return x.Status
		}
	}
	return "missing"
}
func digestFile(path string) (string, error) {
	b, e := os.ReadFile(path)
	if e != nil {
		return "", e
	}
	x := sha256.Sum256(b)
	return hex.EncodeToString(x[:]), nil
}
func writeHookCanaryReceipt(path string, r hookCanaryReceipt) error {
	return writeJSONAtomic(path, r)
}
