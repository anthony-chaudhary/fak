package main

import (
	"bytes"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"sync/atomic"
	"testing"
)

// The real subprocess executes the production chat entry point, including its
// file transport, HTTP planner and kernel tools, rather than a fake child receipt.
func init() {
	if os.Getenv("FAK_OPS_NATIVE_TEST_CHILD") == "1" && len(os.Args) > 1 && os.Args[1] == "chat" {
		cmdChat(os.Args[2:])
		os.Exit(0)
	}
}

func TestOpsNativeRealExecution(t *testing.T) {
	t.Setenv("FAK_OPS_NATIVE_TEST_CHILD", "1")
	for _, mode := range []string{"complete", "turn_cap", "timeout"} {
		t.Run(mode, func(t *testing.T) {
			root := t.TempDir()
			prompt := filepath.Join(root, "prompt.txt")
			artifact := filepath.Join(root, "witness.txt")
			policy := filepath.Join(root, "policy.json")
			receipt := filepath.Join(root, "run.json")
			for path, value := range map[string]string{prompt: "private native task sentinel", policy: `{"allow":["Write"]}`} {
				if err := os.WriteFile(path, []byte(value), 0600); err != nil {
					t.Fatal(err)
				}
			}
			var requests atomic.Int32
			releaseHandler := make(chan struct{})
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				var request map[string]any
				if err := json.NewDecoder(r.Body).Decode(&request); err != nil {
					t.Errorf("provider request: %v", err)
					return
				}
				n := requests.Add(1)
				if mode == "timeout" {
					select {
					case <-r.Context().Done():
					case <-releaseHandler:
					}
					return
				}
				w.Header().Set("Content-Type", "application/json")
				if n > 1 && mode == "complete" {
					fmt.Fprint(w, `{"choices":[{"message":{"role":"assistant","content":"Witness written."},"finish_reason":"stop"}]}`)
					return
				}
				args, _ := json.Marshal(map[string]string{"file_path": artifact, "content": "native mediated witness", "mode": "create"})
				_ = json.NewEncoder(w).Encode(map[string]any{"choices": []any{map[string]any{"message": map[string]any{"role": "assistant", "tool_calls": []any{map[string]any{"id": "write-witness", "type": "function", "function": map[string]any{"name": "Write", "arguments": string(args)}}}}, "finish_reason": "tool_calls"}}})
			}))
			defer server.Close()
			deadline := "10s"
			if mode == "timeout" {
				deadline = "2s"
			}
			var stdout, stderr bytes.Buffer
			code := runOpsRun(&stdout, &stderr, []string{"--harness", "native", "--prompt-file", prompt, "--receipt", receipt, "--provider", "openai", "--model", "fixture", "--base-url", server.URL + "/v1", "--workspace", root, "--policy", policy, "--max-turns", "3", "--timeout", deadline, "--effort", "low"})
			close(releaseHandler)
			data, err := os.ReadFile(receipt)
			if err != nil {
				t.Fatalf("receipt: %v; child stderr: %s", err, &stderr)
			}
			var got opsRunReceipt
			if err := json.Unmarshal(data, &got); err != nil {
				t.Fatal(err)
			}
			if got.Harness != "native" || got.Finished.IsZero() || strings.Contains(string(data), "private native") {
				t.Fatalf("invalid metadata receipt: %s", data)
			}
			if mode == "complete" {
				written, err := os.ReadFile(artifact)
				if code != 0 || got.Status != "succeeded" || err != nil || string(written) != "native mediated witness" || requests.Load() < 2 {
					t.Fatalf("code=%d receipt=%s artifact=%q err=%v stdout=%s stderr=%s", code, data, written, err, &stdout, &stderr)
				}
			} else if code == 0 || got.Status == "succeeded" || (mode == "timeout" && (code != 124 || got.Status != "timed_out")) {
				t.Fatalf("uncompleted run accepted: code=%d receipt=%s stderr=%s", code, data, &stderr)
			}
			if mode == "timeout" && requests.Load() == 0 {
				t.Fatal("timeout did not reach the native provider request")
			}
		})
	}
}

func TestOpsNativeRetainsExplicitPolicy(t *testing.T) {
	t.Setenv("FAK_OPS_NATIVE_TEST_CHILD", "1")
	root := t.TempDir()
	prompt := filepath.Join(root, "prompt.txt")
	receipt := filepath.Join(root, "run.json")
	policyFile := filepath.Join(root, "policy.json")
	outOfScopeFile := filepath.Join(root, "evil.txt")
	permittedFile := filepath.Join(root, "report", "summary.txt")
	if err := os.MkdirAll(filepath.Join(root, "report"), 0755); err != nil {
		t.Fatal(err)
	}

	if err := os.WriteFile(prompt, []byte("native policy retention task"), 0600); err != nil {
		t.Fatal(err)
	}
	policyJSON := `{
		"allow": ["Write", "Bash"],
		"arg_rules": [
			{
				"tool": "Write",
				"arg": "file_path",
				"allow_glob": "report/**",
				"reason": "POLICY_BLOCK"
			},
			{
				"tool": "Bash",
				"arg": "command",
				"allow_exact": "printf 'witness marker\\n'",
				"reason": "POLICY_BLOCK"
			}
		]
	}`
	if err := os.WriteFile(policyFile, []byte(policyJSON), 0600); err != nil {
		t.Fatal(err)
	}

	var requests atomic.Int32
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		n := requests.Add(1)
		switch n {
		case 1:
			// Attempt 1: Write outside permitted scope (should be denied).
			args, _ := json.Marshal(map[string]string{"file_path": "evil.txt", "content": "evil content", "mode": "create"})
			_ = json.NewEncoder(w).Encode(map[string]any{
				"choices": []any{
					map[string]any{
						"message": map[string]any{
							"role": "assistant",
							"tool_calls": []any{
								map[string]any{
									"id":   "write-evil",
									"type": "function",
									"function": map[string]any{
										"name":      "Write",
										"arguments": string(args),
									},
								},
							},
						},
						"finish_reason": "tool_calls",
					},
				},
			})
		case 2:
			// Attempt 2: Bash command violating allow_exact (should be denied).
			args, _ := json.Marshal(map[string]string{"command": "printf 'near match\\n'"})
			_ = json.NewEncoder(w).Encode(map[string]any{
				"choices": []any{
					map[string]any{
						"message": map[string]any{
							"role": "assistant",
							"tool_calls": []any{
								map[string]any{
									"id":   "bash-near-match",
									"type": "function",
									"function": map[string]any{
										"name":      "Bash",
										"arguments": string(args),
									},
								},
							},
						},
						"finish_reason": "tool_calls",
					},
				},
			})
		case 3:
			// Attempt 3: Write inside permitted scope (should succeed).
			args, _ := json.Marshal(map[string]string{"file_path": "report/summary.txt", "content": "report ok", "mode": "create"})
			_ = json.NewEncoder(w).Encode(map[string]any{
				"choices": []any{
					map[string]any{
						"message": map[string]any{
							"role": "assistant",
							"tool_calls": []any{
								map[string]any{
									"id":   "write-report",
									"type": "function",
									"function": map[string]any{
										"name":      "Write",
										"arguments": string(args),
									},
								},
							},
						},
						"finish_reason": "tool_calls",
					},
				},
			})
		default:
			// Final completion.
			fmt.Fprint(w, `{"choices":[{"message":{"role":"assistant","content":"Done."},"finish_reason":"stop"}]}`)
		}
	}))
	defer server.Close()

	var stdout, stderr bytes.Buffer
	code := runOpsRun(&stdout, &stderr, []string{
		"--harness", "native",
		"--prompt-file", prompt,
		"--receipt", receipt,
		"--provider", "openai",
		"--model", "fixture",
		"--base-url", server.URL + "/v1",
		"--workspace", root,
		"--policy", policyFile,
		"--max-turns", "5",
		"--timeout", "10s",
		"--effort", "low",
	})

	if _, err := os.Stat(outOfScopeFile); !os.IsNotExist(err) {
		t.Fatalf("out-of-policy file %s must NOT exist, err=%v", outOfScopeFile, err)
	}
	written, err := os.ReadFile(permittedFile)
	if err != nil || string(written) != "report ok" {
		t.Fatalf("permitted report file %s not written: err=%v, content=%q\nstdout: %s\nstderr: %s", permittedFile, err, written, &stdout, &stderr)
	}

	data, err := os.ReadFile(receipt)
	if err != nil {
		t.Fatalf("receipt: %v; child stderr: %s", err, &stderr)
	}
	var got opsRunReceipt
	if err := json.Unmarshal(data, &got); err != nil {
		t.Fatal(err)
	}
	if code != 0 || got.Status != "succeeded" {
		t.Fatalf("code=%d, status=%q, stderr=%s", code, got.Status, &stderr)
	}
}

func TestOpsNativePolicyExactCommand(t *testing.T) {
	t.Setenv("FAK_OPS_NATIVE_TEST_CHILD", "1")
	root := t.TempDir()
	prompt := filepath.Join(root, "prompt.txt")
	receipt := filepath.Join(root, "run.json")
	policyFile := filepath.Join(root, "policy.json")
	permittedFile := filepath.Join(root, "report", "inventory.txt")
	if err := os.MkdirAll(filepath.Join(root, "report"), 0755); err != nil {
		t.Fatal(err)
	}

	if err := os.WriteFile(prompt, []byte("native exact command inventory task"), 0600); err != nil {
		t.Fatal(err)
	}

	exactCmd := "echo exact-authorized-marker"
	policyJSON := `{
		"posture": "fail_closed",
		"allow": ["Write", "Bash"],
		"arg_rules": [
			{
				"tool": "Write",
				"arg": "file_path",
				"allow_glob": "report/**",
				"reason": "POLICY_BLOCK"
			},
			{
				"tool": "Bash",
				"arg": "command",
				"allow_exact": "echo exact-authorized-marker",
				"reason": "POLICY_BLOCK"
			}
		]
	}`
	if err := os.WriteFile(policyFile, []byte(policyJSON), 0600); err != nil {
		t.Fatal(err)
	}

	var requests atomic.Int32
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var request map[string]any
		_ = json.NewDecoder(r.Body).Decode(&request)
		w.Header().Set("Content-Type", "application/json")
		n := requests.Add(1)
		switch n {
		case 1:
			// Attempt 1: Near-match Bash command (denied by policy).
			args, _ := json.Marshal(map[string]string{"command": "echo near-match-marker"})
			_ = json.NewEncoder(w).Encode(map[string]any{
				"choices": []any{
					map[string]any{
						"message": map[string]any{
							"role": "assistant",
							"tool_calls": []any{
								map[string]any{
									"id":   "bash-near-match",
									"type": "function",
									"function": map[string]any{
										"name":      "Bash",
										"arguments": string(args),
									},
								},
							},
						},
						"finish_reason": "tool_calls",
					},
				},
			})
		case 2:
			// Attempt 2: Exact authorized command but with wrong/escaping cwd (denied).
			args, _ := json.Marshal(map[string]string{"command": exactCmd, "cwd": "../escape"})
			_ = json.NewEncoder(w).Encode(map[string]any{
				"choices": []any{
					map[string]any{
						"message": map[string]any{
							"role": "assistant",
							"tool_calls": []any{
								map[string]any{
									"id":   "bash-wrong-cwd",
									"type": "function",
									"function": map[string]any{
										"name":      "Bash",
										"arguments": string(args),
									},
								},
							},
						},
						"finish_reason": "tool_calls",
					},
				},
			})
		case 3:
			// Attempt 3: Exact authorized command in workspace cwd (should succeed).
			args, _ := json.Marshal(map[string]string{"command": exactCmd})
			_ = json.NewEncoder(w).Encode(map[string]any{
				"choices": []any{
					map[string]any{
						"message": map[string]any{
							"role": "assistant",
							"tool_calls": []any{
								map[string]any{
									"id":   "bash-exact-authorized",
									"type": "function",
									"function": map[string]any{
										"name":      "Bash",
										"arguments": string(args),
									},
								},
							},
						},
						"finish_reason": "tool_calls",
					},
				},
			})
		case 4:
			// Verify that the previous tool execution for bash-exact-authorized returned the exact marker.
			msgs, _ := request["messages"].([]any)
			var markerWitnessed bool
			for _, m := range msgs {
				msgMap, _ := m.(map[string]any)
				if msgMap["role"] == "tool" && msgMap["tool_call_id"] == "bash-exact-authorized" {
					content, _ := msgMap["content"].(string)
					if strings.Contains(content, "exact-authorized-marker") {
						markerWitnessed = true
					}
				}
			}
			content := "failed: command denied"
			if markerWitnessed {
				content = "witness: exact-authorized-marker"
			}
			// Attempt 4: Confined Write with the witnessed marker (should succeed).
			args, _ := json.Marshal(map[string]string{"file_path": "report/inventory.txt", "content": content, "mode": "create"})
			_ = json.NewEncoder(w).Encode(map[string]any{
				"choices": []any{
					map[string]any{
						"message": map[string]any{
							"role": "assistant",
							"tool_calls": []any{
								map[string]any{
									"id":   "write-inventory",
									"type": "function",
									"function": map[string]any{
										"name":      "Write",
										"arguments": string(args),
									},
								},
							},
						},
						"finish_reason": "tool_calls",
					},
				},
			})
		default:
			// Final completion.
			fmt.Fprint(w, `{"choices":[{"message":{"role":"assistant","content":"Operational inventory complete."},"finish_reason":"stop"}]}`)
		}
	}))
	defer server.Close()

	var stdout, stderr bytes.Buffer
	code := runOpsRun(&stdout, &stderr, []string{
		"--harness", "native",
		"--prompt-file", prompt,
		"--receipt", receipt,
		"--provider", "openai",
		"--model", "fixture",
		"--base-url", server.URL + "/v1",
		"--workspace", root,
		"--policy", policyFile,
		"--max-turns", "6",
		"--timeout", "10s",
		"--effort", "low",
	})

	written, err := os.ReadFile(permittedFile)
	if err != nil || string(written) != "witness: exact-authorized-marker" {
		t.Fatalf("permitted report file %s not written: err=%v, content=%q\nstdout: %s\nstderr: %s", permittedFile, err, written, &stdout, &stderr)
	}

	data, err := os.ReadFile(receipt)
	if err != nil {
		t.Fatalf("receipt: %v; child stderr: %s", err, &stderr)
	}
	var got opsRunReceipt
	if err := json.Unmarshal(data, &got); err != nil {
		t.Fatal(err)
	}
	if code != 0 || got.Status != "succeeded" {
		t.Fatalf("code=%d, status=%q, stderr=%s", code, got.Status, &stderr)
	}
}
