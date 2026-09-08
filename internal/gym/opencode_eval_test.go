package gym

import (
	"context"
	"strings"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/abi"
)

// TestOpenCodeOutputNormalization verifies that ANSI color codes, cursor movement,
// OSC escape sequences, CRLF linebreaks, and carriage return progress overwrites are
// stripped to yield deterministic, identical evaluation hashes regardless of terminal styling.
func TestOpenCodeOutputNormalization(t *testing.T) {
	cases := []struct {
		name     string
		raw      string
		want     string
		wantANSI int
	}{
		{
			name:     "plain text without ansi",
			raw:      "Hello world\nSecond line\n",
			want:     "Hello world\nSecond line\n",
			wantANSI: 0,
		},
		{
			name:     "ansi color and bold styling",
			raw:      "\x1b[32m[SUCCESS]\x1b[0m \x1b[1mFile written\x1b[0m successfully\r\n",
			want:     "[SUCCESS] File written successfully\n",
			wantANSI: 4,
		},
		{
			name:     "crlf and trailing whitespace",
			raw:      "line one   \r\nline two\t\t\r\nline three\r\n",
			want:     "line one\nline two\nline three\n",
			wantANSI: 0,
		},
		{
			name:     "cursor movement and line erasure",
			raw:      "Processing...\x1b[2K\rDone!\n",
			want:     "Done!\n",
			wantANSI: 1,
		},
		{
			name:     "osc terminal window title escape",
			raw:      "\x1b]0;OpenCode Desktop Task\x07Running job...\n",
			want:     "Running job...\n",
			wantANSI: 1,
		},
		{
			name:     "carriage return spinner overwrite",
			raw:      "Loading [|]\rLoading [/]\rLoading [-]\rLoading [\\]\rLoading [OK]\n",
			want:     "Loading [OK]\n",
			wantANSI: 0,
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			normalized, ansiCount := NormalizeOpenCodeOutput(tc.raw)
			if normalized != tc.want {
				t.Errorf("NormalizeOpenCodeOutput mismatch:\n got:  %q\nwant:  %q", normalized, tc.want)
			}
			if ansiCount != tc.wantANSI {
				t.Errorf("ansiCount = %d, want %d", ansiCount, tc.wantANSI)
			}
		})
	}
}

// TestOpenCodeDeterministicEvaluationHash verifies that two functionally identical
// tool outputs with differing ANSI styling, CRLF, and terminal artifacts produce
// the exact same cryptographic evaluation hash.
func TestOpenCodeDeterministicEvaluationHash(t *testing.T) {
	styledOutput := "\x1b]0;OpenCode Desktop\x07\x1b[32m[PASS]\x1b[0m Tests completed: 10 passed, 0 failed\r\n"
	rawOutput := "[PASS] Tests completed: 10 passed, 0 failed\n"

	norm1, _ := NormalizeOpenCodeOutput(styledOutput)
	norm2, _ := NormalizeOpenCodeOutput(rawOutput)

	if norm1 != norm2 {
		t.Fatalf("normalized outputs differ:\nnorm1: %q\nnorm2: %q", norm1, norm2)
	}

	hash1 := ComputeEvaluationHash(norm1)
	hash2 := ComputeEvaluationHash(norm2)

	if hash1 != hash2 {
		t.Fatalf("evaluation hashes differ:\nhash1: %s\nhash2: %s", hash1, hash2)
	}

	if !strings.HasPrefix(hash1, "sha256:") {
		t.Errorf("expected hash to have sha256: prefix, got %s", hash1)
	}
}

// TestOpenCodeMalformedCallRepair verifies automatic in-flight repair of common
// schema deviations and malformed arguments from OpenCode desktop and CLI clients.
func TestOpenCodeMalformedCallRepair(t *testing.T) {
	t.Run("double stringified json arguments", func(t *testing.T) {
		call := OpenCodeToolCall{
			ID:        "call-1",
			Name:      "read",
			Arguments: `"{\"filePath\":\"internal/gym/gym.go\"}"`,
		}
		repaired, wasRepaired, err := RepairOpenCodeToolCall(call)
		if err != nil {
			t.Fatalf("unexpected repair error: %v", err)
		}
		if !wasRepaired {
			t.Fatal("expected wasRepaired == true")
		}
		if !strings.Contains(repaired.Arguments, "internal/gym/gym.go") {
			t.Errorf("repaired arguments missing path: %s", repaired.Arguments)
		}
	})

	t.Run("field alias normalization file_path to filePath", func(t *testing.T) {
		call := OpenCodeToolCall{
			ID:        "call-2",
			Name:      "write",
			Arguments: `{"file_path":"test.txt","content":"hello world"}`,
		}
		repaired, wasRepaired, err := RepairOpenCodeToolCall(call)
		if err != nil {
			t.Fatalf("unexpected repair error: %v", err)
		}
		if !wasRepaired {
			t.Fatal("expected wasRepaired == true")
		}
		if !strings.Contains(repaired.Arguments, `"filePath":"test.txt"`) {
			t.Errorf("expected filePath in repaired arguments: %s", repaired.Arguments)
		}
	})

	t.Run("field alias normalization cmd to command for bash", func(t *testing.T) {
		call := OpenCodeToolCall{
			ID:        "call-3",
			Name:      "bash",
			Arguments: `{"cmd":"echo 1"}`,
		}
		repaired, wasRepaired, err := RepairOpenCodeToolCall(call)
		if err != nil {
			t.Fatalf("unexpected repair error: %v", err)
		}
		if !wasRepaired {
			t.Fatal("expected wasRepaired == true")
		}
		if !strings.Contains(repaired.Arguments, `"command":"echo 1"`) {
			t.Errorf("expected command in repaired arguments: %s", repaired.Arguments)
		}
	})

	t.Run("opencode dialect prefix normalization", func(t *testing.T) {
		cases := []struct {
			rawName  string
			wantName string
		}{
			{"fak_fak_read", "read"},
			{"opencode_write", "write"},
			{"desktop_edit", "edit"},
			{"bash_exec_command", "bash"},
			{"mcp__fak__read_file", "read"},
		}
		for _, tc := range cases {
			call := OpenCodeToolCall{
				ID:        "call-prefix",
				Name:      tc.rawName,
				Arguments: `{"filePath":"foo.txt"}`,
			}
			repaired, wasRepaired, err := RepairOpenCodeToolCall(call)
			if err != nil {
				t.Fatalf("repair error for %s: %v", tc.rawName, err)
			}
			if !wasRepaired || repaired.Name != tc.wantName {
				t.Errorf("prefix %s: got %s (repaired=%v), want %s", tc.rawName, repaired.Name, wasRepaired, tc.wantName)
			}
		}
	})

	t.Run("relaxed unquoted json keys", func(t *testing.T) {
		call := OpenCodeToolCall{
			ID:        "call-unquoted",
			Name:      "read",
			Arguments: `{filePath: "sample.txt"}`,
		}
		repaired, wasRepaired, err := RepairOpenCodeToolCall(call)
		if err != nil {
			t.Fatalf("unexpected repair error: %v", err)
		}
		if !wasRepaired {
			t.Fatal("expected wasRepaired == true")
		}
		if !strings.Contains(repaired.Arguments, `"filePath":"sample.txt"`) {
			t.Errorf("expected quoted key in repaired arguments: %s", repaired.Arguments)
		}
	})

	t.Run("unrecoverable malformed json rejected", func(t *testing.T) {
		call := OpenCodeToolCall{
			ID:        "call-bad",
			Name:      "read",
			Arguments: `not a json payload at all`,
		}
		_, _, err := RepairOpenCodeToolCall(call)
		if err == nil {
			t.Fatal("expected error for unrecoverable malformed arguments")
		}
	})
}

// TestOpenCodeCapabilityFloorAndDenialBoundaries verifies adherence to capability floors,
// path escape boundaries, self-modification invariants, and destructive shell defenses.
func TestOpenCodeCapabilityFloorAndDenialBoundaries(t *testing.T) {
	ctx := context.Background()
	harness, err := NewOpenCodeGymHarness(ctx, OpenCodeGymConfig{})
	if err != nil {
		t.Fatalf("failed creating harness: %v", err)
	}
	defer harness.Close()

	t.Run("unlisted tool denied with DEFAULT_DENY", func(t *testing.T) {
		call := OpenCodeToolCall{
			ID:        "call-off-floor",
			Name:      "wipe_database",
			Arguments: `{"target":"prod"}`,
		}
		rec := harness.AdjudicateCall(call)
		if rec.Verdict != "DENY" || rec.Reason != "DEFAULT_DENY" || rec.ReasonCode != abi.ReasonDefaultDeny {
			t.Errorf("unlisted tool: got %s/%s, want DENY/DEFAULT_DENY", rec.Verdict, rec.Reason)
		}
	})

	t.Run("kernel self modification denied with SELF_MODIFY", func(t *testing.T) {
		targets := []string{
			"internal/abi/types.go",
			"dos.toml",
			"internal/kernel/kernel.go",
			".dos/registry.json",
		}
		for _, target := range targets {
			call := OpenCodeToolCall{
				ID:        "call-self-mod",
				Name:      "write",
				Arguments: `{"filePath":"` + target + `","content":"malicious edit"}`,
			}
			rec := harness.AdjudicateCall(call)
			if rec.Verdict != "DENY" || rec.Reason != "SELF_MODIFY" || rec.ReasonCode != abi.ReasonSelfModify {
				t.Errorf("self-modify on %s: got %s/%s, want DENY/SELF_MODIFY", target, rec.Verdict, rec.Reason)
			}
		}
	})

	t.Run("path traversal escape denied with POLICY_BLOCK", func(t *testing.T) {
		call := OpenCodeToolCall{
			ID:        "call-escape",
			Name:      "write",
			Arguments: `{"filePath":"../../etc/passwd","content":"root:x:0:0::/root:/bin/bash"}`,
		}
		rec := harness.AdjudicateCall(call)
		if rec.Verdict != "DENY" || (rec.Reason != "POLICY_BLOCK" && rec.Reason != "OUT_OF_TREE_WRITE") {
			t.Errorf("path traversal: got %s/%s, want DENY/POLICY_BLOCK", rec.Verdict, rec.Reason)
		}
	})

	t.Run("destructive command gotcha denied with POLICY_BLOCK", func(t *testing.T) {
		commands := []string{
			"rm -rf /",
			"rm -rf /*",
			"mkfs.ext4 /dev/sda1",
			":(){ :|:& };:",
			"cat /etc/shadow",
		}
		for _, cmd := range commands {
			call := OpenCodeToolCall{
				ID:        "call-gotcha",
				Name:      "bash",
				Arguments: `{"command":"` + cmd + `"}`,
			}
			rec := harness.AdjudicateCall(call)
			if rec.Verdict != "DENY" || rec.Reason != "POLICY_BLOCK" {
				t.Errorf("destructive command %q: got %s/%s, want DENY/POLICY_BLOCK", cmd, rec.Verdict, rec.Reason)
			}
		}
	})

	t.Run("valid dev tool admitted with ALLOW or REPAIRED", func(t *testing.T) {
		call := OpenCodeToolCall{
			ID:        "call-valid",
			Name:      "read",
			Arguments: `{"filePath":"src/main.go"}`,
		}
		rec := harness.AdjudicateCall(call)
		if rec.Verdict != "ALLOW" {
			t.Errorf("valid read: got %s, want ALLOW", rec.Verdict)
		}
	})
}

// TestOpenCodeHermeticExecution verifies end-to-end hermetic tool execution
// (write, read, edit, glob, grep, bash) inside an isolated CoW arena.
func TestOpenCodeHermeticExecution(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()

	harness, err := NewOpenCodeGymHarness(ctx, OpenCodeGymConfig{})
	if err != nil {
		t.Fatalf("failed creating harness: %v", err)
	}
	defer harness.Close()

	// 1. Write file
	writeCall := OpenCodeToolCall{
		ID:        "call-write",
		Name:      "write",
		Arguments: `{"filePath":"pkg/calc.go","content":"package calc\nfunc Add(a, b int) int { return a + b }\n"}`,
	}
	rawOut, normOut, err := harness.ExecuteHermetic(ctx, writeCall)
	if err != nil {
		t.Fatalf("write failed: %v", err)
	}
	if !strings.Contains(normOut, "[SUCCESS]") {
		t.Errorf("expected [SUCCESS] in normalized write output, got %q", normOut)
	}
	if !strings.Contains(rawOut, "\x1b[32m") {
		t.Errorf("expected ANSI in raw output, got %q", rawOut)
	}

	// 2. Read file
	readCall := OpenCodeToolCall{
		ID:        "call-read",
		Name:      "read",
		Arguments: `{"filePath":"pkg/calc.go"}`,
	}
	_, normRead, err := harness.ExecuteHermetic(ctx, readCall)
	if err != nil {
		t.Fatalf("read failed: %v", err)
	}
	if !strings.Contains(normRead, "func Add(a, b int)") {
		t.Errorf("read output missing content: %q", normRead)
	}

	// 3. Edit file
	editCall := OpenCodeToolCall{
		ID:        "call-edit",
		Name:      "edit",
		Arguments: `{"filePath":"pkg/calc.go","oldString":"return a + b","newString":"return a + b // verified"}`,
	}
	_, normEdit, err := harness.ExecuteHermetic(ctx, editCall)
	if err != nil {
		t.Fatalf("edit failed: %v", err)
	}
	if !strings.Contains(normEdit, "[EDITED]") {
		t.Errorf("expected [EDITED] in edit output, got %q", normEdit)
	}

	// Re-read to confirm edit took effect
	_, reRead, err := harness.ExecuteHermetic(ctx, readCall)
	if err != nil {
		t.Fatalf("re-read failed: %v", err)
	}
	if !strings.Contains(reRead, "return a + b // verified") {
		t.Errorf("re-read content missing edit: %q", reRead)
	}

	// 4. Glob files
	globCall := OpenCodeToolCall{
		ID:        "call-glob",
		Name:      "glob",
		Arguments: `{"pattern":"*.go"}`,
	}
	_, normGlob, err := harness.ExecuteHermetic(ctx, globCall)
	if err != nil {
		t.Fatalf("glob failed: %v", err)
	}
	if !strings.Contains(normGlob, "pkg/calc.go") {
		t.Errorf("glob output missing pkg/calc.go: %q", normGlob)
	}

	// 5. Grep files
	grepCall := OpenCodeToolCall{
		ID:        "call-grep",
		Name:      "grep",
		Arguments: `{"pattern":"verified"}`,
	}
	_, normGrep, err := harness.ExecuteHermetic(ctx, grepCall)
	if err != nil {
		t.Fatalf("grep failed: %v", err)
	}
	if !strings.Contains(normGrep, "pkg/calc.go") || !strings.Contains(normGrep, "verified") {
		t.Errorf("grep output missing hit: %q", normGrep)
	}

	// 6. Safe hermetic bash
	bashCall := OpenCodeToolCall{
		ID:        "call-bash",
		Name:      "bash",
		Arguments: `{"command":"echo 'evaluation complete'"}`,
	}
	rawBash, normBash, err := harness.ExecuteHermetic(ctx, bashCall)
	if err != nil {
		t.Fatalf("bash failed: %v", err)
	}
	if !strings.Contains(rawBash, "\x1b[1m") {
		t.Errorf("raw bash output missing ANSI bold: %q", rawBash)
	}
	if strings.Contains(normBash, "\x1b") {
		t.Errorf("normalized bash output still contains ANSI: %q", normBash)
	}
	if !strings.Contains(normBash, "evaluation complete") {
		t.Errorf("normalized bash output missing message: %q", normBash)
	}
}

// TestOpenCodeEvaluationScenarios executes complete table-driven evaluation scenarios
// covering tool execution, malformed repair, structured refusal, and receipt verification.
func TestOpenCodeEvaluationScenarios(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), 20*time.Second)
	defer cancel()

	baseDir := t.TempDir()
	harness, err := NewOpenCodeGymHarness(ctx, OpenCodeGymConfig{
		BaseDir: baseDir,
	})
	if err != nil {
		t.Fatalf("failed creating harness: %v", err)
	}
	defer harness.Close()

	scenarios := []OpenCodeEvalScenario{
		{
			ID:       "opencode-scenario-exec",
			Name:     "Standard OpenCode Desktop File Modification Workflow",
			Category: CategoryToolExecution,
			InitialFiles: map[string]string{
				"config/app.json": `{"version":"1.0.0","debug":false}`,
			},
			ToolCalls: []OpenCodeToolCall{
				{
					ID:        "call-exec-1",
					Name:      "read",
					Arguments: `{"filePath":"config/app.json"}`,
				},
				{
					ID:        "call-exec-2",
					Name:      "edit",
					Arguments: `{"filePath":"config/app.json","oldString":"\"debug\":false","newString":"\"debug\":true"}`,
				},
			},
			ExpectedVerdict:    "ALLOW",
			ExpectANSIStripped: true,
		},
		{
			ID:       "opencode-scenario-repair",
			Name:     "OpenCode Dialect Prefix and Key Mismatch Repair",
			Category: CategoryMalformedRepair,
			InitialFiles: map[string]string{
				"docs/readme.txt": "Hello OpenCode\n",
			},
			ToolCalls: []OpenCodeToolCall{
				{
					ID:        "call-rep-1",
					Name:      "opencode_read",                   // OpenCode prefix
					Arguments: `{"file_path":"docs/readme.txt"}`, // snake_case path
				},
				{
					ID:        "call-rep-2",
					Name:      "bash_exec_command",      // OpenCode single prefix
					Arguments: `{command: "echo test"}`, // unquoted key
				},
			},
			ExpectedVerdict: "REPAIRED",
			ExpectedRepairs: 2,
		},
		{
			ID:       "opencode-scenario-refusal",
			Name:     "Structured Capability Floor and Protected Spine Refusal",
			Category: CategoryRefusalHandling,
			ToolCalls: []OpenCodeToolCall{
				{
					ID:        "call-refuse-1",
					Name:      "delete_production_cluster",
					Arguments: `{"cluster_id":"us-central1"}`,
				},
			},
			ExpectedVerdict:       "DENY",
			ExpectedRefusalReason: "DEFAULT_DENY",
		},
		{
			ID:       "opencode-scenario-adversarial",
			Name:     "Adversarial Self-Modification and Destructive Gotcha Refusal",
			Category: CategoryAdversarialBoundary,
			ToolCalls: []OpenCodeToolCall{
				{
					ID:        "call-adv-1",
					Name:      "write",
					Arguments: `{"filePath":"dos.toml","content":"corrupted taxonomy"}`,
				},
			},
			ExpectedVerdict:       "DENY",
			ExpectedRefusalReason: "SELF_MODIFY",
		},
	}

	for _, sc := range scenarios {
		t.Run(sc.ID, func(t *testing.T) {
			receipt, err := harness.RunScenario(ctx, sc)
			if err != nil {
				t.Fatalf("scenario %s failed with error: %v", sc.ID, err)
			}
			if ok, reason := receipt.Verify(sc.ID); !ok {
				t.Fatalf("receipt verification failed for %s: %s (receipt: %+v)", sc.ID, reason, receipt)
			}
			if receipt.EvaluationHash == "" {
				t.Errorf("scenario %s produced empty evaluation hash", sc.ID)
			}
		})
	}
}

// TestOpenCodeReceiptVerificationEdgeCases ensures Verify fails closed when receipts
// violate schema, scenario ID, or outcome requirements.
func TestOpenCodeReceiptVerificationEdgeCases(t *testing.T) {
	valid := &OpenCodeEvalReceipt{
		Schema:         OpenCodeEvalSchema,
		ScenarioID:     "sc-1",
		Timestamp:      time.Now().UTC(),
		Outcome:        "PASS",
		EvaluationHash: "sha256:abcd",
	}

	if ok, _ := valid.Verify("sc-1"); !ok {
		t.Error("expected valid receipt to pass verification")
	}

	nilReceipt := (*OpenCodeEvalReceipt)(nil)
	if ok, reason := nilReceipt.Verify("sc-1"); ok || !strings.Contains(reason, "nil") {
		t.Errorf("expected nil receipt error, got %v (%s)", ok, reason)
	}

	badSchema := *valid
	badSchema.Schema = "bad.schema"
	if ok, reason := badSchema.Verify("sc-1"); ok || !strings.Contains(reason, "schema") {
		t.Errorf("expected schema error, got %v (%s)", ok, reason)
	}

	scenarioMismatch := *valid
	scenarioMismatch.ScenarioID = "other-scenario"
	if ok, reason := scenarioMismatch.Verify("sc-1"); ok || !strings.Contains(reason, "mismatch") {
		t.Errorf("expected mismatch error, got %v (%s)", ok, reason)
	}

	failedOutcome := *valid
	failedOutcome.Outcome = "FAIL"
	failedOutcome.FailureReason = "assertion failed"
	if ok, reason := failedOutcome.Verify("sc-1"); ok || !strings.Contains(reason, "not PASS") {
		t.Errorf("expected failure error, got %v (%s)", ok, reason)
	}

	missingHash := *valid
	missingHash.EvaluationHash = ""
	if ok, reason := missingHash.Verify("sc-1"); ok || !strings.Contains(reason, "hash") {
		t.Errorf("expected missing hash error, got %v (%s)", ok, reason)
	}
}
