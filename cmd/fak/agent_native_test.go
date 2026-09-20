package main

import (
	"bytes"
	"encoding/base64"
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/abi"
	"github.com/anthony-chaudhary/fak/internal/adjudicator"
	"github.com/anthony-chaudhary/fak/internal/agent"
	"github.com/anthony-chaudhary/fak/internal/canon"
)

func TestNativeAgentReceiptIsSingleKernelArm(t *testing.T) {
	metrics := agent.ArmMetrics{
		Arm:           "fak",
		Turns:         2,
		TaskCompleted: true,
		FinalAnswer:   "done",
	}
	receipt := newNativeAgentReceipt("fix it", "fixture", metrics)
	if receipt.Schema != nativeAgentReceiptSchema || receipt.Task != "fix it" || receipt.Model != "fixture" {
		t.Fatalf("receipt identity = %#v", receipt)
	}
	if receipt.Metrics.Arm != "fak" || receipt.Metrics.FinalAnswer != "done" {
		t.Fatalf("receipt metrics = %#v", receipt.Metrics)
	}
	body, err := json.Marshal(receipt)
	if err != nil {
		t.Fatal(err)
	}
	if bytes.Contains(body, []byte(`"baseline"`)) {
		t.Fatalf("single-arm receipt leaked benchmark arm: %s", body)
	}
}

func TestNativeAgentReceiptCalls(t *testing.T) {
	t.Run("versioned ordered calls preserve adjudication fields", func(t *testing.T) {
		calls := []agent.CallTrace{
			{Arm: "fak", Turn: 1, Tool: "read_file", Verdict: "ALLOW", By: "policy-floor", Args: `{"path":"README.md"}`, Note: "served by the kernel"},
			{Arm: "fak", Turn: 2, Tool: "write_file", Verdict: "DENY", Reason: "POLICY_BLOCK", By: "workspace-floor", Disposition: "TERMINAL", Args: `{"path":"outside.txt"}`, Note: "outside the granted workspace"},
		}

		body, err := json.Marshal(newHeadlessAgentReceipt("fix it", "fixture", agent.ArmMetrics{}, calls, "", nil))
		if err != nil {
			t.Fatal(err)
		}
		var got struct {
			Schema string `json:"schema"`
			Calls  *struct {
				Schema  string            `json:"schema"`
				Entries []agent.CallTrace `json:"entries"`
			} `json:"calls,omitempty"`
		}
		if err := json.Unmarshal(body, &got); err != nil {
			t.Fatal(err)
		}
		if got.Schema != nativeAgentReceiptSchema {
			t.Fatalf("receipt schema = %q, want %q", got.Schema, nativeAgentReceiptSchema)
		}
		if got.Calls == nil {
			t.Fatalf("calls missing from native receipt: %s", body)
		}
		if got.Calls.Schema != "fak.agent.native.calls.v1" {
			t.Fatalf("calls schema = %q, want fak.agent.native.calls.v1", got.Calls.Schema)
		}
		if len(got.Calls.Entries) != len(calls) {
			t.Fatalf("calls = %#v, want %#v", got.Calls.Entries, calls)
		}
		for i, want := range calls {
			entry := got.Calls.Entries[i]
			if entry.Turn != want.Turn || entry.Tool != want.Tool || entry.Verdict != want.Verdict ||
				entry.Reason != want.Reason || entry.By != want.By || entry.Disposition != want.Disposition {
				t.Fatalf("call[%d] = %#v, want ordered adjudication %#v", i, entry, want)
			}
		}
		var wire struct {
			Calls struct {
				Entries []map[string]json.RawMessage `json:"entries"`
			} `json:"calls"`
		}
		if err := json.Unmarshal(body, &wire); err != nil {
			t.Fatal(err)
		}
		for i, entry := range wire.Calls.Entries {
			for _, unsupported := range []string{"result", "output", "executed", "succeeded"} {
				if _, ok := entry[unsupported]; ok {
					t.Fatalf("call[%d] invented unsupported %q field: %s", i, unsupported, body)
				}
			}
		}
	})

	t.Run("legacy receipt remains valid and omits calls", func(t *testing.T) {
		legacy := []byte(`{"schema":"fak.agent.native.v1","task":"fix it","model":"fixture","metrics":{"arm":"fak"}}`)
		var receipt nativeAgentReceipt
		if err := json.Unmarshal(legacy, &receipt); err != nil {
			t.Fatalf("legacy receipt no longer decodes: %v", err)
		}
		body, err := json.Marshal(receipt)
		if err != nil {
			t.Fatal(err)
		}
		var got map[string]json.RawMessage
		if err := json.Unmarshal(body, &got); err != nil {
			t.Fatal(err)
		}
		if _, ok := got["calls"]; ok {
			t.Fatalf("zero-call legacy receipt emitted optional calls: %s", body)
		}
		if receipt.Schema != nativeAgentReceiptSchema || receipt.Task != "fix it" || receipt.Model != "fixture" {
			t.Fatalf("legacy receipt identity changed: %#v", receipt)
		}
	})

	t.Run("arguments and notes are bounded and secret shaped data is redacted", func(t *testing.T) {
		const secret = "sk-ant-api03-native-receipt-secret-0123456789"
		calls := []agent.CallTrace{{
			Arm: "fak", Turn: 1, Tool: "http_request", Verdict: "DENY", Reason: "SECRET_EXFIL", By: "secret-floor",
			Args: `{"authorization":"Bearer ` + secret + `","padding":"` + strings.Repeat("x", 256) + `"}`,
			Note: "rejected credential " + secret + " " + strings.Repeat("n", 256),
		}}

		body, err := json.Marshal(newHeadlessAgentReceipt("fix it", "fixture", agent.ArmMetrics{}, calls, "", nil))
		if err != nil {
			t.Fatal(err)
		}
		if bytes.Contains(body, []byte(secret)) {
			t.Fatalf("native receipt leaked secret-shaped data: %s", body)
		}
		var got struct {
			Calls *struct {
				Entries []agent.CallTrace `json:"entries"`
			} `json:"calls,omitempty"`
		}
		if err := json.Unmarshal(body, &got); err != nil {
			t.Fatal(err)
		}
		if got.Calls == nil || len(got.Calls.Entries) != 1 {
			t.Fatalf("redacted call trace missing: %s", body)
		}
		entry := got.Calls.Entries[0]
		if len(entry.Args) > 163 || len(entry.Note) > 163 {
			t.Fatalf("call preview is unbounded: args=%d note=%d", len(entry.Args), len(entry.Note))
		}
	})

	t.Run("mixed raw and obfuscated credentials seal the preview", func(t *testing.T) {
		const rawSecret = "ghp_ABCDEFGHIJKLMNOPQRSTUVWXYZ012345"
		const decodedSecret = "sk-abcdef0123456789abcdef0123"
		obfuscatedSecret := base64.StdEncoding.EncodeToString([]byte(decodedSecret))
		mixed := []byte(`{"raw":"` + rawSecret + `","encoded":"` + obfuscatedSecret + `"}`)

		partiallyRedacted, masked := canon.RedactSecrets(mixed)
		if masked != 1 || !canon.Scan(partiallyRedacted).Secret {
			t.Fatalf("fixture must leave the canonical obfuscated credential after one raw mask: masked=%d redacted=%q", masked, partiallyRedacted)
		}
		body, err := json.Marshal(newHeadlessAgentReceipt("fix it", "fixture", agent.ArmMetrics{}, []agent.CallTrace{{
			Arm: "fak", Turn: 1, Tool: "http_request", Verdict: "DENY", Reason: "SECRET_EXFIL", By: "secret-floor",
			Args: string(mixed), Note: "mixed credential rejected",
		}}, "", nil))
		if err != nil {
			t.Fatal(err)
		}
		for _, leaked := range []string{rawSecret, decodedSecret, obfuscatedSecret} {
			if bytes.Contains(body, []byte(leaked)) {
				t.Fatalf("native receipt leaked credential bytes %q: %s", leaked, body)
			}
		}
		if !bytes.Contains(body, []byte(`[redacted:secret]`)) {
			t.Fatalf("mixed credential preview was not sealed: %s", body)
		}
	})
}

func TestNativeAgentOfflinePrintsAnswerAndWritesReceipt(t *testing.T) {
	out := filepath.Join(t.TempDir(), "native.json")
	stdout, stderr := captureAgentStdio(t, func() {
		cmdAgent([]string{"--native", "--offline", "--out", out})
	})
	if !strings.Contains(stdout, "Booked flight UA123") {
		t.Fatalf("stdout did not carry the final answer:\n%s", stdout)
	}
	if strings.Contains(stdout, "turn-use vs now") {
		t.Fatalf("native mode rendered the A/B benchmark:\n%s", stdout)
	}
	if !strings.Contains(stderr, filepath.Dir(out)) {
		t.Fatalf("stderr did not announce receipt directory:\n%s", stderr)
	}
	body, err := os.ReadFile(out)
	if err != nil {
		t.Fatal(err)
	}
	var receipt nativeAgentReceipt
	if err := json.Unmarshal(body, &receipt); err != nil {
		t.Fatal(err)
	}
	if receipt.Schema != nativeAgentReceiptSchema || receipt.Metrics.Arm != "fak" || !receipt.Metrics.TaskCompleted {
		t.Fatalf("receipt = %#v", receipt)
	}
	if receipt.Metrics.FinalAnswer == "" || receipt.Metrics.Turns != 7 {
		t.Fatalf("native run envelope = %#v", receipt.Metrics)
	}
}

func TestRawAgentReceiptIsSingleBaselineArm(t *testing.T) {
	metrics := agent.ArmMetrics{
		Arm:           "baseline",
		Turns:         2,
		TaskCompleted: true,
		FinalAnswer:   "done",
	}
	receipt := newRawAgentReceipt("fix it", "fixture", metrics)
	if receipt.Schema != rawAgentReceiptSchema || receipt.Task != "fix it" || receipt.Model != "fixture" {
		t.Fatalf("receipt identity = %#v", receipt)
	}
	if receipt.Metrics.Arm != "baseline" || receipt.Metrics.FinalAnswer != "done" {
		t.Fatalf("receipt metrics = %#v", receipt.Metrics)
	}
	body, err := json.Marshal(receipt)
	if err != nil {
		t.Fatal(err)
	}
	if bytes.Contains(body, []byte(`"fak"`)) {
		t.Fatalf("single-arm raw receipt leaked kernel arm: %s", body)
	}
}

func TestRawAgentOfflinePrintsAnswerAndWritesReceipt(t *testing.T) {
	out := filepath.Join(t.TempDir(), "raw.json")
	stdout, stderr := captureAgentStdio(t, func() {
		cmdAgent([]string{"--raw", "--offline", "--out", out})
	})
	if !strings.Contains(stdout, "Booked flight UA123") {
		t.Fatalf("stdout did not carry the final answer:\n%s", stdout)
	}
	if strings.Contains(stdout, "turn-use vs now") {
		t.Fatalf("raw mode rendered the A/B benchmark:\n%s", stdout)
	}
	if !strings.Contains(stderr, filepath.Dir(out)) {
		t.Fatalf("stderr did not announce receipt directory:\n%s", stderr)
	}
	body, err := os.ReadFile(out)
	if err != nil {
		t.Fatal(err)
	}
	var receipt rawAgentReceipt
	if err := json.Unmarshal(body, &receipt); err != nil {
		t.Fatal(err)
	}
	if receipt.Schema != rawAgentReceiptSchema || receipt.Metrics.Arm != "baseline" || !receipt.Metrics.TaskCompleted {
		t.Fatalf("receipt = %#v", receipt)
	}
	// Verify zero kernel mediation occurred:
	if receipt.Metrics.Repairs != 0 || receipt.Metrics.VDSOHits != 0 || receipt.Metrics.Denies != 0 || receipt.Metrics.Quarantines != 0 {
		t.Fatalf("kernel mediation occurred in raw mode: %#v", receipt.Metrics)
	}
	// Verify unmediated baseline behavior: destructive operation executed, injection in context, unrepaired syntax error.
	if !receipt.Metrics.DestructiveExecuted {
		t.Fatalf("destructive op was expected to execute in raw baseline mode: %#v", receipt.Metrics)
	}
	if !receipt.Metrics.InjectionInContext {
		t.Fatalf("injection was expected to reach context in raw baseline mode: %#v", receipt.Metrics)
	}
	if receipt.Metrics.ToolErrors == 0 {
		t.Fatalf("tool errors were expected from unmediated args in raw baseline mode: %#v", receipt.Metrics)
	}
	if receipt.Metrics.FinalAnswer == "" || receipt.Metrics.Turns != 9 {
		t.Fatalf("raw run envelope = %#v", receipt.Metrics)
	}
}

func TestRawAgentRespectsTaskOfflineAndOutFlags(t *testing.T) {
	customTask := "Plan trip to New York"
	out := filepath.Join(t.TempDir(), "custom_raw.json")
	captureAgentStdio(t, func() {
		cmdAgent([]string{"--raw", "--task", customTask, "--offline", "--out", out})
	})

	body, err := os.ReadFile(out)
	if err != nil {
		t.Fatalf("failed to read output file: %v", err)
	}
	var receipt rawAgentReceipt
	if err := json.Unmarshal(body, &receipt); err != nil {
		t.Fatal(err)
	}
	if receipt.Schema != rawAgentReceiptSchema {
		t.Fatalf("schema = %q, want %q", receipt.Schema, rawAgentReceiptSchema)
	}
	if receipt.Task != customTask {
		t.Fatalf("task = %q, want %q", receipt.Task, customTask)
	}
	if receipt.Metrics.Arm != "baseline" {
		t.Fatalf("metrics arm = %q, want baseline", receipt.Metrics.Arm)
	}
}

func TestRawAgentModeFlag(t *testing.T) {
	outRaw := filepath.Join(t.TempDir(), "mode_raw.json")
	captureAgentStdio(t, func() {
		cmdAgent([]string{"--mode", "raw", "--offline", "--out", outRaw})
	})
	bodyRaw, err := os.ReadFile(outRaw)
	if err != nil {
		t.Fatal(err)
	}
	var recRaw rawAgentReceipt
	if err := json.Unmarshal(bodyRaw, &recRaw); err != nil {
		t.Fatal(err)
	}
	if recRaw.Schema != rawAgentReceiptSchema || recRaw.Metrics.Arm != "baseline" {
		t.Fatalf("mode raw receipt = %#v", recRaw)
	}

	outNative := filepath.Join(t.TempDir(), "mode_native.json")
	captureAgentStdio(t, func() {
		cmdAgent([]string{"--mode", "native", "--offline", "--out", outNative})
	})
	bodyNative, err := os.ReadFile(outNative)
	if err != nil {
		t.Fatal(err)
	}
	var recNative nativeAgentReceipt
	if err := json.Unmarshal(bodyNative, &recNative); err != nil {
		t.Fatal(err)
	}
	if recNative.Schema != nativeAgentReceiptSchema || recNative.Metrics.Arm != "fak" {
		t.Fatalf("mode native receipt = %#v", recNative)
	}
}

func TestRawAgentZeroAdjudicationsWitness(t *testing.T) {
	// Tighten the adjudicator policy to deny the tools the task uses.
	// If adjudicator were called, these would be denied.
	prevPolicy := adjudicator.Default.PolicySnapshot()
	defer adjudicator.Default.SetPolicy(prevPolicy)
	adjudicator.Default.SetPolicy(adjudicator.Policy{
		Deny: map[string]abi.ReasonCode{
			"get_user":       abi.ReasonPolicyBlock,
			"search_flights": abi.ReasonPolicyBlock,
			"book_flight":    abi.ReasonPolicyBlock,
		},
	})

	out := filepath.Join(t.TempDir(), "unmediated.json")
	captureAgentStdio(t, func() {
		cmdAgent([]string{"--raw", "--offline", "--out", out})
	})

	body, err := os.ReadFile(out)
	if err != nil {
		t.Fatal(err)
	}
	var receipt rawAgentReceipt
	if err := json.Unmarshal(body, &receipt); err != nil {
		t.Fatal(err)
	}
	// Zero adjudications occurred: the policy deny was never triggered because the kernel/adjudicator is bypassed.
	if receipt.Metrics.Denies != 0 {
		t.Fatalf("expected 0 denies in raw mode, got %d", receipt.Metrics.Denies)
	}
	if !receipt.Metrics.TaskCompleted {
		t.Fatalf("task was expected to complete unmediated despite policy block on adjudicator: %#v", receipt.Metrics)
	}
}

func TestResolveAgentMode(t *testing.T) {
	cases := []struct {
		raw        bool
		native     bool
		mode       string
		wantRaw    bool
		wantNative bool
		wantErr    bool
	}{
		{raw: true, native: false, mode: "", wantRaw: true, wantNative: false, wantErr: false},
		{raw: false, native: true, mode: "", wantRaw: false, wantNative: true, wantErr: false},
		{raw: false, native: false, mode: "raw", wantRaw: true, wantNative: false, wantErr: false},
		{raw: false, native: false, mode: "native", wantRaw: false, wantNative: true, wantErr: false},
		{raw: false, native: false, mode: "ab", wantRaw: false, wantNative: false, wantErr: false},
		{raw: false, native: false, mode: "", wantRaw: false, wantNative: false, wantErr: false},
		{raw: true, native: true, mode: "", wantErr: true},
		{raw: true, native: false, mode: "native", wantErr: true},
		{raw: false, native: true, mode: "raw", wantErr: true},
		{raw: false, native: false, mode: "invalid", wantErr: true},
	}
	for _, tc := range cases {
		gotRaw, gotNative, err := resolveAgentMode(tc.raw, tc.native, tc.mode)
		if tc.wantErr {
			if err == nil {
				t.Errorf("resolveAgentMode(%v, %v, %q) expected error, got nil", tc.raw, tc.native, tc.mode)
			}
			continue
		}
		if err != nil {
			t.Errorf("resolveAgentMode(%v, %v, %q) unexpected error: %v", tc.raw, tc.native, tc.mode, err)
			continue
		}
		if gotRaw != tc.wantRaw || gotNative != tc.wantNative {
			t.Errorf("resolveAgentMode(%v, %v, %q) = (%v, %v), want (%v, %v)", tc.raw, tc.native, tc.mode, gotRaw, gotNative, tc.wantRaw, tc.wantNative)
		}
	}
}

// TestResolveStreamMode pins the --stream selector (fak-private#1019): the empty default and
// "auto" both heal a stream-only upstream, "on"/"off" are honored verbatim, and a typo is
// rejected so it can never silently pick the buffered arm.
func TestResolveStreamMode(t *testing.T) {
	cases := []struct {
		in      string
		want    string
		wantErr bool
	}{
		{"", streamAuto, false},
		{"auto", streamAuto, false},
		{"AUTO", streamAuto, false},
		{"on", streamOn, false},
		{"off", streamOff, false},
		{"  On ", streamOn, false},
		{"maybe", "", true},
	}
	for _, tc := range cases {
		got, err := resolveStreamMode(tc.in)
		if tc.wantErr {
			if err == nil {
				t.Errorf("resolveStreamMode(%q) expected error, got %q", tc.in, got)
			}
			continue
		}
		if err != nil {
			t.Errorf("resolveStreamMode(%q) unexpected error: %v", tc.in, err)
			continue
		}
		if got != tc.want {
			t.Errorf("resolveStreamMode(%q) = %q, want %q", tc.in, got, tc.want)
		}
	}
}

// TestPlannerSupportsStreaming pins the CLI gate that decides whether the native arm may use
// the streaming path: an OpenAI-wire planner is streaming-capable, an Anthropic-wire planner
// (not streamable today) and a non-streaming mock planner are not. This is the predicate that
// keeps --stream=auto byte-identical to the old buffered arm for every wire that cannot stream.
func TestPlannerSupportsStreaming(t *testing.T) {
	openai, err := agent.NewProviderHTTPPlanner("openai", "http://127.0.0.1:0", "m", "")
	if err != nil {
		t.Fatal(err)
	}
	if !plannerSupportsStreaming(openai) {
		t.Error("openai wire must be streaming-capable")
	}
	anthropic, err := agent.NewProviderHTTPPlanner("anthropic", "http://127.0.0.1:0", "m", "")
	if err != nil {
		t.Fatal(err)
	}
	if plannerSupportsStreaming(anthropic) {
		t.Error("anthropic wire must NOT be streaming-capable (buffered arm stays byte-identical)")
	}
	if plannerSupportsStreaming(agent.NewMockPlanner("mock")) {
		t.Error("the mock planner (no StreamingPlanner) must NOT be streaming-capable")
	}
}
