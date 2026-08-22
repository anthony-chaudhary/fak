package main_test

import (
	"bufio"
	"bytes"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/harnessserver"
	"github.com/anthony-chaudhary/fak/internal/serverlifecycle"
	"github.com/anthony-chaudhary/fak/internal/serverproduct"
)

type selfcheckResult struct {
	Schema   string `json:"schema"`
	Verdict  string `json:"verdict"`
	Boundary struct {
		CleanProductDirectories int      `json:"clean_product_directories"`
		ServerRoot              string   `json:"server_root"`
		HarnessRoot             string   `json:"harness_root"`
		DistinctRoots           bool     `json:"distinct_roots"`
		ReceiptOnlyHandoff      bool     `json:"receipt_only_handoff"`
		CrossedPaths            []string `json:"crossed_paths"`
		SharedMutablePaths      []string `json:"shared_mutable_paths"`
	} `json:"boundary"`
	Receipt struct {
		Schema           string   `json:"schema"`
		Digest           string   `json:"digest"`
		SpecDigest       string   `json:"spec_digest"`
		Generation       uint64   `json:"generation"`
		ArtifactName     string   `json:"artifact_name"`
		ArtifactDigest   string   `json:"artifact_digest"`
		AdapterName      string   `json:"adapter_name"`
		AdapterVersion   string   `json:"adapter_version"`
		ExecutableDigest string   `json:"executable_digest"`
		ReadinessProbe   string   `json:"readiness_probe"`
		ReadinessDigest  string   `json:"readiness_digest"`
		Capabilities     []string `json:"capabilities"`
		EventLogDigest   string   `json:"event_log_digest"`
		ReceiptUnchanged bool     `json:"receipt_unchanged"`
	} `json:"receipt"`
	Harness struct {
		Schema              string `json:"schema"`
		BindingDigest       string `json:"binding_digest"`
		PinnedReceiptDigest string `json:"pinned_receipt_digest"`
		Generation          uint64 `json:"generation"`
		ModelAlias          string `json:"model_alias"`
		ProtocolFamily      string `json:"protocol_family"`
		ProtocolRevision    string `json:"protocol_revision"`
		ChatCalls           int    `json:"chat_calls"`
		ChatResponse        string `json:"chat_response"`
		LifecycleCalls      int    `json:"lifecycle_calls"`
	} `json:"harness"`
	Teardown struct {
		State                 string `json:"state"`
		Owned                 bool   `json:"owned"`
		InstanceIDMatch       bool   `json:"instance_id_match"`
		GenerationMatch       bool   `json:"generation_match"`
		ProcessIDMatch        bool   `json:"process_id_match"`
		OwnedTeardowns        int    `json:"owned_teardowns"`
		HarnessLifecycleCalls int    `json:"harness_lifecycle_calls"`
	} `json:"teardown"`
	ReadinessProbes         int `json:"readiness_probes"`
	ExternalNetworkRequests int `json:"external_network_requests"`
	Phases                  []struct {
		Name      string `json:"name"`
		ElapsedMS int64  `json:"elapsed_ms"`
	} `json:"phases"`
}

type eventRecord struct {
	Schema    string `json:"schema"`
	Kind      string `json:"kind"`
	Actor     string `json:"actor"`
	Operation string `json:"operation"`
	Path      string `json:"path,omitempty"`
	Success   bool   `json:"success"`
}

func TestSelfcheckCapturesReceiptHandoffAcrossSeparateProducts(t *testing.T) {
	root := t.TempDir()
	ctxDeadline := 45 * time.Second
	cmd := exec.Command("go", "run", ".", "-selfcheck", "-work-root", root)
	cmd.Env = append(os.Environ(), "GOTOOLCHAIN=auto")
	var stdout, stderr bytes.Buffer
	cmd.Stdout, cmd.Stderr = &stdout, &stderr
	done := make(chan error, 1)
	go func() { done <- cmd.Run() }()
	select {
	case err := <-done:
		if err != nil {
			t.Fatalf("selfcheck: %v\nstderr:\n%s", err, stderr.String())
		}
	case <-time.After(ctxDeadline):
		_ = cmd.Process.Kill()
		t.Fatalf("selfcheck exceeded %s", ctxDeadline)
	}

	var got selfcheckResult
	decoder := json.NewDecoder(&stdout)
	decoder.DisallowUnknownFields()
	if err := decoder.Decode(&got); err != nil {
		t.Fatalf("decode selfcheck: %v\n%s", err, stdout.String())
	}
	if got.Schema != "fak.independent-server-selfcheck/v1" || got.Verdict != "pass" {
		t.Fatalf("selfcheck verdict=%+v", got)
	}
	if got.Boundary.CleanProductDirectories != 2 || !got.Boundary.DistinctRoots || !got.Boundary.ReceiptOnlyHandoff ||
		len(got.Boundary.CrossedPaths) != 1 || got.Boundary.CrossedPaths[0] != serverlifecycle.ReceiptFilename || len(got.Boundary.SharedMutablePaths) != 0 {
		t.Fatalf("boundary=%+v", got.Boundary)
	}
	if got.ReadinessProbes != 3 || got.Harness.ChatCalls != 1 || got.Harness.ChatResponse != "HANDOFF_OK" ||
		got.Teardown.OwnedTeardowns != 1 || got.Teardown.HarnessLifecycleCalls != 0 || got.ExternalNetworkRequests != 0 {
		t.Fatalf("operating envelope=%+v", got)
	}
	if len(got.Phases) != 4 {
		t.Fatalf("phases=%+v", got.Phases)
	}
	for i, name := range []string{"init", "up", "harness", "down"} {
		if got.Phases[i].Name != name || got.Phases[i].ElapsedMS < 0 {
			t.Fatalf("phase[%d]=%+v", i, got.Phases[i])
		}
	}

	serverRoot := filepath.Join(root, got.Boundary.ServerRoot)
	harnessRoot := filepath.Join(root, got.Boundary.HarnessRoot)
	assertDistinctRoots(t, serverRoot, harnessRoot)
	receiptPath := filepath.Join(serverRoot, serverlifecycle.ReceiptFilename)
	receiptRaw, err := os.ReadFile(receiptPath)
	if err != nil {
		t.Fatal(err)
	}
	receipt, err := serverproduct.DecodeReceipt(receiptRaw)
	if err != nil {
		t.Fatal(err)
	}
	if digest(receiptRaw) != got.Receipt.Digest || receipt.Generation != got.Receipt.Generation ||
		receipt.Artifact.Digest != got.Receipt.ArtifactDigest || receipt.Readiness.ProbeDigest != got.Receipt.ReadinessDigest {
		t.Fatalf("receipt reread mismatch: receipt=%+v result=%+v", receipt, got.Receipt)
	}
	bindingPath := filepath.Join(harnessRoot, harnessserver.BindingFileName)
	binding, err := harnessserver.ReadBinding(bindingPath)
	if err != nil {
		t.Fatal(err)
	}
	bindingRaw, err := os.ReadFile(bindingPath)
	if err != nil {
		t.Fatal(err)
	}
	if filepath.Clean(binding.ReceiptPath) != filepath.Clean(receiptPath) || binding.HarnessDirectory != filepath.Clean(harnessRoot) ||
		digest(bindingRaw) != got.Harness.BindingDigest || binding.ReceiptDigest != got.Receipt.Digest {
		t.Fatalf("binding reread mismatch: binding=%+v", binding)
	}

	eventRaw, events := readEvents(t, filepath.Join(serverRoot, "events.jsonl"))
	if digest(eventRaw) != got.Receipt.EventLogDigest {
		t.Fatal("event log digest changed after products exited")
	}
	counts := map[string]int{}
	for _, event := range events {
		if event.Schema != "fak.independent-server-event/v1" || !event.Success {
			t.Fatalf("event=%+v", event)
		}
		counts[event.Kind+"/"+event.Actor+"/"+event.Operation]++
	}
	if counts["probe/server-product/health"] != 1 || counts["probe/server-product/models"] != 1 ||
		counts["probe/server-product/readiness-chat"] != 1 || counts["chat/harness-product/chat"] != 1 ||
		counts["lifecycle/server-product/down"] != 1 {
		t.Fatalf("event counts=%v", counts)
	}
	for key, count := range counts {
		if strings.HasPrefix(key, "lifecycle/harness-product/") && count != 0 {
			t.Fatalf("harness owned lifecycle event %q", key)
		}
	}
	stateRaw, err := os.ReadFile(filepath.Join(serverRoot, serverlifecycle.StateFilename))
	if err != nil {
		t.Fatal(err)
	}
	var state struct {
		State string `json:"state"`
	}
	if err := json.Unmarshal(stateRaw, &state); err != nil || state.State != "stopped" {
		t.Fatalf("final server state=%q err=%v", state.State, err)
	}
}

func assertDistinctRoots(t *testing.T, serverRoot, harnessRoot string) {
	t.Helper()
	rel, err := filepath.Rel(serverRoot, harnessRoot)
	if err != nil || rel == "." || !strings.HasPrefix(rel, ".."+string(filepath.Separator)) {
		t.Fatalf("product roots overlap: server=%s harness=%s", serverRoot, harnessRoot)
	}
	serverInfo, err := os.Stat(serverRoot)
	if err != nil {
		t.Fatal(err)
	}
	harnessInfo, err := os.Stat(harnessRoot)
	if err != nil {
		t.Fatal(err)
	}
	if os.SameFile(serverInfo, harnessInfo) {
		t.Fatal("product roots resolve to the same directory")
	}
}

func readEvents(t *testing.T, path string) ([]byte, []eventRecord) {
	t.Helper()
	raw, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	var events []eventRecord
	scanner := bufio.NewScanner(bytes.NewReader(raw))
	for scanner.Scan() {
		var event eventRecord
		if err := json.Unmarshal(scanner.Bytes(), &event); err != nil {
			t.Fatal(err)
		}
		events = append(events, event)
	}
	if err := scanner.Err(); err != nil {
		t.Fatal(err)
	}
	return raw, events
}

func digest(raw []byte) string {
	sum := sha256.Sum256(raw)
	return "sha256:" + hex.EncodeToString(sum[:])
}
