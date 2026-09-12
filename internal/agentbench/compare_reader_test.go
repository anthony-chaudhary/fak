package agentbench

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/agentbench/taskrun"
)

func TestCompareDirectoriesVerifiesFiveSharedC2Receipts(t *testing.T) {
	baseline := writeComparisonArmFixture(t, "baseline", false, false, false, false, false, false)
	candidate := writeComparisonArmFixture(t, "candidate", false, false, false, false, false, false)

	got, err := compareDirectories(baseline, candidate)
	if err != nil {
		t.Fatal(err)
	}
	if got.Status != "NOISY_INCONCLUSIVE" || got.Comparison.Pairs != 5 {
		t.Fatalf("five verified noisy pairs = %+v", got)
	}
	if len(got.BaselineObserved.Replicates) != 5 || len(got.CandidateObserved.Replicates) != 5 {
		t.Fatalf("serialized observations lost paired evidence: %+v", got)
	}
	for i, observed := range got.BaselineObserved.Replicates {
		if observed.Pair != i+1 || observed.Seed != 0xA63E1001+uint64(i) || observed.Order != i%2 || !observed.SharedC2.Qualified || !observed.OwnCapacity.Qualified || len(observed.MetricMillis) == 0 {
			t.Fatalf("baseline observation %d lost identity/metrics/qualification: %+v", i, observed)
		}
	}
	for name, verdict := range got.MetricVerdicts {
		if verdict.Status != "NOISY" {
			t.Fatalf("metric %s verdict=%+v, want noisy", name, verdict)
		}
	}

	t.Run("probe events cannot self-declare steady", func(t *testing.T) {
		probe := writeComparisonArmFixture(t, "probe", true, false, false, false, false, false)
		got, err := compareDirectories(baseline, probe)
		if err != nil {
			t.Fatal(err)
		}
		if got.Status != "INCONCLUSIVE" {
			t.Fatalf("probe-only artifact promoted as shared-C2 steady: %+v", got)
		}
		if len(got.BaselineObserved.Replicates) != 5 || len(got.CandidateObserved.Replicates) != 5 {
			t.Fatalf("inconclusive pair discarded verified observations before/after failure: %+v", got)
		}
		failed := got.CandidateObserved.Replicates[2]
		if failed.Pair != 3 || failed.Seed != 0xA63E1003 || failed.Order != 0 || failed.SharedC2.Qualified || len(failed.MetricMillis) != 0 {
			t.Fatalf("failed pair did not retain identity with unknown qualification: %+v", failed)
		}
		for i, observed := range got.CandidateObserved.Replicates {
			if i != 2 && (!observed.SharedC2.Qualified || len(observed.MetricMillis) == 0) {
				t.Fatalf("valid observation %d around failed pair lost metrics: %+v", i+1, observed)
			}
		}
	})
	t.Run("unknown identity is inconclusive", func(t *testing.T) {
		unknown := writeComparisonArmFixture(t, "unknown", false, true, false, false, false, false)
		got, err := compareDirectories(baseline, unknown)
		if err != nil {
			t.Fatal(err)
		}
		if got.Status != "INCONCLUSIVE" {
			t.Fatalf("unknown tokenizer identity promoted: %+v", got)
		}
	})
	t.Run("self-declared task acceptance is insufficient", func(t *testing.T) {
		failed := writeComparisonArmFixture(t, "failed-tasks", false, false, true, false, false, false)
		got, err := compareDirectories(baseline, failed)
		if err != nil {
			t.Fatal(err)
		}
		if got.Status != "INCONCLUSIVE" {
			t.Fatalf("failed task artifacts promoted by receipt boolean: %+v", got)
		}
	})
	t.Run("rehash cannot hide context identity mismatch", func(t *testing.T) {
		mismatch := writeComparisonArmFixture(t, "context-mismatch", false, false, false, true, false, false)
		got, err := compareDirectories(baseline, mismatch)
		if err != nil {
			t.Fatal(err)
		}
		if got.Status != "INCONCLUSIVE" {
			t.Fatalf("used tokenizer mismatch promoted after honest rehash: %+v", got)
		}
	})
	t.Run("missing optional bootstrap stays unknown", func(t *testing.T) {
		missing := writeComparisonArmFixture(t, "missing-bootstrap", false, false, false, false, false, true)
		got, err := compareDirectories(baseline, missing)
		if err != nil {
			t.Fatal(err)
		}
		if len(got.CandidateObserved.Replicates) != 5 {
			t.Fatalf("missing optional bootstrap dropped otherwise verified observations: %+v", got)
		}
		for _, replicate := range got.CandidateObserved.Replicates {
			if replicate.BootstrapKnown {
				t.Fatalf("completion latency substituted for missing bootstrap: %+v", replicate)
			}
		}
		if _, substituted := got.Comparison.Metrics["bootstrap_ms"]; substituted {
			t.Fatalf("optional unknown bootstrap was compared: %+v", got.Comparison.Metrics)
		}
	})
	t.Run("changed oracle changes immutable task witness", func(t *testing.T) {
		changed := writeComparisonArmFixture(t, "changed-oracle", false, false, false, false, true, false)
		got, err := compareDirectories(baseline, changed)
		if err != nil {
			t.Fatal(err)
		}
		if got.Status != "INCONCLUSIVE" {
			t.Fatalf("changed oracle with same candidate code remained comparable: %+v", got)
		}
	})
	t.Run("tampered artifact is corrupt input", func(t *testing.T) {
		tampered := writeComparisonArmFixture(t, "tampered", false, false, false, false, false, false)
		if err := os.WriteFile(filepath.Join(tampered, "pair-1", "normal-events.jsonl"), []byte("tampered\n"), 0600); err != nil {
			t.Fatal(err)
		}
		if _, err := compareDirectories(baseline, tampered); err == nil {
			t.Fatal("artifact digest corruption returned a normal comparison result")
		}
	})
}

func writeComparisonArmFixture(t *testing.T, armID string, probeOnly, unknownIdentity, failedTasks, contextMismatch, definitionMismatch, omitBootstrap bool) string {
	t.Helper()
	root := t.TempDir()
	refs := make([]comparisonReplicateRef, 0, 5)
	var toolContractDigest, taskWitnessDigest string
	for pair := 1; pair <= 5; pair++ {
		dir := filepath.Join(root, fmt.Sprintf("pair-%d", pair))
		if err := os.MkdirAll(filepath.Join(dir, "tasks-c2"), 0700); err != nil {
			t.Fatal(err)
		}
		artifacts := map[string]string{}
		manifestBody := []byte(`{"schema":"fak.agentbench.manifest.v1","profile":"normal","resolved_model":"fixture-model"}`)
		if err := os.WriteFile(filepath.Join(dir, "manifest.json"), manifestBody, 0600); err != nil {
			t.Fatal(err)
		}
		artifacts["manifest.json"] = comparisonTestDigest(manifestBody)
		contextBody, toolDigest := comparisonContextManifest(t, contextMismatch)
		if err := os.WriteFile(filepath.Join(dir, "normal-context-manifest.json"), contextBody, 0600); err != nil {
			t.Fatal(err)
		}
		artifacts["normal-context-manifest.json"] = comparisonTestDigest(contextBody)
		toolContractDigest = toolDigest
		tasks := make([]map[string]any, 6)
		definitions := make([]struct{ ID, DefinitionSHA256 string }, 6)
		for i := range tasks {
			accepted := !failedTasks
			id := fmt.Sprintf("task-%d", i+1)
			definition := comparisonTestDigest([]byte(fmt.Sprintf("fixture-definition-%d-with-visible-oracle-workflow-policy", i+1)))
			if definitionMismatch && i == 2 {
				definition = comparisonTestDigest([]byte("same candidate code but changed hidden oracle"))
			}
			definitions[i] = struct{ ID, DefinitionSHA256 string }{id, definition}
			tasks[i] = map[string]any{"id": id, "family": fmt.Sprintf("family-%d", i%3), "definition_sha256": definition, "passed": accepted, "accepted": accepted, "external_passed": accepted, "controls": map[string]any{"broken_digest": comparisonTestDigest([]byte("broken" + id)), "fixed_digest": comparisonTestDigest([]byte("fixed" + id))}, "model_observation": map[string]any{"requests": 4, "upstream_requests": 4, "successful_requests": 4, "model_identity_status": "matched", "events_digest": strings.Repeat("a", 64)}}
		}
		taskBody, _ := json.Marshal(map[string]any{"schema": "fak.agentbench.taskrun.v1", "configured_concurrency": 2, "concurrency_qualified": true, "tasks": tasks})
		taskPath := filepath.Join(dir, "tasks-c2", "receipt.json")
		if err := os.WriteFile(taskPath, taskBody, 0600); err != nil {
			t.Fatal(err)
		}
		artifacts["tasks-c2/receipt.json"] = comparisonTestDigest(taskBody)
		definitionBytes, _ := json.Marshal(definitions)
		taskWitnessDigest = comparisonTestDigest(definitionBytes)
		phase := "steady"
		if probeOnly && pair == 3 {
			phase = "probe"
		}
		var eventText strings.Builder
		delta := time.Duration(100) * time.Millisecond
		if armID != "baseline" {
			delta += time.Duration([]int{-2, 2, -1, 1, 0}[pair-1]) * time.Millisecond
		}
		if !omitBootstrap {
			started := time.Date(2026, 9, 10, 11, pair, 0, 0, time.UTC)
			fmt.Fprintf(&eventText, "{\"event\":\"bootstrap\",\"event_type\":\"terminal\",\"phase\":\"precondition\",\"rung\":\"bootstrap-c2\",\"status\":\"completed\",\"scored_request\":false,\"started_at\":%q,\"completed_at\":%q,\"duration_ms\":25}\n", started.Format(time.RFC3339Nano), started.Add(25*time.Millisecond).Format(time.RFC3339Nano))
		}
		for request := 1; request <= 96; request++ {
			started := time.Date(2026, 9, 10, 12, pair, request, 0, time.UTC)
			fmt.Fprintf(&eventText, "{\"event\":\"request_started\",\"event_type\":\"start\",\"phase\":\"start\",\"sequence\":%d,\"request_id\":\"p%d-r%d\",\"rung\":\"steady-c2\",\"session\":\"S%d\",\"turn\":%d,\"status\":\"in_flight\",\"scored_request\":true,\"started_at\":%q}\n", request, pair, request, (request-1)/12+1, (request-1)%12+1, started.Format(time.RFC3339Nano))
			fmt.Fprintf(&eventText, "{\"event\":\"first_output\",\"event_type\":\"first_output\",\"phase\":\"first_output\",\"sequence\":%d,\"request_id\":\"p%d-r%d\",\"rung\":\"steady-c2\",\"session\":\"S%d\",\"turn\":%d,\"status\":\"first_output\",\"scored_request\":true,\"started_at\":%q}\n", request, pair, request, (request-1)/12+1, (request-1)%12+1, started.Add(delta/2).Format(time.RFC3339Nano))
			fmt.Fprintf(&eventText, "{\"event\":\"end\",\"event_type\":\"terminal\",\"phase\":%q,\"sequence\":%d,\"request_id\":\"p%d-r%d\",\"rung\":\"steady-c2\",\"session\":\"S%d\",\"turn\":%d,\"status\":\"completed\",\"service_verdict\":\"OBSERVED_OK\",\"model_identity_status\":\"matched\",\"scored_request\":true,\"started_at\":%q,\"completed_at\":%q,\"duration_ms\":%d}\n", phase, request, pair, request, (request-1)/12+1, (request-1)%12+1, started.Format(time.RFC3339Nano), started.Add(delta).Format(time.RFC3339Nano), delta.Milliseconds())
		}
		events := []byte(eventText.String())
		if err := os.WriteFile(filepath.Join(dir, "normal-events.jsonl"), events, 0600); err != nil {
			t.Fatal(err)
		}
		artifacts["normal-events.jsonl"] = comparisonTestDigest(events)
		var taskReceipt taskrun.Receipt
		if err := json.Unmarshal(taskBody, &taskReceipt); err != nil {
			t.Fatal(err)
		}
		receipt := normalRunReceipt{Schema: "fak.agentbench.normal-run.v1", Status: "PASSED", OverallVerdict: "NORMAL_PASSED", ResolvedModel: "fixture-model", RequestCeiling: 272, ServiceQualification: normalQualification{Qualified: true}, SharedC2Tasks: &taskReceipt, OwnCapacityTasks: &taskReceipt, StartedAt: time.Now().Add(-time.Second), Artifacts: []string{filepath.Join(dir, "manifest.json"), filepath.Join(dir, "normal-context-manifest.json"), filepath.Join(dir, "normal-events.jsonl"), taskPath}}
		if _, err := finishNormalReceipt(receipt, dir, nil); err != nil {
			t.Fatal(err)
		}
		rel := fmt.Sprintf("pair-%d/receipt.json", pair)
		receiptBody, err := os.ReadFile(filepath.Join(root, rel))
		if err != nil {
			t.Fatal(err)
		}
		refs = append(refs, comparisonReplicateRef{Pair: pair, Seed: 0xA63E1001 + uint64(pair-1), Order: (pair - 1) % 2, Receipt: rel, ReceiptSHA256: comparisonTestDigest(receiptBody)})
	}
	tokenizer := "tokenizer-digest"
	if unknownIdentity {
		tokenizer = ""
	}
	arm := comparisonArmArtifact{Schema: comparisonArmSchema, ArmID: armID, ManifestSHA256: "immutable-workload-spec", ModelID: "fixture-model", TokenizerSHA256: tokenizer, RendererSHA256: "renderer", WeightSHA256: "weights", Quantization: "q4", ContextTokens: 32768, SamplingSHA256: "sampling", OutputPolicySHA256: "output", ToolContractSHA256: toolContractDigest, TaskWitnessSHA256: taskWitnessDigest, ResourceSHA256: "resource", LaunchSHA256: "launch", CachePolicySHA256: "cache", LoadScheduleSHA256: "load", QualifiedConcurrency: 2, Replicates: refs}
	body, _ := json.Marshal(arm)
	if err := os.WriteFile(filepath.Join(root, "comparison-arm.json"), body, 0600); err != nil {
		t.Fatal(err)
	}
	return root
}

func comparisonTestDigest(body []byte) string {
	sum := sha256.Sum256(body)
	return hex.EncodeToString(sum[:])
}

func comparisonContextManifest(t *testing.T, mismatch bool) ([]byte, string) {
	t.Helper()
	tools := []any{map[string]any{"type": "function", "function": map[string]any{"name": "Read", "parameters": map[string]any{"type": "object"}}}}
	turns := make([]any, 12)
	for turn := range turns {
		tokenizer := "tokenizer-digest"
		if mismatch && turn == 5 {
			tokenizer = "different-tokenizer"
		}
		output := 128
		if turn >= 8 && turn < 11 {
			output = 256
		} else if turn == 11 {
			output = 1024
		}
		turns[turn] = map[string]any{"ordinal": turn + 1, "output_tokens": output, "request": map[string]any{"messages": []any{map[string]any{"role": "user", "content": fmt.Sprintf("turn-%d", turn+1)}}, "tools": tools, "max_tokens": output}, "encoding": map[string]any{"model_id": "fixture-model", "renderer_id": "renderer", "tokenizer_id": tokenizer, "token_ids": []int{1, 2, 3}, "prompt_tokens": 3, "context_window_tokens": 32768, "reserved_output_tokens": output, "rendered_sha256": strings.Repeat("a", 64)}}
	}
	sessions := make([]any, 8)
	for i := range sessions {
		sessions[i] = map[string]any{"id": fmt.Sprintf("S%d", i+1), "area": "A", "turns": turns}
	}
	preconditions := map[string]any{}
	for _, c := range []int{1, 2, 4, 8} {
		preconditions[fmt.Sprint(c)] = map[string]any{"concurrency": c, "request": map[string]any{"messages": []any{map[string]any{"role": "user", "content": fmt.Sprintf("precondition-c%d", c)}}, "tools": tools, "max_tokens": 128}, "encoding": map[string]any{"model_id": "fixture-model", "renderer_id": "renderer", "tokenizer_id": "tokenizer-digest", "token_ids": []int{1}, "prompt_tokens": 1, "context_window_tokens": 32768, "reserved_output_tokens": 128, "rendered_sha256": strings.Repeat("b", 64)}}
	}
	body, err := json.Marshal(map[string]any{"schema": "fak.agentbench.normal-corpus.v1", "sessions": sessions, "preconditions": preconditions})
	if err != nil {
		t.Fatal(err)
	}
	toolBytes, _ := json.Marshal(tools)
	return body, comparisonTestDigest(toolBytes)
}

func TestCompareCLIUsesVerifiedDirectoryReceipts(t *testing.T) {
	baseline := writeComparisonArmFixture(t, "cli-baseline", false, false, false, false, false, false)
	candidate := writeComparisonArmFixture(t, "cli-candidate", false, false, false, false, false, false)
	var stdout, stderr bytes.Buffer
	if code := RunCLI(context.Background(), &stdout, &stderr, []string{"compare", baseline, candidate}); code != 0 {
		t.Fatalf("compare CLI code=%d stderr=%s", code, stderr.String())
	}
	var receipt compareDirectoryReceipt
	if err := json.Unmarshal(stdout.Bytes(), &receipt); err != nil || receipt.Status != "NOISY_INCONCLUSIVE" {
		t.Fatalf("compare CLI receipt=%+v err=%v stdout=%s", receipt, err, stdout.String())
	}
	stdout.Reset()
	stderr.Reset()
	if code := RunCLI(context.Background(), &stdout, &stderr, []string{"compare", baseline}); code != 2 {
		t.Fatalf("wrong arity code=%d want2", code)
	}
	if err := os.WriteFile(filepath.Join(candidate, "pair-1", "normal-events.jsonl"), []byte("tampered\n"), 0600); err != nil {
		t.Fatal(err)
	}
	stdout.Reset()
	stderr.Reset()
	if code := RunCLI(context.Background(), &stdout, &stderr, []string{"compare", baseline, candidate}); code != 1 {
		t.Fatalf("tampered artifact code=%d want1 stdout=%s stderr=%s", code, stdout.String(), stderr.String())
	}
}
