package taskrun

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"regexp"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/agentbench/taskfixture"
)

func TestMain(m *testing.M) {
	if len(os.Args) >= 5 && os.Args[1] == "bench" && os.Args[2] == "agent" && (os.Args[3] == "task-child" || os.Args[3] == "test-child") {
		config := ""
		for i := 4; i+1 < len(os.Args); i++ {
			if os.Args[i] == "--config" {
				config = os.Args[i+1]
				break
			}
		}
		if config == "" {
			fmt.Fprintln(os.Stderr, "missing --config")
			os.Exit(2)
		}
		var err error
		if os.Args[3] == "task-child" {
			err = RunChild(context.Background(), config, os.Stdout)
		} else {
			err = RunTestChild(context.Background(), config, os.Stdout)
		}
		if err != nil {
			fmt.Fprintln(os.Stderr, err)
			os.Exit(1)
		}
		os.Exit(0)
	}
	os.Exit(m.Run())
}

func TestAgentBenchRejectsFalseSuccess(t *testing.T) {
	requireTaskSandbox(t)
	server, requests := newTaskPlannerServer(t, func(int) taskPlannerReply {
		return taskPlannerReply{content: "Done. The requested fix is complete."}
	})
	defer server.Close()

	ctx, cancel := context.WithTimeout(context.Background(), 90*time.Second)
	defer cancel()
	receipt, err := Run(ctx, taskOptions(t, server.URL, "false-success"))
	if err != nil {
		t.Fatalf("Run false-success cohort: %v", err)
	}
	if got := requests.Load(); got != 6 {
		t.Fatalf("false-success planner requests = %d, want one for each of six fixtures", got)
	}
	assertTaskOutcomes(t, receipt, 6, 0)
}

func TestAgentBenchTurnCap(t *testing.T) {
	requireTaskSandbox(t)
	server, requests := newTaskPlannerServer(t, func(id int) taskPlannerReply {
		return taskPlannerReply{toolID: fmt.Sprintf("read-%d", id), tool: "Read", arguments: `{"file_path":"missing-never-final.txt"}`}
	})
	defer server.Close()

	ctx, cancel := context.WithTimeout(context.Background(), 90*time.Second)
	defer cancel()
	receipt, err := Run(ctx, taskOptions(t, server.URL, "turn-cap"))
	if err != nil {
		t.Fatalf("Run never-final cohort: %v", err)
	}
	if got := requests.Load(); got > 6*12 {
		t.Fatalf("never-final cohort made %d HTTP requests, exceeds six tasks x 12 turns", got)
	}
	if got := requests.Load(); got < 6 {
		t.Fatalf("never-final cohort made only %d HTTP requests, did not exercise every task", got)
	}
	assertTaskOutcomes(t, receipt, 6, 0)
	assertNoTurnCountAbove(t, receipt, 12)
}

func TestAgentBenchUserRepoReadOnly(t *testing.T) {
	requireTaskSandbox(t)
	userRepo := t.TempDir()
	mustTaskWrite(t, filepath.Join(userRepo, "go.mod"), "module example.com/user-work\n\ngo 1.26\n")
	mustTaskWrite(t, filepath.Join(userRepo, "user.go"), "package userwork\nconst Sentinel = \"KEEP\"\n")
	before := taskTreeDigest(t, userRepo)
	oldWD, err := os.Getwd()
	if err != nil {
		t.Fatal(err)
	}
	if err := os.Chdir(userRepo); err != nil {
		t.Fatal(err)
	}
	defer func() {
		if err := os.Chdir(oldWD); err != nil {
			t.Errorf("restore cwd: %v", err)
		}
	}()

	server, _ := newTaskPlannerServer(t, func(int) taskPlannerReply { return taskPlannerReply{content: "No changes needed."} })
	defer server.Close()
	options := taskOptions(t, server.URL, "readonly")
	ctx, cancel := context.WithTimeout(context.Background(), 90*time.Second)
	defer cancel()
	if _, err := Run(ctx, options); err != nil {
		t.Fatalf("Run from user repository: %v", err)
	}
	if after := taskTreeDigest(t, userRepo); after != before {
		t.Fatalf("user repository changed: before %s after %s", before, after)
	}
}

func TestAgentBenchHappyInspectEditTest(t *testing.T) {
	requireTaskSandbox(t)
	fixture := taskfixture.Cases(false)[0]
	var requests atomic.Int64
	var sawVersion, sawSandboxCommand atomic.Bool
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var body map[string]any
		if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
			http.Error(w, err.Error(), http.StatusBadRequest)
			return
		}
		id := int(requests.Add(1))
		var answer taskPlannerReply
		switch id {
		case 1:
			answer = taskPlannerReply{toolID: "inspect", tool: "Read", arguments: `{"file_path":"` + fixture.TargetFile + `"}`}
		case 2:
			version := taskToolResultVersion(body)
			if version == "" {
				http.Error(w, "Read result did not expose a version", http.StatusBadRequest)
				return
			}
			sawVersion.Store(true)
			args, _ := json.Marshal(map[string]any{"file_path": fixture.TargetFile, "content": fixture.FixedSource, "mode": "overwrite", "expected_version": version})
			answer = taskPlannerReply{toolID: "edit", tool: "Write", arguments: string(args)}
		case 3:
			command := taskPromptTestCommand(body)
			if command == "" {
				http.Error(w, "prompt did not expose exact sandbox test command", http.StatusBadRequest)
				return
			}
			sawSandboxCommand.Store(true)
			args, _ := json.Marshal(map[string]any{"command": command})
			answer = taskPlannerReply{toolID: "test", tool: "Bash", arguments: string(args)}
		default:
			answer = taskPlannerReply{content: "The requested change is implemented and its exact test command passed."}
		}
		writeTaskPlannerReply(w, body, answer)
	}))
	defer server.Close()

	options := taskOptions(t, server.URL, "happy")
	options.TaskLimit = 1
	options.Concurrency = 1
	ctx, cancel := context.WithTimeout(context.Background(), 90*time.Second)
	defer cancel()
	receipt, err := Run(ctx, options)
	if err != nil {
		t.Fatalf("Run happy task: %v", err)
	}
	if len(receipt.Tasks) != 1 {
		t.Fatalf("task receipts = %d, want 1", len(receipt.Tasks))
	}
	got := receipt.Tasks[0]
	if !got.Accepted || !got.ExternalPassed {
		t.Fatalf("external oracle rejected happy task: %+v", got)
	}
	if !got.Inspected || !got.Edited || !got.TestSucceeded || got.DiffDigest == "" {
		t.Fatalf("inspect/edit/test/diff evidence incomplete: %+v", got)
	}
	if got.PlannerCalls > 12 || got.PlannerCalls != int(requests.Load()) {
		t.Fatalf("planner calls receipt/HTTP = %d/%d, cap 12", got.PlannerCalls, requests.Load())
	}
	if !sawVersion.Load() || !sawSandboxCommand.Load() {
		t.Fatalf("live transcript evidence missing: read-version=%v sandbox-command=%v", sawVersion.Load(), sawSandboxCommand.Load())
	}
	artifactPath := filepath.Join(options.OutDir, got.ID, "task.json")
	artifactBody, err := os.ReadFile(artifactPath)
	if err != nil {
		t.Fatalf("read persisted task receipt: %v", err)
	}
	var artifact TaskReceipt
	if err := json.Unmarshal(artifactBody, &artifact); err != nil {
		t.Fatalf("decode persisted task receipt: %v", err)
	}
	if artifact.Duration <= 0 || artifact.Duration != got.Duration {
		t.Fatalf("persisted/aggregate duration = %s/%s, want identical positive duration", artifact.Duration, got.Duration)
	}
	if artifact.Accepted != got.Accepted || artifact.Passed != got.Passed {
		t.Fatalf("persisted/aggregate outcome differs: artifact accepted=%v passed=%v; aggregate accepted=%v passed=%v", artifact.Accepted, artifact.Passed, got.Accepted, got.Passed)
	}
}

func TestAgentBenchOracleMustExecute(t *testing.T) {
	requireTaskSandbox(t)
	fixture := taskfixture.Cases(false)[0]
	for _, candidate := range taskfixture.Cases(true) {
		if err := validateCandidateSource(candidate.BrokenSource, candidate.FixedSource); err != nil {
			t.Fatalf("known body-only fix %s rejected: %v", candidate.ID, err)
		}
	}
	malicious := "package fixture\nimport \"os\"\nfunc init(){ os.Exit(0) }\n" + strings.TrimPrefix(fixture.FixedSource, "package fixture\n")
	rawTrial, oracle := t.TempDir(), t.TempDir()
	if err := seedFixture(rawTrial, fixture, malicious, true); err != nil {
		t.Fatal(err)
	}
	mustTaskWrite(t, filepath.Join(oracle, "hidden"), "protected")
	result, err := RunSandboxedTests(context.Background(), rawTrial, oracle)
	if err != nil || result.ExitCode != 0 {
		t.Fatalf("probe no longer demonstrates why fixture-aware validation is required: result=%+v err=%v", result, err)
	}
	directRejects := map[string]string{
		"malformed":             "package fixture\nfunc RetryDelay(",
		"changed import":        malicious,
		"changed package":       strings.Replace(fixture.FixedSource, "package fixture", "package other", 1),
		"changed signature":     strings.Replace(fixture.FixedSource, "attempt, base, cap int", "attempt, base int", 1),
		"changed name":          strings.Replace(fixture.FixedSource, "RetryDelay", "RetryDelayOther", 1),
		"added global":          strings.Replace(fixture.FixedSource, "func RetryDelay", "var escape = os.Exit\n\nfunc RetryDelay", 1),
		"extra function":        fixture.FixedSource + "\nfunc TestMain() { }\n",
		"directive outside":     strings.Replace(fixture.FixedSource, "func RetryDelay", "//go:noinline\nfunc RetryDelay", 1),
		"directive inside body": strings.Replace(fixture.FixedSource, "\tdelay :=", "\t//go:noinline\n\tdelay :=", 1),
		"outside comment":       strings.Replace(fixture.FixedSource, "func RetryDelay", "// changed outside body\nfunc RetryDelay", 1),
		"outside formatting":    strings.Replace(fixture.FixedSource, "package fixture\n\n", "package fixture\n\n\n", 1),
	}
	for name, candidate := range directRejects {
		t.Run("source "+name, func(t *testing.T) {
			if err := validateCandidateSource(fixture.BrokenSource, candidate); err == nil {
				t.Fatal("structural or directive change accepted")
			}
		})
	}
	multiOriginal := "package fixture\n\nvar stable = 1\n\nfunc Target() int { return 1 }\n\nfunc Helper() int { return 2 }\n"
	for name, candidate := range map[string]string{
		"removed declaration":    "package fixture\n\nfunc Target() int { return 3 }\n\nfunc Helper() int { return 2 }\n",
		"reordered declarations": "package fixture\n\nfunc Helper() int { return 2 }\n\nfunc Target() int { return 3 }\n\nvar stable = 1\n",
		"non-target initializer": "package fixture\n\nvar stable = 2\n\nfunc Target() int { return 3 }\n\nfunc Helper() int { return 2 }\n",
	} {
		t.Run("source "+name, func(t *testing.T) {
			if err := validateCandidateSource(multiOriginal, candidate); err == nil {
				t.Fatal("non-target declaration change accepted")
			}
		})
	}

	structuralChanges := map[string]string{
		"import and init": malicious,
		"global":          strings.Replace(fixture.FixedSource, "func RetryDelay", "var hiddenBypass = true\n\nfunc RetryDelay", 1),
		"extra function":  fixture.FixedSource + "\nfunc Unrelated() bool { return true }\n",
		"package":         strings.Replace(fixture.FixedSource, "package fixture", "package other", 1),
	}
	for name, source := range structuralChanges {
		t.Run(name, func(t *testing.T) {
			trial := t.TempDir()
			if err := seedFixture(trial, fixture, source, false); err != nil {
				t.Fatal(err)
			}
			if _, _, err := validateCandidate(trial, fixture); err == nil {
				t.Fatal("fixture acceptance admitted a structural change outside the target function body")
			}
		})
	}
	known := t.TempDir()
	if err := seedFixture(known, fixture, fixture.FixedSource, false); err != nil {
		t.Fatal(err)
	}
	if changed, _, err := validateCandidate(known, fixture); err != nil || !changed {
		t.Fatalf("known body-only fix rejected: changed=%v err=%v", changed, err)
	}
}

type taskPlannerReply struct {
	content   string
	toolID    string
	tool      string
	arguments string
}

func newTaskPlannerServer(t *testing.T, reply func(int) taskPlannerReply) (*httptest.Server, *atomic.Int64) {
	t.Helper()
	var requests atomic.Int64
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var body map[string]any
		if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
			http.Error(w, err.Error(), http.StatusBadRequest)
			return
		}
		if model, _ := body["model"].(string); model != "fixture-model" {
			http.Error(w, "wrong model", http.StatusBadRequest)
			return
		}
		id := int(requests.Add(1))
		writeTaskPlannerReply(w, body, reply(id))
	}))
	return server, &requests
}

func writeTaskPlannerReply(w http.ResponseWriter, body map[string]any, answer taskPlannerReply) {
	message := map[string]any{"role": "assistant", "content": answer.content}
	finish := "stop"
	if answer.tool != "" {
		finish = "tool_calls"
		message["tool_calls"] = []any{map[string]any{
			"id": answer.toolID, "type": "function",
			"function": map[string]any{"name": answer.tool, "arguments": answer.arguments},
		}}
	}
	if stream, _ := body["stream"].(bool); stream {
		w.Header().Set("Content-Type", "text/event-stream")
		encoded, _ := json.Marshal(map[string]any{"model": "fixture-model", "choices": []any{map[string]any{"delta": message, "finish_reason": finish}}, "usage": map[string]any{"prompt_tokens": 32, "completion_tokens": 4, "total_tokens": 36}})
		fmt.Fprintf(w, "data: %s\n\ndata: [DONE]\n\n", encoded)
		return
	}
	writeTaskJSON(w, map[string]any{"model": "fixture-model", "choices": []any{map[string]any{"message": message, "finish_reason": finish}}, "usage": map[string]any{"prompt_tokens": 32, "completion_tokens": 4, "total_tokens": 36}})
}

func taskToolResultVersion(body map[string]any) string {
	texts := taskJSONStrings(body)
	for i := len(texts) - 1; i >= 0; i-- {
		var result any
		if json.Unmarshal([]byte(texts[i]), &result) == nil {
			if version := taskFindNamedString(result, "version"); version != "" {
				return version
			}
		}
	}
	return ""
}

func taskFindNamedString(value any, name string) string {
	switch typed := value.(type) {
	case map[string]any:
		if text, ok := typed[name].(string); ok {
			return text
		}
		for _, item := range typed {
			if text := taskFindNamedString(item, name); text != "" {
				return text
			}
		}
	case []any:
		for _, item := range typed {
			if text := taskFindNamedString(item, name); text != "" {
				return text
			}
		}
	}
	return ""
}

var taskTestCommandPattern = regexp.MustCompile(`[^\s]+ bench agent test-child --config [^\s\x60]+`)

func taskPromptTestCommand(body map[string]any) string {
	for _, text := range taskJSONStrings(body) {
		if command := taskTestCommandPattern.FindString(text); command != "" {
			return command
		}
	}
	return ""
}

func taskJSONStrings(value any) []string {
	var out []string
	var visit func(any)
	visit = func(value any) {
		switch typed := value.(type) {
		case string:
			out = append(out, typed)
		case map[string]any:
			for _, item := range typed {
				visit(item)
			}
		case []any:
			for _, item := range typed {
				visit(item)
			}
		}
	}
	visit(value)
	return out
}

func taskOptions(t *testing.T, endpoint, name string) Options {
	t.Helper()
	executable, err := os.Executable()
	if err != nil {
		t.Fatal(err)
	}
	return Options{Executable: executable, Endpoint: endpoint, Model: "fixture-model", OutDir: filepath.Join(t.TempDir(), name), Concurrency: 2}
}

func requireTaskSandbox(t *testing.T) {
	t.Helper()
	if err := SandboxAvailable(); err != nil {
		t.Skipf("OS sandbox physically unavailable: %v", err)
	}
}

func assertTaskOutcomes(t *testing.T, receipt Receipt, wantTotal, wantAccepted int) {
	t.Helper()
	raw, err := json.Marshal(receipt)
	if err != nil {
		t.Fatal(err)
	}
	var value any
	if err := json.Unmarshal(raw, &value); err != nil {
		t.Fatal(err)
	}
	total, accepted := countTaskAccepted(value)
	if total != wantTotal || accepted != wantAccepted {
		t.Fatalf("task outcomes total/accepted = %d/%d, want %d/%d: %s", total, accepted, wantTotal, wantAccepted, raw)
	}
}

func countTaskAccepted(value any) (total, accepted int) {
	switch typed := value.(type) {
	case map[string]any:
		for key, item := range typed {
			if strings.EqualFold(key, "accepted") {
				if ok, isBool := item.(bool); isBool {
					total++
					if ok {
						accepted++
					}
				}
			}
		}
		for _, item := range typed {
			t, a := countTaskAccepted(item)
			total, accepted = total+t, accepted+a
		}
	case []any:
		for _, item := range typed {
			t, a := countTaskAccepted(item)
			total, accepted = total+t, accepted+a
		}
	}
	return total, accepted
}

func assertNoTurnCountAbove(t *testing.T, receipt Receipt, maximum int) {
	t.Helper()
	raw, err := json.Marshal(receipt)
	if err != nil {
		t.Fatal(err)
	}
	var value any
	if err := json.Unmarshal(raw, &value); err != nil {
		t.Fatal(err)
	}
	var visit func(any)
	visit = func(value any) {
		switch typed := value.(type) {
		case map[string]any:
			for key, item := range typed {
				norm := strings.ReplaceAll(strings.ToLower(key), "_", "")
				if norm == "turns" || norm == "modelturns" {
					if count, ok := item.(float64); ok && count > float64(maximum) {
						t.Errorf("%s = %v, exceeds %d", key, item, maximum)
					}
				}
				visit(item)
			}
		case []any:
			for _, item := range typed {
				visit(item)
			}
		}
	}
	visit(value)
}

func writeTaskJSON(w io.Writer, value any) {
	_ = json.NewEncoder(w).Encode(value)
}

func mustTaskWrite(t *testing.T, path, content string) {
	t.Helper()
	if err := os.WriteFile(path, []byte(content), 0o600); err != nil {
		t.Fatal(err)
	}
}

func taskTreeDigest(t *testing.T, root string) string {
	t.Helper()
	h := sha256.New()
	err := filepath.Walk(root, func(path string, info os.FileInfo, err error) error {
		if err != nil {
			return err
		}
		rel, err := filepath.Rel(root, path)
		if err != nil {
			return err
		}
		fmt.Fprintf(h, "%s\x00%s\x00", filepath.ToSlash(rel), info.Mode())
		if !info.IsDir() {
			body, err := os.ReadFile(path)
			if err != nil {
				return err
			}
			_, _ = h.Write(body)
		}
		return nil
	})
	if err != nil {
		t.Fatal(err)
	}
	return hex.EncodeToString(h.Sum(nil))
}
