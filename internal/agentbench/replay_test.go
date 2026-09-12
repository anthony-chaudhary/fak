package agentbench

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"io/fs"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"reflect"
	"sort"
	"strconv"
	"strings"
	"sync"
	"testing"
	"time"
)

func TestAgentBenchQuick16Requests(t *testing.T) {
	repo := writeAgentBenchRepo(t)
	repoBefore := agentBenchTreeDigest(t, repo)
	outDir := filepath.Join(t.TempDir(), "run")
	upstream := newAgentBenchSSE(t, agentBenchSSEOptions{outDir: outDir})
	defer upstream.Close()

	var stdout, stderr bytes.Buffer
	code := RunReplayCLI(context.Background(), &stdout, &stderr, agentBenchArgs(repo, upstream.URL(), outDir))
	if code != 0 {
		t.Fatalf("RunCLI code = %d, want 0; stderr: %s", code, stderr.String())
	}
	if got := upstream.RequestCount(); got != 16 {
		t.Fatalf("upstream requests = %d, want exactly 16", got)
	}
	if problems := upstream.Problems(); len(problems) != 0 {
		t.Fatalf("streaming request contract violations: %s", strings.Join(problems, "; "))
	}
	if upstream.ManifestMissingAtDispatch() {
		t.Fatal("first upstream dispatch happened before a non-empty manifest.json existed")
	}
	if got := agentBenchTreeDigest(t, repo); got != repoBefore {
		t.Fatalf("source repository changed: before %s, after %s", repoBefore, got)
	}

	events := readAgentBenchEvents(t, outDir)
	terminals := agentBenchScoredTerminalEvents(events)
	seen := assertAgentBenchTerminalMatrix(t, terminals)
	assertAgentBenchLifecycleMatrix(t, events, "release", "enqueue", "dispatch", "first_output", "tool", "end")
	conditionSessions := map[string]map[string]bool{"C1": {}, "C2": {}}
	for identity := range seen {
		conditionSessions[identity.Condition][identity.Session] = true
	}
	for _, condition := range []string{"C1", "C2"} {
		if got := len(conditionSessions[condition]); got != 2 {
			t.Errorf("%s sessions = %d, want 2", condition, got)
		}
	}

	assertAgentBenchFinalArtifacts(t, outDir)
	assertAgentBenchServiceOnly(t, outDir)
	assertAgentBenchNoPromotionClaims(t, outDir)
	if !json.Valid(bytes.TrimSpace(stdout.Bytes())) {
		t.Fatalf("--json stdout is not one JSON value: %q", stdout.String())
	}
}

// RunReplayCLI is a test-only adapter around the package-private replay phase. It deliberately
// cannot be selected through the product CLI; RunCLI always composes the real task phase.
func RunReplayCLI(ctx context.Context, stdout, stderr io.Writer, args []string) int {
	if ctx == nil {
		ctx = context.Background()
	}
	opts, err := parseOptions(stderr, args)
	if err != nil {
		return 2
	}
	if opts.profile != "quick" || opts.plan || opts.endpoint == "" || opts.out == "" || opts.model == "" {
		return RunCLI(ctx, stdout, stderr, args)
	}
	if err := ctx.Err(); err != nil {
		return 130
	}
	repo, err := resolveRepository(opts.repo)
	if err != nil {
		return 2
	}
	repoInput, err := freezeRepositoryInput(repo)
	if err != nil {
		return 1
	}
	inputs := newFrozenInputs(repoInput)
	plan := newQuickPlan(inputs, opts.model, true)
	endpoint, err := normalizeEndpoint(opts.endpoint)
	if err != nil {
		return 2
	}
	out, err := resolveOutput(repo, opts.out)
	if err != nil {
		return 2
	}
	state, runErr := runQuickReplay(ctx, stderr, repo, endpoint, opts.model, out, inputs, plan)
	var summary *runSummary
	if state != nil {
		var closeErr error
		summary, closeErr = state.Close(state.summary)
		if runErr == nil {
			runErr = closeErr
		}
	}
	if summary != nil {
		if err := printValue(stdout, summary, opts.json); err != nil && runErr == nil {
			runErr = err
		}
	}
	if runErr == nil {
		return 0
	}
	_, _ = fmt.Fprintf(stderr, "replay phase: %v\n", runErr)
	if errors.Is(runErr, errCanceled) || errors.Is(runErr, context.Canceled) || errors.Is(runErr, context.DeadlineExceeded) {
		return 130
	}
	return 1
}

func TestAgentBenchPlanNoDispatch(t *testing.T) {
	repo := writeAgentBenchRepo(t)
	outDir := filepath.Join(t.TempDir(), "plan")
	upstream := newAgentBenchSSE(t, agentBenchSSEOptions{})
	defer upstream.Close()

	var stdout, stderr bytes.Buffer
	args := append(agentBenchArgs(repo, upstream.URL(), outDir), "--plan")
	if code := RunCLI(context.Background(), &stdout, &stderr, args); code != 0 {
		t.Fatalf("RunCLI --plan code = %d, want 0; stderr: %s", code, stderr.String())
	}
	if got := upstream.RequestCount(); got != 0 {
		t.Fatalf("--plan dispatched %d upstream requests, want 0", got)
	}
	if !json.Valid(bytes.TrimSpace(stdout.Bytes())) {
		t.Fatalf("--plan --json stdout is not one JSON value: %q", stdout.String())
	}
	plan := strings.ToLower(stdout.String())
	for _, required := range []string{"coverage", "context", "output", "cost", "deadline", "retry", "16"} {
		if !strings.Contains(plan, required) {
			t.Errorf("--plan JSON does not expose required %q bound: %s", required, stdout.String())
		}
	}
	var planValue any
	if err := json.Unmarshal(stdout.Bytes(), &planValue); err != nil {
		t.Fatal(err)
	}
	if digests := agentBenchSHA256Values(planValue); len(digests) < 4 {
		t.Errorf("--plan binds %d SHA-256 artifact hashes, want four frozen inputs: %s", len(digests), stdout.String())
	}

}

func TestAgentBenchPrefixStability(t *testing.T) {
	repo := writeAgentBenchRepo(t)
	outDir := filepath.Join(t.TempDir(), "prefix")
	upstream := newAgentBenchSSE(t, agentBenchSSEOptions{})
	defer upstream.Close()

	var stdout, stderr bytes.Buffer
	if code := RunReplayCLI(context.Background(), &stdout, &stderr, agentBenchArgs(repo, upstream.URL(), outDir)); code != 0 {
		t.Fatalf("RunCLI code = %d; stderr: %s", code, stderr.String())
	}
	requests := upstream.Requests()
	if len(requests) != 16 {
		t.Fatalf("requests = %d, want 16", len(requests))
	}

	for _, request := range requests {
		body, err := json.Marshal(request.Body)
		if err != nil {
			t.Fatal(err)
		}
		if strings.Contains(string(body), "LIVE-OUTPUT-") {
			t.Fatalf("request %d incorporated varied live output instead of the frozen recorded continuation: %s", request.ID, body)
		}
	}
	assertAgentBenchFrozenRung(t, requests[:8], "C1")
	assertAgentBenchFrozenRung(t, requests[8:], "C2")
	assertAgentBenchSharedSourcePrefix(t, requests)
}

func TestAgentBenchBoundedScheduling(t *testing.T) {
	repo := writeAgentBenchRepo(t)
	outDir := filepath.Join(t.TempDir(), "schedule")
	upstream := newAgentBenchSSE(t, agentBenchSSEOptions{
		delay: func(id int) time.Duration {
			return time.Duration(5-id%4) * 15 * time.Millisecond
		},
	})
	defer upstream.Close()

	var stdout, stderr bytes.Buffer
	if code := RunReplayCLI(context.Background(), &stdout, &stderr, agentBenchArgs(repo, upstream.URL(), outDir)); code != 0 {
		t.Fatalf("RunCLI code = %d; stderr: %s", code, stderr.String())
	}
	if upstream.RungsOverlapped() {
		t.Fatal("C2 dispatched before all eight C1 requests reached terminal response state")
	}
	if got := upstream.RungMaxActive(1); got != 1 {
		t.Fatalf("C1 maximum inflight = %d, want exactly 1", got)
	}
	if got := upstream.RungMaxActive(2); got != 2 {
		t.Fatalf("C2 maximum inflight = %d, want exactly 2", got)
	}
	assertAgentBenchNoSessionOverlap(t, upstream.Requests())
	events := readAgentBenchEvents(t, outDir)
	assertAgentBenchRungEventOrder(t, events)
	assertAgentBenchTerminalMatrix(t, agentBenchScoredTerminalEvents(events))
}

func TestAgentBenchCancellation(t *testing.T) {
	t.Run("during replay", func(t *testing.T) {
		repo := writeAgentBenchRepo(t)
		outDir := filepath.Join(t.TempDir(), "cancel")
		upstream := newAgentBenchSSE(t, agentBenchSSEOptions{block: true})
		defer upstream.Close()

		ctx, cancel := context.WithCancel(context.Background())
		defer cancel()
		var stdout, stderr bytes.Buffer
		done := make(chan int, 1)
		go func() {
			done <- RunReplayCLI(ctx, &stdout, &stderr, agentBenchArgs(repo, upstream.URL(), outDir))
		}()
		select {
		case <-upstream.Started():
		case <-time.After(2 * time.Second):
			t.Fatal("benchmark did not dispatch before cancellation deadline")
		}
		waitAgentBenchLifecycle(t, outDir, time.Second, "release", "enqueue", "dispatch")
		cancel()

		select {
		case code := <-done:
			if code == 0 {
				t.Fatalf("canceled RunCLI returned success; stdout: %s", stdout.String())
			}
		case <-time.After(2 * time.Second):
			t.Fatal("RunCLI did not return promptly after context cancellation")
		}
		if got := upstream.RequestCount(); got >= 16 {
			t.Fatalf("cancellation allowed all %d requests to dispatch", got)
		}
		corpus := strings.ToLower(stderr.String() + stdout.String() + readAgentBenchDir(t, outDir))
		if !strings.Contains(corpus, "cancel") {
			t.Fatalf("cancellation is not explicit in diagnostics or artifacts: %q", corpus)
		}
		if raw, err := os.ReadFile(filepath.Join(outDir, "summary.json")); err != nil || !json.Valid(raw) {
			t.Fatalf("cancellation did not leave an atomic valid summary.json: err=%v body=%q", err, raw)
		}
		terminals := agentBenchScoredTerminalEvents(readAgentBenchEvents(t, outDir))
		assertAgentBenchTerminalMatrix(t, terminals)
		assertAgentBenchLifecycleMatrix(t, readAgentBenchEvents(t, outDir), "cancel")
		for i, event := range terminals {
			outcome, ok := agentBenchTerminalOutcome(event)
			if !ok || !strings.Contains(outcome, "cancel") {
				t.Fatalf("planned request terminal %d outcome = %q (present=%v), want canceled", i, outcome, ok)
			}
		}
	})

	t.Run("before terminal close", func(t *testing.T) {
		repo := writeAgentBenchRepo(t)
		outDir := filepath.Join(t.TempDir(), "terminal-cancel")
		upstream := newAgentBenchSSE(t, agentBenchSSEOptions{})
		defer upstream.Close()
		repoInput, err := freezeRepositoryInput(repo)
		if err != nil {
			t.Fatal(err)
		}
		inputs := newFrozenInputs(repoInput)
		plan := newQuickPlan(inputs, "fixture-model", true)
		ctx, cancel := context.WithCancel(context.Background())
		state, err := runQuickReplay(ctx, io.Discard, repo, upstream.URL(), "fixture-model", outDir, inputs, plan)
		if err != nil {
			t.Fatalf("completed replay: %v", err)
		}
		supplied := state.summary
		supplied.Status = "passed"
		supplied.OverallVerdict = "QUICK_SMOKE_PASSED"
		cancel()
		closed, err := state.Close(supplied)
		if err == nil || closed == nil {
			t.Fatalf("Close after parent cancellation = summary %+v, err %v; want terminal failure", closed, err)
		}
		if closed.Status != "incomplete" || closed.OverallVerdict != "INCOMPLETE" {
			t.Fatalf("canceled close retained success: status=%q overall=%q", closed.Status, closed.OverallVerdict)
		}
		var persisted runSummary
		body, readErr := os.ReadFile(filepath.Join(outDir, "summary.json"))
		if readErr != nil || json.Unmarshal(body, &persisted) != nil {
			t.Fatalf("read terminal summary: %v body=%q", readErr, body)
		}
		if persisted.Status != "incomplete" || persisted.OverallVerdict != "INCOMPLETE" {
			t.Fatalf("persisted canceled close retained success: status=%q overall=%q", persisted.Status, persisted.OverallVerdict)
		}
	})
}

func TestAgentBenchHonestUnknowns(t *testing.T) {
	repo := writeAgentBenchRepo(t)
	outDir := filepath.Join(t.TempDir(), "unknowns")
	upstream := newAgentBenchSSE(t, agentBenchSSEOptions{usageKnown: func(id int) bool { return id%2 == 1 }})
	defer upstream.Close()

	var stdout, stderr bytes.Buffer
	if code := RunReplayCLI(context.Background(), &stdout, &stderr, agentBenchArgs(repo, upstream.URL(), outDir)); code != 0 {
		t.Fatalf("RunCLI code = %d; stderr: %s", code, stderr.String())
	}
	if problems := upstream.Problems(); len(problems) != 0 {
		t.Fatalf("streaming request contract violations: %s", strings.Join(problems, "; "))
	}

	usageByIdentity := make(map[agentBenchIdentity]string, 16)
	for i, event := range readAgentBenchEvents(t, outDir) {
		identity, identityOK := agentBenchEventIdentity(event)
		if !identityOK {
			continue
		}
		state, ok := agentBenchUsageState(event)
		if !ok {
			continue
		}
		if state != "known" && state != "unknown" {
			t.Fatalf("event %d usage state = %q", i, state)
		}
		if prior := usageByIdentity[identity]; prior != "" && prior != state {
			t.Fatalf("request %+v changes usage authority from %s to %s", identity, prior, state)
		}
		usageByIdentity[identity] = state
		if state == "unknown" && agentBenchHasUnknownZero(event) {
			t.Fatalf("event %d fabricates zero token usage for an unknown observation: %v", i, event)
		}
	}
	if len(usageByIdentity) != 16 {
		t.Fatalf("requests with explicit usage authority = %d, want 16", len(usageByIdentity))
	}
	known, unknown := 0, 0
	for _, state := range usageByIdentity {
		if state == "known" {
			known++
		} else {
			unknown++
		}
	}
	if known != 8 || unknown != 8 {
		t.Fatalf("usage states known/unknown = %d/%d, want 8/8", known, unknown)
	}
	assertAgentBenchNoPromotionClaims(t, outDir)

	for _, tc := range []struct {
		name       string
		streamMode string
	}{
		{name: "malformed_event", streamMode: "malformed"},
		{name: "missing_done", streamMode: "missing_done"},
		{name: "missing_finish", streamMode: "missing_finish"},
		{name: "inconsistent_usage", streamMode: "inconsistent_usage"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			badOut := filepath.Join(t.TempDir(), tc.name)
			badUpstream := newAgentBenchSSE(t, agentBenchSSEOptions{streamMode: tc.streamMode})
			defer badUpstream.Close()
			var badStdout, badStderr bytes.Buffer
			ctx, cancel := context.WithTimeout(context.Background(), 3*time.Second)
			defer cancel()
			if code := RunReplayCLI(ctx, &badStdout, &badStderr, agentBenchArgs(repo, badUpstream.URL(), badOut)); code == 0 {
				t.Fatalf("strict stream case %s returned success; stdout: %s", tc.streamMode, badStdout.String())
			}
			if got := badUpstream.RequestCount(); got > 24 {
				t.Fatalf("strict stream case %s dispatched %d attempts, exceeds 16 requests + eight replay retries", tc.streamMode, got)
			}
		})
	}
}

type agentBenchSSEOptions struct {
	outDir     string
	delay      func(int) time.Duration
	usageKnown func(int) bool
	block      bool
	streamMode string
}

type agentBenchCapturedRequest struct {
	ID        int
	Identity  agentBenchIdentity
	Body      map[string]any
	Messages  []any
	StartedAt time.Time
	EndedAt   time.Time
}

type agentBenchSSE struct {
	server  *httptest.Server
	opts    agentBenchSSEOptions
	started chan struct{}
	once    sync.Once

	mu              sync.Mutex
	requests        []agentBenchCapturedRequest
	activeByRung    [3]int
	maxActiveByRung [3]int
	completedByRung [3]int
	rungsOverlapped bool
	problems        []string
	manifestMissing bool
}

func newAgentBenchSSE(t *testing.T, opts agentBenchSSEOptions) *agentBenchSSE {
	t.Helper()
	h := &agentBenchSSE{opts: opts, started: make(chan struct{})}
	h.server = httptest.NewServer(http.HandlerFunc(h.serveHTTP))
	return h
}

func (h *agentBenchSSE) serveHTTP(w http.ResponseWriter, r *http.Request) {
	var body map[string]any
	if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
		h.addProblem("decode request: " + err.Error())
		http.Error(w, "invalid json", http.StatusBadRequest)
		return
	}
	messages, _ := body["messages"].([]any)
	if len(messages) == 0 {
		messages, _ = body["input"].([]any)
	}
	stream, _ := body["stream"].(bool)
	streamOptions, _ := body["stream_options"].(map[string]any)
	includeUsage, _ := streamOptions["include_usage"].(bool)
	if !stream {
		h.addProblem("stream must be true")
	}
	if !includeUsage {
		h.addProblem("stream_options.include_usage must be true")
	}
	if model, _ := body["model"].(string); model != "fixture-model" {
		h.addProblem(fmt.Sprintf("model = %q, want fixture-model", model))
	}

	h.mu.Lock()
	id := len(h.requests) + 1
	rung := 1
	if id > 8 {
		rung = 2
		if h.completedByRung[1] != 8 {
			h.rungsOverlapped = true
		}
	}
	identity, _ := agentBenchEventIdentity(body)
	h.requests = append(h.requests, agentBenchCapturedRequest{ID: id, Identity: identity, Body: body, Messages: messages, StartedAt: time.Now()})
	h.activeByRung[rung]++
	if h.activeByRung[rung] > h.maxActiveByRung[rung] {
		h.maxActiveByRung[rung] = h.activeByRung[rung]
	}
	if h.opts.outDir != "" {
		info, err := os.Stat(filepath.Join(h.opts.outDir, "manifest.json"))
		if err != nil || info.Size() == 0 {
			h.manifestMissing = true
		}
	}
	h.mu.Unlock()
	h.once.Do(func() { close(h.started) })
	defer func() {
		h.mu.Lock()
		h.activeByRung[rung]--
		if h.requests[id-1].EndedAt.IsZero() {
			h.completedByRung[rung]++
			h.requests[id-1].EndedAt = time.Now()
		}
		h.mu.Unlock()
	}()

	w.Header().Set("Content-Type", "text/event-stream")
	w.WriteHeader(http.StatusOK)
	if flusher, ok := w.(http.Flusher); ok {
		flusher.Flush()
	}
	if h.opts.block {
		<-r.Context().Done()
		return
	}
	if h.opts.delay != nil {
		timer := time.NewTimer(h.opts.delay(id))
		defer timer.Stop()
		select {
		case <-timer.C:
		case <-r.Context().Done():
			return
		}
	}
	h.writeStream(w, id)
}

func (h *agentBenchSSE) writeStream(w io.Writer, id int) {
	live := fmt.Sprintf("LIVE-OUTPUT-%02d-%s", id, strings.Repeat("x", id%7+1))
	content := fmt.Sprintf("data: {\"id\":\"chunk-%d\",\"model\":\"fixture-model\",\"choices\":[{\"delta\":{\"content\":%s}}]}\n\n", id, strconv.Quote(live))
	tool := fmt.Sprintf("data: {\"model\":\"fixture-model\",\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":0,\"id\":\"call-%d\",\"type\":\"function\",\"function\":{\"name\":\"Read\",\"arguments\":\"{\\\"file_path\\\":\\\"README.md\\\"}\"}}]}}]}\n\n", id)
	usage := fmt.Sprintf("data: {\"model\":\"fixture-model\",\"choices\":[{\"delta\":{},\"finish_reason\":\"tool_calls\"}],\"usage\":{\"prompt_tokens\":%d,\"completion_tokens\":1,\"total_tokens\":%d}}\n\n", 100+id, 101+id)
	switch h.opts.streamMode {
	case "malformed":
		io.WriteString(w, "data: {\"choices\":[\n\n")
	case "missing_done":
		io.WriteString(w, content+tool+usage)
	case "missing_finish":
		io.WriteString(w, content+tool)
		fmt.Fprintf(w, "data: {\"model\":\"fixture-model\",\"choices\":[],\"usage\":{\"prompt_tokens\":%d,\"completion_tokens\":1,\"total_tokens\":%d}}\n\n", 100+id, 101+id)
		h.writeDone(w, id)
	case "inconsistent_usage":
		io.WriteString(w, content+tool)
		io.WriteString(w, "data: {\"model\":\"fixture-model\",\"choices\":[{\"delta\":{},\"finish_reason\":\"tool_calls\"}],\"usage\":{\"prompt_tokens\":10,\"completion_tokens\":2,\"total_tokens\":999}}\n\n")
		h.writeDone(w, id)
	default:
		io.WriteString(w, content+tool)
		known := true
		if h.opts.usageKnown != nil {
			known = h.opts.usageKnown(id)
		}
		if known {
			io.WriteString(w, usage)
		} else {
			io.WriteString(w, "data: {\"model\":\"fixture-model\",\"choices\":[{\"delta\":{},\"finish_reason\":\"tool_calls\"}]}\n\n")
		}
		h.writeDone(w, id)
	}
}

func (h *agentBenchSSE) writeDone(w io.Writer, id int) {
	rung := 1
	if id > 8 {
		rung = 2
	}
	h.mu.Lock()
	if h.requests[id-1].EndedAt.IsZero() {
		h.requests[id-1].EndedAt = time.Now()
		h.completedByRung[rung]++
	}
	h.mu.Unlock()
	io.WriteString(w, "data: [DONE]\n\n")
}

func (h *agentBenchSSE) Close()                   { h.server.Close() }
func (h *agentBenchSSE) URL() string              { return h.server.URL }
func (h *agentBenchSSE) Started() <-chan struct{} { return h.started }

func (h *agentBenchSSE) RequestCount() int {
	h.mu.Lock()
	defer h.mu.Unlock()
	return len(h.requests)
}

func (h *agentBenchSSE) Requests() []agentBenchCapturedRequest {
	h.mu.Lock()
	defer h.mu.Unlock()
	return append([]agentBenchCapturedRequest(nil), h.requests...)
}

func (h *agentBenchSSE) RungMaxActive(rung int) int {
	h.mu.Lock()
	defer h.mu.Unlock()
	return h.maxActiveByRung[rung]
}

func (h *agentBenchSSE) RungsOverlapped() bool {
	h.mu.Lock()
	defer h.mu.Unlock()
	return h.rungsOverlapped
}

func (h *agentBenchSSE) Problems() []string {
	h.mu.Lock()
	defer h.mu.Unlock()
	return append([]string(nil), h.problems...)
}

func (h *agentBenchSSE) ManifestMissingAtDispatch() bool {
	h.mu.Lock()
	defer h.mu.Unlock()
	return h.manifestMissing
}

func (h *agentBenchSSE) addProblem(problem string) {
	h.mu.Lock()
	h.problems = append(h.problems, problem)
	h.mu.Unlock()
}

func agentBenchArgs(repo, endpoint, outDir string) []string {
	return []string{"--profile", "quick", "--repo", repo, "--endpoint", endpoint, "--model", "fixture-model", "--out", outDir, "--json"}
}

func writeAgentBenchRepo(t *testing.T) string {
	t.Helper()
	root := t.TempDir()
	files := map[string]string{
		"go.mod":      "module example.com/agentbenchfixture\n\ngo 1.26\n",
		"README.md":   "# Frozen agent benchmark fixture\n",
		"cmd/app.go":  "package main\n\nfunc main() {}\n",
		"pkg/math.go": "package math\n\nfunc Add(a, b int) int { return a + b }\n",
	}
	for name, body := range files {
		path := filepath.Join(root, filepath.FromSlash(name))
		if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(path, []byte(body), 0o444); err != nil {
			t.Fatal(err)
		}
	}
	return root
}

func agentBenchTreeDigest(t *testing.T, root string) string {
	t.Helper()
	h := sha256.New()
	err := filepath.WalkDir(root, func(path string, entry fs.DirEntry, err error) error {
		if err != nil {
			return err
		}
		rel, err := filepath.Rel(root, path)
		if err != nil {
			return err
		}
		info, err := entry.Info()
		if err != nil {
			return err
		}
		fmt.Fprintf(h, "%s\x00%s\x00", filepath.ToSlash(rel), info.Mode())
		if !entry.IsDir() {
			body, err := os.ReadFile(path)
			if err != nil {
				return err
			}
			h.Write(body)
		}
		return nil
	})
	if err != nil {
		t.Fatal(err)
	}
	return hex.EncodeToString(h.Sum(nil))
}

func readAgentBenchEvents(t *testing.T, outDir string) []map[string]any {
	t.Helper()
	events, err := agentBenchTryReadEvents(outDir)
	if err != nil {
		t.Fatal(err)
	}
	return events
}

func agentBenchTryReadEvents(outDir string) ([]map[string]any, error) {
	body, err := os.ReadFile(filepath.Join(outDir, "events.jsonl"))
	if err != nil {
		return nil, fmt.Errorf("read events.jsonl: %w", err)
	}
	lines := bytes.Split(bytes.TrimSpace(body), []byte("\n"))
	events := make([]map[string]any, 0, len(lines))
	for i, line := range lines {
		if len(bytes.TrimSpace(line)) == 0 {
			continue
		}
		var event map[string]any
		if err := json.Unmarshal(line, &event); err != nil {
			return nil, fmt.Errorf("events.jsonl line %d: %w", i+1, err)
		}
		events = append(events, event)
	}
	return events, nil
}

type agentBenchIdentity struct {
	Condition string
	Session   string
	Turn      int
}

func agentBenchScoredTerminalEvents(events []map[string]any) []map[string]any {
	terminals := make([]map[string]any, 0, 16)
	for _, event := range events {
		if _, ok := agentBenchEventIdentity(event); ok && agentBenchIsTerminalEvent(event) {
			terminals = append(terminals, event)
		}
	}
	return terminals
}

func assertAgentBenchTerminalMatrix(t *testing.T, terminals []map[string]any) map[agentBenchIdentity]int {
	t.Helper()
	if len(terminals) != 16 {
		t.Fatalf("scored request terminal events = %d, want 16 (lifecycle log may contain additional non-terminal events)", len(terminals))
	}
	seen := make(map[agentBenchIdentity]int, 16)
	for i, event := range terminals {
		identity, ok := agentBenchEventIdentity(event)
		if !ok {
			t.Fatalf("terminal event %d has incomplete scored-request identity: %v", i, event)
		}
		seen[identity]++
	}
	if len(seen) != 16 {
		t.Fatalf("unique terminal condition/session/turn identities = %d, want 16: %v", len(seen), seen)
	}
	for identity, count := range seen {
		if count != 1 {
			t.Fatalf("terminal identity %+v occurs %d times, want once", identity, count)
		}
	}
	conditionSessions := map[string]map[string]map[int]bool{"C1": {}, "C2": {}}
	for identity := range seen {
		if conditionSessions[identity.Condition][identity.Session] == nil {
			conditionSessions[identity.Condition][identity.Session] = make(map[int]bool)
		}
		conditionSessions[identity.Condition][identity.Session][identity.Turn] = true
	}
	for _, condition := range []string{"C1", "C2"} {
		if len(conditionSessions[condition]) != 2 {
			t.Fatalf("%s terminal sessions = %d, want 2", condition, len(conditionSessions[condition]))
		}
		for session, turns := range conditionSessions[condition] {
			for turn := 1; turn <= 4; turn++ {
				if !turns[turn] {
					t.Fatalf("%s/%s lacks terminal outcome for turn %d", condition, session, turn)
				}
			}
		}
	}
	return seen
}

func agentBenchEventIdentity(event map[string]any) (agentBenchIdentity, bool) {
	condition, conditionOK := agentBenchStringField(event, "condition", "condition_id", "rung", "concurrency")
	condition = strings.ToUpper(strings.ReplaceAll(strings.TrimSpace(condition), "=", ""))
	if condition == "1" || condition == "2" {
		condition = "C" + condition
	}
	if !conditionOK {
		if concurrency, ok := agentBenchIntField(event, "concurrency", "rung", "c"); ok {
			condition, conditionOK = "C"+strconv.Itoa(concurrency), true
		}
	}
	session, sessionOK := agentBenchStringField(event, "session", "session_id")
	if !sessionOK {
		if sessionIndex, ok := agentBenchIntField(event, "session", "session_index"); ok {
			session, sessionOK = strconv.Itoa(sessionIndex), true
		}
	}
	turn, turnOK := agentBenchIntField(event, "turn", "turn_index")
	identity := agentBenchIdentity{Condition: condition, Session: session, Turn: turn}
	valid := conditionOK && (identity.Condition == "C1" || identity.Condition == "C2") && sessionOK && session != "" && turnOK && turn >= 1 && turn <= 4
	return identity, valid
}

func agentBenchIsTerminalEvent(event map[string]any) bool {
	if terminal, ok := agentBenchBoolField(event, "terminal", "is_terminal"); ok && terminal {
		return true
	}
	labels := agentBenchNamedStrings(event, "event", "event_type", "type", "kind", "phase", "lifecycle", "state", "status", "terminal_state", "outcome")
	explicit := map[string]bool{
		"requestend": true, "requestended": true, "requestcomplete": true, "requestcompleted": true,
		"requestcancel": true, "requestcanceled": true, "requestcancelled": true, "requestfailed": true,
		"requestterminal": true, "attemptterminal": true, "terminal": true, "succeeded": true,
	}
	for _, label := range labels {
		if explicit[agentBenchNormField(label)] {
			return true
		}
	}
	joined := agentBenchNormField(strings.Join(labels, " "))
	if strings.Contains(joined, "tool") || strings.Contains(joined, "stream") || strings.Contains(joined, "firstoutput") {
		return false
	}
	for _, label := range labels {
		switch agentBenchNormField(label) {
		case "end", "ended", "complete", "completed", "cancel", "canceled", "cancelled", "failed", "error":
			return true
		}
	}
	return false
}

func agentBenchTerminalOutcome(event map[string]any) (string, bool) {
	labels := agentBenchNamedStrings(event, "event", "event_type", "type", "kind", "phase", "lifecycle", "state", "status", "terminal_state", "outcome", "reason")
	if len(labels) == 0 {
		return "", false
	}
	return strings.ToLower(strings.Join(labels, " ")), true
}

func assertAgentBenchLifecycleMatrix(t *testing.T, events []map[string]any, phases ...string) {
	t.Helper()
	want := make(map[agentBenchIdentity]bool, 16)
	for _, terminal := range agentBenchScoredTerminalEvents(events) {
		if identity, ok := agentBenchEventIdentity(terminal); ok {
			want[identity] = true
		}
	}
	if len(want) != 16 {
		t.Fatalf("terminal identity set = %d, want 16 before lifecycle audit", len(want))
	}
	seen := make(map[string]map[agentBenchIdentity]bool, len(phases))
	for _, phase := range phases {
		seen[phase] = make(map[agentBenchIdentity]bool, 16)
	}
	for _, event := range events {
		identity, ok := agentBenchEventIdentity(event)
		if !ok || !want[identity] {
			continue
		}
		for _, phase := range agentBenchLifecyclePhases(event) {
			if seen[phase] != nil {
				seen[phase][identity] = true
			}
		}
	}
	for _, phase := range phases {
		if got := len(seen[phase]); got != 16 {
			t.Fatalf("lifecycle phase %s covers %d scored requests, want 16", phase, got)
		}
	}
}

func assertAgentBenchRungEventOrder(t *testing.T, events []map[string]any) {
	t.Helper()
	c1Ended := make(map[agentBenchIdentity]bool, 8)
	c2Seen := false
	for _, event := range events {
		identity, ok := agentBenchEventIdentity(event)
		if !ok {
			continue
		}
		if identity.Condition == "C1" && agentBenchIsTerminalEvent(event) {
			if c2Seen {
				t.Fatalf("C1 terminal %+v was appended after C2 dispatch began", identity)
			}
			c1Ended[identity] = true
		}
		for _, phase := range agentBenchLifecyclePhases(event) {
			if phase == "dispatch" && identity.Condition == "C2" {
				c2Seen = true
				if len(c1Ended) != 8 {
					t.Fatalf("C2 dispatch %+v occurred after only %d/8 C1 terminal outcomes", identity, len(c1Ended))
				}
			}
		}
	}
	if !c2Seen {
		t.Fatal("no C2 dispatch lifecycle event observed")
	}
}

func waitAgentBenchLifecycle(t *testing.T, outDir string, timeout time.Duration, phases ...string) {
	t.Helper()
	deadline := time.Now().Add(timeout)
	for {
		events, err := agentBenchTryReadEvents(outDir)
		if err == nil {
			seen := make(map[string]bool, len(phases))
			for _, event := range events {
				for _, phase := range agentBenchLifecyclePhases(event) {
					seen[phase] = true
				}
			}
			complete := true
			for _, phase := range phases {
				complete = complete && seen[phase]
			}
			if complete {
				return
			}
		}
		if time.Now().After(deadline) {
			t.Fatalf("append-only lifecycle did not expose %v while the first request remained in flight", phases)
		}
		time.Sleep(10 * time.Millisecond)
	}
}

func agentBenchLifecyclePhases(event map[string]any) []string {
	labels := agentBenchNamedStrings(event, "event", "event_type", "type", "kind", "phase", "lifecycle", "state", "status", "terminal_state", "outcome")
	joined := agentBenchNormField(strings.Join(labels, " "))
	var phases []string
	if strings.Contains(joined, "release") {
		phases = append(phases, "release")
	}
	if strings.Contains(joined, "enqueue") || strings.Contains(joined, "queued") {
		phases = append(phases, "enqueue")
	}
	if strings.Contains(joined, "dispatch") {
		phases = append(phases, "dispatch")
	}
	if strings.Contains(joined, "firstoutput") || strings.Contains(joined, "firstbyte") || strings.Contains(joined, "firsttoken") {
		phases = append(phases, "first_output")
	}
	if strings.Contains(joined, "toolcall") || strings.Contains(joined, "toolinvoke") {
		phases = append(phases, "tool")
	}
	if agentBenchIsTerminalEvent(event) {
		phases = append(phases, "end")
	}
	if strings.Contains(joined, "cancel") {
		phases = append(phases, "cancel")
	}
	return phases
}

func agentBenchNamedStrings(value any, names ...string) []string {
	wanted := agentBenchFieldSet(names)
	var found []string
	var visit func(any)
	visit = func(value any) {
		switch typed := value.(type) {
		case map[string]any:
			keys := make([]string, 0, len(typed))
			for key := range typed {
				keys = append(keys, key)
			}
			sort.Strings(keys)
			for _, key := range keys {
				item := typed[key]
				if wanted[agentBenchNormField(key)] {
					if text, ok := item.(string); ok {
						found = append(found, text)
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
	return found
}

func assertAgentBenchFinalArtifacts(t *testing.T, outDir string) {
	t.Helper()
	artifacts := make(map[string][]byte, 4)
	for _, name := range []string{"manifest.json", "summary.json", "receipt.json"} {
		body, err := os.ReadFile(filepath.Join(outDir, name))
		if err != nil {
			t.Fatalf("read %s: %v", name, err)
		}
		if !json.Valid(body) {
			t.Fatalf("%s is not valid JSON: %q", name, body)
		}
		artifacts[name] = body
	}
	temps, err := filepath.Glob(filepath.Join(outDir, "summary.json.tmp*"))
	if err != nil {
		t.Fatal(err)
	}
	if len(temps) != 0 {
		t.Fatalf("atomic summary left temporary files: %v", temps)
	}
	events, err := os.ReadFile(filepath.Join(outDir, "events.jsonl"))
	if err != nil {
		t.Fatal(err)
	}
	var value any
	if err := json.Unmarshal(artifacts["receipt.json"], &value); err != nil {
		t.Fatal(err)
	}
	eventSum := sha256.Sum256(events)
	eventHex := hex.EncodeToString(eventSum[:])
	eventDigest := "sha256:" + eventHex
	if !agentBenchContainsString(value, eventDigest) && !agentBenchContainsString(value, eventHex) {
		t.Fatalf("receipt.json does not bind the exact events.jsonl digest %s: %s", eventDigest, artifacts["receipt.json"])
	}
	if digests := agentBenchSHA256Values(value); len(digests) < 3 {
		t.Fatalf("receipt.json binds %d distinct SHA-256 values, want event log plus frozen inputs: %s", len(digests), artifacts["receipt.json"])
	}
}

func agentBenchSHA256Values(value any) map[string]bool {
	values := make(map[string]bool)
	var visit func(any)
	visit = func(value any) {
		switch typed := value.(type) {
		case string:
			candidate := strings.TrimPrefix(typed, "sha256:")
			if len(candidate) == sha256.Size*2 {
				if _, err := hex.DecodeString(candidate); err == nil && candidate == strings.ToLower(candidate) {
					values[candidate] = true
				}
			}
		case []any:
			for _, item := range typed {
				visit(item)
			}
		case map[string]any:
			for _, item := range typed {
				visit(item)
			}
		}
	}
	visit(value)
	return values
}

func agentBenchContainsString(value any, wanted string) bool {
	switch typed := value.(type) {
	case string:
		return typed == wanted
	case []any:
		for _, item := range typed {
			if agentBenchContainsString(item, wanted) {
				return true
			}
		}
	case map[string]any:
		for _, item := range typed {
			if agentBenchContainsString(item, wanted) {
				return true
			}
		}
	}
	return false
}

func agentBenchStringField(value any, names ...string) (string, bool) {
	wanted := agentBenchFieldSet(names)
	return agentBenchFindString(value, wanted)
}

func agentBenchFindString(value any, wanted map[string]bool) (string, bool) {
	switch typed := value.(type) {
	case map[string]any:
		keys := make([]string, 0, len(typed))
		for key := range typed {
			keys = append(keys, key)
		}
		sort.Strings(keys)
		for _, key := range keys {
			if wanted[agentBenchNormField(key)] {
				if text, ok := typed[key].(string); ok {
					return text, true
				}
			}
		}
		for _, key := range keys {
			if text, ok := agentBenchFindString(typed[key], wanted); ok {
				return text, true
			}
		}
	case []any:
		for _, item := range typed {
			if text, ok := agentBenchFindString(item, wanted); ok {
				return text, true
			}
		}
	}
	return "", false
}

func agentBenchIntField(value any, names ...string) (int, bool) {
	wanted := agentBenchFieldSet(names)
	return agentBenchFindInt(value, wanted)
}

func agentBenchFindInt(value any, wanted map[string]bool) (int, bool) {
	switch typed := value.(type) {
	case map[string]any:
		keys := make([]string, 0, len(typed))
		for key := range typed {
			keys = append(keys, key)
		}
		sort.Strings(keys)
		for _, key := range keys {
			if wanted[agentBenchNormField(key)] {
				if number, ok := typed[key].(float64); ok && number == float64(int(number)) {
					return int(number), true
				}
			}
		}
		for _, key := range keys {
			if number, ok := agentBenchFindInt(typed[key], wanted); ok {
				return number, true
			}
		}
	case []any:
		for _, item := range typed {
			if number, ok := agentBenchFindInt(item, wanted); ok {
				return number, true
			}
		}
	}
	return 0, false
}

func agentBenchFieldSet(names []string) map[string]bool {
	wanted := make(map[string]bool, len(names))
	for _, name := range names {
		wanted[agentBenchNormField(name)] = true
	}
	return wanted
}

func agentBenchNormField(name string) string {
	return strings.NewReplacer("_", "", "-", "").Replace(strings.ToLower(name))
}

func assertAgentBenchFrozenRung(t *testing.T, requests []agentBenchCapturedRequest, condition string) {
	t.Helper()
	if len(requests) != 8 {
		t.Fatalf("%s captured requests = %d, want 8", condition, len(requests))
	}
	ordered := append([]agentBenchCapturedRequest(nil), requests...)
	sort.SliceStable(ordered, func(i, j int) bool { return len(ordered[i].Messages) < len(ordered[j].Messages) })
	depths := make(map[int]int, len(ordered))
	depthCount := make(map[int]int, 4)
	for i, request := range ordered {
		if len(request.Messages) == 0 {
			t.Fatalf("%s request %d has no OpenAI-compatible message history", condition, request.ID)
		}
		depth := 1
		for j := 0; j < i; j++ {
			parent := ordered[j]
			if agentBenchStrictMessagePrefix(parent.Messages, request.Messages) && depths[parent.ID]+1 > depth {
				depth = depths[parent.ID] + 1
			}
		}
		depths[request.ID] = depth
		depthCount[depth]++
	}
	for depth := 1; depth <= 4; depth++ {
		if got := depthCount[depth]; got != 2 {
			t.Fatalf("%s frozen history depth %d requests = %d, want two sessions", condition, depth, got)
		}
	}
	if len(depthCount) != 4 {
		t.Fatalf("%s has unexpected frozen history depths: %v", condition, depthCount)
	}
	hasRecordedToolResult := false
	for _, request := range requests {
		for _, message := range request.Messages {
			mapped, _ := message.(map[string]any)
			role, _ := mapped["role"].(string)
			if role == "tool" {
				hasRecordedToolResult = true
			}
		}
	}
	if !hasRecordedToolResult {
		t.Fatalf("%s histories never expose a frozen recorded tool result", condition)
	}
}

func assertAgentBenchSharedSourcePrefix(t *testing.T, requests []agentBenchCapturedRequest) {
	t.Helper()
	if len(requests) == 0 || len(requests[0].Messages) == 0 {
		t.Fatal("no messages available for shared source-prefix check")
	}
	common := len(requests[0].Messages)
	for _, request := range requests[1:] {
		limit := common
		if len(request.Messages) < limit {
			limit = len(request.Messages)
		}
		matched := 0
		for matched < limit && reflect.DeepEqual(requests[0].Messages[matched], request.Messages[matched]) {
			matched++
		}
		common = matched
	}
	if common == 0 {
		t.Fatal("C1/C2 requests do not retain any byte-stable shared system/source prefix")
	}
}

func assertAgentBenchNoSessionOverlap(t *testing.T, requests []agentBenchCapturedRequest) {
	t.Helper()
	for i, earlier := range requests {
		for j, later := range requests {
			if i == j || earlier.Identity.Condition == "" || earlier.Identity.Session == "" || later.Identity.Condition == "" || later.Identity.Session == "" {
				continue
			}
			if earlier.Identity.Condition == later.Identity.Condition && earlier.Identity.Session == later.Identity.Session && earlier.Identity.Turn < later.Identity.Turn && later.StartedAt.Before(earlier.EndedAt) {
				t.Fatalf("same-session requests overlapped: request %d ended %s after continuation %d started %s", earlier.ID, earlier.EndedAt, later.ID, later.StartedAt)
			}
		}
	}
}

func agentBenchStrictMessagePrefix(prefix, messages []any) bool {
	return len(prefix) < len(messages) && reflect.DeepEqual(prefix, messages[:len(prefix)])
}

func agentBenchUsageState(event map[string]any) (string, bool) {
	if known, ok := agentBenchBoolField(event, "usage_known"); ok {
		if known {
			return "known", true
		}
		return "unknown", true
	}
	if state, ok := agentBenchStringField(event, "usage_status", "usage_availability"); ok {
		state = strings.ToLower(state)
		if state == "known" || state == "available" {
			return "known", true
		}
		if state == "unknown" || state == "unavailable" {
			return "unknown", true
		}
	}
	usage, ok := agentBenchMapField(event, "usage")
	if !ok {
		return "", false
	}
	if known, ok := agentBenchBoolField(usage, "known"); ok {
		if known {
			return "known", true
		}
		return "unknown", true
	}
	if state, ok := agentBenchStringField(usage, "status", "availability"); ok {
		state = strings.ToLower(state)
		if state == "known" || state == "available" {
			return "known", true
		}
		if state == "unknown" || state == "unavailable" {
			return "unknown", true
		}
	}
	return "", false
}

func agentBenchBoolField(value any, names ...string) (bool, bool) {
	wanted := agentBenchFieldSet(names)
	var visit func(any) (bool, bool)
	visit = func(value any) (bool, bool) {
		switch typed := value.(type) {
		case map[string]any:
			for key, item := range typed {
				if wanted[agentBenchNormField(key)] {
					if boolean, ok := item.(bool); ok {
						return boolean, true
					}
				}
			}
			for _, item := range typed {
				if boolean, ok := visit(item); ok {
					return boolean, true
				}
			}
		case []any:
			for _, item := range typed {
				if boolean, ok := visit(item); ok {
					return boolean, true
				}
			}
		}
		return false, false
	}
	return visit(value)
}

func agentBenchMapField(value map[string]any, name string) (map[string]any, bool) {
	wanted := agentBenchNormField(name)
	for key, item := range value {
		if agentBenchNormField(key) == wanted {
			mapped, ok := item.(map[string]any)
			return mapped, ok
		}
	}
	for _, item := range value {
		if mapped, ok := item.(map[string]any); ok {
			if found, ok := agentBenchMapField(mapped, name); ok {
				return found, true
			}
		}
	}
	return nil, false
}

func agentBenchHasUnknownZero(value any) bool {
	switch typed := value.(type) {
	case map[string]any:
		for key, item := range typed {
			norm := agentBenchNormField(key)
			if strings.Contains(norm, "token") {
				if number, ok := item.(float64); ok && number == 0 {
					return true
				}
			}
			if agentBenchHasUnknownZero(item) {
				return true
			}
		}
	case []any:
		for _, item := range typed {
			if agentBenchHasUnknownZero(item) {
				return true
			}
		}
	}
	return false
}

func assertAgentBenchServiceOnly(t *testing.T, outDir string) {
	t.Helper()
	var corpus strings.Builder
	for _, name := range []string{"summary.json", "receipt.json"} {
		body, err := os.ReadFile(filepath.Join(outDir, name))
		if err != nil {
			t.Fatal(err)
		}
		corpus.Write(body)
	}
	if !strings.Contains(strings.ToUpper(corpus.String()), "SERVICE_ONLY_DIAGNOSTIC") {
		t.Fatalf("quick receipt is not labeled SERVICE_ONLY_DIAGNOSTIC: %s", corpus.String())
	}
}

func assertAgentBenchNoPromotionClaims(t *testing.T, outDir string) {
	t.Helper()
	for _, name := range []string{"summary.json", "receipt.json"} {
		body, err := os.ReadFile(filepath.Join(outDir, name))
		if err != nil {
			t.Fatal(err)
		}
		var value any
		if err := json.Unmarshal(body, &value); err != nil {
			t.Fatal(err)
		}
		if field, ok := agentBenchForbiddenClaim(value); ok {
			t.Fatalf("%s makes forbidden hardware/task-acceptance claim %q", name, field)
		}
	}
}

func agentBenchForbiddenClaim(value any) (string, bool) {
	switch typed := value.(type) {
	case map[string]any:
		for key, item := range typed {
			norm := agentBenchNormField(key)
			if strings.Contains(norm, "hardware") {
				return key, true
			}
			if strings.Contains(norm, "taskaccept") {
				if claimed, ok := item.(bool); ok && claimed {
					return key, true
				}
				if claim, ok := item.(string); ok && agentBenchPositiveClaim(claim) {
					return key, true
				}
			}
			if norm == "overall" || strings.Contains(norm, "usability") {
				if claim, ok := item.(string); ok && agentBenchPositiveClaim(claim) {
					return key, true
				}
			}
			if field, ok := agentBenchForbiddenClaim(item); ok {
				return field, true
			}
		}
	case []any:
		for _, item := range typed {
			if field, ok := agentBenchForbiddenClaim(item); ok {
				return field, true
			}
		}
	}
	return "", false
}

func agentBenchPositiveClaim(value string) bool {
	switch agentBenchNormField(value) {
	case "pass", "passed", "accept", "accepted", "satisfied", "usable":
		return true
	default:
		return false
	}
}

func readAgentBenchDir(t *testing.T, root string) string {
	t.Helper()
	var contents strings.Builder
	err := filepath.WalkDir(root, func(path string, entry fs.DirEntry, err error) error {
		if err != nil {
			return err
		}
		if entry.IsDir() {
			return nil
		}
		body, err := os.ReadFile(path)
		if err != nil {
			return err
		}
		contents.Write(body)
		return nil
	})
	if err != nil {
		t.Fatal(err)
	}
	return contents.String()
}
