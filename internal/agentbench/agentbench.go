// Package agentbench implements the bounded, replay-only AgentBench runner.
package agentbench

import (
	"bufio"
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"io"
	"io/fs"
	"net/http"
	"net/url"
	"os"
	"os/exec"
	"path/filepath"
	"sort"
	"strings"
	"sync"
	"time"

	"github.com/anthony-chaudhary/fak/internal/agentbench/profileplan"
	"github.com/anthony-chaudhary/fak/internal/agentbench/taskrun"
)

const (
	quickRequestCount  = 16
	quickMaxInFlight   = 2
	parentTimeout      = 5 * time.Minute
	requestTimeout     = 1 * time.Minute
	progressInterval   = 5 * time.Second
	maxStreamBytes     = 8 << 20
	maxResponseBytes   = 1 << 20
	maxSourceBytes     = 64 << 10
	maxSourceFileBytes = 16 << 10
)

const frozenSystemInput = `You are serving a replay-only AgentBench diagnostic. Use only the frozen inputs and recorded history supplied in this request. Do not modify the source repository, infer hardware, or claim task acceptance.`

var errCanceled = errors.New("agentbench replay canceled")

type options struct {
	profile  string
	repo     string
	endpoint string
	model    string
	out      string
	plan     bool
	json     bool
}

type rungPlan struct {
	Name        string `json:"name"`
	Concurrency int    `json:"concurrency"`
	Sessions    int    `json:"sessions"`
	Turns       int    `json:"turns_per_session"`
	Requests    int    `json:"scored_requests"`
}

type inputDescriptor struct {
	Path   string `json:"path"`
	SHA256 string `json:"sha256"`
	Bytes  int    `json:"bytes"`
}

type tokenizerKnowledge struct {
	Status string `json:"status"`
	Reason string `json:"reason"`
}

type replayPlan struct {
	Schema             string                     `json:"schema"`
	Mode               string                     `json:"mode"`
	Profile            string                     `json:"profile"`
	RequestedModel     string                     `json:"requested_model,omitempty"`
	Rungs              []rungPlan                 `json:"rungs"`
	ScoredRequests     int                        `json:"scored_requests"`
	MaxInFlight        int                        `json:"max_in_flight"`
	InputArtifacts     map[string]inputDescriptor `json:"input_artifacts"`
	ReferenceTokenizer tokenizerKnowledge         `json:"reference_tokenizer"`
	ModelIdentity      string                     `json:"model_identity_policy"`
	CoverageBound      string                     `json:"coverage_bound"`
	ContextBound       string                     `json:"context_bound"`
	OutputBound        string                     `json:"output_bound"`
	CostBound          string                     `json:"cost_bound"`
	DeadlineBound      string                     `json:"deadline_bound"`
	DigestBound        string                     `json:"digest_bound"`
	RetryBound         string                     `json:"retry_bound"`
	Dispatch           bool                       `json:"dispatch"`
	TaskFixtures       int                        `json:"task_fixtures"`
	MaxTaskTurns       int                        `json:"max_task_turns"`
	RequestCeiling     int                        `json:"request_ceiling"`
}

type manifest struct {
	Schema               string                     `json:"schema"`
	RunID                string                     `json:"run_id"`
	Mode                 string                     `json:"mode"`
	Profile              string                     `json:"profile"`
	Repository           string                     `json:"repository"`
	Endpoint             string                     `json:"endpoint"`
	RequestedModel       string                     `json:"requested_model,omitempty"`
	ModelIdentityPolicy  string                     `json:"model_identity_policy"`
	Rungs                []rungPlan                 `json:"rungs"`
	ScoredRequests       int                        `json:"scored_requests"`
	MaxInFlight          int                        `json:"max_in_flight"`
	ParentTimeoutMillis  int64                      `json:"parent_timeout_ms"`
	RequestTimeoutMillis int64                      `json:"request_timeout_ms"`
	MaxStreamBytes       int64                      `json:"max_stream_bytes"`
	MaxResponseBytes     int64                      `json:"max_response_bytes"`
	InputArtifacts       map[string]inputDescriptor `json:"input_artifacts"`
	ReferenceTokenizer   tokenizerKnowledge         `json:"reference_tokenizer"`
	UsagePolicy          string                     `json:"usage_policy"`
	CachePolicy          string                     `json:"cache_policy"`
	OverallVerdict       string                     `json:"overall_verdict"`
	TaskVerdict          string                     `json:"task_verdict"`
	StartedAt            string                     `json:"started_at"`
	TaskFixtures         int                        `json:"task_fixtures"`
	MaxTaskTurns         int                        `json:"max_task_turns"`
	RequestCeiling       int                        `json:"request_ceiling"`
}

type toolFunction struct {
	Name      string `json:"name"`
	Arguments string `json:"arguments"`
}

type toolCall struct {
	ID       string       `json:"id"`
	Type     string       `json:"type"`
	Function toolFunction `json:"function"`
}

type chatMessage struct {
	Role       string     `json:"role"`
	Content    string     `json:"content"`
	ToolCallID string     `json:"tool_call_id,omitempty"`
	ToolCalls  []toolCall `json:"tool_calls,omitempty"`
}

type requestMetadata struct {
	Benchmark     string `json:"benchmark"`
	Profile       string `json:"profile"`
	RequestID     string `json:"request_id,omitempty"`
	Rung          string `json:"rung"`
	Session       string `json:"session"`
	Turn          int    `json:"turn"`
	Sequence      int    `json:"sequence"`
	ScoredRequest bool   `json:"scored_request"`
}

type streamOptions struct {
	IncludeUsage bool `json:"include_usage"`
}

type requestEnvelope struct {
	Model         string            `json:"model,omitempty"`
	Stream        bool              `json:"stream"`
	StreamOptions streamOptions     `json:"stream_options"`
	Messages      []chatMessage     `json:"messages"`
	Tools         []json.RawMessage `json:"tools,omitempty"`
	MaxTokens     int               `json:"max_tokens,omitempty"`
	Metadata      requestMetadata   `json:"metadata"`
}

type observedUsage struct {
	PromptTokens     int `json:"prompt_tokens"`
	CompletionTokens int `json:"completion_tokens"`
	TotalTokens      int `json:"total_tokens"`
}

type lifecycleEvent struct {
	Schema               string         `json:"schema"`
	Event                string         `json:"event"`
	EventType            string         `json:"event_type,omitempty"`
	Phase                string         `json:"phase,omitempty"`
	Sequence             int            `json:"sequence"`
	RequestID            string         `json:"request_id"`
	Rung                 string         `json:"rung"`
	ConditionID          string         `json:"condition_id"`
	Session              string         `json:"session"`
	SessionID            string         `json:"session_id"`
	Turn                 int            `json:"turn"`
	ScoredRequest        bool           `json:"scored_request"`
	Status               string         `json:"status"`
	ServiceVerdict       string         `json:"service_verdict,omitempty"`
	CacheVerdict         string         `json:"cache_verdict,omitempty"`
	TaskVerdict          string         `json:"task_verdict"`
	RequestedModel       string         `json:"requested_model,omitempty"`
	ObservedModel        string         `json:"observed_model,omitempty"`
	ModelIdentityStatus  string         `json:"model_identity_status"`
	FinishReason         string         `json:"finish_reason,omitempty"`
	Response             string         `json:"response,omitempty"`
	UsageKnown           *bool          `json:"usage_known,omitempty"`
	Usage                *observedUsage `json:"usage,omitempty"`
	CacheKnown           *bool          `json:"cache_known,omitempty"`
	CachedPromptTokens   *int           `json:"cached_prompt_tokens,omitempty"`
	Error                string         `json:"error,omitempty"`
	StartedAt            string         `json:"started_at,omitempty"`
	CompletedAt          string         `json:"completed_at,omitempty"`
	DurationMilliseconds int64          `json:"duration_ms,omitempty"`
}

type usageSummary struct {
	KnownRequests    int `json:"known_requests"`
	UnknownRequests  int `json:"unknown_requests"`
	PromptTokens     int `json:"prompt_tokens"`
	CompletionTokens int `json:"completion_tokens"`
	TotalTokens      int `json:"total_tokens"`
}

type cacheSummary struct {
	KnownRequests      int  `json:"known_requests"`
	UnknownRequests    int  `json:"unknown_requests"`
	CachedPromptTokens *int `json:"cached_prompt_tokens,omitempty"`
}

type rungSummary struct {
	Name                 string `json:"name"`
	ConcurrencyLimit     int    `json:"concurrency_limit"`
	ObservedPeakInFlight int    `json:"observed_peak_in_flight"`
	PlannedRequests      int    `json:"planned_requests"`
	TerminalRequests     int    `json:"terminal_requests"`
	CompletedRequests    int    `json:"completed_requests"`
	FailedRequests       int    `json:"failed_requests"`
}

type runSummary struct {
	Schema              string             `json:"schema"`
	RunID               string             `json:"run_id"`
	Mode                string             `json:"mode"`
	Profile             string             `json:"profile"`
	Status              string             `json:"status"`
	OverallVerdict      string             `json:"overall_verdict"`
	ServiceVerdict      string             `json:"service_verdict"`
	CacheVerdict        string             `json:"cache_verdict"`
	TaskVerdict         string             `json:"task_verdict"`
	RequestedModel      string             `json:"requested_model,omitempty"`
	ModelIdentityStatus string             `json:"model_identity_status"`
	PlannedRequests     int                `json:"planned_requests"`
	TerminalRequests    int                `json:"terminal_requests"`
	ScoredRequests      int                `json:"scored_requests"`
	FailedRequests      int                `json:"failed_requests"`
	Rungs               []rungSummary      `json:"rungs"`
	Usage               usageSummary       `json:"usage"`
	Cache               cacheSummary       `json:"cache"`
	ObservedModels      []string           `json:"observed_models"`
	ReferenceTokenizer  tokenizerKnowledge `json:"reference_tokenizer"`
	StartedAt           string             `json:"started_at"`
	CompletedAt         string             `json:"completed_at"`
	Error               string             `json:"error,omitempty"`
	TaskResult          *taskrun.Receipt   `json:"task_result,omitempty"`
	TaskReceiptSHA256   string             `json:"task_receipt_sha256,omitempty"`
	Phase               string             `json:"phase"`
	TaskFixtures        int                `json:"task_fixtures"`
	MaxTaskTurns        int                `json:"max_task_turns"`
	RequestCeiling      int                `json:"request_ceiling"`
}

type receipt struct {
	Schema         string            `json:"schema"`
	RunID          string            `json:"run_id"`
	Algorithm      string            `json:"algorithm"`
	ArtifactSHA256 map[string]string `json:"artifact_sha256"`
	RunSHA256      string            `json:"run_sha256"`
}

type frozenInputs struct {
	system     string
	repo       string
	area       map[string]string
	sourceRoot string
}

type plannedRequest struct {
	sequence int
	rung     string
	session  string
	turn     int
}

type streamObservation struct {
	response           string
	usage              *observedUsage
	cacheKnown         bool
	cachedPromptTokens int
	model              string
	finishReason       string
}

type eventSink struct {
	mu        sync.Mutex
	file      *os.File
	terminal  map[int]lifecycleEvent
	inFlight  map[string]int
	peak      map[string]int
	writeErr  error
	model     string
	lifecycle []lifecycleEvent
}

// RunCLI runs the AgentBench command and returns a process-style exit code.
func RunCLI(ctx context.Context, stdout, stderr io.Writer, args []string) int {
	if ctx == nil {
		ctx = context.Background()
	}
	if stdout == nil {
		stdout = io.Discard
	}
	if stderr == nil {
		stderr = io.Discard
	}
	if len(args) > 0 && args[0] == "compare" {
		if len(args) != 3 {
			fmt.Fprintln(stderr, "usage: fak bench agent compare <baseline-directory> <candidate-directory>")
			return 2
		}
		comparison, err := compareDirectories(args[1], args[2])
		if err != nil {
			fmt.Fprintf(stderr, "agentbench: compare: %v\n", err)
			return 1
		}
		if err := json.NewEncoder(stdout).Encode(comparison); err != nil {
			fmt.Fprintf(stderr, "agentbench: compare output: %v\n", err)
			return 1
		}
		return 0
	}
	if handled, code := runChildCLI(ctx, stdout, stderr, args); handled {
		return code
	}
	opts, err := parseOptions(stderr, args)
	if err != nil {
		return 2
	}
	if opts.profile != "quick" && opts.plan {
		plan, planErr := profileplan.Build(profileplan.Options{Profile: opts.profile, PreferredConcurrency: 2})
		if planErr != nil {
			fmt.Fprintf(stderr, "agentbench: plan: %v\n", planErr)
			return 2
		}
		if err := json.NewEncoder(stdout).Encode(plan); err != nil {
			fmt.Fprintf(stderr, "agentbench: print plan: %v\n", err)
			return 1
		}
		return 0
	}
	if opts.profile == "normal" {
		return runNormalCLICommand(ctx, stdout, stderr, opts)
	}
	if opts.profile != "quick" {
		fmt.Fprintf(stderr, "agentbench: UNSUPPORTED_INCOMPLETE profile %q; use --profile quick\n", opts.profile)
		return 2
	}
	if err := ctx.Err(); err != nil {
		fmt.Fprintln(stderr, "agentbench: replay canceled before planning")
		return 130
	}
	repo, err := resolveRepository(opts.repo)
	if err != nil {
		fmt.Fprintf(stderr, "agentbench: repository: %v\n", err)
		return 2
	}
	repoInput, err := freezeRepositoryInput(repo)
	if err != nil {
		fmt.Fprintf(stderr, "agentbench: freeze repository input: %v\n", err)
		return 1
	}
	inputs := newFrozenInputs(repoInput)
	plan := newQuickPlan(inputs, opts.model, !opts.plan)
	if opts.plan {
		if err := printValue(stdout, plan, opts.json); err != nil {
			fmt.Fprintf(stderr, "agentbench: print plan: %v\n", err)
			return 1
		}
		return 0
	}
	if opts.endpoint == "" {
		fmt.Fprintln(stderr, "agentbench: --endpoint is required unless --plan is set")
		return 2
	}
	if opts.out == "" {
		fmt.Fprintln(stderr, "agentbench: --out is required unless --plan is set")
		return 2
	}
	if opts.model == "" {
		fmt.Fprintln(stderr, "agentbench: --model is required for the quick replay plus task smoke")
		return 2
	}
	endpoint, err := normalizeEndpoint(opts.endpoint)
	if err != nil {
		fmt.Fprintf(stderr, "agentbench: endpoint: %v\n", err)
		return 2
	}
	out, err := resolveOutput(repo, opts.out)
	if err != nil {
		fmt.Fprintf(stderr, "agentbench: output: %v\n", err)
		return 2
	}
	summary, runErr := runQuick(ctx, stderr, repo, endpoint, opts.model, out, inputs, plan)
	if summary != nil {
		if err := printValue(stdout, summary, opts.json); err != nil && runErr == nil {
			runErr = fmt.Errorf("print summary: %w", err)
		}
	}
	if runErr == nil {
		return 0
	}
	if errors.Is(runErr, errCanceled) || errors.Is(runErr, context.Canceled) || errors.Is(runErr, context.DeadlineExceeded) {
		fmt.Fprintln(stderr, "agentbench: replay canceled")
		return 130
	}
	fmt.Fprintf(stderr, "agentbench: %v\n", runErr)
	return 1
}

func parseOptions(stderr io.Writer, args []string) (options, error) {
	opts := options{profile: "normal", repo: "."}
	flags := flag.NewFlagSet("bench agent", flag.ContinueOnError)
	flags.SetOutput(stderr)
	flags.StringVar(&opts.profile, "profile", opts.profile, "replay profile (quick; normal is not yet supported)")
	flags.StringVar(&opts.repo, "repo", opts.repo, "read-only source repository")
	flags.StringVar(&opts.endpoint, "endpoint", "", "OpenAI-compatible streaming endpoint")
	flags.StringVar(&opts.model, "model", "", "explicit model identifier; otherwise require streamed identity")
	flags.StringVar(&opts.out, "out", "", "artifact output directory outside the source repository")
	flags.BoolVar(&opts.plan, "plan", false, "print the replay plan without dispatching")
	flags.BoolVar(&opts.json, "json", false, "print machine-readable output")
	if err := flags.Parse(args); err != nil {
		return options{}, err
	}
	if flags.NArg() != 0 {
		return options{}, fmt.Errorf("unexpected positional arguments: %s", strings.Join(flags.Args(), " "))
	}
	return opts, nil
}

func newFrozenInputs(repoInput string) frozenInputs {
	return frozenInputs{
		system: frozenSystemInput,
		repo:   repoInput,
		area: map[string]string{
			"C1": "Frozen area input: C1 is the concurrency-one service replay rung. Observe service, usage, and cache telemetry only.",
			"C2": "Frozen area input: C2 is the concurrency-two service replay rung. Observe service, usage, and cache telemetry only.",
		},
	}
}

func newNormalInputs(repo, repoInput string) frozenInputs {
	inputs := newFrozenInputs(repoInput)
	inputs.sourceRoot = repo
	return inputs
}

func newQuickPlan(inputs frozenInputs, model string, dispatch bool) replayPlan {
	return replayPlan{
		Schema:         "fak.agentbench.plan.v1",
		Mode:           "replay-plus-task",
		Profile:        "quick",
		RequestedModel: model,
		Rungs:          quickRungs(),
		ScoredRequests: quickRequestCount,
		MaxInFlight:    quickMaxInFlight,
		InputArtifacts: describeInputs(inputs),
		ReferenceTokenizer: tokenizerKnowledge{
			Status: "unknown",
			Reason: "no reference tokenizer is supplied by the replay contract",
		},
		ModelIdentity:  "explicit requested identity is preserved; streamed identity is required and must match when explicit",
		CoverageBound:  "exactly 16 scored replay requests plus one real task fixture capped at 12 model turns",
		ContextBound:   "frozen repository input is capped at 65536 bytes and each source file at 16384 bytes",
		OutputBound:    "each decoded response is capped at 1048576 bytes and each SSE stream at 8388608 bytes",
		CostBound:      "provider price and monetary cost are unknown; only provider-reported token usage is recorded",
		DeadlineBound:  "five-minute run deadline and one-minute per-request deadline",
		DigestBound:    "SHA-256 covers every frozen input and final artifact receipt",
		RetryBound:     "zero retries; exactly one dispatch attempt per scored request",
		Dispatch:       dispatch,
		TaskFixtures:   1,
		MaxTaskTurns:   12,
		RequestCeiling: 28,
	}
}

func quickRungs() []rungPlan {
	return []rungPlan{
		{Name: "C1", Concurrency: 1, Sessions: 2, Turns: 4, Requests: 8},
		{Name: "C2", Concurrency: 2, Sessions: 2, Turns: 4, Requests: 8},
	}
}

func plannedRequests() []plannedRequest {
	requests := make([]plannedRequest, 0, quickRequestCount)
	sequence := 1
	for _, rung := range []string{"C1", "C2"} {
		for session := 1; session <= 2; session++ {
			for turn := 1; turn <= 4; turn++ {
				requests = append(requests, plannedRequest{
					sequence: sequence,
					rung:     rung,
					session:  fmt.Sprintf("%s-S%d", rung, session),
					turn:     turn,
				})
				sequence++
			}
		}
	}
	return requests
}

func runQuick(ctx context.Context, progress io.Writer, repo, endpoint, requestedModel, out string, inputs frozenInputs, plan replayPlan) (*runSummary, error) {
	state, err := runQuickReplay(ctx, progress, repo, endpoint, requestedModel, out, inputs, plan)
	if err != nil {
		return nil, err
	}
	summary := state.summary
	taskReceiptPath := ""
	if state.runCtx.Err() == nil && state.writeErr == nil && summary.FailedRequests == 0 && resolvedRequestedModel(summary, requestedModel) {
		if err := state.setPhase("task"); err == nil {
			taskResult, taskPath, taskErr := runQuickTask(state.runCtx, out, endpoint, requestedModel)
			summary.TaskResult = taskResult
			taskReceiptPath = taskPath
			if taskErr == nil && quickTaskAccepted(taskResult) {
				summary.TaskVerdict = "ACCEPTED"
				summary.OverallVerdict = "QUICK_SMOKE_PASSED"
			} else {
				summary.Status = "incomplete"
				summary.TaskVerdict = "REJECTED"
				summary.OverallVerdict = "INCOMPLETE"
				if taskErr != nil {
					summary.Error = taskErr.Error()
				} else {
					summary.Error = "real task was not accepted by the external verifier"
				}
			}
		}
	}
	if summary.TaskVerdict != "ACCEPTED" {
		summary.Status = "incomplete"
		summary.TaskVerdict = "REJECTED"
		summary.OverallVerdict = "INCOMPLETE"
		if summary.Error == "" {
			summary.Error = "replay service/model witness did not qualify the task phase"
		}
	}
	var taskArtifactErr error
	if taskReceiptPath != "" {
		summary.TaskReceiptSHA256, taskArtifactErr = hashFile(taskReceiptPath)
		if taskArtifactErr != nil {
			summary.Status = "incomplete"
			summary.TaskVerdict = "REJECTED"
			summary.OverallVerdict = "INCOMPLETE"
			summary.Error = "hash task receipt: " + taskArtifactErr.Error()
			taskReceiptPath = ""
		}
	}
	var extras []string
	if taskReceiptPath != "" {
		extras = append(extras, taskReceiptPath)
	}
	final, closeErr := state.Close(summary, extras...)
	if closeErr != nil {
		return final, closeErr
	}
	if taskArtifactErr != nil {
		return final, taskArtifactErr
	}
	if final.FailedRequests != 0 {
		return final, fmt.Errorf("%d replay requests did not complete", final.FailedRequests)
	}
	if final.TaskVerdict != "ACCEPTED" {
		return final, errors.New("quick task smoke incomplete")
	}
	return final, nil
}

func runRung(ctx context.Context, client *http.Client, endpoint, requestedModel string, inputs frozenInputs, sink *eventSink, requests []plannedRequest, concurrency int) {
	bySession := make(map[string][]plannedRequest, 2)
	order := make([]string, 0, 2)
	for _, request := range requests {
		if _, ok := bySession[request.session]; !ok {
			order = append(order, request.session)
		}
		bySession[request.session] = append(bySession[request.session], request)
	}
	if concurrency == 1 {
		for _, session := range order {
			if ctx.Err() != nil {
				return
			}
			runSession(ctx, client, endpoint, requestedModel, inputs, sink, bySession[session])
		}
		return
	}
	var workers sync.WaitGroup
	for _, session := range order {
		if ctx.Err() != nil {
			break
		}
		workers.Add(1)
		go func(sessionRequests []plannedRequest) {
			defer workers.Done()
			runSession(ctx, client, endpoint, requestedModel, inputs, sink, sessionRequests)
		}(bySession[session])
	}
	workers.Wait()
}

func runSession(ctx context.Context, client *http.Client, endpoint, requestedModel string, inputs frozenInputs, sink *eventSink, requests []plannedRequest) {
	for _, request := range requests {
		if ctx.Err() != nil {
			return
		}
		started := time.Now().UTC()
		if err := sink.start(request, requestedModel, started); err != nil {
			return
		}
		for _, phase := range []string{"release", "enqueue", "dispatch"} {
			if err := sink.phase(request, requestedModel, phase, time.Now().UTC()); err != nil {
				return
			}
		}
		requestCtx, cancel := context.WithTimeout(ctx, requestTimeout)
		observation, dispatchErr := dispatchWithProgress(requestCtx, client, endpoint, requestEnvelope{
			Model:         requestedModel,
			Stream:        true,
			StreamOptions: streamOptions{IncludeUsage: true},
			Messages:      frozenMessages(inputs, request),
			Tools:         normalToolContracts(),
			Metadata: requestMetadata{
				Benchmark:     "agentbench",
				Profile:       "quick",
				Rung:          request.rung,
				Session:       request.session,
				Turn:          request.turn,
				Sequence:      request.sequence,
				ScoredRequest: true,
			},
		}, requestedModel,
			func(at time.Time) error { return sink.phase(request, requestedModel, "first_output", at) },
			func(at time.Time) error { return sink.phase(request, requestedModel, "tool_call", at) },
			func(at time.Time) error { return sink.phase(request, requestedModel, "stream_progress", at) })
		requestContextErr := requestCtx.Err()
		cancel()
		completed := time.Now().UTC()
		terminal := terminalEvent(request, requestedModel, started, completed, observation, dispatchErr, requestContextErr, ctx.Err())
		if err := sink.finish(terminal); err != nil {
			return
		}
	}
}

func frozenMessages(inputs frozenInputs, request plannedRequest) []chatMessage {
	messages := []chatMessage{
		{Role: "system", Content: inputs.system},
		{Role: "system", Content: inputs.repo},
		{Role: "system", Content: inputs.area[request.rung]},
	}
	for turn := 1; turn < request.turn; turn++ {
		callID := fmt.Sprintf("recorded-%s-T%d", request.session, turn)
		messages = append(messages,
			chatMessage{Role: "user", Content: recordedUserPrompt(request.rung, turn)},
			chatMessage{
				Role:    "assistant",
				Content: recordedAssistantReply(request.rung, turn),
				ToolCalls: []toolCall{{
					ID:   callID,
					Type: "function",
					Function: toolFunction{
						Name:      "recorded_source_observation",
						Arguments: fmt.Sprintf(`{"rung":%q,"turn":%d}`, request.rung, turn),
					},
				}},
			},
			chatMessage{Role: "tool", Content: recordedToolReply(request.rung, turn), ToolCallID: callID},
		)
	}
	return append(messages, chatMessage{Role: "user", Content: recordedUserPrompt(request.rung, request.turn)})
}

func recordedUserPrompt(rung string, turn int) string {
	prompts := []string{
		"Using the frozen repository context, identify the primary implementation seam relevant to the recorded task.",
		"Using the frozen inputs and recorded history, state the smallest safe change at that seam.",
		"Name the bounded verification that would distinguish the intended behavior from a regression.",
		"Summarize the service-level evidence boundary without claiming hardware characteristics or task acceptance.",
	}
	return fmt.Sprintf("%s replay turn %d: %s", rung, turn, prompts[turn-1])
}

func recordedAssistantReply(rung string, turn int) string {
	return fmt.Sprintf("Recorded assistant history for %s turn %d; this frozen text is not a live endpoint result.", rung, turn)
}

func recordedToolReply(rung string, turn int) string {
	return fmt.Sprintf("Recorded tool observation for %s turn %d; repository access remained read-only.", rung, turn)
}

func terminalEvent(request plannedRequest, requestedModel string, started, completed time.Time, observation streamObservation, dispatchErr, requestContextErr, parentErr error) lifecycleEvent {
	usageKnown := observation.usage != nil
	cacheKnown := observation.cacheKnown
	event := lifecycleEvent{
		Schema:               "fak.agentbench.event.v1",
		Event:                "request_terminal",
		EventType:            "end",
		Phase:                "end",
		Sequence:             request.sequence,
		RequestID:            requestID(request),
		Rung:                 request.rung,
		ConditionID:          request.rung,
		Session:              request.session,
		SessionID:            request.session,
		Turn:                 request.turn,
		ScoredRequest:        true,
		Status:               "completed",
		ServiceVerdict:       "OBSERVED_OK",
		CacheVerdict:         cacheVerdict(cacheKnown),
		TaskVerdict:          "NOT_EVALUATED",
		RequestedModel:       requestedModel,
		ObservedModel:        observation.model,
		ModelIdentityStatus:  modelIdentityStatus(requestedModel, observation.model),
		FinishReason:         observation.finishReason,
		Response:             observation.response,
		UsageKnown:           boolPointer(usageKnown),
		Usage:                observation.usage,
		CacheKnown:           boolPointer(cacheKnown),
		StartedAt:            started.Format(time.RFC3339Nano),
		CompletedAt:          completed.Format(time.RFC3339Nano),
		DurationMilliseconds: completed.Sub(started).Milliseconds(),
	}
	if cacheKnown {
		event.CachedPromptTokens = intPointer(observation.cachedPromptTokens)
	}
	if dispatchErr == nil {
		return event
	}
	event.Status = "failed"
	event.ServiceVerdict = "OBSERVED_FAILURE"
	event.Error = dispatchErr.Error()
	if parentErr != nil {
		event.Status = "canceled"
		event.ServiceVerdict = "CANCELED"
	} else if errors.Is(dispatchErr, context.DeadlineExceeded) || errors.Is(requestContextErr, context.DeadlineExceeded) {
		event.Status = "timed_out"
		event.ServiceVerdict = "OBSERVED_TIMEOUT"
	}
	return event
}

func (sink *eventSink) start(request plannedRequest, requestedModel string, started time.Time) error {
	sink.mu.Lock()
	defer sink.mu.Unlock()
	event := lifecycleEvent{
		Schema:              "fak.agentbench.event.v1",
		Event:               "request_started",
		EventType:           "start",
		Phase:               "start",
		Sequence:            request.sequence,
		RequestID:           requestID(request),
		Rung:                request.rung,
		ConditionID:         request.rung,
		Session:             request.session,
		SessionID:           request.session,
		Turn:                request.turn,
		ScoredRequest:       true,
		Status:              "in_flight",
		TaskVerdict:         "NOT_EVALUATED",
		RequestedModel:      requestedModel,
		ModelIdentityStatus: modelIdentityStatus(requestedModel, ""),
		StartedAt:           started.Format(time.RFC3339Nano),
	}
	if err := sink.appendLocked(event); err != nil {
		return err
	}
	sink.inFlight[request.rung]++
	if sink.inFlight[request.rung] > sink.peak[request.rung] {
		sink.peak[request.rung] = sink.inFlight[request.rung]
	}
	return nil
}

func (sink *eventSink) phase(request plannedRequest, requestedModel, phase string, at time.Time) error {
	sink.mu.Lock()
	defer sink.mu.Unlock()
	eventName := "request_lifecycle"
	if phase == "first_output" {
		eventName = "first_output"
	}
	if phase == "tool_call" {
		eventName = "tool"
	}
	if phase == "stream_progress" {
		eventName = "progress"
	}
	return sink.appendLocked(lifecycleEvent{
		Schema:              "fak.agentbench.event.v1",
		Event:               eventName,
		EventType:           phase,
		Phase:               phase,
		Sequence:            request.sequence,
		RequestID:           requestID(request),
		Rung:                request.rung,
		ConditionID:         request.rung,
		Session:             request.session,
		SessionID:           request.session,
		Turn:                request.turn,
		ScoredRequest:       true,
		Status:              phase,
		TaskVerdict:         "NOT_EVALUATED",
		RequestedModel:      requestedModel,
		ModelIdentityStatus: modelIdentityStatus(requestedModel, ""),
		StartedAt:           at.Format(time.RFC3339Nano),
	})
}

func (sink *eventSink) finish(event lifecycleEvent) error {
	sink.mu.Lock()
	defer sink.mu.Unlock()
	if _, exists := sink.terminal[event.Sequence]; exists {
		return nil
	}
	if sink.inFlight[event.Rung] > 0 {
		sink.inFlight[event.Rung]--
	}
	if event.Status == "completed" {
		if sink.model == "" {
			sink.model = event.ObservedModel
		} else if sink.model != event.ObservedModel {
			event.Status = "failed"
			event.ServiceVerdict = "OBSERVED_FAILURE"
			event.Error = fmt.Sprintf("observed model %q does not match run model %q", event.ObservedModel, sink.model)
		}
	}
	sink.terminal[event.Sequence] = event
	return sink.appendLocked(event)
}

func (sink *eventSink) terminalizeMissing(requests []plannedRequest, requestedModel, status, message string) {
	for _, request := range requests {
		sink.mu.Lock()
		if _, exists := sink.terminal[request.sequence]; exists {
			sink.mu.Unlock()
			continue
		}
		now := time.Now().UTC().Format(time.RFC3339Nano)
		usageKnown := false
		cacheKnown := false
		serviceVerdict := "NOT_DISPATCHED"
		if status == "canceled" {
			serviceVerdict = "CANCELED"
		}
		event := lifecycleEvent{
			Schema:              "fak.agentbench.event.v1",
			Event:               "request_terminal",
			EventType:           "end",
			Phase:               "end",
			Sequence:            request.sequence,
			RequestID:           requestID(request),
			Rung:                request.rung,
			ConditionID:         request.rung,
			Session:             request.session,
			SessionID:           request.session,
			Turn:                request.turn,
			ScoredRequest:       true,
			Status:              status,
			ServiceVerdict:      serviceVerdict,
			CacheVerdict:        "UNKNOWN",
			TaskVerdict:         "NOT_EVALUATED",
			RequestedModel:      requestedModel,
			ModelIdentityStatus: modelIdentityStatus(requestedModel, ""),
			UsageKnown:          &usageKnown,
			CacheKnown:          &cacheKnown,
			Error:               message,
			CompletedAt:         now,
		}
		sink.terminal[request.sequence] = event
		_ = sink.appendLocked(event)
		sink.mu.Unlock()
	}
}

func (sink *eventSink) appendLocked(event lifecycleEvent) error {
	encoded, err := json.Marshal(event)
	if err == nil {
		encoded = append(encoded, '\n')
		_, err = sink.file.Write(encoded)
	}
	if err == nil {
		err = sink.file.Sync()
	}
	if err == nil {
		sink.lifecycle = append(sink.lifecycle, event)
	}
	if err != nil && sink.writeErr == nil {
		sink.writeErr = fmt.Errorf("append lifecycle event: %w", err)
	}
	return err
}

func (sink *eventSink) events() []lifecycleEvent {
	sink.mu.Lock()
	defer sink.mu.Unlock()
	return append([]lifecycleEvent(nil), sink.lifecycle...)
}

func (sink *eventSink) progress() (int, map[string]int) {
	sink.mu.Lock()
	defer sink.mu.Unlock()
	return len(sink.terminal), map[string]int{"C1": sink.peak["C1"], "C2": sink.peak["C2"]}
}

func (sink *eventSink) terminals() []lifecycleEvent {
	sink.mu.Lock()
	defer sink.mu.Unlock()
	result := make([]lifecycleEvent, 0, len(sink.terminal))
	for _, event := range sink.terminal {
		result = append(result, event)
	}
	sort.Slice(result, func(i, j int) bool { return result[i].Sequence < result[j].Sequence })
	return result
}

func (sink *eventSink) peaks() map[string]int {
	sink.mu.Lock()
	defer sink.mu.Unlock()
	result := make(map[string]int, len(sink.peak))
	for name, peak := range sink.peak {
		result[name] = peak
	}
	return result
}

func (sink *eventSink) err() error {
	sink.mu.Lock()
	defer sink.mu.Unlock()
	return sink.writeErr
}

func (sink *eventSink) recordError(err error) {
	if err == nil {
		return
	}
	sink.mu.Lock()
	defer sink.mu.Unlock()
	if sink.writeErr == nil {
		sink.writeErr = err
	}
}

func requestID(request plannedRequest) string {
	return fmt.Sprintf("quick-%s-T%d", request.session, request.turn)
}

func cacheVerdict(known bool) string {
	if known {
		return "OBSERVED_KNOWN"
	}
	return "UNKNOWN"
}

func modelIdentityStatus(requested, observed string) string {
	if observed != "" {
		return "observed"
	}
	if requested != "" {
		return "explicit_unobserved"
	}
	return "unknown"
}

type tokenDetailsWire struct {
	CachedTokens *int `json:"cached_tokens"`
}

type usageWire struct {
	PromptTokens       *int              `json:"prompt_tokens"`
	CompletionTokens   *int              `json:"completion_tokens"`
	InputTokens        *int              `json:"input_tokens"`
	OutputTokens       *int              `json:"output_tokens"`
	TotalTokens        *int              `json:"total_tokens"`
	CachedTokens       *int              `json:"cached_tokens"`
	CacheReadTokens    *int              `json:"cache_read_input_tokens"`
	PromptTokenDetails *tokenDetailsWire `json:"prompt_tokens_details"`
	InputTokenDetails  *tokenDetailsWire `json:"input_tokens_details"`
}

type streamChunk struct {
	Model   string `json:"model"`
	Choices []struct {
		Delta struct {
			Content   string `json:"content"`
			ToolCalls []struct {
				Index    int    `json:"index"`
				ID       string `json:"id"`
				Type     string `json:"type"`
				Function struct {
					Name      string `json:"name"`
					Arguments string `json:"arguments"`
				} `json:"function"`
			} `json:"tool_calls"`
		} `json:"delta"`
		FinishReason *string `json:"finish_reason"`
	} `json:"choices"`
	Usage *usageWire `json:"usage"`
	Error *struct {
		Message string `json:"message"`
	} `json:"error"`
}

type sseState struct {
	done                bool
	content             strings.Builder
	usage               *observedUsage
	usageIncompleteSeen bool
	cacheKnown          bool
	cacheObserved       bool
	cachedTokens        int
	model               string
	finishReason        string
	firstOutput         bool
	toolNotified        bool
	toolNames           map[int]string
	toolIDs             map[int]string
	toolArguments       map[int]string
	allowedTools        map[string]bool
	onFirstOutput       func(time.Time) error
	onTool              func(time.Time) error
	onProgress          func(time.Time) error
}

func dispatch(ctx context.Context, client *http.Client, endpoint string, envelope requestEnvelope, requestedModel string) (streamObservation, error) {
	return dispatchWithProgress(ctx, client, endpoint, envelope, requestedModel, nil, nil, nil)
}

func dispatchWithProgress(ctx context.Context, client *http.Client, endpoint string, envelope requestEnvelope, requestedModel string, onFirstOutput, onTool, onProgress func(time.Time) error) (streamObservation, error) {
	body, err := json.Marshal(envelope)
	if err != nil {
		return streamObservation{}, fmt.Errorf("encode request: %w", err)
	}
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, endpoint, bytes.NewReader(body))
	if err != nil {
		return streamObservation{}, fmt.Errorf("create request: %w", err)
	}
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Accept", "text/event-stream")
	if sessionID := benchmarkSessionID(envelope.Metadata); sessionID != "" {
		req.Header.Set("X-Fak-Session-Id", sessionID)
	}
	if traceID := benchmarkTraceID(envelope.Metadata); traceID != "" {
		req.Header.Set("X-Trace-Id", traceID)
	}
	resp, err := client.Do(req)
	if err != nil {
		return streamObservation{}, fmt.Errorf("dispatch request: %w", err)
	}
	defer resp.Body.Close()
	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		message, _ := io.ReadAll(io.LimitReader(resp.Body, 4<<10))
		return streamObservation{}, fmt.Errorf("endpoint returned %s: %s", resp.Status, strings.TrimSpace(string(message)))
	}
	contentType := strings.ToLower(resp.Header.Get("Content-Type"))
	if !strings.HasPrefix(contentType, "text/event-stream") {
		return streamObservation{}, fmt.Errorf("endpoint returned non-streaming content type %q", resp.Header.Get("Content-Type"))
	}
	limited := &io.LimitedReader{R: resp.Body, N: maxStreamBytes + 1}
	scanner := bufio.NewScanner(limited)
	scanner.Buffer(make([]byte, 4<<10), maxResponseBytes+1)
	state := &sseState{toolNames: map[int]string{}, toolIDs: map[int]string{}, toolArguments: map[int]string{}, allowedTools: declaredToolNames(envelope.Tools), onFirstOutput: onFirstOutput, onTool: onTool, onProgress: onProgress}
	dataLines := make([]string, 0, 1)
	flush := func() error {
		if len(dataLines) == 0 {
			return nil
		}
		data := strings.Join(dataLines, "\n")
		dataLines = dataLines[:0]
		return consumeSSEData(data, state)
	}
	for scanner.Scan() {
		line := scanner.Text()
		if line == "" {
			if err := flush(); err != nil {
				return streamObservation{}, err
			}
			continue
		}
		if strings.HasPrefix(line, ":") {
			continue
		}
		field, value, found := strings.Cut(line, ":")
		if !found {
			field, value = line, ""
		}
		if strings.HasPrefix(value, " ") {
			value = value[1:]
		}
		if field == "data" {
			dataLines = append(dataLines, value)
		}
	}
	if err := scanner.Err(); err != nil {
		return streamObservation{}, fmt.Errorf("read SSE stream: %w", err)
	}
	if limited.N == 0 {
		return streamObservation{}, fmt.Errorf("SSE stream exceeded %d-byte cap", maxStreamBytes)
	}
	if len(dataLines) != 0 {
		return streamObservation{}, errors.New("SSE stream ended with an unterminated event frame")
	}
	if !state.done {
		return streamObservation{}, errors.New("SSE stream ended without [DONE]")
	}
	if state.finishReason == "" {
		return streamObservation{}, errors.New("SSE stream did not report a finish reason")
	}
	if state.model == "" {
		return streamObservation{}, errors.New("SSE stream did not report model identity")
	}
	if requestedModel != "" && state.model != "" && state.model != requestedModel {
		return streamObservation{}, fmt.Errorf("observed model %q does not match requested model %q", state.model, requestedModel)
	}
	return streamObservation{
		response:           state.content.String(),
		usage:              state.usage,
		cacheKnown:         state.cacheKnown,
		cachedPromptTokens: state.cachedTokens,
		model:              state.model,
		finishReason:       state.finishReason,
	}, nil
}

func benchmarkSessionID(metadata requestMetadata) string {
	if metadata.Session == "" {
		return ""
	}
	return fmt.Sprintf("agentbench-%s-%s", metadata.Profile, metadata.Session)
}

func benchmarkTraceID(metadata requestMetadata) string {
	if metadata.Session == "" || metadata.Sequence <= 0 {
		return ""
	}
	return fmt.Sprintf("agentbench-%s-%s-t%d-q%d", metadata.Profile, metadata.Session, metadata.Turn, metadata.Sequence)
}

func consumeSSEData(data string, state *sseState) error {
	if state.done {
		return errors.New("SSE data received after [DONE]")
	}
	if data == "[DONE]" {
		state.done = true
		return nil
	}
	var chunk streamChunk
	if err := json.Unmarshal([]byte(data), &chunk); err != nil {
		return fmt.Errorf("decode SSE data: %w", err)
	}
	if chunk.Error != nil {
		return fmt.Errorf("stream error: %s", chunk.Error.Message)
	}
	if chunk.Model != "" {
		if state.model != "" && state.model != chunk.Model {
			return fmt.Errorf("stream model identity changed from %q to %q", state.model, chunk.Model)
		}
		state.model = chunk.Model
	}
	for _, choice := range chunk.Choices {
		hasProgress := choice.Delta.Content != "" || len(choice.Delta.ToolCalls) != 0
		if hasProgress && state.onProgress != nil {
			if err := state.onProgress(time.Now().UTC()); err != nil {
				return err
			}
		}
		if state.content.Len()+len(choice.Delta.Content) > maxResponseBytes {
			return fmt.Errorf("streamed response exceeded %d-byte cap", maxResponseBytes)
		}
		state.content.WriteString(choice.Delta.Content)
		if choice.Delta.Content != "" && !state.firstOutput {
			state.firstOutput = true
			if state.onFirstOutput != nil {
				if err := state.onFirstOutput(time.Now().UTC()); err != nil {
					return err
				}
			}
		}
		for _, delta := range choice.Delta.ToolCalls {
			if delta.ID != "" {
				state.toolIDs[delta.Index] = delta.ID
			}
			state.toolNames[delta.Index] += delta.Function.Name
			state.toolArguments[delta.Index] += delta.Function.Arguments
			if !state.firstOutput {
				state.firstOutput = true
				if state.onFirstOutput != nil {
					if err := state.onFirstOutput(time.Now().UTC()); err != nil {
						return err
					}
				}
			}
		}
		if choice.FinishReason != nil && *choice.FinishReason != "" {
			if state.finishReason != "" && state.finishReason != *choice.FinishReason {
				return fmt.Errorf("finish reason changed from %q to %q", state.finishReason, *choice.FinishReason)
			}
			state.finishReason = *choice.FinishReason
			if *choice.FinishReason == "tool_calls" && !state.toolNotified {
				valid := len(state.toolArguments) > 0
				for index, arguments := range state.toolArguments {
					var value map[string]any
					if state.toolIDs[index] == "" || !state.allowedTools[state.toolNames[index]] || json.Unmarshal([]byte(arguments), &value) != nil || value == nil {
						valid = false
					}
				}
				if !valid {
					return errors.New("stream tool call was incomplete or had invalid arguments")
				}
				state.toolNotified = true
				if state.onTool != nil {
					if err := state.onTool(time.Now().UTC()); err != nil {
						return err
					}
				}
			}
		}
	}
	if chunk.Usage != nil {
		usage, usageKnown, cacheKnown, cachedTokens, err := validateUsage(chunk.Usage)
		if err != nil {
			return err
		}
		if !usageKnown {
			if state.usage != nil {
				return errors.New("stream usage changed from complete to incomplete")
			}
			state.usageIncompleteSeen = true
			return nil
		}
		if state.usageIncompleteSeen {
			return errors.New("stream usage changed from incomplete to complete")
		}
		if state.usage != nil && *state.usage != *usage {
			return errors.New("stream reported inconsistent usage values")
		}
		if state.cacheObserved && state.cacheKnown != cacheKnown {
			return errors.New("stream reported inconsistent cache knowledge")
		}
		if state.cacheObserved && cacheKnown && state.cachedTokens != cachedTokens {
			return errors.New("stream reported inconsistent cached-token values")
		}
		state.usage = usage
		state.cacheObserved = true
		state.cacheKnown = cacheKnown
		state.cachedTokens = cachedTokens
	}
	return nil
}

func declaredToolNames(raw []json.RawMessage) map[string]bool {
	result := map[string]bool{}
	for _, item := range raw {
		var wire struct {
			Function struct {
				Name string `json:"name"`
			} `json:"function"`
		}
		if json.Unmarshal(item, &wire) == nil && wire.Function.Name != "" {
			result[wire.Function.Name] = true
		}
	}
	return result
}

func validateUsage(wire *usageWire) (*observedUsage, bool, bool, int, error) {
	prompt := wire.PromptTokens
	if prompt == nil {
		prompt = wire.InputTokens
	}
	completion := wire.CompletionTokens
	if completion == nil {
		completion = wire.OutputTokens
	}
	if prompt == nil || completion == nil || wire.TotalTokens == nil {
		return nil, false, false, 0, nil
	}
	if *prompt < 0 || *completion < 0 || *wire.TotalTokens < 0 {
		return nil, false, false, 0, errors.New("stream reported negative usage")
	}
	if *prompt+*completion != *wire.TotalTokens {
		return nil, false, false, 0, errors.New("stream reported inconsistent total token usage")
	}
	usage := &observedUsage{PromptTokens: *prompt, CompletionTokens: *completion, TotalTokens: *wire.TotalTokens}
	details := wire.PromptTokenDetails
	if details == nil {
		details = wire.InputTokenDetails
	}
	var cachedValue *int
	if details != nil {
		cachedValue = details.CachedTokens
	}
	for _, candidate := range []*int{wire.CachedTokens, wire.CacheReadTokens} {
		if candidate == nil {
			continue
		}
		if cachedValue != nil && *cachedValue != *candidate {
			return nil, false, false, 0, errors.New("stream reported inconsistent cached-token aliases")
		}
		cachedValue = candidate
	}
	if cachedValue == nil {
		return usage, true, false, 0, nil
	}
	cached := *cachedValue
	if cached < 0 || cached > *prompt {
		return nil, false, false, 0, errors.New("stream reported invalid cached prompt tokens")
	}
	return usage, true, true, cached, nil
}

func summarize(runID string, started time.Time, requestedModel string, events []lifecycleEvent, peaks map[string]int, tokenizer tokenizerKnowledge, parentErr, writeErr error) runSummary {
	result := runSummary{
		Schema:              "fak.agentbench.summary.v1",
		RunID:               runID,
		Mode:                "replay-plus-task",
		Profile:             "quick",
		Status:              "completed",
		OverallVerdict:      "SERVICE_ONLY_DIAGNOSTIC",
		ServiceVerdict:      "OBSERVED_OK",
		CacheVerdict:        "UNKNOWN",
		TaskVerdict:         "NOT_EVALUATED",
		RequestedModel:      requestedModel,
		ModelIdentityStatus: modelIdentityStatus(requestedModel, ""),
		PlannedRequests:     quickRequestCount,
		TerminalRequests:    len(events),
		ReferenceTokenizer:  tokenizer,
		StartedAt:           started.Format(time.RFC3339Nano),
		CompletedAt:         time.Now().UTC().Format(time.RFC3339Nano),
		Phase:               "replay",
		TaskFixtures:        1,
		MaxTaskTurns:        12,
		RequestCeiling:      28,
		Rungs: []rungSummary{
			{Name: "C1", ConcurrencyLimit: 1, ObservedPeakInFlight: peaks["C1"], PlannedRequests: 8},
			{Name: "C2", ConcurrencyLimit: 2, ObservedPeakInFlight: peaks["C2"], PlannedRequests: 8},
		},
	}
	models := make(map[string]struct{})
	for _, event := range events {
		rungIndex := 0
		if event.Rung == "C2" {
			rungIndex = 1
		}
		result.Rungs[rungIndex].TerminalRequests++
		if event.Status == "completed" {
			result.ScoredRequests++
			result.Rungs[rungIndex].CompletedRequests++
		} else {
			result.FailedRequests++
			result.Rungs[rungIndex].FailedRequests++
		}
		if event.UsageKnown != nil && *event.UsageKnown {
			result.Usage.KnownRequests++
			result.Usage.PromptTokens += event.Usage.PromptTokens
			result.Usage.CompletionTokens += event.Usage.CompletionTokens
			result.Usage.TotalTokens += event.Usage.TotalTokens
		} else {
			result.Usage.UnknownRequests++
		}
		if event.CacheKnown != nil && *event.CacheKnown {
			result.Cache.KnownRequests++
			if result.Cache.CachedPromptTokens == nil {
				result.Cache.CachedPromptTokens = intPointer(0)
			}
			if event.CachedPromptTokens != nil {
				*result.Cache.CachedPromptTokens += *event.CachedPromptTokens
			}
		} else {
			result.Cache.UnknownRequests++
		}
		if event.ObservedModel != "" {
			models[event.ObservedModel] = struct{}{}
		}
	}
	for model := range models {
		result.ObservedModels = append(result.ObservedModels, model)
	}
	sort.Strings(result.ObservedModels)
	if len(result.ObservedModels) != 0 {
		result.ModelIdentityStatus = "observed"
	}
	if result.Cache.KnownRequests == quickRequestCount {
		result.CacheVerdict = "OBSERVED_KNOWN"
	} else if result.Cache.KnownRequests != 0 {
		result.CacheVerdict = "PARTIALLY_KNOWN"
	}
	if result.FailedRequests != 0 {
		result.Status = "failed"
		result.ServiceVerdict = "OBSERVED_FAILURE"
		result.Error = fmt.Sprintf("%d requests did not complete", result.FailedRequests)
	}
	if parentErr != nil {
		result.Status = "canceled"
		result.ServiceVerdict = "CANCELED"
		result.Error = parentErr.Error()
	}
	if writeErr != nil {
		result.Status = "failed"
		result.ServiceVerdict = "ARTIFACT_FAILURE"
		result.Error = writeErr.Error()
	}
	return result
}

func describeInputs(inputs frozenInputs) map[string]inputDescriptor {
	return map[string]inputDescriptor{
		"system":  inputDescription("inputs/system.txt", inputs.system),
		"repo":    inputDescription("inputs/repo.txt", inputs.repo),
		"area_c1": inputDescription("inputs/area-c1.txt", inputs.area["C1"]),
		"area_c2": inputDescription("inputs/area-c2.txt", inputs.area["C2"]),
	}
}

func inputDescription(path, content string) inputDescriptor {
	digest := sha256.Sum256([]byte(content))
	return inputDescriptor{Path: path, SHA256: hex.EncodeToString(digest[:]), Bytes: len(content)}
}

func writeInputArtifacts(out string, inputs frozenInputs) ([]string, error) {
	dir := filepath.Join(out, "inputs")
	if err := os.Mkdir(dir, 0o755); err != nil {
		return nil, err
	}
	artifacts := []struct {
		name    string
		content string
	}{
		{name: "system.txt", content: inputs.system},
		{name: "repo.txt", content: inputs.repo},
		{name: "area-c1.txt", content: inputs.area["C1"]},
		{name: "area-c2.txt", content: inputs.area["C2"]},
	}
	paths := make([]string, 0, len(artifacts))
	for _, artifact := range artifacts {
		path := filepath.Join(dir, artifact.name)
		if err := writeExclusiveBytes(path, []byte(artifact.content)); err != nil {
			return nil, err
		}
		paths = append(paths, path)
	}
	return paths, nil
}

func resolveRepository(path string) (string, error) {
	abs, err := filepath.Abs(path)
	if err != nil {
		return "", err
	}
	real, err := filepath.EvalSymlinks(abs)
	if err != nil {
		return "", err
	}
	info, err := os.Stat(real)
	if err != nil {
		return "", err
	}
	if !info.IsDir() {
		return "", errors.New("path is not a directory")
	}
	return real, nil
}

func resolveOutput(repo, path string) (string, error) {
	abs, err := filepath.Abs(path)
	if err != nil {
		return "", err
	}
	resolved := abs
	if real, evalErr := filepath.EvalSymlinks(abs); evalErr == nil {
		resolved = real
	} else if parent, parentErr := filepath.EvalSymlinks(filepath.Dir(abs)); parentErr == nil {
		resolved = filepath.Join(parent, filepath.Base(abs))
	}
	rel, err := filepath.Rel(repo, resolved)
	if err != nil {
		return "", err
	}
	if rel == "." || (rel != ".." && !strings.HasPrefix(rel, ".."+string(filepath.Separator))) {
		return "", errors.New("must be outside the read-only source repository")
	}
	return resolved, nil
}

func normalizeEndpoint(raw string) (string, error) {
	parsed, err := url.Parse(raw)
	if err != nil {
		return "", err
	}
	if parsed.Scheme != "http" && parsed.Scheme != "https" {
		return "", errors.New("URL scheme must be http or https")
	}
	if parsed.Host == "" {
		return "", errors.New("URL must include a host")
	}
	if parsed.Path == "" || parsed.Path == "/" {
		parsed.Path = "/v1/chat/completions"
	}
	return parsed.String(), nil
}

func freezeRepositoryInput(repo string) (string, error) {
	gitRoot, rootErr := gitText(repo, "rev-parse", "--show-toplevel")
	if rootErr == nil {
		gitRoot, rootErr = filepath.EvalSymlinks(gitRoot)
	}
	if rootErr == nil && gitRoot != repo {
		return "", errors.New("repository path must be the Git worktree root")
	}
	if revision, err := gitText(repo, "rev-parse", "HEAD"); rootErr == nil {
		if err != nil || revision == "" {
			return "", errors.New("Git repository has no readable HEAD revision")
		}
		listed, err := gitText(repo, "ls-tree", "-r", "--name-only", revision)
		if err != nil {
			return "", fmt.Errorf("list committed source: %w", err)
		}
		paths := strings.Split(strings.TrimSpace(listed), "\n")
		sort.Strings(paths)
		return renderFrozenSource(paths, revision, "git HEAD public-source allowlist", func(rel string, limit int) ([]byte, error) {
			return gitBlob(repo, revision+":"+rel, limit)
		})
	}

	paths := make([]string, 0, 32)
	err := filepath.WalkDir(repo, func(path string, entry fs.DirEntry, walkErr error) error {
		if walkErr != nil {
			return walkErr
		}
		if entry.IsDir() {
			if path != repo && (strings.HasPrefix(entry.Name(), ".") || entry.Name() == "node_modules" || entry.Name() == "vendor") {
				return filepath.SkipDir
			}
			return nil
		}
		if entry.Type().IsRegular() && safeSourcePath(path) {
			rel, err := filepath.Rel(repo, path)
			if err != nil {
				return err
			}
			paths = append(paths, rel)
		}
		return nil
	})
	if err != nil {
		return "", err
	}
	sort.Strings(paths)
	return renderFrozenSource(paths, "unversioned", "unversioned fixture public-source allowlist", func(rel string, limit int) ([]byte, error) {
		file, err := os.Open(filepath.Join(repo, rel))
		if err != nil {
			return nil, err
		}
		defer file.Close()
		return io.ReadAll(io.LimitReader(file, int64(limit)))
	})
}

func safeSourcePath(path string) bool {
	base := filepath.Base(path)
	if strings.HasPrefix(base, ".") {
		return false
	}
	return strings.HasSuffix(base, ".go") || base == "go.mod" || base == "go.sum" || strings.EqualFold(base, "README.md")
}

func renderFrozenSource(paths []string, revision, selection string, read func(string, int) ([]byte, error)) (string, error) {
	var snapshot strings.Builder
	fmt.Fprintf(&snapshot, "Frozen read-only repository source follows.\nRevision: %s\nSelection: %s\n", revision, selection)
	remaining := maxSourceBytes
	included := 0
	truncated := false
	for _, rel := range paths {
		if rel == "" || !safeSourcePath(rel) {
			continue
		}
		if remaining <= 0 {
			truncated = true
			break
		}
		limit := maxSourceFileBytes
		if limit > remaining {
			limit = remaining
		}
		content, err := read(rel, limit)
		if err != nil {
			return "", err
		}
		if bytes.IndexByte(content, 0) >= 0 {
			continue
		}
		fmt.Fprintf(&snapshot, "\n--- %s ---\n", filepath.ToSlash(rel))
		snapshot.Write(content)
		remaining -= len(content)
		included++
		if len(content) == limit {
			truncated = true
		}
	}
	fmt.Fprintf(&snapshot, "\nSnapshot: included_files=%d byte_cap=%d truncated=%t\n", included, maxSourceBytes, truncated)
	return snapshot.String(), nil
}

func gitText(repo string, args ...string) (string, error) {
	cmd := exec.Command("git", append([]string{"-C", repo}, args...)...)
	out, err := cmd.Output()
	return strings.TrimSpace(string(out)), err
}

func gitBlob(repo, object string, limit int) ([]byte, error) {
	cmd := exec.Command("git", "-C", repo, "show", object)
	pipe, err := cmd.StdoutPipe()
	if err != nil {
		return nil, err
	}
	if err := cmd.Start(); err != nil {
		return nil, err
	}
	content, readErr := io.ReadAll(io.LimitReader(pipe, int64(limit)))
	_, drainErr := io.Copy(io.Discard, pipe)
	waitErr := cmd.Wait()
	if readErr != nil {
		return nil, readErr
	}
	if drainErr != nil {
		return nil, drainErr
	}
	if waitErr != nil {
		return nil, waitErr
	}
	return content, nil
}

func prepareOutput(out string) error {
	if err := os.MkdirAll(out, 0o755); err != nil {
		return fmt.Errorf("create output directory: %w", err)
	}
	for _, name := range []string{"manifest.json", "inputs", "events.jsonl", "summary.json", "receipt.json"} {
		_, err := os.Lstat(filepath.Join(out, name))
		if err == nil {
			return fmt.Errorf("artifact %s already exists", name)
		}
		if !errors.Is(err, os.ErrNotExist) {
			return err
		}
	}
	return nil
}

func writeExclusiveBytes(path string, content []byte) error {
	file, err := os.OpenFile(path, os.O_CREATE|os.O_EXCL|os.O_WRONLY, 0o644)
	if err != nil {
		return err
	}
	_, writeErr := file.Write(content)
	syncErr := file.Sync()
	closeErr := file.Close()
	if writeErr != nil {
		return writeErr
	}
	if syncErr != nil {
		return syncErr
	}
	return closeErr
}

func writeExclusiveJSON(path string, value any) error {
	file, err := os.OpenFile(path, os.O_CREATE|os.O_EXCL|os.O_WRONLY, 0o644)
	if err != nil {
		return err
	}
	encoder := json.NewEncoder(file)
	encoder.SetIndent("", "  ")
	encodeErr := encoder.Encode(value)
	syncErr := file.Sync()
	closeErr := file.Close()
	if encodeErr != nil {
		return encodeErr
	}
	if syncErr != nil {
		return syncErr
	}
	return closeErr
}

func writeAtomicJSON(path string, value any) error {
	dir := filepath.Dir(path)
	temp, err := os.CreateTemp(dir, "."+filepath.Base(path)+"-*")
	if err != nil {
		return err
	}
	tempPath := temp.Name()
	keep := false
	defer func() {
		if !keep {
			_ = os.Remove(tempPath)
		}
	}()
	encoder := json.NewEncoder(temp)
	encoder.SetIndent("", "  ")
	if err := encoder.Encode(value); err != nil {
		_ = temp.Close()
		return err
	}
	if err := temp.Sync(); err != nil {
		_ = temp.Close()
		return err
	}
	if err := temp.Close(); err != nil {
		return err
	}
	if err := os.Rename(tempPath, path); err != nil {
		return err
	}
	keep = true
	return nil
}

func writeReceiptAtomic(path, out, runID string, artifacts []string) error {
	hashes := make(map[string]string, len(artifacts))
	names := make([]string, 0, len(artifacts))
	for _, artifact := range artifacts {
		name, err := filepath.Rel(out, artifact)
		if err != nil {
			return err
		}
		name = filepath.ToSlash(name)
		digest, err := hashFile(artifact)
		if err != nil {
			return err
		}
		hashes[name] = digest
		names = append(names, name)
	}
	sort.Strings(names)
	runHasher := sha256.New()
	for _, name := range names {
		_, _ = io.WriteString(runHasher, name)
		_, _ = runHasher.Write([]byte{0})
		_, _ = io.WriteString(runHasher, hashes[name])
		_, _ = runHasher.Write([]byte{'\n'})
	}
	value := receipt{
		Schema:         "fak.agentbench.receipt.v1",
		RunID:          runID,
		Algorithm:      "sha256",
		ArtifactSHA256: hashes,
		RunSHA256:      hex.EncodeToString(runHasher.Sum(nil)),
	}
	return writeAtomicJSON(path, value)
}

func hashFile(path string) (string, error) {
	file, err := os.Open(path)
	if err != nil {
		return "", err
	}
	hasher := sha256.New()
	_, copyErr := io.Copy(hasher, file)
	closeErr := file.Close()
	if copyErr != nil {
		return "", copyErr
	}
	if closeErr != nil {
		return "", closeErr
	}
	return hex.EncodeToString(hasher.Sum(nil)), nil
}

func printValue(writer io.Writer, value any, asJSON bool) error {
	if asJSON {
		return json.NewEncoder(writer).Encode(value)
	}
	switch value := value.(type) {
	case replayPlan:
		_, err := fmt.Fprintf(writer, "agentbench quick plan: %d scored replay requests plus %d task (max %d turns); C1 concurrency=1, C2 concurrency=2; request ceiling=%d; dispatch=%t\n", value.ScoredRequests, value.TaskFixtures, value.MaxTaskTurns, value.RequestCeiling, value.Dispatch)
		return err
	case *runSummary:
		_, err := fmt.Fprintf(writer, "agentbench quick replay: status=%s terminal=%d/%d service=%s cache=%s task=%s overall=%s\n", value.Status, value.TerminalRequests, value.PlannedRequests, value.ServiceVerdict, value.CacheVerdict, value.TaskVerdict, value.OverallVerdict)
		return err
	default:
		return json.NewEncoder(writer).Encode(value)
	}
}

func boolPointer(value bool) *bool { return &value }

func intPointer(value int) *int { return &value }
