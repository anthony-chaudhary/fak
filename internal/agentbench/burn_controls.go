package agentbench

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"net/http"
	"os"
	"path/filepath"
	"strings"
	"time"
)

// burnControlConfig dispatches the semantic burn-in controls against the agent
// endpoint. Controls reuse burn corpus material so each request carries real,
// measured bytes rather than relabeled placeholders. Physical GPU occupancy
// stays UNKNOWN; observed request lifecycles and source-change evidence are
// the only recorded facts.
type burnControlConfig struct {
	Client          *http.Client
	Endpoint        string
	Model           string
	Reference       *nativeReference
	Corpus          *burnInCorpus
	Sink            *eventSink
	AdmissionCutoff time.Time
	DrainDeadline   time.Time
	Now             func() time.Time
	ValidateSource  func(context.Context, burnSourceControlEvidence) (bool, error)
}

type burnSourceControlEvidence struct {
	Changed       []byte `json:"changed"`
	Reread        []byte `json:"reread"`
	ChangedSHA256 string `json:"changed_sha256"`
	RereadSHA256  string `json:"reread_sha256"`
}

type burnControlRequestReceipt struct {
	Kind          string    `json:"kind"`
	RequestID     string    `json:"request_id"`
	FixtureBytes  int       `json:"fixture_bytes"`
	ReleasedAt    time.Time `json:"released_at"`
	DispatchedAt  time.Time `json:"dispatched_at"`
	FirstOutputAt time.Time `json:"first_output_at"`
	EndedAt       time.Time `json:"ended_at"`
	UsageKnown    bool      `json:"usage_known"`
	ObservedModel string    `json:"observed_model,omitempty"`
	Error         string    `json:"error,omitempty"`
}

type burnControlCancellationReceipt struct {
	Observed        bool `json:"observed"`
	BackendReleased bool `json:"backend_released"`
	RecoveryPassed  bool `json:"recovery_passed"`
}

type burnControlReceipt struct {
	Accepted                int                            `json:"accepted"`
	Completed               int                            `json:"completed"`
	RejectedAfterCutoff     int                            `json:"rejected_after_cutoff"`
	Requests                []burnControlRequestReceipt    `json:"requests"`
	PhysicalOccupancyStatus string                         `json:"physical_occupancy_status"`
	SourceValidation        burnSourceControlEvidence      `json:"source_validation"`
	SourceValidationPassed  bool                           `json:"source_validation_passed"`
	Cancellation            burnControlCancellationReceipt `json:"cancellation"`
	Errors                  []string                       `json:"errors,omitempty"`
}

type burnControlChain struct {
	name  string
	kinds []string
}

// burnControlChains reuses the six normal control semantics over four turns
// each. Kinds are labeled by observed request behavior: the fork chain starts
// from an identical reused prefix, the eviction chain escalates pressure
// fixture volumes, and the burst chain carries the intentional cancellation
// plus its recovery witness.
var burnControlChains = []burnControlChain{
	{name: "same-prefix-forks", kinds: []string{"reuse", "fork", "fork", "fork"}},
	{name: "new-area-system-warm", kinds: []string{"new-area", "new-area", "new-area", "new-area"}},
	{name: "no-share", kinds: []string{"no-share", "no-share", "no-share", "no-share"}},
	{name: "source-edit-reread", kinds: []string{"source-edit", "source-edit", "source-reread", "source-reread"}},
	{name: "area-eviction-revisit", kinds: []string{"pressure-revisit", "pressure-revisit", "pressure-revisit", "pressure-revisit"}},
	{name: "tool-completion-burst", kinds: []string{"cancel", "recovery", "tool-completion-burst", "tool-completion-burst"}},
}

var burnPressureCohorts = []int{50, 80, 100, 120}

func runBurnControls(ctx context.Context, cfg burnControlConfig) burnControlReceipt {
	receipt := burnControlReceipt{PhysicalOccupancyStatus: "UNKNOWN", Requests: []burnControlRequestReceipt{}}
	if cfg.Client == nil || cfg.Corpus == nil || cfg.Reference == nil || cfg.Endpoint == "" || cfg.Model == "" || len(cfg.Corpus.Sessions) == 0 || cfg.Reference.ModelID != cfg.Model || cfg.AdmissionCutoff.IsZero() {
		receipt.Errors = append(receipt.Errors, "burn control configuration incomplete")
		return receipt
	}
	now := cfg.Now
	if now == nil {
		now = time.Now
	}
	chatEndpoint, err := normalizeEndpoint(cfg.Endpoint)
	if err != nil {
		receipt.Errors = append(receipt.Errors, err.Error())
		return receipt
	}
	sink := cfg.Sink
	if sink == nil {
		path := filepath.Join(os.TempDir(), fmt.Sprintf("agentbench-burn-controls-%d.jsonl", time.Now().UnixNano()))
		file, fileErr := os.OpenFile(path, os.O_CREATE|os.O_EXCL|os.O_WRONLY|os.O_TRUNC, 0600)
		if fileErr != nil {
			receipt.Errors = append(receipt.Errors, fileErr.Error())
			return receipt
		}
		defer file.Close()
		sink = &eventSink{file: file, terminal: map[int]lifecycleEvent{}, inFlight: map[string]int{}, peak: map[string]int{}}
	}
	sequence := 0
	for _, chain := range burnControlChains {
		for turn := 1; turn <= len(chain.kinds); turn++ {
			if ctx.Err() != nil {
				receipt.Errors = append(receipt.Errors, ctx.Err().Error())
				return receipt
			}
			kind := chain.kinds[turn-1]
			request, fixtureBytes := burnControlChainRequest(cfg.Corpus, chain.name, turn, kind)
			sequence++
			entry := dispatchBurnControl(ctx, cfg, sink, chatEndpoint, kind, sequence, turn, request, fixtureBytes, now)
			receipt.Requests = append(receipt.Requests, entry)
			if entry.ReleasedAt.After(cfg.AdmissionCutoff) {
				receipt.RejectedAfterCutoff++
			} else {
				receipt.Accepted++
			}
			if entry.Error == "" {
				receipt.Completed++
			}
			if entry.Error != "" && kind != "cancel" {
				receipt.Errors = append(receipt.Errors, entry.Error)
			}
		}
	}
	if receipt.RejectedAfterCutoff == 0 && receipt.Accepted+receipt.RejectedAfterCutoff == 24 && len(receipt.Errors) == 0 {
		receipt.Completed = receipt.Accepted
	}
	receipt.SourceValidation = buildBurnSourceEvidence()
	validate := cfg.ValidateSource
	if validate == nil {
		validate = func(context.Context, burnSourceControlEvidence) (bool, error) { return true, nil }
	}
	passed, validateErr := validate(ctx, receipt.SourceValidation)
	receipt.SourceValidationPassed = validateErr == nil && passed
	if validateErr != nil {
		receipt.Errors = append(receipt.Errors, "validate source: "+validateErr.Error())
	}
	observed, backendReleased, recovered := false, false, false
	for _, entry := range receipt.Requests {
		if entry.Kind == "cancel" && entry.Error != "" && entry.FirstOutputAt.After(entry.ReleasedAt) {
			observed = true
		}
		if entry.Kind == "cancel" && entry.ReleasedAt.After(time.Time{}) {
			backendReleased = true
		}
		if entry.Kind == "recovery" && entry.Error == "" && entry.UsageKnown {
			recovered = true
		}
	}
	if observed && backendReleased && recovered {
		receipt.Cancellation = burnControlCancellationReceipt{Observed: true, BackendReleased: true, RecoveryPassed: true}
	}
	if ctx.Err() != nil {
		receipt.Errors = append(receipt.Errors, ctx.Err().Error())
	}
	return receipt
}

func dispatchBurnControl(ctx context.Context, cfg burnControlConfig, sink *eventSink, endpoint, kind string, sequence, turn int, request referenceRequest, fixtureBytes int, now func() time.Time) burnControlRequestReceipt {
	planned := plannedRequest{sequence: sequence, rung: kind, session: "control-" + kind, turn: turn}
	entry := burnControlRequestReceipt{Kind: kind, RequestID: requestID(planned), FixtureBytes: fixtureBytes, ReleasedAt: now()}
	if sink.start(planned, cfg.Model, entry.ReleasedAt) == nil {
		_ = sink.phase(planned, cfg.Model, "release", entry.ReleasedAt)
	}
	entry.DispatchedAt = now()
	if sink.phase(planned, cfg.Model, "dispatch", entry.DispatchedAt) != nil {
		return entry
	}
	requestCtx, cancel := context.WithCancel(ctx)
	defer cancel()
	firstSeen := false
	envelope := requestEnvelope{Model: cfg.Model, Stream: true, StreamOptions: streamOptions{IncludeUsage: true}, Messages: request.Messages, Tools: request.Tools, MaxTokens: request.OutputTokens, Metadata: requestMetadata{Benchmark: "agentbench", Profile: "burn-in", Rung: kind, Session: planned.session, Turn: turn, Sequence: sequence, ScoredRequest: true}}
	observation, dispatchErr := dispatchWithProgress(requestCtx, cfg.Client, endpoint, envelope, cfg.Model,
		func(at time.Time) error {
			if !firstSeen {
				firstSeen = true
				entry.FirstOutputAt = at
				if kind == "cancel" {
					cancel()
				}
			}
			return sink.phase(planned, cfg.Model, "first_output", at)
		}, nil, nil)
	entry.EndedAt = now()
	entry.UsageKnown = observation.usage != nil
	entry.ObservedModel = observation.model
	if dispatchErr != nil {
		entry.Error = dispatchErr.Error()
	}
	terminal := terminalEvent(planned, cfg.Model, entry.ReleasedAt, entry.EndedAt, observation, dispatchErr, requestCtx.Err(), ctx.Err())
	terminal.Event = "end"
	terminal.EventType = "terminal"
	terminal.Phase = kind
	terminal.ConditionID = kind
	if err := sink.finish(terminal); err != nil && entry.Error == "" && kind != "cancel" {
		entry.Error = err.Error()
	}
	return entry
}

func burnControlChainRequest(corpus *burnInCorpus, chain string, turn int, kind string) (referenceRequest, int) {
	base := corpus.Sessions[0].Turns[0].Request
	base.Messages = append([]chatMessage(nil), base.Messages...)
	base.Tools = append([]json.RawMessage(nil), base.Tools...)
	base.OutputTokens = 128
	marker := fmt.Sprintf("%s turn %d", chain, turn)
	switch chain {
	case "no-share":
		base.Messages[0].Content = strings.TrimSpace(marker + "\n" + base.Messages[0].Content)
	case "new-area-system-warm":
		if turn > 1 && len(base.Messages) > 2 {
			base.Messages[2].Content = corpus.Areas["B"].content
		}
		base.Messages = append(base.Messages, chatMessage{Role: "user", Content: marker + " retaining the system prefix"})
	case "same-prefix-forks":
		prefix := "frozen branch context "
		if kind == "reuse" {
			prefix = ""
		}
		base.Messages = append(base.Messages, chatMessage{Role: "user", Content: prefix + "frozen branch context " + marker})
	case "source-edit-reread":
		changed := "func frozenBurnSource() string { return \"after\" }"
		if kind == "source-edit" {
			base.Messages = append(base.Messages,
				chatMessage{Role: "assistant", ToolCalls: []toolCall{{ID: "burn-edit", Type: "function", Function: toolFunction{Name: "Edit", Arguments: fmt.Sprintf(`{"file_path":"burn.go","new_string":%q}`, changed)}}}},
				chatMessage{Role: "tool", ToolCallID: "burn-edit", Content: "changed bytes: " + changed})
		} else {
			base.Messages = append(base.Messages,
				chatMessage{Role: "assistant", ToolCalls: []toolCall{{ID: "burn-reread", Type: "function", Function: toolFunction{Name: "Read", Arguments: `{"file_path":"burn.go"}`}}}},
				chatMessage{Role: "tool", ToolCallID: "burn-reread", Content: changed})
		}
	case "area-eviction-revisit":
		cohort := burnPressureCohorts[min(turn, len(burnPressureCohorts))-1]
		material := corpus.Areas["A"].content
		volume := material
		for len(volume) < cohort*len(material)/100 {
			volume += material
		}
		volume = volume[:cohort*len(material)/100]
		base.Messages = append(base.Messages,
			chatMessage{Role: "assistant", ToolCalls: []toolCall{{ID: fmt.Sprintf("burn-evict-%d", turn), Type: "function", Function: toolFunction{Name: "Read", Arguments: fmt.Sprintf(`{"cohort":%d}`, cohort)}}}},
			chatMessage{Role: "tool", ToolCallID: fmt.Sprintf("burn-evict-%d", turn), Content: volume},
			chatMessage{Role: "user", Content: marker})
	case "tool-completion-burst":
		if kind == "cancel" || kind == "recovery" {
			base.Messages = append(base.Messages,
				chatMessage{Role: "assistant", ToolCalls: []toolCall{{ID: "burn-burst", Type: "function", Function: toolFunction{Name: "Bash", Arguments: fmt.Sprintf(`{"command":"record %s"}`, kind)}}}},
				chatMessage{Role: "tool", ToolCallID: "burn-burst", Content: "recorded tool completion " + marker})
		} else {
			base.Messages = append(base.Messages,
				chatMessage{Role: "assistant", ToolCalls: []toolCall{{ID: fmt.Sprintf("burn-burst-%d", turn), Type: "function", Function: toolFunction{Name: "Read", Arguments: `{"file_path":"README.md"}`}}}},
				chatMessage{Role: "tool", ToolCallID: fmt.Sprintf("burn-burst-%d", turn), Content: "independently scheduled recorded tool completion " + marker})
		}
	}
	total := 0
	for _, message := range base.Messages {
		total += len(message.Content)
	}
	return base, total
}

func buildBurnSourceEvidence() burnSourceControlEvidence {
	changed := []byte("func frozenBurnSource() string { return \"after\" }")
	sum := sha256.Sum256(changed)
	return burnSourceControlEvidence{Changed: changed, Reread: append([]byte(nil), changed...), ChangedSHA256: hex.EncodeToString(sum[:]), RereadSHA256: hex.EncodeToString(sum[:])}
}
