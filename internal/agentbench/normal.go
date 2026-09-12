package agentbench

import (
	"context"
	"crypto/rand"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"time"

	"github.com/anthony-chaudhary/fak/internal/agentbench/profileplan"
)

type normalCell struct {
	Name, Phase                                          string
	Concurrency, Sessions, TurnsPerSession, OutputTokens int
}

type normalReplayResult struct {
	events                    []lifecycleEvent
	peaks                     map[string]int
	selectedConcurrency       int
	highestPassingConcurrency int
	qualification             normalQualification
}

type normalCapacitySelection struct {
	Qualified      bool
	Selected       int
	HighestPassing int
	Reason         string
}

var normalControls = []string{
	"same-prefix-forks", "new-area-system-warm", "no-share",
	"source-edit-reread", "area-eviction-revisit", "tool-completion-burst",
}

func compileNormalSchedule(plan profileplan.Plan) ([]normalCell, error) {
	if plan.Profile != "normal" || plan.ProbeTurns != 64 || plan.SteadyTurns != 96 || plan.ControlTurns != 24 {
		return nil, errors.New("agentbench normal: invalid immutable plan geometry")
	}
	var cells []normalCell
	for _, c := range []int{1, 2, 4, 8} {
		cells = append(cells, normalCell{Name: fmt.Sprintf("probe-c%d", c), Phase: "probe", Concurrency: c, Sessions: 8, TurnsPerSession: 2, OutputTokens: 128})
	}
	cells = append(cells, normalCell{Name: "steady", Phase: "steady", Concurrency: plan.PreferredConcurrency, Sessions: 8, TurnsPerSession: 12})
	for _, name := range normalControls {
		sessions, turns := 1, 4
		if name == "no-share" {
			sessions, turns = 4, 1
		}
		if name == "tool-completion-burst" {
			sessions, turns = plan.PreferredConcurrency, 4/plan.PreferredConcurrency
		}
		cells = append(cells, normalCell{Name: name, Phase: "controls", Concurrency: plan.PreferredConcurrency, Sessions: sessions, TurnsPerSession: turns, OutputTokens: 256})
	}
	return cells, nil
}

func runNormalReplay(ctx context.Context, progress io.Writer, endpoint, model, out string, inputs frozenInputs) (*normalReplayResult, error) {
	chatEndpoint, err := normalizeEndpoint(endpoint)
	if err != nil {
		return nil, err
	}
	plan, err := profileplan.Build(profileplan.Options{Profile: "normal", PreferredConcurrency: 2})
	if err != nil {
		return nil, err
	}
	cells, err := compileNormalSchedule(plan)
	if err != nil {
		return nil, err
	}
	if err := os.MkdirAll(out, 0700); err != nil {
		return nil, err
	}
	f, err := os.OpenFile(filepath.Join(out, "normal-events.jsonl"), os.O_CREATE|os.O_EXCL|os.O_WRONLY|os.O_APPEND, 0600)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	transport := http.DefaultTransport.(*http.Transport).Clone()
	transport.MaxConnsPerHost = 8
	transport.MaxIdleConnsPerHost = 8
	defer transport.CloseIdleConnections()
	client := &http.Client{Transport: transport, Timeout: requestTimeout, CheckRedirect: func(*http.Request, []*http.Request) error { return http.ErrUseLastResponse }}
	reference, err := discoverNativeReference(ctx, client, endpoint, model)
	if err != nil {
		return nil, err
	}
	if reference.ContextWindowTokens < plan.MinimumModelContextTokens {
		return nil, fmt.Errorf("normal context %d below %d", reference.ContextWindowTokens, plan.MinimumModelContextTokens)
	}
	if inputs.sourceRoot == "" {
		return nil, errors.New("normal replay requires the frozen corpus source root")
	}
	corpus, err := buildNormalCorpus(ctx, reference, inputs.sourceRoot)
	if err != nil {
		return nil, fmt.Errorf("build exact normal corpus: %w", err)
	}
	if err := isolateNormalCampaign(ctx, reference, corpus); err != nil {
		return nil, fmt.Errorf("freeze normal campaign namespace: %w", err)
	}
	if err := writeExclusiveJSON(filepath.Join(out, "normal-context-manifest.json"), corpus); err != nil {
		return nil, fmt.Errorf("persist exact normal context before dispatch: %w", err)
	}
	keys := make([]string, 0, len(cells))
	for _, c := range cells {
		keys = append(keys, c.Name)
	}
	inflight, peaks := map[string]int{}, map[string]int{}
	for _, k := range keys {
		inflight[k] = 0
		peaks[k] = 0
	}
	for _, c := range []int{1, 2, 4, 8} {
		key := fmt.Sprintf("C%d", c)
		inflight[key] = 0
		peaks[key] = 0
	}
	sink := &eventSink{file: f, terminal: make(map[int]lifecycleEvent, 188), inFlight: inflight, peak: peaks}
	runCtx, cancel := context.WithTimeout(ctx, 30*time.Minute)
	defer cancel()
	if err := runNormalPreconditions(runCtx, client, chatEndpoint, model, reference, corpus, sink); err != nil {
		return &normalReplayResult{events: sink.terminals(), peaks: sink.peaks()}, err
	}
	sequence := 0
	selectedRuntime := 0
	for i := range cells {
		cell := cells[i]
		if runCtx.Err() != nil {
			break
		}
		if cell.Phase != "probe" && selectedRuntime == 0 {
			selection := selectNormalCapacity(plan, sink.terminals())
			if !selection.Qualified {
				return &normalReplayResult{events: sink.terminals(), peaks: sink.peaks()}, errors.New(selection.Reason)
			}
			selectedRuntime = selection.Selected
		}
		if cell.Phase == "steady" || cell.Phase == "controls" {
			cell.Concurrency = selectedRuntime
			if cell.Name == "tool-completion-burst" {
				cell.Sessions = min(selectedRuntime, 4)
				cell.TurnsPerSession = 4 / cell.Sessions
			}
			cells[i] = cell
		}
		fmt.Fprintf(progress, "agentbench normal phase=%s cell=%s\n", cell.Phase, cell.Name)
		requests := make([]plannedRequest, 0, cell.Sessions*cell.TurnsPerSession)
		for s := 1; s <= cell.Sessions; s++ {
			for turn := 1; turn <= cell.TurnsPerSession; turn++ {
				sequence++
				requests = append(requests, plannedRequest{sequence: sequence, rung: cell.Name, session: fmt.Sprintf("%s-S%d", cell.Name, s), turn: turn})
			}
		}
		runNormalCell(runCtx, client, chatEndpoint, model, corpus, reference, sink, cell, requests)
		if hardNormalFailure(sink.terminals(), cell.Name) {
			break
		}
	}
	terminals := sink.terminals()
	all := sink.events()
	scored := 0
	for _, event := range terminals {
		if event.ScoredRequest {
			scored++
		}
	}
	if scored != 184 {
		return &normalReplayResult{events: all, peaks: sink.peaks()}, fmt.Errorf("normal replay incomplete: %d/184 scored terminal requests", scored)
	}
	selection := selectNormalCapacity(plan, terminals)
	result := &normalReplayResult{events: all, peaks: sink.peaks(), selectedConcurrency: selection.Selected, highestPassingConcurrency: selection.HighestPassing}
	if sink.err() != nil {
		return result, sink.err()
	}
	if !selection.Qualified {
		return result, errors.New(selection.Reason)
	}
	if err := validateNormalCells(cells, terminals, result.peaks); err != nil {
		return result, err
	}
	result.qualification = summarizeNormalQualification(plan, cells, all, selection.Selected)
	if !result.qualification.Qualified {
		return result, fmt.Errorf("%s: %s", result.qualification.Reason, strings.Join(result.qualification.Unknowns, "; "))
	}
	return result, nil
}

// isolateNormalCampaign gives every execution a fresh early token namespace while
// keeping one namespace shared by every session in a rung. The exact namespace and
// its reference encoding are persisted in the context manifest before dispatch.
func isolateNormalCampaign(ctx context.Context, encoder referenceEncoder, corpus *normalCorpus) error {
	var raw [16]byte
	if _, err := rand.Read(raw[:]); err != nil {
		return err
	}
	campaign := fmt.Sprintf("normal-campaign-%x", raw[:])
	commonOffset := -1
	commonTokens := 0
	for _, concurrency := range []int{1, 2, 4, 8} {
		prior, ok := corpus.Preconditions[concurrency]
		if !ok || len(prior.Request.Messages) == 0 {
			return fmt.Errorf("missing C%d precondition", concurrency)
		}
		nonce := fmt.Sprintf("Campaign namespace: %s\nEquivalent rung namespace: normal-rung-%02d\n", campaign, concurrency)
		request := prior.Request
		request.Messages = append([]chatMessage(nil), prior.Request.Messages...)
		request.Messages[0].Content = nonce + request.Messages[0].Content
		encoded, err := encoder.Encode(ctx, request)
		if err != nil {
			return err
		}
		if commonTokens == 0 {
			commonTokens = encoded.PromptTokens
		} else if encoded.PromptTokens != commonTokens {
			return errors.New("campaign rung namespaces do not preserve reference token count")
		}
		offset := commonTokenPrefix(prior.Encoding.TokenIDs, encoded.TokenIDs)
		if commonOffset < 0 {
			commonOffset = offset
		} else if offset != commonOffset {
			return errors.New("campaign rung namespaces do not occupy the same verified token position")
		}
		corpus.Preconditions[concurrency] = normalPrecondition{
			Concurrency: concurrency, PromptTokens: encoded.PromptTokens,
			NonceTokenOffset: offset, Nonce: nonce, EncodingSHA256: encoded.RenderedSHA256,
			Request: request, Encoding: encoded,
		}
	}
	return nil
}

func runNormalPreconditions(ctx context.Context, client *http.Client, endpoint, model string, reference *nativeReference, corpus *normalCorpus, sink *eventSink) error {
	for index, c := range []int{1, 2, 4, 8} {
		pre := corpus.Preconditions[c]
		req := plannedRequest{sequence: 185 + index, rung: fmt.Sprintf("C%d", c), session: fmt.Sprintf("precondition-C%d", c), turn: 1}
		started := time.Now().UTC()
		if err := sink.start(req, model, started); err != nil {
			return err
		}
		if _, err := reference.Encode(ctx, pre.Request); err != nil {
			return err
		}
		observation, dispatchErr := dispatchWithProgress(ctx, client, endpoint, requestEnvelope{Model: model, Stream: true, StreamOptions: streamOptions{IncludeUsage: true}, Messages: pre.Request.Messages, Tools: pre.Request.Tools, MaxTokens: pre.Request.OutputTokens, Metadata: requestMetadata{Benchmark: "agentbench", Profile: "normal-precondition", Rung: req.rung, Session: req.session, Turn: 1, Sequence: req.sequence, ScoredRequest: false}}, model, nil, nil, nil)
		terminal := terminalEvent(req, model, started, time.Now().UTC(), observation, dispatchErr, ctx.Err(), ctx.Err())
		terminal.Event = "end"
		terminal.EventType = "terminal"
		terminal.Phase = "precondition"
		terminal.ConditionID = "unique-early-prefix-equivalence"
		terminal.ScoredRequest = false
		terminal.CacheVerdict = "UNKNOWN"
		if err := sink.finish(terminal); err != nil {
			return err
		}
		if dispatchErr != nil {
			return dispatchErr
		}
	}
	return nil
}

func runNormalCell(ctx context.Context, client *http.Client, endpoint, model string, corpus *normalCorpus, reference *nativeReference, sink *eventSink, cell normalCell, requests []plannedRequest) {
	sem := make(chan struct{}, cell.Concurrency)
	var wg sync.WaitGroup
	prepared := make(map[int]referenceRequest, len(requests))
	for i, request := range requests {
		session := i/cell.TurnsPerSession + 1
		wire := normalCorpusRequest(corpus, cell, request, session)
		if err := validateControlRequest(cell.Name, wire); err != nil {
			return
		}
		if _, err := reference.Encode(ctx, wire); err != nil {
			return
		}
		prepared[request.sequence] = wire
	}
	if cell.Name == "tool-completion-burst" {
		var ready sync.WaitGroup
		ready.Add(len(requests))
		release := make(chan struct{})
		for i, req := range requests {
			wg.Add(1)
			go func(req plannedRequest, session int) {
				defer wg.Done()
				started := prepareNormalRequest(sink, req, model)
				ready.Done()
				select {
				case <-release:
				case <-ctx.Done():
					return
				}
				select {
				case sem <- struct{}{}:
				case <-ctx.Done():
					return
				}
				runNormalRequest(ctx, client, endpoint, model, sink, cell, req, prepared[req.sequence], started)
				<-sem
			}(req, i%cell.Sessions+1)
		}
		ready.Wait()
		close(release)
		wg.Wait()
		return
	}
	for session := 1; session <= cell.Sessions; session++ {
		session := session
		wg.Add(1)
		go func() {
			defer wg.Done()
			for turn := 1; turn <= cell.TurnsPerSession; turn++ {
				idx := (session-1)*cell.TurnsPerSession + turn - 1
				req := requests[idx]
				started := prepareNormalRequest(sink, req, model)
				select {
				case sem <- struct{}{}:
				case <-ctx.Done():
					return
				}
				runNormalRequest(ctx, client, endpoint, model, sink, cell, req, prepared[req.sequence], started)
				<-sem
				if cell.Phase == "steady" {
					if wait := normalToolWait(turn); wait > 0 {
						select {
						case <-time.After(wait):
						case <-ctx.Done():
							return
						}
					}
				}
			}
		}()
	}
	wg.Wait()
}

func normalToolWait(turn int) time.Duration {
	switch turn {
	case 2:
		return 200 * time.Millisecond
	case 5:
		return time.Second
	case 8:
		return 5 * time.Second
	case 11:
		return 15 * time.Second
	}
	return 0
}

func prepareNormalRequest(sink *eventSink, request plannedRequest, model string) time.Time {
	started := time.Now().UTC()
	if normalStartQueued(sink, request, model, started) != nil {
		return started
	}
	_ = sink.phase(request, model, "release", time.Now().UTC())
	_ = sink.phase(request, model, "enqueue", time.Now().UTC())
	return started
}

func runNormalRequest(ctx context.Context, client *http.Client, endpoint, model string, sink *eventSink, cell normalCell, request plannedRequest, wire referenceRequest, started time.Time) {
	normalMarkDispatched(sink, request.rung)
	if sink.phase(request, model, "dispatch", time.Now().UTC()) != nil {
		return
	}
	messages, tools, cap := wire.Messages, wire.Tools, wire.OutputTokens
	var observation streamObservation
	var dispatchErr error
	observation, dispatchErr = dispatchWithProgress(ctx, client, endpoint, requestEnvelope{Model: model, Stream: true, StreamOptions: streamOptions{IncludeUsage: true}, Messages: messages, Tools: tools, MaxTokens: cap, Metadata: requestMetadata{Benchmark: "agentbench", Profile: "normal", Rung: cell.Name, Session: request.session, Turn: request.turn, Sequence: request.sequence, ScoredRequest: true}}, model,
		func(at time.Time) error { return sink.phase(request, model, "first_output", at) },
		func(at time.Time) error { return sink.phase(request, model, "tool_call", at) },
		func(at time.Time) error { return sink.phase(request, model, "stream_progress", at) })
	completed := time.Now().UTC()
	terminal := terminalEvent(request, model, started, completed, observation, dispatchErr, ctx.Err(), ctx.Err())
	terminal.Event = "end"
	terminal.EventType = "terminal"
	terminal.Phase = cell.Phase
	terminal.ConditionID = cell.Name
	_ = sink.finish(terminal)
}

func validateControlRequest(name string, request referenceRequest) error {
	if name != "source-edit-reread" {
		return nil
	}
	edit, reread, bound := false, false, false
	for _, message := range request.Messages {
		for _, call := range message.ToolCalls {
			if call.Function.Name == "Edit" {
				edit = true
			}
			if call.Function.Name == "Read" {
				reread = true
			}
		}
		if message.Role == "tool" && message.ToolCallID != "" {
			bound = true
		}
	}
	if !edit || !reread || !bound {
		return errors.New("source-edit-reread control lacks bound edit decision and reread")
	}
	return nil
}

func validateNormalCells(cells []normalCell, events []lifecycleEvent, peaks map[string]int) error {
	for _, cell := range cells {
		count := 0
		for _, event := range events {
			if event.Rung == cell.Name && (event.Event == "end" || event.EventType == "terminal") {
				count++
				if event.Status != "completed" || event.ModelIdentityStatus != "observed" || event.UsageKnown == nil || !*event.UsageKnown {
					return fmt.Errorf("normal cell %s has an unqualified terminal", cell.Name)
				}
			}
		}
		if count != cell.Sessions*cell.TurnsPerSession {
			return fmt.Errorf("normal cell %s terminals %d/%d", cell.Name, count, cell.Sessions*cell.TurnsPerSession)
		}
		if cell.Name == "tool-completion-burst" && peaks[cell.Name] < cell.Concurrency {
			return errors.New("tool-completion-burst did not witness request overlap")
		}
	}
	return nil
}

func normalStartQueued(sink *eventSink, request plannedRequest, model string, started time.Time) error {
	sink.mu.Lock()
	defer sink.mu.Unlock()
	return sink.appendLocked(lifecycleEvent{Schema: "fak.agentbench.event.v1", Event: "request_started", EventType: "start", Phase: "start", Sequence: request.sequence, RequestID: requestID(request), Rung: request.rung, ConditionID: request.rung, Session: request.session, SessionID: request.session, Turn: request.turn, ScoredRequest: true, Status: "queued", TaskVerdict: "NOT_EVALUATED", RequestedModel: model, ModelIdentityStatus: modelIdentityStatus(model, ""), StartedAt: started.Format(time.RFC3339Nano)})
}

func normalMarkDispatched(sink *eventSink, rung string) {
	sink.mu.Lock()
	defer sink.mu.Unlock()
	sink.inFlight[rung]++
	if sink.inFlight[rung] > sink.peak[rung] {
		sink.peak[rung] = sink.inFlight[rung]
	}
}

func normalCorpusRequest(corpus *normalCorpus, cell normalCell, request plannedRequest, session int) referenceRequest {
	var base referenceRequest
	if cell.Phase == "probe" {
		index := min(max(session, 1), len(corpus.Sessions)) - 1
		base = corpus.Sessions[index].Turns[min(request.turn, 2)-1].Request
		base.Messages = append([]chatMessage(nil), base.Messages...)
		if len(base.Messages) > 0 {
			base.Messages[0].Content = corpus.Preconditions[cell.Concurrency].Nonce + base.Messages[0].Content
		}
	} else if cell.Phase == "controls" {
		base = referenceRequest{Messages: []chatMessage{{Role: "system", Content: corpus.SystemTools.content}, {Role: "system", Content: corpus.RepositoryMap.content}, {Role: "system", Content: corpus.Areas["A"].content}, {Role: "user", Content: fmt.Sprintf("%s turn %d", cell.Name, request.turn)}}, Tools: normalToolContracts(), OutputTokens: 256}
		if cell.Name == "same-prefix-forks" {
			base.Messages[3].Content = "frozen common branch point"
		}
	} else {
		index := min(max(session, 1), len(corpus.Sessions)) - 1
		base = corpus.Sessions[index].Turns[min(request.turn, len(corpus.Sessions[index].Turns))-1].Request
	}
	base.Messages = append([]chatMessage(nil), base.Messages...)
	base.Tools = append([]json.RawMessage(nil), base.Tools...)
	return applyNormalControl(corpus, cell.Name, request, session, base)
}

func normalMessages(inputs frozenInputs, request plannedRequest, cell normalCell) []chatMessage {
	area := inputs.area["A"]
	if strings.HasSuffix(request.session, "S7") || strings.HasSuffix(request.session, "S8") {
		area = inputs.area["B"]
	}
	messages := []chatMessage{{Role: "system", Content: inputs.system}, {Role: "system", Content: inputs.repo}, {Role: "system", Content: area}}
	if cell.Name == "no-share" {
		messages[0].Content += " isolated nonce " + request.session
	}
	if cell.Name == "source-edit-reread" && request.turn >= 3 {
		messages = append(messages,
			chatMessage{Role: "assistant", ToolCalls: []toolCall{{ID: "edit-1", Type: "function", Function: toolFunction{Name: "Edit", Arguments: `{"file_path":"target.go","old_string":"before","new_string":"after"}`}}}},
			chatMessage{Role: "tool", ToolCallID: "edit-1", Content: "recorded edit applied"},
			chatMessage{Role: "assistant", ToolCalls: []toolCall{{ID: "read-1", Type: "function", Function: toolFunction{Name: "Read", Arguments: `{"file_path":"target.go"}`}}}},
			chatMessage{Role: "tool", ToolCallID: "read-1", Content: "recorded reread after edit"})
	}
	return append(messages, chatMessage{Role: "user", Content: fmt.Sprintf("%s turn %d", cell.Name, request.turn)})
}

func hardNormalFailure(events []lifecycleEvent, cell string) bool {
	for _, e := range events {
		if e.Rung == cell && e.Status != "completed" && (e.Status == "canceled" || e.Status == "timed_out") {
			return true
		}
	}
	return false
}

func selectNormalCapacity(plan profileplan.Plan, events []lifecycleEvent) normalCapacitySelection {
	result := normalCapacitySelection{Reason: "probe cells are incomplete or failed"}
	if err := validateProbePreconditions(events); err != nil {
		result.Reason = err.Error()
		return result
	}
	throughput := map[int]float64{}
	for _, c := range plan.CapacitySelection.ProbeConcurrency {
		name := fmt.Sprintf("C%d", c)
		var count int
		var total float64
		var first, last time.Time
		passing := true
		for _, e := range events {
			if e.Phase == "probe" && (e.Rung == name || e.Rung == fmt.Sprintf("probe-c%d", c)) && ((e.Event == "end" || e.Event == "request_terminal") || (e.EventType == "terminal" || e.EventType == "end")) {
				count++
				if e.Status != "completed" || e.ModelIdentityStatus != "matched" && e.ModelIdentityStatus != "observed" || e.UsageKnown == nil || !*e.UsageKnown {
					passing = false
					continue
				}
				total += float64(max(1, e.DurationMilliseconds))
				started, serr := time.Parse(time.RFC3339Nano, e.StartedAt)
				completed, cerr := time.Parse(time.RFC3339Nano, e.CompletedAt)
				if serr == nil && cerr == nil {
					if first.IsZero() || started.Before(first) {
						first = started
					}
					if last.IsZero() || completed.After(last) {
						last = completed
					}
				}
			}
		}
		if count != 16 {
			result.Reason = fmt.Sprintf("probe C%d has %d/16 terminals", c, count)
			return result
		}
		if !passing {
			continue
		}
		if !first.IsZero() && last.After(first) {
			throughput[c] = float64(count) / last.Sub(first).Seconds()
		} else {
			throughput[c] = 1000 / (total / float64(count))
		}
		result.HighestPassing = c
	}
	if len(throughput) == 0 {
		result.Reason = "no complete passing probe cell"
		return result
	}
	best := 0.0
	for _, score := range throughput {
		if score > best {
			best = score
		}
	}
	for _, c := range plan.CapacitySelection.ProbeConcurrency {
		score, ok := throughput[c]
		if ok && score >= best*plan.CapacitySelection.ThroughputWithinBest {
			result.Selected = c
			break
		}
	}
	result.Qualified = result.Selected > 0
	result.Reason = ""
	return result
}

func validateProbePreconditions(events []lifecycleEvent) error {
	for _, c := range []int{1, 2, 4, 8} {
		rung := fmt.Sprintf("C%d", c)
		count := 0
		for _, e := range events {
			if e.Phase == "precondition" && e.Rung == rung && e.ConditionID == "unique-early-prefix-equivalence" && !e.ScoredRequest {
				count++
				if e.Status != "completed" || e.ModelIdentityStatus != "observed" {
					return fmt.Errorf("precondition %s was not observed complete", rung)
				}
			}
		}
		if count != 1 {
			return fmt.Errorf("precondition %s count %d, want 1", rung, count)
		}
	}
	return nil
}
