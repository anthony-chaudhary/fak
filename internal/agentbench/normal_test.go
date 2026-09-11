package agentbench

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"path/filepath"
	"reflect"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/agentbench/profileplan"
)

func TestAgentBenchNormal184Replay(t *testing.T) {
	plan, err := profileplan.Build(profileplan.Options{Profile: "normal", PreferredConcurrency: 2})
	if err != nil {
		t.Fatal(err)
	}
	schedule, err := compileNormalSchedule(plan)
	if err != nil {
		t.Fatal(err)
	}
	phaseRequests := map[string]int{}
	probe := map[int]normalCell{}
	controls := map[string]bool{}
	for _, cell := range schedule {
		phaseRequests[cell.Phase] += cell.Sessions * cell.TurnsPerSession
		if cell.Phase == "probe" {
			probe[cell.Concurrency] = cell
		}
		if cell.Phase == "controls" {
			if cell.Name == "" || controls[cell.Name] {
				t.Fatalf("control cell name missing/duplicate: %+v", cell)
			}
			controls[cell.Name] = true
		}
	}
	if phaseRequests["probe"] != 64 || phaseRequests["steady"] != 96 || phaseRequests["controls"] != 24 || len(controls) != 6 {
		t.Fatalf("normal schedule geometry = %#v controls=%d, want 64/96/24 and six controls", phaseRequests, len(controls))
	}
	for _, concurrency := range []int{1, 2, 4, 8} {
		cell, ok := probe[concurrency]
		if !ok || cell.Sessions != 8 || cell.TurnsPerSession != 2 {
			t.Fatalf("probe C%d = %+v present=%v, want 8 sessions x 2 turns", concurrency, cell, ok)
		}
	}
	steady := 0
	for _, cell := range schedule {
		if cell.Phase == "steady" {
			steady++
			if cell.Concurrency != 2 || cell.Sessions != 8 || cell.TurnsPerSession != 12 {
				t.Fatalf("steady cell = %+v, want C2 8x12", cell)
			}
		}
	}
	if steady != 1 {
		t.Fatalf("steady cells = %d, want one fixed selected-capacity cell", steady)
	}

	repo := normalCorpusRepo(t, true)
	repo, err = filepath.EvalSymlinks(repo)
	if err != nil {
		t.Fatal(err)
	}
	repoInput, err := freezeRepositoryInput(repo)
	if err != nil {
		t.Fatal(err)
	}
	inputs := newNormalInputs(repo, repoInput)
	server, chatCalls, capCounts, preconditions, shallowProbes := normalReferenceServer(t, 32768)
	defer server.Close()
	result, err := runNormalReplay(context.Background(), io.Discard, server.URL, "fixture-model", t.TempDir(), inputs)
	if err != nil {
		t.Fatalf("run normal replay: %v", err)
	}
	if got := int(chatCalls.Load()); got != 188 {
		t.Fatalf("normal chat requests = %d, want 184 scored plus four priming", got)
	}
	terminals, scored := 0, 0
	for _, event := range result.events {
		if event.Event == "end" || event.EventType == "terminal" {
			terminals++
			if event.ScoredRequest {
				scored++
			}
		}
	}
	if terminals != 188 || scored != 184 || preconditions.Load() != 4 {
		t.Fatalf("terminal/scored/priming = %d/%d/%d, want 188/184/4", terminals, scored, preconditions.Load())
	}
	if shallowProbes.Load() != 0 {
		t.Fatalf("%d scored probe requests used the layer-only preconditioning prompt instead of full 13k-15k session context", shallowProbes.Load())
	}
	if result.selectedConcurrency != 4 || result.highestPassingConcurrency != 8 {
		t.Fatalf("capacity result selected/highest = %d/%d, want 4/8", result.selectedConcurrency, result.highestPassingConcurrency)
	}
	if !result.qualification.Qualified || !result.qualification.SLOs.WarmTTFTP90.Known || !result.qualification.SLOs.ProgressGapMaximum.Known {
		t.Fatalf("real streamed wire did not produce qualifying TTFT/progress evidence: %+v", result.qualification)
	}
	if capCounts[128] < 64 || capCounts[256] < 24 || capCounts[1024] < 8 {
		t.Fatalf("wire output-cap mix = %#v", capCounts)
	}
	reject, rejectedChats, _, _, _ := normalReferenceServer(t, 16384)
	defer reject.Close()
	if _, err := runNormalReplay(context.Background(), io.Discard, reject.URL, "fixture-model", t.TempDir(), inputs); err == nil {
		t.Fatal("normal replay admitted context below 32768")
	}
	if rejectedChats.Load() != 0 {
		t.Fatalf("context rejection dispatched %d chats", rejectedChats.Load())
	}

	t.Run("capacity requires complete observed cells", func(t *testing.T) {
		events := normalCapacityEvents()
		if selection := selectNormalCapacity(plan, events); selection.Qualified {
			t.Fatalf("capacity qualified without explicit per-rung preconditioning: %+v", selection)
		}
		events = append(normalPreconditionEvents(), events...)
		selection := selectNormalCapacity(plan, events)
		if !selection.Qualified || selection.Selected != 4 || selection.HighestPassing != 8 {
			t.Fatalf("capacity selection = %+v, want selected C4/highest passing C8", selection)
		}
		selection = selectNormalCapacity(plan, events[:len(events)-1])
		if selection.Qualified || selection.Reason == "" {
			t.Fatalf("incomplete probe cell qualified: %+v", selection)
		}
	})
}

func TestNormalControlSemantics(t *testing.T) {
	inputs := frozenInputs{system: "system", repo: "repo", area: map[string]string{"A": "area-a", "B": "area-b"}}
	steady := normalCell{Name: "steady", Phase: "steady", Concurrency: 2, Sessions: 8, TurnsPerSession: 12}
	sessionA := normalMessages(inputs, plannedRequest{session: "steady-S1", turn: 1}, steady)
	sessionB := normalMessages(inputs, plannedRequest{session: "steady-S7", turn: 1}, steady)
	if reflect.DeepEqual(sessionA, sessionB) || !normalMessagesContain(sessionA, "area-a") || !normalMessagesContain(sessionB, "area-b") {
		t.Fatalf("steady session areas do not implement fixed 6A/2B assignment: S1=%+v S7=%+v", sessionA, sessionB)
	}

	noShare := normalCell{Name: "no-share", Phase: "controls", Concurrency: 2, Sessions: 2, TurnsPerSession: 2}
	left := normalMessages(inputs, plannedRequest{session: "no-share-S1", turn: 1}, noShare)
	right := normalMessages(inputs, plannedRequest{session: "no-share-S2", turn: 1}, noShare)
	if reflect.DeepEqual(left, right) {
		t.Fatalf("no-share control retained a shareable identical prefix across sessions: %+v", left)
	}

	edit := normalCell{Name: "source-edit-reread", Phase: "controls", Concurrency: 2, Sessions: 1, TurnsPerSession: 4}
	messages := normalMessages(inputs, plannedRequest{session: "source-edit-reread-S1", turn: 3}, edit)
	sawAssistantCall, sawBoundToolResult := false, false
	for _, message := range messages {
		if message.Role == "assistant" && len(message.ToolCalls) > 0 {
			sawAssistantCall = true
		}
		if message.Role == "tool" && message.ToolCallID != "" {
			sawBoundToolResult = true
		}
	}
	if !sawAssistantCall || !sawBoundToolResult {
		t.Fatalf("source-edit-reread control lacks actual assistant edit/reread tool history: %+v", messages)
	}

	plan, err := profileplan.Build(profileplan.Options{Profile: "normal", PreferredConcurrency: 2})
	if err != nil {
		t.Fatal(err)
	}
	schedule, err := compileNormalSchedule(plan)
	if err != nil {
		t.Fatal(err)
	}
	for _, cell := range schedule {
		if cell.Name == "tool-completion-burst" && (cell.Sessions < cell.Concurrency || cell.Concurrency < 2) {
			t.Fatalf("tool-completion-burst cannot create open-loop C2 overlap: %+v", cell)
		}
	}
}

func TestNormalControlsUseRealBoundCorpusHistory(t *testing.T) {
	corpus, err := buildNormalCorpus(context.Background(), &normalCorpusEncoder{}, normalCorpusRepo(t, true))
	if err != nil {
		t.Fatal(err)
	}
	cell := func(name string) normalCell {
		plan, _ := profileplan.Build(profileplan.Options{Profile: "normal", PreferredConcurrency: 2})
		cells, _ := compileNormalSchedule(plan)
		for _, candidate := range cells {
			if candidate.Name == name {
				return candidate
			}
		}
		t.Fatalf("missing control %s", name)
		return normalCell{}
	}

	newArea := cell("new-area-system-warm")
	warmed := normalCorpusRequest(corpus, newArea, plannedRequest{session: "new-area-S1", turn: 1}, 1)
	areaTurn := normalCorpusRequest(corpus, newArea, plannedRequest{session: "new-area-S1", turn: 2}, 1)
	if !normalMessagesContain(warmed.Messages, corpus.Areas["A"].content) || !normalMessagesContain(areaTurn.Messages, corpus.Areas["B"].content) || normalMessagesContain(areaTurn.Messages, corpus.Areas["A"].content) {
		t.Fatalf("new-area control did not execute a logical A-warm then actual-B chain")
	}
	revisit := cell("area-eviction-revisit")
	sequence := []string{"A", "B", "", "A"}
	for turn, area := range sequence {
		request := normalCorpusRequest(corpus, revisit, plannedRequest{session: "revisit-S1", turn: turn + 1}, 1)
		if area != "" && !normalMessagesContain(request.Messages, corpus.Areas[area].content) {
			t.Fatalf("eviction/revisit turn %d lacks actual area %s bytes", turn+1, area)
		}
		if area == "" && !normalMessagesContain(request.Messages, "pressure") {
			t.Fatalf("eviction/revisit turn %d lacks explicit pressure material", turn+1)
		}
	}

	noShare := cell("no-share")
	if noShare.Sessions*noShare.TurnsPerSession != 4 {
		t.Fatalf("no-share geometry = %+v, want four independently isolated requests", noShare)
	}
	prefixes := map[string]bool{}
	for turn := 1; turn <= 4; turn++ {
		request := normalCorpusRequest(corpus, noShare, plannedRequest{session: "no-share-S1", turn: turn}, 1)
		prefix := request.Messages[0].Content
		if prefixes[prefix] {
			t.Fatalf("no-share request %d reused prefix %q", turn, prefix)
		}
		prefixes[prefix] = true
	}

	edit := normalCorpusRequest(corpus, cell("source-edit-reread"), plannedRequest{session: "edit-S1", turn: 3}, 1)
	if normalMessagesContain(edit.Messages, "recorded source edit generation") || normalMessagesContain(edit.Messages, "recorded reread after edit") {
		t.Fatal("source edit/reread control used marker strings instead of actual before/after source bytes")
	}
	if !toolResultsAreBound(edit.Messages) {
		t.Fatal("source edit/reread history contains an unbound tool result")
	}
	burst := normalCorpusRequest(corpus, cell("tool-completion-burst"), plannedRequest{session: "burst-S1", turn: 1}, 1)
	if !toolResultsAreBound(burst.Messages) {
		t.Fatal("tool completion burst contains a tool result without its assistant tool call")
	}
}

func TestNormalPrefixControlsUseExactTokenPositions(t *testing.T) {
	corpus, err := buildNormalCorpus(context.Background(), &normalCorpusEncoder{}, normalCorpusRepo(t, true))
	if err != nil {
		t.Fatal(err)
	}
	encode := func(request referenceRequest) ([]byte, []int) {
		raw, err := json.Marshal(request)
		if err != nil {
			t.Fatal(err)
		}
		ids := make([]int, len(raw))
		for i, token := range raw {
			ids[i] = int(token)
		}
		return raw, ids
	}
	control := func(name string) normalCell {
		return normalCell{Name: name, Phase: "controls", Concurrency: 2, Sessions: 1, TurnsPerSession: 4, OutputTokens: 256}
	}

	var forkIDs [][]int
	var forkRequests []referenceRequest
	for turn := 1; turn <= 4; turn++ {
		request := normalCorpusRequest(corpus, control("same-prefix-forks"), plannedRequest{session: "fork-S1", turn: turn}, 1)
		_, ids := encode(request)
		forkIDs = append(forkIDs, ids)
		forkRequests = append(forkRequests, request)
	}
	baseMessages := forkRequests[0].Messages[:len(forkRequests[0].Messages)-1]
	forkSuffixes := map[string]bool{}
	for i, request := range forkRequests {
		if !reflect.DeepEqual(request.Messages[:len(request.Messages)-1], baseMessages) || !reflect.DeepEqual(request.Tools, forkRequests[0].Tools) {
			t.Fatalf("same-prefix fork %d changed the frozen base or tool contract", i+1)
		}
		suffix := request.Messages[len(request.Messages)-1].Content
		forkSuffixes[suffix] = true
		if i > 0 {
			prefix := commonTokenPrefix(forkIDs[0], forkIDs[i])
			if prefix*100 < min(len(forkIDs[0]), len(forkIDs[i]))*80 {
				t.Fatalf("same-prefix fork %d diverged before the substantial frozen branch: prefix=%d tokens=%d", i+1, prefix, len(forkIDs[0]))
			}
		}
	}
	if len(forkSuffixes) != 4 {
		t.Fatalf("same-prefix control did not produce four distinct continuations: %d", len(forkSuffixes))
	}

	var noShare [][]int
	for turn := 1; turn <= 4; turn++ {
		_, ids := encode(normalCorpusRequest(corpus, control("no-share"), plannedRequest{session: "no-share-S1", turn: turn}, 1))
		noShare = append(noShare, ids)
	}
	for i := 1; i < len(noShare); i++ {
		if prefix := commonTokenPrefix(noShare[0], noShare[i]); prefix >= 128 {
			t.Fatalf("no-share request %d first diverged after a reusable %d-token prefix", i+1, prefix)
		}
	}

	_, areaA := encode(normalCorpusRequest(corpus, control("new-area-system-warm"), plannedRequest{session: "area-S1", turn: 1}, 1))
	_, areaB := encode(normalCorpusRequest(corpus, control("new-area-system-warm"), plannedRequest{session: "area-S1", turn: 2}, 1))
	if prefix := commonTokenPrefix(areaA, areaB); prefix <= len(corpus.SystemTools.content)+len(corpus.RepositoryMap.content) || prefix >= min(len(areaA), len(areaB)) {
		t.Fatalf("new-area A→B did not retain only the exact warm system/repository prefix: prefix=%d", prefix)
	}

	nonceOffset := -1
	for _, concurrency := range []int{1, 2, 4, 8} {
		request := normalCorpusRequest(corpus, normalCell{Name: fmt.Sprintf("probe-c%d", concurrency), Phase: "probe", Concurrency: concurrency}, plannedRequest{session: "probe-S1", turn: 1}, 1)
		raw, _ := encode(request)
		offset := bytes.Index(raw, []byte(corpus.Preconditions[concurrency].Nonce))
		encodedArea, _ := json.Marshal(corpus.Areas["A"].content[:64])
		areaOffset := bytes.Index(raw, encodedArea[1:len(encodedArea)-1])
		if offset < 0 || areaOffset < 0 || offset >= areaOffset {
			t.Fatalf("C%d namespace token position = %d, area=%d; namespace must precede shared corpus", concurrency, offset, areaOffset)
		}
		if nonceOffset < 0 {
			nonceOffset = offset
		} else if offset != nonceOffset {
			t.Fatalf("C%d namespace position=%d want equivalent %d", concurrency, offset, nonceOffset)
		}
	}
}

func toolResultsAreBound(messages []chatMessage) bool {
	seen := map[string]bool{}
	for _, message := range messages {
		if message.Role == "assistant" {
			for _, call := range message.ToolCalls {
				seen[call.ID] = true
			}
		}
		if message.Role == "tool" && (message.ToolCallID == "" || !seen[message.ToolCallID]) {
			return false
		}
	}
	return true
}

func TestNormalCapacityUsesObservedCellThroughput(t *testing.T) {
	plan, err := profileplan.Build(profileplan.Options{Profile: "normal", PreferredConcurrency: 2})
	if err != nil {
		t.Fatal(err)
	}
	events := normalPreconditionEvents()
	events = append(events, capacityTimeline(1, 100*time.Millisecond, true)...)
	events = append(events, capacityTimeline(2, 50*time.Millisecond, true)...)
	events = append(events, capacityTimeline(4, 25*time.Millisecond, false)...)
	events = append(events, capacityTimeline(8, 12*time.Millisecond, false)...)
	if got := selectNormalCapacity(plan, events); !got.Qualified || got.Selected != 2 || got.HighestPassing != 2 {
		t.Fatalf("equal per-request latency but twice-throughput C2 selection = %+v", got)
	}

	tied := normalPreconditionEvents()
	tied = append(tied, capacityTimeline(1, 45*time.Millisecond, true)...)
	tied = append(tied, capacityTimeline(2, 100*time.Millisecond, true)...)
	tied = append(tied, capacityTimeline(4, 25*time.Millisecond, false)...)
	tied = append(tied, capacityTimeline(8, 12*time.Millisecond, false)...)
	if got := selectNormalCapacity(plan, tied); !got.Qualified || got.Selected != 1 || got.HighestPassing != 2 {
		t.Fatalf("throughput tie did not choose smallest passing rung: %+v", got)
	}
}

func capacityTimeline(concurrency int, releaseEvery time.Duration, passing bool) []lifecycleEvent {
	base := time.Date(2026, 9, 10, 14, 0, 0, 0, time.UTC)
	known := true
	events := make([]lifecycleEvent, 0, 16)
	for i := 0; i < 16; i++ {
		started := base.Add(time.Duration(i/concurrency) * releaseEvery)
		status := "completed"
		if !passing && i == 0 {
			status = "failed"
		}
		events = append(events, lifecycleEvent{Sequence: i + 1, RequestID: fmt.Sprintf("C%d-%d", concurrency, i), Rung: fmt.Sprintf("C%d", concurrency), Phase: "probe", Event: "end", EventType: "terminal", Status: status, ModelIdentityStatus: "observed", UsageKnown: &known, Usage: &observedUsage{PromptTokens: 100, CompletionTokens: 1, TotalTokens: 101}, StartedAt: started.Format(time.RFC3339Nano), CompletedAt: started.Add(100 * time.Millisecond).Format(time.RFC3339Nano), DurationMilliseconds: 100})
	}
	return events
}

func normalMessagesContain(messages []chatMessage, text string) bool {
	for _, message := range messages {
		if strings.Contains(message.Content, text) {
			return true
		}
	}
	return false
}

func normalCapacityEvents() []lifecycleEvent {
	var events []lifecycleEvent
	sequence := 1
	for _, cell := range []struct {
		concurrency int
		duration    int64
	}{{1, 100}, {2, 52}, {4, 25}, {8, 26}} {
		for request := 1; request <= 16; request++ {
			known := true
			events = append(events, lifecycleEvent{
				Sequence: sequence, RequestID: "probe", Rung: "C" + string(rune('0'+cell.concurrency)), Phase: "probe",
				Event: "end", EventType: "terminal", Status: "completed", ServiceVerdict: "OBSERVED_PASS",
				RequestedModel: "fixture-model", ObservedModel: "fixture-model", ModelIdentityStatus: "matched",
				UsageKnown: &known, Usage: &observedUsage{PromptTokens: 100, CompletionTokens: 10, TotalTokens: 110},
				DurationMilliseconds: cell.duration,
			})
			sequence++
		}
	}
	return events
}

func normalPreconditionEvents() []lifecycleEvent {
	known := true
	events := make([]lifecycleEvent, 0, 4)
	for i, concurrency := range []int{1, 2, 4, 8} {
		events = append(events, lifecycleEvent{
			Sequence: i + 1, RequestID: fmt.Sprintf("precondition-c%d", concurrency), Rung: fmt.Sprintf("C%d", concurrency),
			Phase: "precondition", ConditionID: "unique-early-prefix-equivalence", Event: "end", EventType: "terminal",
			ScoredRequest: false, Status: "completed", ServiceVerdict: "OBSERVED_OK", CacheVerdict: "UNKNOWN",
			RequestedModel: "fixture-model", ObservedModel: "fixture-model", ModelIdentityStatus: "observed",
			UsageKnown: &known, Usage: &observedUsage{PromptTokens: 14000, CompletionTokens: 1, TotalTokens: 14001},
		})
	}
	return events
}

func normalReferenceServer(t *testing.T, contextTokens int) (*httptest.Server, *atomic.Int64, map[int]int, *atomic.Int64, *atomic.Int64) {
	t.Helper()
	var chats atomic.Int64
	var preconditions atomic.Int64
	var shallowProbes atomic.Int64
	var burstEntered atomic.Int64
	var burstReleaseOnce sync.Once
	burstRelease := make(chan struct{})
	caps := map[int]int{}
	var capsMu sync.Mutex
	var server *httptest.Server
	server = httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/v1/models":
			_ = json.NewEncoder(w).Encode(map[string]any{"data": []any{map[string]any{"id": "fixture-model", "context_length": contextTokens, "fak_capabilities": map[string]any{"prompt_tokenization": map[string]any{"endpoint": "/v1/fak/tokenize"}}}}})
		case "/v1/fak/tokenize":
			var req map[string]any
			_ = json.NewDecoder(r.Body).Decode(&req)
			reserve, _ := req["max_tokens"].(float64)
			body, _ := json.Marshal(req)
			text := strings.ToLower(string(body))
			tokens := 14000
			switch {
			case reserve == 0 && strings.Contains(text, "tool"):
				tokens = 4000
			case reserve == 0 && strings.Contains(text, "repository"):
				tokens = 2000
			case reserve == 0:
				tokens = 8000
			default:
				tokens += (strings.Count(text, `"role"`) - 1) * 1400
				if tokens > 31744 {
					tokens = 31744
				}
			}
			ids := make([]int, tokens)
			for i := range ids {
				ids[i] = i % 251
			}
			digest := fmt.Sprintf("%x", sha256.Sum256(body))
			_ = json.NewEncoder(w).Encode(map[string]any{"model_id": "fixture-model", "renderer_id": "fixture-renderer", "tokenizer_id": "fixture-tokenizer", "token_ids": ids, "prompt_tokens": tokens, "context_window_tokens": contextTokens, "reserved_output_tokens": int(reserve), "rendered_sha256": digest})
		case "/v1/chat/completions":
			var req map[string]any
			_ = json.NewDecoder(r.Body).Decode(&req)
			if raw, ok := req["max_tokens"].(float64); ok {
				capsMu.Lock()
				caps[int(raw)]++
				capsMu.Unlock()
			}
			if metadata, _ := req["metadata"].(map[string]any); metadata != nil {
				phase, _ := metadata["profile"].(string)
				rung, _ := metadata["rung"].(string)
				if phase == "normal-precondition" {
					preconditions.Add(1)
				} else if strings.HasPrefix(rung, "probe-c") && strings.Contains(strings.ToLower(fmt.Sprint(req["messages"])), "preconditioning marker") {
					shallowProbes.Add(1)
				}
				switch metadata["rung"] {
				case "probe-c1":
					time.Sleep(200 * time.Millisecond)
				case "probe-c2":
					time.Sleep(100 * time.Millisecond)
				case "probe-c4":
					time.Sleep(5 * time.Millisecond)
				case "probe-c8":
					time.Sleep(50 * time.Millisecond)
				case "tool-completion-burst":
					if burstEntered.Add(1) == 2 {
						burstReleaseOnce.Do(func() { close(burstRelease) })
					}
					select {
					case <-burstRelease:
					case <-r.Context().Done():
						return
					}
				}
			}
			id := chats.Add(1)
			w.Header().Set("Content-Type", "text/event-stream")
			fmt.Fprintf(w, "data: {\"model\":\"fixture-model\",\"choices\":[{\"delta\":{\"content\":\"ok-%d\"},\"finish_reason\":\"stop\"}],\"usage\":{\"prompt_tokens\":3,\"completion_tokens\":1,\"total_tokens\":4}}\n\ndata: [DONE]\n\n", id)
		default:
			http.NotFound(w, r)
		}
	}))
	return server, &chats, caps, &preconditions, &shallowProbes
}
