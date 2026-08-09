package gateway

// ep_completions_fanout_test.go — #5523: EP follower fanout on the LEGACY
// text-completion wire (POST /v1/completions).
//
// The fanout bridge was wired to the chat route only, in two independent ways:
// handleCompletions never called startEPFanoutFollowers, and epFanoutURLsFromEnv
// hardcoded the /v1/chat/completions suffix — so even a call from the legacy handler
// would have asked every follower for a DIFFERENT route with a different request
// schema than the front rank was serving. Either one alone leaves the front rank
// entering a multi-rank decode no follower was released into.
//
// What is asserted here is the RELEASE, not the collective: these tests model the
// topology in-process with no weights and no device (collectivePlanner parks in a
// modeled per-step collective, standing in for the sharded decode). Whether the real
// consequence on multi-rank hardware is a hang, a timeout, or a silently degraded
// single-rank answer is INFERRED from the AllReduce contract in startEPFanoutFollowers'
// doc comment, not observed — no multi-rank deployment was run.
//
// Note the failure is invisible in single-rank use: with FAK_EP_FANOUT_ADDRS unset
// epFanoutURLsFromEnv returns nothing and both routes no-op. Every test below
// therefore configures a real follower rank, or it would assert nothing.
//
// Every wait is bounded, so these can fail loudly but never hang a suite.

import (
	"bytes"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/agent"
)

// epFollowerCall records one mirrored request as a follower rank saw it: which ROUTE
// the front rank asked for, the body it mirrored, and the recursion-guard header it
// stamped. startFollowerRank (ep_stream_stall_test.go) reports only the body, which
// cannot distinguish "asked for the route the front rank is serving" from "asked for
// the chat route regardless" — the second half of #5523.
type epFollowerCall struct {
	path   string
	body   string
	header string
}

// startRecordingFollowerRank stands up a fake EP follower rank, points
// FAK_EP_FANOUT_ADDRS at it, and returns the channel its mirrored requests land on.
// It answers with a whole, immediately-terminating body on both wires so the front
// rank's bounded fanout wait returns whether the mirror was streaming or not.
func startRecordingFollowerRank(t *testing.T) chan epFollowerCall {
	t.Helper()
	calls := make(chan epFollowerCall, 4)
	follower := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		raw, _ := io.ReadAll(r.Body)
		select {
		case calls <- epFollowerCall{path: r.URL.Path, body: string(raw), header: r.Header.Get(epFollowerHeader)}:
		default:
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"object":"text_completion","choices":[{"text":"rank"}]}`))
	}))
	t.Cleanup(follower.Close)
	t.Setenv("FAK_EP_FANOUT_ADDRS", follower.URL)
	return calls
}

// assertLegacyFollowerCall checks one mirrored request carries everything a follower
// rank needs to run the SAME forward pass as the front rank: the legacy route the
// front rank is actually serving, the recursion-guard header, and the identical body.
func assertLegacyFollowerCall(t *testing.T, got epFollowerCall, wantBody []byte) {
	t.Helper()
	if got.path != "/v1/completions" {
		t.Fatalf("follower rank was asked for %q, want %q — the front rank is serving the LEGACY wire, and the chat route takes a different request schema (#5523)", got.path, "/v1/completions")
	}
	if got.header != "1" {
		t.Fatalf("follower rank %s = %q, want 1 — without it a follower that is itself a fak front rank fans out again", epFollowerHeader, got.header)
	}
	if got.body != string(wantBody) {
		t.Fatalf("follower rank body = %q, want the mirrored request %q", got.body, wantBody)
	}
}

// TestCompletionsReleasesFollowerRanksIntoTheCollective is the #5523 witness for the
// buffered legacy wire. The front rank is parked in the modeled collective; a follower
// rank must ALREADY have been released into the same decode, over the same route.
//
// Against the unfixed handler no follower is contacted at all, so the wait for the
// mirrored request fails here.
func TestCompletionsReleasesFollowerRanksIntoTheCollective(t *testing.T) {
	calls := startRecordingFollowerRank(t)

	planner := newCollectivePlanner("test-model", "rank0 decoded through the collective", nil)
	var releaseOnce sync.Once
	release := func() { releaseOnce.Do(func() { close(planner.release) }) }

	srv := newTestServer(t)
	srv.planner = planner
	ts := httptest.NewServer(srv.Handler())
	t.Cleanup(ts.Close)
	// Registered AFTER ts.Close so LIFO cleanup frees the parked decode FIRST; see the
	// sibling repro in ep_completions_stall_test.go.
	t.Cleanup(release)

	body, err := json.Marshal(CompletionRequest{
		Model:     "test-model",
		Prompt:    json.RawMessage(`"decode across ranks"`),
		MaxTokens: 16,
	})
	if err != nil {
		t.Fatal(err)
	}
	heads := make(chan *http.Response, 1)
	fails := make(chan error, 1)
	go func() {
		resp, err := http.Post(ts.URL+"/v1/completions", "application/json", bytes.NewReader(body))
		if err != nil {
			fails <- err
			return
		}
		heads <- resp
	}()

	// The front rank really is inside the modeled collective at this point — this is not
	// a race against a no-op planner.
	select {
	case <-planner.joined:
	case err := <-fails:
		t.Fatalf("legacy completion request failed: %v", err)
	case <-time.After(epStallBudget):
		t.Fatal("the modeled decode never entered the collective — the repro is not exercising the served path")
	}

	select {
	case got := <-calls:
		assertLegacyFollowerCall(t, got, body)
	case err := <-fails:
		t.Fatalf("legacy completion request failed: %v", err)
	case <-time.After(epStallBudget):
		t.Fatal("no follower rank was released into the legacy-completions decode — the front rank is alone in a multi-rank collective (#5523)")
	}

	release()
	select {
	case resp := <-heads:
		defer resp.Body.Close()
		if resp.StatusCode != http.StatusOK {
			t.Fatalf("status = %d, want 200", resp.StatusCode)
		}
		var out CompletionResponse
		if err := json.NewDecoder(resp.Body).Decode(&out); err != nil {
			t.Fatalf("decode legacy response: %v", err)
		}
		if len(out.Choices) != 1 || out.Choices[0].Text != planner.content {
			t.Fatalf("choices = %+v, want one choice with the front rank's own text %q", out.Choices, planner.content)
		}
	case err := <-fails:
		t.Fatalf("legacy completion request failed: %v", err)
	case <-time.After(15 * time.Second):
		t.Fatal("the legacy completion never returned after the collective was released")
	}
}

// TestCompletionsStreamingReleasesFollowerRanksIntoTheCollective is the same witness
// for stream:true. It is a separate case because the streaming half of the chat route
// regressed independently once (#4855: a stream:true request returned before starting
// any follower), and because the legacy stream path now opens an SSE preamble (#5514)
// BEFORE the decode — the fanout has to happen on the same side of that split as it
// does on the chat wire, not after it.
func TestCompletionsStreamingReleasesFollowerRanksIntoTheCollective(t *testing.T) {
	calls := startRecordingFollowerRank(t)

	planner := newCollectivePlanner("test-model", "rank0 decoded through the collective", nil)
	var releaseOnce sync.Once
	release := func() { releaseOnce.Do(func() { close(planner.release) }) }

	srv := newTestServer(t)
	srv.planner = planner
	ts := httptest.NewServer(srv.Handler())
	t.Cleanup(ts.Close)
	t.Cleanup(release)

	body, err := json.Marshal(CompletionRequest{
		Model:     "test-model",
		Prompt:    json.RawMessage(`"decode across ranks"`),
		Stream:    true,
		MaxTokens: 16,
	})
	if err != nil {
		t.Fatal(err)
	}
	tap := tapChatStream(ts.URL+"/v1/completions", body)

	// The #5514 preamble must still land first: the fanout is added to that flush, never
	// in place of it.
	resp := tap.waitHead(t, epStallBudget, "no HTTP status line/headers before the modeled collective join")
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("status = %d, want 200", resp.StatusCode)
	}
	opening := decodeCompletionSSEChunk(t, tap.waitFrame(t, epStallBudget, "no opening SSE chunk before the modeled collective join"))
	if opening.Object != "text_completion" {
		t.Fatalf("opening object = %q, want text_completion", opening.Object)
	}

	select {
	case <-planner.joined:
	case <-time.After(epStallBudget):
		t.Fatal("the modeled decode never entered the collective — the repro is not exercising the served path")
	}

	select {
	case got := <-calls:
		assertLegacyFollowerCall(t, got, body)
	case <-time.After(epStallBudget):
		t.Fatal("no follower rank was released into the streaming legacy-completions decode (#5523)")
	}

	release()
	rest := tap.drain(t, 15*time.Second)
	var text strings.Builder
	var sawDone bool
	for _, line := range rest {
		if line == "data: [DONE]" {
			sawDone = true
			continue
		}
		text.WriteString(decodeCompletionSSEChunk(t, line).Choices[0].Text)
	}
	if got := text.String(); got != planner.content {
		t.Fatalf("reassembled streamed text = %q, want the front rank's own decode %q", got, planner.content)
	}
	if !sawDone {
		t.Fatalf("stream never terminated with [DONE]: %v", rest)
	}
}

// TestCompletionsFollowerRequestDoesNotFanOutAgain pins the recursion guard on the
// route #5523 newly fans out from. A follower rank that is itself a fak gateway serves
// the mirrored request through this same handler; if it fanned out in turn, one client
// request would become an exponential broadcast across the fleet. The guard is
// startEPFanoutFollowers' first statement and is route-independent — this asserts the
// LEGACY entry point really does reach it, with followers configured (unconfigured, the
// no-op path would make the assertion vacuous).
func TestCompletionsFollowerRequestDoesNotFanOutAgain(t *testing.T) {
	calls := startRecordingFollowerRank(t)

	srv := newTestServer(t)
	srv.planner = stubPlanner{comp: &agent.Completion{
		Message:      agent.Message{Role: agent.RoleAssistant, Content: "follower decoded"},
		FinishReason: "stop",
	}}
	ts := httptest.NewServer(srv.Handler())
	t.Cleanup(ts.Close)

	body, err := json.Marshal(CompletionRequest{
		Model:  "test-model",
		Prompt: json.RawMessage(`"mirrored from the front rank"`),
	})
	if err != nil {
		t.Fatal(err)
	}
	req, err := http.NewRequest(http.MethodPost, ts.URL+"/v1/completions", bytes.NewReader(body))
	if err != nil {
		t.Fatal(err)
	}
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set(epFollowerHeader, "1")
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		t.Fatalf("mirrored legacy request failed: %v", err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("status = %d, want 200 — a follower must still SERVE the mirrored request", resp.StatusCode)
	}
	if _, err := io.Copy(io.Discard, resp.Body); err != nil {
		t.Fatalf("drain follower response: %v", err)
	}

	// A BOUNDED negative window, not a bare `default`: the handler's fanout wait is a
	// deferred call, so a recursive mirror could still be in flight when the client's
	// own response has already landed. Waiting a beat is what makes the absence real.
	select {
	case got := <-calls:
		t.Fatalf("a follower's own legacy request fanned out again to %q — one client request would become an exponential EP broadcast", got.path)
	case <-time.After(time.Second):
	}
}
