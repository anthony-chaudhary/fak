package gateway

// ep_wire_fanout_test.go — #5528: per-wire EP follower-fanout witnesses for the three
// routes this issue newly covers (Anthropic messages, OpenAI Responses, native Gemini).
//
// ep_fanout_coverage_test.go asserts THAT each covered route contacts a follower rank.
// This file asserts the property that makes that contact worth anything: the release
// happens BEFORE the front rank enters the decode. A fanout issued after the collective
// was already joined would satisfy a "was a follower contacted?" check and still leave
// rank 0 stranded, so each wire is driven against a planner that PARKS in a modeled
// per-step collective and the mirrored request must arrive while it is parked.
//
// Streaming and non-streaming are separate cases per wire on purpose: the two arms of the
// chat wire failed independently (#4855, then #5523), and these three wires split their
// arms in three different places — the Anthropic wire on a body field routed through three
// different streaming implementations, the Responses wire on a body field that only
// changes the render, and the Gemini wire on the URL METHOD rather than the body at all.
//
// As in the #5523 tests, what is modeled is the topology, not the hardware: no weights, no
// device, and the real multi-rank consequence of a missing release is inferred from the
// AllReduce contract rather than observed. Every wait is bounded, so these fail loudly but
// never hang a suite.

import (
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/agent"
)

// epReleasePrecedesDecode drives one arm of one wire and requires that a follower rank was
// released into the decode WHILE the front rank is parked in the modeled collective, on
// wantRoute, with the identical body and the recursion guard set.
func epReleasePrecedesDecode(t *testing.T, urlPath, body, wantRoute string) {
	t.Helper()
	calls := startRecordingFollowerRank(t)

	planner := newCollectivePlanner("test-model", "rank0 decoded through the collective", nil)
	var releaseOnce sync.Once
	release := func() { releaseOnce.Do(func() { close(planner.release) }) }

	srv := newTestServer(t)
	srv.planner = planner
	ts := httptest.NewServer(srv.Handler())
	t.Cleanup(ts.Close)
	// Registered AFTER ts.Close so LIFO cleanup frees the parked decode FIRST — otherwise
	// Close waits on a handler this test is holding open.
	t.Cleanup(release)

	done := make(chan error, 1)
	go func() {
		resp, err := http.Post(ts.URL+urlPath, "application/json", strings.NewReader(body))
		if err != nil {
			done <- err
			return
		}
		defer resp.Body.Close()
		_, err = io.Copy(io.Discard, resp.Body)
		done <- err
	}()

	// The front rank really is inside the modeled collective at this point — this is not a
	// race against a planner that already returned.
	select {
	case <-planner.joined:
	case err := <-done:
		t.Fatalf("POST %s returned without ever entering the modeled collective (err=%v) — the witness is not exercising the served decode path", urlPath, err)
	case <-time.After(epStallBudget):
		t.Fatalf("the modeled decode never entered the collective for %s — the witness is not exercising the served decode path", urlPath)
	}

	select {
	case got := <-calls:
		if got.path != wantRoute {
			t.Fatalf("follower rank was asked for %q, want %q — a follower asked for a wire whose request schema differs from the one the front rank is serving answers 400 and never joins the decode (#5523)", got.path, wantRoute)
		}
		if got.header != "1" {
			t.Fatalf("follower rank %s = %q, want 1 — without it a follower that is itself a fak front rank fans out again", epFollowerHeader, got.header)
		}
		if got.body != body {
			t.Fatalf("follower rank body = %q, want the mirrored request %q", got.body, body)
		}
	case err := <-done:
		t.Fatalf("POST %s finished before any follower rank was released (err=%v)", urlPath, err)
	case <-time.After(epStallBudget):
		t.Fatalf("no follower rank was released into the %s decode while the front rank sat in the collective — rank 0 is alone in a multi-rank collective (#5528)", urlPath)
	}

	release()
	select {
	case err := <-done:
		if err != nil {
			t.Fatalf("draining the front rank's response failed: %v", err)
		}
	case <-time.After(15 * time.Second):
		t.Fatalf("POST %s never returned after the collective was released", urlPath)
	}
}

const (
	epMessagesBody       = `{"model":"test-model","max_tokens":16,"messages":[{"role":"user","content":"decode across ranks"}]}`
	epMessagesStreamBody = `{"model":"test-model","max_tokens":16,"messages":[{"role":"user","content":"decode across ranks"}],"stream":true}`
	epResponsesBody      = `{"model":"test-model","input":"decode across ranks","max_output_tokens":16}`
	epResponsesStream    = `{"model":"test-model","input":"decode across ranks","max_output_tokens":16,"stream":true}`
	epGeminiBody         = `{"contents":[{"role":"user","parts":[{"text":"decode across ranks"}]}]}`
)

// TestAnthropicMessagesReleasesFollowerRanksIntoTheCollective is the #5528 witness for the
// buffered Anthropic wire.
func TestAnthropicMessagesReleasesFollowerRanksIntoTheCollective(t *testing.T) {
	epReleasePrecedesDecode(t, "/v1/messages", epMessagesBody, epRouteMessages)
}

// TestAnthropicMessagesStreamingReleasesFollowerRanksIntoTheCollective is the same witness
// for stream:true. Separate from the buffered case because this wire routes streaming
// through three different implementations depending on the upstream (live passthrough,
// live planner translation, ping-then-synthesize) — the release has to sit ahead of all
// three, which is only true if it happens before the request is even decoded.
func TestAnthropicMessagesStreamingReleasesFollowerRanksIntoTheCollective(t *testing.T) {
	epReleasePrecedesDecode(t, "/v1/messages", epMessagesStreamBody, epRouteMessages)
}

// TestResponsesReleasesFollowerRanksIntoTheCollective is the #5528 witness for the
// buffered Responses wire — the wire Codex speaks.
func TestResponsesReleasesFollowerRanksIntoTheCollective(t *testing.T) {
	epReleasePrecedesDecode(t, "/v1/responses", epResponsesBody, epRouteResponses)
}

// TestResponsesStreamingReleasesFollowerRanksIntoTheCollective is the same witness for
// stream:true on the Responses wire.
func TestResponsesStreamingReleasesFollowerRanksIntoTheCollective(t *testing.T) {
	epReleasePrecedesDecode(t, "/v1/responses", epResponsesStream, epRouteResponses)
}

// TestGeminiGenerateContentReleasesFollowerRanksIntoTheCollective is the #5528 witness for
// the native Gemini wire.
func TestGeminiGenerateContentReleasesFollowerRanksIntoTheCollective(t *testing.T) {
	epReleasePrecedesDecode(t, "/v1beta/models/gemini-test:generateContent", epGeminiBody, epRouteGeminiGenerateContent)
}

// TestGeminiStreamGenerateContentReleasesFollowerRanksIntoTheCollective is the streaming
// arm. On this wire streaming is chosen by the URL METHOD, not by a body field, so the two
// arms differ in the one input the fanout is forbidden to read — and both must still
// mirror onto the SAME buffered follower route (see epRouteGeminiGenerateContent).
func TestGeminiStreamGenerateContentReleasesFollowerRanksIntoTheCollective(t *testing.T) {
	epReleasePrecedesDecode(t, "/v1beta/models/gemini-test:streamGenerateContent", epGeminiBody, epRouteGeminiGenerateContent)
}

// TestGeminiFollowerRouteIsNotDerivedFromTheRequestPath is the security floor #5523
// established, pinned on the one wire where it is not automatic.
//
// /v1beta/models/{model}:{method} carries two client-chosen segments. If either reached the
// follower URL, an inbound request would be choosing part of the address the front rank
// then dials on the operator's own network. Two hostile-shaped model ids are mirrored here;
// neither may appear in what the follower is asked for, and the route must be the constant
// regardless of which method the client hit.
func TestGeminiFollowerRouteIsNotDerivedFromTheRequestPath(t *testing.T) {
	for _, path := range []string{
		"/v1beta/models/..%2f..%2fadmin:generateContent",
		"/v1beta/models/evil.example.com%2Fexfil:streamGenerateContent",
	} {
		t.Run(path, func(t *testing.T) {
			calls := startRecordingFollowerRank(t)
			ts := epFanoutProbeServer(t)

			resp, err := http.Post(ts.URL+path, "application/json", strings.NewReader(epGeminiBody))
			if err != nil {
				t.Fatalf("front-rank request failed: %v", err)
			}
			_, _ = io.Copy(io.Discard, resp.Body)
			resp.Body.Close()

			select {
			case got := <-calls:
				if got.path != epRouteGeminiGenerateContent {
					t.Fatalf("follower rank was asked for %q — the follower URL must be the constant %q, never anything read out of the inbound path (#5523)", got.path, epRouteGeminiGenerateContent)
				}
			case <-time.After(epStallBudget):
				t.Fatal("no follower rank was contacted at all — the assertion would be vacuous")
			}
		})
	}
}

// TestResponsesDenialRecoveryReleasesFollowerRanksExactlyOnce pins the claim made in
// handleResponses about the #5212 denial-recovery arm.
//
// That arm runs a SECOND completeServed inside one HTTP turn. It is covered by the single
// release at the top of the handler because the follower rank serves the same mirrored
// body through the same handler and re-derives the recovery itself — so the correct number
// of releases is exactly one. Two would push the ranks into a decode they are already
// running; zero for the second sample is what a naive reading of "one request, one decode"
// would give.
//
// Non-vacuity: the scenario really does take the recovery arm — the planner is sampled
// twice — so if the second sample needed its own release, this test is where that shows up
// as a second mirrored request.
func TestResponsesDenialRecoveryReleasesFollowerRanksExactlyOnce(t *testing.T) {
	calls := startRecordingFollowerRank(t)

	srv := newTestServer(t)
	planner := &sequencePlanner{comps: []*agent.Completion{
		toolCallTurn("call_denied", "deny_shell", `{"command":"bash install.sh"}`),
		toolCallTurn("call_safe", "allow_read", `{"path":"AGENTS.md"}`),
	}}
	srv.planner = planner
	ts := httptest.NewServer(srv.Handler())
	t.Cleanup(ts.Close)

	body := `{"model":"test-model","input":"fix the install script",` +
		`"tools":[{"type":"function","name":"deny_shell"},{"type":"function","name":"allow_read"}]}`
	resp, err := http.Post(ts.URL+"/v1/responses", "application/json", strings.NewReader(body))
	if err != nil {
		t.Fatalf("responses request failed: %v", err)
	}
	_, _ = io.Copy(io.Discard, resp.Body)
	resp.Body.Close()

	// The turn really took the recovery arm: two samples, i.e. two forward passes on the
	// front rank inside ONE inbound request.
	if got := planner.calls(); got != 2 {
		t.Fatalf("planner calls = %d, want 2 — this turn did not take the #5212 denial-recovery arm, so it witnesses nothing about it", got)
	}

	select {
	case got := <-calls:
		if got.path != epRouteResponses {
			t.Fatalf("follower rank was asked for %q, want %q", got.path, epRouteResponses)
		}
	case <-time.After(epStallBudget):
		t.Fatal("no follower rank was released into the denial-recovery turn (#5528)")
	}
	// A BOUNDED negative window: the second release, if one existed, would follow the first
	// by however long the first sample took.
	select {
	case extra := <-calls:
		t.Fatalf("the denial-recovery sample released the follower ranks a SECOND time (%q) — the follower serves the same body through the same handler and re-derives the recovery itself, so a second mirror pushes it into a decode it is already running", extra.path)
	case <-time.After(time.Second):
	}
}
