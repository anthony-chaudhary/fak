package gateway

// ep_fanout_coverage_test.go — #5528: the whole-server EP follower-fanout coverage gate.
//
// #5523 fixed ONE uncovered wire (the legacy text-completion route) by inspection. The
// reason the same gap could then sit unnoticed on four more served decode paths is that
// nothing ever asserted the property at the level it actually holds at: EVERY route the
// server registers either releases the follower ranks into its decode, or is named here
// with a written reason why it has no decode to release them into.
//
// So this file is keyed off the LIVE route table — (*Server).routeTable(), the same
// single source of truth openapi_spec_test.go ranges over — rather than a free-standing
// list. A new route with no classification fails TestEPFanoutCoverageClassifiesEveryServedRoute,
// and a route classified as covered that does not actually contact a follower rank fails
// TestEveryEPFanoutCoveredRouteReleasesFollowerRanks. A new served wire therefore cannot
// land without either wiring the bridge or writing down why it does not need it.
//
// What is asserted is the RELEASE, not the collective. Like the #5523 tests, the topology
// is modeled in-process with no weights and no device; whether the real consequence on
// multi-rank hardware is a hang, a timeout, or a silently degraded single-rank answer is
// INFERRED from the AllReduce contract in startEPFanoutFollowers' doc comment, not
// observed. No multi-rank deployment was run.
//
// Note the whole defect is invisible on a single-rank serve: with FAK_EP_FANOUT_ADDRS
// unset, epFanoutURLsFromEnv returns nothing and every route no-ops. Every case below
// therefore stands up a real follower rank, or it would assert nothing.

import (
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/agent"
)

// epFanoutProbe drives ONE arm of ONE covered route end to end and says which follower
// route that arm must mirror onto. Streaming and non-streaming are separate probes on
// purpose: during #4855 the streaming arm of the chat wire returned before starting any
// follower while the buffered arm was fine, and during #5523 the two arms of the legacy
// wire failed differently again. One probe per wire would have passed through both.
type epFanoutProbe struct {
	// pattern is the routeTable() registration pattern this arm exercises. It is what
	// ties a probe back to the served surface for the exhaustiveness gate.
	pattern string
	// name distinguishes the arms of one pattern in test output.
	name string
	// urlPath is a concrete path that dispatches to pattern.
	urlPath string
	// body is a minimal well-formed request for that wire.
	body string
	// wantRoute is the epRoute* constant the follower rank must be asked for. Asserting
	// the ROUTE, not merely "some follower was contacted", is the second half of #5523:
	// a follower asked for a wire whose request schema differs from the one the front
	// rank is serving answers 400 and never joins the decode.
	wantRoute string
}

// epFanoutProbes is the covered half of the classification: every served route that
// enters a model decode, and how to drive each of its arms.
var epFanoutProbes = []epFanoutProbe{
	{
		pattern: "/v1/chat/completions", name: "buffered",
		urlPath:   "/v1/chat/completions",
		body:      `{"model":"test-model","messages":[{"role":"user","content":"decode across ranks"}],"max_tokens":16}`,
		wantRoute: epRouteChatCompletions,
	},
	{
		pattern: "/v1/chat/completions", name: "streaming",
		urlPath:   "/v1/chat/completions",
		body:      `{"model":"test-model","messages":[{"role":"user","content":"decode across ranks"}],"max_tokens":16,"stream":true}`,
		wantRoute: epRouteChatCompletions,
	},
	{
		pattern: "/v1/completions", name: "buffered",
		urlPath:   "/v1/completions",
		body:      `{"model":"test-model","prompt":"decode across ranks","max_tokens":16}`,
		wantRoute: epRouteCompletions,
	},
	{
		pattern: "/v1/completions", name: "streaming",
		urlPath:   "/v1/completions",
		body:      `{"model":"test-model","prompt":"decode across ranks","max_tokens":16,"stream":true}`,
		wantRoute: epRouteCompletions,
	},
	{
		pattern: "/v1/messages", name: "buffered",
		urlPath:   "/v1/messages",
		body:      `{"model":"test-model","max_tokens":16,"messages":[{"role":"user","content":"decode across ranks"}]}`,
		wantRoute: epRouteMessages,
	},
	{
		pattern: "/v1/messages", name: "streaming",
		urlPath:   "/v1/messages",
		body:      `{"model":"test-model","max_tokens":16,"messages":[{"role":"user","content":"decode across ranks"}],"stream":true}`,
		wantRoute: epRouteMessages,
	},
	{
		pattern: "/v1/responses", name: "buffered",
		urlPath:   "/v1/responses",
		body:      `{"model":"test-model","input":"decode across ranks","max_output_tokens":16}`,
		wantRoute: epRouteResponses,
	},
	{
		pattern: "/v1/responses", name: "streaming",
		urlPath:   "/v1/responses",
		body:      `{"model":"test-model","input":"decode across ranks","max_output_tokens":16,"stream":true}`,
		wantRoute: epRouteResponses,
	},
	{
		// The Gemini wire chooses streaming by METHOD, not by a body field, so its two
		// arms are two different inbound URLs. Both mirror onto the SAME buffered
		// follower route — see epRouteGeminiGenerateContent for why.
		pattern: "/v1beta/", name: "generateContent",
		urlPath:   "/v1beta/models/gemini-test:generateContent",
		body:      `{"contents":[{"role":"user","parts":[{"text":"decode across ranks"}]}]}`,
		wantRoute: epRouteGeminiGenerateContent,
	},
	{
		pattern: "/v1beta/", name: "streamGenerateContent",
		urlPath:   "/v1beta/models/gemini-test:streamGenerateContent",
		body:      `{"contents":[{"role":"user","parts":[{"text":"decode across ranks"}]}]}`,
		wantRoute: epRouteGeminiGenerateContent,
	},
}

// Exemption reasons. A route is exempt because of a PROPERTY of the route, written down
// once and shared by every route that has it — not because wiring it looked awkward.
const (
	// epExemptNoDecode: a control-plane / observability route. It answers from gateway
	// state, a policy reload, a ledger or a provider hook and never calls a planner, so
	// there is no forward pass for a follower rank to be released into.
	epExemptNoDecode = "control/observability route: answers from gateway state and never enters a model decode, so there is no collective for a follower rank to join"

	// epExemptLocalCompute: answered by a deterministic local computation (a hash-derived
	// embedding, a token estimate, a classifier over the request text). Same conclusion as
	// epExemptNoDecode, different cause, so it is named separately rather than folded in.
	epExemptLocalCompute = "answered by a deterministic local computation, not a planner call: no rank enters a decode"

	// epExemptOwnedLoop: the route runs fak's OWN multi-turn governed loop. The bridge's
	// unit is one inbound body == one forward pass; an owned loop is N forward passes whose
	// turns 2..N are transcripts synthesized server-side AFTER tool calls with real side
	// effects. Mirroring the inbound body would reproduce only turn 1 and then let the ranks
	// diverge — and would have every follower rank execute the tool calls a second time.
	// Covering this needs a rank barrier inside the loop, not a request mirror.
	//
	// #5532 settled what that barrier is, and it is not something to invent: it ALREADY
	// exists as model.EPDecodeCoordinator / model.RunEPFollower (#4835), announced from
	// Session.Prefill and Session.Step. That unit is FINER than a turn and therefore
	// route-agnostic — an owned loop that decodes through those two entry points keeps every
	// rank in step for free, and a follower parked in RunEPFollower never tokenizes, never
	// samples and structurally cannot execute a tool. What is missing is the serve wiring
	// that installs it (it has no non-test caller today), not a primitive. So this exemption
	// is not provisional pending a new design: an owned-loop route is exempt from the
	// REQUEST-MIRROR bridge permanently, and the follow-on work is #4835's wiring.
	epExemptOwnedLoop = "runs fak's own multi-turn governed loop: N forward passes and real tool side effects from one inbound body, which a request mirror cannot reproduce without diverging the ranks; the in-loop barrier that covers it already exists as model.EPDecodeCoordinator/RunEPFollower (#4835) and only lacks serve wiring (#5532)"
)

// The classification above is by ROUTE PATTERN, and one pattern is MODE-SENSITIVE: under
// `fak serve --native` the /v1/messages handler branches into serveNativeMessages, which
// drives agent.RunArm(fak=true) — the same owned governed loop epExemptOwnedLoop describes,
// on a pattern classified COVERED. #5532 measured the consequence: with one follower rank
// configured, a two-turn native request drove 4 forward passes and 2 tool-result turns
// across the ranks instead of 2 and 1, i.e. the follower re-ran the whole loop and dispatched
// the tool a second time. handleAnthropicMessages therefore skips the fanout when s.native
// is set, and TestNativeMessagesFanoutDoesNotDuplicateTheOwnedLoop
// (ep_owned_loop_fanout_test.go) pins that. The probes below construct a NON-native server,
// so what they classify as covered is the proxy arm, which is the arm the mirror is sound for.
// The residual is stated plainly: a native multi-rank serve enters its collectives on rank 0
// alone until #4835's coordinator is wired into the serve path.

// epFanoutExemptRoutes names every served route that does NOT release follower ranks,
// with the reason. Being on this list is a claim, and the claim is checkable: each reason
// asserts the route reaches no planner.
var epFanoutExemptRoutes = map[string]string{
	"/": "read-only discovery homepage has no generation or expert work to fan out",
	// A2A Agent-to-Agent surface (#1019) — a task-record control plane. handleA2ASendMessage
	// validates a method against the registry and files a task record; no planner call.
	"/a2a/v1/messages":   epExemptNoDecode,
	"/a2a/v1/tasks":      epExemptNoDecode,
	"/a2a/v1/agent-card": epExemptNoDecode,
	"/a2a/v1/tasks/":     epExemptNoDecode,

	// embedText/moderation classification are hash-derived and local (embeddings.go,
	// moderations.go); EstimateAnthropicTokens is arithmetic over the decoded request.
	"/v1/embeddings":            epExemptLocalCompute,
	"/v1/moderations":           epExemptLocalCompute,
	"/v1/messages/count_tokens": epExemptLocalCompute,

	// The fak-native surface: syscall adjudication, admission, revocation, ledger reads,
	// policy/route reloads, trace and session control, lease planes. All gateway state.
	"/v1/fak/syscall":               epExemptNoDecode,
	"/v1/fak/cache/posture":         epExemptNoDecode,
	"/v1/fak/adjudicate":            epExemptNoDecode,
	"/v1/fak/admit":                 epExemptNoDecode,
	"/v1/fak/changes":               epExemptNoDecode,
	"/v1/fak/events":                epExemptNoDecode,
	"/v1/fak/vcache/score":          epExemptNoDecode,
	"/v1/fak/vcache/actions":        epExemptNoDecode,
	"/v1/fak/usage/cache-alignment": epExemptNoDecode,
	"/v1/fak/session-audit/actions": epExemptNoDecode,
	"/v1/fak/ctxvalue":              epExemptNoDecode,
	"/v1/fak/revoke":                epExemptNoDecode,
	"/v1/fak/context/change":        epExemptNoDecode,
	"/v1/fak/policy":                epExemptNoDecode,
	"/v1/fak/policy/reload":         epExemptNoDecode,
	"/v1/fak/route/reload":          epExemptNoDecode,
	"/v1/fak/trace/reset":           epExemptNoDecode,
	"/v1/fak/trace/":                epExemptNoDecode,
	"/v1/fak/session/changes":       epExemptNoDecode,
	"/v1/fak/session/":              epExemptNoDecode,
	"/v1/fak/fleet":                 epExemptNoDecode,
	"/v1/fak/sessions":              epExemptNoDecode,
	"/v1/fak/observation":           epExemptNoDecode,
	"/v1/fak/tasks":                 epExemptNoDecode,
	"/v1/fak/sharedtask/":           epExemptNoDecode,
	"/v1/fak/loops":                 epExemptNoDecode,
	"/v1/fak/account/rehome":        epExemptNoDecode,
	"/v1/fak/discovery/":            epExemptNoDecode,

	// The agent-runtime spine drives agent.RunGovernedArm over the server's planner —
	// a real decode, but an owned multi-turn one.
	"/v1/fak/agent/sessions": epExemptOwnedLoop,

	"/v1/models":                epExemptNoDecode,
	"/v1/control/config":        epExemptNoDecode,
	"/v1/fak/control/config":    epExemptNoDecode,
	"/v1/control/apply":         epExemptNoDecode,
	"/v1/fak/control/apply":     epExemptNoDecode,
	"/v1/control/events":        epExemptNoDecode,
	"/v1/fak/control/events":    epExemptNoDecode,
	"/v1/control/telemetry":     epExemptNoDecode,
	"/v1/fak/control/telemetry": epExemptNoDecode,
	"/v1/leases":                epExemptNoDecode,
	"/v1/leases/":               epExemptNoDecode,
	"/v1/sessions":              epExemptNoDecode,
	"/mcp":                      epExemptNoDecode,
	"/healthz":                  epExemptNoDecode,
	"/metrics":                  epExemptNoDecode,
	"/debug/vars":               epExemptNoDecode,
	"/debug/guard-audit":        epExemptNoDecode,
}

// epFanoutCoveredPatterns is the set of routeTable() patterns epFanoutProbes exercises.
func epFanoutCoveredPatterns() map[string]bool {
	covered := make(map[string]bool, len(epFanoutProbes))
	for _, p := range epFanoutProbes {
		covered[p.pattern] = true
	}
	return covered
}

// TestEPFanoutCoverageClassifiesEveryServedRoute is the drift half of the gate: every
// route the server registers is either probed as covered or named as exempt, exactly one
// of the two, and no classification survives the route it described.
//
// This is what makes the behavioral test below non-circular. Without it, a new served
// decode path could be added with no probe and no exemption and the suite would stay
// green by simply not looking at it — which is precisely how #5528 happened.
func TestEPFanoutCoverageClassifiesEveryServedRoute(t *testing.T) {
	covered := epFanoutCoveredPatterns()
	served := make(map[string]bool)
	for _, rt := range (&Server{}).routeTable() {
		served[rt.pattern] = true
		_, exempt := epFanoutExemptRoutes[rt.pattern]
		switch {
		case covered[rt.pattern] && exempt:
			t.Errorf("route %q is classified BOTH covered and exempt — pick one", rt.pattern)
		case !covered[rt.pattern] && !exempt:
			t.Errorf("route %q is served by routeTable() but unclassified for EP follower fanout: either add an epFanoutProbe (and call startEPFanoutFollowers from its handler) or name it in epFanoutExemptRoutes with a written reason (#5528)", rt.pattern)
		}
	}
	for pattern, reason := range epFanoutExemptRoutes {
		if !served[pattern] {
			t.Errorf("epFanoutExemptRoutes names %q, which routeTable() no longer serves — drop the stale exemption", pattern)
		}
		if strings.TrimSpace(reason) == "" {
			t.Errorf("route %q is exempt with an empty reason — an exemption without a reason is a silent drop", pattern)
		}
	}
	for _, p := range epFanoutProbes {
		if !served[p.pattern] {
			t.Errorf("epFanoutProbes drives %q, which routeTable() no longer serves — drop the stale probe", p.pattern)
		}
	}
}

// TestEveryEPFanoutCoveredRouteReleasesFollowerRanks is the behavioral half, and the
// assertion #5528 exists to add: drive every covered arm against a real follower rank and
// require that the rank was contacted, on the route the front rank is serving, with the
// identical body and the recursion guard set.
//
// Against the unwired handlers this fails on /v1/messages, /v1/responses and /v1beta/ —
// no follower is contacted at all on those wires.
func TestEveryEPFanoutCoveredRouteReleasesFollowerRanks(t *testing.T) {
	for _, probe := range epFanoutProbes {
		t.Run(probe.pattern+" "+probe.name, func(t *testing.T) {
			calls := startRecordingFollowerRank(t)
			ts := epFanoutProbeServer(t)

			resp, err := http.Post(ts.URL+probe.urlPath, "application/json", strings.NewReader(probe.body))
			if err != nil {
				t.Fatalf("front-rank request to %s failed: %v", probe.urlPath, err)
			}
			_, _ = io.Copy(io.Discard, resp.Body)
			resp.Body.Close()

			// The front rank's fanout wait is a deferred call, so by the time the client's
			// own response has fully landed the mirrored request has already been made and
			// answered. The budget covers scheduling only.
			select {
			case got := <-calls:
				if got.path != probe.wantRoute {
					t.Fatalf("follower rank was asked for %q, want %q — a follower asked for a wire whose request schema differs from the one the front rank is serving answers 400 and never joins the decode (#5523)", got.path, probe.wantRoute)
				}
				if got.header != "1" {
					t.Fatalf("follower rank %s = %q, want 1 — without it a follower that is itself a fak front rank fans out again", epFollowerHeader, got.header)
				}
				if got.body != probe.body {
					t.Fatalf("follower rank body = %q, want the mirrored request %q — a follower running a different forward pass is not in the same collective", got.body, probe.body)
				}
			case <-time.After(epStallBudget):
				t.Fatalf("no follower rank was released into the decode served by %s (%s arm): the front rank enters a multi-rank collective alone (#5528)", probe.pattern, probe.name)
			}
		})
	}
}

// TestEPFanoutCoveredRoutesDoNotFanOutAgain pins the recursion guard on every covered
// wire at once. A follower rank that is itself a fak gateway serves the mirrored request
// through these same handlers; if any of them fanned out in turn, one client request
// would become an exponential broadcast across the fleet.
//
// The guard is startEPFanoutFollowers' first statement and is route-independent, so what
// this asserts per wire is that the handler reaches it — with followers CONFIGURED, since
// the unconfigured no-op path would make the assertion vacuous.
func TestEPFanoutCoveredRoutesDoNotFanOutAgain(t *testing.T) {
	for _, probe := range epFanoutProbes {
		t.Run(probe.pattern+" "+probe.name, func(t *testing.T) {
			calls := startRecordingFollowerRank(t)
			ts := epFanoutProbeServer(t)

			req, err := http.NewRequest(http.MethodPost, ts.URL+probe.urlPath, strings.NewReader(probe.body))
			if err != nil {
				t.Fatal(err)
			}
			req.Header.Set("Content-Type", "application/json")
			req.Header.Set(epFollowerHeader, "1")
			resp, err := http.DefaultClient.Do(req)
			if err != nil {
				t.Fatalf("mirrored request to %s failed: %v", probe.urlPath, err)
			}
			_, _ = io.Copy(io.Discard, resp.Body)
			resp.Body.Close()

			// A BOUNDED negative window, not a bare default: a recursive mirror could still
			// be in flight when the client's own response has landed. Waiting is what makes
			// the absence real.
			select {
			case got := <-calls:
				t.Fatalf("a follower's own %s request fanned out again to %q — one client request would become an exponential EP broadcast", probe.pattern, got.path)
			case <-time.After(time.Second):
			}
		})
	}
}

// epFanoutProbeServer stands up the front rank the probes drive: the standard test server
// with a planner that answers immediately, so what a probe measures is the fanout and not
// a decode. The ordering witnesses in ep_wire_fanout_test.go use a planner that PARKS in a
// modeled collective instead, which is what proves the release happens BEFORE the decode.
func epFanoutProbeServer(t *testing.T) *httptest.Server {
	t.Helper()
	srv := newTestServer(t)
	srv.planner = stubPlanner{comp: &agent.Completion{
		Message:      agent.Message{Role: agent.RoleAssistant, Content: "front rank decoded"},
		FinishReason: "stop",
		Model:        "test-model",
		Usage:        agent.Usage{PromptTokens: 6, CompletionTokens: 4, TotalTokens: 10},
	}}
	ts := httptest.NewServer(srv.Handler())
	t.Cleanup(ts.Close)
	return ts
}
