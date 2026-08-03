package gateway

// stream_progress_timeout_test.go — #5545: the streaming CONTENT-progress deadline (#5486)
// has to be reachable from a gateway.Config, and the operator's off switch has to really
// turn it off.
//
// #5486 shipped the deadline reading an environment variable; the relocation to the config
// surface (agent.DefaultStreamProgressTimeout + HTTPPlanner.StreamProgressTimeout, resolved
// by agent.(*HTTPPlanner).streamProgressWindow) closed internal/envconfiglint's CONFIG_NOT_ENV
// ratchet but left NOTHING setting the field — a hard 300s deadline with no operator route and
// no escape hatch. These cases pin both halves of the route: the Config value lands on every
// proxy planner (newConfiguredHTTPPlanner, the one place a Config-derived planner knob is set),
// and the value the planner ends up with is the window the stall reader actually arms.

import (
	"context"
	"errors"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/agent"
)

// streamProgressOff is the Config encoding for "no content-progress deadline": a NEGATIVE
// duration, which streamProgressWindow resolves to a disabled window. It mirrors
// cmd/fak/serve.go's streamProgressTimeoutOff, which is what `--stream-progress-timeout 0`
// (the house off spelling) is translated into at the front door.
const streamProgressOff = -1 * time.Second

// TestNewConfiguredHTTPPlannerCarriesStreamProgressTimeout pins the threading seam itself:
// the Config value rides onto the planner VERBATIM, in the planner field's own encoding, so
// zero keeps meaning "unconfigured, take agent.DefaultStreamProgressTimeout" and the negative
// off switch survives the hop instead of being normalized away into a real window.
func TestNewConfiguredHTTPPlannerCarriesStreamProgressTimeout(t *testing.T) {
	for _, c := range []struct {
		name string
		set  time.Duration
		want time.Duration
	}{
		{"an unset Config leaves the planner's zero (the agent default)", 0, 0},
		{"an in-band window rides through verbatim", 45 * time.Second, 45 * time.Second},
		{"the off switch keeps its negative encoding", streamProgressOff, streamProgressOff},
		{"an out-of-band window is NOT clamped here (the agent resolver decides)", 601 * time.Second, 601 * time.Second},
	} {
		t.Run(c.name, func(t *testing.T) {
			p, err := newConfiguredHTTPPlanner(Config{
				Provider:              "openai",
				APIKey:                "test-key",
				StreamProgressTimeout: c.set,
			}, "m", "https://example.invalid")
			if err != nil {
				t.Fatalf("newConfiguredHTTPPlanner: %v", err)
			}
			if p.StreamProgressTimeout != c.want {
				t.Fatalf("planner StreamProgressTimeout = %s, want %s", p.StreamProgressTimeout, c.want)
			}
		})
	}
}

// TestNewProxyPlannerPropagatesStreamProgressTimeout is the same fact one level up, on the
// path serve actually takes: a lone upstream, and every member of a replica fleet (the
// per-replica loop is the second newConfiguredHTTPPlanner call site, and a knob that reached
// only the lone-upstream planner would be silently absent from a fleet).
func TestNewProxyPlannerPropagatesStreamProgressTimeout(t *testing.T) {
	cfg := Config{Provider: "openai", APIKey: "test-key", StreamProgressTimeout: streamProgressOff}

	lone, err := newProxyPlanner(cfg, "m", []string{"https://a.example"})
	if err != nil {
		t.Fatalf("newProxyPlanner (lone): %v", err)
	}
	hp, ok := lone.(*agent.HTTPPlanner)
	if !ok {
		t.Fatalf("planner = %T, want *agent.HTTPPlanner", lone)
	}
	if hp.StreamProgressTimeout != streamProgressOff {
		t.Fatalf("lone-upstream planner StreamProgressTimeout = %s, want %s", hp.StreamProgressTimeout, streamProgressOff)
	}

	fleet, err := newProxyPlanner(cfg, "m", []string{"https://a.example", "https://b.example"})
	if err != nil {
		t.Fatalf("newProxyPlanner (fleet): %v", err)
	}
	if _, ok := fleet.(*agent.HTTPPlanner); ok {
		t.Fatal("two base URLs must build a replica router, not a bare planner")
	}
}

// TestStreamProgressTimeoutFromConfigReachesTheStallReader is the end-to-end witness, and the
// only one that can tell a threaded knob from a threaded-looking one: it drives a real
// keepalive-only upstream through a planner built from a gateway.Config and reads back WHICH
// window the stall reader armed.
//
//   - The ARMED arm proves the Config value reaches streamProgressWindow: the turn ends as a
//     no-progress stall carrying the configured 6s window. A Config that never reached the
//     planner would leave the 300s default and this arm would still be streaming keepalives
//     when the test gave up.
//   - The OFF arm proves the escape hatch: the same warm-but-unadvancing upstream, with the
//     negative off encoding, runs PAST the armed arm's window and completes with its content.
//
// Between them the chain is closed. (The off arm cannot distinguish "disabled" from "some
// window longer than this test runs" by wall clock alone — nothing can, short of waiting out
// the 300s default. What pins that last link is the resolver contract itself, which
// internal/agent's TestStreamProgressWindowResolvesTheConfigField holds: a negative
// StreamProgressTimeout resolves to a zero window, and newStallReader arms no progress timer
// for a zero window.)
func TestStreamProgressTimeoutFromConfigReachesTheStallReader(t *testing.T) {
	// Lower the IDLE (inter-byte) deadline: newStallReader raises a progress window shorter
	// than the idle window up to it, so with the 60s default idle window a 6s progress window
	// would silently become 60s and this test would measure nothing.
	t.Setenv("FAK_STREAM_STALL_TIMEOUT_S", "5")
	const armedWindow = 6 * time.Second

	type result struct {
		comp    *agent.Completion
		err     error
		elapsed time.Duration
	}
	run := func(url string, window time.Duration, out chan<- result) {
		p, err := newConfiguredHTTPPlanner(Config{
			Provider:              "openai",
			APIKey:                "test-key",
			StreamProgressTimeout: window,
		}, "m", url)
		if err != nil {
			out <- result{err: err}
			return
		}
		start := time.Now()
		comp, cerr := p.CompleteStream(context.Background(), func(string) error { return nil },
			[]agent.Message{{Role: agent.RoleUser, Content: "hi"}}, nil)
		out <- result{comp: comp, err: cerr, elapsed: time.Since(start)}
	}

	// The armed upstream never finishes on its own inside this test; the deadline is what ends
	// the turn. The off upstream stays warm well past the armed arm's window and then finishes.
	armedSrv := warmSSEUpstream(t, time.Minute)
	offSrv := warmSSEUpstream(t, 8*time.Second)

	armedCh, offCh := make(chan result, 1), make(chan result, 1)
	go run(armedSrv.URL, armedWindow, armedCh)
	go run(offSrv.URL, streamProgressOff, offCh)

	await := func(name string, ch <-chan result) result {
		t.Helper()
		select {
		case r := <-ch:
			return r
		case <-time.After(90 * time.Second):
			t.Fatalf("%s arm: CompleteStream never returned — a keepalive-only upstream is still riding the whole-request ceiling", name)
			return result{}
		}
	}

	armed := await("armed", armedCh)
	var stalled *agent.UpstreamStalledError
	if !errors.As(armed.err, &stalled) {
		t.Fatalf("armed arm: err = %v (%T), want *agent.UpstreamStalledError — the Config window never reached the stall reader", armed.err, armed.err)
	}
	if !errors.Is(armed.err, agent.ErrUpstreamStalled) {
		t.Fatalf("armed arm: errors.Is(err, ErrUpstreamStalled) = false, err = %v", armed.err)
	}
	// The window the error reports IS streamProgressWindow's answer for this Config.
	if stalled.Idle != armedWindow {
		t.Fatalf("armed arm: stalled window = %s, want the configured %s (the deadline that fired was not the one gateway.Config asked for)", stalled.Idle, armedWindow)
	}
	if !strings.Contains(stalled.Error(), "no content progress") {
		t.Fatalf("armed arm: %q must name the no-progress cause, not report a silent upstream", stalled.Error())
	}
	if armed.elapsed < armedWindow {
		t.Fatalf("armed arm: returned after %s, sooner than the %s window — something other than the progress deadline ended the turn", armed.elapsed, armedWindow)
	}

	off := await("off", offCh)
	if off.err != nil {
		t.Fatalf("off arm: a disabled progress deadline still cut the turn: %v", off.err)
	}
	if off.comp == nil || off.comp.Message.Content != "Hello" {
		t.Fatalf("off arm: content = %+v, want the completed \"Hello\"", off.comp)
	}
	if off.elapsed <= armedWindow {
		t.Fatalf("off arm: returned in %s, inside the %s window the armed arm was cut at — it outran nothing and proves nothing", off.elapsed, armedWindow)
	}
}

// warmSSEUpstream is the stalled-but-WARM upstream of #5486: it opens a healthy OpenAI-wire
// stream, then emits nothing but keepalive comments and empty-delta chunks — bytes that re-arm
// the inter-byte deadline while advancing the turn not at all — for warm, and only then
// finishes the turn. An idle deadline alone can never fire against it.
func warmSSEUpstream(t *testing.T, warm time.Duration) *httptest.Server {
	t.Helper()
	const opener = "data: {\"choices\":[{\"delta\":{\"role\":\"assistant\"},\"finish_reason\":null}]}\n\n" +
		"data: {\"choices\":[{\"delta\":{\"content\":\"Hel\"},\"finish_reason\":null}]}\n\n"
	const keepalive = ": keepalive\n\n" +
		"data: {\"choices\":[{\"delta\":{},\"finish_reason\":null}]}\n\n"
	const finish = "data: {\"choices\":[{\"delta\":{\"content\":\"lo\"},\"finish_reason\":\"stop\"}]}\n\n" +
		"data: [DONE]\n\n"

	release := make(chan struct{})
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		w.WriteHeader(http.StatusOK)
		f, _ := w.(http.Flusher)
		send := func(s string) bool {
			if _, err := io.WriteString(w, s); err != nil {
				return false
			}
			if f != nil {
				f.Flush()
			}
			return true
		}
		if !send(opener) {
			return
		}
		done := time.After(warm)
		for {
			select {
			case <-r.Context().Done(): // the deadline tripped and the stall reader closed the body
				return
			case <-release: // test over
				return
			case <-done:
				send(finish)
				return
			case <-time.After(250 * time.Millisecond):
			}
			if !send(keepalive) {
				return
			}
		}
	}))
	t.Cleanup(func() {
		close(release)
		srv.Close()
	})
	return srv
}
