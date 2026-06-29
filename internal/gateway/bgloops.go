package gateway

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"os"
	"strconv"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/bgloop"
	"github.com/anthony-chaudhary/fak/internal/marketing"
	"github.com/anthony-chaudhary/fak/internal/scoreboard"
)

// defaultHeartbeatInterval is how often the built-in kernel heartbeat loop ticks. It
// is the always-on witness that the in-kernel loop runtime keeps progressing while
// `fak serve` is up; FAK_BGLOOP_HEARTBEAT_S overrides it (a value <= 0 disables the
// built-in loop, leaving an empty-but-live supervisor a host can register into).
const defaultHeartbeatInterval = 30 * time.Second

// newBgloopSupervisor builds the gateway's in-kernel background-loop supervisor and
// registers the built-in loops. It never returns nil. The loops are not running yet:
// Serve starts them on the lifecycle context (startLoops) and joins them on shutdown
// (stopLoops). This is the in-kernel RUNTIME complement to internal/loopmgr's durable
// ledger — the supervisor that actually keeps loops progressing, where loopmgr only
// records the events a producer emits. See docs/notes/LONG-RUNNING-AGENT-LOOPS.
func newBgloopSupervisor(s *Server) *bgloop.Supervisor {
	sup := bgloop.New()
	if iv := heartbeatInterval(); iv > 0 {
		_ = sup.Register(bgloop.Loop{
			Name:     "heartbeat",
			Interval: iv,
			// A pure liveness pulse: the climbing tick counter (and its
			// fak_bgloop_ticks_total metric / its /v1/fak/loops row) is the witness
			// that the kernel's loop runtime is alive without an external scheduler
			// firing it. Cheap and always safe; the real value is the runtime + the
			// registration seam other subsystems extend.
			Tick: func(context.Context) error { return nil },
		})
	}
	if iv := marketingLoopInterval(); iv > 0 {
		_ = sup.Register(bgloop.Loop{
			Name:     "marketing",
			Interval: iv,
			// The AUTO marketing surface: each tick reads the high-water mark, gathers
			// genuinely-new WITNESSED ships (hooks.StampOf trailer|direct), and posts one
			// digest to #marketing — idempotently, so a re-tick over the same commits is a
			// no-op. Opt-in (FAK_MARKETING_LOOP), inert until a channel/token resolve, and it
			// inherits backoff + the admit gate + /metrics + /v1/fak/loops from the supervisor.
			Tick: s.marketingTick,
		})
	}
	return sup
}

// marketingLoopInterval reads FAK_MARKETING_LOOP / FAK_MARKETING_LOOP_S: the loop is OFF
// unless FAK_MARKETING_LOOP is truthy (1/true/on), then ticks every FAK_MARKETING_LOOP_S
// seconds (default 1h). A non-positive _S also disables it. Off-by-default because a marketing
// post is an outward-facing side effect — a host opts in, it is never automatic on serve.
func marketingLoopInterval() time.Duration {
	switch strings.ToLower(strings.TrimSpace(os.Getenv("FAK_MARKETING_LOOP"))) {
	case "1", "true", "on", "yes":
	default:
		return 0
	}
	iv := time.Hour
	if v := strings.TrimSpace(os.Getenv("FAK_MARKETING_LOOP_S")); v != "" {
		if n, err := strconv.Atoi(v); err == nil {
			if n <= 0 {
				return 0
			}
			iv = time.Duration(n) * time.Second
		}
	}
	return iv
}

// marketingTick is the in-kernel marketing loop body: resolve the #marketing channel/token,
// build a dedupe-aware Poster, and run one idempotent marketing.Tick over the repo. An
// unresolved channel/token is NOT an error — the loop stays armed but inert (like an
// unconfigured scoreboard) until a host configures FAK_MARKETING_CHANNEL/TOKEN. A tick that
// finds no new witnessed ships is a clean no-op. The repo root is the gateway's working dir
// (where `fak serve` runs); the source label is "serve".
func (s *Server) marketingTick(ctx context.Context) error {
	ch := marketing.ResolveChannel()
	if ch == "" {
		return nil // inert until configured — not a failure
	}
	client, err := scoreboard.NewClient(marketing.ResolveToken())
	if err != nil {
		return nil // no token yet — inert, not a failure (a real token error surfaces on post)
	}
	opts := marketing.Opts{
		Root:   ".",
		Source: "serve",
		Poster: marketingLoopPoster{client: client, channel: ch},
	}
	res, err := opts.Tick(ctx, time.Now())
	if err != nil {
		return err // a real post/transport error -> the supervisor records it + backs off
	}
	if res.Status == "posted" {
		s.logf("fak marketing: posted %d witnessed ship(s) to #marketing ts=%s", res.NewShips, res.PostedTS)
	}
	return nil
}

// marketingLoopPoster adapts the scoreboard client to the marketing.Poster seam for the
// in-kernel loop (the gateway twin of cmd/fak's marketingPoster). It keys the dedupe-aware
// PostWithUpdate on the artifact's stable DedupeKey, the backstop behind the high-water CAS.
type marketingLoopPoster struct {
	client  *scoreboard.Client
	channel string
}

func (p marketingLoopPoster) PostArtifact(ctx context.Context, a marketing.Artifact) (string, error) {
	return p.client.PostWithUpdate(ctx, p.channel, scoreboard.Update{Title: a.DedupeKey}, a.Text(), a.Blocks())
}

// heartbeatInterval reads FAK_BGLOOP_HEARTBEAT_S (seconds); unset uses the default,
// a non-positive value disables the built-in heartbeat loop.
func heartbeatInterval() time.Duration {
	if v := strings.TrimSpace(os.Getenv("FAK_BGLOOP_HEARTBEAT_S")); v != "" {
		if n, err := strconv.Atoi(v); err == nil {
			if n <= 0 {
				return 0
			}
			return time.Duration(n) * time.Second
		}
	}
	return defaultHeartbeatInterval
}

// startLoops launches the supervised background loops on the serve lifecycle context.
// Called once from Serve, right after the listener is bound and the gateway is ready.
func (s *Server) startLoops(ctx context.Context) {
	if s.loops == nil {
		return
	}
	s.loops.Start(ctx)
}

// stopLoops cancels the background loops and joins their goroutines within a bounded
// window, so a wedged loop cannot block the gateway's own graceful shutdown — it logs
// a timeout naming the stuck loop instead of hanging.
func (s *Server) stopLoops() {
	if s.loops == nil {
		return
	}
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	if err := s.loops.Shutdown(ctx); err != nil {
		s.logf("fak bgloop: %v", err)
	}
}

// BgloopsResponse is the GET /v1/fak/loops body: a snapshot of every supervised
// in-kernel background loop. It is the stable wire shape the `fak bgloop status`
// client decodes (mirroring SessionListResponse for `fak ps`).
type BgloopsResponse struct {
	Schema string          `json:"schema"`
	Loops  []bgloop.Status `json:"loops"`
}

// handleFakLoops serves GET /v1/fak/loops — a JSON snapshot of every supervised
// in-kernel background loop and its live progress (state, ticks, errors, panics,
// restarts, last tick, next tick). It is the observability half of the loop control
// plane: the in-kernel runtime view that complements `fak loop status` (the durable
// loopmgr ledger of externally-scheduled fires).
func (s *Server) handleFakLoops(w http.ResponseWriter, r *http.Request) {
	if !requireMethod(w, r, http.MethodGet) {
		return
	}
	loops := []bgloop.Status{}
	if s.loops != nil {
		loops = s.loops.Snapshot()
	}
	w.Header().Set("Content-Type", "application/json")
	_ = json.NewEncoder(w).Encode(BgloopsResponse{Schema: "fak.bgloops.v1", Loops: loops})
}

// writeBgloopMetrics appends the in-kernel background-loop Prometheus family to the
// /metrics render: per-loop tick/error/panic/restart/pause counters, the last-tick
// timestamp gauge, and an up gauge (1 = idle/running/backoff/paused, 0 = stopped).
// Called from renderMetrics.
func (s *Server) writeBgloopMetrics(b *strings.Builder) {
	if s.loops == nil {
		return
	}
	snap := s.loops.Snapshot()
	writeHelpType(b, "fak_bgloop_registered", "In-kernel background loops registered with the supervisor.", "gauge")
	fmt.Fprintf(b, "fak_bgloop_registered %d\n", len(snap))
	if len(snap) == 0 {
		return
	}
	writeHelpType(b, "fak_bgloop_ticks_total", "Background-loop ticks that completed cleanly, by loop.", "counter")
	for _, st := range snap {
		fmt.Fprintf(b, "fak_bgloop_ticks_total{loop=\"%s\"} %d\n", promQuote(st.Name), st.Ticks)
	}
	writeHelpType(b, "fak_bgloop_errors_total", "Background-loop ticks that returned an error, by loop.", "counter")
	for _, st := range snap {
		fmt.Fprintf(b, "fak_bgloop_errors_total{loop=\"%s\"} %d\n", promQuote(st.Name), st.Errors)
	}
	writeHelpType(b, "fak_bgloop_panics_total", "Background-loop ticks that panicked and were recovered, by loop.", "counter")
	for _, st := range snap {
		fmt.Fprintf(b, "fak_bgloop_panics_total{loop=\"%s\"} %d\n", promQuote(st.Name), st.Panics)
	}
	writeHelpType(b, "fak_bgloop_restarts_total", "Background-loop backoff restarts (errors + panics), by loop.", "counter")
	for _, st := range snap {
		fmt.Fprintf(b, "fak_bgloop_restarts_total{loop=\"%s\"} %d\n", promQuote(st.Name), st.Restarts)
	}
	writeHelpType(b, "fak_bgloop_pauses_total", "Background-loop fires the admit gate refused, by loop.", "counter")
	for _, st := range snap {
		fmt.Fprintf(b, "fak_bgloop_pauses_total{loop=\"%s\"} %d\n", promQuote(st.Name), st.Pauses)
	}
	writeHelpType(b, "fak_bgloop_last_tick_timestamp_seconds", "Unix time of each background loop's last completed tick (0 if none yet).", "gauge")
	for _, st := range snap {
		var ts int64
		if !st.LastTickAt.IsZero() {
			ts = st.LastTickAt.Unix()
		}
		fmt.Fprintf(b, "fak_bgloop_last_tick_timestamp_seconds{loop=\"%s\"} %d\n", promQuote(st.Name), ts)
	}
	writeHelpType(b, "fak_bgloop_up", "Background-loop liveness: 1 when not stopped, 0 when stopped.", "gauge")
	for _, st := range snap {
		up := 1
		if st.State == bgloop.StateStopped {
			up = 0
		}
		fmt.Fprintf(b, "fak_bgloop_up{loop=\"%s\",state=\"%s\"} %d\n", promQuote(st.Name), promQuote(string(st.State)), up)
	}
}
