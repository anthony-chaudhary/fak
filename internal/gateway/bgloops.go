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
	return sup
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
