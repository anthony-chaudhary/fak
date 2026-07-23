package gateway

// session_ceiling.go — the deployment-boundary SESSION SATURATION signal (#3425,
// epic #3256, Workstream E: the all-in-one deployment). #3292 benches the per-agent
// governed OVERHEAD; this answers the operator's other two sizing questions: how many
// concurrent governed loops does this box hold, and how do I see it saturating BEFORE
// the in-flight loops degrade.
//
// An operator configures a ceiling — the maximum number of concurrent governed
// sessions the deployment holds — via the env var FAK_MAX_SESSIONS (0/unset =
// unbounded, the historical default). The runtime then:
//
//   1. Exposes live saturation vs that ceiling as a /metrics gauge
//      (fak_gateway_session_saturation, writeSessionSaturationMetrics) an
//      operator/autoscaler can act on. The live count is the SAME session-registry
//      fold /metrics already publishes as fak_sessions.
//   2. BACKPRESSURES a NEW session's admission past the ceiling with the closed
//      reason SESSION_CEILING_SATURATED (sessionCeilingRefusal, wired into
//      beginServedSessionTurn). It refuses the next NEW session — a trace already
//      resident in the registry is ALWAYS admitted, so an in-flight loop is never
//      sacrificed to admit a new one. This is the deployment-boundary projection of
//      the fleet's existing admission machinery; doctrine: no silent caps — when the
//      deployment bounds concurrency it SAYS so, with a reason, and fails closed.
//
// Inert by default (gen/next posture): a zero ceiling emits no gauge, refuses
// nothing, and leaves the request path byte-for-byte historical — the whole signal
// is gated behind an explicit operator ceiling. It is a coarse admission gate, not a
// hard mutex: two new sessions racing the last slot can both admit (the count is a
// read-time fold, not a reservation) — the intended semantics for a saturation
// backpressure, which shapes load rather than enforcing an exact cap.

import (
	"context"
	"fmt"
	"os"
	"strconv"
	"strings"
)

// ReasonSessionCeilingSaturated is the closed-vocabulary reason a new-session
// admission carries when the live governed-session count is at/over the operator's
// configured ceiling. Declared in dos.toml as [reasons.SESSION_CEILING_SATURATED];
// the binding is asserted by session_ceiling_test.go so dos_check_reason resolves it.
const ReasonSessionCeilingSaturated = "SESSION_CEILING_SATURATED"

// envMaxSessions names the operator-configured concurrent-governed-session ceiling.
// Documented in docs/fak/deployment-guide.md (the sizing model). 0/unset = unbounded.
const envMaxSessions = "FAK_MAX_SESSIONS"

// SessionSaturation is the deployment-level headroom readout: the live governed-
// session count against the configured ceiling. Bounded=false means no ceiling is
// configured (unbounded) — Ratio and Headroom are then not meaningful and marshal 0.
type SessionSaturation struct {
	Ceiling  int     `json:"ceiling"`
	Live     int     `json:"live"`
	Ratio    float64 `json:"ratio"`    // live/ceiling in [0,1+]; 0 when unbounded
	Headroom int     `json:"headroom"` // ceiling-live, floored at 0; 0 when unbounded
	Bounded  bool    `json:"bounded"`
}

// sessionSaturation folds a live count and a ceiling into the readout. ceiling<=0 is
// unbounded (Bounded=false). A negative live count is clamped to 0 (defensive).
func sessionSaturation(live, ceiling int) SessionSaturation {
	if live < 0 {
		live = 0
	}
	if ceiling <= 0 {
		return SessionSaturation{Live: live, Bounded: false}
	}
	headroom := ceiling - live
	if headroom < 0 {
		headroom = 0
	}
	return SessionSaturation{
		Ceiling:  ceiling,
		Live:     live,
		Ratio:    float64(live) / float64(ceiling),
		Headroom: headroom,
		Bounded:  true,
	}
}

// admitNewSession reports whether a NEW session may be admitted under this readout.
// resident=true (the trace is already a live governed session) always admits — an
// in-flight loop is never refused. A new trace is refused once live >= ceiling.
func (sat SessionSaturation) admitNewSession(resident bool) bool {
	if !sat.Bounded || resident {
		return true
	}
	return sat.Live < sat.Ceiling
}

// configuredSessionCeiling reads the operator's FAK_MAX_SESSIONS ceiling. An unset,
// empty, non-numeric, or negative value is 0 (unbounded) — fail-open to the
// historical path, never a surprise cap from a typo.
func configuredSessionCeiling() int {
	raw := strings.TrimSpace(os.Getenv(envMaxSessions))
	if raw == "" {
		return 0
	}
	n, err := strconv.Atoi(raw)
	if err != nil || n < 0 {
		return 0
	}
	return n
}

// sessionSaturationNow reads the live governed-session count against the configured
// ceiling. It returns Bounded=false (and never scans the registry) when no ceiling is
// set, so the default serve path pays nothing.
func (s *Server) sessionSaturationNow(ctx context.Context) SessionSaturation {
	ceiling := configuredSessionCeiling()
	if ceiling <= 0 || s.listSessions == nil {
		return SessionSaturation{Bounded: false}
	}
	return sessionSaturation(len(s.listSessions(ctx)), ceiling)
}

// sessionCeilingRefusal reports the refusal state (if any) a NEW session's turn must
// carry because the deployment is at its configured session ceiling. nil ⇒ admit. A
// trace already resident in the live registry is never refused (no in-flight loop is
// sacrificed). No-op (nil) when no ceiling is configured or no registry is wired.
func (s *Server) sessionCeilingRefusal(ctx context.Context, trace string) *SessionState {
	ceiling := configuredSessionCeiling()
	if ceiling <= 0 || trace == "" || s.listSessions == nil {
		return nil
	}
	live := s.listSessions(ctx)
	resident := false
	for _, st := range live {
		if st.TraceID == trace {
			resident = true
			break
		}
	}
	if sessionSaturation(len(live), ceiling).admitNewSession(resident) {
		return nil
	}
	return &SessionState{TraceID: trace, Reason: ReasonSessionCeilingSaturated}
}

// writeSessionSaturationMetrics renders the deployment-boundary SESSION SATURATION
// signal (#3425): the live governed-session count against the operator's configured
// ceiling (FAK_MAX_SESSIONS), so an operator/autoscaler can act on headroom before
// the in-flight loops degrade. Absent entirely until a ceiling is configured (a zero
// ceiling is "unbounded" — the default serve path emits nothing here and pays no
// extra registry scan), the same quiet-until-armed shape writeSpendGovernorMetrics
// uses.
func (s *Server) writeSessionSaturationMetrics(b *strings.Builder) {
	sat := s.sessionSaturationNow(context.Background())
	if !sat.Bounded {
		return
	}
	writeHelpType(b, "fak_gateway_session_ceiling",
		"Configured maximum concurrent governed sessions the deployment holds (#3425, FAK_MAX_SESSIONS). Absent entirely when unbounded (no ceiling configured).",
		"gauge")
	fmt.Fprintf(b, "fak_gateway_session_ceiling %d\n", sat.Ceiling)
	writeHelpType(b, "fak_gateway_sessions_live",
		"Live governed sessions counted against the configured ceiling (read-time fold of the session registry, the same source as fak_sessions).",
		"gauge")
	fmt.Fprintf(b, "fak_gateway_sessions_live %d\n", sat.Live)
	writeHelpType(b, "fak_gateway_session_saturation",
		"Deployment session saturation: live governed sessions / configured ceiling, in [0,1+]. At/past 1.0 a NEW session is backpressured with SESSION_CEILING_SATURATED; in-flight loops are never sacrificed. Watch it to size a host and to scale out before saturation degrades the loops already in flight.",
		"gauge")
	fmt.Fprintf(b, "fak_gateway_session_saturation %s\n", promFloat(sat.Ratio))
}
