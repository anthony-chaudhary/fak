// Package stepbatoncapture is the capture arm of the pre-resume step-advice stamp:
// the LIVE-side producer that reads the gateway's managed-context report while the
// trace is still alive and projects it into a durable stepbaton.Stamp.
//
// It is the piece internal/stepbaton deliberately left out (see that package's SCOPE
// note): stepbaton takes plain scalars and never imports the gateway, so the projection
// from a live gateway report has to live somewhere else. It lives here — a small package
// of its own — rather than inline in the guard hook for one reason: testability. The
// guard's cmd/fak is a large main binary, so a fetch+select+project+write folded into a
// hook could only ever be exercised through the whole hook; extracted here, the
// projection is a pure, httptest-exercised unit and the guard-hook seam shrinks to a
// single fail-open call.
//
// The read side is the gateway's GET /v1/fak/ctxvalue (internal/gateway ctxvalue.go,
// CtxValueSnapshot). This package decodes only the handful of wire fields it projects,
// into LOCAL structs — not the gateway types — so, like stepbaton, it carries no gateway
// dependency and cannot be poisoned by an unrelated gateway change: it is decoding a
// stable wire schema (fak-ctxvalue-report/1), not binding a Go type.
//
// Everything here is fail-open by construction. A capture runs inline on a guard
// boundary (a Stop or PreCompact hook) that must never be gated on it: "nothing to carry"
// (no gateway configured, or an empty snapshot) is a decidable non-error, and a genuine
// failure (unreachable gateway, non-2xx, undecodable body) is returned for the caller to
// log and move past — it can never change a hook's exit code.
package stepbatoncapture

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/stepbaton"
)

// maxCtxValueRespBytes bounds the decoded body — a misbehaving or compromised gateway
// must not stream an unbounded 200 into the hook's memory. It sits well above a realistic
// multi-session snapshot (the same posture session_cmd's req uses for its success body).
const maxCtxValueRespBytes = 1 << 20 // 1 MiB

// defaultTimeout bounds the whole GET when the caller passes no client of its own. The
// capture runs inline on a latency-sensitive hook boundary, so it must never hang the
// resume/turn-end path waiting on a slow or wedged gateway.
const defaultTimeout = 2 * time.Second

// wireSnapshot / wireReport mirror ONLY the fields this package projects out of the
// gateway's CtxValueSnapshot wire body (internal/gateway ctxvalue.go). Kept local — not
// imported — so the package stays gateway-free; the json tags are the load-bearing
// contract and match the gateway's struct tags verbatim.
type wireSnapshot struct {
	Sessions []wireReport `json:"sessions"`
}

type wireReport struct {
	TraceID string `json:"trace_id"`
	Tokens  struct {
		ResidentTokens int `json:"resident_tokens"`
		BudgetTokens   int `json:"budget_tokens"`
	} `json:"tokens"`
	Turns struct {
		TurnsObserved int `json:"turns_observed"`
	} `json:"turns"`
	Session struct {
		Phase string `json:"phase"`
	} `json:"session"`
	StepAdvice struct {
		StepClass string `json:"step_class"`
		Basis     string `json:"basis"`
		Reason    string `json:"reason"`
	} `json:"step_advice"`
}

// Options are the inputs to a single capture.
type Options struct {
	// BaseURL is the gateway base the hook already resolves (from ANTHROPIC_BASE_URL).
	// ctxvalueURL is forgiving about its shape — a bare base, a .../v1 base, or the
	// .../metrics scrape URL the hook derives all normalize to the ctxvalue read URL.
	BaseURL string
	// Bearer is an optional Authorization bearer; "" sends no auth header (the localhost
	// guard gateway serves ctxvalue unauthenticated, matching the /metrics scrape).
	Bearer string
	// TraceHint is the session id the hook read from stdin, preferred when several
	// sessions are live. "" falls back to the sole/most-live session (see selectReport).
	TraceHint string
	// SHA is the git commit observed at capture — the successor's anchor. "" is allowed.
	SHA string
	// Dir is the durable per-session directory (the guard's guardAuditDir).
	Dir string
	// SessionID keys the per-session stamp file (stepbaton.Path); the hook's stdin id.
	SessionID string
	// Client is an optional bounded HTTP client; nil uses a defaultTimeout client.
	Client *http.Client
}

// Capture fetches the live managed-context snapshot, selects the most-live session, and
// projects it into a stamp. ok is false with a NIL error when there is nothing live to
// stamp — no gateway configured, or the snapshot carries no session — so the caller
// treats "nothing to carry" the fail-open way the gateway itself treats an unknown trace.
// A non-nil error is a genuine failure (unreachable gateway, non-2xx, undecodable body).
func Capture(ctx context.Context, o Options) (stepbaton.Stamp, bool, error) {
	url := ctxvalueURL(o.BaseURL)
	if url == "" {
		return stepbaton.Stamp{}, false, nil // no gateway configured — fail open
	}
	snap, err := fetch(ctx, o.Client, url, o.Bearer)
	if err != nil {
		return stepbaton.Stamp{}, false, err
	}
	rep, ok := selectReport(snap, o.TraceHint)
	if !ok {
		return stepbaton.Stamp{}, false, nil // no live session — nothing to carry
	}
	return project(rep, o.SHA), true, nil
}

// CaptureAndWrite is Capture plus a durable, atomic write to the per-session stamp path.
// It writes UNCONDITIONALLY whenever a live session is found — including the any/unknown
// classes ShouldCarry() will later suppress — because the file is the "last live
// decision": overwriting it on every boundary is what keeps a stale checkpoint from
// outliving the pressure that produced it. When there is nothing live to stamp it leaves
// any prior stamp untouched, so a one-off unreachable gateway never erases a good stamp.
func CaptureAndWrite(ctx context.Context, o Options) (stepbaton.Stamp, bool, error) {
	stamp, ok, err := Capture(ctx, o)
	if err != nil || !ok {
		return stamp, ok, err
	}
	if err := stepbaton.Write(stepbaton.Path(o.Dir, o.SessionID), stamp); err != nil {
		return stamp, false, err
	}
	return stamp, true, nil
}

// fetch GETs the ctxvalue read URL and decodes the bounded body. It asks for ALL sessions
// (no ?trace= filter) so selectReport can fall back to the sole/most-live session even
// when the hook's stdin session id does not equal the gateway's trace key — a server-side
// filter to an unknown trace would return an empty list and lose that fallback.
func fetch(ctx context.Context, client *http.Client, url, bearer string) (wireSnapshot, error) {
	if client == nil {
		client = &http.Client{Timeout: defaultTimeout}
	}
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, url, nil)
	if err != nil {
		return wireSnapshot{}, fmt.Errorf("stepbatoncapture: build request: %w", err)
	}
	if b := strings.TrimSpace(bearer); b != "" {
		req.Header.Set("Authorization", "Bearer "+b)
	}
	resp, err := client.Do(req)
	if err != nil {
		return wireSnapshot{}, fmt.Errorf("stepbatoncapture: GET %s: %w", url, err)
	}
	defer resp.Body.Close()
	if resp.StatusCode/100 != 2 {
		return wireSnapshot{}, fmt.Errorf("stepbatoncapture: GET %s: status %d", url, resp.StatusCode)
	}
	var snap wireSnapshot
	if err := json.NewDecoder(io.LimitReader(resp.Body, maxCtxValueRespBytes)).Decode(&snap); err != nil {
		return wireSnapshot{}, fmt.Errorf("stepbatoncapture: decode: %w", err)
	}
	return snap, nil
}

// selectReport chooses the report to stamp:
//
//  1. an exact trace_id == hint match (the caller knows its own trace), else
//  2. the sole session when exactly one is live (the common `fak guard` case), else
//  3. the session with the most observed turns (the most-live of several), else
//  4. nothing (an empty snapshot) — ok=false.
func selectReport(snap wireSnapshot, hint string) (wireReport, bool) {
	if hint = strings.TrimSpace(hint); hint != "" {
		for _, r := range snap.Sessions {
			if r.TraceID == hint {
				return r, true
			}
		}
	}
	switch len(snap.Sessions) {
	case 0:
		return wireReport{}, false
	case 1:
		return snap.Sessions[0], true
	}
	best := snap.Sessions[0]
	for _, r := range snap.Sessions[1:] {
		if r.Turns.TurnsObserved > best.Turns.TurnsObserved {
			best = r
		}
	}
	return best, true
}

// project maps a chosen wire report + capture SHA into a stamp. The step class is
// normalized fail-closed inside stepbaton.New, so an off-vocabulary class from a future
// gateway becomes StepUnknown rather than a confident wrong carry.
func project(r wireReport, sha string) stepbaton.Stamp {
	return stepbaton.New(
		r.TraceID,
		r.StepAdvice.StepClass,
		r.StepAdvice.Basis,
		r.StepAdvice.Reason,
		r.Session.Phase,
		sha,
		r.Tokens.ResidentTokens,
		r.Tokens.BudgetTokens,
	)
}

// ctxvalueURL normalizes a gateway base into the ctxvalue read URL. It is deliberately
// forgiving about what the guard hook hands it: the hook resolves ANTHROPIC_BASE_URL into
// a `/metrics` scrape URL for its own posture read, so this accepts a bare base, a
// `.../v1` base, or that `.../metrics` URL and lands them all on `<base>/v1/fak/ctxvalue`.
// An empty base returns "" so the caller fails open (no gateway configured).
func ctxvalueURL(base string) string {
	base = strings.TrimRight(strings.TrimSpace(base), "/")
	base = strings.TrimSuffix(base, "/metrics")
	base = strings.TrimRight(base, "/")
	base = strings.TrimSuffix(base, "/v1")
	base = strings.TrimRight(base, "/")
	if base == "" {
		return ""
	}
	return base + "/v1/fak/ctxvalue"
}
