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
	"os"
	"path/filepath"
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

// --- Orphan reap for the per-session step-advice sidecars (issue #5353) ---
//
// CaptureAndWrite above rewrites one stepadvice-<id>.json per live trace every
// turn end, so a running trace owns exactly one file. But when the trace ends
// nothing deletes it: the file is now a disposable KB sidecar with no reader,
// and orphans accrete one per dead trace (193 seen on the live tree — pure count
// growth, the WIP-ref leak shape). These two best-effort helpers, kept beside the
// writer so the delete shape can never drift from the write shape, bound that
// growth. ReapClosedAdvice removes the CLOSING trace's own file on the clean-exit
// (SessionEnd) boundary; SweepStaleAdvice removes any sidecar a crashed trace
// left behind — one whose clean-exit delete never ran — once it is older than a
// grace floor, so a live trace's current-turn file is never taken.

// advicePrefix / adviceSuffix bracket a per-session sidecar's base name: the
// exact shape stepbaton.Path writes (stepadvice-<sanitized id>.json). Kept beside
// the sweep that globs for them so the match can never drift from the write path.
const advicePrefix = "stepadvice-"
const adviceSuffix = ".json"

// DefaultStaleFloor is the grace age below which SweepStaleAdvice will not touch a
// sidecar. It sits far above any plausible turn cadence (a live trace rewrites its
// file every turn end, seconds to minutes apart), so a current-turn file is never
// mistaken for an orphan; only a file untouched for a full day — necessarily from
// a trace whose clean-exit delete never ran — is swept.
const DefaultStaleFloor = 24 * time.Hour

// ReapClosedAdvice removes the closing trace's own stepadvice-<id>.json from dir —
// the clean-exit half of the reap, called on the SessionEnd boundary where the
// trace is gone and the file has no reader. It deletes exactly the one path
// stepbaton.Path would write for id. BEST-EFFORT: an already-absent file is a
// success (nil), and any other remove error is returned for the caller to log,
// never to fail a hook (mirroring the toolproc journal compactor's contract). An
// empty dir or id is a no-op.
func ReapClosedAdvice(dir, id string) error {
	if strings.TrimSpace(dir) == "" || strings.TrimSpace(id) == "" {
		return nil
	}
	if err := os.Remove(stepbaton.Path(dir, id)); err != nil && !os.IsNotExist(err) {
		return err
	}
	return nil
}

// SweepStaleAdvice removes every stepadvice-*.json in dir last modified before
// now-floor and returns the count removed — the crash-recovery half. A host that
// dies without a clean SessionEnd never runs ReapClosedAdvice, so its sidecar
// would leak forever; the age floor collects it while KEEPING any file a live
// trace rewrote within the floor (its current-turn file). BEST-EFFORT throughout:
// a missing dir is not an error (nothing to sweep), a stat or remove failure on
// one entry is skipped so a single stuck file never aborts the pass, a
// concurrent peer delete of the same orphan simply is not counted, and floor <= 0
// falls back to DefaultStaleFloor so a mis-wired zero can never sweep a live file.
func SweepStaleAdvice(dir string, floor time.Duration, now time.Time) (int, error) {
	if strings.TrimSpace(dir) == "" {
		return 0, nil
	}
	if floor <= 0 {
		floor = DefaultStaleFloor
	}
	entries, err := os.ReadDir(dir)
	if err != nil {
		if os.IsNotExist(err) {
			return 0, nil // no dir yet — nothing to sweep
		}
		return 0, err
	}
	cutoff := now.Add(-floor)
	removed := 0
	for _, e := range entries {
		if e.IsDir() {
			continue
		}
		name := e.Name()
		if !strings.HasPrefix(name, advicePrefix) || !strings.HasSuffix(name, adviceSuffix) {
			continue
		}
		info, err := e.Info()
		if err != nil {
			continue // races a peer write — keep it, a later pass retries
		}
		if info.ModTime().After(cutoff) {
			continue // younger than the floor — a live trace's current file
		}
		if err := os.Remove(filepath.Join(dir, name)); err == nil {
			removed++
		}
	}
	return removed, nil
}
