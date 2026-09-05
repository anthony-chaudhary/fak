// Package stepbatoncapture captures live managed-context advice reports from
// the gateway and projects them into durable stepbaton stamps before trace rotation.
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

// maxCtxValueRespBytes bounds decoded response body size to 1 MiB.
const maxCtxValueRespBytes = 1 << 20

// defaultTimeout bounds HTTP requests when no client is provided.
const defaultTimeout = 2 * time.Second

// wireSnapshot and wireReport mirror gateway CtxValueSnapshot JSON wire fields.
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

// Options configures a managed-context report capture.
type Options struct {
	BaseURL   string
	Bearer    string
	TraceHint string
	SHA       string
	Dir       string
	SessionID string
	Client    *http.Client
}

// Capture retrieves the gateway context snapshot and projects the active session advice.
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

// CaptureAndWrite executes Capture and atomically persists the resulting stamp to disk.
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

// fetch performs a bounded GET request to retrieve the context snapshot.
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

// selectReport chooses the report matching hint, the sole session, or the most active.
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

// project constructs a stepbaton Stamp from a report and commit SHA.
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

// ctxvalueURL normalizes a gateway base URL to the ctxvalue endpoint path.
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

const (
	advicePrefix = "stepadvice-"
	adviceSuffix = ".json"
)

// DefaultStaleFloor is the default age threshold after which advice sidecars are swept.
const DefaultStaleFloor = 24 * time.Hour

// ReapClosedAdvice deletes the advice sidecar for a closed session.
func ReapClosedAdvice(dir, id string) error {
	if strings.TrimSpace(dir) == "" || strings.TrimSpace(id) == "" {
		return nil
	}
	if err := os.Remove(stepbaton.Path(dir, id)); err != nil && !os.IsNotExist(err) {
		return err
	}
	return nil
}

// SweepStaleAdvice removes advice sidecars modified prior to now minus floor duration.
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
