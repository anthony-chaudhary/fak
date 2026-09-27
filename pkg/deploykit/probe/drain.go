package probe

import (
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"math"
	"net/http"
	"strconv"
	"time"
)

// Drain statuses. Only DrainStatusDrained with zero active connections is a
// successful drain; the other two still arrive with HTTP 200.
const (
	DrainStatusDrained  = "drained"
	DrainStatusTimedOut = "timed_out"
	DrainStatusAborted  = "aborted"
)

// DefaultDrainTimeout is the drain window when the request names none, or
// names one that does not parse or is not positive.
const DefaultDrainTimeout = 15 * time.Second

// DrainResponse is the /v1/admin/drain body. Field names, order and tags match
// the AdminDrainResponse fak-server already writes, so the two encode
// byte-identically.
type DrainResponse struct {
	Status            string `json:"status"`
	ActiveConnections int64  `json:"active_connections"`
	ElapsedMS         int64  `json:"elapsed_ms"`
	// ObservationSnapshot is the bounded in-flight request snapshot taken at
	// drain time. It is omitted when absent, which keeps the legacy shape.
	ObservationSnapshot []json.RawMessage `json:"observation_snapshot,omitempty"`
}

// Drained reports whether the response is a complete drain: status "drained"
// and no connection still active.
func (d DrainResponse) Drained() bool {
	return d.Status == DrainStatusDrained && d.ActiveConnections == 0
}

// ErrNotDrained reports a well-formed drain response that is not a complete
// drain (timed out, aborted, connections still active, or a non-200 status).
var ErrNotDrained = errors.New("probe: drain incomplete")

// maxDrainResponse bounds a drain body. A response can carry up to 256
// observation rows, well past the legacy 64 KiB limit.
const maxDrainResponse = 16 << 20

// DecodeDrainResponse decodes and validates a /v1/admin/drain reply. It accepts
// only the closed response shape (unknown fields and trailing data are
// refused), and it returns ErrNotDrained unless the HTTP status is 200 and the
// body is a complete drain. The decoded response is returned whenever the body
// parsed, so a caller can still record a timed-out or aborted drain.
func DecodeDrainResponse(status int, body io.Reader) (DrainResponse, error) {
	if body == nil {
		return DrainResponse{}, errors.New("probe: drain response has no body")
	}
	var resp DrainResponse
	dec := json.NewDecoder(io.LimitReader(body, maxDrainResponse))
	dec.DisallowUnknownFields()
	if err := dec.Decode(&resp); err != nil {
		return DrainResponse{}, fmt.Errorf("probe: decode drain response: %w", err)
	}
	var trailing any
	if err := dec.Decode(&trailing); !errors.Is(err, io.EOF) {
		if err == nil {
			return DrainResponse{}, errors.New("probe: drain response contains multiple JSON values")
		}
		return DrainResponse{}, fmt.Errorf("probe: decode drain response trailing data: %w", err)
	}
	if status != http.StatusOK || !resp.Drained() {
		return resp, fmt.Errorf("%w (http=%d status=%q active=%d)", ErrNotDrained, status, clip(resp.Status), resp.ActiveConnections)
	}
	return resp, nil
}

// maxDrainRequest bounds the optional JSON request body of a drain call.
const maxDrainRequest = 64 << 10

// drainTimeout reads the drain window the way fak-server does: the "timeout"
// query parameter first (a Go duration or whole seconds), else a JSON body
// {"timeout": "<duration or seconds>"} or {"timeout_seconds": n}. Anything
// missing, unparseable or not positive falls back to DefaultDrainTimeout.
//
// One deliberate difference: a whole-second count too large for a Duration
// falls back to DefaultDrainTimeout, where fak-server multiplies without an
// overflow check and can end up with an arbitrary wrapped window.
func drainTimeout(r *http.Request) time.Duration {
	timeout := DefaultDrainTimeout
	if q := r.URL.Query().Get("timeout"); q != "" {
		timeout = parseDrainTimeout(q, timeout)
	} else if r.Body != nil {
		raw, _ := io.ReadAll(io.LimitReader(r.Body, maxDrainRequest))
		var req struct {
			Timeout        string `json:"timeout"`
			TimeoutSeconds int    `json:"timeout_seconds"`
		}
		if len(raw) > 0 && json.Unmarshal(raw, &req) == nil {
			if req.Timeout != "" {
				timeout = parseDrainTimeout(req.Timeout, timeout)
			} else if req.TimeoutSeconds > 0 {
				timeout = seconds(int64(req.TimeoutSeconds), timeout)
			}
		}
	}
	if timeout <= 0 {
		timeout = DefaultDrainTimeout
	}
	return timeout
}

func parseDrainTimeout(s string, fallback time.Duration) time.Duration {
	if d, err := time.ParseDuration(s); err == nil {
		return d
	}
	if n, err := strconv.ParseInt(s, 10, 64); err == nil {
		return seconds(n, fallback)
	}
	return fallback
}

// seconds converts whole seconds to a Duration, refusing values that would
// overflow it.
func seconds(n int64, fallback time.Duration) time.Duration {
	if n > math.MaxInt64/int64(time.Second) || n < math.MinInt64/int64(time.Second) {
		return fallback
	}
	return time.Duration(n) * time.Second
}
