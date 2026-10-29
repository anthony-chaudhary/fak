package gateway

import (
	"encoding/json"
	"math"
	"net/http"
	"strconv"
	"strings"
	"sync"
	"sync/atomic"
	"time"

	"github.com/anthony-chaudhary/fak/internal/modelperfobs"
	"github.com/anthony-chaudhary/fak/internal/nativeperf"
	"github.com/anthony-chaudhary/fak/internal/perfledger"
)

// requestPhaseForwardPath names the receipt layout: engine-measured prefill and
// decode laid end to end, decode-window kernel/synchronization nested in decode.
const requestPhaseForwardPath = "request_engine_phases"

// requestPhaseRing is the bounded, drop-oldest ring of fak-request-phases/1
// envelopes served at /v1/fak/perf/phases. refused counts native rows whose
// receipt failed to build or validate; they are omitted, never emitted partial.
type requestPhaseRing struct {
	mu      sync.Mutex
	rows    []modelperfobs.RequestPhaseReceipt
	dropped bool
	refused atomic.Uint64
}

func (r *requestPhaseRing) add(row modelperfobs.RequestPhaseReceipt) {
	r.mu.Lock()
	r.rows = append(r.rows, row)
	if len(r.rows) > perfledger.RingCap {
		r.rows = append([]modelperfobs.RequestPhaseReceipt(nil), r.rows[len(r.rows)-perfledger.RingCap:]...)
		r.dropped = true
	}
	r.mu.Unlock()
}

func (r *requestPhaseRing) snapshot(n int) ([]modelperfobs.RequestPhaseReceipt, bool) {
	r.mu.Lock()
	defer r.mu.Unlock()
	start := 0
	if n > 0 && n < len(r.rows) {
		start = len(r.rows) - n
	}
	return append([]modelperfobs.RequestPhaseReceipt(nil), r.rows[start:]...), r.dropped
}

// recordRequestPhases books one committed native perf row as a full-request
// phase receipt. Non-native rows (no engine anatomy or no engine phase) are not
// receipt candidates and are skipped without counting a refusal.
func (m *gatewayMetrics) recordRequestPhases(rec perfledger.Record, decode nativeperf.PhaseReceipt, decodeOK bool) {
	if m == nil || rec.Error != "" || rec.Engine == nil || rec.NativeTiming == nil {
		return
	}
	row, ok := buildRequestPhaseReceipt(rec, decode, decodeOK)
	if !ok {
		m.requestPhases.refused.Add(1)
		return
	}
	m.requestPhases.add(row)
}

// buildRequestPhaseReceipt reconciles the row's engine-measured phases to a
// complete fak-native receipt. The receipt wall is the measured span only; the
// rest of the request wall (admission, tokenization, model load, output, gaps)
// has no measured offset at this seam and is reported as unattributed_ms rather
// than booked into a fabricated phase.
func buildRequestPhaseReceipt(rec perfledger.Record, decode nativeperf.PhaseReceipt, decodeOK bool) (modelperfobs.RequestPhaseReceipt, bool) {
	n := rec.NativeTiming
	if strings.TrimSpace(rec.RequestID) == "" || strings.TrimSpace(rec.Model) == "" {
		return modelperfobs.RequestPhaseReceipt{}, false
	}
	prefill, okP := msDuration(n.PrefillMS)
	dec, okD := msDuration(n.DecodeMS)
	e2e, okE := msDuration(rec.E2EMS)
	if !okP || !okD || !okE {
		return modelperfobs.RequestPhaseReceipt{}, false
	}
	wall := prefill + dec
	if wall <= 0 || wall > e2e {
		return modelperfobs.RequestPhaseReceipt{}, false
	}
	backend := strings.TrimSpace(rec.Identity.Backend)
	if backend == "" && decodeOK {
		backend = strings.TrimSpace(decode.Backend)
	}
	if backend == "" {
		backend = "fak"
	}
	recorder := nativeperf.NewPhaseRecorder("inkernel", backend, requestPhaseForwardPath, wall)
	recorder.SetFallbackActive(rec.PrefillRoute != nil && rec.PrefillRoute.FallbackActive)
	if prefill > 0 && recorder.Add(nativeperf.PhasePrefill, "", 0, prefill, nativeperf.WorkActive) != nil {
		return modelperfobs.RequestPhaseReceipt{}, false
	}
	if dec > 0 {
		if recorder.Add(nativeperf.PhaseDecode, "", prefill, wall, nativeperf.WorkActive) != nil {
			return modelperfobs.RequestPhaseReceipt{}, false
		}
		if decodeOK {
			kernel := exclusivePhaseWall(decode, nativeperf.PhaseKernel)
			syncWait := exclusivePhaseWall(decode, nativeperf.PhaseSynchronization)
			// The decode window and decode_ms are separate clocks; nest only a
			// split that fits, never stretch decode to hold it.
			if kernel+syncWait <= dec {
				if kernel > 0 && recorder.Add(nativeperf.PhaseKernel, nativeperf.PhaseDecode, prefill, prefill+kernel, nativeperf.WorkActive) != nil {
					return modelperfobs.RequestPhaseReceipt{}, false
				}
				if syncWait > 0 && recorder.Add(nativeperf.PhaseSynchronization, nativeperf.PhaseDecode, prefill+kernel, prefill+kernel+syncWait, nativeperf.WorkWait) != nil {
					return modelperfobs.RequestPhaseReceipt{}, false
				}
			}
		}
	}
	receipt, err := recorder.Finalize(0)
	if err != nil {
		return modelperfobs.RequestPhaseReceipt{}, false
	}
	unattributed := float64(e2e-wall) / float64(time.Millisecond)
	return modelperfobs.RequestPhaseReceipt{
		Schema:         modelperfobs.RequestPhaseSchema,
		RequestID:      rec.RequestID,
		Model:          rec.Model,
		Backend:        backend,
		Receipt:        receipt,
		UnattributedMS: &unattributed,
	}, true
}

func exclusivePhaseWall(r nativeperf.PhaseReceipt, phase nativeperf.Phase) time.Duration {
	for _, p := range r.Phases {
		if p.Phase == phase && p.Exclusive.Wall > 0 {
			return p.Exclusive.Wall
		}
	}
	return 0
}

// msDuration converts a nonnegative finite millisecond value; an absent (zero)
// phase is a zero duration.
func msDuration(ms float64) (time.Duration, bool) {
	if ms < 0 || math.IsNaN(ms) || math.IsInf(ms, 0) || ms > float64(math.MaxInt64)/float64(time.Millisecond) {
		return 0, false
	}
	return time.Duration(math.Round(ms * float64(time.Millisecond))), true
}

// handleFakPerfPhases serves GET /v1/fak/perf/phases: the retained
// fak-request-phases/1 envelopes as JSONL, oldest first, keyed by the same
// request_id the /v1/fak/perf/recent rows carry. It is the --phases input of
// `fak model-observe time-burndown`.
//
//	?n=<int> newest envelopes to return (default all retained)
func (s *Server) handleFakPerfPhases(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet && r.Method != http.MethodHead {
		w.Header().Set("Allow", "GET, HEAD")
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}
	n := 0
	if raw := strings.TrimSpace(r.URL.Query().Get("n")); raw != "" {
		v, err := strconv.Atoi(raw)
		if err != nil || v <= 0 {
			http.Error(w, "n must be a positive integer", http.StatusBadRequest)
			return
		}
		n = v
	}
	var rows []modelperfobs.RequestPhaseReceipt
	var dropped bool
	var refused uint64
	if s != nil && s.metrics != nil {
		rows, dropped = s.metrics.requestPhases.snapshot(n)
		refused = s.metrics.requestPhases.refused.Load()
	}
	w.Header().Set("Content-Type", "application/x-ndjson")
	w.Header().Set("X-Fak-Phases-Refused", strconv.FormatUint(refused, 10))
	w.Header().Set("X-Fak-Phases-Capped", strconv.FormatBool(dropped))
	w.WriteHeader(http.StatusOK)
	if r.Method == http.MethodHead {
		return
	}
	enc := json.NewEncoder(w)
	for _, row := range rows {
		if enc.Encode(row) != nil {
			return
		}
	}
}
