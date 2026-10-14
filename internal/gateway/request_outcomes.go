package gateway

import (
	"context"
	"net/http"
	"net/url"
	"strings"
	"sync"
	"sync/atomic"
	"time"
	"unicode/utf8"

	"github.com/anthony-chaudhary/fak/internal/agent"
	"github.com/anthony-chaudhary/fak/pkg/traceabi"
)

const (
	requestOutcomeSchema        = "fak.request.outcome.v1"
	requestOutcomeLimit         = 4096
	requestOutcomeMaxAge        = 168 * time.Hour
	requestOutcomeIdentityLimit = 512
)

type requestOutcomeKey struct {
	owner, traceID, traceparent string
}

// Only scalar identity and timing survive a request. Never retain a request,
// completion, native decode summary, credential, error, or transcript here.
type requestOutcomeRecord struct {
	key               requestOutcomeKey
	sequence          uint64
	started, finished time.Time
	outcome, source   string
}

// This is a lossy, process-local view, not a retry-safety authority. Production
// keys include a fresh server-minted span and have exactly one terminal producer.
// First-terminal wins: duplicate offers never replace or refresh a retained row.
// Contention drops a NEW offer without affecting unrelated retained history.
type requestOutcomeStore struct {
	mu                           sync.Mutex
	rows                         map[requestOutcomeKey]requestOutcomeRecord
	offers                       atomic.Uint64
	dropped                      atomic.Uint64
	evicted, expired, duplicates uint64
}

type requestOutcomeAccounting struct {
	Offers     uint64 `json:"offers"`
	Dropped    uint64 `json:"dropped"`
	Evicted    uint64 `json:"evicted"`
	Expired    uint64 `json:"expired"`
	Duplicates uint64 `json:"duplicates"`
}

func validOutcomeIdentity(value string) bool {
	return value != "" && len(value) <= requestOutcomeIdentityLimit &&
		utf8.ValidString(value) && strings.TrimSpace(value) == value &&
		!strings.ContainsAny(value, "\r\n\x00")
}

func validOutcomeKey(key requestOutcomeKey) bool {
	if len(key.owner) > requestOutcomeIdentityLimit || !utf8.ValidString(key.owner) ||
		!validOutcomeIdentity(key.traceID) || len(key.traceparent) != 55 {
		return false
	}
	tc, err := parseTraceparent(key.traceparent)
	return err == nil && tc.String() == key.traceparent
}

func (s *requestOutcomeStore) expire(now time.Time) {
	for key, row := range s.rows {
		if !now.Before(row.finished.Add(requestOutcomeMaxAge)) {
			delete(s.rows, key)
			s.expired++
		}
	}
}

func (s *requestOutcomeStore) offer(row requestOutcomeRecord, now time.Time) {
	s.offers.Add(1)
	if !validOutcomeKey(row.key) || row.sequence == 0 || row.started.IsZero() ||
		row.finished.Before(row.started) || !now.Before(row.finished.Add(requestOutcomeMaxAge)) {
		s.dropped.Add(1)
		return
	}
	if !s.mu.TryLock() {
		s.dropped.Add(1)
		return
	}
	defer s.mu.Unlock()
	s.expire(now)
	if _, exists := s.rows[row.key]; exists {
		s.duplicates++
		return
	}
	if len(s.rows) == requestOutcomeLimit {
		var oldest requestOutcomeRecord
		for _, existing := range s.rows {
			if oldest.sequence == 0 || existing.finished.Before(oldest.finished) ||
				(existing.finished.Equal(oldest.finished) && existing.sequence < oldest.sequence) {
				oldest = existing
			}
		}
		delete(s.rows, oldest.key)
		s.evicted++
	}
	if s.rows == nil {
		s.rows = make(map[requestOutcomeKey]requestOutcomeRecord)
	}
	// Clone bounded strings rather than retaining a slice of a larger buffer.
	row.key = requestOutcomeKey{strings.Clone(row.key.owner), strings.Clone(row.key.traceID), strings.Clone(row.key.traceparent)}
	s.rows[row.key] = row
}

func (s *requestOutcomeStore) lookup(key requestOutcomeKey, now time.Time) (requestOutcomeRecord, bool, requestOutcomeAccounting) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.expire(now)
	row, ok := s.rows[key]
	return row, ok, requestOutcomeAccounting{
		Offers: s.offers.Load(), Dropped: s.dropped.Load(),
		Evicted: s.evicted, Expired: s.expired, Duplicates: s.duplicates,
	}
}

// Malformed query strings also select the mandatory-auth path. URL.Query would
// silently drop malformed identity parameters and restore the legacy live read.
func requestOutcomeQuery(r *http.Request) (url.Values, bool, error) {
	if r.URL.Path != "/v1/fak/observation/requests" {
		return nil, false, nil
	}
	values, err := url.ParseQuery(r.URL.RawQuery)
	_, trace := values["trace_id"]
	_, parent := values["traceparent"]
	return values, err != nil || trace || parent, err
}

func requiresRequestOutcomeAuth(r *http.Request) bool {
	_, selected, _ := requestOutcomeQuery(r)
	return selected
}

type requestOutcomeWireRecord struct {
	TraceID        string                  `json:"trace_id"`
	Traceparent    string                  `json:"traceparent"`
	Started        int64                   `json:"started_unix_nanos"`
	Finished       int64                   `json:"finished_unix_nanos"`
	Outcome        string                  `json:"outcome"`
	TerminalSource string                  `json:"terminal_source"`
	Causal         traceabi.CausalEnvelope `json:"causal"`
}

type requestOutcomeWire struct {
	Schema        string                    `json:"schema"`
	Availability  string                    `json:"availability"`
	Scope         string                    `json:"scope"`
	Persistence   string                    `json:"persistence"`
	MaxRecords    int                       `json:"max_records"`
	MaxAgeSeconds int64                     `json:"max_age_seconds"`
	Record        *requestOutcomeWireRecord `json:"record,omitempty"`
}

func (s *Server) handleRequestOutcome(w http.ResponseWriter, r *http.Request, values url.Values, queryErr error) {
	w.Header().Set("Cache-Control", "no-store")
	// Do not include raw query values in refusal messages.
	if queryErr != nil || len(values["trace_id"]) != 1 || len(values["traceparent"]) != 1 {
		writeErrCode(w, http.StatusBadRequest, "outcome_identity_invalid", "require exactly one trace_id and traceparent")
		return
	}
	key := requestOutcomeKey{principalFromContext(r.Context()), values.Get("trace_id"), values.Get("traceparent")}
	if !validOutcomeKey(key) {
		writeErrCode(w, http.StatusBadRequest, "outcome_identity_invalid", "require bounded canonical identities")
		return
	}
	out := requestOutcomeWire{
		Schema: requestOutcomeSchema, Availability: "not_found", Scope: "buffered_generation",
		Persistence: "memory_only", MaxRecords: requestOutcomeLimit,
		MaxAgeSeconds: int64(requestOutcomeMaxAge / time.Second),
	}
	if s.metrics != nil {
		row, ok, _ := s.metrics.requestOutcomes.lookup(key, time.Now())
		if ok {
			tc, _ := parseTraceparent(row.key.traceparent)
			out.Availability = "observed"
			out.Record = &requestOutcomeWireRecord{
				TraceID: row.key.traceID, Traceparent: row.key.traceparent,
				Started: row.started.UnixNano(), Finished: row.finished.UnixNano(),
				Outcome: row.outcome, TerminalSource: row.source,
				Causal: traceabi.CausalEnvelope{
					Schema: traceabi.Schema, TraceID: tc.TraceID, TraceFormat: traceabi.TraceW3C,
					SpanID: tc.ParentID, ParentStatus: traceabi.ParentUnknown,
					Sequence: row.sequence, LogicalID: row.key.traceparent,
					Operation: "legacy_buffered_generation", Phase: "terminal_observation",
					Correlations: []traceabi.Correlation{{Kind: traceabi.CorrelationProducerNative, Value: row.key.traceID}},
					SourceSchema: requestOutcomeSchema, SourceVersion: "1", MappingStatus: traceabi.MappingUnknown,
				},
			}
			writeJSON(w, http.StatusOK, out)
			return
		}
	}
	writeJSON(w, http.StatusNotFound, out)
}

// Capture the ACTUALLY selected planner before invoking completeServed. A clean
// return with request-owned NativeDecode is positive native generation evidence.
// Normalized finish text, custom/proxy summaries, errors and cancellation are not.
// The single finalizer is called just after settlement; its deferred fallback
// covers early exits/panics. Later socket failures/cancellation cannot replace it.
func (s *Server) beginRequestOutcome(ctx context.Context, owner, acceptedTrace string, headers http.Header) func(*agent.Completion, error) {
	selected, native := s.chatPlanner(ctx).(*agent.InKernelPlanner)
	key := requestOutcomeKey{owner, acceptedTrace, headers.Get(traceparentHeader)}
	valid := acceptedTrace == headers.Get(traceHeader)
	row := requestOutcomeRecord{key: key, sequence: inflightIDFromCtx(ctx), started: time.Now()}
	done := false
	return func(comp *agent.Completion, err error) {
		if done {
			return
		}
		done = true
		if s.metrics == nil {
			return
		}
		row.finished = time.Now()
		row.outcome, row.source = "completion_unknown", "served_call_unproven"
		if valid && native && selected != nil && err == nil && comp != nil && comp.NativeDecode != nil {
			switch comp.FinishReason {
			case "stop", "length", "tool_calls":
				row.outcome, row.source = "succeeded", "native_generate_return"
			}
		}
		if !valid {
			row.key.traceID = "" // admission/header disagreement must fail closed
		}
		s.metrics.requestOutcomes.offer(row, row.finished)
	}
}
