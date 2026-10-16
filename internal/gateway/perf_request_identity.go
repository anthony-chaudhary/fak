package gateway

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"hash"
	"io"
	"math"
	"net/http"
	"strings"
	"sync"

	"github.com/anthony-chaudhary/fak/internal/enginestep"
	"github.com/anthony-chaudhary/fak/internal/perfledger"
)

type requestObservation struct {
	mu                 sync.Mutex
	requestID, traceID string
	inboundRequestID   string
	digest             hash.Hash
	read, expected     int64
	complete, failed   bool
	admitted           bool
	lifecycle          modelLoadLifecycle
	configSHA          string
	phaseContext       context.Context
}

type observationBody struct {
	io.ReadCloser
	observation *requestObservation
}

func (b *observationBody) Read(p []byte) (int, error) {
	n, err := b.ReadCloser.Read(p)
	o := b.observation
	o.mu.Lock()
	_, _ = o.digest.Write(p[:n])
	o.read += int64(n)
	if err != nil && err != io.EOF {
		o.failed = true
	}
	if err == io.EOF || (o.expected >= 0 && o.read == o.expected) {
		o.complete = true
	}
	if o.expected >= 0 && o.read > o.expected {
		o.failed = true
	}
	o.mu.Unlock()
	return n, err
}

func newRequestObservation(r *http.Request) *requestObservation {
	o := &requestObservation{requestID: randomHex(16), inboundRequestID: inboundRequestID(r.Header), digest: sha256.New(), expected: r.ContentLength}
	if r.Body != nil && r.Body != http.NoBody {
		r.Body = &observationBody{ReadCloser: r.Body, observation: o}
	}
	return o
}

func (s *Server) admitRequestObservation(r *http.Request, trace string) {
	o := perfSourceFrom(r.Context()).observation
	if o == nil {
		return
	}
	o.mu.Lock()
	defer o.mu.Unlock()
	o.traceID = trace
	if o.admitted {
		return
	}
	o.admitted = true
	o.lifecycle = s.captureModelLoadLifecycle()
	config, err := json.Marshal(s.ScalarConfig())
	if err == nil {
		sum := sha256.Sum256(config)
		o.configSHA = hex.EncodeToString(sum[:])
	}
}

func requestObservationContext(r *http.Request) (string, string) {
	o := perfSourceFrom(r.Context()).observation
	if o == nil {
		return "", ""
	}
	o.mu.Lock()
	defer o.mu.Unlock()
	return o.requestID, o.lifecycle.LoadID
}

func bindRequestPhaseContext(r *http.Request, ctx context.Context) {
	o := perfSourceFrom(r.Context()).observation
	if o == nil {
		return
	}
	o.mu.Lock()
	o.phaseContext = ctx
	o.mu.Unlock()
}

func requestPartialNativePhases(ctx context.Context) *perfledger.PartialNativePhases {
	phases := enginestep.RequestPhaseObservationsFromContext(ctx)
	out := &perfledger.PartialNativePhases{Scope: "request_phase_elapsed"}
	if p := phases.Prefill; p != nil && p.Count > 0 && p.DurationNS >= 0 {
		ms := float64(p.DurationNS) / 1e6
		out.PrefillMS, out.PrefillObservations = &ms, p.Count
	}
	if p := phases.Decode; p != nil && p.Count > 0 && p.DurationNS >= 0 {
		ms := float64(p.DurationNS) / 1e6
		out.DecodeMS, out.DecodeObservations = &ms, p.Count
	}
	if out.PrefillMS == nil && out.DecodeMS == nil {
		return nil
	}
	return out
}

func applyRequestObservation(o *requestObservation, rec *perfledger.Record) {
	if o == nil || rec == nil {
		return
	}
	o.mu.Lock()
	defer o.mu.Unlock()
	rec.RequestID, rec.TraceID, rec.InboundRequestID = o.requestID, o.traceID, o.inboundRequestID
	rec.PartialNativePhases = requestPartialNativePhases(o.phaseContext)
	if o.complete && !o.failed && o.read > 0 {
		rec.WorkloadSHA256 = hex.EncodeToString(o.digest.Sum(nil))
		rec.WorkloadIdentityMode = "raw_request_body_sha256_v1"
	}
	if o.configSHA != "" {
		rec.ConfigSHA256, rec.ConfigIdentityMode = o.configSHA, "scalar_v1"
	}
	if v := o.lifecycle; v.LoadID != "" {
		rec.LoadLifecycle = &perfledger.LoadLifecycle{LoadID: v.LoadID, RecordedUnixMS: v.RecordedUnixMS, RequestOrdinal: v.RequestOrdinal, InitialAfterLoad: v.RequestOrdinal == 1}
		if v.LoadSeconds > 0 && !math.IsNaN(v.LoadSeconds) && !math.IsInf(v.LoadSeconds, 0) {
			rec.LoadLifecycle.LoadMS = v.LoadSeconds * 1000
		}
	}
}

const maxInboundRequestIDLen = 128

// inboundRequestID reads the caller's id before ensureHTTPTrace rewrites the
// trace headers: X-Request-ID when it is a bounded printable token, else the
// W3C traceparent trace id.
func inboundRequestID(h http.Header) string {
	if id := strings.TrimSpace(h.Get("X-Request-ID")); id != "" && len(id) <= maxInboundRequestIDLen && printableToken(id) {
		return id
	}
	if tp, err := parseTraceparent(h.Get(traceparentHeader)); err == nil {
		return tp.TraceID
	}
	return ""
}

func printableToken(s string) bool {
	for i := 0; i < len(s); i++ {
		if c := s[i]; c <= ' ' || c > '~' || c == '"' || c == '\\' {
			return false
		}
	}
	return true
}
