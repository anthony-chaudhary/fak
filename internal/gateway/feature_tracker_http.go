package gateway

import (
	"net/http"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/agent"
)

const (
	HeaderFeaturesEnabled   = "X-Fak-Features-Enabled"
	HeaderFeaturesUsed      = "X-Fak-Features-Used"
	HeaderFeaturesUsedFinal = "X-Fak-Features-Used-Final"
)

// withFeatureActivations sits outside metrics so the same request tracker reaches
// access logging and the response writer observes recovered error responses too.
func (s *Server) withFeatureActivations(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/v1/chat/completions", "/v1/messages", "/v1/fak/syscall":
		default:
			next.ServeHTTP(w, r)
			return
		}
		tracker := NewFeatureActivationTracker(s.FeatureSnapshot())
		r = r.WithContext(WithFeatureActivationTracker(r.Context(), tracker))
		r = r.WithContext(agent.WithNativeCompactionObserver(r.Context(), func(observation agent.NativeCompactionObservation) {
			if observation.DroppedMessages > 0 {
				tracker.RecordActivation(FeatureCompactHistory, FeatureOutcomeUsed)
			}
		}))
		wrapped, finish := newFeatureActivationResponseWriter(w, tracker)
		// Finalization runs only after normal return, including panics recovered by
		// withMetrics. An aborted transport must not publish a completed trailer.
		next.ServeHTTP(wrapped, r)
		snapshot, first := finish()
		if first && s.metrics != nil {
			s.metrics.featureActivations.ObserveFinal(snapshot)
		}
	})
}

// newFeatureActivationResponseWriter snapshots evidence at the first final header
// commit, without buffering response bytes. Finish must be called after request
// workers join and before ServeHTTP returns; its boolean gates metrics/log folding.
// A nil tracker leaves the writer and its capabilities unchanged.
func newFeatureActivationResponseWriter(w http.ResponseWriter, tracker *FeatureActivationTracker) (http.ResponseWriter, func() (FeatureActivationSnapshot, bool)) {
	if tracker == nil {
		return w, tracker.Finalize
	}
	r := &featureActivationResponseWriter{ResponseWriter: w, tracker: tracker}
	r.declareTrailer()
	if _, ok := w.(http.Flusher); ok {
		return &featureActivationFlusher{featureActivationFlushError: &featureActivationFlushError{featureActivationResponseWriter: r}}, r.finish
	}
	if _, ok := w.(interface{ FlushError() error }); ok {
		return &featureActivationFlushError{featureActivationResponseWriter: r}, r.finish
	}
	return r, r.finish
}

type featureActivationResponseWriter struct {
	http.ResponseWriter
	tracker   *FeatureActivationTracker
	committed bool
}

func (w *featureActivationResponseWriter) Unwrap() http.ResponseWriter {
	return w.ResponseWriter
}

func (w *featureActivationResponseWriter) declareTrailer() {
	for _, declaration := range w.Header().Values("Trailer") {
		for _, name := range strings.Split(declaration, ",") {
			if strings.EqualFold(strings.TrimSpace(name), HeaderFeaturesUsedFinal) {
				return
			}
		}
	}
	w.Header().Add("Trailer", HeaderFeaturesUsedFinal)
}

func (w *featureActivationResponseWriter) WriteHeader(status int) {
	if w.committed {
		return
	}
	if status >= 100 && status < 200 && status != http.StatusSwitchingProtocols {
		w.ResponseWriter.WriteHeader(status)
		return
	}
	snapshot := w.tracker.Snapshot()
	w.declareTrailer()
	h := w.Header()
	h.Set(HeaderFeaturesEnabled, joinActivationFeatures(snapshot.Enabled))
	h.Set(HeaderFeaturesUsed, joinActivationFeatures(snapshot.Used))
	// A declared trailer needs streaming framing on HTTP/1.1, even for a buffered
	// handler that supplied a body length. Do not let a stale initial value leak.
	h.Del("Content-Length")
	h.Del(HeaderFeaturesUsedFinal)
	w.ResponseWriter.WriteHeader(status)
	w.committed = true
}

func (w *featureActivationResponseWriter) Write(body []byte) (int, error) {
	if !w.committed {
		w.WriteHeader(http.StatusOK)
	}
	return w.ResponseWriter.Write(body)
}

func (w *featureActivationResponseWriter) finish() (FeatureActivationSnapshot, bool) {
	snapshot, first := w.tracker.Finalize()
	if first {
		if !w.committed {
			w.WriteHeader(http.StatusOK)
		}
		if w.tracker.completeEvidence() {
			w.Header().Set(HeaderFeaturesUsedFinal, joinActivationFeatures(snapshot.Used))
		}
	}
	return snapshot, first
}

// Only wrappers around an actual Flusher implement http.Flusher. FlushError is
// retained separately for ResponseController users without inventing Flush support.
type featureActivationFlushError struct {
	*featureActivationResponseWriter
}

func (w *featureActivationFlushError) FlushError() error {
	if !w.committed {
		w.WriteHeader(http.StatusOK)
	}
	if f, ok := w.ResponseWriter.(interface{ FlushError() error }); ok {
		return f.FlushError()
	}
	w.ResponseWriter.(http.Flusher).Flush()
	return nil
}

type featureActivationFlusher struct {
	*featureActivationFlushError
}

func (w *featureActivationFlusher) Flush() {
	_ = w.FlushError()
}

func joinActivationFeatures(features []ServeFeature) string {
	names := make([]string, len(features))
	for i, feature := range features {
		names[i] = string(feature)
	}
	return strings.Join(names, ",")
}
