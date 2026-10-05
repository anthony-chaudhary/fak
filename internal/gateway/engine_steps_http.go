package gateway

import (
	"net/http"
	"strconv"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/enginestep"
	"github.com/anthony-chaudhary/fak/internal/stepobs"
)

// engineStepsDefaultRecent is how many ring records /v1/fak/observation/engine
// returns when the caller does not ask (?n=). MaxRecent bounds any request.
const engineStepsDefaultRecent = 32

// engineStepsSchema versions the /v1/fak/observation/engine envelope.
const engineStepsSchema = "fak-observation-engine/1"

// engineStepsResponse is the bounded agent read of the native serving loop: the
// planner kind (so a proxy gateway reads as "not native", never as an idle
// engine) and the enginestep snapshot.
type engineStepsResponse struct {
	Schema  string              `json:"schema"`
	Planner string              `json:"planner"`
	Native  bool                `json:"native"`
	Engine  enginestep.Snapshot `json:"engine"`
}

// nativeEngineServing reports whether /v1/* chat is answered by the in-process
// native planner, i.e. whether the enginestep recorder describes this server.
func (s *Server) nativeEngineServing() bool {
	if s == nil {
		return false
	}
	switch plannerKind(s.planner) {
	case "inkernel", "dual":
		return true
	default:
		return false
	}
}

// writeEngineStepMetrics renders the fak_engine_* continuous-batching cycle
// families when this server runs the native planner: the cycle itself
// (enginestep), then its two sub-seams — every kernel call and every planner-step
// leg (stepobs). A proxy/mock gateway emits nothing rather than a phantom idle engine.
func (s *Server) writeEngineStepMetrics(b *strings.Builder) {
	if b == nil || !s.nativeEngineServing() {
		return
	}
	enginestep.Default.WritePrometheus(b)
	stepobs.Default.WritePrometheus(b)
}

// handleFakObservationEngine serves GET /v1/fak/observation/engine: the bounded
// per-step view of the native continuous-batching loop for agents.
//
//	?n=<int>        recent ring records to include (default 32, max 512, 0 = none)
//	?kind=<kind>    filter records: decode_step | prefill_chunk | cohort | phase
//	?format=compact one text line instead of JSON
func (s *Server) handleFakObservationEngine(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet && r.Method != http.MethodHead {
		w.Header().Set("Allow", "GET, HEAD")
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}
	q := r.URL.Query()
	n := engineStepsDefaultRecent
	if raw := q.Get("n"); raw != "" {
		v, err := strconv.Atoi(raw)
		if err != nil || v < 0 {
			http.Error(w, "n must be a non-negative integer", http.StatusBadRequest)
			return
		}
		n = v
	}
	kind := q.Get("kind")
	switch kind {
	case "", enginestep.KindDecodeStep, enginestep.KindPrefillChunk, enginestep.KindCohort, enginestep.KindPhase:
	default:
		http.Error(w, "kind must be one of decode_step, prefill_chunk, cohort, phase", http.StatusBadRequest)
		return
	}
	native := s.nativeEngineServing()
	snap := enginestep.Default.Snapshot(n, kind)
	if q.Get("format") == "compact" {
		w.Header().Set("Content-Type", "text/plain; charset=utf-8")
		if !native {
			_, _ = w.Write([]byte("ENGINE not native (planner=" + plannerKind(s.planner) + ")\n"))
			return
		}
		_, _ = w.Write([]byte(snap.Compact() + "\n"))
		return
	}
	writeJSON(w, http.StatusOK, engineStepsResponse{
		Schema:  engineStepsSchema,
		Planner: plannerKind(s.planner),
		Native:  native,
		Engine:  snap,
	})
}
