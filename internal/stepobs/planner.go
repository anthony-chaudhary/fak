package stepobs

import (
	"io"
	"strconv"
	"time"

	"github.com/anthony-chaudhary/fak/internal/enginestep"
)

// plannerRegistry folds planner-step legs over the CLOSED StepKind vocabulary. Because
// the vocabulary is fixed, the series set is fixed: every kind is always allocated and
// always rendered, so a dashboard can read "seat legs are zero" as a real answer rather
// than "the seat series is missing".
type plannerRegistry struct {
	seconds      map[StepKind]*histogram
	eventsByKind map[StepKind]uint64
	events       uint64 // every leg folded, including any whose kind overflowed
	kindOverflow uint64 // legs dropped because their kind was outside the vocabulary
}

func (pr *plannerRegistry) init() {
	pr.seconds = make(map[StepKind]*histogram, len(StepKinds))
	pr.eventsByKind = make(map[StepKind]uint64, len(StepKinds))
	for _, k := range StepKinds {
		pr.seconds[k] = newHistogram(secondsBuckets)
	}
}

func (pr *plannerRegistry) observe(kind StepKind, d time.Duration) {
	h, ok := pr.seconds[kind]
	if !ok {
		// An unknown kind is counted and dropped, never folded into a neighbouring
		// label and never turned into a new label value: a producer that invents kinds
		// must be visible as a number, not as a wider board.
		pr.kindOverflow++
		return
	}
	h.observe(seconds(d))
	pr.eventsByKind[kind]++
	pr.events++
}

func (pr *plannerRegistry) write(w io.Writer) {
	helpType(w, MetricPlannerStepSeconds,
		"Duration of one planner-step leg in seconds, over the closed kind vocabulary (step, tool, admission, seat, verdict). step is one model turn (prefill/decode/sampling); seat is a slot or device-seat wait; admission is an admission decision.",
		"histogram")
	for _, k := range StepKinds {
		pr.seconds[k].write(w, MetricPlannerStepSeconds, "kind=\""+quoteLabel(string(k))+"\"")
	}

	helpType(w, MetricPlannerStepEvents,
		"Planner-step legs observed, by closed kind vocabulary.",
		"counter")
	for _, k := range StepKinds {
		writeString(w, MetricPlannerStepEvents+"{kind=\""+quoteLabel(string(k))+"\"} "+strconv.FormatUint(pr.eventsByKind[k], 10)+"\n")
	}

	observed := 0
	if pr.events > 0 {
		observed = 1
	}
	helpType(w, MetricPlannerStepObserved,
		"1 only when a planner-step observer is genuinely attached AND fed at least one real leg; 0 means the sub-planner producer is ABSENT, never that zero steps ran.",
		"gauge")
	writeString(w, MetricPlannerStepObserved+" "+strconv.Itoa(observed)+"\n")

	helpType(w, MetricPlannerStepKindOverflowTotal,
		"Planner-step legs dropped because their kind fell outside the closed vocabulary, rather than being folded into a neighbouring label.",
		"counter")
	writeString(w, MetricPlannerStepKindOverflowTotal+" "+strconv.FormatUint(pr.kindOverflow, 10)+"\n")
}

// ObservePlannerStep folds one planner-step leg. kind must be a member of the closed
// StepKind vocabulary; anything else is counted as an overflow and dropped.
func (r *Recorder) ObservePlannerStep(kind StepKind, d time.Duration) {
	if r == nil {
		return
	}
	r.mu.Lock()
	defer r.mu.Unlock()
	r.planner.observe(kind, d)
}

// ObserveEngineStep folds one microagent span leg by its closed MicroSpanKind. The kind
// arrives as a plain string so this package does not have to import internal/metrics —
// metrics sits below agent, and agent reaches this package, so the metrics dependency
// would be a cycle. gateway holds the typed bridge; the values are the same strings.
func (r *Recorder) ObserveEngineStep(kind string, d time.Duration) {
	if r == nil {
		return
	}
	r.mu.Lock()
	defer r.mu.Unlock()
	r.planner.observe(StepKind(kind), d)
}

// ObserveEnginePhase folds one native serving-loop phase, mapping it onto the closed
// planner-step vocabulary. A phase that names no span leg records nothing.
func (r *Recorder) ObserveEnginePhase(p enginestep.Phase, d time.Duration) {
	kind, ok := StepKindForEnginePhase(p)
	if !ok {
		return
	}
	r.ObservePlannerStep(kind, d)
}
