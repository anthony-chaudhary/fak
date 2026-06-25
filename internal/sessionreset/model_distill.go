package sessionreset

import (
	"strconv"
	"strings"
)

// model_distill.go — the OPTIONAL model-call task distiller (#741). The built-in
// taskDistill (contributors.go) is deterministic and EXTRACTIVE: objective = first user
// line, latest = last user line. It is cheap and reproducible but blind to the MIDDLE of
// the session — the decisions made, the blockers hit, the state the work is actually in.
// This contributor closes that gap by asking a model to read the WHOLE transcript and
// write a real "where we are" recap.
//
// WHY IT IS OPT-IN. A model call costs tokens and latency, and it is — by construction —
// the ONE non-deterministic contributor (its output depends on a provider, not just the
// Input). The package's purity/determinism contract (sessionreset.go) is about the
// package's OWN code: this file imports no provider and makes no call on its own; the
// SummarizeFunc seam is injected by the host, so nondeterminism enters ONLY when a host
// deliberately opts in by calling RegisterModelDistiller. The default fold (the four
// built-ins) stays pure. That is why this is registered on demand, never at init.
//
// GRACEFUL DEGRADATION. Every failure path DECLINES (ok=false) rather than poisoning the
// seed: a nil seam, a transcript too short to be worth a model call, a model error, or an
// empty summary all fall through to the deterministic taskDistill, which still fires. A
// reset never blocks on the model distiller succeeding.

// SummarizeFunc is the model-call seam. It receives a fully-formed prompt (the recap
// instruction + the rendered transcript) and returns the model's "where we are" summary,
// or an error if the call fails/times out. The host injects a provider-backed
// implementation; sessionreset never imports a provider, so its tier (2) is unchanged.
// Implementations should be context-bounded by the host — this package neither sets nor
// sees a timeout.
type SummarizeFunc func(prompt string) (string, error)

// DefaultModelDistillMinChars is the rendered-transcript length below which the model
// distiller declines: a session this short is fully covered by the cheap extractive
// taskDistill, so paying for a model call buys nothing. The host can override it.
const DefaultModelDistillMinChars = 400

// ModelDistillOrder places the model recap right after the extractive taskDistill (20),
// so when both are registered the model recap reads as the richer companion to the
// one-line objective/latest, not a replacement scattered elsewhere in the seed.
const ModelDistillOrder = 21

// ModelDistillInstruction is the recap instruction prepended to the transcript before the
// model call. It lives here (not in the host) because this package owns what a carryover
// recap MEANS — decisions, blockers, current state — so every host that opts in asks for
// the same shape. The host's only job is to run the model on the resulting prompt.
const ModelDistillInstruction = "You are summarizing a long assistant session that is being reset at its token budget. " +
	"Write a compact \"where we are\" recap for the fresh session that continues this work. " +
	"Cover, in this order and only if present: the standing objective, decisions already made, " +
	"open blockers or unknowns, and the current state / next step. Be terse and factual — no preamble, " +
	"no restating these instructions. Output only the recap."

// modelDistill is the opt-in model-call contributor. It carries its injected seam, the
// cost gate, and its render order so the same type serves any host configuration.
type modelDistill struct {
	summarize SummarizeFunc
	minChars  int
	order     int
}

// Name is distinct from the extractive "task_distill" so both can coexist in one fold and
// the audit surface (Registered/Parts) can tell which produced a given recap.
func (modelDistill) Name() string { return "task_distill_model" }

// Contribute renders the transcript, gates on cost, runs the injected model seam, and
// folds the result. Any decline reason is recorded in Meta (for the audit surface) and
// returns ok=false so the deterministic distiller still covers the slot.
func (m modelDistill) Contribute(in Input) (Part, bool) {
	order := m.order
	if order == 0 {
		order = ModelDistillOrder
	}
	if m.summarize == nil {
		return Part{Name: "task_distill_model", Order: order,
			Meta: map[string]string{"skipped": "no_model_seam"}}, false
	}
	transcript := renderTranscriptForModel(in.Messages)
	min := m.minChars
	if min <= 0 {
		min = DefaultModelDistillMinChars
	}
	if len(transcript) < min {
		return Part{Name: "task_distill_model", Order: order,
			Meta: map[string]string{"skipped": "below_min_chars", "chars": strconv.Itoa(len(transcript))}}, false
	}
	summary, err := m.summarize(ModelDistillInstruction + "\n\n--- transcript ---\n" + transcript)
	if err != nil {
		return Part{Name: "task_distill_model", Order: order,
			Meta: map[string]string{"skipped": "model_call_failed"}}, false
	}
	summary = strings.TrimSpace(summary)
	if summary == "" {
		return Part{Name: "task_distill_model", Order: order,
			Meta: map[string]string{"skipped": "empty_summary"}}, false
	}
	text := "Where we are (model recap):\n" + clip(summary, 1200)
	return Part{Name: "task_distill_model", Order: order, Text: text,
		Meta: map[string]string{"source": "model", "transcript_chars": strconv.Itoa(len(transcript))}}, true
}

// NewModelDistiller builds the model-call contributor WITHOUT registering it — for a host
// that folds its own registry or a test that wants the contributor in isolation. A nil fn
// yields a contributor that always declines (no_model_seam), so it is never a panic. A
// minChars <= 0 uses DefaultModelDistillMinChars.
func NewModelDistiller(fn SummarizeFunc, minChars int) Contributor {
	return modelDistill{summarize: fn, minChars: minChars, order: ModelDistillOrder}
}

// RegisterModelDistiller opts the process into the model-call task distiller and returns
// the registered contributor (for inspection/tests). It is the cost gate's on-switch: the
// distiller is absent from the default fold until a host calls this with a model seam. A
// nil fn is a no-op — it registers nothing and returns nil — so a host that resolves no
// model degrades to the deterministic distiller rather than registering a dead contributor.
func RegisterModelDistiller(fn SummarizeFunc, minChars int) Contributor {
	if fn == nil {
		return nil
	}
	c := NewModelDistiller(fn, minChars)
	Register(c)
	return c
}

// renderTranscriptForModel flattens the transcript into the "role: content" form the model
// reads. Empty lines are dropped; nothing is reordered or summarized here — the WHOLE
// remaining transcript is fed so the model, not this package, decides what matters.
func renderTranscriptForModel(msgs []Msg) string {
	var b strings.Builder
	for _, m := range msgs {
		c := strings.TrimSpace(m.Content)
		if c == "" {
			continue
		}
		b.WriteString(m.Role)
		b.WriteString(": ")
		b.WriteString(c)
		b.WriteString("\n")
	}
	return strings.TrimRight(b.String(), "\n")
}
