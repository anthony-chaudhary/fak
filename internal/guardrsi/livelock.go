package guardrsi

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"io"
	"strings"
)

const (
	LivelockSchema           = "guardrsi.livelock/1"
	LivelockEvent            = "LIVELOCK_DETECTED"
	DefaultLivelockThreshold = 3
	// DefaultLivelockFuseFactor multiplies the advisory threshold to get the fuse
	// count. The advisory note fires at `threshold` and repeats for the next few
	// identical outcomes; if the run keeps going and reaches threshold*factor, the
	// model has demonstrably ignored the nudge, so the envelope arms Fuse and the
	// caller converts the still-admitted call into a hard refusal. A purely advisory
	// note lets an unresponsive loop burn tokens forever (the gateway kept admitting
	// the identical call); the fuse is the structural backstop that actually breaks it.
	DefaultLivelockFuseFactor = 2
	// DefaultLivelockAbortFactor multiplies the advisory threshold to get the ABORT
	// count — the second, higher rung above the fuse. The fuse (threshold*2) converts
	// the repeated call into a per-tool RETRYABLE refusal, but a model that ignores
	// even the fuse keeps re-proposing the same call every turn and the loop never
	// ends (worker #2704: an identical call fused at 6, then spun to ~125k tokens
	// because a RETRYABLE refusal is auto-continued). At threshold*factor (=9 for the
	// default 3) the envelope arms Escalate, and the caller stamps the refusal
	// TERMINAL so the turn becomes a deny-all — the bounded give-up path that actually
	// stops the session. It sits strictly ABOVE the fuse so the soft advisory and the
	// retryable fuse always fire first; the terminal stop is a last resort.
	DefaultLivelockAbortFactor = 3
)

// LivelockObservation is one repeated tool-call outcome, identified without carrying
// raw arguments. TraceID scopes the consecutive run to one agent session.
type LivelockObservation struct {
	TraceID     string
	Tool        string
	ArgsDigest  string
	Verdict     string
	Reason      string
	Disposition string
}

// LivelockEnvelope is the structured nudge returned once an identical tool call
// repeats enough times in one session.
type LivelockEnvelope struct {
	Schema          string `json:"schema"`
	Event           string `json:"event"`
	TraceID         string `json:"trace_id,omitempty"`
	Tool            string `json:"tool"`
	ArgsDigest      string `json:"args_digest,omitempty"`
	FailureHash     string `json:"failure_hash"`
	Verdict         string `json:"verdict"`
	Reason          string `json:"reason,omitempty"`
	Disposition     string `json:"disposition,omitempty"`
	RepeatCount     int    `json:"repeat_count"`
	SuggestedChange string `json:"suggested_change"`
	// Fuse is true once the consecutive run reaches the fuse count (>= advisory
	// threshold). It tells the caller the loop must be broken now — an admitted call
	// carrying Fuse should be converted into a hard refusal rather than admitted
	// again. Fuse never fires before the advisory has already fired at least once.
	Fuse bool `json:"fuse,omitempty"`
	// Escalate is true once the run reaches the ABORT count (threshold*abortFactor),
	// strictly above the fuse count. It tells the caller the fuse's per-tool refusal
	// was itself ignored turn after turn, so the refusal must be stamped TERMINAL
	// (non-retryable) and the turn allowed to become a deny-all — the bounded stop that
	// ends a session a soft/retryable rung never could. Escalate implies Fuse.
	Escalate bool `json:"escalate,omitempty"`
}

type livelockRun struct {
	key   string
	count int
	last  LivelockObservation
}

// LivelockDetector tracks consecutive identical tool-call outcomes per trace. It is intentionally
// small and caller-synchronized; gateway guards it with its server mutex.
type LivelockDetector struct {
	threshold int
	fuse      int
	abort     int
	byTrace   map[string]livelockRun
}

func NewLivelockDetector(threshold int) *LivelockDetector {
	if threshold <= 0 {
		threshold = DefaultLivelockThreshold
	}
	return &LivelockDetector{
		threshold: threshold,
		fuse:      threshold * DefaultLivelockFuseFactor,
		abort:     threshold * DefaultLivelockAbortFactor,
		byTrace:   map[string]livelockRun{},
	}
}

// NewLivelockDetectorWithFuse builds a detector whose advisory fires at `threshold`
// and whose fuse arms at `fuse`. A fuse <= threshold is clamped up to threshold so
// the fuse can never precede the first advisory. A non-positive fuse disables the
// fuse entirely (envelope.Fuse stays false — advisory-only, the pre-fuse behavior).
func NewLivelockDetectorWithFuse(threshold, fuse int) *LivelockDetector {
	d := NewLivelockDetector(threshold)
	switch {
	case fuse <= 0:
		d.fuse = -1  // sentinel: fuse explicitly disabled, advisory-only
		d.abort = -1 // no fuse => no terminal rung either (advisory-only detector)
	case fuse < d.threshold:
		d.fuse = d.threshold
	default:
		d.fuse = fuse
	}
	// Keep the terminal (abort) rung strictly above the fuse so the retryable fuse
	// always fires first. The default abort is threshold*abortFactor; if an explicit
	// fuse was set at or above that, push abort one threshold beyond it.
	if d.fuse > 0 && d.abort <= d.fuse {
		d.abort = d.fuse + d.threshold
	}
	return d
}

// ArgsDigest returns a content-free identity for tool arguments. Valid JSON is compacted
// through encoding/json first, which makes object key order and whitespace irrelevant.
func ArgsDigest(raw string) string {
	b := []byte(strings.TrimSpace(raw))
	if len(b) == 0 {
		b = []byte("{}")
	}
	if canon, ok := canonicalJSON(b); ok {
		b = canon
	}
	sum := sha256.Sum256(b)
	return "sha256:" + hex.EncodeToString(sum[:])
}

func canonicalJSON(raw []byte) ([]byte, bool) {
	dec := json.NewDecoder(strings.NewReader(string(raw)))
	dec.UseNumber()
	var v any
	if err := dec.Decode(&v); err != nil {
		return nil, false
	}
	var extra any
	if err := dec.Decode(&extra); err != io.EOF {
		return nil, false
	}
	b, err := json.Marshal(v)
	return b, err == nil
}

func (d *LivelockDetector) Clear(trace string) {
	if d == nil || trace == "" {
		return
	}
	delete(d.byTrace, trace)
}

func (d *LivelockDetector) ObserveFailure(obs LivelockObservation) (LivelockEnvelope, bool) {
	return d.observe(obs)
}

func (d *LivelockDetector) ObserveAllowed(obs LivelockObservation) (LivelockEnvelope, bool) {
	obs.Verdict = "ALLOW"
	return d.ObserveAdmitted(obs)
}

func (d *LivelockDetector) ObserveAdmitted(obs LivelockObservation) (LivelockEnvelope, bool) {
	if strings.TrimSpace(obs.Verdict) == "" {
		obs.Verdict = "ALLOW"
	}
	return d.observe(obs)
}

func (d *LivelockDetector) observe(obs LivelockObservation) (LivelockEnvelope, bool) {
	if d == nil || strings.TrimSpace(obs.TraceID) == "" {
		return LivelockEnvelope{}, false
	}
	if d.threshold <= 0 {
		d.threshold = DefaultLivelockThreshold
	}
	if d.fuse == 0 && d.threshold > 0 {
		// Zero-valued detector (constructed as &LivelockDetector{} rather than via a
		// constructor): default the fuse to the standard factor so the backstop is
		// present. An explicit opt-out is stored as the -1 sentinel, not 0, so this only
		// rescues the never-initialized case.
		d.fuse = d.threshold * DefaultLivelockFuseFactor
	}
	if d.abort == 0 && d.threshold > 0 {
		// Same zero-value rescue for the terminal rung (see fuse above): an explicit
		// opt-out is the -1 sentinel, so 0 only means never-initialized.
		d.abort = d.threshold * DefaultLivelockAbortFactor
	}
	if d.byTrace == nil {
		d.byTrace = map[string]livelockRun{}
	}
	obs.TraceID = strings.TrimSpace(obs.TraceID)
	obs.Tool = strings.TrimSpace(obs.Tool)
	obs.ArgsDigest = strings.TrimSpace(obs.ArgsDigest)
	obs.Verdict = strings.ToUpper(strings.TrimSpace(obs.Verdict))
	obs.Reason = strings.TrimSpace(obs.Reason)
	obs.Disposition = strings.TrimSpace(obs.Disposition)
	key := livelockKey(obs)
	run := d.byTrace[obs.TraceID]
	if run.key == key {
		run.count++
		run.last = obs
	} else {
		run = livelockRun{key: key, count: 1, last: obs}
	}
	d.byTrace[obs.TraceID] = run
	if run.count < d.threshold {
		return LivelockEnvelope{}, false
	}
	return LivelockEnvelope{
		Schema:          LivelockSchema,
		Event:           LivelockEvent,
		TraceID:         obs.TraceID,
		Tool:            obs.Tool,
		ArgsDigest:      obs.ArgsDigest,
		FailureHash:     failureHash(key),
		Verdict:         obs.Verdict,
		Reason:          obs.Reason,
		Disposition:     obs.Disposition,
		RepeatCount:     run.count,
		SuggestedChange: suggestedLivelockChange(obs),
		Fuse:            d.fuse > 0 && run.count >= d.fuse,
		Escalate:        d.abort > 0 && run.count >= d.abort,
	}, true
}

func livelockKey(obs LivelockObservation) string {
	return obs.Tool + "\x00" + obs.ArgsDigest + "\x00" + obs.Verdict + "\x00" + obs.Reason + "\x00" + obs.Disposition
}

func failureHash(key string) string {
	sum := sha256.Sum256([]byte(key))
	return "sha256:" + hex.EncodeToString(sum[:])
}

func suggestedLivelockChange(obs LivelockObservation) string {
	switch strings.ToUpper(obs.Disposition) {
	case "WAIT":
		return "change_approach_wait_or_fetch_merge_before_retry"
	case "ESCALATE":
		return "change_approach_escalate_or_not_yet_with_witness"
	}
	if strings.EqualFold(obs.Verdict, "ALLOW") || strings.EqualFold(obs.Verdict, "TRANSFORM") {
		return "change_approach_stop_repeating_successful_call_or_summarize_result"
	}
	switch strings.ToUpper(obs.Reason) {
	case "STALE_BASE_DELETION", "OFF_TRUNK", "MERGE_IN_PROGRESS", "COLLISION_RISK":
		return "change_approach_fetch_merge_or_wait_for_lease"
	case "LOOP_DONE_UNWITNESSED", "CORE_SELF_MODIFY", "INDETERMINATE":
		return "change_approach_escalate_or_not_yet_with_witness"
	default:
		return "change_approach_fetch_merge_escalate_or_not_yet_with_witness"
	}
}
