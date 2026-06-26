// Package tracesink is a payload-bearing, IFC-labeled TRAJECTORY SINK.
//
// WHY THIS IS NET-NEW (issue #499). The policy-replay spine (turnbench.RunPolicyReplay)
// proves K-policy replay on trace FIXTURES whose turnbench.Call carries its Args
// PAYLOAD, so a content-inspecting policy (arg-rules, redact-fields, glob/byte caps,
// self-modify-by-command) can re-adjudicate faithfully. But the durable decision
// journal (internal/journal's Row) stores arg/result DIGESTS — sha256 content hashes
// (Row.ArgsDigest / Row.ResultDigest), never the bytes. A policy that inspects an
// arg VALUE cannot re-adjudicate against a hash. So "replay last month's production
// sessions through a candidate policy" needs a payload-bearing sink: a recorder that
// captures the LIVE bytes a call carried into a turnbench.Trace, which then round-trips
// through RunPolicyReplay. That is this package.
//
// It lives in its OWN package (not internal/journal) only to break an import cycle:
// journal is blank-imported by internal/registrations, which turnbench imports, so a
// journal->turnbench edge would close a loop. tracesink imports turnbench (read-only)
// and nothing imports tracesink, so the recorder sits cleanly downstream of the spine.
//
// WHERE IT CAPTURES. The sink is an abi.Emitter scoped to EvSubmit — the one event
// the kernel fires with the call's ORIGINAL Args Ref, BEFORE a grammar TRANSFORM
// rewrites them and before dispatch. That is the model-observed call: exactly the
// (tool, args, meta) RunPolicyReplay re-feeds. Capturing post-transform would record
// the kernel's repair, not what the model emitted, and the replay would not re-exercise
// the grammar rung. The emitter runs SYNCHRONOUSLY inside Submit, so it resolves the
// args bytes immediately — before any later mutation of the shared *ToolCall.
//
// THE IFC LABEL + THE EGRESS FLOOR (the load-bearing honesty half). Every captured
// call carries its IFC provenance taint (per-call Meta key trace_taint) and the sink's
// world-version / capture provenance. Crucially the sink RESPECTS ITS OWN EGRESS FLOOR:
// it must not become a side channel that persists, in plaintext, a payload the kernel's
// IFC sink-gate would refuse to let egress. Before retaining a payload the sink asks the
// SAME ifc.Classify the live SinkGate asks; if the call is a sensitive sink AND the data
// flowing into it is tainted (the call's own args, or the session high-water mark), the
// sink REDACTS the payload to a digest-only placeholder rather than persisting the bytes.
// The recorded Trace is then itself safe to read back: a content-inspecting policy still
// sees that the call HAPPENED (tool+meta+taint), but the floor's egress decision is not
// laundered by the recorder. A capture that redacts is marked (Meta key payload_redacted)
// so the replay knows the bytes are a placeholder, not the original argument.
//
// COMPLETENESS ("trace is total"). The sink counts every EvSubmit it is offered and
// every call it actually recorded; Dropped() is the gap. An out-of-band path (an event
// kind the sink does not subscribe to, an unresolvable Ref) is COUNTED, never silently
// dropped — Total() == Recorded()+Dropped() is the invariant a witness checks before
// trusting any fleet-scale claim built on a captured corpus.
package tracesink

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"sort"
	"sync"
	"time"

	"github.com/anthony-chaudhary/fak/internal/abi"
	"github.com/anthony-chaudhary/fak/internal/ifc"
	"github.com/anthony-chaudhary/fak/internal/provenance"
	"github.com/anthony-chaudhary/fak/internal/turnbench"
)

// Per-call Meta keys the sink stamps onto every captured turnbench.Call. They are
// additive provenance the replay and an auditor read; WorkloadHash folds Meta, so a
// relabel changes the recorded workload identity (the keys are part of the capture).
const (
	// MetaTaint is the IFC provenance taint of the call's source/flow at capture time
	// (trusted | tainted | quarantined). This is the label the issue asks the sink to
	// retain payloads UNDER.
	MetaTaint = "trace_taint"
	// MetaWorld pins the capture's world-version so two recordings made under different
	// kernel states are distinguishable.
	MetaWorld = "trace_world"
	// MetaRedacted is "true" when the egress floor forced the payload to a digest
	// placeholder; the original Args are NOT present (the floor would not let them out).
	MetaRedacted = "payload_redacted"
	// MetaArgsDigest is the sha256 of the ORIGINAL args, recorded even when redacted so
	// a content-addressed identity survives the redaction.
	MetaArgsDigest = "args_digest"
	// MetaCaptureSeq is the kernel ToolCall.SeqNo of the captured submission — the join
	// key back to the durable Row stream for the same call.
	MetaCaptureSeq = "capture_seq"
)

// TraceSink captures a live run's submitted calls into a payload-bearing turnbench.Trace.
// It is an abi.Emitter (register it with abi.RegisterEmitter, or attach it to one kernel
// for a scoped recording). The zero value is not usable; construct with NewTraceSink.
//
// It is the data-plane dual of Journal: Journal persists the DECISION (verdict + digest)
// tamper-evidently; TraceSink persists the model-observed CALL (tool + payload + label)
// replayably. They share this package because they record the same lifecycle, at the same
// EvSubmit/EvDecide seam, from the same abi.Event.
type TraceSink struct {
	mu      sync.Mutex
	sliceID string
	res     abi.Resolver
	ledger  *ifc.Ledger
	policy  ifc.Policy
	world   string
	clock   func() time.Time

	calls    []turnbench.Call
	total    uint64 // every EvSubmit offered
	recorded uint64 // calls actually appended
	dropped  uint64 // offered but not recorded (unresolvable / out-of-band)
	redacted uint64 // recorded with the payload replaced by a digest placeholder
}

// Options configures a TraceSink. All fields are optional with safe defaults.
type Options struct {
	// SliceID names the recorded trace (defaults to a timestamped id).
	SliceID string
	// Resolver materializes a call's Args Ref to bytes. Defaults to abi.ActiveResolver.
	Resolver abi.Resolver
	// Ledger is the IFC high-water-mark ledger consulted for the egress floor. Defaults
	// to ifc.Default (the process-wide ledger the live gates raise).
	Ledger *ifc.Ledger
	// Policy is the IFC sink-classification policy. Zero value uses ifc defaults.
	Policy ifc.Policy
	// World pins the recorded world-version (defaults to a timestamp).
	World string
	// Clock is injectable for deterministic tests.
	Clock func() time.Time
}

// NewTraceSink builds a recorder. Defaults fill in the live process resolver + ledger,
// so a caller can NewTraceSink(Options{}) and attach it to a kernel run.
func NewTraceSink(o Options) *TraceSink {
	clock := o.Clock
	if clock == nil {
		clock = time.Now
	}
	res := o.Resolver
	if res == nil {
		res = abi.ActiveResolver()
	}
	ledger := o.Ledger
	if ledger == nil {
		ledger = ifc.Default
	}
	sliceID := o.SliceID
	if sliceID == "" {
		sliceID = "trace-capture-" + clock().UTC().Format("20060102T150405Z")
	}
	world := o.World
	if world == "" {
		world = clock().UTC().Format(time.RFC3339Nano)
	}
	return &TraceSink{
		sliceID: sliceID,
		res:     res,
		ledger:  ledger,
		policy:  o.Policy,
		world:   world,
		clock:   clock,
	}
}

// Subscriptions scopes the sink to EvSubmit — the call-bearing, pre-transform seam. The
// kernel delivers only that kind to the sink, so the recorder adds nothing to the
// EvDispatch/EvComplete path (and never double-records the same call from EvDecide).
func (s *TraceSink) Subscriptions() []abi.EventKind { return []abi.EventKind{abi.EvSubmit} }

// Emit implements abi.Emitter. It records one submitted call's payload-bearing,
// IFC-labeled turnbench.Call, applying the egress floor. It never blocks the kernel and
// never panics; a call it cannot resolve is COUNTED as dropped (completeness), not lost.
func (s *TraceSink) Emit(ev abi.Event) {
	if ev.Kind != abi.EvSubmit {
		return // out-of-band for this sink; Subscriptions already scopes us, belt-and-braces
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	s.total++

	c := ev.Call
	if c == nil {
		s.dropped++
		return
	}

	args, ok := s.resolveArgs(c)
	if !ok {
		// An unresolvable Ref is an out-of-band tool path we must not silently drop.
		s.dropped++
		return
	}

	digest := "sha256:" + hex.EncodeToString(sha256Sum(args))
	taint := s.flowTaint(c)

	meta := mergeMeta(c.Meta, map[string]string{
		MetaTaint:      taintName(taint),
		MetaWorld:      s.world,
		MetaArgsDigest: digest,
		MetaCaptureSeq: utoa(c.SeqNo),
	})

	call := turnbench.Call{Tool: c.Tool, Meta: meta}

	// THE EGRESS FLOOR. Persist the raw payload ONLY if the sink's own policy would let
	// it egress. A sensitive-sink call fed tainted data is exactly what the live SinkGate
	// denies; the recorder must not be the side channel that leaks those bytes in clear.
	if s.egressBlocked(c) {
		call.Meta[MetaRedacted] = "true"
		// A digest-only placeholder: the call is still replayable for verdict/floor
		// adjudication (tool+meta+taint drive deny/arg-name rules), but the bytes the
		// floor would block are not in the recording.
		call.Args = json.RawMessage(`{"__redacted__":"` + digest + `"}`)
		s.redacted++
	} else {
		call.Args = json.RawMessage(append([]byte(nil), args...))
	}

	s.calls = append(s.calls, call)
	s.recorded++
}

// Trace returns the assembled payload-bearing trace — the artifact that round-trips
// through turnbench.RunPolicyReplay / turnbench.Run. The returned Trace is a copy; the
// sink keeps recording into its own slice.
func (s *TraceSink) Trace() *turnbench.Trace {
	s.mu.Lock()
	defer s.mu.Unlock()
	calls := make([]turnbench.Call, len(s.calls))
	copy(calls, s.calls)
	return &turnbench.Trace{SliceID: s.sliceID, Calls: calls}
}

// Stats reports the completeness counters: every call offered, every one recorded, the
// gap (dropped, the out-of-band path that MUST be zero for a total capture), and how many
// recorded calls had their payload redacted by the egress floor.
func (s *TraceSink) Stats() (total, recorded, dropped, redacted uint64) {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.total, s.recorded, s.dropped, s.redacted
}

// Total is every EvSubmit the sink was offered.
func (s *TraceSink) Total() uint64 { return load(s, func() uint64 { return s.total }) }

// Recorded is the number of calls written into the trace.
func (s *TraceSink) Recorded() uint64 { return load(s, func() uint64 { return s.recorded }) }

// Dropped is the number of offered calls the sink could NOT record (unresolvable Ref /
// nil call). For a TOTAL capture this is zero; a non-zero value is the honest signal that
// the trace is incomplete and a fleet-scale claim built on it must be qualified.
func (s *TraceSink) Dropped() uint64 { return load(s, func() uint64 { return s.dropped }) }

// Complete reports the "trace is total" witness: every offered call was recorded, none
// fell to an out-of-band path. Total() == Recorded()+Dropped() is an invariant the sink
// maintains by construction; Complete() additionally asserts Dropped()==0.
func (s *TraceSink) Complete() bool {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.dropped == 0 && s.total == s.recorded
}

// ---------------------------------------------------------------------------
// helpers
// ---------------------------------------------------------------------------

func (s *TraceSink) resolveArgs(c *abi.ToolCall) ([]byte, bool) {
	r := c.Args
	if r.Kind == abi.RefInline {
		if len(r.Inline) == 0 {
			return []byte("{}"), true
		}
		return r.Inline, true
	}
	if s.res == nil {
		return nil, false
	}
	b, err := s.res.Resolve(context.Background(), r)
	if err != nil {
		return nil, false
	}
	if len(b) == 0 {
		b = []byte("{}")
	}
	return b, true
}

// flowTaint is the IFC taint flowing through this call: the most restrictive of the
// call's own provenance (its source class), the session high-water mark in the ledger,
// and the call's Args.Taint — but Args.Taint is consulted ONLY for its non-default
// Quarantined value. abi.TaintTainted is the enum ZERO value, so an unstamped Ref is
// indistinguishable from a tainted one; trusting it would mislabel every trusted-local
// read as tainted (the exact trap ifc.SinkGate documents). This is the label the captured
// payload is retained UNDER.
func (s *TraceSink) flowTaint(c *abi.ToolCall) abi.TaintLabel {
	t := provenance.Taint(c, nil) // source-class taint (TrustedLocal => trusted, else tainted)
	if s.ledger != nil {
		if hw := s.ledger.Level(c.TraceID); rank(hw) > rank(t) {
			t = hw
		}
	}
	if c.Args.Taint == abi.TaintQuarantined && rank(abi.TaintQuarantined) > rank(t) {
		t = abi.TaintQuarantined
	}
	return t
}

// egressBlocked is the sink's own egress floor, and it mirrors ifc.SinkGate.Adjudicate
// EXACTLY — the recorder must be no more and no less restrictive than the live floor, or
// it would either leak what the floor blocks or redact a payload the floor would have
// served (which silently corrupts a capture-fidelity replay). The flow into the sink is
// the session's control-flow HIGH-WATER MARK (the ledger), NOT the call's source-class
// provenance: the live gate uses the ledger because the source-class default (Tainted) is
// the enum zero value and would block every egress (and would, e.g., redact a fetch_policy
// the live run legitimately serves while the session is still clean). Args.Taint is
// consulted only for its non-default Quarantined value, exactly as SinkGate does.
func (s *TraceSink) egressBlocked(c *abi.ToolCall) bool {
	sink := ifc.Classify(context.Background(), c, s.policy)
	if sink == ifc.SinkNone {
		return false // not a sink: nothing to gate
	}
	flow := abi.TaintTrusted
	if s.ledger != nil {
		flow = s.ledger.Level(c.TraceID)
	}
	if c.Args.Taint == abi.TaintQuarantined && rank(abi.TaintQuarantined) > rank(flow) {
		flow = abi.TaintQuarantined
	}
	if !ifc.Dangerous(flow) {
		return false // clean data to a sink is fine to record
	}
	// Only the sink classes this policy gates are blocked — mirror the live gate's
	// gated-set decision (the default exempts EXEC; StrictGatedSinks gates it too), or
	// the recorder would redact a Bash the live floor serves.
	if !s.policy.Gates(sink) {
		return false
	}
	// A tainted flow into a sensitive sink. Honor the policy's authorization escape
	// exactly as the live gate would, so the recorder is no more (and no less)
	// restrictive than the floor it mirrors.
	if s.policy.Authorize != nil && s.policy.Authorize(c, sink) {
		return false
	}
	return true
}

func rank(t abi.TaintLabel) int {
	switch t {
	case abi.TaintTrusted:
		return 0
	case abi.TaintTainted:
		return 1
	case abi.TaintQuarantined:
		return 2
	}
	return 1 // unknown => tainted (fail-closed)
}

func taintName(t abi.TaintLabel) string {
	switch t {
	case abi.TaintTrusted:
		return "trusted"
	case abi.TaintTainted:
		return "tainted"
	case abi.TaintQuarantined:
		return "quarantined"
	}
	return "unknown"
}

func mergeMeta(base, add map[string]string) map[string]string {
	out := make(map[string]string, len(base)+len(add))
	keys := make([]string, 0, len(base))
	for k := range base {
		keys = append(keys, k)
	}
	sort.Strings(keys)
	for _, k := range keys {
		out[k] = base[k]
	}
	for k, v := range add {
		out[k] = v
	}
	return out
}

func sha256Sum(b []byte) []byte {
	h := sha256.Sum256(b)
	return h[:]
}

func utoa(n uint64) string {
	if n == 0 {
		return "0"
	}
	var buf [20]byte
	i := len(buf)
	for n > 0 {
		i--
		buf[i] = byte('0' + n%10)
		n /= 10
	}
	return string(buf[i:])
}

func load(s *TraceSink, f func() uint64) uint64 {
	s.mu.Lock()
	defer s.mu.Unlock()
	return f()
}
