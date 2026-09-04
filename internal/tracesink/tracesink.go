// Package tracesink provides a payload-bearing, IFC-labeled trajectory sink.
//
// Unlike the durable decision journal which persists only cryptographic digests,
// TraceSink captures raw argument payloads and provenance metadata from submitted
// tool calls into a turnbench.Trace suitable for offline policy replay. Captures
// enforce the kernel egress floor by redacting sensitive payloads to digest
// placeholders when tainted data flows to sensitive sinks.
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
	"github.com/anthony-chaudhary/fak/internal/pathutil"
	"github.com/anthony-chaudhary/fak/internal/provenance"
	"github.com/anthony-chaudhary/fak/internal/turnbench"
)

const (
	// MetaTaint records the IFC provenance taint of the call's source or flow at capture time.
	MetaTaint = "trace_taint"
	// MetaWorld pins the capture world-version for reproducible replay comparison.
	MetaWorld = "trace_world"
	// MetaRedacted flags whether the egress floor redacted the payload to a digest placeholder.
	MetaRedacted = "payload_redacted"
	// MetaArgsDigest preserves the sha256 of the original args for content-addressed identity.
	MetaArgsDigest = "args_digest"
	// MetaCaptureSeq records the kernel ToolCall sequence number of the captured submission.
	MetaCaptureSeq = "capture_seq"
	// MetaCaptureRefused marks a whole-source ingest refusal recorded as an audit row.
	MetaCaptureRefused = "capture_refused"
	// MetaCaptureReason records why an ingest source was refused by security classification.
	MetaCaptureReason = "capture_reason"
	// MetaCaptureSourceDigest records the digest of an ingest source denied during capture.
	MetaCaptureSourceDigest = "capture_source_digest"
)

// TraceSink captures submitted kernel calls into a payload-bearing turnbench.Trace.
type TraceSink struct {
	mu      sync.Mutex
	sliceID string
	res     abi.Resolver
	ledger  *ifc.Ledger
	policy  ifc.Policy
	world   string
	clock   func() time.Time

	calls    []turnbench.Call
	total    uint64
	recorded uint64
	dropped  uint64
	redacted uint64
	refused  uint64
}

// Options configures a TraceSink recorder instance.
type Options struct {
	// SliceID names the recorded trace artifact.
	SliceID string
	// Resolver materializes call argument references to byte payloads.
	Resolver abi.Resolver
	// Ledger tracks IFC taint levels checked against the egress floor.
	Ledger *ifc.Ledger
	// Policy defines the IFC sink classification rules.
	Policy ifc.Policy
	// World pins the recorded kernel world-version string.
	World string
	// Clock provides time timestamps for deterministic trace generation.
	Clock func() time.Time
}

// NewTraceSink constructs an active recorder initialized with safe defaults.
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

// Subscriptions scopes event delivery strictly to EvSubmit pre-transform calls.
func (s *TraceSink) Subscriptions() []abi.EventKind { return []abi.EventKind{abi.EvSubmit} }

// Emit captures one submitted call into the trace while applying egress gating.
func (s *TraceSink) Emit(ev abi.Event) {
	if ev.Kind != abi.EvSubmit {
		return
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
		s.dropped++
		return
	}

	digest := "sha256:" + hex.EncodeToString(sha256Sum(args))
	if match := pathutil.CheckCaptureJSON(args, c.Meta); match.Refused {
		call := turnbench.Call{
			Tool: c.Tool,
			Meta: map[string]string{
				MetaWorld:               s.world,
				MetaArgsDigest:          digest,
				MetaCaptureSeq:          utoa(c.SeqNo),
				MetaCaptureRefused:      "true",
				MetaCaptureReason:       match.Reason,
				MetaCaptureSourceDigest: match.SourceDigest,
			},
			Args: json.RawMessage(`{"__capture_refused__":"` + match.SourceDigest + `"}`),
		}
		s.calls = append(s.calls, call)
		s.recorded++
		s.refused++
		return
	}
	taint := s.flowTaint(c)

	meta := mergeMeta(c.Meta, map[string]string{
		MetaTaint:      taintName(taint),
		MetaWorld:      s.world,
		MetaArgsDigest: digest,
		MetaCaptureSeq: utoa(c.SeqNo),
	})

	call := turnbench.Call{Tool: c.Tool, Meta: meta}

	if s.egressBlocked(c) {
		call.Meta[MetaRedacted] = "true"
		call.Args = json.RawMessage(`{"__redacted__":"` + digest + `"}`)
		s.redacted++
	} else {
		call.Args = json.RawMessage(append([]byte(nil), args...))
	}

	s.calls = append(s.calls, call)
	s.recorded++
}

// Trace returns a snapshot copy of the assembled payload-bearing trace.
func (s *TraceSink) Trace() *turnbench.Trace {
	s.mu.Lock()
	defer s.mu.Unlock()
	calls := make([]turnbench.Call, len(s.calls))
	copy(calls, s.calls)
	return &turnbench.Trace{SliceID: s.sliceID, Calls: calls}
}

// Stats reports total offered, recorded, dropped, and redacted call counts.
func (s *TraceSink) Stats() (total, recorded, dropped, redacted uint64) {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.total, s.recorded, s.dropped, s.redacted
}

// Total returns the count of every EvSubmit event offered to the sink.
func (s *TraceSink) Total() uint64 { return load(s, func() uint64 { return s.total }) }

// Recorded returns the count of calls written into the trace.
func (s *TraceSink) Recorded() uint64 { return load(s, func() uint64 { return s.recorded }) }

// Dropped returns the count of offered calls that could not be resolved.
func (s *TraceSink) Dropped() uint64 { return load(s, func() uint64 { return s.dropped }) }

// Refused returns the count of denied sources captured as digest audit rows.
func (s *TraceSink) Refused() uint64 { return load(s, func() uint64 { return s.refused }) }

// Complete reports whether every offered call was recorded with zero drops.
func (s *TraceSink) Complete() bool {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.dropped == 0 && s.total == s.recorded
}

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

func (s *TraceSink) flowTaint(c *abi.ToolCall) abi.TaintLabel {
	t := provenance.Taint(c, nil)
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

func (s *TraceSink) egressBlocked(c *abi.ToolCall) bool {
	sink := ifc.Classify(context.Background(), c, s.policy)
	if sink == ifc.SinkNone {
		return false
	}
	flow := abi.TaintTrusted
	if s.ledger != nil {
		flow = s.ledger.Level(c.TraceID)
	}
	if c.Args.Taint == abi.TaintQuarantined && rank(abi.TaintQuarantined) > rank(flow) {
		flow = abi.TaintQuarantined
	}
	if !ifc.Dangerous(flow) {
		return false
	}
	if !s.policy.Gates(sink) {
		return false
	}
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
	return 1
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
