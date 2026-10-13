// Package stepspan provides bounded, asynchronous step telemetry shared by all planes.
package stepspan

import (
	"crypto/rand"
	"encoding/hex"
	"errors"
	"strings"
)

const Schema = "fak.step-span.v1"

type Plane string

const (
	Ops     Plane = "ops"
	Agent   Plane = "agent"
	Planner Plane = "planner"
	Router  Plane = "router"
	Gateway Plane = "gateway"
	Sched   Plane = "sched"
	Engine  Plane = "engine"
	Kernel  Plane = "kernel"
)

type Kind string

const (
	Run           Kind = "run"
	Wave          Kind = "wave"
	Spawn         Kind = "spawn"
	Invocation    Kind = "invocation"
	Turn          Kind = "turn"
	ToolCall      Kind = "tool_call"
	LLMCall       Kind = "llm_call"
	Adjudicate    Kind = "adjudicate"
	Plan          Kind = "plan"
	PlanStep      Kind = "plan_step"
	Observe       Kind = "observe"
	Replan        Kind = "replan"
	Request       Kind = "request"
	Admission     Kind = "admission"
	RouteDecision Kind = "route_decision"
	Attempt       Kind = "attempt"
	AdmissionWait Kind = "admission_wait"
	Stream        Kind = "stream"
	Tick          Kind = "tick"
	PrefixLookup  Kind = "prefix_lookup"
	PrefillChunk  Kind = "prefill_chunk"
	Cohort        Kind = "cohort"
	DecodeStep    Kind = "decode_step"
	SpecRound     Kind = "spec_round"
	Sample        Kind = "sample"
	KVAlloc       Kind = "kv_alloc"
	KVEvict       Kind = "kv_evict"
	Preempt       Kind = "preempt"
	Layer         Kind = "layer"
	Op            Kind = "op"
)

func Kinds(p Plane) []Kind {
	switch p {
	case Ops:
		return []Kind{Run, Wave, Spawn, Invocation}
	case Agent:
		return []Kind{Turn, ToolCall, LLMCall, Adjudicate}
	case Planner:
		return []Kind{Plan, PlanStep, Observe, Replan}
	case Router:
		return []Kind{Request, Admission, RouteDecision, Attempt}
	case Gateway:
		return []Kind{Request, AdmissionWait, Stream}
	case Sched:
		return []Kind{Tick}
	case Engine:
		return []Kind{PrefixLookup, PrefillChunk, Cohort, DecodeStep, SpecRound, Sample, KVAlloc, KVEvict, Preempt}
	case Kernel:
		return []Kind{Layer, Op}
	}
	return nil
}

type Status string

const (
	OK        Status = "ok"
	Error     Status = "error"
	Refused   Status = "refused"
	Dropped   Status = "dropped"
	Preempted Status = "preempted"
	Cancelled Status = "cancelled"
)

type Link struct {
	TraceID string `json:"trace_id"`
	SpanID  string `json:"span_id"`
}
type IDs struct {
	RunID     string   `json:"run_id,omitempty"`
	SessionID string   `json:"session_id,omitempty"`
	Turn      int      `json:"turn,omitempty"`
	CallID    string   `json:"call_id,omitempty"`
	RequestID string   `json:"request_id,omitempty"`
	Attempt   int      `json:"attempt,omitempty"`
	SeqIDs    []string `json:"seq_ids,omitempty"`
	CohortID  uint64   `json:"cohort_id,omitempty"`
	SchedIter uint64   `json:"sched_iter,omitempty"`
}
type Attrs struct {
	Path          string `json:"path,omitempty"`
	Lanes         int    `json:"lanes,omitempty"`
	Tokens        int    `json:"tokens,omitempty"`
	PrefillTokens int    `json:"prefill_tokens,omitempty"`
	PrefixMatched int    `json:"prefix_matched,omitempty"`
	Running       int    `json:"running,omitempty"`
	Waiting       int    `json:"waiting,omitempty"`
	TokenBudget   int    `json:"token_budget,omitempty"`
	Proposed      int    `json:"proposed,omitempty"`
	Accepted      int    `json:"accepted,omitempty"`
	Backend       string `json:"backend,omitempty"`
	Layer         int    `json:"layer,omitempty"`
	Op            string `json:"op,omitempty"`
	Model         string `json:"model,omitempty"`
	Tool          string `json:"tool,omitempty"`
	Verdict       string `json:"verdict,omitempty"`
}
type Span struct {
	Schema        string `json:"schema"`
	TraceID       string `json:"trace_id"`
	SpanID        string `json:"span_id"`
	ParentSpanID  string `json:"parent_span_id,omitempty"`
	Links         []Link `json:"links,omitempty"`
	Plane         Plane  `json:"plane"`
	Kind          Kind   `json:"kind"`
	Name          string `json:"name"`
	StartUnixNano int64  `json:"start_unix_nano"`
	EndUnixNano   int64  `json:"end_unix_nano"`
	Status        Status `json:"status"`
	Host          string `json:"host"`
	PID           int    `json:"pid"`
	Service       string `json:"service"`
	IDs           IDs    `json:"ids"`
	Attrs         Attrs  `json:"attrs"`
}

var ErrInvalidSpan = errors.New("invalid step span")
var ErrInvalidTraceparent = errors.New("invalid W3C traceparent")

func validID(s string, n int) bool {
	if len(s) != n || s == strings.Repeat("0", n) {
		return false
	}
	for _, c := range s {
		if !(c >= '0' && c <= '9' || c >= 'a' && c <= 'f') {
			return false
		}
	}
	return true
}
func ValidTraceID(s string) bool { return validID(s, 32) }
func ValidSpanID(s string) bool  { return validID(s, 16) }
func newID(n int) (string, error) {
	b := make([]byte, n)
	for {
		if _, err := rand.Read(b); err != nil {
			return "", err
		}
		s := hex.EncodeToString(b)
		if validID(s, n*2) {
			return s, nil
		}
	}
}
func NewTraceID() (string, error) { return newID(16) }
func NewSpanID() (string, error)  { return newID(8) }

type TraceContext struct {
	TraceID string
	SpanID  string
	Flags   byte
}

func (c TraceContext) Traceparent() (string, error) {
	if !ValidTraceID(c.TraceID) || !ValidSpanID(c.SpanID) {
		return "", ErrInvalidTraceparent
	}
	return "00-" + c.TraceID + "-" + c.SpanID + "-" + hex.EncodeToString([]byte{c.Flags}), nil
}
func ParseTraceparent(s string) (TraceContext, error) {
	parts := strings.Split(s, "-")
	if len(parts) != 4 || parts[0] != "00" || !ValidTraceID(parts[1]) || !ValidSpanID(parts[2]) || len(parts[3]) != 2 || strings.ToLower(parts[3]) != parts[3] {
		return TraceContext{}, ErrInvalidTraceparent
	}
	b, err := hex.DecodeString(parts[3])
	if err != nil {
		return TraceContext{}, ErrInvalidTraceparent
	}
	return TraceContext{parts[1], parts[2], b[0]}, nil
}
func (s Span) Validate() error {
	if s.Schema != Schema || !ValidTraceID(s.TraceID) || !ValidSpanID(s.SpanID) || s.ParentSpanID != "" && !ValidSpanID(s.ParentSpanID) || s.StartUnixNano < 0 || s.EndUnixNano < s.StartUnixNano {
		return ErrInvalidSpan
	}
	found := false
	for _, k := range Kinds(s.Plane) {
		if k == s.Kind {
			found = true
			break
		}
	}
	if !found {
		return ErrInvalidSpan
	}
	switch s.Status {
	case OK, Error, Refused, Dropped, Preempted, Cancelled:
	default:
		return ErrInvalidSpan
	}
	for _, l := range s.Links {
		if !ValidTraceID(l.TraceID) || !ValidSpanID(l.SpanID) {
			return ErrInvalidSpan
		}
	}
	return nil
}
