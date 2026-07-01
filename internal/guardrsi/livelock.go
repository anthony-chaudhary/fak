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
)

// LivelockObservation is one refused/quarantined tool-call outcome, identified without
// carrying raw arguments. TraceID scopes the consecutive run to one agent session.
type LivelockObservation struct {
	TraceID     string
	Tool        string
	ArgsDigest  string
	Verdict     string
	Reason      string
	Disposition string
}

// LivelockEnvelope is the structured nudge returned once an identical failing call
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
}

type livelockRun struct {
	key   string
	count int
	last  LivelockObservation
}

// LivelockDetector tracks consecutive identical failures per trace. It is intentionally
// small and caller-synchronized; gateway guards it with its server mutex.
type LivelockDetector struct {
	threshold int
	byTrace   map[string]livelockRun
}

func NewLivelockDetector(threshold int) *LivelockDetector {
	if threshold <= 0 {
		threshold = DefaultLivelockThreshold
	}
	return &LivelockDetector{threshold: threshold, byTrace: map[string]livelockRun{}}
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
	if d == nil || strings.TrimSpace(obs.TraceID) == "" {
		return LivelockEnvelope{}, false
	}
	if d.threshold <= 0 {
		d.threshold = DefaultLivelockThreshold
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
	switch strings.ToUpper(obs.Reason) {
	case "STALE_BASE_DELETION", "OFF_TRUNK", "MERGE_IN_PROGRESS", "COLLISION_RISK":
		return "change_approach_fetch_merge_or_wait_for_lease"
	case "LOOP_DONE_UNWITNESSED", "CORE_SELF_MODIFY", "INDETERMINATE":
		return "change_approach_escalate_or_not_yet_with_witness"
	default:
		return "change_approach_fetch_merge_escalate_or_not_yet_with_witness"
	}
}
