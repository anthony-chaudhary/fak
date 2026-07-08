package memq

import (
	"errors"
	"fmt"
	"regexp"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/memoryread"
)

// Deny-by-structure memory-write adjudication (#2912). Hermes curates durable
// memory by asking the model to honor a prose "Do NOT capture" list in the review
// prompt (environment failures, negative tool claims, transient errors) —
// enforcement is hope-the-model-complies, and #3006 is the failure: a 74 KiB skill
// body copied verbatim into a single memory entry. A list in a prompt is not an
// invariant; a distracted or injected model writes junk to durable memory.
//
// fak adjudicates a tool call deny-by-structure; a durable memory write is just
// another governed syscall. This is that structural arm: a candidate durable write
// is judged by SHAPE — its size, and the lexical signature of the three junk classes
// the Hermes list names but cannot enforce (a transient environment error, a
// negative tool-INVOCATION claim, a one-off session narrative bound to this run) —
// and refused with a reason from a closed vocabulary BEFORE it reaches storage. No
// prompt, no model call, no override string: structure decides, so an injected or
// distracted model cannot talk its way past it (unlike the ephemeral gate in
// ephemeral_gate.go, whose situational refusal a caller MAY override with an explicit
// reclassification). The one-off-narrative arm is issue #2836's delta over #2912.
//
// The rules are deliberately conservative — a durable fact is a distilled
// generalization, so the refusals target the shapes that are junk BY CONSTRUCTION
// (a document-sized verbatim blob; a bare event line reporting a transient failure
// or a failed tool invocation) and leave a legitimate distilled fact untouched even
// when it MENTIONS a retry, a missing binary, or a rate-limit header. Over-refusing
// an honest report is the named risk (the issue's confusion note); the rules err
// toward under-refusing. In particular the negative-tool arm targets a failed
// INVOCATION ("failed to run X", "the call exited nonzero") — a session-transient
// narrative — NOT a durable CAPABILITY fact ("grep is not on PATH on this host"),
// which is exactly the kind of environment fact worth keeping. Likewise the one-off
// arm targets a fact BOUND to this run by a temporal deictic ("this session", "just
// now") — the fourth "Do NOT capture" class — NOT a durable fact that merely names a
// session/run/turn ("the session token expires in 30m").
const (
	// AdmitOK is the verdict for a candidate write that passes every structural rule.
	AdmitOK = "ok"
	// RefuseOversizeVerbatim refuses a durable write whose body exceeds the
	// single-fact byte bound — a document-sized verbatim copy (the #3006 74 KiB
	// skill body), never a distilled fact.
	RefuseOversizeVerbatim = "oversize_verbatim"
	// RefuseTransientError refuses a durable write whose fact is a transient
	// environment failure (a connection reset, a timeout, a rate-limit wall) — a
	// one-off event, not a durable invariant.
	RefuseTransientError = "transient_error"
	// RefuseNegativeToolClaim refuses a durable write whose fact is a negative
	// tool-INVOCATION claim (a call that failed / exited nonzero / produced no
	// output this run) — a session-transient narrative, not a durable fact.
	RefuseNegativeToolClaim = "negative_tool_claim"
	// RefuseOneOffNarrative refuses a durable write whose fact is a one-off session
	// narrative — a thin event line anchored to THIS run/session/turn by a temporal
	// deictic ("this session", "just now", "earlier today"). A durable fact is a
	// timeless generalization and never needs to bind itself to the current run; the
	// bound narrative is the fourth "Do NOT capture" class the Hermes prose list names
	// but cannot enforce (#2836).
	RefuseOneOffNarrative = "one_off_narrative"
)

// MaxDurableFactBytes is the single-fact size bound. A durable memory entry is one
// distilled fact (a sentence or short paragraph); anything past this is a
// document-sized verbatim copy that belongs in docs/, not a memory cell. #3006's
// 74 KiB write is more than 4x over. The bound sits well above any real distilled
// fact (a few hundred bytes to low-single-KiB), so it refuses the copied-document
// shape without touching a legitimate long fact.
const MaxDurableFactBytes = 16 << 10 // 16 KiB

// thinFactBytes bounds the lexical rules to a "thin" fact — a bare event line or a
// single sentence. A junk transient/failed-tool write IS its event line; a distilled
// durable fact that legitimately discusses retries or a failed probe is either longer
// than this or, if short, does not carry a failure-report phrasing. Requiring thinness
// keeps an incidental mention deep in a rich fact from tripping a refusal.
const thinFactBytes = 400

// ErrWriteRefused wraps a deny-by-structure refusal of a durable write (#2912). The
// verdict's closed-vocabulary Reason and its Detail are formatted into the error; a
// caller that needs the machine token can call AdjudicateMemoryWrite directly.
var ErrWriteRefused = errors.New("memq: durable memory write refused by structure")

// AdmissionVerdict is the deny-by-structure ruling on a candidate durable memory
// write. Reason is drawn from the closed vocabulary above and is AdmitOK exactly
// when Admit is true; Detail names the offending structure (the measured size, the
// matched phrase) so a refusal is auditable from the verdict alone.
type AdmissionVerdict struct {
	Admit  bool
	Reason string
	Detail string
}

// transientErrorRE matches the FAILURE-REPORT phrasings of a transient environment
// error — the event, not a durable fact that merely names the mechanism. It matches
// "the request timed out" but not "set the timeout to 30s"; "rate limited" / "rate
// limit exceeded" but not "respects the rate-limit header".
var transientErrorRE = regexp.MustCompile(`(?i)(` +
	`connection (refused|reset|timed out)|` +
	`econnrefused|econnreset|etimedout|epipe|` +
	`(request|operation|call|read|write|it) timed out|` +
	`timed out (waiting|after|while|connecting)|` +
	`rate ?limited|` +
	`rate ?limit (exceeded|hit|reached)|` +
	`too many requests|` +
	`(temporarily|momentarily) unavailable|` +
	`service unavailable|server unavailable|` +
	`deadline exceeded|` +
	`try again later|please try again|` +
	`transient (error|failure)|` +
	`broken pipe|` +
	`network (is )?unreachable` +
	`)`)

// negativeToolClaimRE matches the FAILED-INVOCATION phrasings of a tool call — a
// session-transient narrative ("failed to run X", "the call exited nonzero"). It
// deliberately does NOT match a durable CAPABILITY fact ("not on PATH", "not
// installed", "not available", "command not found"): those can be genuine, durable
// host facts worth keeping, and over-refusing them is the issue's named risk.
var negativeToolClaimRE = regexp.MustCompile(`(?i)(` +
	`failed to (run|invoke|call|execute|exec|spawn|launch|start|connect|open|read|write|fetch|load|reach)|` +
	`(command|tool|call|invocation|the command|the tool) failed|` +
	`returned (a |an )?(non-?zero|error|failure)|` +
	`exited (with )?(code |status )?(non-?zero|[1-9])|` +
	`non-?zero exit|` +
	`no output (from|returned|produced)|` +
	`error (running|invoking|calling|executing|spawning|launching)|` +
	`could not (run|invoke|execute|exec|spawn|start|launch|reach)|` +
	`unable to (run|invoke|execute|exec|spawn|start|launch|reach|connect)` +
	`)`)

// oneOffNarrativeRE matches the SESSION-DEICTIC anchor of a one-off narrative — a
// bare event line bound to THIS run/session/turn ("this session", "this run", "this
// turn", "just now", "earlier today", "a moment ago"). A durable fact is a timeless
// generalization and never needs to say "this session", so the deictic binding is the
// structural tell of a one-off event, not a durable invariant. It deliberately does
// NOT match a durable fact that merely names a session/run/turn WITHOUT the deictic
// binding ("the session token expires in 30m", "a dry run is safe to repeat") — over-
// refusing those is the issue's named risk (#2836). The \b guards keep "this run"
// from firing on "this runtime"/"this running".
var oneOffNarrativeRE = regexp.MustCompile(`(?i)(` +
	`this (session|run|turn)\b|` +
	`\bjust now\b|` +
	`\bearlier today\b|` +
	`\bmoments? ago\b` +
	`)`)

// AdjudicateMemoryWrite is the deny-by-structure rule set (#2912). It judges a
// candidate durable memory write by structure alone and returns a verdict whose
// Reason is from the closed vocabulary above. It is pure and deterministic: the same
// body always yields the same verdict, with no model call, clock, or I/O — so a
// #3006-style verbatim write is refused the same way every time.
func AdjudicateMemoryWrite(body []byte) AdmissionVerdict {
	// Size is judged on the RAW write (frontmatter included): a verbatim blob is
	// oversize whether or not it carries a YAML header. Short-circuit here so a huge
	// body is never scanned lexically.
	if n := len(body); n > MaxDurableFactBytes {
		return AdmissionVerdict{
			Reason: RefuseOversizeVerbatim,
			Detail: fmt.Sprintf("%d bytes exceeds the %d-byte single-fact bound", n, MaxDurableFactBytes),
		}
	}
	// The lexical rules judge the FACT (frontmatter stripped, trimmed) — the durable
	// fact's load-bearing text, not its metadata block.
	fact := strings.TrimSpace(memoryread.StripFrontmatter(string(body)))
	if fact == "" {
		return AdmissionVerdict{Admit: true, Reason: AdmitOK}
	}
	if thin := len(fact) <= thinFactBytes && strings.Count(fact, "\n") <= 4; thin {
		if m := transientErrorRE.FindString(fact); m != "" {
			return AdmissionVerdict{Reason: RefuseTransientError, Detail: "fact is a transient-error narrative: " + strings.ToLower(m)}
		}
		if m := negativeToolClaimRE.FindString(fact); m != "" {
			return AdmissionVerdict{Reason: RefuseNegativeToolClaim, Detail: "fact is a failed-tool-invocation narrative: " + strings.ToLower(m)}
		}
		if m := oneOffNarrativeRE.FindString(fact); m != "" {
			return AdmissionVerdict{Reason: RefuseOneOffNarrative, Detail: "fact is a one-off session narrative: " + strings.ToLower(m)}
		}
	}
	return AdmissionVerdict{Admit: true, Reason: AdmitOK}
}
