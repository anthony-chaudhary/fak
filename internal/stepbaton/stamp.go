// Package stepbaton is the pre-resume step-advice stamp: the durable, cross-restart
// carrier of the managed-context DECISION (a StepClass) captured while the trace is
// still live, so a resuming successor can read what the window pressure WAS before the
// trace rotated. It exists because of the witnessed resume-boundary break
// (docs/notes/CONCEPT-IDEAL-WORKING-CONDITIONS-2026-07-09.md §2.3, "consumed by us"):
// the gateway's live step-advice (internal/gateway ctxvalue.go, CtxStepAdvice) is
// scoped to one gateway/trace lifetime, so at SessionStart on a resume the trace
// reports phase:fresh / step_class:unknown and the pre-resume decision is gone. The
// only fix is to capture the decision at the LAST live moment (a PreCompact or Stop
// hook, trace still alive) and stash it to a durable per-session file the resume path
// can read back — a write+read pair across the process boundary.
//
// SCOPE — this is the PRODUCER-CORE rung only. It defines the stamp vocabulary,
// persists it durably (atomic write), and reads it back. It deliberately does NOT:
//   - capture the live report itself (the hook that projects a gateway.CtxValueReport
//     into New() lands in a guard hook — separate wiring), nor
//   - inject the carried line on resume (the consumer lands in the SessionStart rule /
//     internal/sessionsteer, concurrent #3512 territory — separate wiring).
// Like internal/relay's baton and armtriggers, it takes plain SCALARS, never a
// gateway type, so it stays dependency-free and cannot form an import cycle: the hook
// that already holds the gateway report does the projection and calls New().
package stepbaton

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strings"
)

// Schema is the wire/schema tag every stamp carries. Bump it (never mutate a shipped
// field's meaning) when the stamp shape changes, so a stale file from an older writer
// is recognizably a different schema rather than silently misread.
const Schema = "fak.stepadvice.stamp.v1"

// The closed step-advice vocabulary, mirrored verbatim from internal/gateway
// ctxvalue.go (StepClassAny..StepClassUnknown). Kept as its own copy — not imported —
// so this package stays free of the gateway dependency; ValidStepClass is the guard
// against the two vocabularies drifting (a stamp written with an off-vocabulary class
// is normalized to Unknown, exactly the gateway's own fail-closed default).
const (
	StepAny        = "any"
	StepBounded    = "bounded"
	StepCheckpoint = "checkpoint"
	StepRebuild    = "rebuild"
	StepUnknown    = "unknown"
)

// ValidStepClass reports whether s is a member of the closed vocabulary.
func ValidStepClass(s string) bool {
	switch s {
	case StepAny, StepBounded, StepCheckpoint, StepRebuild, StepUnknown:
		return true
	default:
		return false
	}
}

// NormalizeStepClass folds any input to a vocabulary member, failing CLOSED to
// StepUnknown for anything off-vocabulary — the same conservative default the gateway's
// adviseCtxStep uses when it has no evidence. A carryover we cannot classify must never
// masquerade as a confident "any".
func NormalizeStepClass(s string) string {
	s = strings.TrimSpace(s)
	if ValidStepClass(s) {
		return s
	}
	return StepUnknown
}

// Stamp is one durable capture of the managed-context decision at a live boundary. It
// carries the DECISION plus the numbers behind it — never a transcript recap — so the
// successor can weigh it, not obey it. Every field is a scalar the capturing hook reads
// off the live gateway.CtxValueReport at capture time.
type Stamp struct {
	// Schema is the version tag (Schema const). New() stamps it.
	Schema string `json:"schema"`
	// TraceID is the trace this was captured FROM — lineage only. The successor records
	// it to link its provenance back here; it never trusts the trace to still be live
	// (on resume it is not).
	TraceID string `json:"trace_id,omitempty"`
	// StepClass is the carried decision: a closed-vocabulary member (fail-closed to
	// StepUnknown). This is the load-bearing field — the reason the stamp exists.
	StepClass string `json:"step_class"`
	// Basis is the gateway's deciding axis (token_headroom | event_cadence |
	// context_event | none), carried so the successor can see WHY the class fired.
	Basis string `json:"basis,omitempty"`
	// Reason is the gateway's one-line, display-only explanation. Never a recap.
	Reason string `json:"reason,omitempty"`
	// ResidentTokens / BudgetTokens are the deciding numbers at capture: observed
	// resident context against the budget it was measured on. Headroom is derivable
	// (budget-resident) so it is not stored, keeping the stamp minimal and unambiguous.
	ResidentTokens int `json:"resident_tokens,omitempty"`
	BudgetTokens   int `json:"budget_tokens,omitempty"`
	// Phase is the session phase at capture (e.g. "guard"): a LIVE phase, the very thing
	// a resume's "fresh" cannot recover.
	Phase string `json:"phase,omitempty"`
	// CapturedAtSHA is the git commit observed at capture — the anchor that lets the
	// successor situate the decision against ground truth. Empty is allowed (no anchor
	// observed); a consumer treats an empty anchor as "situate loosely", never an error.
	CapturedAtSHA string `json:"captured_at_sha,omitempty"`
}

// New builds a stamp from the scalars a capturing hook reads off the live report,
// normalizing the step class fail-closed and stamping the schema. It is the single
// projection seam: a hook holding a gateway.CtxValueReport maps its fields to these
// arguments, so this package never imports the gateway.
func New(traceID, stepClass, basis, reason, phase, capturedAtSHA string, residentTokens, budgetTokens int) Stamp {
	return Stamp{
		Schema:         Schema,
		TraceID:        strings.TrimSpace(traceID),
		StepClass:      NormalizeStepClass(stepClass),
		Basis:          strings.TrimSpace(basis),
		Reason:         strings.TrimSpace(reason),
		ResidentTokens: residentTokens,
		BudgetTokens:   budgetTokens,
		Phase:          strings.TrimSpace(phase),
		CapturedAtSHA:  strings.TrimSpace(capturedAtSHA),
	}
}

// ShouldCarry reports whether this decision is worth injecting on resume: only the
// classes that change what a successor should DO — checkpoint (land in-flight state
// first) and rebuild (re-anchor from durable state first). any / bounded / unknown add
// no steer, so carrying them would be noise. Advisory: the consumer may override, but
// co-locating the predicate keeps that consumer a one-liner and pins the policy under
// test here.
func (s Stamp) ShouldCarry() bool {
	return s.StepClass == StepCheckpoint || s.StepClass == StepRebuild
}

// Line renders the one-line carryover a resume consumer injects into the successor's
// first-turn context — the human/agent-readable form of the decision. Fixed shape so a
// scan reads cleanly; omits empty optional fields.
func (s Stamp) Line() string {
	var b strings.Builder
	fmt.Fprintf(&b, "managed-context carryover: last live step=%s", s.StepClass)
	if s.Basis != "" {
		fmt.Fprintf(&b, " basis=%s", s.Basis)
	}
	if s.BudgetTokens > 0 {
		fmt.Fprintf(&b, " resident=%d budget=%d", s.ResidentTokens, s.BudgetTokens)
	}
	if s.Phase != "" {
		fmt.Fprintf(&b, " phase=%s", s.Phase)
	}
	if s.CapturedAtSHA != "" {
		fmt.Fprintf(&b, " at=%s", s.CapturedAtSHA)
	}
	if s.Reason != "" {
		fmt.Fprintf(&b, " reason=%q", s.Reason)
	}
	return b.String()
}

// Marshal encodes a stamp as indented JSON with a trailing newline: the durable file is
// meant to be human-inspectable (an operator can cat it), and the byte shape is stable
// because encoding/json emits struct fields in declaration order.
func Marshal(s Stamp) ([]byte, error) {
	data, err := json.MarshalIndent(s, "", "  ")
	if err != nil {
		return nil, err
	}
	return append(data, '\n'), nil
}

// Unmarshal decodes a stamp from its wire bytes.
func Unmarshal(data []byte) (Stamp, error) {
	var s Stamp
	if err := json.Unmarshal(data, &s); err != nil {
		return Stamp{}, err
	}
	return s, nil
}

// Path is the durable per-session file path for a stamp: <dir>/stepadvice-<id>.json.
// The session id is sanitized to a single safe path segment (any character outside
// [A-Za-z0-9._-] becomes '_') so a hostile or oddly-shaped id can never escape dir via
// a separator or "..".
func Path(dir, sessionID string) string {
	return filepath.Join(dir, "stepadvice-"+sanitizeSegment(sessionID)+".json")
}

func sanitizeSegment(id string) string {
	id = strings.TrimSpace(id)
	if id == "" {
		return "unknown"
	}
	var b strings.Builder
	for _, r := range id {
		switch {
		case r >= 'a' && r <= 'z', r >= 'A' && r <= 'Z', r >= '0' && r <= '9', r == '.', r == '_', r == '-':
			b.WriteRune(r)
		default:
			b.WriteRune('_')
		}
	}
	out := b.String()
	// A segment of all-dots (".", "..") would still be path-significant after the
	// per-rune pass (dots are allowed), so fold those to a safe literal.
	if strings.Trim(out, ".") == "" {
		return "unknown"
	}
	return out
}

// Write persists the stamp to path atomically: it writes a sibling temp file and
// renames it over path, so a failed or partial write leaves any pre-existing stamp
// intact rather than truncating it (the same all-or-nothing swap the guard settings
// writer uses). The parent directory is created if absent. On Windows the rename is a
// same-directory replace (MoveFileEx), matching the guard writer's assumption.
func Write(path string, s Stamp) error {
	data, err := Marshal(s)
	if err != nil {
		return err
	}
	dir := filepath.Dir(path)
	if err := os.MkdirAll(dir, 0o700); err != nil {
		return err
	}
	tmp, err := os.CreateTemp(dir, filepath.Base(path)+".tmp-*")
	if err != nil {
		return err
	}
	tmpName := tmp.Name()
	defer os.Remove(tmpName) // no-op once the rename below succeeds
	if _, err := tmp.Write(data); err != nil {
		tmp.Close()
		return err
	}
	if err := tmp.Chmod(0o600); err != nil {
		tmp.Close()
		return err
	}
	if err := tmp.Close(); err != nil {
		return err
	}
	return os.Rename(tmpName, path)
}

// Read loads the stamp at path. An ABSENT file is not an error: it returns
// (zero, false, nil), so a resume consumer that finds no carryover is decidable and
// simply injects nothing (fail-open, the way the gateway treats an unknown trace). A
// present-but-corrupt file returns a non-nil error so the caller can log it rather than
// silently treat garbage as "no carryover".
func Read(path string) (Stamp, bool, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		if os.IsNotExist(err) {
			return Stamp{}, false, nil
		}
		return Stamp{}, false, err
	}
	s, err := Unmarshal(data)
	if err != nil {
		return Stamp{}, false, fmt.Errorf("stepbaton: read %s: %w", path, err)
	}
	return s, true, nil
}
