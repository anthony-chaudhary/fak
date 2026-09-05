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
	// CargoCultPrefixSchema versions the cargo-cult command prefixing detection
	// record emitted when a leading advisory/recovery sub-command repeats across
	// consecutive tool calls.
	CargoCultPrefixSchema = "guardrsi.cargo-cult-prefix/1"
	CargoCultPrefixEvent  = "CARGO_CULT_PREFIX_DETECTED"
	// SuggestedChangeCargoCultPrefix is the recommended actionable change when an
	// agent repeatedly prefixes substantive commands with advisory/recovery commands.
	SuggestedChangeCargoCultPrefix = "change_approach_remove_cargo_cult_command_prefix"
	// RefusalCheckpointSchema versions the typed recovery checkpoint nested in a
	// livelock envelope. It is deliberately additive to LivelockSchema: existing
	// callers still receive the advisory/fuse fields, while recovery-aware callers
	// can keep the goal active instead of rendering another denial-only final turn.
	RefusalCheckpointSchema = "guardrsi.refusal-checkpoint/1"
	RefusalCheckpointEvent  = "REFUSAL_RECOVERY_CHECKPOINT"
	// DefaultSemanticRefusalCheckpoint is the hard UX bound for semantically
	// equivalent SELF_MODIFY refusals. The security decision remains DENY, but by
	// the fourth refusal the caller receives a typed recoverable pause rather than
	// needing to infer recovery from another varied shell command.
	DefaultSemanticRefusalCheckpoint = 4
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

// RefusalCheckpointOutcome is the closed outcome vocabulary for a repeated
// semantic refusal checkpoint.
type RefusalCheckpointOutcome string

const (
	// RefusalPausedRecoverable means the refused effect remains prohibited, the
	// active goal should be preserved, and the caller must choose a sanctioned
	// actuator or escalate with a witness rather than retrying command mutations.
	RefusalPausedRecoverable RefusalCheckpointOutcome = "paused_recoverable"
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
	// SemanticIdentity is an optional stable, content-free fingerprint of the
	// guarded target and intended effect. Supplying it lets the detector recognize
	// the same refusal across different tools and argument strings without exposing
	// either value. SELF_MODIFY observations without one use the conservative
	// (reason, tool) fallback so today's gateway catches command-mutation loops;
	// callers that know target/effect boundaries should always supply the sharper
	// identity so genuinely different routes remain distinct.
	SemanticIdentity string
	// Command is an optional direct shell command string. Supplying it lets the
	// detector decompose compound shell commands (e.g. splitting on ';' or '&&')
	// and track repetition of leading advisory/recovery sub-commands across consecutive turns.
	Command string
	// RawArgs is optional raw tool arguments (JSON or plain string) from which
	// a command string can be extracted if Command is omitted.
	RawArgs string
}

// CargoCultPrefixRecord is the typed cargo-cult prefix record emitted when a
// leading advisory/recovery sub-command repeats across consecutive compound commands.
type CargoCultPrefixRecord struct {
	Schema      string `json:"schema"`
	Event       string `json:"event"`
	Prefix      string `json:"prefix"`
	RepeatCount int    `json:"repeat_count"`
	NextAction  string `json:"next_action"`
}

// RefusalCheckpoint is the typed same-turn recovery contract emitted on the
// fourth semantically equivalent SELF_MODIFY refusal. Confirmable is false for
// SELF_MODIFY: replaying or confirming the same bytes never overrides the floor.
// GoalAction instructs the harness to retain the active objective while paused.
type RefusalCheckpoint struct {
	Schema           string                   `json:"schema"`
	Event            string                   `json:"event"`
	Outcome          RefusalCheckpointOutcome `json:"outcome"`
	Reason           string                   `json:"reason"`
	SemanticIdentity string                   `json:"semantic_identity"`
	RepeatCount      int                      `json:"repeat_count"`
	Confirmable      bool                     `json:"confirmable"`
	GoalAction       string                   `json:"goal_action"`
	NextAction       string                   `json:"next_action"`
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
	// IdentityKind is set for semantic refusal folding; an omitted value retains
	// the original exact-call wire shape. SemanticIdentity is always content-free.
	IdentityKind     string `json:"identity_kind,omitempty"`
	SemanticIdentity string `json:"semantic_identity,omitempty"`
	// Checkpoint is present from the fourth equivalent SELF_MODIFY refusal onward.
	// It does not weaken the DENY; it changes the control response from another
	// terminal-looking denial into an explicit recoverable pause contract.
	Checkpoint *RefusalCheckpoint `json:"checkpoint,omitempty"`
	// CommandPrefix and Prefix are the extracted leading sub-command when a compound
	// command prefix repeat is detected.
	CommandPrefix   string                 `json:"command_prefix,omitempty"`
	Prefix          string                 `json:"prefix,omitempty"`
	CargoCultPrefix *CargoCultPrefixRecord `json:"cargo_cult_prefix,omitempty"`
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

type prefixRun struct {
	prefix string
	count  int
	last   LivelockObservation
}

// LivelockDetector tracks consecutive identical tool-call outcomes per trace. It is intentionally
// small and caller-synchronized; gateway guards it with its server mutex.
type LivelockDetector struct {
	threshold     int
	fuse          int
	abort         int
	byTrace       map[string]livelockRun
	prefixByTrace map[string]prefixRun
}

func NewLivelockDetector(threshold int) *LivelockDetector {
	if threshold <= 0 {
		threshold = DefaultLivelockThreshold
	}
	return &LivelockDetector{
		threshold:     threshold,
		fuse:          threshold * DefaultLivelockFuseFactor,
		abort:         threshold * DefaultLivelockAbortFactor,
		byTrace:       map[string]livelockRun{},
		prefixByTrace: map[string]prefixRun{},
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
	if d.prefixByTrace != nil {
		delete(d.prefixByTrace, trace)
	}
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
	traceID := strings.TrimSpace(obs.TraceID)
	if d != nil && traceID != "" && d.byTrace != nil {
		if run, ok := d.byTrace[traceID]; ok && isRefusalObservation(run.last) {
			delete(d.byTrace, traceID)
		}
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
	if d.prefixByTrace == nil {
		d.prefixByTrace = map[string]prefixRun{}
	}
	obs.TraceID = strings.TrimSpace(obs.TraceID)
	obs.Tool = strings.TrimSpace(obs.Tool)
	obs.Command = strings.TrimSpace(obs.Command)
	obs.RawArgs = strings.TrimSpace(obs.RawArgs)
	obs.ArgsDigest = strings.TrimSpace(obs.ArgsDigest)
	if obs.ArgsDigest == "" {
		if obs.Command != "" {
			obs.ArgsDigest = ArgsDigest(obs.Command)
		} else if obs.RawArgs != "" {
			obs.ArgsDigest = ArgsDigest(obs.RawArgs)
		}
	}
	if obs.Tool == "" && obs.Command != "" {
		obs.Tool = "Bash"
	}
	obs.Verdict = strings.ToUpper(strings.TrimSpace(obs.Verdict))
	obs.Reason = strings.TrimSpace(obs.Reason)
	obs.Disposition = strings.TrimSpace(obs.Disposition)
	obs.SemanticIdentity = strings.TrimSpace(obs.SemanticIdentity)

	// Cargo-cult prefix extraction and tracking across consecutive tool calls.
	cmd := extractCommand(obs)
	prefix, _, isCompound := DecomposeCompoundCommand(cmd)
	isCargoCultCandidate := isCompound && IsAdvisoryOrRecoveryCommand(prefix)

	var pRun prefixRun
	if isCargoCultCandidate {
		pRun = d.prefixByTrace[obs.TraceID]
		if strings.EqualFold(pRun.prefix, prefix) {
			if d.abort <= 0 || pRun.count < d.abort+1 {
				pRun.count++
			}
			pRun.last = obs
		} else {
			pRun = prefixRun{prefix: prefix, count: 1, last: obs}
		}
		d.prefixByTrace[obs.TraceID] = pRun
	} else {
		delete(d.prefixByTrace, obs.TraceID)
	}

	key, semanticIdentity := livelockKey(obs)
	run := d.byTrace[obs.TraceID]
	if run.key == key {
		// Saturate at the terminal abort rung. Once the structural loop-break is
		// armed, a hostile/stuck caller cannot grow this per-trace counter without
		// bound across further ticks; the terminal envelope remains sticky.
		if d.abort <= 0 || run.count < d.abort+1 {
			run.count++
		}
		run.last = obs
	} else {
		run = livelockRun{key: key, count: 1, last: obs}
	}
	d.byTrace[obs.TraceID] = run
	checkpoint := semanticRefusalCheckpoint(obs, semanticIdentity, run.count)

	if isCargoCultCandidate && pRun.count >= d.threshold && run.count < d.threshold {
		reason := obs.Reason
		if reason == "" {
			reason = "CARGO_CULT_PREFIX"
		}
		prefixHash := failureHash("cargo-cult-prefix\x00" + strings.ToLower(prefix))
		record := &CargoCultPrefixRecord{
			Schema:      CargoCultPrefixSchema,
			Event:       CargoCultPrefixEvent,
			Prefix:      prefix,
			RepeatCount: pRun.count,
			NextAction:  "remove leading advisory/recovery sub-command prefix and execute substantive command directly",
		}
		return LivelockEnvelope{
			Schema:           LivelockSchema,
			Event:            CargoCultPrefixEvent,
			TraceID:          obs.TraceID,
			Tool:             obs.Tool,
			ArgsDigest:       obs.ArgsDigest,
			FailureHash:      prefixHash,
			Verdict:          obs.Verdict,
			Reason:           reason,
			Disposition:      obs.Disposition,
			RepeatCount:      pRun.count,
			SuggestedChange:  SuggestedChangeCargoCultPrefix,
			IdentityKind:     "cargo_cult_prefix",
			SemanticIdentity: prefixHash,
			CommandPrefix:    prefix,
			Prefix:           prefix,
			CargoCultPrefix:  record,
			Fuse:             d.fuse > 0 && pRun.count >= d.fuse,
			Escalate:         d.abort > 0 && pRun.count >= d.abort,
		}, true
	}

	if run.count < d.threshold && checkpoint == nil {
		return LivelockEnvelope{}, false
	}
	identityKind := ""
	if semanticIdentity != "" {
		identityKind = "semantic_refusal"
	}
	env := LivelockEnvelope{
		Schema:           LivelockSchema,
		Event:            LivelockEvent,
		TraceID:          obs.TraceID,
		Tool:             obs.Tool,
		ArgsDigest:       obs.ArgsDigest,
		FailureHash:      failureHash(key),
		Verdict:          obs.Verdict,
		Reason:           obs.Reason,
		Disposition:      obs.Disposition,
		RepeatCount:      run.count,
		SuggestedChange:  suggestedLivelockChange(obs, checkpoint != nil),
		IdentityKind:     identityKind,
		SemanticIdentity: semanticIdentity,
		Checkpoint:       checkpoint,
		Fuse:             d.fuse > 0 && run.count >= d.fuse,
		Escalate:         d.abort > 0 && run.count >= d.abort,
	}
	if isCargoCultCandidate && pRun.count >= d.threshold {
		env.CommandPrefix = prefix
		env.Prefix = prefix
		env.CargoCultPrefix = &CargoCultPrefixRecord{
			Schema:      CargoCultPrefixSchema,
			Event:       CargoCultPrefixEvent,
			Prefix:      prefix,
			RepeatCount: pRun.count,
			NextAction:  "remove leading advisory/recovery sub-command prefix and execute substantive command directly",
		}
	}
	return env, true
}

func livelockKey(obs LivelockObservation) (key, semanticIdentity string) {
	if isRefusalObservation(obs) {
		source := strings.TrimSpace(obs.SemanticIdentity)
		if source != "" {
			source = strings.ToUpper(obs.Reason) + "\x00" + strings.ToLower(obs.Tool) + "\x00" + source
		} else if strings.EqualFold(obs.Reason, "SELF_MODIFY") {
			// The gateway currently has only the reason/tool projection at this seam:
			// arguments are content-free digests, so the guarded target cannot be
			// recovered here. Grouping SELF_MODIFY by (reason, tool) is a conservative
			// fallback that catches command mutation without weakening the deny. A
			// caller-provided SemanticIdentity replaces it when target/effect are known.
			source = "fallback\x00" + strings.ToUpper(obs.Reason) + "\x00" + strings.ToLower(obs.Tool)
		}
		if source != "" {
			semanticIdentity = failureHash("semantic-refusal\x00" + source)
			return "semantic-refusal\x00" + semanticIdentity + "\x00" + obs.Verdict + "\x00" + strings.ToUpper(obs.Reason) + "\x00" + strings.ToLower(obs.Tool), semanticIdentity
		}
	}
	key = obs.Tool + "\x00" + obs.ArgsDigest + "\x00" + obs.Verdict + "\x00" + obs.Reason + "\x00" + obs.Disposition
	if obs.SemanticIdentity != "" {
		key += "\x00" + obs.SemanticIdentity
	}
	return key, ""
}

func isRefusalObservation(obs LivelockObservation) bool {
	switch strings.ToUpper(strings.TrimSpace(obs.Verdict)) {
	case "DENY", "QUARANTINE":
		return true
	default:
		return false
	}
}

func semanticRefusalCheckpoint(obs LivelockObservation, identity string, count int) *RefusalCheckpoint {
	if identity == "" || !strings.EqualFold(obs.Reason, "SELF_MODIFY") || count < DefaultSemanticRefusalCheckpoint {
		return nil
	}
	return &RefusalCheckpoint{
		Schema:           RefusalCheckpointSchema,
		Event:            RefusalCheckpointEvent,
		Outcome:          RefusalPausedRecoverable,
		Reason:           "SELF_MODIFY",
		SemanticIdentity: identity,
		RepeatCount:      count,
		Confirmable:      false,
		GoalAction:       "preserve_active",
		NextAction:       "select a sanctioned actuator, or escalate with a witness",
	}
}

func failureHash(key string) string {
	sum := sha256.Sum256([]byte(key))
	return "sha256:" + hex.EncodeToString(sum[:])
}

func suggestedLivelockChange(obs LivelockObservation, checkpoint bool) string {
	if checkpoint {
		return string(RefusalPausedRecoverable)
	}
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

// DecomposeCompoundCommand splits a shell command on the first ';' or '&&' outside quotes,
// returning the leading prefix sub-command, the trailing suffix, and true if compound.
func DecomposeCompoundCommand(cmd string) (prefix, suffix string, isCompound bool) {
	cmd = strings.TrimSpace(cmd)
	if cmd == "" {
		return "", "", false
	}
	inSingle := false
	inDouble := false
	escaped := false
	for i := 0; i < len(cmd); i++ {
		c := cmd[i]
		if escaped {
			escaped = false
			continue
		}
		if c == '\\' {
			escaped = true
			continue
		}
		if c == '\'' && !inDouble {
			inSingle = !inSingle
			continue
		}
		if c == '"' && !inSingle {
			inDouble = !inDouble
			continue
		}
		if !inSingle && !inDouble {
			if c == ';' {
				if i+1 < len(cmd) && cmd[i+1] == ';' {
					i++
					continue
				}
				p := strings.TrimSpace(cmd[:i])
				s := strings.TrimSpace(cmd[i+1:])
				if p != "" && s != "" {
					return p, s, true
				}
			}
			if c == '&' && i+1 < len(cmd) && cmd[i+1] == '&' {
				p := strings.TrimSpace(cmd[:i])
				s := strings.TrimSpace(cmd[i+2:])
				if p != "" && s != "" {
					return p, s, true
				}
			}
		}
	}
	return "", "", false
}

// IsAdvisoryOrRecoveryCommand reports whether a shell command or prefix matches
// advisory or recovery patterns in the fak/dos ecosystem.
func IsAdvisoryOrRecoveryCommand(cmd string) bool {
	s := strings.ToLower(strings.TrimSpace(cmd))
	if s == "" {
		return false
	}
	if strings.HasPrefix(s, "dos ") ||
		strings.HasPrefix(s, "fak recover") ||
		strings.HasPrefix(s, "fak doctor") ||
		strings.HasPrefix(s, "fak man") ||
		strings.HasPrefix(s, "fak help") ||
		strings.Contains(s, "wedge") ||
		strings.Contains(s, "recover") ||
		strings.Contains(s, "--explain") ||
		strings.Contains(s, "advisory") ||
		strings.Contains(s, "check-reason") ||
		strings.Contains(s, "check_reason") ||
		strings.Contains(s, "refuse-reasons") ||
		strings.Contains(s, "refuse_reasons") ||
		strings.Contains(s, "cargo") ||
		strings.Contains(s, "prefix") {
		return true
	}
	return false
}

func extractCommand(obs LivelockObservation) string {
	if cmd := strings.TrimSpace(obs.Command); cmd != "" {
		return cmd
	}
	raw := strings.TrimSpace(obs.RawArgs)
	if raw == "" {
		return ""
	}
	var obj map[string]any
	if err := json.Unmarshal([]byte(raw), &obj); err == nil {
		for _, key := range []string{"command", "cmd", "input", "script"} {
			if v, ok := obj[key]; ok {
				if s, ok := v.(string); ok && strings.TrimSpace(s) != "" {
					return strings.TrimSpace(s)
				}
			}
		}
	}
	return raw
}
