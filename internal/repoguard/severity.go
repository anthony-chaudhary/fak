// severity.go — the per-reason severity model for the repo-guard PreToolUse hook.
//
// Every classifier rung in this package (OUT_OF_TREE_WRITE, INTERACTIVE_HANG,
// LIVE_MONITOR_OUTPUT_READ, FOREGROUND_SLEEP, WORKSPACE_PATH_UNMAPPED,
// FOREGROUND_NETWORK_LOOP) is a best-effort heuristic — the
// docs say so plainly ("it raises the floor, it is not a sandbox"). Hard-denying
// a tool call on a heuristic is too rigid: cross-repo work in a fleet host's
// `work/` tree of sibling repos is ROUTINE, not anomalous, and even a *warning*
// is injected into the agent's context where it can steer the model or waste a
// turn. So a classified violation does not imply a decision — the DECISION is a
// severity resolved per reason, defaulting permissive, with the hard blocks a
// one-env-flag opt-in for the operator who wants them.
//
// This file is the single source of truth for "given a reason, what do we do?".
// It is PURE: no env reads (the command layer owns those), no clock, hermetically
// testable like the rest of the core.
package repoguard

import "strings"

// Severity is what the hook does about a classified violation, ordered from most
// permissive to most strict. The ordering is meaningful: the global FAK_REPO_GUARD
// master switch CAPS severity (warn can only soften, never escalate), which is a
// numeric min() over this order.
type Severity int

const (
	// SeverityOff drops the finding entirely: no journal row, no stderr, allow.
	SeverityOff Severity = iota
	// SeverityRecord is SILENT: append a durable journal row, emit NOTHING on
	// stderr, allow. Nothing enters the model's context — the guard proves it
	// caught something (countable via --summary) without perturbing the agent.
	SeverityRecord
	// SeverityWarn is advisory: a structured stderr pointer + fix, a journal row,
	// allow. The agent SEES it (which is the point when the fix genuinely helps).
	SeverityWarn
	// SeverityDeny is the hard block: deny via the PreToolUse decision protocol
	// plus a journal row. This is opt-in per reason under the default posture.
	SeverityDeny
)

// String renders the level for journal decision labels and --check copy.
func (s Severity) String() string {
	switch s {
	case SeverityOff:
		return "off"
	case SeverityRecord:
		return "record"
	case SeverityWarn:
		return "warn"
	case SeverityDeny:
		return "deny"
	default:
		return "deny" // unknown value fails safe toward the strict end
	}
}

// DecisionLabel is the journal `decision` field for a resolved severity. Off has
// no row (the caller drops it before recording), so it maps to "" defensively.
func (s Severity) DecisionLabel() string {
	switch s {
	case SeverityRecord:
		return "record"
	case SeverityWarn:
		return "advisory"
	case SeverityDeny:
		return "deny"
	default:
		return ""
	}
}

// defaultSeverity is the permissive default posture. No rung hard-denies by
// default; `deny` is a per-reason opt-in. The routine/false-positive-prone rungs
// are SILENT (record) so they never enter the model's context; the rungs whose
// fix-hint genuinely helps the agent stay at warn.
var defaultSeverity = map[string]Severity{
	ReasonBuildCacheCleanRace:           SeverityDeny,   // deletes shared compiler state while peers may still consume it
	guardReason:                         SeverityRecord, // OUT_OF_TREE_WRITE: routine cross-repo work; a placement convention
	ReasonLiveMonitorOutputRead:         SeverityRecord, // niche, harmless-if-wrong anti-pattern
	ReasonInteractiveHang:               SeverityWarn,   // the non-interactive-form hint avoids a wasted turn
	ReasonForegroundSleep:               SeverityWarn,   // the background-wait hint avoids a wasted turn
	ReasonWorkspacePathUnmapped:         SeverityWarn,   // the correct-path hint avoids a wasted turn
	ReasonForegroundNetworkLoop:         SeverityWarn,   // the batch/background hint avoids a killed-mid-loop turn
	ReasonForegroundPowerShellInventory: SeverityWarn,   // bounded host inventory avoids the two-minute foreground kill
	ReasonUndeclaredLeaf:                SeverityWarn,   // #2082: the declare-it-now hint at edit time beats a refused commit many turns later
}

// DefaultSeverity returns the default severity for a reason. An UNKNOWN reason
// resolves to SeverityDeny (fail-safe): any refusal-class reason added later
// denies until it is explicitly softened in this table — we never silently
// downgrade something unclassified.
func DefaultSeverity(reason string) Severity {
	if s, ok := defaultSeverity[reason]; ok {
		return s
	}
	return SeverityDeny
}

// ResolveSeverity is the precedence resolver — the one function the hook consults
// to turn a reason into a decision:
//
//  1. global mode "off"  -> SeverityOff  (the master switch skips everything)
//  2. global mode "warn" -> the resolved severity CAPPED at SeverityWarn
//     (the master switch may soften, never escalate)
//  3. an explicit per-reason override wins over the default
//  4. otherwise the default table
//
// globalMode is the raw FAK_REPO_GUARD value already lower-cased/trimmed by the
// caller ("" or "enforce" both mean the normal per-reason path).
func ResolveSeverity(reason string, overrides map[string]Severity, globalMode string) Severity {
	switch globalMode {
	case "off":
		return SeverityOff
	case "warn":
		return capSeverity(resolvePerReason(reason, overrides), SeverityWarn)
	default:
		return resolvePerReason(reason, overrides)
	}
}

func resolvePerReason(reason string, overrides map[string]Severity) Severity {
	if s, ok := overrides[reason]; ok {
		return s
	}
	return DefaultSeverity(reason)
}

func capSeverity(s, cap Severity) Severity {
	if s > cap {
		return cap
	}
	return s
}

// ParseSeverity parses one level token. Case-insensitive; ok is false for an
// unrecognized token so the caller can skip a malformed pair rather than guess.
func ParseSeverity(s string) (Severity, bool) {
	switch strings.ToLower(strings.TrimSpace(s)) {
	case "off":
		return SeverityOff, true
	case "record":
		return SeverityRecord, true
	case "warn", "advisory":
		return SeverityWarn, true
	case "deny", "enforce", "block":
		return SeverityDeny, true
	default:
		return SeverityOff, false
	}
}

// ParseSeverityOverrides parses the FAK_REPO_GUARD_SEVERITY spec:
// `REASON=level,REASON=level`. Tolerant — a malformed pair (no `=`, unknown
// level) is skipped, not fatal, so one typo never wedges the whole guard. Reason
// tokens are upper-cased to match the exported Reason* constants. Returns nil for
// an empty/blank spec.
func ParseSeverityOverrides(spec string) map[string]Severity {
	spec = strings.TrimSpace(spec)
	if spec == "" {
		return nil
	}
	out := map[string]Severity{}
	for _, pair := range strings.Split(spec, ",") {
		pair = strings.TrimSpace(pair)
		if pair == "" {
			continue
		}
		eq := strings.IndexByte(pair, '=')
		if eq < 0 {
			continue // no level given — skip
		}
		reason := strings.ToUpper(strings.TrimSpace(pair[:eq]))
		level, ok := ParseSeverity(pair[eq+1:])
		if reason == "" || !ok {
			continue
		}
		out[reason] = level
	}
	if len(out) == 0 {
		return nil
	}
	return out
}
