package stopgate

import "fmt"

const (
	DefaultWarnAt          = 3
	DefaultFinalAt         = 7
	DefaultMax             = 9
	DefaultSameStop        = 6
	DefaultToolFeedbackMax = 25
)

// ContinueReason is the NUDGE-rung instruction fed back to the model when fak
// resumes the agent past a deny-all stop.
const ContinueReason = "fak guard: heads-up — your previous turn ended before acting because its tool call(s) are waiting on a shape the capability floor can admit (reported upstream as end_turn). You can continue right now. The in-band `[fak]` note on that turn labels each call as `Tool (REASON/DISPOSITION)` — let that reason point the way. Most reasons just invite a small RESHAPING the floor will welcome: for MISROUTE, reach for the tool or argument shape it expects; for SELF_MODIFY, the floor is protecting a guarded write target (VERSION, .dos/, internal/…), so aim the write at an unguarded path, split a compound command to isolate it, or leave the guarded part out; for LEASE_HELD, another agent holds that tree, so narrow your paths or pick up other work; for a preview-confirm pause, re-send the same call with the confirm key it asked for. A few reasons are protected on purpose — a TERMINAL disposition (e.g. SECRET_EXFIL, TRUST_VIOLATION) is a deliberate boundary, so the clean win there is a different task. Choose an ALLOWED alternative and keep the work moving; and if a protected boundary is all that stands between you and the last step, that is a fine, complete place to stop — note it in one line (`no allowed path: <reason>`) and finish cleanly."

// StageMessage generates the guidance message for the blind deny-all ladder at a given stage.
func StageMessage(stage Stage, consecutive, maxN int) string {
	switch stage {
	case StageWarn:
		return fmt.Sprintf("fak guard: the last %d turns each closed while the capability floor was still waiting for a shape it can admit, so the same approach keeps returning. Good moment to try a fresh angle: if the remaining work is reachable under this floor, take a different allowed action now — a different tool, a narrower command, or a path the floor welcomes. If a protected boundary is all that's left, note it on one line (`no allowed path: <reason>`) and finish cleanly — that is a good, complete outcome. (Auto-continue %d of %d before fak lets the turn end.)", consecutive, consecutive, maxN)
	case StageFinal:
		return fmt.Sprintf("fak guard: last auto-continue (%d of %d). After %d turns still waiting on a shape the floor can admit, fak will let the session wrap up. If there is an allowed way forward, take it on this turn; otherwise note what's protecting the last step on one line (`no allowed path: <reason>`) and finish cleanly now, so the stop is your own call — a complete, expected ending.", consecutive, maxN, maxN)
	default:
		return ContinueReason
	}
}

// GiveUpMessage is the operator-facing stand-down note when the blind deny-all ladder gives up.
func GiveUpMessage(consecutive, maxN int) string {
	return fmt.Sprintf("fak guard Stop: standing down after %d consecutive deny-all turns (every proposed tool call set aside; %d > max %d) — allowing the stop so the loop cannot spin. To keep the agent moving, inspect why the floor sets everything aside (fak guard --dump-policy) or raise --deny-all-continue=max=N; --deny-all-continue off disables this layer.", consecutive, consecutive, maxN)
}

// SameStageMessage generates the guidance message for the same-issue deny-all ladder.
func SameStageMessage(stage Stage, sameConsecutive, stop int) string {
	switch stage {
	case StageWarn:
		return fmt.Sprintf("fak guard: you have now ended %d turns in a row proposing the IDENTICAL refused action — the capability floor is setting aside the very same tool call, for the same reason, each time. Repeating it will not change the verdict. Try a genuinely different angle now: a different tool, a narrower command, or a path the floor welcomes. The in-band `[fak]` note labels the block as `Tool (REASON/DISPOSITION)` — let that reason point the way. If a protected boundary is all that's left, note it on one line (`no allowed path: <reason>`) and finish cleanly — a complete, expected outcome. (Auto-continue %d of %d identical repeats before fak lets the turn end.)", sameConsecutive, sameConsecutive, stop)
	case StageFinal:
		return fmt.Sprintf("fak guard: last auto-continue (%d of %d). You have proposed the IDENTICAL refused action %d turns running; one more and fak will let the session wrap up. This is a genuine repeat, not exploration — if there is an allowed way forward, take a DIFFERENT action on this turn; otherwise note what is protecting the last step on one line (`no allowed path: <reason>`) and finish cleanly now, so the stop is your own call.", sameConsecutive, stop, sameConsecutive)
	default:
		return ContinueReason
	}
}

// SameGiveUpMessage is the operator-facing stand-down note when the same-issue ladder gives up.
func SameGiveUpMessage(sameConsecutive, stop int) string {
	return fmt.Sprintf("fak guard Stop: standing down after %d turns proposing the IDENTICAL refused action (same tool + same reason; %d >= same-issue give-up %d) — a genuine repeated same issue, not exploration, so allowing the stop keeps the loop from spinning. A session hitting a FRESH block each turn is never stopped here. To keep the agent moving, inspect why the floor sets that same call aside (fak guard --dump-policy) or raise --deny-all-continue=same-stop=N; --deny-all-continue off disables this layer.", sameConsecutive, sameConsecutive, stop)
}

// ToolFeedbackMessage generates the continuation guidance for retryable tool-feedback turns.
func ToolFeedbackMessage(consecutive int) string {
	return fmt.Sprintf("fak guard: the previous %d turn(s) ended after retryable tool-call feedback, not a session stop. The proposed tool call(s) were just malformed or otherwise model-fixable, so fak returned per-call feedback and kept the task alive. Fix the JSON/arguments/tool shape and continue — this is a routine retry, so keep going.", consecutive)
}

// ToolFeedbackGiveUpMessage is the operator-facing stand-down line when tool-feedback continues pass the ceiling.
func ToolFeedbackGiveUpMessage(consecutive, bound int) string {
	return fmt.Sprintf("fak guard Stop: standing down — %d consecutive tool-feedback turn(s) exceeded the continue bound (%d). The model kept emitting malformed/misrouted tool calls without landing one, so fak is allowing the stop instead of holding the turn open indefinitely. Raise FAK_GUARD_TOOL_FEEDBACK_MAX to extend the bound.", consecutive, bound)
}

// NormalizeDenyAllThresholds ensures 1 <= warn <= final <= max total ordering.
func NormalizeDenyAllThresholds(warnAt, finalAt, maxN int) (int, int, int) {
	if maxN <= 0 {
		maxN = DefaultMax
	}
	if warnAt < 1 {
		warnAt = 1
	}
	if warnAt > maxN {
		warnAt = maxN
	}
	if finalAt < warnAt {
		finalAt = warnAt
	}
	if finalAt > maxN {
		finalAt = maxN
	}
	return warnAt, finalAt, maxN
}

// NormalizeSameStop derives warn/final rungs from the single same-issue stop depth.
func NormalizeSameStop(stop int) (warnAt, finalAt, stopN int) {
	if stop < 2 {
		stop = DefaultSameStop
	}
	finalAt = stop - 1
	warnAt = stop - 3
	if warnAt < 1 {
		warnAt = 1
	}
	if finalAt < warnAt {
		finalAt = warnAt
	}
	return warnAt, finalAt, stop
}

// StageForDenyAll maps consecutive count to its blind ladder rung.
func StageForDenyAll(consecutive, warnAt, finalAt, maxN int) Stage {
	warnAt, finalAt, maxN = NormalizeDenyAllThresholds(warnAt, finalAt, maxN)
	switch {
	case consecutive <= 0:
		return StageAllow
	case consecutive > maxN:
		return StageGiveUp
	case consecutive >= finalAt:
		return StageFinal
	case consecutive >= warnAt:
		return StageWarn
	default:
		return StageNudge
	}
}

// StageForSameIssue maps same-issue consecutive count to its ladder rung.
func StageForSameIssue(sameConsecutive, stop int) Stage {
	warnAt, finalAt, stopN := NormalizeSameStop(stop)
	switch {
	case sameConsecutive <= 0:
		return StageAllow
	case sameConsecutive >= stopN:
		return StageGiveUp
	case sameConsecutive >= finalAt:
		return StageFinal
	case sameConsecutive >= warnAt:
		return StageWarn
	default:
		return StageNudge
	}
}

// LadderConfig configures the graduated deny-all and tool-feedback back-off ladders.
type LadderConfig struct {
	WarnAt          int
	FinalAt         int
	Max             int
	SameStop        int
	ToolFeedbackMax int
	Mode            Mode
}

// DefaultLadderConfig returns standard production defaults.
func DefaultLadderConfig() LadderConfig {
	return LadderConfig{
		WarnAt:          DefaultWarnAt,
		FinalAt:         DefaultFinalAt,
		Max:             DefaultMax,
		SameStop:        DefaultSameStop,
		ToolFeedbackMax: DefaultToolFeedbackMax,
		Mode:            ModeEnforce,
	}
}

// Ladder evaluates deny-all and tool-feedback retry/stand-down thresholds.
type Ladder struct {
	cfg LadderConfig
}

// NewLadder constructs a Ladder with normalized thresholds.
func NewLadder(cfg LadderConfig) *Ladder {
	w, f, m := NormalizeDenyAllThresholds(cfg.WarnAt, cfg.FinalAt, cfg.Max)
	_, _, s := NormalizeSameStop(cfg.SameStop)
	tfMax := cfg.ToolFeedbackMax
	if tfMax <= 0 {
		tfMax = DefaultToolFeedbackMax
	}
	mode := cfg.Mode
	if mode == "" {
		mode = ModeEnforce
	}
	return &Ladder{
		cfg: LadderConfig{
			WarnAt:          w,
			FinalAt:         f,
			Max:             m,
			SameStop:        s,
			ToolFeedbackMax: tfMax,
			Mode:            mode,
		},
	}
}

// EvaluateDenyAll evaluates consecutive deny-all counts and returns a Decision.
func (l *Ladder) EvaluateDenyAll(consecutive, sameConsecutive int, useSame bool) Decision {
	mode := l.cfg.Mode
	if mode == "" {
		mode = ModeEnforce
	}

	if useSame {
		stage := StageForSameIssue(sameConsecutive, l.cfg.SameStop)
		depth := sameConsecutive
		bound := l.cfg.SameStop
		signal := "same-issue"
		if depth <= 0 {
			signal = "clean"
		}

		if stage == StageAllow {
			return Decision{
				Action:      ActionAllow,
				Stage:       StageAllow,
				Disposition: DispCleanCompletion,
				Kind:        KindClean,
				ExitCode:    0,
				Blocked:     false,
				Signal:      "clean",
				Depth:       0,
				Bound:       bound,
			}
		}

		if stage == StageGiveUp {
			return Decision{
				Action:      ActionAllow,
				Stage:       stage,
				Disposition: DispSameIssueGiveUp,
				Kind:        KindStandDown,
				Blocked:     false,
				ExitCode:    0,
				Signal:      signal,
				Depth:       depth,
				Bound:       bound,
				OperatorMsg: SameGiveUpMessage(depth, bound),
			}
		}

		if mode == ModeOff {
			return Decision{
				Action:      ActionAllow,
				Stage:       stage,
				Disposition: DispModeOff,
				Kind:        KindOff,
				ExitCode:    0,
				Blocked:     false,
				Signal:      signal,
				Depth:       depth,
				Bound:       bound,
			}
		}

		block := stage == StageNudge || stage == StageWarn || stage == StageFinal
		if mode == ModeShadow {
			return Decision{
				Action:      ActionAllow,
				Stage:       stage,
				Disposition: DispShadow,
				Kind:        KindShadow,
				Blocked:     block,
				ExitCode:    0,
				Signal:      signal,
				Depth:       depth,
				Bound:       bound,
				Guidance:    SameStageMessage(stage, depth, bound),
			}
		}

		if block {
			return Decision{
				Action:      ActionContinue,
				Stage:       stage,
				Disposition: DispSameIssueContinue,
				Kind:        KindContinue,
				Blocked:     true,
				ExitCode:    2,
				Signal:      signal,
				Depth:       depth,
				Bound:       bound,
				Guidance:    SameStageMessage(stage, depth, bound),
			}
		}

		return Decision{
			Action:      ActionAllow,
			Stage:       StageAllow,
			Disposition: DispCleanCompletion,
			Kind:        KindClean,
			ExitCode:    0,
			Blocked:     false,
			Signal:      "clean",
			Depth:       0,
			Bound:       bound,
		}
	}

	// Blind ladder path
	stage := StageForDenyAll(consecutive, l.cfg.WarnAt, l.cfg.FinalAt, l.cfg.Max)
	depth := consecutive
	bound := l.cfg.Max
	signal := "blind"
	if depth <= 0 {
		signal = "clean"
	}

	if stage == StageAllow {
		return Decision{
			Action:      ActionAllow,
			Stage:       StageAllow,
			Disposition: DispCleanCompletion,
			Kind:        KindClean,
			ExitCode:    0,
			Blocked:     false,
			Signal:      "clean",
			Depth:       0,
			Bound:       bound,
		}
	}

	if stage == StageGiveUp {
		return Decision{
			Action:      ActionAllow,
			Stage:       stage,
			Disposition: DispBlindGiveUp,
			Kind:        KindStandDown,
			Blocked:     false,
			ExitCode:    0,
			Signal:      signal,
			Depth:       depth,
			Bound:       bound,
			OperatorMsg: GiveUpMessage(depth, bound),
		}
	}

	if mode == ModeOff {
		return Decision{
			Action:      ActionAllow,
			Stage:       stage,
			Disposition: DispModeOff,
			Kind:        KindOff,
			ExitCode:    0,
			Blocked:     false,
			Signal:      signal,
			Depth:       depth,
			Bound:       bound,
		}
	}

	block := stage == StageNudge || stage == StageWarn || stage == StageFinal
	if mode == ModeShadow {
		return Decision{
			Action:      ActionAllow,
			Stage:       stage,
			Disposition: DispShadow,
			Kind:        KindShadow,
			Blocked:     block,
			ExitCode:    0,
			Signal:      signal,
			Depth:       depth,
			Bound:       bound,
			Guidance:    StageMessage(stage, depth, bound),
		}
	}

	if block {
		return Decision{
			Action:      ActionContinue,
			Stage:       stage,
			Disposition: DispDenyAllContinue,
			Kind:        KindContinue,
			Blocked:     true,
			ExitCode:    2,
			Signal:      signal,
			Depth:       depth,
			Bound:       bound,
			Guidance:    StageMessage(stage, depth, bound),
		}
	}

	return Decision{
		Action:      ActionAllow,
		Stage:       StageAllow,
		Disposition: DispCleanCompletion,
		Kind:        KindClean,
		ExitCode:    0,
		Blocked:     false,
		Signal:      "clean",
		Depth:       0,
		Bound:       bound,
	}
}

// EvaluateToolFeedback evaluates consecutive retryable tool feedback counts.
func (l *Ladder) EvaluateToolFeedback(consecutive int) Decision {
	bound := l.cfg.ToolFeedbackMax
	if bound <= 0 {
		bound = DefaultToolFeedbackMax
	}
	mode := l.cfg.Mode
	if mode == "" {
		mode = ModeEnforce
	}

	if mode == ModeOff {
		return Decision{
			Action:      ActionAllow,
			Stage:       StageAllow,
			Disposition: DispModeOff,
			Kind:        KindOff,
			ExitCode:    0,
			Blocked:     false,
			Signal:      "tool-feedback",
			Depth:       consecutive,
			Bound:       bound,
		}
	}

	if consecutive > bound {
		if mode == ModeShadow {
			return Decision{
				Action:      ActionAllow,
				Stage:       StageGiveUp,
				Disposition: DispShadow,
				Kind:        KindShadow,
				Blocked:     false,
				ExitCode:    0,
				Signal:      "tool-feedback",
				Depth:       consecutive,
				Bound:       bound,
				OperatorMsg: fmt.Sprintf("fak guard Stop: shadow would stand down on tool-feedback past bound (tool_feedback_consecutive=%d bound=%d)", consecutive, bound),
			}
		}
		return Decision{
			Action:      ActionAllow,
			Stage:       StageGiveUp,
			Disposition: DispToolFeedbackGiveUp,
			Kind:        KindStandDown,
			Blocked:     false,
			ExitCode:    0,
			Signal:      "tool-feedback",
			Depth:       consecutive,
			Bound:       bound,
			OperatorMsg: ToolFeedbackGiveUpMessage(consecutive, bound),
		}
	}

	if mode == ModeShadow {
		return Decision{
			Action:      ActionAllow,
			Stage:       StageNudge,
			Disposition: DispShadow,
			Kind:        KindShadow,
			Blocked:     true,
			ExitCode:    0,
			Signal:      "tool-feedback",
			Depth:       consecutive,
			Bound:       bound,
			Guidance:    ToolFeedbackMessage(consecutive),
		}
	}

	return Decision{
		Action:      ActionContinue,
		Stage:       StageNudge,
		Disposition: DispToolFeedbackContinue,
		Kind:        KindContinue,
		Blocked:     true,
		ExitCode:    2,
		Signal:      "tool-feedback",
		Depth:       consecutive,
		Bound:       bound,
		Guidance:    ToolFeedbackMessage(consecutive),
	}
}

// EvaluateDenyAll evaluates the deny-all ladder with given configuration.
func EvaluateDenyAll(cfg LadderConfig, consecutive, sameConsecutive int, useSame bool) Decision {
	return NewLadder(cfg).EvaluateDenyAll(consecutive, sameConsecutive, useSame)
}

// EvaluateToolFeedback evaluates tool feedback with given configuration.
func EvaluateToolFeedback(cfg LadderConfig, consecutive int) Decision {
	return NewLadder(cfg).EvaluateToolFeedback(consecutive)
}
