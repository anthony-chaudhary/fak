package polymodel

// specthrottle.go — an adaptive draft-length throttle for the SpecDecode speculation
// (companion to specdecode.go, #4877; study-borrow from TensorRT-LLM's acceptance-rate
// speculation control, #5257 / epic #5256). SpecDecode is LOSSLESS for any drafter — a
// bad drafter never corrupts the stream, it only costs extra verify passes with few
// accepts. This throttle is the pure CONTROL that keeps speculation from being net
// NEGATIVE: it watches the real per-round acceptance rate SpecDecode already tracks
// (AcceptedDrafts over DraftedTokens) and adapts the next draft length K — shrinking it
// toward the floor and finally to zero (stop drafting) when acceptance is persistently
// poor, growing it back toward the max when acceptance is high, and holding steady in a
// mid band so it does not flap.
//
// PURE, DETERMINISTIC, WALL-CLOCK-FREE. Like the rest of this tier-1 leaf it imports
// nothing internal and touches no GPU: it consumes plain (accepted, proposed) counts per
// draft round and returns the next K. The host binds the thresholds and feeds the counts
// from a finished round; the throttle owns only the rolling-window bookkeeping. It never
// changes WHAT SpecDecode commits — the stream stays token-identical to plain greedy decode
// — only WHETHER, and how far, the next round drafts.
//
// THE RULE (one-line intuition). Keep an O(1) rolling-sum window of the last N rounds'
// (accepted, proposed) counts. rate = accepted-sum / proposed-sum over the window. Below
// the low threshold: shrink K toward the floor, and after a streak of poor windows stop
// drafting (K=0). Above the high threshold: grow K toward the max (or resume from the
// floor if drafting was stopped). Between the two thresholds: hold — the low/high band is
// the hysteresis that stops K from oscillating on noise. A degenerate round (no tokens
// proposed, or impossible counts) fails closed to a full stop rather than trusting it.

// DraftLengthThrottleConfig sets the adaptive throttle's fixed bounds. Zero fields take
// safe defaults (see NewDraftLengthThrottle), so the zero value is a usable throttle.
type DraftLengthThrottleConfig struct {
	// Window is how many recent draft rounds the acceptance rate averages over. A larger
	// window reacts more slowly and is steadier; a smaller one adapts faster. Default 4.
	Window int
	// Low is the acceptance-rate floor: at or below it the throttle shrinks K (and, after a
	// poor streak, stops). Default 0.3.
	Low float64
	// High is the acceptance-rate ceiling: strictly above it the throttle grows K. The
	// [Low, High] mid band holds K steady (the hysteresis). Default 0.7. If a caller passes
	// High < Low it is raised to Low (the band collapses to hold-only).
	High float64
	// Floor is the smallest NON-zero draft length the throttle shrinks to before a stop —
	// the shortest speculation still worth a verify pass. Default 1.
	Floor int
	// Max caps how far a high acceptance rate may grow K. Default 8.
	Max int
	// Step is how much K moves per adaptation (grow or shrink). Default 1.
	Step int
	// StopStreak is how many consecutive below-Low windows force a full stop (K=0, drafting
	// off). Default 3. A high rate anywhere in between clears the streak.
	StopStreak int
	// Start is the initial draft length before the window fills (the warmup K). Clamped into
	// [0, Max]. Default 4.
	Start int
}

// ThrottleVerdict is one adaptation step's outcome: the next draft length to use, whether
// to draft at all, the windowed acceptance rate it decided on, and a one-word Reason
// ("warmup", "grow", "shrink", "hold", "stop", or "degenerate").
type ThrottleVerdict struct {
	// Length is the next draft length K the host should propose. 0 means draft nothing this
	// round (plain single-token decode).
	Length int
	// Drafting is Length > 0 — a convenience the host can branch on.
	Drafting bool
	// Rate is the windowed acceptance rate the verdict used (0 until the window fills).
	Rate float64
	// Reason names which arm fired, for honest observability.
	Reason string
}

// DraftLengthThrottle is the stateful adaptive controller. Feed it one (accepted,
// proposed) pair per finished draft round with Record; it returns the next draft length.
// It is NOT safe for concurrent use — one throttle per running SpecDecode.
type DraftLengthThrottle struct {
	window     int
	low, high  float64
	floor, max int
	step       int
	stopStreak int

	samples []accSample
	idx     int
	filled  int
	accSum  int
	propSum int

	length   int
	drafting bool
	poor     int
	rate     float64
}

// accSample is one draft round's acceptance accounting held in the rolling window.
type accSample struct {
	accepted, proposed int
}

// NewDraftLengthThrottle builds a throttle from cfg, substituting safe defaults for zero
// or out-of-range fields so the zero-value config yields a sensible controller.
func NewDraftLengthThrottle(cfg DraftLengthThrottleConfig) *DraftLengthThrottle {
	window := cfg.Window
	if window <= 0 {
		window = 4
	}
	low := cfg.Low
	if low <= 0 {
		low = 0.3
	}
	high := cfg.High
	if high <= 0 {
		high = 0.7
	}
	if high < low {
		high = low // band collapses to hold-only rather than inverting
	}
	floor := cfg.Floor
	if floor <= 0 {
		floor = 1
	}
	max := cfg.Max
	if max <= 0 {
		max = 8
	}
	if floor > max {
		floor = max
	}
	step := cfg.Step
	if step <= 0 {
		step = 1
	}
	stop := cfg.StopStreak
	if stop <= 0 {
		stop = 3
	}
	start := cfg.Start
	if start <= 0 {
		start = 4 // zero/unset warms up at a middling K
	}
	if start > max {
		start = max
	}
	return &DraftLengthThrottle{
		window:     window,
		low:        low,
		high:       high,
		floor:      floor,
		max:        max,
		step:       step,
		stopStreak: stop,
		samples:    make([]accSample, window),
		length:     start,
		drafting:   start > 0,
	}
}

// Record folds one finished draft round's counts into the rolling window and returns the
// adapted next draft length. accepted is how many drafted tokens the verify pass committed;
// proposed is how many were drafted (SpecDecodeRun's AcceptedDrafts / DraftedTokens deltas
// for the round). A degenerate round — nothing proposed, negative counts, or more accepted
// than proposed — fails closed to a full stop and is not folded into the window.
func (t *DraftLengthThrottle) Record(accepted, proposed int) ThrottleVerdict {
	if proposed <= 0 || accepted < 0 || accepted > proposed {
		t.length = 0
		t.drafting = false
		return t.verdictOf("degenerate")
	}

	// O(1) rolling sums: drop the oldest sample once the window is full, then admit the new.
	if t.filled == t.window {
		old := t.samples[t.idx]
		t.accSum -= old.accepted
		t.propSum -= old.proposed
	} else {
		t.filled++
	}
	t.samples[t.idx] = accSample{accepted: accepted, proposed: proposed}
	t.idx = (t.idx + 1) % t.window
	t.accSum += accepted
	t.propSum += proposed

	// Warmup: hold the current length until the window has enough samples to be meaningful.
	if t.filled < t.window {
		return t.verdictOf("warmup")
	}

	t.rate = float64(t.accSum) / float64(t.propSum)
	switch {
	case t.rate < t.low:
		// Poor acceptance: shrink toward the floor, and after a streak of poor windows stop.
		t.poor++
		t.length -= t.step
		if t.length < t.floor {
			t.length = t.floor
		}
		if t.poor >= t.stopStreak {
			t.length = 0
			t.drafting = false
			return t.verdictOf("stop")
		}
		t.drafting = t.length > 0
		return t.verdictOf("shrink")
	case t.rate > t.high:
		// High acceptance: clear the poor streak and grow, resuming from the floor if a prior
		// stop had turned drafting off (a probe round with proposed>0 fed the throttle).
		t.poor = 0
		if !t.drafting {
			t.length = t.floor
			t.drafting = true
		} else {
			t.length += t.step
			if t.length > t.max {
				t.length = t.max
			}
		}
		return t.verdictOf("grow")
	default:
		// Mid band: hold steady — this is the hysteresis that prevents flapping on noise.
		t.poor = 0
		t.drafting = t.length > 0
		return t.verdictOf("hold")
	}
}

// verdictOf snapshots the current state as a ThrottleVerdict with the given reason word.
func (t *DraftLengthThrottle) verdictOf(reason string) ThrottleVerdict {
	return ThrottleVerdict{Length: t.length, Drafting: t.drafting, Rate: t.rate, Reason: reason}
}

// Length is the throttle's current draft length K.
func (t *DraftLengthThrottle) Length() int { return t.length }

// Drafting reports whether the throttle currently wants any draft (Length > 0).
func (t *DraftLengthThrottle) Drafting() bool { return t.drafting }

// Rate is the most recent windowed acceptance rate (0 until the window first fills).
func (t *DraftLengthThrottle) Rate() float64 { return t.rate }
