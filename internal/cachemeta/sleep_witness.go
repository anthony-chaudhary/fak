package cachemeta

// sleep_witness.go — issue #1730: bridge fak session dormancy to upstream vLLM
// lifecycle controls (pause / sleep / wake) with EXPLICIT cache-loss evidence.
//
// vLLM's sleep mode (docs/features/sleep_mode.md) can release GPU memory and
// DISCARD the KV cache: level 1 offloads weights to CPU memory but forgets KV,
// level 2 forgets both weights and KV. A pause (scheduler stop) keeps KV resident.
// A prefix-cache reset forgets KV while the engine stays up. The danger for a
// reuse layer is a STALE warm belief: fak's PrefixStabilityTracker (prefix_score.go)
// may still report a prefix "warm" after the engine underneath it silently threw
// the KV away. This file is the pure, wall-clock-free witness that says — per
// dormancy action — whether KV was preserved, forgotten, or unknown, and whether a
// warm-prefix belief must therefore be demoted cold.
//
// This leaf owns ONLY the belief/witness logic. It never calls a vLLM control
// endpoint (that is an engine adapter's job, the same boundary
// external_invalidation.go keeps) and it does not itself scrape or emit metrics — it
// exposes the honest Serving bit a metrics/`session ls` layer reads so it never
// reports the engine healthy-serving while it is asleep.

// EngineSleepLevel mirrors vLLM's sleep-mode levels. Both non-zero levels discard
// the KV cache; they differ only in whether weights are also forgotten.
type EngineSleepLevel int

const (
	// SleepNone is the awake engine — no dormancy requested.
	SleepNone EngineSleepLevel = 0
	// SleepLevel1 offloads weights to CPU memory and FORGETS the KV cache.
	SleepLevel1 EngineSleepLevel = 1
	// SleepLevel2 forgets BOTH weights and the KV cache.
	SleepLevel2 EngineSleepLevel = 2
)

// EngineDormancyAction is the closed set of session-driven lifecycle actions fak
// maps onto a vLLM worker. Only a fak session lifecycle action drives these — an
// out-of-band vLLM `/sleep` is not modeled here, by design (the acceptance rule
// "pause/sleep only through a fak session lifecycle action").
type EngineDormancyAction string

const (
	// DormancyPause stops the scheduler but keeps KV resident in place.
	DormancyPause EngineDormancyAction = "pause"
	// DormancySleep issues vLLM `/sleep` at a level; the KV cache is forgotten.
	DormancySleep EngineDormancyAction = "sleep"
	// DormancyReset resets the prefix cache; KV is forgotten, engine stays up.
	DormancyReset EngineDormancyAction = "reset"
	// DormancyWake issues vLLM `/wake_up`; the engine serves again, but the KV
	// forgotten during sleep is NOT restored — it must be refilled.
	DormancyWake EngineDormancyAction = "wake"
)

// KVDisposition is the witnessed verdict of what happened to the KV cache. It is
// deliberately a three-value vocabulary: "unknown" is a first-class answer, never
// silently collapsed to "preserved" (that would let a stale warm belief survive).
type KVDisposition string

const (
	// KVPreserved — KV stayed resident (a pause).
	KVPreserved KVDisposition = "preserved"
	// KVForgotten — KV was discarded (a sleep at any level, or a reset).
	KVForgotten KVDisposition = "forgotten"
	// KVDispositionUnknown — fak cannot prove KV survived; treated fail-closed as
	// potentially lost, so warm beliefs are demoted.
	KVDispositionUnknown KVDisposition = "unknown"
)

// EngineLifecyclePhase is the session-facing dormancy phase a metrics/`session ls`
// surface reports. It is separate from KVDisposition: a phase says whether the
// engine can serve, the disposition says what happened to the cache.
type EngineLifecyclePhase string

const (
	// PhaseServing — the engine is up and can serve requests.
	PhaseServing EngineLifecyclePhase = "serving"
	// PhasePaused — the scheduler is stopped; KV is intact but no request runs.
	PhasePaused EngineLifecyclePhase = "paused"
	// PhaseSleeping — the engine is asleep; it CANNOT serve and KV is gone.
	PhaseSleeping EngineLifecyclePhase = "sleeping"
	// PhaseWaking — a wake was requested; the engine is returning to serving.
	PhaseWaking EngineLifecyclePhase = "waking"
	// PhaseError — the dormancy action was ill-formed (e.g. sleep with no level).
	PhaseError EngineLifecyclePhase = "error"
)

// SleepWitness is one immutable row recording the cache-loss consequence of a
// dormancy action. It is the evidence the acceptance criteria ask for: it says
// whether KV was preserved/forgotten/unknown, whether warm prefix beliefs must be
// demoted cold, and — honestly — whether the engine can serve (never true while
// asleep).
type SleepWitness struct {
	Action EngineDormancyAction
	Level  EngineSleepLevel
	Phase  EngineLifecyclePhase
	KV     KVDisposition
	// WarmPrefixesCold is true when this action invalidates warm-prefix beliefs:
	// any KV loss (or an unknown disposition, fail-closed) demotes them.
	WarmPrefixesCold bool
	// Serving is the honest health bit for a metrics/`session ls` layer: false
	// whenever the engine cannot actually serve (paused, sleeping, error), so a
	// dashboard never reads "healthy" for an asleep worker.
	Serving bool
	Reason  string
}

// WitnessDormancy lowers a session dormancy action into its cache-loss witness. It
// is pure: no clock, no engine call, no I/O. A sleep with a non-positive level is
// ill-formed and fails closed (unknown KV, warm beliefs demoted).
func WitnessDormancy(action EngineDormancyAction, level EngineSleepLevel) SleepWitness {
	switch action {
	case DormancyPause:
		return SleepWitness{
			Action: action, Phase: PhasePaused, KV: KVPreserved,
			WarmPrefixesCold: false, Serving: false,
			Reason: "scheduler paused; KV cache preserved in place",
		}
	case DormancySleep:
		if level != SleepLevel1 && level != SleepLevel2 {
			return SleepWitness{
				Action: action, Level: level, Phase: PhaseError, KV: KVDispositionUnknown,
				WarmPrefixesCold: true, Serving: false,
				Reason: "sleep requires level 1 or 2; disposition unknown, warm beliefs demoted fail-closed",
			}
		}
		return SleepWitness{
			Action: action, Level: level, Phase: PhaseSleeping, KV: KVForgotten,
			WarmPrefixesCold: true, Serving: false,
			Reason: sleepReason(level),
		}
	case DormancyReset:
		return SleepWitness{
			Action: action, Phase: PhaseServing, KV: KVForgotten,
			WarmPrefixesCold: true, Serving: true,
			Reason: "prefix cache reset; KV forgotten while engine stays up",
		}
	case DormancyWake:
		return SleepWitness{
			Action: action, Phase: PhaseServing, KV: KVDispositionUnknown,
			WarmPrefixesCold: true, Serving: true,
			Reason: "engine woke; KV forgotten during sleep is not restored — warm beliefs stay cold until a fresh cache signal",
		}
	default:
		return SleepWitness{
			Action: action, Phase: PhaseError, KV: KVDispositionUnknown,
			WarmPrefixesCold: true, Serving: false,
			Reason: "unknown dormancy action; disposition unknown, warm beliefs demoted fail-closed",
		}
	}
}

func sleepReason(level EngineSleepLevel) string {
	if level == SleepLevel2 {
		return "vLLM sleep level 2: weights and KV cache forgotten"
	}
	return "vLLM sleep level 1: weights offloaded to CPU, KV cache forgotten"
}

// WarmHitGate decides whether a resume may report a WARM prefix hit against a vLLM
// worker that may have slept/reset underneath it. It starts unproven: no warm hit
// is allowed until a fresh cache signal (a vLLM BlockStored event proving the KV is
// resident again) revalidates the belief. Any forgotten/unknown-KV witness drives
// it back to unproven, so a stale warm belief cannot survive dormancy — the exact
// acceptance rule "resume refuses a warm hit after sleep/reset unless a fresh
// BlockStored/cache signal revalidates it".
//
// It is a small, single-session state machine (not safe for concurrent use without
// external synchronization), the same contract as PrefixStabilityTracker.
type WarmHitGate struct {
	revalidated bool
}

// ObserveDormancy folds a dormancy witness into the gate. A witness that demotes
// warm prefixes (any KV loss or unknown disposition) invalidates the belief.
func (g *WarmHitGate) ObserveDormancy(w SleepWitness) {
	if w.WarmPrefixesCold {
		g.revalidated = false
	}
}

// ObserveCacheSignal records a fresh resident-KV signal (a vLLM BlockStored event)
// that revalidates the prefix belief. This is the ONLY thing that re-warms the gate
// after a sleep/reset.
func (g *WarmHitGate) ObserveCacheSignal() { g.revalidated = true }

// WarmHitAllowed reports whether a resume may currently claim a warm prefix hit.
func (g *WarmHitGate) WarmHitAllowed() bool { return g.revalidated }
