package promptmmu

// config.go is the flag + observability rung (epic #751 rung 4, #754): the curate
// mode the host flag selects, and the legible drop record the gateway logs out of
// band. It mirrors the co-travel / reset-on-budget gate convention: a closed mode
// vocabulary with a SAFE default, a shadow rung that computes the plan without
// mutating the body, and an off rung. Curate is the single mode-gated host branch
// (off/shadow/on over a REAL body); the cmd/fak flag string + the gateway log
// emission are the deferred cross-lane halves; the mode semantics + the
// shadow-never-mutates guard live here so the spine and its host agree on one closed
// set and the host cannot re-implement the guard wrongly.

// Mode is the closed curate-mode vocabulary the host flag resolves to. The DEFAULT
// is on (curate out of the box, epic posture); shadow is the safe dogfood rung that
// computes the plan and reports what WOULD prune WITHOUT mutating the body; off
// disables the prune entirely.
type Mode string

const (
	// ModeOn curates: compute the plan AND apply the cache-safe splice. The
	// default posture (an empty/unset flag resolves here via ParseMode).
	ModeOn Mode = "on"
	// ModeShadow computes the plan and reports the WouldPrune set but does NOT
	// mutate the body — the safe dogfood rung (invariant 3: a drop is auditable
	// out of band before it is ever applied).
	ModeShadow Mode = "shadow"
	// ModeOff disables the curate: the body is forwarded byte-identical.
	ModeOff Mode = "off"
)

// DefaultMode is the out-of-the-box posture: curate both directions (epic #751).
const DefaultMode = ModeOn

// ParseMode resolves a host flag string to a Mode. Empty (flag unset) ⇒ the
// default-on posture; an unrecognized value resolves to DefaultMode with ok=false
// so the host can warn rather than silently disabling the curate (fail toward the
// epic's on-by-default posture, never silently off). Recognized values are
// case-insensitive-exact: "on", "shadow", "off".
func ParseMode(s string) (m Mode, ok bool) {
	switch s {
	case "":
		return DefaultMode, true
	case string(ModeOn):
		return ModeOn, true
	case string(ModeShadow):
		return ModeShadow, true
	case string(ModeOff):
		return ModeOff, true
	default:
		return DefaultMode, false
	}
}

// Applies reports whether this mode mutates the body (only ModeOn does). The host
// calls CompactInboundTools regardless to get the WouldPrune set, then applies the
// result only when Applies is true — so ModeShadow and ModeOff share one code path
// that differs by this single bit.
func (m Mode) Applies() bool { return m == ModeOn }

// Computes reports whether this mode computes the plan at all. ModeOff short-circuits
// (no plan, no log); ModeOn and ModeShadow both compute (the shadow rung logs without
// mutating).
func (m Mode) Computes() bool { return m == ModeOn || m == ModeShadow }

// Observation is the legible, content-free record of one curate decision, for the
// host to log out of band (invariant 3 — no silent vanish). It carries only tool
// NAMES and counts, never any prompt/result bytes, so it is safe to emit to a log
// or a metric. In ModeShadow, Applied is false and WouldPrune names the drop that
// was withheld; in ModeOn, Applied is true and WouldPrune == the names actually
// removed.
type Observation struct {
	// Mode is the curate mode in force for this decision.
	Mode Mode
	// WouldPrune is the tool names the plan selected, in tools[] order. In ModeOn
	// these were actually removed; in ModeShadow they were computed but withheld.
	WouldPrune []string
	// Applied reports whether the body was actually mutated (true only in ModeOn
	// when the splice changed the body).
	Applied bool
	// SkipReason names why no prune was applied when Applied is false: a curate
	// mode that did not fire (ModeShadow / ModeOff) or a spine identity
	// (PruneResult.SkipReason). Empty when Applied is true.
	SkipReason string
}

// Observe folds a mode and a spine PruneResult into the legible Observation the host
// logs. It is the one place the mode's "compute vs apply" bit is reconciled with the
// spine's identity-vs-change verdict, so the host never has to re-derive it:
//
//   - ModeOff ⇒ nothing computed; WouldPrune empty, Applied false, SkipReason "off".
//   - ModeShadow ⇒ WouldPrune names what the spine WOULD have dropped (res.Pruned
//     on a change, else the spine's named identity reason), Applied false.
//   - ModeOn ⇒ Applied == res.Changed; WouldPrune == res.Pruned; on identity the
//     SkipReason is the spine's named reason.
//
// res is the result of running CompactInboundTools with the plan. For ModeShadow the
// host still runs the spine (to learn what WOULD prune) but does NOT swap in res.Body.
func Observe(mode Mode, res PruneResult) Observation {
	obs := Observation{Mode: mode}
	switch mode {
	case ModeOff:
		obs.SkipReason = string(ModeOff)
		return obs
	case ModeShadow:
		obs.WouldPrune = res.Pruned
		if !res.Changed {
			obs.SkipReason = res.SkipReason
		}
		return obs
	default: // ModeOn (and any future-default mode)
		obs.Applied = res.Changed
		obs.WouldPrune = res.Pruned
		if !res.Changed {
			obs.SkipReason = res.SkipReason
		}
		return obs
	}
}

// Curate is the single mode-gated host branch: given the curate Mode, the outbound
// request bytes, the spine plan, and the spine's decode callback, it returns the body
// the host MUST forward upstream plus the legible Observation to log out of band. It
// is the one place the off/shadow/on decision over a REAL body lives, so the cmd/fak
// flag wiring (the deferred cross-lane half) only resolves a flag string to a Mode and
// then calls this — it never re-implements the shadow guard and so cannot get it wrong.
//
// The load-bearing safety invariant (epic #751 invariant 3 + the rung-4 shadow rung)
// is enforced HERE, in code, not left as a prose contract on the caller:
//
//   - ModeOff   ⇒ the spine never runs; the returned body IS raw (same backing slice).
//   - ModeShadow ⇒ the spine runs to learn what WOULD prune, but the returned body IS
//     raw (same backing slice) — the body is NEVER mutated, only the Observation logs
//     the withheld drop. This is the safe dogfood rung.
//   - ModeOn    ⇒ the spine runs and its result body is forwarded; on a real prune the
//     body is the spliced (shorter) bytes, on a spine identity it is raw.
//
// Only ModeOn can return a body whose bytes differ from raw, and only when the spine
// itself proved the splice cache-safe. The caller can detect "did anything change?"
// via the returned Observation.Applied (true only on a ModeOn change). decode is the
// spine's re-decode proof callback (nil skips only the parse re-check; see
// CompactInboundTools); it is unused in ModeOff (the spine never runs).
func Curate(mode Mode, raw []byte, plan ToolPlan, decode func([]byte) error) (body []byte, obs Observation) {
	if !mode.Computes() {
		// ModeOff: short-circuit. The spine never runs and raw is forwarded verbatim.
		return raw, Observe(mode, PruneResult{})
	}
	res := CompactInboundTools(raw, plan, decode)
	obs = Observe(mode, res)
	if mode.Applies() {
		// ModeOn: forward the spine's (possibly spliced) body.
		return res.Body, obs
	}
	// ModeShadow: the spine ran to populate the Observation, but the body is NEVER
	// swapped — forward the original bytes unchanged (the shadow-never-mutates guard).
	return raw, obs
}
