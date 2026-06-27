package promptmmu

// config.go is the flag + observability rung (epic #751 rung 4, #754): the curate
// mode the host flag selects, and the legible drop record the gateway logs out of
// band. It mirrors the co-travel / reset-on-budget gate convention: a closed mode
// vocabulary with a SAFE default, a shadow rung that computes the plan without
// mutating the body, and an off rung. The cmd/fak flag string + the gateway log
// emission are the deferred cross-lane halves; the mode semantics live here so the
// spine and its host agree on the same closed set.

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
