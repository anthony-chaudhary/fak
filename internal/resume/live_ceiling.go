package resume

// Host-wide live-resume ceiling resolution + provenance. AdmitSource enforces a
// SourcePolicy.MaxLiveResumes; this file answers the sibling question the policy
// leaves open — where does that number COME from. Historically it was a static
// constant 4 hard-fallen-back in three places (the admit CLI flag default, the
// watchdog tick, and a fresh policy file), with no way to change it but hand-
// authoring .fak/resume-source-policy.json (#5093). ResolveMaxLiveResumes turns
// that single number into a resolved value carrying its source, so an operator can
// set it via env/param OR let it scale with healthy-seat headroom, and
// `fak resume admit --explain` can show which rail actually decided it.
//
// This mirrors DeriveWatchdogCap (headroom.go), the sibling derivation for the
// per-TICK launch cap (#3564): the per-tick cap bounds how many launches fire in
// one tick, this bounds how many `claude --resume` are alive at once across ticks —
// the dimension the per-source 529 burst wall actually keys on. Both scale to the
// same healthy-seat evidence; neither reads a clock or does I/O.

import "fmt"

// DefaultMaxLiveResumes is the static fallback the ceiling collapses to when no
// policy value, env/param override, or seat census is available. It preserves the
// historical constant so an un-wired host behaves exactly as it did before #5093.
const DefaultMaxLiveResumes = 4

// MaxLiveResumesSource names where a resolved ceiling came from, a closed set so
// `--explain` reports provenance as a routable token, not free-text.
type MaxLiveResumesSource string

const (
	// MaxLiveFromFlag: an explicit CLI --max-live argument to `fak resume admit` — a
	// one-off invocation's override, the strongest signal in the admit path.
	MaxLiveFromFlag MaxLiveResumesSource = "flag"
	// MaxLiveFromEnv: an explicit FAK_RESUME_MAX_LIVE env / -MaxLiveResumes param —
	// the operator's live knob, highest precedence among the standing rails (flag-
	// over-file, the same order applyResumeSourceFlagOverrides takes).
	MaxLiveFromEnv MaxLiveResumesSource = "env"
	// MaxLiveFromConfigFile: an explicit max_live_resumes in the source policy JSON —
	// the durable hand-authored value, used when no env/param overrides it.
	MaxLiveFromConfigFile MaxLiveResumesSource = "config_file"
	// MaxLiveFromDerived: scaled from healthy-seat headroom, so the ceiling grows with
	// the pool instead of a stale constant turning eligible sessions away.
	MaxLiveFromDerived MaxLiveResumesSource = "derived_seats"
	// MaxLiveFromDefault: the static DefaultMaxLiveResumes fallback (nothing else set).
	MaxLiveFromDefault MaxLiveResumesSource = "default"
)

// MaxLiveResumesInput carries the candidate sources in precedence order. An absent
// source is left at its zero value and skipped, so a caller supplies only what it has.
type MaxLiveResumesInput struct {
	// EnvPresent + EnvValue: the FAK_RESUME_MAX_LIVE / -MaxLiveResumes override. When
	// EnvPresent, EnvValue wins outright (0 meaning "explicitly disable the ceiling").
	EnvPresent bool
	EnvValue   int

	// ConfigValue is the explicit max_live_resumes from the loaded policy file (<=0 =
	// unset). Used when no env override is present.
	ConfigValue int

	// Seats is the healthy-seat census for derivation (nil/empty = no derivation
	// possible, fall through to the default).
	Seats []HeadroomSeat

	// Floor is the derivation floor AND the static default when nothing else applies
	// (<1 ⇒ DefaultMaxLiveResumes). Ceiling caps the derived value (<=0 ⇒ no upper
	// bound beyond the floor). SeatCap is the safe live resumes per healthy seat
	// (<1 ⇒ 1).
	Floor   int
	Ceiling int
	SeatCap int
}

// ResolvedMaxLiveResumes is the ceiling plus its provenance, the shape `--explain`
// surfaces so an operator sees both the number and the rail that produced it.
type ResolvedMaxLiveResumes struct {
	Value  int                  `json:"value"`
	Source MaxLiveResumesSource `json:"source"`
	Detail string               `json:"detail"`
}

// ResolveMaxLiveResumes picks the host-wide live-resume ceiling from the highest-
// precedence source present: explicit env/param → explicit policy-file value →
// healthy-seat derivation → static default. It is pure: no clock, no I/O, no env
// read (the caller supplies EnvPresent/EnvValue) — every input is a fold the shell
// computes, matching AdmitSource.
func ResolveMaxLiveResumes(in MaxLiveResumesInput) ResolvedMaxLiveResumes {
	floor := in.Floor
	if floor < 1 {
		floor = DefaultMaxLiveResumes
	}
	switch {
	case in.EnvPresent:
		v := in.EnvValue
		if v < 0 {
			v = 0
		}
		return ResolvedMaxLiveResumes{
			Value:  v,
			Source: MaxLiveFromEnv,
			Detail: "FAK_RESUME_MAX_LIVE / -MaxLiveResumes override",
		}
	case in.ConfigValue > 0:
		return ResolvedMaxLiveResumes{
			Value:  in.ConfigValue,
			Source: MaxLiveFromConfigFile,
			Detail: "explicit max_live_resumes in the source policy file",
		}
	case len(in.Seats) > 0:
		value, healthy := DeriveMaxLiveResumes(in.Seats, floor, in.Ceiling, in.SeatCap)
		return ResolvedMaxLiveResumes{
			Value:  value,
			Source: MaxLiveFromDerived,
			Detail: fmt.Sprintf("derived from %d healthy seat(s): floor=%d ceiling=%s per_seat=%d",
				healthy, floor, ceilingLabel(in.Ceiling), seatCapOrOne(in.SeatCap)),
		}
	default:
		return ResolvedMaxLiveResumes{
			Value:  floor,
			Source: MaxLiveFromDefault,
			Detail: "static default (no env/param override, policy value, or seat census)",
		}
	}
}

// DeriveMaxLiveResumes scales the live ceiling to currently-healthy seat headroom:
// healthy seats × per-seat cap, clamped to [floor, ceiling]. It counts a seat healthy
// exactly as DeriveWatchdogCap does (available and not throttled), so the per-tick and
// live-ceiling derivations agree on the pool. Returns the derived ceiling and the
// healthy-seat count (for the provenance detail). A zero/negative ceiling means no
// upper bound beyond the floor.
func DeriveMaxLiveResumes(seats []HeadroomSeat, floor, ceiling, seatCap int) (int, int) {
	if floor < 1 {
		floor = DefaultMaxLiveResumes
	}
	if seatCap < 1 {
		seatCap = 1
	}
	healthy := 0
	for _, s := range seats {
		if s.Available && !s.Throttled {
			healthy++
		}
	}
	value := healthy * seatCap
	if value < floor {
		value = floor
	}
	if ceiling > 0 && value > ceiling {
		value = ceiling
	}
	return value, healthy
}

func ceilingLabel(ceiling int) string {
	if ceiling <= 0 {
		return "none"
	}
	return fmt.Sprintf("%d", ceiling)
}

func seatCapOrOne(seatCap int) int {
	if seatCap < 1 {
		return 1
	}
	return seatCap
}
