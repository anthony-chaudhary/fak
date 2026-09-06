package policy

import (
	"embed"
	"fmt"
	"strings"
)

// The least-agency preset ladder (issue #3276).
//
// Forrester's AEGIS framing of "least agency" bounds the DECISIONS an agent may
// take, not merely the data it may touch; Gartner's companion guidance is that
// uniform governance across autonomy levels is itself a failure mode. fak
// default-denies with no policy, which is the right floor but the wrong first
// step: authoring a manifest from scratch is the price of the first policy
// decision. These four presets are that ladder, pre-authored and witnessed —
// an adopter names a rung and dials up as trust grows.
//
// The rungs, in ascending order of WRITE / side-effect agency:
//
//	read-only      observe and plan; no writes, no side effects
//	propose-only   may draft (diff/plan); apply, push, and deploy are refused
//	bounded-write  may write inside a declared region; nothing outside it
//	autonomous     the broad loop; still floored by the self-modify rung
//
// The ladder orders the CONSEQUENCE a rung may cause, and it is not a superset
// chain over permitted tool NAMES. propose-only alone admits the planning family
// (ExitPlanMode, diff_infra, plan_deploy, validate_terraform, and the describe_ /
// dryrun_ prefixes); the two rungs above it DEFAULT_DENY those names while
// permitting strictly higher-consequence writes. So a rung is not a superset of
// the rung below — the set of permitted names is a lattice, and only the
// write/side-effect axis is totally ordered. TestPresetLadderWidensTheWriteAxis
// witnesses the axis that IS ordered; whether the upper rungs should also inherit
// the planning family is a live design question, not an accident this comment
// papers over.
//
// Each rung is a real fak-policy/v1 manifest, embedded in the binary and loaded
// through the SAME ParseRuntime path as any on-disk --policy file, so a preset
// gets no validation exemption: unknown fields are refused, and every deny cites
// the closed refusal vocabulary. TestLeastAgencyPresetsRoundTrip is the
// can't-rot gate.
//
// What a rung IS and IS NOT. A preset encodes the capability floor — which tool
// NAMES and ARGUMENT VALUES the agent may invoke. It is a permissions floor, not
// a detection guarantee. The load-bearing refusals are the structural ones
// (DEFAULT_DENY for any unlisted tool; the in-kernel self-modify rung, which
// decides on shape, not text). See examples/presets/README.md.
//
// Honest reduction, named rather than hidden: `bounded-write` confines writes to
// a STATIC region glob, because arg_rules.allow_glob is static manifest text.
// fak-policy/v1 has no per-session dynamic lease token an arg rule can bind to,
// so the tier's witnessed boundary is "refuses an out-of-REGION write", not
// "refuses an out-of-LEASE write". Binding the glob to a live lease at dispatch
// would touch the adjudicator/session layer; it is tracked separately.

//go:embed presets/*.json
var presetFS embed.FS

// Built-in least-agency preset names, in ascending order of permitted agency.
const (
	PresetReadOnly     = "read-only"
	PresetProposeOnly  = "propose-only"
	PresetBoundedWrite = "bounded-write"
	PresetAutonomous   = "autonomous"
)

// presetLadder defines the ascending write-agency order:
// upgrading a tier is the one-line change of naming the next rung.
var presetLadder = []string{
	PresetReadOnly,
	PresetProposeOnly,
	PresetBoundedWrite,
	PresetAutonomous,
}

// isLadderRung reports whether name is one of the four reviewed rungs.
func isLadderRung(name string) bool {
	for _, rung := range presetLadder {
		if rung == name {
			return true
		}
	}
	return false
}

// PresetNames returns the built-in preset names in ascending write-agency order.
// The returned slice is a copy — a caller cannot reorder the ladder in place.
func PresetNames() []string {
	out := make([]string, len(presetLadder))
	copy(out, presetLadder)
	return out
}

// PresetManifest returns the raw embedded fak-policy/v1 manifest bytes for a
// named preset. An unknown name fails loud and names the valid rungs: a refusal
// is a redirect, not a dead end.
//
// Ladder membership is checked BEFORE the embedded read, and that order is the
// security property. `go:embed presets/*.json` ships every JSON file in the
// directory, but only the four ladder rungs are round-tripped and tier-witnessed
// by the tests. Reading by raw filename would let an unreviewed manifest dropped
// into presets/ — say a wide-open one — load through LoadPreset and be enforced
// as a capability floor without a single test ever having graded it. The gate,
// plus TestEmbeddedPresetsAreExactlyTheLadder, makes that unreachable from both
// ends: a file with no rung cannot be named, and a file with no rung fails CI.
func PresetManifest(name string) ([]byte, error) {
	if !isLadderRung(name) {
		return nil, fmt.Errorf("policy: unknown preset %q; valid rungs (ascending write agency): %s",
			name, strings.Join(PresetNames(), ", "))
	}
	b, err := presetFS.ReadFile("presets/" + name + ".json")
	if err != nil { // a ladder rung with no embedded file: a build-time breakage, not operator error
		return nil, fmt.Errorf("policy: preset %q is a ladder rung but has no embedded manifest: %w", name, err)
	}
	return b, nil
}

// LoadPreset resolves a built-in least-agency preset by name into the full
// boot-time policy set, applying the same validation as any on-disk manifest.
// Callers wanting only the name-level capability floor take rt.Adjudicator, as
// Load does.
func LoadPreset(name string) (Runtime, error) {
	b, err := PresetManifest(name)
	if err != nil {
		return Runtime{}, err
	}
	rt, err := ParseRuntime(b)
	if err != nil {
		return Runtime{}, fmt.Errorf("policy preset %q: %w", name, err)
	}
	return rt, nil
}
