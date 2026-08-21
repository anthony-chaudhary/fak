package syspromptmmu

import (
	"fmt"
	"sort"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/cachemeta"
)

// style.go — #5051 (epic #1258, builds on #3308): the user-facing *style selection* surface
// over the numeric terseness levels steering.go already produces.
//
// steering.go owns the mechanism: a CLOSED level set 1..4 whose text is a fixed literal,
// emitted as an after-breakpoint overlay segment so it can never bust the cached prefix.
// What it does not own is a way for a person to ASK for it — the only knob is a raw integer
// in FAK_STEERING_LEVEL, which is neither discoverable nor self-describing. This file adds
// the naming layer: a small closed set of style names mapped onto those same levels, so
// frugality is a deliberate, visible choice rather than a buried env integer.
//
// Clean-room borrow of a TECHNIQUE only: the idea of a *named conversation style* that
// adjusts the system prompt (rather than a numeric knob) is the honest core of distill's
// `/distill` style mode. No source text was copied; the naming layer is re-derived in Go
// against fak's own overlay seams. Deliberately NOT adopted: distill's output-format DSL.
// fak steers with calibrated prose, which stays readable and cannot corrupt a response shape
// the caller did not ask for — a style here selects a level, never a wire format.
//
// Cache safety is INHERITED, not re-argued: a style resolves to a level and the bytes come
// from the #3308 producer, so every guarantee steering_test.go proves (byte-stable per
// level, distinct across levels, resident-prefix digest unmoved across a splice) holds
// verbatim. This file introduces no new request bytes and is not a second source of truth
// for the steering text — TestStyleSegmentMatchesProducer pins that.
//
// Scope fence: this is the SELECTION surface and its producer-side read-out. It does NOT
// wire steering into the live request path — that is #5047, still open — so selecting a
// style changes what DescribeStyle reports and nothing about a live request until that
// wiring lands. Advisory, opt-in, off by default; no savings claim without a holdout.
//
// Tier: mechanism (2). Imports cachemeta(1) + stdlib only.

// StyleEnvVar is the operator-facing knob: a NAME, not a number. It sits beside
// SteeringEnvVar (the raw numeric level) rather than replacing it — unset or unrecognized
// leaves steering off, so the forward path never moves by accident.
const StyleEnvVar = "FAK_STYLE"

// The closed style vocabulary. Each name is a readable spelling of one point on the #3308
// calibrated scale; StyleFull is the off end and doubles as the fail-safe every refusal
// collapses to, so "I chose off" and "your value was refused" produce the same steering
// outcome (no block) while staying distinguishable in the read-out via Known.
const (
	StyleFamilyNative   = "native"
	StyleFamilyCaveman  = "caveman"
	StyleFamilyOriginal = "original"
)

const (
	StyleFull    = "full"    // level 0 — no steering block at all
	StyleConcise = "concise" // level 1 — mild: trim filler, keep substance
	StyleBrief   = "brief"   // level 2 — moderate: skip preamble and recap
	StyleTerse   = "terse"   // level 3 — strong: essential content only
	StyleMinimal = "minimal" // level 4 — maximal: the requested result, nothing else
)

// styleLevels is the CLOSED name→level map. It is deliberately small and ordinal: one name
// per level, so the scale stays unambiguous and a read-out can present the set in a
// meaningful order. TestStyleLevelClosedSet pins these pairings because changing one
// silently re-steers every session that selected that name.
var styleLevels = map[string]int{
	StyleFull:                 SteeringOff,
	StyleConcise:              1,
	StyleBrief:                2,
	StyleTerse:                3,
	StyleMinimal:              4,
	"native:low":              1,
	"native:medium":           2,
	"native:high":             3,
	"caveman:native:low":      1,
	"caveman:native:medium":   2,
	"caveman:native:high":     3,
	"caveman:original:low":    1,
	"caveman:original:medium": 2,
	"caveman:original:high":   3,
}

// StyleReadout is the read-out: everything needed to SEE and PROVE the selection seam
// without calling a model — which style is in force, the level it maps to, whether a block
// is actually appended, and the exact bytes plus the witness that identifies them. This is
// `fak headroom`'s discipline applied to steering: the operator reads the real segment, not
// a description of it. All fields are comparable so two read-outs of one style can be
// compared directly (byte-stability).
type StyleReadout struct {
	// Style is the canonical style in force — always a member of the closed set, and
	// always StyleFull when the requested name was refused.
	Style     string
	Family    string
	Intensity string
	// Level is the terseness level Style maps to (SteeringOff when steering is off).
	Level int
	// Known reports whether the requested name was in the closed set. False means it was
	// refused and fell back to StyleFull; it does NOT by itself mean steering is on, since
	// an explicit `full` is Known and still off.
	Known bool
	// Applied reports whether a steering block is actually appended. False for StyleFull
	// and for every refused name — the explicit off and the fail-safe are one state.
	Applied bool
	// Segment is the exact text appended after the cache breakpoint, empty when steering is
	// off. Never a summary — the literal bytes the producer emits.
	Segment string
	// Witness is the producer's content-derived witness for Segment, empty when off. It
	// makes drift between the read-out and the producer detectable rather than cosmetic.
	Witness          string
	SourceRevision   string
	SourceDigest     string
	ActivationSource string
	DisableCommand   string
}

// StyleLevel maps a style name to its terseness level. The set is CLOSED and the refusal is
// fail-safe: an unknown name returns (SteeringOff, false) — never a guess, never a nearest
// match, never a fabricated block. Input is trimmed and lowercased so `FAK_STYLE=Terse` or a
// stray trailing space resolves, which widens the accepted SPELLING of a member without
// widening the SET of members.
func StyleLevel(name string) (int, bool) {
	level, ok := styleLevels[canonicalStyle(name)]
	if !ok {
		return SteeringOff, false
	}
	return level, true
}

// canonicalStyle normalizes an operator-supplied name to its comparison form. Kept separate
// from StyleLevel so DescribeStyle can report the canonical spelling it resolved to.
func canonicalStyle(name string) string {
	canonical := strings.ToLower(strings.TrimSpace(name))
	// User-facing Caveman shorthand selects fak's safe native implementation. The canonical
	// readout always expands the implementation slot so captures never confuse it with a
	// future provenance-checked `caveman:original:*` adapter.
	parts := strings.Split(canonical, ":")
	if len(parts) == 2 && parts[0] == StyleFamilyCaveman {
		canonical = StyleFamilyCaveman + ":" + StyleFamilyNative + ":" + parts[1]
	}
	return canonical
}

// StyleNames lists the closed vocabulary in ascending terseness order (full first), for a
// read-out or a flag's help text. Ordered by level rather than alphabetically because the
// set is a scale, and presenting it out of order misrepresents what the names mean.
func StyleNames() []string {
	names := make([]string, 0, len(styleLevels))
	for name := range styleLevels {
		names = append(names, name)
	}
	sort.Slice(names, func(i, j int) bool { return styleLevels[names[i]] < styleLevels[names[j]] })
	return names
}

// styleIdentity separates implementation family from intensity. Legacy names are native
// aliases. "original" is deliberately absent: foreign prompt bytes must be provenance checked.
func styleIdentity(name string, level int) (family, intensity string) {
	canonical := canonicalStyle(name)
	parts := strings.Split(canonical, ":")
	if len(parts) == 3 && parts[0] == StyleFamilyCaveman && (parts[1] == StyleFamilyNative || parts[1] == StyleFamilyOriginal) {
		return StyleFamilyCaveman + ":" + parts[1], parts[2]
	}
	if len(parts) == 2 {
		return parts[0], parts[1]
	}
	if canonical == StyleFull {
		return StyleFamilyNative, "off"
	}
	return StyleFamilyNative, map[int]string{1: "low", 2: "medium", 3: "high", 4: "ultra"}[level]
}

// DescribeStyle resolves a raw, operator-supplied name into the full read-out, applying the
// fail-safe: anything outside the closed set — including the empty string — resolves to
// StyleFull with Known=false and carries no block, so the request path stays exactly where
// it was while the caller can still report that the name was not understood.
//
// The segment bytes come straight from the #3308 producer, so this read-out can never drift
// into being a second source of truth for the steering text.
func DescribeStyle(name string) StyleReadout {
	level, known := StyleLevel(name)
	if !known {
		return StyleReadout{Style: StyleFull, Family: StyleFamilyNative, Intensity: "off", Level: SteeringOff}
	}
	family, intensity := styleIdentity(name, level)
	out := StyleReadout{Style: canonicalStyle(name), Family: family, Intensity: intensity, Level: level, Known: true, ActivationSource: StyleEnvVar, DisableCommand: "set " + StyleEnvVar + "=" + StyleFull}
	var seg cachemeta.PromptSegment
	var ok bool
	if family == StyleFamilyCaveman+":"+StyleFamilyOriginal {
		seg, ok = cavemanOriginalSegment(intensity)
		if !ok {
			return StyleReadout{Style: StyleFull, Family: StyleFamilyNative, Intensity: "off", Level: SteeringOff}
		}
		out.SourceRevision = CavemanOriginalRevision
		out.SourceDigest = CavemanOriginalSourceDigest
	} else {
		seg, ok = SteeringSegment(level)
	}
	if ok {
		out.Applied = true
		out.Segment = string(seg.Content)
		out.Witness = seg.Witness
	}
	return out
}

// ResolveStyle is the strict request-boundary form of DescribeStyle. DescribeStyle stays
// total for read-only diagnostics, while callers that are about to start a run use this
// form so an unknown or reserved selection cannot silently collapse to full/off.
func ResolveStyle(name string) (StyleReadout, error) {
	readout := DescribeStyle(name)
	if !readout.Known {
		return readout, fmt.Errorf("unknown response profile %q", strings.TrimSpace(name))
	}
	return readout, nil
}

// StyleFromEnv resolves the style selected by StyleEnvVar, reading through the supplied
// lookup. The lookup is injected rather than calling os.Getenv directly so the selection
// surface is testable without mutating process state, and so an embedder can source the
// selection from session policy instead of the environment. A nil lookup is treated as
// "nothing selected" — steering off — rather than falling back to the ambient environment,
// which keeps the zero-config path deterministic.
func StyleFromEnv(getenv func(string) string) StyleReadout {
	if getenv == nil {
		return DescribeStyle("")
	}
	return DescribeStyle(getenv(StyleEnvVar))
}

// StyleSegment is the producer entry for a NAMED style — the counterpart to
// SteeringSegment(level) for callers that hold a name. It inherits the exact bytes, so a
// style and its level are indistinguishable on the wire. StyleFull and every refused name
// return ok false and the zero segment, which must not be appended.
func StyleSegment(name string) (cachemeta.PromptSegment, bool) {
	readout := DescribeStyle(name)
	if !readout.Applied {
		return cachemeta.PromptSegment{}, false
	}
	if readout.Family == StyleFamilyCaveman+":"+StyleFamilyOriginal {
		return cavemanOriginalSegment(readout.Intensity)
	}
	return SteeringSegment(readout.Level)
}
