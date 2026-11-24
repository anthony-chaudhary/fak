package routing

import (
	"errors"
	"fmt"
	"regexp"
	"strconv"
	"strings"
	"unicode"
)

// ErrInvalidDiscoveryFact: a discovery fact or derived tag is malformed or
// outside the closed vocabulary. Wrapped errors name the offending field/value.
var ErrInvalidDiscoveryFact = errors.New("routing: invalid GPU discovery fact")

// EngineDiscovery carries the GPU discovery facts the fabric turns into
// capability tags. Pure data; the zero value means "nothing known".
type EngineDiscovery struct {
	// VRAMGiB is total device VRAM in GiB. 0 = unknown/unadvertised.
	VRAMGiB int `json:"vram_gib,omitempty"`
	// ComputeCapability is the CUDA compute capability as "major.minor"
	// (e.g. "8.6"). "" = unknown.
	ComputeCapability string `json:"compute_capability,omitempty"`
	// TPWidth is the tensor-parallel width (ranks) the engine runs with.
	// 0 = unknown.
	TPWidth int `json:"tp_width,omitempty"`
	// MIGProfile is the MIG profile id when the device is a MIG instance
	// (e.g. "1g.5gb"). "" = not a MIG instance / unknown.
	MIGProfile string `json:"mig_profile,omitempty"`
}

// VRAMClass is a closed vocabulary bucket over total device VRAM. It is
// declaration only: the bucket boundaries live in VRAMClassFor and are never
// inferred from anything else.
type VRAMClass string

const (
	// VRAMClassUnknown is the fact-absent sentinel: no tag emitted.
	VRAMClassUnknown VRAMClass = ""
	// VRAMClassLt8 covers (0, 8) GiB.
	VRAMClassLt8 VRAMClass = "lt8"
	// VRAMClass8 covers [8, 16) GiB.
	VRAMClass8 VRAMClass = "8g"
	// VRAMClass16 covers [16, 32) GiB.
	VRAMClass16 VRAMClass = "16g"
	// VRAMClass32 covers [32, 64) GiB.
	VRAMClass32 VRAMClass = "32g"
	// VRAMClass64 covers [64, 96) GiB.
	VRAMClass64 VRAMClass = "64g"
	// VRAMClass96 covers [96, ∞) GiB.
	VRAMClass96 VRAMClass = "96g"
)

// vramClasses is the single source of truth for the closed VRAM-class
// vocabulary, in declaration order. VRAMClasses returns a copy.
var vramClasses = []VRAMClass{
	VRAMClassLt8,
	VRAMClass8,
	VRAMClass16,
	VRAMClass32,
	VRAMClass64,
	VRAMClass96,
}

// VRAMClasses returns every known VRAM class (for a witness/registry test
// that the vocabulary stays closed), in declaration order.
func VRAMClasses() []VRAMClass {
	out := make([]VRAMClass, len(vramClasses))
	copy(out, vramClasses)
	return out
}

// IsKnown reports whether c is a member of the closed VRAM-class vocabulary.
// VRAMClassUnknown ("") is NOT known: an unknown class is a refusal, never a
// guess.
func (c VRAMClass) IsKnown() bool {
	for _, known := range vramClasses {
		if c == known {
			return true
		}
	}
	return false
}

// VRAMClassFor maps total VRAM in GiB to its closed-vocabulary class. Zero
// means unknown and yields VRAMClassUnknown with no error; a negative count
// is a refusal (ErrInvalidDiscoveryFact), never a guess. Boundaries: 7->lt8,
// 8->8g, 15->8g, 16->16g, 31->16g, 32->32g, 63->32g, 64->64g, 95->64g,
// 96->96g.
func VRAMClassFor(gib int) (VRAMClass, error) {
	switch {
	case gib == 0:
		return VRAMClassUnknown, nil
	case gib < 0:
		return VRAMClassUnknown, fmt.Errorf("%w: VRAMGiB=%d is negative", ErrInvalidDiscoveryFact, gib)
	case gib < 8:
		return VRAMClassLt8, nil
	case gib < 16:
		return VRAMClass8, nil
	case gib < 32:
		return VRAMClass16, nil
	case gib < 64:
		return VRAMClass32, nil
	case gib < 96:
		return VRAMClass64, nil
	default:
		return VRAMClass96, nil
	}
}

// CCFamily is a closed vocabulary over CUDA compute-capability generations.
// GPU class == compute-capability family. It is declaration only: the CC
// mapping lives in CCFamilyFor and is never inferred from a device name.
type CCFamily string

const (
	// CCFamilyUnknown is the fact-absent sentinel: no tag emitted.
	CCFamilyUnknown CCFamily = ""
	// CCFamilyPascal covers 6.x.
	CCFamilyPascal CCFamily = "pascal"
	// CCFamilyVolta covers 7.0, 7.2.
	CCFamilyVolta CCFamily = "volta"
	// CCFamilyTuring covers 7.5.
	CCFamilyTuring CCFamily = "turing"
	// CCFamilyAmpere covers 8.0, 8.6, 8.7.
	CCFamilyAmpere CCFamily = "ampere"
	// CCFamilyAda covers 8.9.
	CCFamilyAda CCFamily = "ada"
	// CCFamilyHopper covers 9.0.
	CCFamilyHopper CCFamily = "hopper"
	// CCFamilyBlackwell covers 10.x, 12.x.
	CCFamilyBlackwell CCFamily = "blackwell"
)

// ccFamilies is the single source of truth for the closed CC-family
// vocabulary, in declaration order. CCFamilies returns a copy.
var ccFamilies = []CCFamily{
	CCFamilyPascal,
	CCFamilyVolta,
	CCFamilyTuring,
	CCFamilyAmpere,
	CCFamilyAda,
	CCFamilyHopper,
	CCFamilyBlackwell,
}

// CCFamilies returns every known CC family (for a witness/registry test that
// the vocabulary stays closed), in declaration order.
func CCFamilies() []CCFamily {
	out := make([]CCFamily, len(ccFamilies))
	copy(out, ccFamilies)
	return out
}

// IsKnown reports whether f is a member of the closed CC-family vocabulary.
// CCFamilyUnknown ("") is NOT known: an unknown family is a refusal, never a
// guess.
func (f CCFamily) IsKnown() bool {
	for _, known := range ccFamilies {
		if f == known {
			return true
		}
	}
	return false
}

// ccPattern is the well-formed compute-capability shape "major.minor".
var ccPattern = regexp.MustCompile(`^\d+\.\d+$`)

// CCFamilyFor maps a "major.minor" compute capability to its closed-vocabulary
// family. "" means unknown and yields CCFamilyUnknown with no error. A
// well-formed but unmapped CC (e.g. "5.0", "8.3", "11.0") is a refusal, never
// a guess; a malformed CC is the same refusal. Both wrap
// ErrInvalidDiscoveryFact.
func CCFamilyFor(computeCapability string) (CCFamily, error) {
	if computeCapability == "" {
		return CCFamilyUnknown, nil
	}
	if !ccPattern.MatchString(computeCapability) {
		return CCFamilyUnknown, fmt.Errorf("%w: compute capability %q is malformed", ErrInvalidDiscoveryFact, computeCapability)
	}
	parts := strings.SplitN(computeCapability, ".", 2)
	major, errMajor := strconv.Atoi(parts[0])
	minor, errMinor := strconv.Atoi(parts[1])
	if errMajor != nil || errMinor != nil {
		return CCFamilyUnknown, fmt.Errorf("%w: compute capability %q is malformed", ErrInvalidDiscoveryFact, computeCapability)
	}
	switch major {
	case 6:
		return CCFamilyPascal, nil
	case 7:
		switch minor {
		case 0, 2:
			return CCFamilyVolta, nil
		case 5:
			return CCFamilyTuring, nil
		}
	case 8:
		switch minor {
		case 0, 6, 7:
			return CCFamilyAmpere, nil
		case 9:
			return CCFamilyAda, nil
		}
	case 9:
		if minor == 0 {
			return CCFamilyHopper, nil
		}
	case 10, 12:
		return CCFamilyBlackwell, nil
	}
	return CCFamilyUnknown, fmt.Errorf("%w: compute capability %q has no known family", ErrInvalidDiscoveryFact, computeCapability)
}

// Tag builder prefixes. A derived tag is always prefix + canonical body.
const (
	// TagPrefixVRAM heads a VRAM-class tag ("vram:32g").
	TagPrefixVRAM = "vram:"
	// TagPrefixCC heads a CC-family GPU-class tag ("cc:ampere").
	TagPrefixCC = "cc:"
	// TagPrefixTP heads a tensor-parallel width tag ("tp:2").
	TagPrefixTP = "tp:"
	// TagPrefixMIG heads a MIG profile tag ("mig:1g.5gb").
	TagPrefixMIG = "mig:"
)

// VRAMClassTag renders the vram: tag for a class. An unknown class (including
// VRAMClassUnknown) is a refusal, never an empty tag.
func VRAMClassTag(c VRAMClass) (string, error) {
	if !c.IsKnown() {
		return "", fmt.Errorf("%w: VRAM class %q is not known", ErrInvalidDiscoveryFact, c)
	}
	return TagPrefixVRAM + string(c), nil
}

// CCFamilyTag renders the cc: tag for a family. An unknown family (including
// CCFamilyUnknown) is a refusal, never an empty tag.
func CCFamilyTag(f CCFamily) (string, error) {
	if !f.IsKnown() {
		return "", fmt.Errorf("%w: CC family %q is not known", ErrInvalidDiscoveryFact, f)
	}
	return TagPrefixCC + string(f), nil
}

// TPTag renders the tp: tag for a tensor-parallel width. Widths below 1 are a
// refusal: a zero width means unknown and must not become a tag.
func TPTag(width int) (string, error) {
	if width < 1 {
		return "", fmt.Errorf("%w: TP width %d is below 1", ErrInvalidDiscoveryFact, width)
	}
	return TagPrefixTP + strconv.Itoa(width), nil
}

// MIGProfileTag renders the mig: tag for a MIG profile id: trimmed and
// lowercased (" 1G.5GB " -> "mig:1g.5gb"). Empty, whitespace-only, or
// interior-whitespace profiles are a refusal, never a partial tag.
func MIGProfileTag(profile string) (string, error) {
	trimmed := strings.TrimSpace(profile)
	if trimmed == "" {
		return "", fmt.Errorf("%w: MIG profile %q is empty", ErrInvalidDiscoveryFact, profile)
	}
	if strings.IndexFunc(trimmed, unicode.IsSpace) >= 0 {
		return "", fmt.Errorf("%w: MIG profile %q has interior whitespace", ErrInvalidDiscoveryFact, profile)
	}
	return TagPrefixMIG + strings.ToLower(trimmed), nil
}

// TagsFromDiscovery derives the capability tags for one discovery record: at
// most one tag per family, in FIXED canonical order vram, cc, tp, mig. An
// absent fact contributes no tag; an invalid fact fails loud with no partial
// result — the first invalid fact in canonical order returns (nil, error)
// wrapping ErrInvalidDiscoveryFact. Zero-value discovery yields (nil, nil).
func TagsFromDiscovery(d EngineDiscovery) ([]string, error) {
	var tags []string
	if d.VRAMGiB != 0 {
		class, err := VRAMClassFor(d.VRAMGiB)
		if err != nil {
			return nil, fmt.Errorf("discovery vram_gib: %w", err)
		}
		tag, err := VRAMClassTag(class)
		if err != nil {
			return nil, fmt.Errorf("discovery vram_gib: %w", err)
		}
		tags = append(tags, tag)
	}
	if d.ComputeCapability != "" {
		family, err := CCFamilyFor(d.ComputeCapability)
		if err != nil {
			return nil, fmt.Errorf("discovery compute_capability: %w", err)
		}
		tag, err := CCFamilyTag(family)
		if err != nil {
			return nil, fmt.Errorf("discovery compute_capability: %w", err)
		}
		tags = append(tags, tag)
	}
	if d.TPWidth != 0 {
		tag, err := TPTag(d.TPWidth)
		if err != nil {
			return nil, fmt.Errorf("discovery tp_width: %w", err)
		}
		tags = append(tags, tag)
	}
	if d.MIGProfile != "" {
		tag, err := MIGProfileTag(d.MIGProfile)
		if err != nil {
			return nil, fmt.Errorf("discovery mig_profile: %w", err)
		}
		tags = append(tags, tag)
	}
	return tags, nil
}

// PopulateTags appends the tags derived from d to e.Tags in canonical order,
// skipping any derived tag the engine already carries. Existing tags are
// NEVER removed or reordered: degradation rungs match on "primary"/"small" via
// HasTag, and superset semantics must survive. A nil receiver or an invalid
// discovery is a refusal (ErrInvalidDiscoveryFact) with e.Tags unchanged.
func (e *Engine) PopulateTags(d EngineDiscovery) error {
	if e == nil {
		return fmt.Errorf("%w: nil engine", ErrInvalidDiscoveryFact)
	}
	derived, err := TagsFromDiscovery(d)
	if err != nil {
		return err
	}
	have := make(map[string]bool, len(e.Tags))
	for _, t := range e.Tags {
		have[t] = true
	}
	for _, tag := range derived {
		if have[tag] {
			continue
		}
		e.Tags = append(e.Tags, tag)
		have[tag] = true
	}
	return nil
}

// RequireGPUClass returns the TaskRequest.RequireTags value restricting
// assignment to engines carrying the cc: class tag for family. GPU class ==
// compute-capability family. Selection itself is the existing fabric path:
// RequireTags -> Engine.HasTag -> ReasonMissingTag ("MISSING_TAG") rejection.
// An unknown family is a refusal, never a wildcard.
func RequireGPUClass(family CCFamily) ([]string, error) {
	tag, err := CCFamilyTag(family)
	if err != nil {
		return nil, err
	}
	return []string{tag}, nil
}
