package disambiguation

import (
	"errors"
	"fmt"
	"sort"
)

const LifecycleSourceSelfTestSchemaVersion = "fak-disambiguation-lifecycle-source-self-test/1"

type LadderKind string

const (
	LadderLifecycle  LadderKind = "lifecycle"
	LadderActivation LadderKind = "activation"
	LadderMaturity   LadderKind = "maturity"
)

var ErrLadderSpelling = errors.New("incompatible ladder spelling")

var ladderDefinitions = map[LadderKind]struct {
	canonical string
	accepted  map[string]bool
}{
	LadderLifecycle:  {"index lifecycle class", map[string]bool{"current": true, "versioned": true, "research": true, "archived": true}},
	LadderActivation: {"activation posture", map[string]bool{"off": true, "shadow": true, "on": true}},
	LadderMaturity:   {"capability maturity rung", map[string]bool{"proposed": true, "prototyped": true, "tested": true, "dogfooded": true, "default": true}},
}

type LifecycleLadderResolution struct {
	Ladder        LadderKind `json:"ladder"`
	CanonicalTerm string     `json:"canonical_term"`
	Spellings     []string   `json:"spellings"`
	SourcePath    string     `json:"source_path"`
}

type LifecycleSourceSelfTestReport struct {
	Schema                        string                      `json:"schema"`
	IndexVersion                  string                      `json:"index_version"`
	Ladders                       []LifecycleLadderResolution `json:"ladders"`
	IncompatibleSpellingsRejected bool                        `json:"incompatible_spellings_rejected"`
}

func ResolveLadderSpelling(kind LadderKind, spelling string) (QueryResponse, error) {
	definition, ok := ladderDefinitions[kind]
	if !ok || !definition.accepted[spelling] {
		return QueryResponse{}, fmt.Errorf("%w: %s=%q", ErrLadderSpelling, kind, spelling)
	}
	return Query(definition.canonical)
}

func RunLifecycleSourceSelfTest() (LifecycleSourceSelfTestReport, error) {
	report := LifecycleSourceSelfTestReport{Schema: LifecycleSourceSelfTestSchemaVersion, IndexVersion: PublicIndexVersion}
	order := []LadderKind{LadderLifecycle, LadderActivation, LadderMaturity}
	for _, kind := range order {
		definition := ladderDefinitions[kind]
		resolved, err := Query(definition.canonical)
		if err != nil {
			return report, err
		}
		spellings := make([]string, 0, len(definition.accepted))
		for spelling := range definition.accepted {
			spellings = append(spellings, spelling)
		}
		sort.Strings(spellings)
		report.Ladders = append(report.Ladders, LifecycleLadderResolution{Ladder: kind, CanonicalTerm: resolved.Entry.Identity.CanonicalTerm, Spellings: spellings, SourcePath: resolved.Entry.Sources[0].Locator})
	}
	_, errLifecycle := ResolveLadderSpelling(LadderLifecycle, "shadow")
	_, errActivation := ResolveLadderSpelling(LadderActivation, "archived")
	_, errMaturity := ResolveLadderSpelling(LadderMaturity, "benchmarked")
	report.IncompatibleSpellingsRejected = errors.Is(errLifecycle, ErrLadderSpelling) && errors.Is(errActivation, ErrLadderSpelling) && errors.Is(errMaturity, ErrLadderSpelling)
	if !report.IncompatibleSpellingsRejected {
		return report, errors.New("incompatible ladder spellings accepted")
	}
	return report, nil
}
