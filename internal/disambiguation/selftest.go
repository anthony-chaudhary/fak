package disambiguation

import (
	"encoding/json"
	"fmt"
	"strings"
	"time"
)

// SelfTestReport is the stable, machine-readable result of exercising the v1
// contract without filesystem, network, or private-source dependencies.
type SelfTestReport struct {
	Schema            string                   `json:"schema"`
	CompleteAccepted  bool                     `json:"complete_accepted"`
	OmissionsRejected []string                 `json:"omissions_rejected"`
	Freshness         FreshnessSelfCheckReport `json:"freshness"`
}

// SelfTest is the compact compatibility seam used by package tests and callers.
// Detailed omission evidence is available from RunSelfTest.
func SelfTest() error {
	_, err := RunSelfTest()
	return err
}

// RunSelfTest proves through the strict JSON reader that one complete record is
// accepted and every path declared required by Descriptor is rejected when
// omitted. Identity aliases is included: its array must be present on the wire,
// although [] is a valid value.
func RunSelfTest() (SelfTestReport, error) {
	completeJSON, err := json.Marshal(SelfTestEntry())
	if err != nil {
		return SelfTestReport{}, fmt.Errorf("encode complete record: %w", err)
	}
	report := SelfTestReport{Schema: EntrySchemaVersion, Freshness: FreshnessSelfCheck()}
	if !report.Freshness.Passed {
		return report, fmt.Errorf("freshness four-state self-test failed")
	}
	if _, err := ParseEntry(completeJSON); err != nil {
		return report, fmt.Errorf("complete record rejected: %w", err)
	}
	report.CompleteAccepted = true

	for _, path := range Descriptor().Required {
		candidate, err := jsonWithoutPath(completeJSON, path)
		if err != nil {
			return report, fmt.Errorf("prepare missing %s: %w", path, err)
		}
		if _, err := ParseEntry(candidate); err == nil {
			return report, fmt.Errorf("missing %s was accepted", path)
		}
		report.OmissionsRejected = append(report.OmissionsRejected, path)
	}
	return report, nil
}

func jsonWithoutPath(src []byte, path string) ([]byte, error) {
	var root map[string]any
	if err := json.Unmarshal(src, &root); err != nil {
		return nil, err
	}
	parts := strings.Split(path, ".")
	var current any = root
	for i, part := range parts {
		array := strings.HasSuffix(part, "[]")
		key := strings.TrimSuffix(part, "[]")
		object, ok := current.(map[string]any)
		if !ok {
			return nil, fmt.Errorf("%q parent is not an object", part)
		}
		if i == len(parts)-1 {
			if _, exists := object[key]; !exists {
				return nil, fmt.Errorf("%q is absent from complete fixture", key)
			}
			delete(object, key)
			return json.Marshal(root)
		}
		current, ok = object[key]
		if !ok {
			return nil, fmt.Errorf("%q is absent from complete fixture", key)
		}
		if array {
			values, ok := current.([]any)
			if !ok || len(values) == 0 {
				return nil, fmt.Errorf("%q is not a populated array", key)
			}
			current = values[0]
		}
	}
	return nil, fmt.Errorf("empty required path")
}

// SelfTestEntry returns a deterministic complete public-source record used by
// the package self-test and CLI witness. It is sample data, not an index writer.
func SelfTestEntry() Entry {
	return Entry{
		Schema: EntrySchemaVersion,
		Identity: Identity{
			CanonicalTerm: "agent kernel",
			Aliases:       []string{"fused agent kernel"},
		},
		Definition: "The fak management boundary that governs model traffic, tool effects, context, and recovery.",
		Contrasts: []Contrast{{
			CanonicalTerm:       "compute kernel",
			Explanation:         "An arithmetic routine executed by a processor; it does not govern an agent's tool effects.",
			RequiredPair:        boolPointer(true),
			ForbiddenConflation: boolPointer(true),
		}},
		Scope: Scope{Kind: "product", Value: "fak"},
		Owner: Owner{Leaf: "kernel", Lane: "kernel"},
		Sources: []SourceWitness{{
			Kind: "document", Locator: "README.md#how-it-works", Revision: "self-test",
		}},
		Freshness: Freshness{
			Verdict: FreshnessFresh, ReasonCode: "SOURCE_CURRENT",
			CheckedAt: time.Date(2026, 8, 10, 0, 0, 0, 0, time.UTC).Format(time.RFC3339),
			Probe:     "hermetic-self-test",
		},
		Lifecycle: Lifecycle{Class: LifecycleCurrent, Rollout: RolloutOn},
	}
}
