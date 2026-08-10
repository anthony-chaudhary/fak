package lightgapport

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"sort"
)

const Schema = "fak-lightgap-portability/1"

type Witness struct {
	Path string `json:"path"`
	Test string `json:"test"`
}

type Swap struct {
	ID           string             `json:"id"`
	Fak          Witness            `json:"fak"`
	Alternatives map[string]Witness `json:"alternatives"`
}

type Report struct {
	Schema string `json:"schema"`
	Swaps  []Swap `json:"swaps"`
}

// Contract is the lock-in/exit-path instrument used by the lightgap scorecard.
// A swap is witnessed only when the named committed test exists; prose and
// feature lists are deliberately not accepted as substitutes.
func Contract() Report {
	alts := func() map[string]Witness {
		return map[string]Witness{
			"claude-code-bare": {}, "raw-sglang": {}, "formal-isolation": {}, "llm-tracing": {},
		}
	}
	mk := func(id, path, test string) Swap {
		return Swap{ID: id, Fak: Witness{Path: path, Test: test}, Alternatives: alts()}
	}
	return Report{Schema: Schema, Swaps: []Swap{
		mk("agent", "cmd/fak/guard_harness_profiles_test.go", "TestHarnessProfileFloorCoverage"),
		mk("provider", "internal/accounts/seat_endpoint_test.go", "TestEnvOverlayCarriesEndpointAlone"),
		mk("model", "internal/engine/sglang_test.go", "TestSGLangGovernanceResolvesToEngineSGLang"),
		mk("host", "internal/covmatrix/precision_test.go", "TestCPUReferencePrecisionsMatchSource"),
		mk("policy", "internal/policy/hotreload_test.go", "TestHotReloadSwapsLiveFloorWithoutRestart"),
	}}
}

func Load(root string) (Report, error) {
	r := Contract()
	for i := range r.Swaps {
		if err := checkWitness(root, r.Swaps[i].Fak); err != nil {
			return Report{}, fmt.Errorf("%s fak witness: %w", r.Swaps[i].ID, err)
		}
		for alt, w := range r.Swaps[i].Alternatives {
			if w.Path == "" || w.Test == "" {
				continue
			}
			if err := checkWitness(root, w); err != nil {
				return Report{}, fmt.Errorf("%s %s witness: %w", r.Swaps[i].ID, alt, err)
			}
		}
	}
	return r, nil
}

func checkWitness(root string, w Witness) error {
	b, err := os.ReadFile(filepath.Join(root, filepath.FromSlash(w.Path)))
	if err != nil {
		return err
	}
	needle := []byte("func " + w.Test + "(")
	if !contains(b, needle) {
		return fmt.Errorf("%s does not define %s", w.Path, w.Test)
	}
	return nil
}
func contains(b, sub []byte) bool {
	if len(sub) == 0 {
		return true
	}
	for i := 0; i+len(sub) <= len(b); i++ {
		ok := true
		for j := range sub {
			if b[i+j] != sub[j] {
				ok = false
				break
			}
		}
		if ok {
			return true
		}
	}
	return false
}

func Write(path string, r Report) error {
	sort.Slice(r.Swaps, func(i, j int) bool { return r.Swaps[i].ID < r.Swaps[j].ID })
	b, err := json.MarshalIndent(r, "", "  ")
	if err != nil {
		return err
	}
	b = append(b, '\n')
	return os.WriteFile(path, b, 0o644)
}
