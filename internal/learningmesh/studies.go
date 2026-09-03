package learningmesh

import (
	"fmt"
	"sort"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/studylink"
)

// comparatorProviders maps provider names to canonical tokens (#9887).
var comparatorProviders = map[string]string{
	"llama.cpp": "llama.cpp",
	"llamacpp":  "llama.cpp",
	"vllm":      "vllm",
	"dynamo":    "dynamo",
	"sglang":    "sglang",
	"mlx":       "mlx",
	"mlx-lm":    "mlx",
}

// StudyMechanism describes a borrowed mechanism identified in an upstream study.
type StudyMechanism struct {
	ID                 string      `json:"id"`
	Name               string      `json:"name"`
	Rule               string      `json:"rule"`
	Hardware           string      `json:"hardware"` // e.g. "nvidia", "amd", "apple"
	Backend            string      `json:"backend"`  // e.g. "cuda", "rocm", "metal"
	Model              string      `json:"model,omitempty"`
	Workload           string      `json:"workload,omitempty"`
	DefaultDisposition Disposition `json:"default_disposition,omitempty"`
}

// ProviderStudyInput binds an upstream engine study to the learning-mesh input schema (#9887).
type ProviderStudyInput struct {
	Provider     string           `json:"provider"` // "llama.cpp", "vllm", "dynamo", "sglang", "mlx"
	StudyDocPath string           `json:"study_doc_path"`
	Revision     string           `json:"revision"`
	RecordDigest string           `json:"record_digest"`
	Mechanisms   []StudyMechanism `json:"mechanisms"`
}

// NormalizeProvider maps input provider strings into canonical framework tokens.
func NormalizeProvider(provider string) (string, error) {
	norm := strings.ToLower(strings.TrimSpace(provider))
	if canon, ok := comparatorProviders[norm]; ok {
		return canon, nil
	}
	return "", fmt.Errorf("learningmesh: unsupported comparator provider %q; must be one of: llama.cpp, vllm, dynamo, sglang, mlx", provider)
}

// IngestStudies compiles provider study inputs into a validated, provider-neutral Ledger (#9887).
// Comparator runtimes remain explicit benchmark, parity, interop, or borrowing sources;
// product execution stays strictly fak-native.
func IngestStudies(studies []ProviderStudyInput, targets []Envelope) (Ledger, error) {
	if len(studies) == 0 {
		return Ledger{}, fmt.Errorf("learningmesh: at least one study input is required")
	}
	if len(targets) == 0 {
		return Ledger{}, fmt.Errorf("learningmesh: at least one target is required")
	}

	byID := make(map[string]Mechanism)
	for _, study := range studies {
		canonProvider, err := NormalizeProvider(study.Provider)
		if err != nil {
			return Ledger{}, err
		}
		path := strings.TrimSpace(study.StudyDocPath)
		if path == "" {
			return Ledger{}, fmt.Errorf("learningmesh: study doc path is required for provider %s", canonProvider)
		}
		rev := strings.TrimSpace(study.Revision)
		if rev == "" {
			return Ledger{}, fmt.Errorf("learningmesh: revision is required for study %s", path)
		}

		for _, sm := range study.Mechanisms {
			id := strings.TrimSpace(sm.ID)
			if id == "" {
				return Ledger{}, fmt.Errorf("learningmesh: mechanism ID is required in study %s", path)
			}
			if _, exists := byID[id]; exists {
				return Ledger{}, fmt.Errorf("learningmesh: duplicate mechanism ID %q across studies", id)
			}

			hardware := strings.ToLower(strings.TrimSpace(sm.Hardware))
			if hardware == "" {
				hardware = "generic"
			}
			backend := strings.ToLower(strings.TrimSpace(sm.Backend))

			disp := sm.DefaultDisposition
			if disp == "" {
				disp = Adapt
			}

			mech := Mechanism{
				ID:        id,
				Mechanism: sm.Name,
				Rule:      sm.Rule,
				Source: Envelope{
					ID:        fmt.Sprintf("%s-%s", canonProvider, id),
					Hardware:  hardware,
					Backend:   backend,
					Framework: canonProvider,
					Engine:    canonProvider,
					Model:     sm.Model,
					Workload:  sm.Workload,
					Purpose:   "study",
					Role:      "baseline",
				},
				Provenance: Provenance{
					Artifacts: []studylink.Artifact{
						{
							Kind:         "repo-doc",
							ID:           path,
							Revision:     receiptRevision(rev),
							Path:         path,
							Exact:        false,
							RecordDigest: study.RecordDigest,
						},
					},
				},
				DefaultDisposition: disp,
				Rules: []Rule{
					{
						Target: Selector{
							Engine: "fak-native",
						},
						Disposition: disp,
						Reason:      fmt.Sprintf("borrow provider-neutral %s mechanism from %s into fak-native execution", sm.Name, canonProvider),
					},
				},
			}
			byID[id] = mech
		}
	}

	mechanisms := make([]Mechanism, 0, len(byID))
	for _, m := range byID {
		mechanisms = append(mechanisms, m)
	}
	sort.Slice(mechanisms, func(i, j int) bool { return mechanisms[i].ID < mechanisms[j].ID })

	return Ledger{
		Schema:     InputSchema,
		Mechanisms: mechanisms,
		Targets:    targets,
	}, nil
}
