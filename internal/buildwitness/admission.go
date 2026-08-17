package buildwitness

import (
	"encoding/json"
	"os"
	"path/filepath"

	"github.com/anthony-chaudhary/fak/internal/workdelivery"
)

// AdmissionOverlay turns work-delivery declarations into the overlay used by the
// structural build witness. Excluded source is recorded but not active; admitted
// source remains visible. Declaration errors fail closed.
type AdmissionOverlay struct {
	CompileSet workdelivery.CompileSet
	Overlay    admissionOverlay
}

type admissionOverlay struct {
	Replace map[string]string `json:"Replace"`
}

func BuildAdmissionOverlay(root string, declarationPaths ...string) (AdmissionOverlay, error) {
	set, err := workdelivery.LoadCompileSet(declarationPaths...)
	if err != nil {
		return AdmissionOverlay{}, err
	}
	empty := filepath.Join(filepath.Clean(root), ".fak-empty-go-overlay")
	replace := make(map[string]string, len(set.Excluded))
	for _, path := range set.Excluded {
		replace[filepath.Join(filepath.Clean(root), filepath.FromSlash(path))] = empty
	}
	return AdmissionOverlay{CompileSet: set, Overlay: admissionOverlay{Replace: replace}}, nil
}

func WriteAdmissionOverlay(path string, plan AdmissionOverlay) error {
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		return err
	}
	data, err := json.MarshalIndent(plan.Overlay, "", "  ")
	if err != nil {
		return err
	}
	data = append(data, '\n')
	return os.WriteFile(path, data, 0o644)
}
