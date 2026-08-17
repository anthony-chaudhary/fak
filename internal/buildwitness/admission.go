package buildwitness

import (
	"path/filepath"

	"github.com/anthony-chaudhary/fak/internal/buildoverlay"
	"github.com/anthony-chaudhary/fak/internal/workdelivery"
)

// AdmissionOverlay turns work-delivery declarations into the overlay used by the
// structural build witness. Excluded source is recorded but not active; admitted
// source remains visible. Declaration errors fail closed.
type AdmissionOverlay struct {
	CompileSet workdelivery.CompileSet
	Overlay    buildoverlay.Overlay
}

func BuildAdmissionOverlay(root string, declarationPaths ...string) (AdmissionOverlay, error) {
	set, err := workdelivery.LoadCompileSet(declarationPaths...)
	if err != nil {
		return AdmissionOverlay{}, err
	}
	return AdmissionOverlay{CompileSet: set, Overlay: buildoverlay.Build(filepath.Clean(root), set.Excluded)}, nil
}
