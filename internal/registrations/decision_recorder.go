package registrations

import (
	"github.com/anthony-chaudhary/fak/internal/gitgate"
	"github.com/anthony-chaudhary/fak/internal/witness"
)

func init() {
	// Attach the decisions side-ref sink to gitgate in the default fak binary.
	gitgate.Default.SetRecorder(witness.NewRecorder())
}
