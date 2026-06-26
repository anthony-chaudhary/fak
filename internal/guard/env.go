package guard

import (
	"os"

	"github.com/anthony-chaudhary/fak/internal/secretload"
)

func landlockChildEnv(extra ...string) []string {
	return secretload.SandboxEnv(os.Environ(), extra...)
}
