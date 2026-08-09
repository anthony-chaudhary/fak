// Package testenv provides the credential-free process boundary used by the
// repository test entry point.
package testenv

import (
	"fmt"
	"os"
	"os/exec"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/envconfiglint"
)

// WithoutCredentials returns env with every credential-shaped variable removed.
// envconfiglint.IsSecretName is the repository-wide source of truth, so a newly
// introduced provider key/token/secret/password name is fenced automatically.
func WithoutCredentials(env []string) []string {
	clean := make([]string, 0, len(env))
	for _, entry := range env {
		name, _, ok := strings.Cut(entry, "=")
		if ok && envconfiglint.IsSecretName(name) {
			continue
		}
		clean = append(clean, entry)
	}
	return clean
}

// Run executes argv with inherited non-credential environment only.
func Run(argv []string) error {
	if len(argv) == 0 {
		return fmt.Errorf("testenv: command is required")
	}
	cmd := exec.Command(argv[0], argv[1:]...)
	cmd.Env = WithoutCredentials(os.Environ())
	cmd.Stdin = os.Stdin
	cmd.Stdout = os.Stdout
	cmd.Stderr = os.Stderr
	if err := cmd.Run(); err != nil {
		return fmt.Errorf("testenv: %w", err)
	}
	return nil
}
