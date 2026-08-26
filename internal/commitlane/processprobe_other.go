//go:build !windows

package commitlane

import (
	"bytes"
	"context"
	"fmt"
	"os/exec"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/windowgate"
)

func runWindowsProcessJSON(ctx context.Context, name string, args ...string) ([]byte, error) {
	cmd := exec.CommandContext(ctx, name, args...)
	windowgate.ConfigureBackgroundCommand(cmd)
	var stderr bytes.Buffer
	cmd.Stderr = &stderr
	out, err := cmd.Output()
	if err != nil {
		detail := strings.TrimSpace(stderr.String())
		if detail == "" {
			detail = err.Error()
		}
		return nil, fmt.Errorf("%s: %s", name, detail)
	}
	return out, nil
}
