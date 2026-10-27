package validate

import (
	"context"
	"fmt"
	"os"
	"os/exec"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/windowgate"
)

func validateBuildImportersWithin(ctx context.Context, dir string, wsl bool, targets []string) (string, bool) {
	return validateBuildImporters(ctx, targets, validateBuildNullOutput(os.DevNull, wsl), func(ctx context.Context, args ...string) (string, bool) {
		if args[0] == "list" || args[0] == "env" {
			// Metadata must not contain successful module-download chatter from stderr.
			cmd := validateBuildImporterListCommand(ctx, dir, wsl, args)
			out, err := cmd.Output()
			if err != nil {
				if exitErr, ok := err.(*exec.ExitError); ok {
					out = append(out, exitErr.Stderr...)
				}
				return validateCommandDetail(out, err), false
			}
			return string(out), true
		}
		return validateRunGoEnvWithin(ctx, dir, wsl, nil, args...)
	})
}

func validateBuildNullOutput(hostNull string, wsl bool) string {
	if wsl {
		return "/dev/null"
	}
	return hostNull
}

func validateBuildImporterListCommand(ctx context.Context, dir string, wsl bool, args []string) *exec.Cmd {
	var cmd *exec.Cmd
	if wsl {
		command := "set -euo pipefail; cd " + posixQuote(dir) + "; exec go"
		for _, arg := range args {
			command += " " + posixQuote(arg)
		}
		cmd = windowgate.CommandContext(ctx, "wsl.exe", "-e", "bash", "-lc", command)
	} else {
		cmd = windowgate.CommandContext(ctx, "go", args...)
		cmd.Dir = dir
	}
	windowgate.ConfigureBackgroundCommand(cmd)
	return cmd
}

// validateBuildImporters keeps the selected build closure intact, including every
// command's link step. A single multi-command go build retains all linked binaries
// in its work directory until the entire invocation finishes. Build libraries
// together, then link commands one at a time so Go can release each work directory.
// All invocations share the original validation context and its remaining budget.
func validateBuildImporters(ctx context.Context, targets []string, null string, run func(context.Context, ...string) (string, bool)) (string, bool) {
	if len(targets) == 0 {
		return "", true
	}
	invoke := func(args ...string) (string, bool) {
		if err := ctx.Err(); err != nil {
			return err.Error(), false
		}
		return run(ctx, args...)
	}
	// Unqualified gcflags/ldflags and non-default build modes can make the
	// command-line package set significant; -work also disables Go's cleanup.
	// Preserve the original invocation for unqualified flag configurations.
	flags, ok := invoke("env", "GOFLAGS")
	if !ok {
		return "read importer build flags: " + flags, false
	}
	for _, flag := range strings.Fields(flags) {
		if flag != "-p=1" {
			return invoke(validateGoCheckArgs("build", targets)...)
		}
	}
	// Resolve package kinds once, in the same workspace and runner as the build.
	// Do not infer commands from directory names: main packages can live anywhere.
	// Match the build VCS policy: metadata must not require VCS stamping that
	// the actual build explicitly disables in disposable validation workspaces.
	const format = "{{.Name}} {{.ImportPath}}{{range .Match}} {{.}}{{end}}"
	out, ok := invoke(append([]string{"list", "-trimpath", "-buildvcs=false", "-f", format}, targets...)...)
	if !ok {
		return "select importer build packages: " + out, false
	}
	var libraries, commands []string
	seen := make(map[string]bool)
	matched := make(map[string]bool, len(targets))
	for _, target := range targets {
		matched[target] = false
	}
	for _, line := range strings.Split(strings.TrimSpace(out), "\n") {
		fields := strings.Fields(line)
		if len(fields) < 3 || seen[fields[1]] {
			return fmt.Sprintf("select importer build packages: invalid metadata %q", line), false
		}
		seen[fields[1]] = true
		for _, target := range fields[2:] {
			previous, expected := matched[target]
			if !expected || previous {
				return fmt.Sprintf("select importer build packages: unexpected or repeated target %q", target), false
			}
			matched[target] = true
		}
		if fields[0] == "main" {
			commands = append(commands, fields[1])
		} else {
			libraries = append(libraries, fields[1])
		}
	}
	for _, target := range targets {
		if !matched[target] {
			return "select importer build packages: missing target " + target, false
		}
	}
	build := func(packages []string) (string, bool) {
		// Explicit null output prevents a singleton command from installing its
		// executable in the validation source root. It does not skip linking.
		args := validateGoCheckArgs("build", nil)
		args = append(args, "-o", null)
		args = append(args, packages...)
		out, ok := invoke(args...)
		if !ok {
			return "build " + strings.Join(packages, " ") + ": " + out, false
		}
		return "", true
	}
	if len(libraries) > 0 {
		if out, ok := build(libraries); !ok {
			return out, false
		}
	}
	for _, pkg := range commands {
		if out, ok := build([]string{pkg}); !ok {
			return out, false
		}
	}
	return "", true
}
