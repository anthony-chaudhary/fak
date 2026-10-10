package validate

import (
	"context"
	"errors"
	"io"
	"os"
	"runtime"
	"sort"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/buildwitness"
	"github.com/anthony-chaudhary/fak/internal/windowgate"
)

// crossplatform.go — the two commit-time halves of the 2026-10-10 single-OS-compile floor.
// Commit f9d4cca4 added internal/ctxmmu/closed_windows.go; Go read the suffix as GOOS=windows, so
// every Linux build of internal/ctxmmu broke while the Windows checkout that landed it compiled.
//
//   - platform_filename refuses an owned .go file whose basename implies a GOOS/GOARCH constraint
//     its source never states with a //go:build line (buildwitness.CheckPlatformFilenames).
//   - cross_build compiles the changed packages for the OTHER operating system than the one the
//     build phase just used, so a symbol that only exists on one OS cannot land green.

// validateBuildGOOS is the GOOS the build/vet phases compiled for: a WSL workspace builds Linux
// even on a Windows host.
func validateBuildGOOS(hostGOOS string, wslWorkspace bool) string {
	if wslWorkspace {
		return "linux"
	}
	return hostGOOS
}

// validateCrossGOOS names the operating systems the cross_build phase compiles for: Linux and
// Windows are the two the fleet runs, so each covers the other; any other host checks both.
func validateCrossGOOS(buildGOOS string) []string {
	switch buildGOOS {
	case "linux":
		return []string{"windows"}
	case "windows":
		return []string{"linux"}
	default:
		return []string{"linux", "windows"}
	}
}

func runValidatePlatformPhases(ctx context.Context, stdout io.Writer, res *validateResult, recorder *validateRecorder, sourceRoot, dir string, paths []string, fileToPkg map[string]string, wslWorkspace, asJSON bool) (int, bool) {
	phase := recorder.start("platform_filename")
	bad, err := buildwitness.CheckPlatformFilenames(sourceRoot, paths)
	if err != nil {
		recordValidateFailure(res, phase, "platform_filename", err.Error(), err)
	} else if len(bad) > 0 {
		files := make([]string, 0, len(bad))
		details := make([]string, 0, len(bad))
		for _, v := range bad {
			files = append(files, v.Path)
			details = append(details, v.String())
		}
		phase.finishAs("failed", "filename implies a platform the source does not declare")
		res.OK = false
		res.Failures = append(res.Failures, ciPreflightFailure{Step: "platform_filename", Detail: strings.Join(details, "\n"), Files: files})
	} else {
		phase.finish(nil)
	}

	buildable := validateBuildablePackages(fileToPkg)
	var changed []string
	for _, pkg := range res.Tested {
		if buildable[pkg] {
			changed = append(changed, pkg)
		}
	}
	if len(changed) == 0 {
		recorder.skip("cross_build", "no changed package with non-test Go files")
		return 0, false
	}
	targets := packagePatternsForRoot(dir, changed, fileToPkg)
	phase = recorder.start("cross_build")
	return runValidateCheckPhase(stdout, res, recorder, phase, "cross_build", errors.New("changed package does not compile for another GOOS"), asJSON, func() (string, bool) {
		for _, goos := range validateCrossGOOS(validateBuildGOOS(runtime.GOOS, wslWorkspace)) {
			if detail, ok := validateCrossBuildWithin(ctx, dir, wslWorkspace, goos, targets); !ok {
				return "GOOS=" + goos + ": " + detail, false
			}
		}
		return "", true
	})
}

// validateCrossBuildWithin compiles targets for goos. A package whose every file is excluded on
// goos (a deliberately Windows-only package, say) is not a failure, and naming it would make go
// build refuse at load time and hide real compile errors in the rest, so go list first narrows
// targets to the packages that have Go files on goos.
func validateCrossBuildWithin(ctx context.Context, dir string, wsl bool, goos string, targets []string) (string, bool) {
	env := []string{"GOOS=" + goos, "CGO_ENABLED=0"}
	listArgs := append([]string{"list", "-e", "-f", "{{if .GoFiles}}{{.ImportPath}}{{end}}"}, targets...)
	out, ok := validateRunGoEnvWithin(ctx, dir, wsl, env, listArgs...)
	if !ok {
		return out, false
	}
	var pkgs []string
	for _, line := range strings.Split(out, "\n") {
		// An import path holds no space or colon; anything else is toolchain chatter.
		if line = strings.TrimSpace(line); line != "" && !strings.ContainsAny(line, " \t:") {
			pkgs = append(pkgs, line)
		}
	}
	if len(pkgs) == 0 {
		return "", true
	}
	sort.Strings(pkgs)
	out, ok = validateRunGoEnvWithin(ctx, dir, wsl, env, validateGoCheckArgs("build", pkgs)...)
	return out, ok
}

// validateRunGoEnvWithin runs `go args...` with extra environment in root, natively or through
// WSL, returning combined output and success.
func validateRunGoEnvWithin(ctx context.Context, root string, wsl bool, env []string, args ...string) (string, bool) {
	if wsl {
		argv := append([]string{"env"}, env...)
		argv = append(argv, "go")
		out, err := runValidateWSLCommandWithin(ctx, root, append(argv, args...)...)
		return strings.TrimSpace(string(out)), err == nil
	}
	cmd := windowgate.CommandContext(ctx, "go", args...)
	cmd.Dir = root
	cmd.Env = append(os.Environ(), env...)
	windowgate.ConfigureBackgroundCommand(cmd)
	out, err := cmd.CombinedOutput()
	return strings.TrimSpace(string(out)), err == nil
}
