package validate

import (
	"context"
	"errors"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/windowgate"
)

// validateWSLOverlayTreeDir and validateWSLOverlayScriptName lay out the
// Windows-side overlay stage: the owned files are copied under tree/, and the
// copy/delete script sits beside it so it can never collide with an owned path.
const (
	validateWSLOverlayTreeDir    = "tree"
	validateWSLOverlayScriptName = "overlay.sh"
)

// overlayMinePathsWSLWithin copies the owned paths into the WSL workspace.
//
// The copy/delete commands grow with the owned path set, so they travel in a
// script file run from the stage directory (`wsl.exe --cd <stage> bash
// ./overlay.sh`) rather than as one `bash -lc` argument. The wsl.exe command
// line therefore stays the same size however many paths are owned: a 524-path
// stale-base land delta once composed a ~145K-char argument, which Windows
// CreateProcess refuses (limit 32,767) before WSL starts, and the failure
// surfaced with an empty message.
func overlayMinePathsWSLWithin(ctx context.Context, srcRoot, wslRoot string, paths []string, checked func(string)) error {
	stage, err := os.MkdirTemp("", "fak-validate-overlay-*")
	if err != nil {
		return err
	}
	defer os.RemoveAll(stage)
	tree := filepath.Join(stage, validateWSLOverlayTreeDir)
	if err := overlayMinePathsWithin(ctx, srcRoot, tree, paths, nil); err != nil {
		return err
	}
	script, err := validateWSLOverlayScript(tree, wslRoot, paths)
	if err != nil {
		return err
	}
	if err := os.WriteFile(filepath.Join(stage, validateWSLOverlayScriptName), []byte(script), 0o600); err != nil {
		return err
	}
	cmd := windowgate.CommandContext(ctx, "wsl.exe", "--cd", stage, "bash", "./"+validateWSLOverlayScriptName)
	windowgate.ConfigureBackgroundCommand(cmd)
	out, err := cmd.CombinedOutput()
	if err != nil {
		if ctx.Err() != nil {
			return ctx.Err()
		}
		return validateWSLOverlayFailure(err, out, len(paths), len(script), validateCommandLineChars(cmd.Args))
	}
	if checked != nil {
		for _, rel := range paths {
			checked(rel)
		}
	}
	return nil
}

// validateWSLOverlayScript renders the bash script that applies the staged
// overlay to wslRoot. It runs with the stage directory as its working
// directory, so staged sources are addressed relative to it and no WSL
// translation of the Windows stage path is needed. An owned path absent from
// the staged tree is a deletion.
func validateWSLOverlayScript(tree, wslRoot string, paths []string) (string, error) {
	root := strings.TrimSuffix(wslRoot, "/")
	var b strings.Builder
	b.WriteString("set -euo pipefail\n")
	for _, rel := range paths {
		slash := filepath.ToSlash(rel)
		dst := root + "/" + slash
		_, statErr := os.Stat(filepath.Join(tree, filepath.FromSlash(rel)))
		switch {
		case statErr == nil:
			parent := root + "/" + filepath.ToSlash(filepath.Dir(filepath.FromSlash(rel)))
			src := validateWSLOverlayTreeDir + "/" + slash
			fmt.Fprintf(&b, "mkdir -p -- %s\ncp -- %s %s\n", posixQuote(parent), posixQuote(src), posixQuote(dst))
		case os.IsNotExist(statErr):
			fmt.Fprintf(&b, "rm -rf -- %s\n", posixQuote(dst))
		default:
			return "", statErr
		}
	}
	return b.String(), nil
}

// validateWSLOverlayFailure names a failed overlay so the error is never empty:
// the child's exit code (-1 when wsl.exe never started), the owned path count,
// the script size, and the wsl.exe command-line length, then the combined
// output, or a placeholder when there was none.
func validateWSLOverlayFailure(runErr error, out []byte, pathCount, scriptBytes, commandChars int) error {
	exitCode := -1
	var exitErr *exec.ExitError
	if errors.As(runErr, &exitErr) {
		exitCode = exitErr.ExitCode()
	}
	detail := strings.TrimSpace(string(out))
	if detail == "" {
		detail = "no output from wsl.exe"
	}
	return fmt.Errorf("overlay owned paths in WSL: %v (exit_code=%d paths=%d script_bytes=%d command_chars=%d): %s",
		runErr, exitCode, pathCount, scriptBytes, commandChars, detail)
}

// validateCommandLineChars approximates the length of the command line Windows
// receives for argv (arguments joined by single spaces, before quoting).
func validateCommandLineChars(argv []string) int {
	return len(strings.Join(argv, " "))
}
