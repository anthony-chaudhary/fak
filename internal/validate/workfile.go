package validate

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/windowgate"
)

const (
	publicModulePath  = "github.com/anthony-chaudhary/fak"
	privateModulePath = "github.com/anthony-chaudhary/fak-private"
)

// prepareValidateWorkfile rebinds an inherited native workspace to the extracted
// candidate while retaining its companions and directives. The legacy two-module
// synthesis remains the fallback when there is no inherited workspace or for WSL.
func prepareValidateWorkfile(ctx context.Context, sourceRoot, extractedRoot string, wsl bool) error {
	modulePath, err := modulePathAt(sourceRoot)
	if err != nil || modulePath == publicModulePath {
		return nil
	}
	if modulePath != privateModulePath && nearestWorkfile(sourceRoot) == "" {
		return nil
	}
	if workspaceFile := nearestWorkfile(sourceRoot); !wsl && workspaceFile != "" {
		return rebindValidateWorkfile(ctx, sourceRoot, extractedRoot, workspaceFile, modulePath)
	}
	publicRoot, err := locatePublicModuleRoot(ctx, sourceRoot)
	if err != nil {
		return err
	}
	if wsl {
		publicWSL, ok := wslMountPath(publicRoot)
		if !ok {
			return fmt.Errorf("map public module root %q into WSL", publicRoot)
		}
		script := "rm -f -- go.work; exec go work init . " + posixQuote(publicWSL)
		out, runErr := runValidateWSLCommandWithin(ctx, extractedRoot, "sh", "-c", script)
		if runErr != nil {
			return fmt.Errorf("synthesize private validation workspace in WSL: %s", validateCommandDetail(out, runErr))
		}
		return nil
	}
	if err := os.Remove(filepath.Join(extractedRoot, "go.work")); err != nil && !os.IsNotExist(err) {
		return fmt.Errorf("remove extracted go.work: %w", err)
	}
	cmd := exec.CommandContext(ctx, "go", "work", "init", ".", publicRoot)
	cmd.Dir = extractedRoot
	windowgate.ConfigureBackgroundCommand(cmd)
	if out, runErr := cmd.CombinedOutput(); runErr != nil {
		return fmt.Errorf("synthesize private validation workspace: %s", validateCommandDetail(out, runErr))
	}
	return nil
}

func locatePublicModuleRoot(ctx context.Context, sourceRoot string) (string, error) {
	if workspaceFile := nearestWorkfile(sourceRoot); workspaceFile != "" {
		publicRoot, err := publicModuleRootFromWorkfile(ctx, workspaceFile)
		if err != nil {
			return "", err
		}
		return publicRoot, nil
	}
	if _, file, _, ok := runtime.Caller(0); ok {
		candidate := filepath.Clean(filepath.Join(filepath.Dir(file), "..", ".."))
		if modulePath, err := modulePathAt(candidate); err == nil && modulePath == publicModulePath {
			return candidate, nil
		}
	}
	for dir := filepath.Clean(sourceRoot); ; dir = filepath.Dir(dir) {
		candidate := filepath.Join(filepath.Dir(dir), "fak")
		if modulePath, err := modulePathAt(candidate); err == nil && modulePath == publicModulePath {
			return candidate, nil
		}
		parent := filepath.Dir(dir)
		if parent == dir {
			break
		}
	}
	return "", fmt.Errorf("locate public module %s for private validation root %q", publicModulePath, sourceRoot)
}

func nearestWorkfile(root string) string {
	for dir := filepath.Clean(root); ; dir = filepath.Dir(dir) {
		candidate := filepath.Join(dir, "go.work")
		if info, err := os.Stat(candidate); err == nil && !info.IsDir() {
			return candidate
		}
		parent := filepath.Dir(dir)
		if parent == dir {
			return ""
		}
	}
}

func publicModuleRootFromWorkfile(ctx context.Context, workfile string) (string, error) {
	cmd := exec.CommandContext(ctx, "go", "work", "edit", "-json", workfile)
	windowgate.ConfigureBackgroundCommand(cmd)
	out, err := cmd.CombinedOutput()
	if err != nil {
		return "", fmt.Errorf("read source workspace %q: %s", workfile, validateCommandDetail(out, err))
	}
	var workspace struct {
		Use []struct {
			DiskPath string
		}
	}
	if err := json.Unmarshal(out, &workspace); err != nil {
		return "", fmt.Errorf("decode source workspace %q: %w", workfile, err)
	}
	workdir := filepath.Dir(workfile)
	for _, use := range workspace.Use {
		candidate := use.DiskPath
		if !filepath.IsAbs(candidate) {
			candidate = filepath.Join(workdir, candidate)
		}
		candidate = filepath.Clean(candidate)
		if modulePath, err := modulePathAt(candidate); err == nil && modulePath == publicModulePath {
			return candidate, nil
		}
	}
	return "", fmt.Errorf("source workspace %q does not use public module %s", workfile, publicModulePath)
}

func modulePathAt(root string) (string, error) {
	data, err := os.ReadFile(filepath.Join(root, "go.mod"))
	if err != nil {
		return "", err
	}
	for _, line := range strings.Split(string(data), "\n") {
		fields := strings.Fields(line)
		if len(fields) == 2 && fields[0] == "module" {
			return fields[1], nil
		}
	}
	return "", fmt.Errorf("go.mod at %q has no module directive", root)
}

type validateWorkfile struct {
	Use     []struct{ DiskPath string }
	Replace []struct {
		Old, New struct{ Path, Version string }
	}
}

// rebindValidateWorkfile edits a private copy with Go's own syntax-preserving
// editor. Go may normalize block ordering; no workspace directive is rebuilt
// from the JSON inspection, and versioned replacements are left untouched.
func rebindValidateWorkfile(ctx context.Context, sourceRoot, extractedRoot, workfile, modulePath string) error {
	workfile, err := filepath.Abs(workfile)
	if err != nil {
		return err
	}
	extractedRoot, err = filepath.Abs(extractedRoot)
	if err != nil {
		return err
	}
	workspaceInfo, err := os.Stat(filepath.Dir(workfile))
	if err != nil {
		return err
	}
	sourceInfo, err := os.Stat(sourceRoot)
	if err != nil {
		return err
	}
	extractedInfo, err := os.Stat(extractedRoot)
	if err != nil {
		return err
	}
	if os.SameFile(sourceInfo, extractedInfo) || os.SameFile(workspaceInfo, extractedInfo) {
		return fmt.Errorf("validation workspace requires a separate extracted root")
	}
	if extractedModule, err := modulePathAt(extractedRoot); err != nil || extractedModule != modulePath {
		return fmt.Errorf("extracted validation module identity does not match %q: module=%q error=%v", modulePath, extractedModule, err)
	}
	original, err := os.ReadFile(workfile)
	if err != nil {
		return err
	}
	cmd := exec.CommandContext(ctx, "go", "work", "edit", "-json", workfile)
	windowgate.ConfigureBackgroundCommand(cmd)
	out, err := cmd.CombinedOutput()
	if err != nil {
		return fmt.Errorf("inspect validation workspace %q: %s", workfile, validateCommandDetail(out, err))
	}
	var workspace validateWorkfile
	if err := json.Unmarshal(out, &workspace); err != nil {
		return fmt.Errorf("decode validation workspace %q: %w", workfile, err)
	}
	args, err := validateWorkfileEdits(workspace, filepath.Dir(workfile), sourceRoot, extractedRoot, modulePath)
	if err != nil {
		return err
	}
	current, err := os.ReadFile(workfile)
	if err != nil {
		return err
	}
	if !bytes.Equal(original, current) {
		return fmt.Errorf("source workspace changed during validation preparation")
	}
	privateWorkfile := filepath.Join(extractedRoot, "go.work")
	// Remove any extracted symlink before writing; never follow it into a source
	// workspace. All identity and edit-argument checks above precede this write.
	if err := os.Remove(privateWorkfile); err != nil && !os.IsNotExist(err) {
		return fmt.Errorf("remove extracted go.work: %w", err)
	}
	f, err := os.OpenFile(privateWorkfile, os.O_WRONLY|os.O_CREATE|os.O_EXCL, 0o600)
	if err != nil {
		return err
	}
	_, writeErr := f.Write(original)
	closeErr := f.Close()
	if writeErr != nil {
		return writeErr
	}
	if closeErr != nil {
		return closeErr
	}
	args = append(args, privateWorkfile)
	cmd = exec.CommandContext(ctx, "go", args...)
	cmd.Dir = extractedRoot
	windowgate.ConfigureBackgroundCommand(cmd)
	if out, err := cmd.CombinedOutput(); err != nil {
		return fmt.Errorf("rebind validation workspace: %s", validateCommandDetail(out, err))
	}
	return nil
}

func validateWorkfileEdits(workspace validateWorkfile, workdir, sourceRoot, extractedRoot, modulePath string) ([]string, error) {
	anchor := func(path string) string {
		if !filepath.IsAbs(path) {
			path = filepath.Join(workdir, path)
		}
		return filepath.Clean(path)
	}
	sourceInfo, err := os.Stat(sourceRoot)
	if err != nil {
		return nil, err
	}
	args := []string{"work", "edit"}
	var uses []string
	matches, publicMatches := 0, 0
	for _, use := range workspace.Use {
		root := anchor(use.DiskPath)
		identity, err := modulePathAt(root)
		if err != nil {
			return nil, fmt.Errorf("read workspace module %q: %w", root, err)
		}
		if identity == publicModulePath {
			publicMatches++
		}
		if identity == modulePath {
			matches++
			info, err := os.Stat(root)
			if err != nil {
				return nil, err
			}
			if !os.SameFile(sourceInfo, info) {
				return nil, fmt.Errorf("workspace module %q resolves to %q instead of source root %q", modulePath, root, sourceRoot)
			}
			root = extractedRoot
		}
		args = append(args, "-dropuse="+use.DiskPath)
		uses = append(uses, "-use="+root)
	}
	if matches != 1 || publicMatches != 1 {
		return nil, fmt.Errorf("validation workspace requires exactly one source module %q and one public module %q; found %d and %d", modulePath, publicModulePath, matches, publicMatches)
	}
	args = append(args, uses...)
	for _, replacement := range workspace.Replace {
		if replacement.New.Version != "" || filepath.IsAbs(replacement.New.Path) {
			continue
		}
		path := anchor(replacement.New.Path)
		// The Go editor's flag syntax reserves @ for versions and trims the
		// replacement value. Refuse unrepresentable paths rather than retargeting.
		if strings.Contains(path, "@") || strings.TrimSpace(path) != path {
			return nil, fmt.Errorf("local workspace replacement path %q cannot be safely expressed to go work edit", path)
		}
		old := replacement.Old.Path
		if replacement.Old.Version != "" {
			old += "@" + replacement.Old.Version
		}
		args = append(args, "-replace="+old+"="+path)
	}
	return args, nil
}
