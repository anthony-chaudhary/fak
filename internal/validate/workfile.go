package validate

import (
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

// prepareValidateWorkfile replaces the extracted private workspace's relative
// sibling reference with the public module root that supplied this validator.
// The committed private go.work contains "use . ../fak"; after extraction into
// a temporary directory that sibling does not exist.
func prepareValidateWorkfile(ctx context.Context, sourceRoot, extractedRoot string, wsl bool) error {
	modulePath, err := modulePathAt(sourceRoot)
	if err != nil || modulePath == publicModulePath {
		return nil
	}
	if modulePath != privateModulePath && nearestWorkfile(sourceRoot) == "" {
		return nil
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
