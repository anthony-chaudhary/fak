package compute_test

import (
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"testing"
)

// toPOSIXPath renders a Windows host path in the form the resolved bash expects.
// Git-for-Windows (MSYS) bash translates "C:\dir" as "/c/dir" and splits PATH on
// ':' — a retained "C:/dir" is read as two entries ("C" and "/dir") and every stub
// lookup inside the script fails with exit 127. WSL bash instead mounts the same
// path as "/mnt/c/dir". A non-Windows host only needs the slash normalization.
func toPOSIXPath(path, bash string) string {
	slash := filepath.ToSlash(path)
	if runtime.GOOS != "windows" {
		return slash
	}
	volume := filepath.VolumeName(path)
	if volume == "" {
		return slash
	}
	drive := strings.ToLower(strings.TrimSuffix(volume, ":"))
	rest := strings.TrimPrefix(slash, volume)
	if strings.Contains(strings.ToLower(bash), `\windows\system32\bash.exe`) {
		return "/mnt/" + drive + rest // WSL bash.
	}
	return "/" + drive + rest // Git-for-Windows (MSYS) bash.
}

// TestToPOSIXPathTranslatesWindowsDrive pins the drive-letter translation that
// makes the stub PATH and script path resolvable under both WSL and Git-for-Windows
// bash. Git bash MUST get a "C:/dir" -> "/c/dir" form; a retained drive colon makes
// bash split the PATH at the colon and lose the stub, which is the exit-127 defect.
func TestToPOSIXPathTranslatesWindowsDrive(t *testing.T) {
	if runtime.GOOS != "windows" {
		t.Skip("drive-letter translation is a Windows-host concern")
	}
	const winPath = `C:\tmp\bin`
	if got, want := toPOSIXPath(winPath, `C:\Program Files\Git\bin\bash.exe`), "/c/tmp/bin"; got != want {
		t.Fatalf("MSYS translation = %q, want %q", got, want)
	}
	if got, want := toPOSIXPath(winPath, `C:\Windows\System32\bash.exe`), "/mnt/c/tmp/bin"; got != want {
		t.Fatalf("WSL translation = %q, want %q", got, want)
	}
}

func TestGPUAcceptanceScriptsResolveCleanEnvironmentToolkit(t *testing.T) {
	matches, err := filepath.Glob(filepath.Join("..", "..", "tools", "run_*_acceptance_on_gpu.sh"))
	if err != nil {
		t.Fatal(err)
	}
	if len(matches) == 0 {
		t.Fatal("no GPU acceptance scripts found")
	}
	for _, path := range matches {
		data, err := os.ReadFile(path)
		if err != nil {
			t.Fatal(err)
		}
		script := string(data)
		if strings.Contains(script, "${CUDA_HOME:-$HOME/") {
			t.Errorf("%s expands unset HOME under `set -u`", filepath.Base(path))
		}
		for _, cache := range []string{"GOCACHE", "GOMODCACHE", "GOPATH"} {
			if !strings.Contains(script, "export "+cache+"=") {
				t.Errorf("%s does not provision %s for a HOME-less clean session", filepath.Base(path), cache)
			}
		}
		if !strings.Contains(script, "[ -x /usr/local/cuda/bin/nvcc ]") {
			t.Errorf("%s does not probe the standard CUDA toolkit outside PATH", filepath.Base(path))
		}
	}
}
