package validate_test

import (
	"bytes"
	"encoding/json"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/validate"
)

// TestRunRefusesImplicitPlatformFilename replays commit f9d4cca4 end to end: an owned
// closed_windows.go (meaning "closed windows", not GOOS=windows) defines a helper that an
// unsuffixed file calls. Whatever the host, the build compiles for one OS and the delta must be
// refused: platform_filename names the file, and on a Windows host — where the build phase is
// green, exactly as it was for the incident — cross_build is red for Linux.
func TestRunRefusesImplicitPlatformFilename(t *testing.T) {
	if testing.Short() {
		t.Skip("integration test; skipped under -short")
	}
	root := t.TempDir()
	commitFixture(t, root, map[string]string{
		"go.mod":         "module fixture.test/xplat\n\ngo 1.26\n",
		"ctxmmu/base.go": "package ctxmmu\n\nfunc Base() int { return 1 }\n",
	})
	writeFixtureFile(t, root+"/ctxmmu/compactor.go", "package ctxmmu\n\nfunc Compact() int { return fileWindows() }\n")
	writeFixtureFile(t, root+"/ctxmmu/closed_windows.go", "package ctxmmu\n\nfunc fileWindows() int { return 2 }\n")

	got := runPlatformValidate(t, root, "ctxmmu/compactor.go", "ctxmmu/closed_windows.go")
	if got.OK {
		t.Fatalf("implicit closed_windows.go delta validated green: %+v", got)
	}
	if status, _ := phaseStatus(got, "platform_filename"); status != "failed" {
		t.Fatalf("platform_filename status=%q, want failed; phases=%+v", status, got.Phases)
	}
	build, _ := phaseStatus(got, "build")
	cross, _ := phaseStatus(got, "cross_build")
	if build == "ok" && cross != "failed" {
		t.Fatalf("build compiled for the host OS but cross_build=%q did not catch the other OS; phases=%+v", cross, got.Phases)
	}
}

// TestRunAdmitsExplicitPlatformPair is the passing half: the same helper split into a build-tagged
// _windows.go and a //go:build !windows sibling compiles on every OS and states its platform.
func TestRunAdmitsExplicitPlatformPair(t *testing.T) {
	if testing.Short() {
		t.Skip("integration test; skipped under -short")
	}
	root := t.TempDir()
	commitFixture(t, root, map[string]string{
		"go.mod":         "module fixture.test/xplat\n\ngo 1.26\n",
		"ctxmmu/base.go": "package ctxmmu\n\nfunc Base() int { return 1 }\n",
	})
	writeFixtureFile(t, root+"/ctxmmu/compactor.go", "package ctxmmu\n\nfunc Compact() int { return fileWindows() }\n")
	writeFixtureFile(t, root+"/ctxmmu/closed_windows.go", "//go:build windows\n\npackage ctxmmu\n\nfunc fileWindows() int { return 2 }\n")
	writeFixtureFile(t, root+"/ctxmmu/closed_other.go", "//go:build !windows\n\npackage ctxmmu\n\nfunc fileWindows() int { return 3 }\n")

	got := runPlatformValidate(t, root, "ctxmmu/compactor.go", "ctxmmu/closed_windows.go", "ctxmmu/closed_other.go")
	if !got.OK {
		t.Fatalf("explicit platform pair refused: %+v", got)
	}
	for _, name := range []string{"build", "platform_filename", "cross_build"} {
		if status, ok := phaseStatus(got, name); !ok || status != "ok" {
			t.Fatalf("phase %q status=%q present=%v; phases=%+v", name, status, ok, got.Phases)
		}
	}
}

func runPlatformValidate(t *testing.T, root string, mine ...string) validationReceipt {
	t.Helper()
	args := []string{"--root", root, "--wsl-tests=false", "--progress=false", "--json"}
	for _, m := range mine {
		args = append(args, "--mine", m)
	}
	var stdout, stderr bytes.Buffer
	validate.Run(&stdout, &stderr, args)
	var got validationReceipt
	if err := json.Unmarshal(stdout.Bytes(), &got); err != nil {
		t.Fatalf("decode receipt: %v\nstdout=%s\nstderr=%s", err, stdout.String(), stderr.String())
	}
	return got
}
