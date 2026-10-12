package selfupdatecmd

import (
	"os"
	"path/filepath"
	"runtime/debug"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/selfinstall"
)

func runningExecutable(t *testing.T) string {
	t.Helper()
	exe, err := os.Executable()
	if err != nil {
		t.Fatalf("os.Executable: %v", err)
	}
	abs, err := filepath.Abs(filepath.Clean(exe))
	if err != nil {
		t.Fatalf("abs %q: %v", exe, err)
	}
	return abs
}

func samePath(a, b string) bool {
	return strings.EqualFold(filepath.Clean(a), filepath.Clean(b))
}

// fak-test:runtime fast est=10ms lane=default
func TestFakMainPackageIsModuleCmdFak(t *testing.T) {
	bi, ok := debug.ReadBuildInfo()
	if !ok || bi.Main.Path == "" {
		t.Skip("no build info for the test binary")
	}
	if want := bi.Main.Path + "/cmd/fak"; fakMainPackage != want {
		t.Fatalf("fakMainPackage = %q, want %q", fakMainPackage, want)
	}
}

// A go test binary is not the cmd/fak main package, exactly like the standalone
// fak-selfupdate helper: neither may name itself a fak hot copy.
// fak-test:runtime fast est=50ms lane=default
func TestSelfUpdateHostOmitsNonFakInvoker(t *testing.T) {
	home := t.TempDir()
	t.Setenv("HOME", home)
	t.Setenv("USERPROFILE", home)
	repo := t.TempDir()
	exe := runningExecutable(t)

	if runningAsFak() {
		t.Fatal("runningAsFak() = true inside a go test binary")
	}
	h := selfUpdateHost(repo)
	if h.Scheduled != "" {
		t.Fatalf("Host.Scheduled = %q, want empty for a non-fak executable", h.Scheduled)
	}
	if h.RepoRoot != repo {
		t.Fatalf("Host.RepoRoot = %q, want %q", h.RepoRoot, repo)
	}

	for _, c := range selfinstall.Roles(h) {
		if c.Role == selfinstall.RoleScheduled {
			t.Fatalf("Roles contains RoleScheduled at %q for a non-fak executable", c.Path)
		}
		if samePath(c.Path, exe) {
			t.Fatalf("Roles names the running executable %q as role %q", exe, c.Role)
		}
	}

	probed := []string{}
	probe := func(p string) (string, bool, bool) {
		probed = append(probed, p)
		return "deadbeef", false, true
	}
	census := selfinstall.Census(h, probe)
	for _, p := range probed {
		if samePath(p, exe) {
			t.Fatalf("Census probed the running executable %q", exe)
		}
	}
	audit := selfinstall.AuditCopies(census, "cafef00d")
	for _, set := range [][]selfinstall.Role{audit.Divergent, audit.Dirty, audit.Unattested, audit.Missing} {
		for _, r := range set {
			if r == selfinstall.RoleScheduled {
				t.Fatalf("audit grades RoleScheduled for a non-fak executable: %+v", audit.Partition())
			}
		}
	}

	for _, target := range []string{filepath.Join(home, "elsewhere", "fak"), ""} {
		for _, p := range selfUpdateSiblings(repo, target) {
			if samePath(p, exe) {
				t.Fatalf("selfUpdateSiblings(%q) returns the running executable %q as a write target", target, exe)
			}
		}
	}
}

// When the invoker IS a fak build (fak self-update under a scheduled task), the scheduled
// role stays in the table and remains a convergence target.
// fak-test:runtime fast est=20ms lane=default
func TestScheduledRolePreservedForFakInvoker(t *testing.T) {
	dir := t.TempDir()
	fakExe := filepath.Join(dir, "scheduled", "fak")
	if err := os.MkdirAll(filepath.Dir(fakExe), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(fakExe, []byte("bin"), 0o755); err != nil {
		t.Fatal(err)
	}
	copies := selfinstall.Roles(selfinstall.Host{RepoRoot: dir, Scheduled: fakExe})
	found := false
	for _, c := range copies {
		if c.Role == selfinstall.RoleScheduled {
			found = true
			if !samePath(c.Path, fakExe) || !c.Present {
				t.Fatalf("RoleScheduled = %+v, want present at %q", c, fakExe)
			}
		}
	}
	if !found {
		t.Fatal("Roles omits RoleScheduled for a non-empty Scheduled path")
	}
	if !selfinstall.Convergeable(selfinstall.RoleScheduled) {
		t.Fatal("RoleScheduled is no longer convergeable")
	}
	targets := selfinstall.ConvergeTargets(copies, filepath.Join(dir, "other", "fak"))
	hit := false
	for _, p := range targets {
		if samePath(p, fakExe) {
			hit = true
		}
	}
	if !hit {
		t.Fatalf("ConvergeTargets = %v, want it to include the scheduled fak %q", targets, fakExe)
	}
}
