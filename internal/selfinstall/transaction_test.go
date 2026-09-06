package selfinstall

import (
	"errors"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/launchshim"
)

func TestRunTransactionSharedSourceUpdatesAllTargets(t *testing.T) {
	dir := t.TempDir()
	source := writeTransactionFile(t, dir, "source", "new")
	targets := []string{
		writeTransactionFile(t, dir, "c-target", "old-c"),
		writeTransactionFile(t, dir, "a-target", "old-a"),
		writeTransactionFile(t, dir, "b-target", "old-b"),
	}

	result := RunTransaction([]Copy{
		{Source: source, Target: targets[0]},
		{Source: source, Target: targets[1]},
		{Source: source, Target: targets[2]},
	}, OSSwap)

	updated, ok := result.(Updated)
	if !ok {
		t.Fatalf("result = %#v, want Updated", result)
	}
	if updated.Attempted != 3 || updated.Changed != 3 || updated.Err != nil || len(updated.RollbackErrors) != 0 {
		t.Fatalf("result = %#v", updated)
	}
	assertTransactionContents(t, source, "new")
	for _, target := range targets {
		assertTransactionContents(t, target, "new")
	}
	assertNoTransactionDebris(t, dir)
}

func TestRunLaunchTransactionDigestEqualSkipsActivation(t *testing.T) {
	dir := t.TempDir()
	source := writeTransactionFile(t, dir, "source", "verified-identical")
	target := writeTransactionFile(t, dir, "target", "verified-identical")
	activations := 0

	result := RunLaunchTransaction([]Copy{{Source: source, Target: target}}, target, func(source, target string) error {
		activations++
		return OSSwap(source, target)
	})

	updated, ok := result.(Updated)
	if !ok {
		t.Fatalf("result = %#v, want Updated", result)
	}
	if updated.Attempted != 1 || updated.Changed != 0 || activations != 0 {
		t.Fatalf("result=%#v activations=%d, want one considered copy and zero activation", updated, activations)
	}
	if _, err := os.Stat(launchshim.UpdateStatePath(target)); !errors.Is(err, os.ErrNotExist) {
		t.Fatalf("digest-equal no-op published launch state: %v", err)
	}
	assertTransactionContents(t, target, "verified-identical")
	assertNoTransactionDebris(t, dir)
}

func TestRunTransactionExecutableModeRepairsTarget(t *testing.T) {
	if !supportsPOSIXPermissions() {
		t.Skip("skipping POSIX executable mode repair test on unsupported platform")
	}

	dir := t.TempDir()
	contents := "binary-content-identical"
	source := writeTransactionFileWithMode(t, dir, "source", contents, 0o755)
	target := writeTransactionFileWithMode(t, dir, "target", contents, 0o644)

	targetInfoBefore, err := os.Stat(target)
	if err != nil {
		t.Fatal(err)
	}
	if targetInfoBefore.Mode().Perm()&0111 != 0 {
		t.Fatalf("target should not be executable before transaction, got %o", targetInfoBefore.Mode().Perm())
	}

	activations := 0
	result := RunTransaction([]Copy{{Source: source, Target: target}}, func(src, dst string) error {
		activations++
		return OSSwap(src, dst)
	})

	updated, ok := result.(Updated)
	if !ok {
		t.Fatalf("result = %#v, want Updated", result)
	}
	if updated.Attempted != 1 || updated.Changed != 1 || activations != 1 {
		t.Fatalf("result = %#v, activations = %d, want 1 attempted, 1 changed, 1 activation", updated, activations)
	}

	targetInfoAfter, err := os.Stat(target)
	if err != nil {
		t.Fatal(err)
	}
	if targetInfoAfter.Mode().Perm()&0111 == 0 {
		t.Fatalf("target should be executable after transaction, got %o", targetInfoAfter.Mode().Perm())
	}
	assertTransactionContents(t, target, contents)
	assertNoTransactionDebris(t, dir)

	// Second run should be a no-op since target mode is now converged.
	secondActivations := 0
	secondResult := RunTransaction([]Copy{{Source: source, Target: target}}, func(src, dst string) error {
		secondActivations++
		return OSSwap(src, dst)
	})
	secondUpdated, ok := secondResult.(Updated)
	if !ok {
		t.Fatalf("second result = %#v, want Updated", secondResult)
	}
	if secondUpdated.Attempted != 1 || secondUpdated.Changed != 0 || secondActivations != 0 {
		t.Fatalf("second result = %#v, activations = %d, want no-op", secondUpdated, secondActivations)
	}
}

func TestRunLaunchTransactionExecutableModeRepairsTarget(t *testing.T) {
	if !supportsPOSIXPermissions() {
		t.Skip("skipping POSIX executable mode repair test on unsupported platform")
	}

	dir := t.TempDir()
	contents := "launch-binary-identical"
	source := writeTransactionFileWithMode(t, dir, "source", contents, 0o755)
	target := writeTransactionFileWithMode(t, dir, "target", contents, 0o644)

	activations := 0
	result := RunLaunchTransaction([]Copy{{Source: source, Target: target}}, target, func(src, dst string) error {
		activations++
		return OSSwap(src, dst)
	})

	updated, ok := result.(Updated)
	if !ok {
		t.Fatalf("result = %#v, want Updated", result)
	}
	if updated.Attempted != 1 || updated.Changed != 1 || activations != 1 {
		t.Fatalf("result = %#v, activations = %d, want 1 attempted, 1 changed, 1 activation", updated, activations)
	}

	targetInfo, err := os.Stat(target)
	if err != nil {
		t.Fatal(err)
	}
	if targetInfo.Mode().Perm()&0111 == 0 {
		t.Fatalf("target should be executable after transaction, got %o", targetInfo.Mode().Perm())
	}
	assertTransactionContents(t, target, contents)
	assertNoTransactionDebris(t, dir)
}

func TestRunTransactionExecutableModeRollbackRestoresTargetMode(t *testing.T) {
	if !supportsPOSIXPermissions() {
		t.Skip("skipping POSIX executable mode repair test on unsupported platform")
	}

	dir := t.TempDir()
	contents := "rollback-binary-identical"
	source := writeTransactionFileWithMode(t, dir, "source", contents, 0o755)
	target := writeTransactionFileWithMode(t, dir, "target", contents, 0o644)

	result := RunTransaction([]Copy{{Source: source, Target: target}}, func(src, dst string) error {
		return errors.New("injected activation failure")
	})

	rolledBack, ok := result.(RolledBack)
	if !ok || rolledBack.Err == nil {
		t.Fatalf("result = %#v, want RolledBack", result)
	}
	targetInfo, err := os.Stat(target)
	if err != nil {
		t.Fatal(err)
	}
	if targetInfo.Mode().Perm()&0111 != 0 {
		t.Fatalf("target should remain non-executable after rollback, got %o", targetInfo.Mode().Perm())
	}
	assertTransactionContents(t, target, contents)
	assertNoTransactionDebris(t, dir)
}

func TestExecutableModeNeedsRepairFromPerm(t *testing.T) {
	tests := []struct {
		name       string
		sourcePerm os.FileMode
		targetPerm os.FileMode
		want       bool
	}{
		{name: "0755 vs 0644 needs repair", sourcePerm: 0o755, targetPerm: 0o644, want: true},
		{name: "0755 vs 0755 already executable", sourcePerm: 0o755, targetPerm: 0o755, want: false},
		{name: "0644 vs 0644 neither executable", sourcePerm: 0o644, targetPerm: 0o644, want: false},
		{name: "0700 vs 0644 owner executable needs repair", sourcePerm: 0o700, targetPerm: 0o644, want: true},
		{name: "0100 vs 0644 minimal executable needs repair", sourcePerm: 0o100, targetPerm: 0o644, want: true},
		{name: "0755 vs 0700 target already executable", sourcePerm: 0o755, targetPerm: 0o700, want: false},
		{name: "0644 vs 0755 source not executable", sourcePerm: 0o644, targetPerm: 0o755, want: false},
		{name: "0600 vs 0644 neither executable", sourcePerm: 0o600, targetPerm: 0o644, want: false},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := executableModeNeedsRepairFromPerm(tt.sourcePerm, tt.targetPerm)
			if got != tt.want {
				t.Fatalf("executableModeNeedsRepairFromPerm(%o, %o) = %v, want %v", tt.sourcePerm, tt.targetPerm, got, tt.want)
			}
		})
	}
}

func TestComponentPlanExecutableModeRepairsTarget(t *testing.T) {
	if !supportsPOSIXPermissions() {
		t.Skip("skipping POSIX executable mode repair test on unsupported platform")
	}

	dir := t.TempDir()
	contents := "component-identical"
	source := writeTransactionFileWithMode(t, dir, "source", contents, 0o755)
	target := writeTransactionFileWithMode(t, dir, "target", contents, 0o644)

	components := []Component{
		{Name: "fak", Source: source, Target: target, CompatibilityGroup: "launcher"},
	}
	plans, err := PlanComponents(components)
	if err != nil {
		t.Fatal(err)
	}
	if len(plans) != 1 {
		t.Fatalf("plans = %#v, want 1 plan", plans)
	}
	if plans[0].Activation != ComponentActivate {
		t.Fatalf("plan activation = %v, want %v", plans[0].Activation, ComponentActivate)
	}
	if plans[0].Rollback != "restore_installed_artifact" {
		t.Fatalf("plan rollback = %v, want restore_installed_artifact", plans[0].Rollback)
	}

	copies := CopiesForActivation(components, plans)
	if len(copies) != 1 || copies[0].Target != target {
		t.Fatalf("copies = %#v, want 1 copy for target %q", copies, target)
	}
}

func TestRunTransactionSecondActivationFailureRestoresFirst(t *testing.T) {
	dir := t.TempDir()
	source := writeTransactionFile(t, dir, "source", "new")
	a := writeTransactionFile(t, dir, "a-target", "old-a")
	b := writeTransactionFile(t, dir, "b-target", "old-b")
	c := writeTransactionFile(t, dir, "c-target", "old-c")
	calls := 0
	swap := func(source, target string) error {
		calls++
		if calls == 2 {
			return errors.New("injected activation failure")
		}
		return OSSwap(source, target)
	}

	result := RunTransaction([]Copy{
		{Source: source, Target: c},
		{Source: source, Target: a},
		{Source: source, Target: b},
	}, swap)

	rolledBack, ok := result.(RolledBack)
	if !ok {
		t.Fatalf("result = %#v, want RolledBack", result)
	}
	if rolledBack.Attempted != 2 || rolledBack.Changed != 1 || rolledBack.Err == nil || len(rolledBack.RollbackErrors) != 0 {
		t.Fatalf("result = %#v", rolledBack)
	}
	if calls != 3 {
		t.Fatalf("swap calls = %d, want 3", calls)
	}
	assertTransactionContents(t, source, "new")
	assertTransactionContents(t, a, "old-a")
	assertTransactionContents(t, b, "old-b")
	assertTransactionContents(t, c, "old-c")
	assertNoTransactionDebris(t, dir)
}

func TestRunTransactionReportsRollbackFailure(t *testing.T) {
	dir := t.TempDir()
	source := writeTransactionFile(t, dir, "source", "new")
	a := writeTransactionFile(t, dir, "a-target", "old-a")
	b := writeTransactionFile(t, dir, "b-target", "old-b")
	calls := 0
	swap := func(source, target string) error {
		calls++
		switch calls {
		case 2:
			return errors.New("injected activation failure")
		case 3:
			return errors.New("injected rollback failure")
		default:
			return OSSwap(source, target)
		}
	}

	result := RunTransaction([]Copy{{Source: source, Target: b}, {Source: source, Target: a}}, swap)

	failed, ok := result.(RollbackFailed)
	if !ok {
		t.Fatalf("result = %#v, want RollbackFailed", result)
	}
	if failed.Attempted != 2 || failed.Changed != 1 || failed.Err == nil || len(failed.RollbackErrors) != 1 {
		t.Fatalf("result = %#v", failed)
	}
	if !strings.Contains(failed.RollbackErrors[0].Error(), "injected rollback failure") {
		t.Fatalf("rollback error = %q", failed.RollbackErrors[0])
	}
	assertTransactionContents(t, source, "new")
	assertTransactionContents(t, a, "new")
	assertTransactionContents(t, b, "old-b")
	assertNoTransactionDebris(t, dir)
}

func TestRunTransactionPreflightFailureMutatesNothing(t *testing.T) {
	dir := t.TempDir()
	source := writeTransactionFile(t, dir, "source", "new")
	a := writeTransactionFile(t, dir, "a-target", "old-a")
	missing := filepath.Join(dir, "missing-target")
	called := false

	result := RunTransaction([]Copy{
		{Source: source, Target: a},
		{Source: source, Target: missing},
	}, func(source, target string) error {
		called = true
		return OSSwap(source, target)
	})

	rolledBack, ok := result.(RolledBack)
	if !ok || rolledBack.Err == nil {
		t.Fatalf("result = %#v, want preflight RolledBack error", result)
	}
	if called {
		t.Fatal("swapper called after preflight failure")
	}
	assertTransactionContents(t, source, "new")
	assertTransactionContents(t, a, "old-a")
	if _, err := os.Stat(missing); !errors.Is(err, os.ErrNotExist) {
		t.Fatalf("missing target stat error = %v", err)
	}
	assertNoTransactionDebris(t, dir)
}

func TestRunTransactionRejectsInvalidCopiesBeforeMutation(t *testing.T) {
	dir := t.TempDir()
	source := writeTransactionFile(t, dir, "source", "new")
	target := writeTransactionFile(t, dir, "target", "old")
	tests := map[string][]Copy{
		"no copies":        nil,
		"empty source":     {{Target: target}},
		"empty target":     {{Source: source}},
		"duplicate target": {{Source: source, Target: target}, {Source: source, Target: filepath.Join(dir, ".", "target")}},
		"missing source":   {{Source: filepath.Join(dir, "missing-source"), Target: target}},
	}
	for name, copies := range tests {
		t.Run(name, func(t *testing.T) {
			called := false
			result := RunTransaction(copies, func(source, target string) error {
				called = true
				return nil
			})
			rolledBack, ok := result.(RolledBack)
			if !ok || rolledBack.Err == nil {
				t.Fatalf("result = %#v, want preflight RolledBack error", result)
			}
			if called {
				t.Fatal("swapper called")
			}
			assertTransactionContents(t, source, "new")
			assertTransactionContents(t, target, "old")
			assertNoTransactionDebris(t, dir)
		})
	}
}

func TestLaunchTransactionHoldsPriorAtReplacementBoundary(t *testing.T) {
	dir := t.TempDir()
	source := writeTransactionFile(t, dir, "source", "new")
	target := writeTransactionFile(t, dir, "target", "old")

	replacementEntered := make(chan struct{})
	releaseReplacement := make(chan struct{})
	resultc := make(chan TransactionResult, 1)
	go func() {
		resultc <- RunLaunchTransaction([]Copy{{Source: source, Target: target}}, target, func(source, target string) error {
			close(replacementEntered)
			<-releaseReplacement
			return OSSwap(source, target)
		})
	}()
	<-replacementEntered

	selected, err := launchshim.ResolveExecutable(target, launchshim.UpdatePolicyPrior, time.Second)
	if err != nil || selected != LaunchPriorPath(target) {
		t.Fatalf("prior selected=%q err=%v, want %q", selected, err, LaunchPriorPath(target))
	}
	assertTransactionContents(t, selected, "old")
	if _, err := launchshim.ResolveExecutable(target, launchshim.UpdatePolicyFail, time.Second); err == nil || !strings.Contains(err.Error(), "self-update is replacing") {
		t.Fatalf("strict failure=%v", err)
	}

	close(releaseReplacement)
	if result := <-resultc; reflect.TypeOf(result) != reflect.TypeOf(Updated{}) {
		t.Fatalf("transaction result=%#v, want Updated", result)
	}
	selected, err = launchshim.ResolveExecutable(target, launchshim.UpdatePolicyWait, time.Second)
	if err != nil || selected != target {
		t.Fatalf("completed transaction selected=%q err=%v, want %q", selected, err, target)
	}
	assertTransactionContents(t, target, "new")
	assertTransactionContents(t, LaunchPriorPath(target), "old")
	if _, err := os.Stat(launchshim.UpdateStatePath(target)); !errors.Is(err, os.ErrNotExist) {
		t.Fatalf("launch state remains after completion: %v", err)
	}
}

func TestLaunchPriorPathKeepsWindowsExecutableSuffix(t *testing.T) {
	got := LaunchPriorPath(filepath.Join("dir", "fak.exe"))
	if filepath.Ext(got) != ".exe" || !strings.Contains(filepath.Base(got), "self-update-prior") {
		t.Fatalf("prior path=%q", got)
	}
}

func writeTransactionFile(t *testing.T, dir, name, contents string) string {
	t.Helper()
	return writeTransactionFileWithMode(t, dir, name, contents, 0o755)
}

func writeTransactionFileWithMode(t *testing.T, dir, name, contents string, mode os.FileMode) string {
	t.Helper()
	path := filepath.Join(dir, name)
	if err := os.WriteFile(path, []byte(contents), mode); err != nil {
		t.Fatal(err)
	}
	if err := os.Chmod(path, mode); err != nil {
		t.Fatal(err)
	}
	return path
}

func assertTransactionContents(t *testing.T, path, want string) {
	t.Helper()
	got, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	if string(got) != want {
		t.Fatalf("%s contents = %q, want %q", filepath.Base(path), got, want)
	}
}

func assertNoTransactionDebris(t *testing.T, dir string) {
	t.Helper()
	entries, err := os.ReadDir(dir)
	if err != nil {
		t.Fatal(err)
	}
	var debris []string
	for _, entry := range entries {
		if strings.Contains(entry.Name(), ".selfinstall-") {
			debris = append(debris, entry.Name())
		}
	}
	if !reflect.DeepEqual(debris, []string(nil)) {
		t.Fatalf("transaction debris = %v", debris)
	}
}
