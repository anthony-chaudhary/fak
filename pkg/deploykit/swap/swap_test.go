package swap

import (
	"bufio"
	"bytes"
	"crypto/sha256"
	"errors"
	"fmt"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"runtime"
	"sort"
	"strings"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/selfinstall"
	"github.com/anthony-chaudhary/fak/pkg/deploykit"
)

// helperEnv selects the helper-process mode when the test binary re-runs itself.
const helperEnv = "DEPLOYKIT_SWAP_HELPER"

// TestHelperProcess is not a test. The tests below re-run the test binary with helperEnv set
// so a real child process can hold a running .exe or a lock.
func TestHelperProcess(t *testing.T) {
	mode := os.Getenv(helperEnv)
	if mode == "" {
		return
	}
	// Never outlive an abandoned parent.
	time.AfterFunc(2*time.Minute, func() { os.Exit(3) })
	switch mode {
	case "block":
		// Answer each stdin line with "pong", so the parent can prove this image is still
		// executing; exit on EOF.
		in := bufio.NewScanner(os.Stdin)
		for in.Scan() {
			fmt.Println("pong")
		}
	case "print":
		fmt.Print("helper-ok")
	case "lock":
		release, err := Lock(os.Getenv(helperEnv+"_DIR"), os.Getenv(helperEnv+"_NAME"))
		if err != nil {
			fmt.Printf("lock-error %v\n", err)
			os.Exit(2)
		}
		fmt.Println("locked")
		_, _ = io.Copy(io.Discard, os.Stdin)
		release()
	default:
		os.Exit(4)
	}
	os.Exit(0)
}

// helper is a running child process started from exe in the given mode. Closing its stdin
// ends it.
type helper struct {
	cmd    *exec.Cmd
	stdin  io.WriteCloser
	stdout *bufio.Reader
}

func startHelper(t *testing.T, exe, mode string, env ...string) *helper {
	t.Helper()
	cmd := exec.Command(exe, "-test.run=^TestHelperProcess$")
	cmd.Env = append(append(os.Environ(), helperEnv+"="+mode), env...)
	stdin, err := cmd.StdinPipe()
	if err != nil {
		t.Fatal(err)
	}
	stdout, err := cmd.StdoutPipe()
	if err != nil {
		t.Fatal(err)
	}
	if err := cmd.Start(); err != nil {
		t.Fatalf("start helper %s: %v", mode, err)
	}
	h := &helper{cmd: cmd, stdin: stdin, stdout: bufio.NewReader(stdout)}
	t.Cleanup(func() {
		_ = h.stdin.Close()
		_ = h.cmd.Process.Kill()
		_ = h.cmd.Wait()
	})
	return h
}

// stop closes the helper's stdin and waits for it to exit.
func (h *helper) stop(t *testing.T) {
	t.Helper()
	_ = h.stdin.Close()
	if err := h.cmd.Wait(); err != nil {
		t.Fatalf("helper exit: %v", err)
	}
}

// ping round-trips one line through a "block" helper, proving it is still executing.
func (h *helper) ping(t *testing.T) {
	t.Helper()
	if _, err := io.WriteString(h.stdin, "ping\n"); err != nil {
		t.Fatalf("ping helper: %v", err)
	}
	line, err := h.stdout.ReadString('\n')
	if err != nil || strings.TrimSpace(line) != "pong" {
		t.Fatalf("helper did not answer: line=%q err=%v", line, err)
	}
}

func testExecutable(t *testing.T) string {
	t.Helper()
	exe, err := os.Executable()
	if err != nil {
		t.Fatal(err)
	}
	return exe
}

func writeFile(t *testing.T, path, body string) {
	t.Helper()
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(path, []byte(body), 0o755); err != nil {
		t.Fatal(err)
	}
}

func readFile(t *testing.T, path string) string {
	t.Helper()
	b, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	return string(b)
}

func dirNames(t *testing.T, dir string) []string {
	t.Helper()
	entries, err := os.ReadDir(dir)
	if err != nil {
		t.Fatal(err)
	}
	var names []string
	for _, e := range entries {
		names = append(names, e.Name())
	}
	sort.Strings(names)
	return names
}

func TestSwapReplacesTarget(t *testing.T) {
	dir := t.TempDir()
	src, dst := filepath.Join(dir, "candidate"), filepath.Join(dir, "app")
	writeFile(t, src, "new")
	writeFile(t, dst, "old")
	if err := Swap(src, dst); err != nil {
		t.Fatalf("Swap: %v", err)
	}
	if got := readFile(t, dst); got != "new" {
		t.Fatalf("target = %q, want new", got)
	}
	if _, err := os.Stat(src); !errors.Is(err, os.ErrNotExist) {
		t.Fatalf("Swap must consume src; stat err = %v", err)
	}
}

// TestSwapRunningExeWindows replaces an .exe while a process launched from it is still
// running, then launches the replacement. Windows refuses to overwrite a mapped image, which
// is the case OSSwap's rename-aside exists for.
func TestSwapRunningExeWindows(t *testing.T) {
	if runtime.GOOS != "windows" {
		t.Skip("TestSwapRunningExeWindows: the rename-aside path only exists on Windows; the unix rename is covered by TestSwapReplacesTarget")
	}
	dir := t.TempDir()
	self, err := os.ReadFile(testExecutable(t))
	if err != nil {
		t.Fatal(err)
	}
	app := filepath.Join(dir, "app.exe")
	if err := os.WriteFile(app, self, 0o755); err != nil {
		t.Fatal(err)
	}
	running := startHelper(t, app, "block")
	running.ping(t)

	// Precondition: a plain replace of the running image fails, so the swap below really
	// exercises the running-exe path.
	probe := filepath.Join(dir, "probe.exe")
	writeFile(t, probe, "probe")
	if err := os.Rename(probe, app); err == nil {
		t.Fatal("precondition: os.Rename over a running .exe succeeded; the running-exe path is not exercised")
	}

	// The replacement is the same image plus an overlay marker, so it stays launchable and is
	// byte-distinguishable from the running one.
	next := append(append([]byte(nil), self...), []byte("\ndeploykit-swap-marker\n")...)
	candidate := filepath.Join(dir, "app.exe.candidate")
	if err := os.WriteFile(candidate, next, 0o755); err != nil {
		t.Fatal(err)
	}
	if err := Swap(candidate, app); err != nil {
		t.Fatalf("Swap over running exe: %v", err)
	}
	got, err := os.ReadFile(app)
	if err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(got, next) {
		t.Fatal("running exe was not replaced with the candidate bytes")
	}
	running.ping(t) // the old image, now renamed aside, is still executing
	cmd := exec.Command(app, "-test.run=^TestHelperProcess$")
	cmd.Env = append(os.Environ(), helperEnv+"=print")
	out, err := cmd.CombinedOutput()
	if err != nil || string(out) != "helper-ok" {
		t.Fatalf("replacement exe did not launch: err=%v out=%q", err, out)
	}
	running.stop(t)
}

func TestReapAsidesUsesRealLiveness(t *testing.T) {
	exe := testExecutable(t)
	exited := exec.Command(exe, "-test.run=^TestHelperProcess$")
	exited.Env = append(os.Environ(), helperEnv+"=print")
	if err := exited.Run(); err != nil {
		t.Fatalf("run exited helper: %v", err)
	}
	live := startHelper(t, exe, "block")

	dir := t.TempDir()
	target := filepath.Join(dir, "app.exe")
	writeFile(t, target, "live")
	aside := func(pid int) string { return fmt.Sprintf("%s.old.%d.0", target, pid) }
	dead, mine, alive := aside(exited.Process.Pid), aside(os.Getpid()), aside(live.cmd.Process.Pid)
	kept := []string{mine, alive, target + ".old", target + ".old-20260925-101010"}
	for _, p := range append([]string{dead}, kept...) {
		writeFile(t, p, "aside")
	}

	reaped := ReapAsides(target)
	if len(reaped) != 1 || reaped[0] != dead {
		t.Fatalf("ReapAsides = %v, want only the dead-owner aside %s", reaped, dead)
	}
	for _, p := range append([]string{target}, kept...) {
		if _, err := os.Stat(p); err != nil {
			t.Fatalf("%s must survive: %v", filepath.Base(p), err)
		}
	}
	live.stop(t)
}

// transactionFixture makes three targets a, b, c (activated in that lexical order) and a
// candidate for each in a separate directory.
func transactionFixture(t *testing.T) (copies []Copy, targets []string, pre map[string]string) {
	t.Helper()
	root := t.TempDir()
	pre = map[string]string{}
	for _, name := range []string{"a", "b", "c"} {
		target := filepath.Join(root, "install", name)
		source := filepath.Join(root, "candidates", name)
		writeFile(t, target, "old-"+name)
		writeFile(t, source, "new-"+name)
		copies = append(copies, Copy{Source: source, Target: target})
		targets = append(targets, target)
		pre[target] = "old-" + name
	}
	return copies, targets, pre
}

// failingSwapper performs the real swap except on the listed (1-based) calls.
func failingSwapper(failOn ...int) (selfinstall.Swapper, *int) {
	calls := 0
	return func(src, dst string) error {
		calls++
		for _, n := range failOn {
			if calls == n {
				return fmt.Errorf("induced swap failure on call %d (%s)", n, filepath.Base(dst))
			}
		}
		return Swap(src, dst)
	}, &calls
}

func TestTransactionUpdatesAll(t *testing.T) {
	copies, targets, _ := transactionFixture(t)
	res := Transaction(copies)
	u, ok := res.(Updated)
	if !ok || u.Changed != 3 || u.Err != nil {
		t.Fatalf("Transaction = %#v, want Updated with 3 changed", res)
	}
	for i, target := range targets {
		if got := readFile(t, target); got != "new-"+[]string{"a", "b", "c"}[i] {
			t.Fatalf("%s = %q after Updated", target, got)
		}
		if _, err := os.Stat(copies[i].Source); err != nil {
			t.Fatalf("Transaction must not consume source %s: %v", copies[i].Source, err)
		}
	}
	if names := dirNames(t, filepath.Dir(targets[0])); strings.Join(names, ",") != "a,b,c" {
		t.Fatalf("install dir = %v, want only a,b,c (no staging residue)", names)
	}
}

// TestTransactionInducedFailureRestoresAll is the negative witness: the swap of target 2 of 3
// fails, and every target is left byte-identical to its pre-state.
func TestTransactionInducedFailureRestoresAll(t *testing.T) {
	copies, targets, pre := transactionFixture(t)
	swapper, calls := failingSwapper(2) // call 1 activates a, call 2 would activate b
	res := transaction(copies, swapper)
	rb, ok := res.(RolledBack)
	if !ok {
		t.Fatalf("Transaction = %#v, want RolledBack", res)
	}
	if rb.Attempted != 2 || rb.Changed != 1 || rb.Err == nil || !strings.Contains(rb.Err.Error(), "induced") {
		t.Fatalf("RolledBack = %+v, want Attempted=2 Changed=1 with the induced error", rb)
	}
	if *calls != 3 { // activate a, fail b, restore a
		t.Fatalf("swapper calls = %d, want 3", *calls)
	}
	for _, target := range targets {
		if got := readFile(t, target); got != pre[target] {
			t.Fatalf("%s = %q after rollback, want pre-state %q", filepath.Base(target), got, pre[target])
		}
	}
	if names := dirNames(t, filepath.Dir(targets[0])); strings.Join(names, ",") != "a,b,c" {
		t.Fatalf("install dir = %v, want only a,b,c (no staging residue)", names)
	}
}

func TestTransactionRollbackFailedPreservesSnapshot(t *testing.T) {
	copies, targets, pre := transactionFixture(t)
	// Calls: 1 activate a, 2 activate b, 3 fail c, 4 fail restoring b, 5 restore a.
	swapper, _ := failingSwapper(3, 4)
	res := transaction(copies, swapper)
	rf, ok := res.(RollbackFailed)
	if !ok {
		t.Fatalf("Transaction = %#v, want RollbackFailed", res)
	}
	if rf.Changed != 2 || len(rf.RollbackErrors) != 1 || len(rf.Snapshots) != 1 {
		t.Fatalf("RollbackFailed = %+v, want Changed=2 with one failed restore", rf)
	}
	if got := readFile(t, rf.Snapshots[0]); got != pre[targets[1]] {
		t.Fatalf("preserved snapshot = %q, want b's pre-state %q", got, pre[targets[1]])
	}
	if got := readFile(t, targets[0]); got != pre[targets[0]] {
		t.Fatalf("a = %q, want it restored to %q", got, pre[targets[0]])
	}
}

// TestLockRefusesSecondHolder is the negative witness: a second Lock on the same name, in this
// process or another, returns ErrBusy until the holder releases.
func TestLockRefusesSecondHolder(t *testing.T) {
	dir := t.TempDir()
	release, err := Lock(dir, "router")
	if err != nil {
		t.Fatalf("first Lock: %v", err)
	}
	if _, err := os.Stat(filepath.Join(dir, "router.deploy.lock")); err != nil {
		t.Fatalf("lock file: %v", err)
	}
	if _, err := Lock(dir, "router"); !errors.Is(err, ErrBusy) {
		t.Fatalf("second Lock err = %v, want ErrBusy", err)
	}
	other, err := Lock(dir, "dashboard")
	if err != nil {
		t.Fatalf("a different deployable must not contend: %v", err)
	}
	other()
	release()
	release() // idempotent

	again, err := Lock(dir, "router")
	if err != nil {
		t.Fatalf("Lock after release: %v", err)
	}
	again()

	// Cross-process: a child holds the lock and this process is refused. The lock frees both
	// when the child releases it and when the child is killed without releasing.
	for _, kill := range []bool{false, true} {
		child := startHelper(t, testExecutable(t), "lock", helperEnv+"_DIR="+dir, helperEnv+"_NAME=router")
		line, err := child.stdout.ReadString('\n')
		if err != nil || strings.TrimSpace(line) != "locked" {
			t.Fatalf("child lock: line=%q err=%v", line, err)
		}
		if _, err := Lock(dir, "router"); !errors.Is(err, ErrBusy) {
			t.Fatalf("Lock while another process holds it (kill=%t): err = %v, want ErrBusy", kill, err)
		}
		if kill {
			_ = child.cmd.Process.Kill()
			_ = child.cmd.Wait()
		} else {
			child.stop(t)
		}
		// Windows drops a killed holder's byte-range lock asynchronously; allow it a moment.
		deadline := time.Now().Add(10 * time.Second)
		for {
			after, err := Lock(dir, "router")
			if err == nil {
				after()
				break
			}
			if !errors.Is(err, ErrBusy) || time.Now().After(deadline) {
				t.Fatalf("Lock after the holder exited (kill=%t): %v", kill, err)
			}
			time.Sleep(50 * time.Millisecond)
		}
	}
}

func TestLockRejectsNonComponentName(t *testing.T) {
	for _, name := range []string{"", " ", ".", "..", "a/b", `a\b`, "../x", "foo:bar", "a*b", "a\x00b"} {
		if release, err := Lock(t.TempDir(), name); err == nil {
			release()
			t.Errorf("Lock(%q) succeeded, want a name error", name)
		}
	}
}

var priorName = regexp.MustCompile(`^app\.deploy-prior-[0-9a-f]{12}\.exe$`)

func TestPriorSlotPathContentAddressed(t *testing.T) {
	dir := t.TempDir()
	sum := sha256.Sum256([]byte("v1"))
	got := PriorSlotPath(filepath.Join(dir, "app.exe"), sum)
	if !priorName.MatchString(filepath.Base(got)) || filepath.Dir(got) != dir {
		t.Fatalf("PriorSlotPath = %s, want <dir>/app.deploy-prior-<sha12>.exe", got)
	}
	if want := fmt.Sprintf("%x", sum)[:12]; !strings.Contains(got, want) {
		t.Fatalf("PriorSlotPath = %s, want sha12 %s", got, want)
	}
	if again := PriorSlotPath(filepath.Join(dir, "app.exe"), sha256.Sum256([]byte("v1"))); again != got {
		t.Fatalf("identical bytes gave %s and %s", got, again)
	}
	if other := PriorSlotPath(filepath.Join(dir, "app.exe"), sha256.Sum256([]byte("v2"))); other == got {
		t.Fatal("different bytes gave the same slot")
	}
	if plain := PriorSlotPath(filepath.Join(dir, "fak"), sum); filepath.Base(plain) != "fak.deploy-prior-"+fmt.Sprintf("%x", sum)[:12] {
		t.Fatalf("non-exe slot = %s", plain)
	}
	if upper := filepath.Base(PriorSlotPath(filepath.Join(dir, "APP.EXE"), sum)); !strings.HasSuffix(upper, ".EXE") {
		t.Fatalf("exe suffix must stay last: %s", upper)
	}
}

func TestSavePriorAndPruneSlots(t *testing.T) {
	dir := t.TempDir()
	target := filepath.Join(dir, "app.exe")
	legacy := target + ".old-20260925-101010"
	writeFile(t, legacy, "legacy")

	var slots []string
	for _, body := range []string{"v1", "v2", "v3"} {
		writeFile(t, target, body)
		slot, err := SavePrior(target)
		if err != nil {
			t.Fatalf("SavePrior %s: %v", body, err)
		}
		if slot != PriorSlotPath(target, sha256.Sum256([]byte(body))) || readFile(t, slot) != body {
			t.Fatalf("SavePrior %s wrote %s with %q", body, slot, readFile(t, slot))
		}
		slots = append(slots, slot)
	}
	// Only the new slot name is written: no staging temp files or swap-asides are left behind.
	wantNames := []string{filepath.Base(target), filepath.Base(legacy)}
	for _, s := range slots {
		wantNames = append(wantNames, filepath.Base(s))
	}
	sort.Strings(wantNames)
	if names := dirNames(t, dir); strings.Join(names, ",") != strings.Join(wantNames, ",") {
		t.Fatalf("dir = %v, want exactly %v", names, wantNames)
	}

	// Re-saving identical bytes reuses the same file (no rewrite) and refreshes its mtime.
	past := time.Now().Add(-2 * time.Hour)
	if err := os.Chtimes(slots[2], past, past); err != nil {
		t.Fatal(err)
	}
	before, err := os.Stat(slots[2])
	if err != nil {
		t.Fatal(err)
	}
	if again, err := SavePrior(target); err != nil || again != slots[2] {
		t.Fatalf("SavePrior identical = %s, %v; want %s", again, err, slots[2])
	}
	after, err := os.Stat(slots[2])
	if err != nil {
		t.Fatal(err)
	}
	if !os.SameFile(before, after) {
		t.Fatal("SavePrior rewrote a slot that already held identical bytes")
	}
	if !after.ModTime().After(past.Add(time.Hour)) {
		t.Fatalf("reused slot mtime = %v, want it refreshed", after.ModTime())
	}
	if n := len(PriorSlots(target)); n != 3 {
		t.Fatalf("PriorSlots = %d, want 3", n)
	}
	for _, s := range LegacySlots(target) {
		if s.Kind == KindPrior || strings.Contains(s.Path, priorMarker) {
			t.Fatalf("LegacySlots reported the new slot %s", s.Path)
		}
	}

	base := time.Now().Add(-time.Hour)
	for i, slot := range slots {
		at := base.Add(time.Duration(i) * time.Minute)
		if err := os.Chtimes(slot, at, at); err != nil {
			t.Fatal(err)
		}
	}
	removed, err := PruneSlots(target, deploykit.RollbackPolicy{KeepSlots: 2})
	if err != nil || len(removed) != 1 || removed[0] != slots[0] {
		t.Fatalf("PruneSlots keep=2 = %v, %v; want only the oldest %s", removed, err, slots[0])
	}
	removed, err = PruneSlots(target, deploykit.RollbackPolicy{})
	if err != nil || len(removed) != 1 || removed[0] != slots[1] {
		t.Fatalf("PruneSlots keep=0 = %v, %v; want %s removed and the newest kept", removed, err, slots[1])
	}
	if left := PriorSlots(target); len(left) != 1 || left[0].Path != slots[2] {
		t.Fatalf("remaining slots = %v, want only %s", left, slots[2])
	}
	for _, p := range []string{target, legacy} {
		if _, err := os.Stat(p); err != nil {
			t.Fatalf("pruning must not touch %s: %v", filepath.Base(p), err)
		}
	}
}

const (
	commitA = "0123456789abcdef0123456789abcdef01234567"
	commitB = "89abcdef0123456789abcdef0123456789abcdef"
	commitC = "fedcba9876543210fedcba9876543210fedcba98"
)

// setMod backdates path to ago before now, so ModTime ordering does not depend on write speed.
func setMod(t *testing.T, path string, ago time.Duration) {
	t.Helper()
	at := time.Now().Add(-ago)
	if err := os.Chtimes(path, at, at); err != nil {
		t.Fatal(err)
	}
}

// checkSlots asserts LegacySlots returned exactly want (path -> kind), each with a ModTime.
func checkSlots(t *testing.T, got []Slot, want map[string]SlotKind) {
	t.Helper()
	seen := map[string]SlotKind{}
	for _, s := range got {
		if s.ModTime.IsZero() {
			t.Errorf("%s has no ModTime", s.Path)
		}
		seen[s.Path] = s.Kind
	}
	for p, kind := range want {
		if seen[p] != kind {
			t.Errorf("%s: kind = %q, want %q", p, seen[p], kind)
		}
	}
	for p, kind := range seen {
		if _, ok := want[p]; !ok {
			t.Errorf("unexpected legacy slot %s (%s)", p, kind)
		}
	}
}

// TestLegacySlotsClassifiesEveryLegacyName covers every legacy writer's name and every
// neighbour that must not be mistaken for a slot of this target.
func TestLegacySlotsClassifiesEveryLegacyName(t *testing.T) {
	dir := t.TempDir()
	stage := filepath.Join(t.TempDir(), "stage")
	target := filepath.Join(dir, "fak-server.exe")
	router := filepath.Join(dir, commitA+"-"+commitB, "fak-server.exe")
	want := map[string]SlotKind{
		target + ".old-20260925-101010":                        KindOld,
		target + ".old-1758790000000000000":                    KindOld,
		target + ".bak":                                        KindBak,
		filepath.Join(dir, "fak-server.self-update-prior.exe"): KindSelfUpdatePrior,
		router: KindRouterCommit,
		filepath.Join(stage, "fak-server.exe.predeploy"): KindPredeploy,
	}
	for p := range want {
		writeFile(t, p, "slot")
	}

	ignored := []string{
		target, // live
		target + ".old",
		target + ".old.1234.0",
		target + ".old.1234.overflow",
		filepath.Join(dir, "fak-server.deploy-prior-0123456789ab.exe"),
		filepath.Join(dir, ".fak-server.exe.selfinstall-stage-1"),
		// Backups of other names in the same directory are not this target's slots.
		filepath.Join(dir, "other.exe.bak"),
		filepath.Join(dir, "other.exe.old-20260925-101010"),
		filepath.Join(dir, "fak.predeploy"),
		filepath.Join(dir, "spirv.predeploy", "kernel.spv"),
		filepath.Join(dir, "spirv.bak", "kernel.spv"),
		filepath.Join(dir, "abc-def", "fak-server.exe"),
		filepath.Join(dir, commitA+"-"+commitC, "readme.txt"), // commit dir without the binary
		// The staged candidate and the upgrade's markers are not rollback copies.
		filepath.Join(stage, "fak-server.exe"),
		filepath.Join(stage, "fak-server.exe.predeploy.absent"),
		filepath.Join(stage, "fak-server.exe.predeploy.pending", "x"),
	}
	for _, p := range ignored {
		writeFile(t, p, "not a slot")
	}
	setMod(t, router, time.Hour) // a real previous candidate predates the live target

	checkSlots(t, LegacySlots(target, stage), want)

	// The name classifier alone agrees, so a remote listing can reuse it.
	for p, kind := range want {
		name := filepath.Base(p)
		if kind == KindRouterCommit {
			name = filepath.Base(filepath.Dir(p))
		}
		if got, ok := LegacyKind(target, name); !ok || got != kind {
			t.Errorf("LegacyKind(%s) = %q, %t; want %q", name, got, ok, kind)
		}
	}
}

// TestLegacySlotsHaloUpgradeLayout uses the upgrade writers' real layout: the binary backup and
// SPIR-V directory backup sit next to the live paths, and the ".predeploy" snapshots of both
// sit in a separate staging directory.
func TestLegacySlotsHaloUpgradeLayout(t *testing.T) {
	root := t.TempDir()
	bin, share, stage := filepath.Join(root, "bin"), filepath.Join(root, "share"), filepath.Join(root, "stage")
	fak, spirv := filepath.Join(bin, "fak"), filepath.Join(share, "spirv")
	writeFile(t, fak, "live")
	writeFile(t, fak+".bak", "prior")
	writeFile(t, filepath.Join(spirv, "kernel.spv"), "live")
	writeFile(t, filepath.Join(share, "spirv.bak", "kernel.spv"), "prior")
	writeFile(t, filepath.Join(stage, "fak.predeploy"), "prior")
	writeFile(t, filepath.Join(stage, "spirv.predeploy", "kernel.spv"), "prior")

	checkSlots(t, LegacySlots(fak, stage), map[string]SlotKind{
		fak + ".bak":                          KindBak,
		filepath.Join(stage, "fak.predeploy"): KindPredeploy,
	})
	checkSlots(t, LegacySlots(spirv, stage), map[string]SlotKind{
		filepath.Join(share, "spirv.bak"):       KindBak,
		filepath.Join(stage, "spirv.predeploy"): KindPredeploy,
	})
}

// TestLegacySlotsInsideRouterCommitDir reports the previous candidate and never the newer one a
// failed update left behind.
func TestLegacySlotsInsideRouterCommitDir(t *testing.T) {
	root := t.TempDir()
	target := filepath.Join(root, commitA+"-"+commitB, "fak-server.exe")
	previous := filepath.Join(root, commitC+"-"+commitB, "fak-server.exe")
	failed := filepath.Join(root, commitB+"-"+commitA, "fak-server.exe")
	writeFile(t, target, "live")
	writeFile(t, previous, "prior")
	writeFile(t, failed, "never activated")
	setMod(t, previous, 2*time.Hour)
	setMod(t, target, time.Hour)
	slots := LegacySlots(target)
	if len(slots) != 1 || slots[0].Path != previous || slots[0].Kind != KindRouterCommit {
		t.Fatalf("LegacySlots = %+v, want only the previous candidate %s", slots, previous)
	}
}

func TestLegacySlotsNonExeTargetAndMissingDir(t *testing.T) {
	dir := t.TempDir()
	target := filepath.Join(dir, "fak")
	writeFile(t, target, "live")
	prior := filepath.Join(dir, "fak.self-update-prior")
	writeFile(t, prior, "prior")
	slots := LegacySlots(target)
	if len(slots) != 1 || slots[0].Path != prior || slots[0].Kind != KindSelfUpdatePrior {
		t.Fatalf("LegacySlots = %+v, want %s", slots, prior)
	}
	if slots := LegacySlots(filepath.Join(dir, "missing", "fak")); slots != nil {
		t.Fatalf("missing dir = %+v, want nil", slots)
	}
}
