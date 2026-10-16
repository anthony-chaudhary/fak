package devcmd

import (
	"bytes"
	"context"
	"errors"
	"fmt"
	"io"
	"net"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"sync"
	"testing"
	"time"
)

// companionLiveSkipReason reports why the companion live proxy tests must skip,
// or "" when the sibling checkout is buildable and complete. Their precondition
// is a *current, buildable* sibling, not merely a present one: a sibling left
// missing a module or its tickets dir would otherwise fail the package red even
// though the fak under test is fine. Probes the gate build (a `go list` cannot
// detect a missing module; `go build` can) and the tickets directory. See
// anthony-chaudhary/fak#13840.
func companionLiveSkipReason(t *testing.T, privRoot string) string {
	t.Helper()
	if privRoot == "" {
		return "companion fak-private not present"
	}
	if _, err := os.Stat(filepath.Join(privRoot, "docs", "tickets")); err != nil {
		return "companion fak-private lacks docs/tickets"
	}
	deadline := time.Now().Add(30 * time.Second)
	if outer, ok := t.Deadline(); ok && outer.Add(-2*time.Second).Before(deadline) {
		deadline = outer.Add(-2 * time.Second)
	}
	ctx, cancel := context.WithDeadline(context.Background(), deadline)
	defer cancel()
	// Waiting for the in-process probe slot consumes the same deadline.
	select {
	case companionProbeSlot <- struct{}{}:
		defer func() { <-companionProbeSlot }()
	case <-ctx.Done():
		return fmt.Sprintf("companion build probe admission expired: %v", ctx.Err())
	}
	out := filepath.Join(t.TempDir(), "fak-boundary-probe")
	cmd := exec.CommandContext(ctx, "go", "-C", privRoot, "build", "-p=1", "-o", out, "./cmd/fak-boundary")
	cmd.Env = companionProbeEnv(filepath.Dir(out))
	detail, err := runCompanionProbe(cmd)
	if err != nil {
		if ctx.Err() != nil {
			return fmt.Sprintf("companion build probe deadline exceeded; buildability unverified: %v", ctx.Err())
		}
		return fmt.Sprintf("companion fak-private cmd/fak-boundary does not build: %v: %s", err, strings.TrimSpace(detail))
	}
	return ""
}

func TestCompanionBridge_RootResolution(t *testing.T) {
	// 1. Explicit FAK_PRIVATE_ROOT environment variable pointing to a valid dir
	tempPrivate := t.TempDir()
	if err := os.WriteFile(filepath.Join(tempPrivate, "go.mod"), []byte("module github.com/anthony-chaudhary/fak-private\n"), 0644); err != nil {
		t.Fatal(err)
	}

	t.Setenv("FAK_PRIVATE_ROOT", tempPrivate)
	resolved := ResolveCompanionRoot("")
	if resolved != tempPrivate {
		t.Errorf("expected %q, got %q", tempPrivate, resolved)
	}

	// 2. Non-existent FAK_PRIVATE_ROOT should not be accepted
	t.Setenv("FAK_PRIVATE_ROOT", filepath.Join(tempPrivate, "does_not_exist"))
	resolvedBad := ResolveCompanionRoot(t.TempDir())
	if resolvedBad == filepath.Join(tempPrivate, "does_not_exist") {
		t.Errorf("expected non-existent FAK_PRIVATE_ROOT to be rejected, got %q", resolvedBad)
	}

	// 3. Sibling discovery
	tempParent := t.TempDir()
	fakeFak := filepath.Join(tempParent, "fak")
	fakePriv := filepath.Join(tempParent, "fak-private")
	if err := os.MkdirAll(fakeFak, 0755); err != nil {
		t.Fatal(err)
	}
	if err := os.MkdirAll(fakePriv, 0755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(fakePriv, "go.mod"), []byte("module github.com/anthony-chaudhary/fak-private\n"), 0644); err != nil {
		t.Fatal(err)
	}

	t.Setenv("FAK_PRIVATE_ROOT", "")
	resolvedSibling := ResolveCompanionRoot(fakeFak)
	if filepath.Clean(resolvedSibling) != filepath.Clean(fakePriv) {
		t.Errorf("expected sibling %q, got %q", fakePriv, resolvedSibling)
	}
}

func TestCompanionBridge_MissingCompanion(t *testing.T) {
	// Force companion to be missing
	origOverride := overrideCompanionRoots
	overrideCompanionRoots = func() (string, string) {
		return t.TempDir(), ""
	}
	defer func() {
		overrideCompanionRoots = origOverride
	}()

	// 1. RunCompanionGate without --help
	var stdout, stderr bytes.Buffer
	code := RunCompanionGate(&stdout, &stderr, []string{"check", "--staged"})
	if code != 1 {
		t.Errorf("expected exit code 1 for missing companion, got %d", code)
	}
	if !strings.Contains(stderr.String(), "fak-dev gate: companion repository (fak-private) not found. Set FAK_PRIVATE_ROOT or clone fak-private as sibling.") {
		t.Errorf("unexpected stderr: %s", stderr.String())
	}

	// 2. RunCompanionGate with --help (should exit 0 with notice and help)
	stdout.Reset()
	stderr.Reset()
	code = RunCompanionGate(&stdout, &stderr, []string{"--help"})
	if code != 0 {
		t.Errorf("expected exit code 0 for --help on missing companion, got %d", code)
	}
	if !strings.Contains(stdout.String(), "fak-dev gate: companion repository (fak-private) not found") {
		t.Errorf("expected notice in stdout for --help, got: %s", stdout.String())
	}
	if !strings.Contains(stdout.String(), "Usage: fak-dev gate") {
		t.Errorf("expected usage in stdout for --help, got: %s", stdout.String())
	}

	// 3. RunCompanionProvenance without --help
	stdout.Reset()
	stderr.Reset()
	code = RunCompanionProvenance(&stdout, &stderr, []string{"audit", "--commit", "HEAD"})
	if code != 1 {
		t.Errorf("expected exit code 1 for missing companion, got %d", code)
	}
	if !strings.Contains(stderr.String(), "fak-dev provenance: companion repository (fak-private) not found. Set FAK_PRIVATE_ROOT or clone fak-private as sibling.") {
		t.Errorf("unexpected stderr: %s", stderr.String())
	}

	// 4. RunCompanionProvenance with --help
	stdout.Reset()
	stderr.Reset()
	code = RunCompanionProvenance(&stdout, &stderr, []string{"--help"})
	if code != 0 {
		t.Errorf("expected exit code 0 for --help on missing companion, got %d", code)
	}
	if !strings.Contains(stdout.String(), "fak-dev provenance: companion repository (fak-private) not found") {
		t.Errorf("expected notice in stdout for --help, got: %s", stdout.String())
	}
	if !strings.Contains(stdout.String(), "Usage: fak-dev provenance") {
		t.Errorf("expected usage in stdout for --help, got: %s", stdout.String())
	}

	// 5. RunCompanionAuditLeak without --help
	stdout.Reset()
	stderr.Reset()
	code = RunCompanionAuditLeak(&stdout, &stderr, []string{"--staged"})
	if code != 1 {
		t.Errorf("expected exit code 1 for missing companion, got %d", code)
	}
	if !strings.Contains(stderr.String(), "fak-dev audit-leak: companion repository (fak-private) not found. Set FAK_PRIVATE_ROOT or clone fak-private as sibling.") {
		t.Errorf("unexpected stderr: %s", stderr.String())
	}

	// 6. RunCompanionAuditLeak with --help
	stdout.Reset()
	stderr.Reset()
	code = RunCompanionAuditLeak(&stdout, &stderr, []string{"--help"})
	if code != 0 {
		t.Errorf("expected exit code 0 for --help on missing companion, got %d", code)
	}
	if !strings.Contains(stdout.String(), "fak-dev audit-leak: companion repository (fak-private) not found") {
		t.Errorf("expected notice in stdout for --help, got: %s", stdout.String())
	}
	if !strings.Contains(stdout.String(), "Usage: fak-dev audit-leak") {
		t.Errorf("expected usage in stdout for --help, got: %s", stdout.String())
	}

	// 7. RunCompanionStrix without --help
	stdout.Reset()
	stderr.Reset()
	code = RunCompanionStrix(&stdout, &stderr, []string{"key"})
	if code != 1 {
		t.Errorf("expected exit code 1 for missing companion, got %d", code)
	}
	if !strings.Contains(stderr.String(), "fak-dev strix: companion repository (fak-private) not found. Set FAK_PRIVATE_ROOT or clone fak-private as sibling.") {
		t.Errorf("unexpected stderr: %s", stderr.String())
	}

	// 8. RunCompanionStrix with --help
	stdout.Reset()
	stderr.Reset()
	code = RunCompanionStrix(&stdout, &stderr, []string{"--help"})
	if code != 0 {
		t.Errorf("expected exit code 0 for --help on missing companion, got %d", code)
	}
	if !strings.Contains(stdout.String(), "fak-dev strix: companion repository (fak-private) not found") {
		t.Errorf("expected notice in stdout for --help, got: %s", stdout.String())
	}
	if !strings.Contains(stdout.String(), "Usage: fak-dev strix") {
		t.Errorf("expected usage in stdout for --help, got: %s", stdout.String())
	}
}

func TestCompanionBridge_InvocationHelpers(t *testing.T) {
	if !isBoundaryCheckInvocation([]string{"check"}) {
		t.Error("expected 'check' to be a boundary check invocation")
	}
	if !isBoundaryCheckInvocation([]string{"check", "--staged"}) {
		t.Error("expected 'check --staged' to be a boundary check invocation")
	}
	if !isBoundaryCheckInvocation([]string{"--staged"}) {
		t.Error("expected '--staged' to be a boundary check invocation")
	}
	if isBoundaryCheckInvocation([]string{"--help"}) {
		t.Error("expected '--help' NOT to be a boundary check invocation")
	}
	if isBoundaryCheckInvocation([]string{"-h"}) {
		t.Error("expected '-h' NOT to be a boundary check invocation")
	}
	if isBoundaryCheckInvocation([]string{"query", "concept"}) {
		t.Error("expected 'query' NOT to be a boundary check invocation")
	}
	if isBoundaryCheckInvocation([]string{"explain", "path"}) {
		t.Error("expected 'explain' NOT to be a boundary check invocation")
	}
	if isBoundaryCheckInvocation(nil) {
		t.Error("expected empty args NOT to be a boundary check invocation")
	}

	if !hasFlag([]string{"--foo", "--bar"}, "--foo") {
		t.Error("expected hasFlag to find --foo")
	}
	if !hasFlag([]string{"--dir=test"}, "--dir") {
		t.Error("expected hasFlag to find --dir in --dir=test")
	}
	if hasFlag([]string{"--foo"}, "--bar") {
		t.Error("expected hasFlag not to find --bar")
	}
}

func TestCompanionBridge_Live(t *testing.T) {
	_, privRoot := ResolveCompanionRoots()
	if reason := companionLiveSkipReason(t, privRoot); reason != "" {
		t.Skip("skipping live companion bridge tests: " + reason)
	}

	// 1. Gate --help
	var stdout, stderr bytes.Buffer
	code := runCompanionLiveGate(t, &stdout, &stderr, []string{"--help"})
	if code != 0 {
		t.Fatalf("gate --help failed with code %d. stderr: %s", code, stderr.String())
	}
	if !strings.Contains(stdout.String(), "fak-boundary") && !strings.Contains(stdout.String(), "Subcommands:") {
		t.Errorf("unexpected gate --help output: %s", stdout.String())
	}

	// 2. Gate check --staged
	stdout.Reset()
	stderr.Reset()
	code = runCompanionLiveGate(t, &stdout, &stderr, []string{"check", "--staged"})
	if code != 0 {
		t.Fatalf("gate check --staged failed with code %d. stderr: %s", code, stderr.String())
	}
	if !strings.Contains(stdout.String(), "5-Gate Import Encapsulation") {
		t.Errorf("unexpected gate check output: %s", stdout.String())
	}

	// 3. Provenance --help
	stdout.Reset()
	stderr.Reset()
	code = RunCompanionProvenance(&stdout, &stderr, []string{"--help"})
	if code != 0 {
		t.Fatalf("provenance --help failed with code %d. stderr: %s", code, stderr.String())
	}
	if !strings.Contains(stdout.String(), "fak-sync provenance") && !strings.Contains(stdout.String(), "Subcommands:") {
		t.Errorf("unexpected provenance --help output: %s", stdout.String())
	}

	// 4. Provenance audit --commit HEAD
	stdout.Reset()
	stderr.Reset()
	code = RunCompanionProvenance(&stdout, &stderr, []string{"audit", "--commit", "HEAD"})
	if code != 0 {
		t.Fatalf("provenance audit failed with code %d. stderr: %s", code, stderr.String())
	}
	if !strings.Contains(stdout.String(), "Context Provenance Audit Report") {
		t.Errorf("unexpected provenance audit output: %s", stdout.String())
	}

	// 5. Audit-Leak --help
	stdout.Reset()
	stderr.Reset()
	code = RunCompanionAuditLeak(&stdout, &stderr, []string{"--help"})
	if code != 0 {
		t.Fatalf("audit-leak --help failed with code %d. stderr: %s", code, stderr.String())
	}
	if !strings.Contains(stdout.String(), "Usage: fak-dev audit-leak") {
		t.Errorf("unexpected audit-leak --help output: %s", stdout.String())
	}

	// 6. Audit-Leak --staged
	stdout.Reset()
	stderr.Reset()
	code = RunCompanionAuditLeak(&stdout, &stderr, []string{"--staged"})
	if code != 0 {
		t.Fatalf("audit-leak --staged failed with code %d. stderr: %s", code, stderr.String())
	}

	// 7. Strix --help
	stdout.Reset()
	stderr.Reset()
	code = RunCompanionStrix(&stdout, &stderr, []string{"--help"})
	if code != 0 {
		t.Fatalf("strix --help failed with code %d. stderr: %s", code, stderr.String())
	}
	if !strings.Contains(stdout.String(), "Usage: fak-dev strix") {
		t.Errorf("unexpected strix --help output: %s", stdout.String())
	}
}

const companionProbeOutputLimit = 300

var companionProbeSlot = make(chan struct{}, 1)

func companionProbeEnv(tmp string) []string {
	// Native helper TestMain and Go build temporary files belong to this probe.
	return append(os.Environ(), "GOMAXPROCS=1", "TMPDIR="+tmp, "TMP="+tmp, "TEMP="+tmp)
}

// The output cap applies while copying, not after an unbounded CombinedOutput.
// Both streams share the same writer; the mutex also covers distinct copy loops.
type companionProbeOutput struct {
	mu   sync.Mutex
	data []byte
}

func (w *companionProbeOutput) Write(p []byte) (int, error) {
	w.mu.Lock()
	defer w.mu.Unlock()
	n := len(p)
	if room := companionProbeOutputLimit - len(w.data); room > 0 {
		w.data = append(w.data, p[:min(room, n)]...)
	}
	return n, nil
}

func runCompanionProbe(cmd *exec.Cmd) (string, error) {
	var output companionProbeOutput
	cmd.Stdout, cmd.Stderr = &output, &output
	// Bound inherited-pipe drain after the direct child exits or is cancelled.
	cmd.WaitDelay = time.Second
	err := runOwnedCompanionProbe(cmd)
	output.mu.Lock()
	defer output.mu.Unlock()
	return string(output.data), err
}

// Dispatch private native fixtures before Git-isolating TestMain. Ordinary test
// invocations do not match these modes and retain the normal TestMain contract.
func init() {
	mode := ""
	if len(os.Args) >= 4 {
		mode = os.Args[len(os.Args)-3]
	}
	switch mode {
	case "companion-probe-descendant", "companion-probe-owner", "companion-probe-leader", "companion-probe-owner-target":
		if err := companionProbeFixture(mode, os.Args[len(os.Args)-2], os.Args[len(os.Args)-1]); err != nil {
			fmt.Fprintln(os.Stderr, err)
			os.Exit(1)
		}
		os.Exit(0)
	}
	if len(os.Args) > 1 && os.Args[len(os.Args)-1] == "companion-probe-child" {
		fmt.Fprint(os.Stdout, strings.Repeat("x", 4096))
		time.Sleep(time.Minute)
		os.Exit(0)
	}
}

func companionProbeFixture(mode, address, ready string) error {
	switch mode {
	case "companion-probe-descendant":
		conn, err := net.DialTimeout("tcp", address, 3*time.Second)
		if err != nil {
			return err
		}
		defer conn.Close()
		if err := conn.SetReadDeadline(time.Now().Add(10 * time.Second)); err != nil {
			return err
		}
		if err := os.WriteFile(ready, []byte("ready"), 0600); err != nil {
			return err
		}
		_, err = io.Copy(io.Discard, conn)
		return err
	case "companion-probe-owner":
		ctx, cancel := context.WithTimeout(context.Background(), 8*time.Second)
		defer cancel()
		probe := exec.CommandContext(ctx, os.Args[0], "-test.run=^TestCompanionProbeBounds$", "--", "companion-probe-owner-target", address, ready)
		probe.Env = os.Environ()
		_, err := runCompanionProbe(probe)
		return err
	default:
		child := exec.Command(os.Args[0], "-test.run=^TestCompanionProbeBounds$", "--", "companion-probe-descendant", address, ready)
		child.Stdout, child.Stderr = os.Stdout, os.Stderr
		if err := child.Start(); err != nil {
			return err
		}
		// The descendant inherits the probe's pipes and process ownership.
		for until := time.Now().Add(3 * time.Second); time.Now().Before(until); {
			if _, err := os.Stat(ready); err == nil {
				if mode == "companion-probe-owner-target" {
					time.Sleep(8 * time.Second)
				}
				return nil
			}
			time.Sleep(10 * time.Millisecond)
		}
		return errors.New("descendant readiness timeout")
	}
}

// fak-test:runtime fast est=4s lane=default
func TestCompanionProbeBounds(t *testing.T) {
	// Re-exec the native test binary; this regression never launches a compiler.
	var out companionProbeOutput
	if n, err := out.Write([]byte(strings.Repeat("x", 4096))); n != 4096 || err != nil || len(out.data) != companionProbeOutputLimit {
		t.Fatalf("bounded writer: n=%d err=%v bytes=%d", n, err, len(out.data))
	}
	ctx, cancel := context.WithTimeout(context.Background(), time.Second)
	defer cancel()
	cmd := exec.CommandContext(ctx, os.Args[0], "-test.run=^TestCompanionProbeBounds$", "--", "companion-probe-child")
	cmd.Env = companionProbeEnv(t.TempDir())
	started := time.Now()
	detail, err := runCompanionProbe(cmd)
	if err == nil || !errors.Is(ctx.Err(), context.DeadlineExceeded) {
		t.Fatalf("probe did not expire: err=%v context=%v", err, ctx.Err())
	}
	if elapsed := time.Since(started); elapsed > 5*time.Second {
		t.Fatalf("probe cancellation took %s", elapsed)
	}
	if len(detail) != companionProbeOutputLimit {
		t.Fatalf("probe child did not exercise capped output: %d bytes", len(detail))
	}
	if cmd.ProcessState == nil {
		t.Fatal("probe returned without waiting for its direct child")
	}

	listener, err := net.ListenTCP("tcp", &net.TCPAddr{IP: net.IPv4(127, 0, 0, 1)})
	if err != nil {
		t.Fatal(err)
	}
	defer listener.Close()
	if err := listener.SetDeadline(time.Now().Add(5 * time.Second)); err != nil {
		t.Fatal(err)
	}
	treeCtx, treeCancel := context.WithTimeout(context.Background(), 8*time.Second)
	defer treeCancel()
	leader := exec.CommandContext(treeCtx, os.Args[0], "-test.run=^TestCompanionProbeBounds$", "--", "companion-probe-leader", listener.Addr().String(), filepath.Join(t.TempDir(), "ready"))
	leader.Env = companionProbeEnv(t.TempDir())
	done := make(chan error, 1)
	go func() { _, err := runCompanionProbe(leader); done <- err }()
	conn, err := listener.AcceptTCP()
	if err != nil {
		treeCancel()
		<-done
		t.Fatal(err)
	}
	defer conn.Close()
	if err := <-done; !errors.Is(err, exec.ErrWaitDelay) {
		t.Fatalf("early leader exit did not exercise inherited-pipe timeout: %v", err)
	}
	if leader.ProcessState == nil {
		t.Fatal("early leader was not waited")
	}
	if err := conn.SetReadDeadline(time.Now().Add(time.Second)); err != nil {
		t.Fatal(err)
	}
	var one [1]byte
	if _, err := conn.Read(one[:]); !companionProbeSocketClosed(err) {
		t.Fatalf("owned descendant survived probe return: %v", err)
	}

	// Kill only the exact owner child. Its anchor is outside this process group;
	// EOF therefore witnesses the parent-liveness pipe, not outer group cleanup.
	ownerListener, err := net.ListenTCP("tcp", &net.TCPAddr{IP: net.IPv4(127, 0, 0, 1)})
	if err != nil {
		t.Fatal(err)
	}
	defer ownerListener.Close()
	if err := ownerListener.SetDeadline(time.Now().Add(5 * time.Second)); err != nil {
		t.Fatal(err)
	}
	ownerCtx, ownerCancel := context.WithTimeout(context.Background(), 8*time.Second)
	defer ownerCancel()
	owner := exec.CommandContext(ownerCtx, os.Args[0], "-test.run=^TestCompanionProbeBounds$", "--", "companion-probe-owner", ownerListener.Addr().String(), filepath.Join(t.TempDir(), "ready"))
	owner.Env = companionProbeEnv(t.TempDir())
	if err := owner.Start(); err != nil {
		t.Fatal(err)
	}
	ownerDone := make(chan error, 1)
	go func() { ownerDone <- owner.Wait() }()
	ownerConn, err := ownerListener.AcceptTCP()
	if err != nil {
		_ = owner.Process.Kill()
		<-ownerDone
		t.Fatal(err)
	}
	defer ownerConn.Close()
	killErr := owner.Process.Kill()
	waitErr := <-ownerDone
	if killErr != nil || waitErr == nil || owner.ProcessState == nil {
		t.Fatalf("owner termination was not witnessed: kill=%v wait=%v", killErr, waitErr)
	}
	if err := ownerConn.SetReadDeadline(time.Now().Add(time.Second)); err != nil {
		t.Fatal(err)
	}
	if _, err := ownerConn.Read(one[:]); !companionProbeSocketClosed(err) {
		t.Fatalf("descendant survived parent-owner termination: %v", err)
	}
}
