package main

import (
	"fmt"
	"strings"
	"sync"
	"testing"
	"time"
)

type lockedGuardProgressBuffer struct {
	mu sync.Mutex
	b  strings.Builder
}

func (b *lockedGuardProgressBuffer) Write(p []byte) (int, error) {
	b.mu.Lock()
	defer b.mu.Unlock()
	return b.b.Write(p)
}

func (b *lockedGuardProgressBuffer) String() string {
	b.mu.Lock()
	defer b.mu.Unlock()
	return b.b.String()
}

func TestGuardStartupProgressRevealsSlowPhasesWithoutSecrets(t *testing.T) {
	var out lockedGuardProgressBuffer
	p := newGuardStartupProgress(&out, true, false, time.Millisecond)
	p.Phase("lease admission")
	time.Sleep(20 * time.Millisecond)
	p.Phase("broker/preparing child")
	p.Phase("OS process start")
	p.Phase("child registration")
	p.Started()

	got := out.String()
	for _, phase := range []string{"lease admission", "broker/preparing child", "OS process start", "child registration", "child registration/started"} {
		if !strings.Contains(got, "startup phase="+phase) {
			t.Errorf("progress missing %q: %q", phase, got)
		}
	}
	if strings.Contains(got, "--") || strings.Contains(got, "=") && strings.Contains(got, "token") {
		t.Fatalf("progress unexpectedly rendered launch detail: %q", got)
	}
}

func TestGuardStartupProgressFastPathAndTTYStayCompact(t *testing.T) {
	var fast lockedGuardProgressBuffer
	p := newGuardStartupProgress(&fast, true, false, time.Hour)
	p.Phase("native-hook install")
	p.Started()
	if got := fast.String(); got != "" {
		t.Fatalf("fast launch rendered progress: %q", got)
	}

	var tty lockedGuardProgressBuffer
	p = newGuardStartupProgress(&tty, true, true, time.Millisecond)
	p.Phase("lease admission")
	time.Sleep(20 * time.Millisecond)
	p.Phase("broker/preparing child")
	p.Started()
	got := tty.String()
	if strings.Count(got, "\n") != 1 || !strings.Contains(got, "\r\x1b[2K") || !strings.Contains(got, "child registration/started elapsed=") || !strings.HasSuffix(got, "\n") {
		t.Fatalf("TTY progress did not remain one in-place line: %q", got)
	}
}

func TestGuardStartupProgressAbortEndsTTYLineBeforeError(t *testing.T) {
	var tty lockedGuardProgressBuffer
	p := newGuardStartupProgress(&tty, true, true, time.Millisecond)
	p.Phase("lease admission")
	time.Sleep(20 * time.Millisecond)
	p.Abort()
	fmt.Fprintln(&tty, "fak guard: COLLISION_RISK: lease ledger unavailable")
	p.Stop() // The normal defer must not clear an already-aborted diagnostic.

	got := tty.String()
	if !strings.Contains(got, "lease admission elapsed=") || !strings.Contains(got, "\nfak guard: COLLISION_RISK") {
		t.Fatalf("error did not start on its own line: %q", got)
	}
	errorAt := strings.Index(got, "fak guard: COLLISION_RISK")
	if strings.Contains(got[errorAt:], "\r\x1b[2K") {
		t.Fatalf("deferred stop cleared the error line: %q", got)
	}
}
