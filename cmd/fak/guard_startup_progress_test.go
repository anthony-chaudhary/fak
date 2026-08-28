package main

import (
	"fmt"
	"strings"
	"sync"
	"testing"
	"time"
)

type lockedGuardProgressBuffer struct {
	mu         sync.Mutex
	b          strings.Builder
	firstWrite chan struct{}
	writeOnce  sync.Once
}

func (b *lockedGuardProgressBuffer) Write(p []byte) (int, error) {
	b.mu.Lock()
	defer b.mu.Unlock()
	n, err := b.b.Write(p)
	if b.firstWrite != nil {
		b.writeOnce.Do(func() { close(b.firstWrite) })
	}
	return n, err
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

// TestGuardDefaultTerminalBytesAreOnlyDelayedProgress composes the production banner decision,
// banner renderer, and delayed TTY progress surface. A healthy default must leave captured
// terminal bytes containing only the transient progress row: no report/profile/configuration,
// animation frame, or persistent settle content may leak through.
func TestGuardDefaultTerminalBytesAreOnlyDelayedProgress(t *testing.T) {
	mode, err := guardBannerModeDecision("auto", false, true, true)
	if err != nil {
		t.Fatal(err)
	}
	if mode != guardBannerProgress {
		t.Fatalf("auto mode = %q, want %q", mode, guardBannerProgress)
	}

	const report = "FULL STARTUP REPORT\nresponse profile: caveman\nwork profile: ponytail\nidentity/configuration\n"
	view := guardStartupView{
		bannerMode: mode,
		gwURL:      "http://127.0.0.1:9",
		command:    []string{"codex"},
	}
	out := lockedGuardProgressBuffer{firstWrite: make(chan struct{})}
	writeGuardStartupBanner(&out, view, report, true, false, "", 80)
	if got := out.String(); got != "" {
		t.Fatalf("default banner emitted before delayed progress: %q", got)
	}

	p := newGuardStartupProgress(&out, true, true, time.Millisecond)
	p.Phase("lease admission")
	select {
	case <-out.firstWrite:
	case <-time.After(time.Second):
		t.Fatal("delayed TTY progress did not render before timeout")
	}
	p.Phase("broker/preparing child")
	p.Started()

	got := out.String()
	if got == "" || !strings.Contains(got, guardLaunchClearLine+"fak guard · starting: ") {
		t.Fatalf("delayed TTY progress did not render: %q", got)
	}
	for _, forbidden := range []string{
		"FULL STARTUP REPORT", "response profile", "work profile", "identity/configuration",
		"fak info --startup", "gateway http", "kernel floor", "arming capability floor",
		"opening gateway", "linking kernel", string(guardLaunchFilledCell), string(guardLaunchEmptyCell),
	} {
		if strings.Contains(got, forbidden) {
			t.Errorf("default terminal bytes leaked %q: %q", forbidden, got)
		}
	}
	for _, fragment := range strings.Split(got, guardLaunchClearLine) {
		if fragment == "" {
			continue
		}
		if !strings.HasPrefix(fragment, "fak guard · starting: ") {
			t.Errorf("captured non-progress terminal bytes: %q in %q", fragment, got)
		}
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
