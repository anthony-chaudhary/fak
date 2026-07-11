package toolprocgate

import "testing"

// fault is a tiny constructor for a console-fault row at a time/surface/session.
func fault(atMS int64, surface, session string) ConsoleFaultEvent {
	return ConsoleFaultEvent{Class: ConsoleHostFailFast, AtMS: atMS, Surface: surface, Session: session}
}

// TestDecideContainment pins the protection ladder: the widest-blast-radius
// condition that holds wins, and a clean history admits.
func TestDecideContainment(t *testing.T) {
	pol := DefaultContainmentPolicy()
	now := int64(1_000_000)

	t.Run("clean history admits", func(t *testing.T) {
		dec := DecideContainment(pol, nil, ContainmentRequest{Surface: "conpty-A", LiveOnSurface: 0, NowMS: now})
		if !dec.Admit || dec.Verdict != ContainAdmit {
			t.Fatalf("want ADMIT, got %q (admit=%v)", dec.Verdict, dec.Admit)
		}
	})

	t.Run("co-location cap bounds blast radius", func(t *testing.T) {
		dec := DecideContainment(pol, nil, ContainmentRequest{Surface: "conpty-A", LiveOnSurface: 3, NowMS: now})
		if dec.Admit || dec.Verdict != ContainRefuseColocation {
			t.Fatalf("want REFUSE_COLOCATION, got %q (admit=%v)", dec.Verdict, dec.Admit)
		}
	})

	t.Run("repeated surface faults quarantine the surface", func(t *testing.T) {
		faults := []ConsoleFaultEvent{
			fault(now-1000, "conpty-A", "s1"),
			fault(now-2000, "conpty-A", "s1"),
		}
		dec := DecideContainment(pol, faults, ContainmentRequest{Surface: "conpty-A", LiveOnSurface: 0, NowMS: now})
		if dec.Admit || dec.Verdict != ContainQuarantineSurface {
			t.Fatalf("want QUARANTINE_SURFACE, got %q (admit=%v)", dec.Verdict, dec.Admit)
		}
		if dec.SurfaceFaults != 2 {
			t.Errorf("surface_faults = %d, want 2", dec.SurfaceFaults)
		}
	})

	t.Run("cross-session storm opens the fleet breaker (most severe wins)", func(t *testing.T) {
		// 5 faults across 3 sessions AND 2 on the requested surface: breaker must
		// win over the surface-quarantine that also holds.
		faults := []ConsoleFaultEvent{
			fault(now-1000, "conpty-A", "s1"),
			fault(now-1100, "conpty-A", "s1"),
			fault(now-1200, "conpty-B", "s2"),
			fault(now-1300, "conpty-C", "s3"),
			fault(now-1400, "conpty-D", "s3"),
		}
		dec := DecideContainment(pol, faults, ContainmentRequest{Surface: "conpty-A", LiveOnSurface: 0, NowMS: now})
		if dec.Admit || dec.Verdict != ContainBreakerOpen {
			t.Fatalf("want BREAKER_OPEN, got %q (admit=%v)", dec.Verdict, dec.Admit)
		}
		if dec.WindowSessions != 3 || dec.WindowFaults != 5 {
			t.Errorf("evidence = %d faults / %d sessions, want 5/3", dec.WindowFaults, dec.WindowSessions)
		}
	})

	t.Run("single-session storm does NOT open the breaker", func(t *testing.T) {
		// 5 faults but all one session: not a cross-session cascade. It should
		// fall through to surface quarantine (2 on conpty-A), not the breaker.
		faults := []ConsoleFaultEvent{
			fault(now-1000, "conpty-A", "s1"),
			fault(now-1100, "conpty-A", "s1"),
			fault(now-1200, "conpty-B", "s1"),
			fault(now-1300, "conpty-C", "s1"),
			fault(now-1400, "conpty-D", "s1"),
		}
		dec := DecideContainment(pol, faults, ContainmentRequest{Surface: "conpty-A", LiveOnSurface: 0, NowMS: now})
		if dec.Verdict != ContainQuarantineSurface {
			t.Fatalf("want QUARANTINE_SURFACE (single session), got %q", dec.Verdict)
		}
	})

	t.Run("stale faults outside the window are cleared", func(t *testing.T) {
		// Two faults on the surface but both older than the 5-min window.
		faults := []ConsoleFaultEvent{
			fault(now-10*60*1000, "conpty-A", "s1"),
			fault(now-11*60*1000, "conpty-A", "s1"),
		}
		dec := DecideContainment(pol, faults, ContainmentRequest{Surface: "conpty-A", LiveOnSurface: 0, NowMS: now})
		if !dec.Admit {
			t.Fatalf("stale faults should clear -> ADMIT, got %q", dec.Verdict)
		}
		if dec.WindowFaults != 0 {
			t.Errorf("window_faults = %d, want 0 (stale)", dec.WindowFaults)
		}
	})

	t.Run("unknown clock counts all faults (fail-protective)", func(t *testing.T) {
		// NowMS=0 disables the window: old faults still count, so a re-crash loop
		// is caught even without a clock.
		faults := []ConsoleFaultEvent{
			fault(1, "conpty-A", "s1"),
			fault(2, "conpty-A", "s1"),
		}
		dec := DecideContainment(pol, faults, ContainmentRequest{Surface: "conpty-A", NowMS: 0})
		if dec.Verdict != ContainQuarantineSurface {
			t.Fatalf("unknown clock should still quarantine, got %q", dec.Verdict)
		}
	})
}

// TestSortSurfaceLoads pins the blast-radius map ordering.
func TestSortSurfaceLoads(t *testing.T) {
	got := SortSurfaceLoads([]ContainmentSurfaceLoad{
		{"conpty-B", 1}, {"conpty-A", 3}, {"conpty-C", 3},
	})
	if got[0].Surface != "conpty-A" || got[0].Live != 3 {
		t.Errorf("most-concentrated surface should sort first, got %+v", got[0])
	}
	if got[2].Surface != "conpty-B" {
		t.Errorf("least-loaded should sort last, got %+v", got[2])
	}
}
