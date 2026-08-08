package harnessres

import (
	"testing"
	"time"
)

// TestTurnFragmentRendersKernelAxes pins the compact per-turn fragment (#2050): the exact
// glanceable shape the guard appends to its `--debug-stats` economy line.
func TestTurnFragmentRendersKernelAxes(t *testing.T) {
	s := Snapshot{
		Elapsed: 10 * time.Second,
		Kernel: Half{
			CPUUser: time.Second, CPUSys: 200 * time.Millisecond, HaveCPU: true,
			RSSBytes: 64 << 20, HaveRSS: true,
			IOReadBytes: 5 << 20, IOWriteBytes: 300 << 10, HaveIO: true,
		},
	}
	const want = "res: cpu=12% rss=64MB io=5.0MBr/0.3MBw"
	if got := s.TurnFragment(); got != want {
		t.Errorf("TurnFragment() = %q, want %q", got, want)
	}
}

// TestTurnFragmentOmitsUnreadableAxes is the package's standing rule applied to this
// renderer: an axis the platform could not read is ABSENT, never a fabricated 0. A
// snapshot with nothing observed renders "" so the caller appends nothing at all.
func TestTurnFragmentOmitsUnreadableAxes(t *testing.T) {
	cases := []struct {
		name string
		snap Snapshot
		want string
	}{
		{"nothing observed", Snapshot{Elapsed: time.Second}, ""},
		{"zero elapsed drops cpu, keeps rss", Snapshot{
			Kernel: Half{CPUUser: time.Second, HaveCPU: true, RSSBytes: 32 << 20, HaveRSS: true},
		}, "res: rss=32MB"},
		{"io only", Snapshot{
			Kernel: Half{IOReadBytes: 2 << 30, IOWriteBytes: 0, HaveIO: true},
		}, "res: io=2.0GBr/0.0MBw"},
		{"cpu only", Snapshot{
			Elapsed: 4 * time.Second,
			Kernel:  Half{CPUSys: 2 * time.Second, HaveCPU: true},
		}, "res: cpu=50%"},
		// An observed ZERO is not the same as an unreadable axis: the presence bit is set,
		// so the axis renders rather than vanishing.
		{"observed zero rss still renders", Snapshot{
			Kernel: Half{RSSBytes: 0, HaveRSS: true},
		}, "res: rss=0.0MB"},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			if got := tc.snap.TurnFragment(); got != tc.want {
				t.Errorf("TurnFragment() = %q, want %q", got, tc.want)
			}
		})
	}
}

// TestTurnFragmentIgnoresAgentHalf is the honesty guard on the shape choice: the agent
// half is folded from the child's EXIT state, so mid-session it is empty. A live turn
// must never borrow it — a fragment built only from agent numbers stays silent.
func TestTurnFragmentIgnoresAgentHalf(t *testing.T) {
	s := Snapshot{
		Elapsed: 10 * time.Second,
		Agent: Half{
			CPUUser: 5 * time.Second, HaveCPU: true,
			RSSBytes: 512 << 20, HaveRSS: true,
			IOReadBytes: 1 << 20, HaveIO: true,
		},
	}
	if got := s.TurnFragment(); got != "" {
		t.Errorf("TurnFragment() with only the agent half = %q, want %q", got, "")
	}
}

// TestTurnFragmentHasNoOuterSpace pins the contract the caller relies on: the caller owns
// the separator, so the fragment must not smuggle its own leading/trailing space.
func TestTurnFragmentHasNoOuterSpace(t *testing.T) {
	s := Snapshot{Elapsed: time.Second, Kernel: Half{RSSBytes: 1 << 20, HaveRSS: true}}
	got := s.TurnFragment()
	if got == "" {
		t.Fatal("TurnFragment() = \"\", want a rendered fragment")
	}
	if got[0] == ' ' || got[len(got)-1] == ' ' {
		t.Errorf("TurnFragment() = %q, want no leading/trailing space", got)
	}
}
