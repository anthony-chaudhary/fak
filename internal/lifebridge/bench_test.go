package lifebridge

import (
	"testing"

	"github.com/anthony-chaudhary/fak/internal/loopmgr"
	"github.com/anthony-chaudhary/fak/internal/session"
)

// BenchmarkLifeBridge exercises bidirectional state conversions in a tight loop.
func BenchmarkLifeBridge(b *testing.B) {
	runs := []session.RunState{
		session.Running,
		session.Paused,
		session.Draining,
		session.Stopped,
		session.Throttled,
	}
	loops := []loopmgr.LoopState{
		loopmgr.StateRunning,
		loopmgr.StatePaused,
		loopmgr.StateDraining,
		loopmgr.StateStopped,
		loopmgr.StateArmed,
		loopmgr.StateDisabled,
	}

	b.ResetTimer()
	b.ReportAllocs()

	for i := 0; i < b.N; i++ {
		for _, rs := range runs {
			_, _ = RunToLoop(rs)
		}
		for _, ls := range loops {
			_, _ = LoopToRun(ls)
		}
	}
}
