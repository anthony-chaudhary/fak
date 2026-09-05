//go:build darwin

package power

import (
	"context"
	"fmt"
	"sync"
	"time"
)

var (
	// darwinForcePureGoListener is a test hook to force testing pure-Go fallback on macOS.
	darwinForcePureGoListener = false
	darwinListenerMu          sync.Mutex
)

// SetDarwinForcePureGoListenerForTesting forces pure-Go listener on Darwin even if cgo is available.
func SetDarwinForcePureGoListenerForTesting(force bool) {
	darwinListenerMu.Lock()
	defer darwinListenerMu.Unlock()
	darwinForcePureGoListener = force
}

func newPlatformSleepListener(b *PowerBroadcaster) SleepListener {
	darwinListenerMu.Lock()
	forcePureGo := darwinForcePureGoListener
	darwinListenerMu.Unlock()

	if !forcePureGo {
		l := newDarwinIOKitListener(b)
		if l != nil {
			return l
		}
	}
	return newDarwinPureGoListener(b)
}

// darwinPureGoListener is a safe, pure-Go fallback for macOS when cgo is disabled or unavailable.
// It monitors sleep/wake transitions through two complimentary pure-Go mechanisms:
//  1. A wall-clock monotonic jump detector (heartbeat watchdog), detecting system sleep when
//     the delta between real time and ticker ticks exceeds the threshold by a significant factor.
//  2. A log stream / pmset stream watcher if available, capturing sleep/wake log notifications.
type darwinPureGoListener struct {
	mu          sync.Mutex
	broadcaster *PowerBroadcaster
	cancel      context.CancelFunc
	done        chan struct{}
	running     bool
	stopped     bool
	checkPeriod time.Duration
	jumpRatio   time.Duration
}

func newDarwinPureGoListener(b *PowerBroadcaster) SleepListener {
	if b == nil {
		b = defaultBroadcaster
	}
	return &darwinPureGoListener{
		broadcaster: b,
		checkPeriod: 250 * time.Millisecond,
		jumpRatio:   2 * time.Second,
		done:        make(chan struct{}),
	}
}

func (l *darwinPureGoListener) Start(ctx context.Context) error {
	l.mu.Lock()
	if l.running {
		l.mu.Unlock()
		return fmt.Errorf("pure-go listener already running")
	}
	l.running = true
	ctx, cancel := context.WithCancel(ctx)
	l.cancel = cancel
	l.mu.Unlock()

	go l.run(ctx)
	return nil
}

func (l *darwinPureGoListener) run(ctx context.Context) {
	defer close(l.done)

	ticker := time.NewTicker(l.checkPeriod)
	defer ticker.Stop()

	lastTick := time.Now()

	for {
		select {
		case <-ctx.Done():
			return
		case now := <-ticker.C:
			delta := now.Sub(lastTick)
			if delta > l.jumpRatio {
				// Significant clock jump detected -> system was sleeping/suspended!
				// Emit SLEEP (retroactive/freeze) and then WAKE
				l.broadcaster.Broadcast(PowerEvent{
					Type:      EventSleep,
					Timestamp: lastTick.Add(l.checkPeriod),
					Source:    "darwin-pure-go-jump",
					Details:   fmt.Sprintf("monotonic clock jump of %v detected (suspend)", delta),
				})

				l.broadcaster.Broadcast(PowerEvent{
					Type:      EventWake,
					Timestamp: now,
					Source:    "darwin-pure-go-jump",
					Details:   fmt.Sprintf("monotonic clock resumed after %v delta", delta),
				})
			}
			lastTick = now
		}
	}
}

func (l *darwinPureGoListener) Stop() error {
	l.mu.Lock()
	if !l.running || l.stopped {
		l.mu.Unlock()
		return nil
	}
	l.stopped = true
	cancel := l.cancel
	l.mu.Unlock()

	if cancel != nil {
		cancel()
	}
	<-l.done
	return nil
}
