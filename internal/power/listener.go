package power

import (
	"context"
	"sync"
)

// noOpSleepListener provides a safe fallback implementation when platform listeners are unavailable.
type noOpSleepListener struct {
	mu      sync.Mutex
	running bool
	stopped bool
}

func newNoOpSleepListener() SleepListener {
	return &noOpSleepListener{}
}

func (n *noOpSleepListener) Start(ctx context.Context) error {
	n.mu.Lock()
	defer n.mu.Unlock()
	n.running = true
	return nil
}

func (n *noOpSleepListener) Stop() error {
	n.mu.Lock()
	defer n.mu.Unlock()
	n.running = false
	n.stopped = true
	return nil
}

// NewSleepListener creates a platform-appropriate SleepListener that broadcasts power events
// to the specified broadcaster (or the package-level default broadcaster if nil).
func NewSleepListener(b *PowerBroadcaster) SleepListener {
	if b == nil {
		b = defaultBroadcaster
	}
	return newPlatformSleepListener(b)
}
