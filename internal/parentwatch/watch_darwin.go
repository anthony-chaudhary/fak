//go:build darwin

package parentwatch

import (
	"context"
	"errors"
	"sync"

	"golang.org/x/sys/unix"
)

func watch(ctx context.Context, cancel context.CancelFunc, id identity) func() {
	// A known creation time arms the PID-reuse guard, which the kqueue NOTE_EXIT
	// registration cannot observe (it fires only on this PID's exit). Fall back
	// to polling so a reused PID still cancels; StartTime is unavailable on
	// darwin today, so this path is exercised only if that changes.
	if id.startOK {
		return pollWatch(ctx, cancel, id)
	}

	kq, err := unix.Kqueue()
	if err != nil {
		return pollWatch(ctx, cancel, id)
	}

	var change unix.Kevent_t
	unix.SetKevent(&change, id.pid, unix.EVFILT_PROC, unix.EV_ADD|unix.EV_ENABLE)
	change.Fflags = unix.NOTE_EXIT

	n, err := keventRegister(kq, &change)
	if err != nil {
		unix.Close(kq)
		if errors.Is(err, unix.ESRCH) {
			cancel()
			return func() {}
		}
		return pollWatch(ctx, cancel, id)
	}
	if n == 0 {
		unix.Close(kq)
		return pollWatch(ctx, cancel, id)
	}

	go func() {
		events := make([]unix.Kevent_t, 1)
		for {
			_, err := unix.Kevent(kq, nil, events, nil)
			if errors.Is(err, unix.EINTR) {
				continue
			}
			if err == nil {
				cancel()
			}
			return
		}
	}()

	var once sync.Once
	return func() {
		once.Do(func() {
			unix.Close(kq)
		})
	}
}

func keventRegister(kq int, change *unix.Kevent_t) (int, error) {
	for {
		n, err := unix.Kevent(kq, []unix.Kevent_t{*change}, nil, nil)
		if errors.Is(err, unix.EINTR) {
			continue
		}
		return n, err
	}
}
