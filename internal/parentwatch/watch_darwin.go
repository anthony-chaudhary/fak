//go:build darwin

package parentwatch

import (
	"context"
	"errors"
	"sync"

	"golang.org/x/sys/unix"
)

func watch(ctx context.Context, cancel context.CancelFunc, parentPID int) func() {
	kq, err := unix.Kqueue()
	if err != nil {
		return pollWatch(ctx, cancel, parentPID)
	}

	var change unix.Kevent_t
	unix.SetKevent(&change, parentPID, unix.EVFILT_PROC, unix.EV_ADD|unix.EV_ENABLE)
	change.Fflags = unix.NOTE_EXIT

	n, err := keventRegister(kq, &change)
	if err != nil {
		unix.Close(kq)
		if errors.Is(err, unix.ESRCH) {
			cancel()
			return func() {}
		}
		return pollWatch(ctx, cancel, parentPID)
	}
	if n == 0 {
		unix.Close(kq)
		return pollWatch(ctx, cancel, parentPID)
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
