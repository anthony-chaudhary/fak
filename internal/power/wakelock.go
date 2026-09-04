package power

import (
	"context"
	"errors"
	"fmt"
	"sync"
	"time"
)

// Releaser represents an acquired power assertion token that can be released.
type Releaser interface {
	Release() error
}

// WakeFlags specifies options for the power assertion.
type WakeFlags uint32

const (
	// PreventSystemSleep prevents the OS from entering idle sleep while operations are running.
	PreventSystemSleep WakeFlags = 1 << iota
	// PreventDisplaySleep prevents the display from entering sleep or turning off.
	PreventDisplaySleep
)

// platformLock is implemented by OS-specific power management backends.
type platformLock interface {
	Release() error
}

// noOpLock provides a no-op fallback platformLock for unsupported OS or fallback paths.
type noOpLock struct{}

func (n *noOpLock) Release() error {
	return nil
}

// WakeLock represents an active OS power assertion or wake lock.
// It is safe for concurrent use by multiple goroutines and supports reentrant acquisitions.
type WakeLock struct {
	mu       sync.Mutex
	reason   string
	flags    WakeFlags
	refCount int
	active   bool
	released bool
	impl     platformLock
	timer    *time.Timer
	cancel   context.CancelFunc
}

// NewWakeLock creates and activates a new independent WakeLock.
func NewWakeLock(reason string, flags WakeFlags) (*WakeLock, error) {
	if flags == 0 {
		flags = PreventSystemSleep
	}
	impl, err := platformAcquire(reason, flags)
	if err != nil {
		return nil, fmt.Errorf("power assertion acquire: %w", err)
	}
	return &WakeLock{
		reason:   reason,
		flags:    flags,
		refCount: 1,
		active:   true,
		impl:     impl,
	}, nil
}

// Active reports whether the wake lock is currently held.
func (w *WakeLock) Active() bool {
	w.mu.Lock()
	defer w.mu.Unlock()
	return w.active && !w.released
}

// Reason returns the reason given when acquiring the wake lock.
func (w *WakeLock) Reason() string {
	w.mu.Lock()
	defer w.mu.Unlock()
	return w.reason
}

// Flags returns the flags configured on the wake lock.
func (w *WakeLock) Flags() WakeFlags {
	w.mu.Lock()
	defer w.mu.Unlock()
	return w.flags
}

// RefCount returns the current reentrant acquisition count.
func (w *WakeLock) RefCount() int {
	w.mu.Lock()
	defer w.mu.Unlock()
	if w.released {
		return 0
	}
	return w.refCount
}

// Acquire increments the reentrant acquisition count and returns a Releaser.
func (w *WakeLock) Acquire() (Releaser, error) {
	w.mu.Lock()
	defer w.mu.Unlock()
	if w.released || !w.active {
		return nil, errors.New("wake lock already released")
	}
	w.refCount++
	return &reentrantToken{lock: w}, nil
}

// AcquireWithTimeout increments the reentrant acquisition count and automatically
// releases it after the given duration if not released sooner.
func (w *WakeLock) AcquireWithTimeout(timeout time.Duration) (Releaser, error) {
	r, err := w.Acquire()
	if err != nil {
		return nil, err
	}
	if timeout <= 0 {
		return r, nil
	}
	ctx, cancel := context.WithTimeout(context.Background(), timeout)
	tr := &timeoutReleaser{
		releaser: r,
		cancel:   cancel,
	}
	go func() {
		select {
		case <-ctx.Done():
			_ = tr.Release()
		}
	}()
	return tr, nil
}

// Release decrements the acquisition count and releases the underlying OS assertion
// when the reference count drops to zero. Calling Release multiple times is idempotent.
func (w *WakeLock) Release() error {
	w.mu.Lock()
	defer w.mu.Unlock()
	if w.released {
		return nil
	}
	w.refCount--
	if w.refCount <= 0 {
		w.released = true
		w.active = false
		if w.timer != nil {
			w.timer.Stop()
			w.timer = nil
		}
		if w.cancel != nil {
			w.cancel()
			w.cancel = nil
		}
		if w.impl != nil {
			err := w.impl.Release()
			w.impl = nil
			return err
		}
	}
	return nil
}

type reentrantToken struct {
	lock *WakeLock
	once sync.Once
	err  error
}

func (t *reentrantToken) Release() error {
	t.once.Do(func() {
		t.err = t.lock.Release()
	})
	return t.err
}

type timeoutReleaser struct {
	releaser Releaser
	cancel   context.CancelFunc
	once     sync.Once
	err      error
}

func (tr *timeoutReleaser) Release() error {
	tr.once.Do(func() {
		if tr.cancel != nil {
			tr.cancel()
		}
		tr.err = tr.releaser.Release()
	})
	return tr.err
}

var (
	globalMu   sync.Mutex
	globalLock *WakeLock
)

// Acquire acquires a shared, reentrant wake lock with the given reason and flags.
// The underlying OS assertion is held as long as at least one acquired token is active.
func Acquire(reason string, flags WakeFlags) (Releaser, error) {
	globalMu.Lock()
	defer globalMu.Unlock()

	if flags == 0 {
		flags = PreventSystemSleep
	}

	if globalLock == nil || !globalLock.Active() {
		lock, err := NewWakeLock(reason, flags)
		if err != nil {
			return nil, err
		}
		globalLock = lock
		return &globalToken{lock: lock}, nil
	}

	// Reentrant acquisition on existing global lock
	token, err := globalLock.Acquire()
	if err != nil {
		// If prior global lock was released in a race, create a fresh one
		lock, err := NewWakeLock(reason, flags)
		if err != nil {
			return nil, err
		}
		globalLock = lock
		return &globalToken{lock: lock}, nil
	}
	return &globalToken{lock: globalLock, token: token}, nil
}

type globalToken struct {
	lock  *WakeLock
	token Releaser
	once  sync.Once
	err   error
}

func (gt *globalToken) Release() error {
	gt.once.Do(func() {
		globalMu.Lock()
		defer globalMu.Unlock()
		if gt.token != nil {
			gt.err = gt.token.Release()
		} else if gt.lock != nil {
			gt.err = gt.lock.Release()
		}
	})
	return gt.err
}

// AcquireWithTimeout acquires a wake lock that automatically releases after timeout.
func AcquireWithTimeout(reason string, flags WakeFlags, timeout time.Duration) (Releaser, error) {
	r, err := Acquire(reason, flags)
	if err != nil {
		return nil, err
	}
	if timeout <= 0 {
		return r, nil
	}
	ctx, cancel := context.WithTimeout(context.Background(), timeout)
	tr := &timeoutReleaser{
		releaser: r,
		cancel:   cancel,
	}
	go func() {
		select {
		case <-ctx.Done():
			_ = tr.Release()
		}
	}()
	return tr, nil
}

// AcquireContext acquires a wake lock that automatically releases when ctx is done.
func AcquireContext(ctx context.Context, reason string, flags WakeFlags) (Releaser, error) {
	r, err := Acquire(reason, flags)
	if err != nil {
		return nil, err
	}
	go func() {
		select {
		case <-ctx.Done():
			_ = r.Release()
		}
	}()
	return r, nil
}

// IsActive reports whether the package-level shared wake lock is active.
func IsActive() bool {
	globalMu.Lock()
	defer globalMu.Unlock()
	return globalLock != nil && globalLock.Active()
}

// GlobalRefCount reports the reference count of the package-level shared wake lock.
func GlobalRefCount() int {
	globalMu.Lock()
	defer globalMu.Unlock()
	if globalLock == nil {
		return 0
	}
	return globalLock.RefCount()
}

// ResetGlobalForTesting resets the package-level lock state for isolated testing.
func ResetGlobalForTesting() {
	globalMu.Lock()
	defer globalMu.Unlock()
	if globalLock != nil {
		_ = globalLock.Release()
		globalLock = nil
	}
}
