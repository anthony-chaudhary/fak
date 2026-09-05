package power

import (
	"context"
	"fmt"
	"sync"
	"time"
)

// EventType indicates whether the system is preparing to sleep, resuming, or in an error state.
type EventType string

const (
	// EventSleep indicates the system is about to sleep or suspend.
	EventSleep EventType = "SLEEP"
	// EventWake indicates the system has resumed from sleep or suspend.
	EventWake EventType = "WAKE"
)

// PowerEvent represents an OS sleep/wake power event.
type PowerEvent struct {
	Type      EventType
	Timestamp time.Time
	Source    string // e.g. "iokit", "cgo", "pure-go", "pmset", "broadcast", "mock"
	Details   string
}

func (e PowerEvent) String() string {
	return fmt.Sprintf("PowerEvent{Type: %s, Source: %s, Timestamp: %s, Details: %s}",
		e.Type, e.Source, e.Timestamp.Format(time.RFC3339Nano), e.Details)
}

// PowerObserver receives notifications for system suspend and resume lifecycle transitions.
type PowerObserver interface {
	OnSuspend(event PowerEvent)
	OnResume(event PowerEvent)
}

// SleepListener represents an active listener that can observe power events or be stopped.
type SleepListener interface {
	Start(ctx context.Context) error
	Stop() error
}

// ObserverFunc is an adapter allowing bare functions to act as PowerObservers.
type ObserverFunc struct {
	SuspendFn func(PowerEvent)
	ResumeFn  func(PowerEvent)
}

// OnSuspend dispatches to SuspendFn if non-nil.
func (o ObserverFunc) OnSuspend(e PowerEvent) {
	if o.SuspendFn != nil {
		o.SuspendFn(e)
	}
}

// OnResume dispatches to ResumeFn if non-nil.
func (o ObserverFunc) OnResume(e PowerEvent) {
	if o.ResumeFn != nil {
		o.ResumeFn(e)
	}
}

// PowerBroadcaster manages registration and dispatch of power events to observers.
type PowerBroadcaster struct {
	mu        sync.RWMutex
	nextID    uint64
	observers map[uint64]PowerObserver
	listeners map[uint64]func(PowerEvent)
}

// NewPowerBroadcaster creates an empty PowerBroadcaster.
func NewPowerBroadcaster() *PowerBroadcaster {
	return &PowerBroadcaster{
		observers: make(map[uint64]PowerObserver),
		listeners: make(map[uint64]func(PowerEvent)),
	}
}

// RegisterObserver registers a PowerObserver. The returned cancel func unregisters it.
func (b *PowerBroadcaster) RegisterObserver(obs PowerObserver) (cancel func()) {
	if obs == nil {
		return func() {}
	}
	b.mu.Lock()
	defer b.mu.Unlock()
	id := b.nextID
	b.nextID++
	b.observers[id] = obs
	return func() {
		b.mu.Lock()
		defer b.mu.Unlock()
		delete(b.observers, id)
	}
}

// RegisterFunc registers a callback function for all power events.
func (b *PowerBroadcaster) RegisterFunc(fn func(PowerEvent)) (cancel func()) {
	if fn == nil {
		return func() {}
	}
	b.mu.Lock()
	defer b.mu.Unlock()
	id := b.nextID
	b.nextID++
	b.listeners[id] = fn
	return func() {
		b.mu.Lock()
		defer b.mu.Unlock()
		delete(b.listeners, id)
	}
}

// Broadcast dispatches a PowerEvent synchronously (in order) to all registered observers and callbacks.
func (b *PowerBroadcaster) Broadcast(event PowerEvent) {
	if event.Timestamp.IsZero() {
		event.Timestamp = time.Now()
	}

	b.mu.RLock()
	observers := make([]PowerObserver, 0, len(b.observers))
	for _, obs := range b.observers {
		observers = append(observers, obs)
	}
	listeners := make([]func(PowerEvent), 0, len(b.listeners))
	for _, fn := range b.listeners {
		listeners = append(listeners, fn)
	}
	b.mu.RUnlock()

	for _, obs := range observers {
		switch event.Type {
		case EventSleep:
			obs.OnSuspend(event)
		case EventWake:
			obs.OnResume(event)
		}
	}

	for _, fn := range listeners {
		fn(event)
	}
}

// ObserverCount returns the current number of registered observers and callbacks.
func (b *PowerBroadcaster) ObserverCount() int {
	b.mu.RLock()
	defer b.mu.RUnlock()
	return len(b.observers) + len(b.listeners)
}

// Package-level global broadcaster for system-wide power notifications.
var (
	defaultBroadcaster = NewPowerBroadcaster()
)

// RegisterSleepObserver registers a PowerObserver with the package-level default broadcaster.
func RegisterSleepObserver(obs PowerObserver) (cancel func()) {
	return defaultBroadcaster.RegisterObserver(obs)
}

// RegisterSleepFunc registers a power event callback with the package-level default broadcaster.
func RegisterSleepFunc(fn func(PowerEvent)) (cancel func()) {
	return defaultBroadcaster.RegisterFunc(fn)
}

// BroadcastEvent dispatches an event via the package-level default broadcaster.
func BroadcastEvent(event PowerEvent) {
	defaultBroadcaster.Broadcast(event)
}

// LeaseFreezeState represents the lease lifecycle state across sleep/wake transitions.
type LeaseFreezeState string

const (
	// StateActive indicates normal operation where watchdog and lease timers tick.
	StateActive LeaseFreezeState = "ACTIVE"
	// StateSuspendedFreeze indicates the system is sleeping/suspending; lease is frozen to prevent split-brain expiration.
	StateSuspendedFreeze LeaseFreezeState = "SUSPENDED_FREEZE"
	// StateNeedsReadjudication indicates system resumed from sleep and leases require re-adjudication against peer state.
	StateNeedsReadjudication LeaseFreezeState = "NEEDS_READJUDICATION"
)

// LeaseFreezeCoordinator manages the lease freeze state machine in response to power events.
// When an OS sleep event occurs, held leases transition to SUSPENDED_FREEZE.
// When an OS wake event occurs, leases transition to NEEDS_READJUDICATION and wake hooks fire.
type LeaseFreezeCoordinator struct {
	mu           sync.RWMutex
	state        LeaseFreezeState
	frozenAt     time.Time
	resumedAt    time.Time
	suspendCount int64
	resumeCount  int64
	onFreeze     []func(event PowerEvent)
	onThaw       []func(event PowerEvent)
	cancelReg    func()
}

// NewLeaseFreezeCoordinator creates a new coordinator and subscribes to the provided broadcaster.
// If broadcaster is nil, defaultBroadcaster is used.
func NewLeaseFreezeCoordinator(b *PowerBroadcaster) *LeaseFreezeCoordinator {
	if b == nil {
		b = defaultBroadcaster
	}
	c := &LeaseFreezeCoordinator{
		state: StateActive,
	}
	c.cancelReg = b.RegisterObserver(c)
	return c
}

// Close unregisters the coordinator from its broadcaster.
func (c *LeaseFreezeCoordinator) Close() {
	c.mu.Lock()
	defer c.mu.Unlock()
	if c.cancelReg != nil {
		c.cancelReg()
		c.cancelReg = nil
	}
}

// State returns the current freeze state.
func (c *LeaseFreezeCoordinator) State() LeaseFreezeState {
	c.mu.RLock()
	defer c.mu.RUnlock()
	return c.state
}

// IsFrozen reports whether the coordinator is currently in SUSPENDED_FREEZE.
func (c *LeaseFreezeCoordinator) IsFrozen() bool {
	c.mu.RLock()
	defer c.mu.RUnlock()
	return c.state == StateSuspendedFreeze
}

// NeedsReadjudication reports whether the coordinator requires re-adjudication after waking.
func (c *LeaseFreezeCoordinator) NeedsReadjudication() bool {
	c.mu.RLock()
	defer c.mu.RUnlock()
	return c.state == StateNeedsReadjudication
}

// SuspendCount returns total sleep transitions witnessed.
func (c *LeaseFreezeCoordinator) SuspendCount() int64 {
	c.mu.RLock()
	defer c.mu.RUnlock()
	return c.suspendCount
}

// ResumeCount returns total wake transitions witnessed.
func (c *LeaseFreezeCoordinator) ResumeCount() int64 {
	c.mu.RLock()
	defer c.mu.RUnlock()
	return c.resumeCount
}

// FrozenDuration returns how long the current freeze has lasted, or the duration of the last freeze if active.
func (c *LeaseFreezeCoordinator) FrozenDuration() time.Duration {
	c.mu.RLock()
	defer c.mu.RUnlock()
	if c.state == StateSuspendedFreeze {
		if c.frozenAt.IsZero() {
			return 0
		}
		return time.Since(c.frozenAt)
	}
	if !c.frozenAt.IsZero() && !c.resumedAt.IsZero() && c.resumedAt.After(c.frozenAt) {
		return c.resumedAt.Sub(c.frozenAt)
	}
	return 0
}

// OnFreeze registers a hook called whenever transitioning into SUSPENDED_FREEZE.
func (c *LeaseFreezeCoordinator) OnFreeze(fn func(PowerEvent)) {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.onFreeze = append(c.onFreeze, fn)
}

// OnThaw registers a hook called whenever transitioning into NEEDS_READJUDICATION or ACTIVE.
func (c *LeaseFreezeCoordinator) OnThaw(fn func(PowerEvent)) {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.onThaw = append(c.onThaw, fn)
}

// MarkAdjudicated marks the lease state back to ACTIVE after re-adjudication finishes cleanly.
func (c *LeaseFreezeCoordinator) MarkAdjudicated() {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.state = StateActive
}

// OnSuspend handles sleep events, moving state to SUSPENDED_FREEZE.
func (c *LeaseFreezeCoordinator) OnSuspend(event PowerEvent) {
	c.mu.Lock()
	c.state = StateSuspendedFreeze
	c.frozenAt = event.Timestamp
	c.suspendCount++
	callbacks := make([]func(PowerEvent), len(c.onFreeze))
	copy(callbacks, c.onFreeze)
	c.mu.Unlock()

	for _, fn := range callbacks {
		fn(event)
	}
}

// OnResume handles wake events, moving state to NEEDS_READJUDICATION.
func (c *LeaseFreezeCoordinator) OnResume(event PowerEvent) {
	c.mu.Lock()
	c.state = StateNeedsReadjudication
	c.resumedAt = event.Timestamp
	c.resumeCount++
	callbacks := make([]func(PowerEvent), len(c.onThaw))
	copy(callbacks, c.onThaw)
	c.mu.Unlock()

	for _, fn := range callbacks {
		fn(event)
	}
}
