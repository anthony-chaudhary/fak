//go:build darwin

package macobs

/*
#cgo LDFLAGS: -framework CoreFoundation
#include <dispatch/dispatch.h>
#include <stdlib.h>
#include <stdint.h>

typedef struct {
	uintptr_t sub_id;
	dispatch_source_t source;
	dispatch_queue_t queue;
} memory_pressure_ctx_t;

extern void goDarwinMemoryPressureDispatch(uintptr_t subID, unsigned long flags);

static void handleDarwinMemoryPressure(void *context) {
	memory_pressure_ctx_t *ctx = (memory_pressure_ctx_t*)context;
	if (ctx && ctx->source) {
		unsigned long flags = dispatch_source_get_data(ctx->source);
		goDarwinMemoryPressureDispatch(ctx->sub_id, flags);
	}
}

static memory_pressure_ctx_t* startDarwinMemoryPressure(uintptr_t subID) {
	memory_pressure_ctx_t *ctx = (memory_pressure_ctx_t*)malloc(sizeof(memory_pressure_ctx_t));
	if (!ctx) {
		return NULL;
	}
	ctx->sub_id = subID;
	ctx->queue = dispatch_queue_create("fak.macobs.memorypressure", DISPATCH_QUEUE_SERIAL);
	if (!ctx->queue) {
		free(ctx);
		return NULL;
	}
	ctx->source = dispatch_source_create(
		DISPATCH_SOURCE_TYPE_MEMORYPRESSURE,
		0,
		DISPATCH_MEMORYPRESSURE_NORMAL | DISPATCH_MEMORYPRESSURE_WARN | DISPATCH_MEMORYPRESSURE_CRITICAL,
		ctx->queue
	);
	if (!ctx->source) {
		dispatch_release(ctx->queue);
		free(ctx);
		return NULL;
	}
	dispatch_set_context(ctx->source, ctx);
	dispatch_source_set_event_handler_f(ctx->source, handleDarwinMemoryPressure);
	dispatch_resume(ctx->source);
	return ctx;
}

static void stopDarwinMemoryPressure(memory_pressure_ctx_t *ctx) {
	if (!ctx) {
		return;
	}
	if (ctx->source) {
		dispatch_source_cancel(ctx->source);
		dispatch_release(ctx->source);
		ctx->source = NULL;
	}
	if (ctx->queue) {
		dispatch_release(ctx->queue);
		ctx->queue = NULL;
	}
	free(ctx);
}
*/
import "C"

import (
	"context"
	"fmt"
	"sync"
	"sync/atomic"
	"time"
)

const (
	darwinMemoryPressureNormal   = 0x01
	darwinMemoryPressureWarn     = 0x02
	darwinMemoryPressureCritical = 0x04
)

var (
	darwinSubscribers   = make(map[uintptr]*DarwinMemoryPressureSubscriber)
	darwinSubscribersMu sync.RWMutex
	nextSubscriberID    uint64
)

//export goDarwinMemoryPressureDispatch
func goDarwinMemoryPressureDispatch(subID C.uintptr_t, flags C.ulong) {
	darwinSubscribersMu.RLock()
	sub := darwinSubscribers[uintptr(subID)]
	darwinSubscribersMu.RUnlock()

	if sub != nil {
		sub.handleFlags(uint64(flags), "darwin_dispatch_source")
	}
}

// DarwinMemoryPressureSubscriber subscribes to macOS kernel memory pressure notifications via GCD.
type DarwinMemoryPressureSubscriber struct {
	id           uintptr
	ctx          *C.memory_pressure_ctx_t
	mu           sync.RWMutex
	handlers     map[uint64]PressureHandler
	nextHandler  uint64
	currentLevel MemoryPressureLevel
	running      bool
	cancel       context.CancelFunc
	done         chan struct{}
}

// NewDarwinMemoryPressureSubscriber instantiates a subscriber listening to Darwin memory pressure events.
func NewDarwinMemoryPressureSubscriber() *DarwinMemoryPressureSubscriber {
	id := uintptr(atomic.AddUint64(&nextSubscriberID, 1))
	return &DarwinMemoryPressureSubscriber{
		id:           id,
		handlers:     make(map[uint64]PressureHandler),
		currentLevel: PressureNormal,
	}
}

// Start registers the GCD dispatch source and begins listening for kernel memory pressure notifications.
func (s *DarwinMemoryPressureSubscriber) Start(ctx context.Context) error {
	s.mu.Lock()
	if s.running {
		s.mu.Unlock()
		return nil
	}
	s.running = true
	s.done = make(chan struct{})
	sCtx, cancel := context.WithCancel(ctx)
	s.cancel = cancel

	darwinSubscribersMu.Lock()
	darwinSubscribers[s.id] = s
	darwinSubscribersMu.Unlock()

	ctxPtr := C.startDarwinMemoryPressure(C.uintptr_t(s.id))
	if ctxPtr == nil {
		darwinSubscribersMu.Lock()
		delete(darwinSubscribers, s.id)
		darwinSubscribersMu.Unlock()
		s.running = false
		s.mu.Unlock()
		return fmt.Errorf("failed to create Darwin GCD memory pressure source")
	}
	s.ctx = ctxPtr
	s.mu.Unlock()

	go func() {
		select {
		case <-sCtx.Done():
			_ = s.Stop()
		case <-s.done:
		}
	}()

	return nil
}

// Stop cancels and releases the underlying GCD memory pressure source.
func (s *DarwinMemoryPressureSubscriber) Stop() error {
	s.mu.Lock()
	if !s.running {
		s.mu.Unlock()
		return nil
	}
	s.running = false
	ctxPtr := s.ctx
	s.ctx = nil
	cancel := s.cancel
	s.cancel = nil
	close(s.done)
	s.mu.Unlock()

	darwinSubscribersMu.Lock()
	delete(darwinSubscribers, s.id)
	darwinSubscribersMu.Unlock()

	if cancel != nil {
		cancel()
	}

	if ctxPtr != nil {
		C.stopDarwinMemoryPressure(ctxPtr)
	}
	return nil
}

// CurrentLevel returns the most recently received memory pressure level.
func (s *DarwinMemoryPressureSubscriber) CurrentLevel() MemoryPressureLevel {
	s.mu.RLock()
	defer s.mu.RUnlock()
	return s.currentLevel
}

// Subscribe attaches a PressureHandler callback and returns an unsubscribe function.
func (s *DarwinMemoryPressureSubscriber) Subscribe(handler PressureHandler) func() {
	s.mu.Lock()
	hid := s.nextHandler
	s.nextHandler++
	s.handlers[hid] = handler
	s.mu.Unlock()

	return func() {
		s.mu.Lock()
		delete(s.handlers, hid)
		s.mu.Unlock()
	}
}

// SimulatePressure allows programmatically dispatching a simulated Darwin memory pressure event.
func (s *DarwinMemoryPressureSubscriber) SimulatePressure(level MemoryPressureLevel) {
	var flags uint64
	switch level {
	case PressureNormal:
		flags = darwinMemoryPressureNormal
	case PressureWarn:
		flags = darwinMemoryPressureWarn
	case PressureCritical:
		flags = darwinMemoryPressureCritical
	default:
		flags = 0
	}
	s.handleFlags(flags, "simulated_darwin_gcd")
}

func (s *DarwinMemoryPressureSubscriber) handleFlags(flags uint64, source string) {
	level := parseDarwinPressureFlags(flags)

	s.mu.Lock()
	s.currentLevel = level
	handlers := make([]PressureHandler, 0, len(s.handlers))
	for _, h := range s.handlers {
		handlers = append(handlers, h)
	}
	s.mu.Unlock()

	evt := PressureEvent{
		Level:     level,
		Timestamp: time.Now().UTC(),
		RawFlags:  flags,
		Source:    source,
		Details:   fmt.Sprintf("Darwin GCD memory pressure event (flags=0x%x, level=%s)", flags, level),
	}

	for _, h := range handlers {
		h(evt)
	}
}

func parseDarwinPressureFlags(flags uint64) MemoryPressureLevel {
	if flags&darwinMemoryPressureCritical != 0 {
		return PressureCritical
	}
	if flags&darwinMemoryPressureWarn != 0 {
		return PressureWarn
	}
	if flags&darwinMemoryPressureNormal != 0 {
		return PressureNormal
	}
	return PressureUnknown
}

func init() {
	newPlatformSubscriber = func() MemoryPressureSubscriber {
		return NewDarwinMemoryPressureSubscriber()
	}
}
