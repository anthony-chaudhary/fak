//go:build darwin && cgo

package power

/*
#cgo LDFLAGS: -framework IOKit -framework CoreFoundation
#include <stdint.h>
#include <CoreFoundation/CoreFoundation.h>
#include <IOKit/pwr_mgt/IOPMLib.h>
#include <IOKit/IOMessage.h>

extern void goDarwinPowerCallback(void *refCon, io_service_t service, natural_t messageType, void *messageArgument);

static io_connect_t registerDarwinPowerNotifications(uintptr_t identity, IONotificationPortRef *notifyPort, io_object_t *notifierObject, CFRunLoopRef *runLoopRef) {
    io_connect_t rootPort;
    rootPort = IORegisterForSystemPower((void *)identity, notifyPort, (IOServiceInterestCallback)goDarwinPowerCallback, notifierObject);
    if (!rootPort) {
        return 0;
    }

    CFRunLoopSourceRef rls = IONotificationPortGetRunLoopSource(*notifyPort);
    CFRunLoopRef rl = CFRunLoopGetCurrent();
    *runLoopRef = rl;
    CFRunLoopAddSource(rl, rls, kCFRunLoopDefaultMode);
    return rootPort;
}

static void deregisterDarwinPowerNotifications(io_connect_t rootPort, IONotificationPortRef notifyPort, io_object_t notifierObject, CFRunLoopRef rl) {
    if (notifyPort && rl) {
        CFRunLoopSourceRef rls = IONotificationPortGetRunLoopSource(notifyPort);
        if (rls) {
            CFRunLoopRemoveSource(rl, rls, kCFRunLoopDefaultMode);
        }
    }
    if (notifierObject) {
        IODeregisterForSystemPower(&notifierObject);
    }
    if (rootPort) {
        IOServiceClose(rootPort);
    }
    if (notifyPort) {
        IONotificationPortDestroy(notifyPort);
    }
}

static void allowDarwinPowerChange(io_connect_t rootPort, long messageArg) {
    if (rootPort) {
        IOAllowPowerChange(rootPort, messageArg);
    }
}

static void runDarwinRunLoop() {
    CFRunLoopRun();
}

static void stopDarwinRunLoop(CFRunLoopRef rl) {
    if (rl) {
        CFRunLoopStop(rl);
    }
}

// Exercise the callback ABI without registering or sending an OS notification.
static void invokeDarwinPowerCallback(uintptr_t identity, unsigned int messageType, long argument) {
    goDarwinPowerCallback((void *)identity, 0, messageType, (void *)(uintptr_t)argument);
}
*/
import "C"

import (
	"context"
	"fmt"
	"runtime"
	"runtime/cgo"
	"sync"
	"time"
	"unsafe"
)

// Registration and teardown are serialized; callback ownership is carried by
// IOKit's refCon, never a process-global "current listener".
var darwinCGORegistrationMu sync.Mutex

const (
	darwinCanSleep     = uint32(C.kIOMessageCanSystemSleep)
	darwinWillSleep    = uint32(C.kIOMessageSystemWillSleep)
	darwinWillPowerOn  = uint32(C.kIOMessageSystemWillPowerOn)
	darwinHasPoweredOn = uint32(C.kIOMessageSystemHasPoweredOn)
)

type darwinPowerPhase uint8

const (
	darwinAwake darwinPowerPhase = iota
	darwinSleeping
	darwinWaking
)

//export goDarwinPowerCallback
func goDarwinPowerCallback(refCon unsafePointer, service C.io_service_t, messageType C.natural_t, messageArgument unsafePointer) {
	l := cgo.Handle(uintptr(refCon)).Value().(*darwinCGOListener)
	l.dispatchPowerMessage(uint32(messageType), int64(uintptr(messageArgument)))
}

// invokeDarwinCallback enters the real C-to-Go callback with SDK message values.
// Callers supplying synthetic arguments must install a per-listener ack sink.
func invokeDarwinCallback(identity cgo.Handle, message uint32, argument int64) {
	C.invokeDarwinPowerCallback(C.uintptr_t(identity), C.uint(message), C.long(argument))
}

// dispatchPowerMessage is the native callback dispatcher. A listener's run loop
// delivers notifications serially. The acknowledgement sink is local to that
// listener so injected dispatch never needs a real IOKit acknowledgement token.
func (l *darwinCGOListener) dispatchPowerMessage(msgType uint32, arg int64) {
	event := PowerEvent{Timestamp: time.Now(), Source: "iokit-cgo"}
	switch msgType {
	case darwinCanSleep:
		l.acknowledge(arg)
		return
	case darwinWillSleep:
		// Finalization also runs if a synchronous observer panics. Propagate the
		// panic; this callback does not define an observer recovery policy.
		defer l.acknowledge(arg)
		if l.phase != darwinAwake {
			return
		}
		l.phase = darwinSleeping
		event.Type = EventSleep
		event.Details = "kIOMessageSystemWillSleep"
	case darwinWillPowerOn:
		if l.phase == darwinSleeping {
			l.phase = darwinWaking
		}
		return
	case darwinHasPoweredOn:
		if l.phase == darwinAwake {
			return
		}
		l.phase = darwinAwake
		event.Type = EventWake
		event.Details = "kIOMessageSystemHasPoweredOn"
	default:
		return
	}
	l.broadcaster.Broadcast(event)
}

func (l *darwinCGOListener) acknowledge(arg int64) {
	if l.ack != nil {
		l.ack(arg)
		return
	}
	C.allowDarwinPowerChange(l.rootPort, C.long(arg))
}

// resetCycle is called only before a registration's run loop starts.
func (l *darwinCGOListener) resetCycle() { l.phase = darwinAwake }

type unsafePointer = unsafe.Pointer

type darwinCGOListener struct {
	phase       darwinPowerPhase
	ack         func(int64)
	mu          sync.Mutex
	broadcaster *PowerBroadcaster
	rootPort    C.io_connect_t
	notifyPort  C.IONotificationPortRef
	notifierObj C.io_object_t
	runLoop     C.CFRunLoopRef
	ready       chan struct{}
	done        chan struct{}
	running     bool
	stopped     bool
}

func newDarwinIOKitListener(b *PowerBroadcaster) SleepListener {
	if b == nil {
		b = defaultBroadcaster
	}
	return &darwinCGOListener{
		broadcaster: b,
		ready:       make(chan struct{}),
		done:        make(chan struct{}),
	}
}

func (l *darwinCGOListener) Start(ctx context.Context) error {
	l.mu.Lock()
	if l.running {
		l.mu.Unlock()
		return fmt.Errorf("iokit listener already running")
	}
	l.running = true
	l.stopped = false
	l.ready = make(chan struct{})
	l.done = make(chan struct{})
	l.resetCycle()
	l.mu.Unlock()

	errCh := make(chan error, 1)

	go func() {
		// CFRunLoop must run on an OS thread locked to the loop
		runtime.LockOSThread()
		defer runtime.UnlockOSThread()

		identity := cgo.NewHandle(l)
		defer identity.Delete()

		var (
			np  C.IONotificationPortRef
			obj C.io_object_t
			rl  C.CFRunLoopRef
		)

		darwinCGORegistrationMu.Lock()
		port := C.registerDarwinPowerNotifications(C.uintptr_t(identity), &np, &obj, &rl)
		darwinCGORegistrationMu.Unlock()
		if port == 0 {
			l.mu.Lock()
			l.running = false
			close(l.ready)
			close(l.done)
			l.mu.Unlock()
			errCh <- fmt.Errorf("IORegisterForSystemPower failed")
			return
		}

		l.mu.Lock()
		l.rootPort = port
		l.notifyPort = np
		l.notifierObj = obj
		l.runLoop = rl

		l.mu.Unlock()

		close(l.ready)
		errCh <- nil

		// Blocks until C.stopDarwinRunLoop is called
		C.runDarwinRunLoop()

		// Cleanup after loop exits
		l.mu.Lock()
		rootPort := l.rootPort
		notifyPort := l.notifyPort
		notifierObj := l.notifierObj
		runLoop := l.runLoop
		l.rootPort = 0
		l.notifyPort = C.IONotificationPortRef(nil)
		l.notifierObj = 0
		l.runLoop = 0

		darwinCGORegistrationMu.Lock()
		C.deregisterDarwinPowerNotifications(rootPort, notifyPort, notifierObj, runLoop)
		darwinCGORegistrationMu.Unlock()
		close(l.done)
		l.running = false
		l.mu.Unlock()
	}()

	select {
	case err := <-errCh:
		if err != nil {
			return err
		}
	case <-ctx.Done():
		_ = l.Stop()
		return ctx.Err()
	}

	return nil
}

func (l *darwinCGOListener) Stop() error {
	l.mu.Lock()
	if !l.running || l.stopped {
		l.mu.Unlock()
		return nil
	}
	l.stopped = true
	rl := l.runLoop
	done := l.done
	l.mu.Unlock()

	if rl != 0 {
		C.stopDarwinRunLoop(rl)
	}

	select {
	case <-done:
	case <-time.After(2 * time.Second):
	}

	return nil
}
