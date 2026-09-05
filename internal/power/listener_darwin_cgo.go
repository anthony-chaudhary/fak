//go:build darwin && cgo

package power

/*
#cgo LDFLAGS: -framework IOKit -framework CoreFoundation
#include <CoreFoundation/CoreFoundation.h>
#include <IOKit/pwr_mgt/IOPMLib.h>
#include <IOKit/IOMessage.h>

extern void goDarwinPowerCallback(void *refCon, io_service_t service, natural_t messageType, void *messageArgument);

static io_connect_t registerDarwinPowerNotifications(IONotificationPortRef *notifyPort, io_object_t *notifierObject, CFRunLoopRef *runLoopRef) {
    io_connect_t rootPort;
    rootPort = IORegisterForSystemPower(NULL, notifyPort, (IOServiceInterestCallback)goDarwinPowerCallback, notifierObject);
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

static unsigned int getMsgCanSleep() { return kIOMessageCanSystemSleep; }
static unsigned int getMsgSystemWillSleep() { return kIOMessageSystemWillSleep; }
static unsigned int getMsgSystemWillPowerOn() { return kIOMessageSystemWillPowerOn; }
static unsigned int getMsgSystemHasPoweredOn() { return kIOMessageSystemHasPoweredOn; }
*/
import "C"

import (
	"context"
	"fmt"
	"runtime"
	"sync"
	"time"
	"unsafe"
)

var (
	darwinCGORootPort  C.io_connect_t
	darwinCGOMu        sync.Mutex
	darwinCGOBroadcast func(PowerEvent)
)

//export goDarwinPowerCallback
func goDarwinPowerCallback(refCon unsafePointer, service C.io_service_t, messageType C.natural_t, messageArgument unsafePointer) {
	msgType := uint32(messageType)
	arg := C.long(uintptr(messageArgument))

	darwinCGOMu.Lock()
	rootPort := darwinCGORootPort
	broadcast := darwinCGOBroadcast
	darwinCGOMu.Unlock()

	var event PowerEvent
	event.Timestamp = time.Now()
	event.Source = "iokit-cgo"

	switch msgType {
	case uint32(C.getMsgCanSleep()):
		// System queries whether sleep can occur. We allow it and emit SLEEP event.
		event.Type = EventSleep
		event.Details = "kIOMessageCanSystemSleep"
		if rootPort != 0 {
			C.allowDarwinPowerChange(rootPort, arg)
		}
	case uint32(C.getMsgSystemWillSleep()):
		// System will sleep. Acknowledge notification and emit SLEEP event.
		event.Type = EventSleep
		event.Details = "kIOMessageSystemWillSleep"
		if rootPort != 0 {
			C.allowDarwinPowerChange(rootPort, arg)
		}
	case uint32(C.getMsgSystemWillPowerOn()):
		event.Type = EventWake
		event.Details = "kIOMessageSystemWillPowerOn"
	case uint32(C.getMsgSystemHasPoweredOn()):
		event.Type = EventWake
		event.Details = "kIOMessageSystemHasPoweredOn"
	default:
		return
	}

	if broadcast != nil {
		broadcast(event)
	}
}

type unsafePointer = unsafe.Pointer

type darwinCGOListener struct {
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
	l.mu.Unlock()

	errCh := make(chan error, 1)

	go func() {
		// CFRunLoop must run on an OS thread locked to the loop
		runtime.LockOSThread()
		defer runtime.UnlockOSThread()

		darwinCGOMu.Lock()
		darwinCGOBroadcast = func(e PowerEvent) {
			l.broadcaster.Broadcast(e)
		}
		darwinCGOMu.Unlock()

		var (
			np  C.IONotificationPortRef
			obj C.io_object_t
			rl  C.CFRunLoopRef
		)

		port := C.registerDarwinPowerNotifications(&np, &obj, &rl)
		if port == 0 {
			errCh <- fmt.Errorf("IORegisterForSystemPower failed")
			close(l.ready)
			close(l.done)
			return
		}

		l.mu.Lock()
		l.rootPort = port
		l.notifyPort = np
		l.notifierObj = obj
		l.runLoop = rl

		darwinCGOMu.Lock()
		darwinCGORootPort = port
		darwinCGOMu.Unlock()

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

		darwinCGOMu.Lock()
		darwinCGORootPort = 0
		darwinCGOMu.Unlock()

		C.deregisterDarwinPowerNotifications(rootPort, notifyPort, notifierObj, runLoop)
		l.mu.Unlock()

		close(l.done)
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
	l.mu.Unlock()

	if rl != 0 {
		C.stopDarwinRunLoop(rl)
	}

	select {
	case <-l.done:
	case <-time.After(2 * time.Second):
	}

	return nil
}
