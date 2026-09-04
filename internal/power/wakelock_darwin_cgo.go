//go:build darwin && cgo

package power

/*
#cgo LDFLAGS: -framework IOKit -framework CoreFoundation
#include <CoreFoundation/CoreFoundation.h>
#include <IOKit/pwr_mgt/IOPMLib.h>

static IOPMAssertionID createDarwinAssertion(int displaySleep, const char* reason, IOReturn* ret) {
	CFStringRef cfType = displaySleep ?
		kIOPMAssertionTypePreventUserIdleDisplaySleep :
		kIOPMAssertionTypePreventUserIdleSystemSleep;
	CFStringRef cfReason = CFStringCreateWithCString(kCFAllocatorDefault, reason, kCFStringEncodingUTF8);
	IOPMAssertionID assertionID = kIOPMNullAssertionID;
	*ret = IOPMAssertionCreateWithName(cfType, kIOPMAssertionLevelOn, cfReason, &assertionID);
	if (cfReason) {
		CFRelease(cfReason);
	}
	return assertionID;
}

static IOReturn releaseDarwinAssertion(IOPMAssertionID assertionID) {
	return IOPMAssertionRelease(assertionID);
}
*/
import "C"

import (
	"fmt"
	"sync"
	"unsafe"
)

type iokitLock struct {
	id     C.IOPMAssertionID
	mu     sync.Mutex
	closed bool
}

func acquireDarwinIOKit(reason string, flags WakeFlags) (platformLock, error) {
	cReason := C.CString(reason)
	defer C.free(unsafe.Pointer(cReason))

	displaySleep := 0
	if flags&PreventDisplaySleep != 0 {
		displaySleep = 1
	}

	var ret C.IOReturn
	assertionID := C.createDarwinAssertion(C.int(displaySleep), cReason, &ret)
	if ret != C.kIOReturnSuccess || assertionID == C.kIOPMNullAssertionID {
		return nil, fmt.Errorf("IOPMAssertionCreateWithName failed: return code %d", int(ret))
	}

	return &iokitLock{id: assertionID}, nil
}

func (l *iokitLock) Release() error {
	l.mu.Lock()
	defer l.mu.Unlock()
	if l.closed {
		return nil
	}
	l.closed = true
	ret := C.releaseDarwinAssertion(l.id)
	if ret != C.kIOReturnSuccess {
		return fmt.Errorf("IOPMAssertionRelease failed: return code %d", int(ret))
	}
	return nil
}
