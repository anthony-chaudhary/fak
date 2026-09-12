//go:build darwin && cgo

package power

import (
	"fmt"
	"reflect"
	"runtime/cgo"
	"testing"
)

// Injected native callback regression with an acknowledgement spy. This does
// not register with IOKit, suspend the machine, or acknowledge a real OS token.
func TestDarwinSuspendAcknowledgementOrder(t *testing.T) {
	b := NewPowerBroadcaster()
	l := newDarwinIOKitListener(b).(*darwinCGOListener)
	handle := cgo.NewHandle(l)
	defer handle.Delete()
	var trace []string
	var token int64
	acked := map[int64]bool{}
	l.ack = func(arg int64) {
		if acked[arg] {
			t.Errorf("duplicate acknowledgement for %d", arg)
		}
		acked[arg] = true
		trace = append(trace, fmt.Sprintf("ack:%d", arg))
	}
	panicOnSleep := false
	sentinel := &struct{}{}
	b.RegisterObserver(ObserverFunc{
		SuspendFn: func(e PowerEvent) {
			if acked[token] {
				t.Errorf("sleep observer entered after acknowledgement %d", token)
			}
			if e.Source != "iokit-cgo" || e.Details != "kIOMessageSystemWillSleep" || e.Timestamp.IsZero() {
				t.Errorf("unexpected committed sleep event: %+v", e)
			}
			trace = append(trace, "sleep:enter")
			if panicOnSleep {
				panic(sentinel)
			}
			trace = append(trace, "sleep:return")
		},
		ResumeFn: func(e PowerEvent) {
			if e.Source != "iokit-cgo" || e.Details != "kIOMessageSystemHasPoweredOn" {
				t.Errorf("unexpected wake: %+v", e)
			}
			trace = append(trace, "wake")
		},
	})
	for _, step := range []struct {
		name            string
		message         uint32
		argument        int64
		restart, panics bool
		want            []string
	}{
		{name: "wake without sleep", message: darwinHasPoweredOn},
		{name: "early power on", message: darwinWillPowerOn},
		{name: "query", message: darwinCanSleep, argument: 1, want: []string{"ack:1"}},
		{name: "committed sleep", message: darwinWillSleep, argument: 2, want: []string{"sleep:enter", "sleep:return", "ack:2"}},
		{name: "duplicate sleep", message: darwinWillSleep, argument: 3, want: []string{"ack:3"}},
		{name: "will power on", message: darwinWillPowerOn},
		{name: "duplicate while waking", message: darwinWillSleep, argument: 4, want: []string{"ack:4"}},
		{name: "powered on", message: darwinHasPoweredOn, want: []string{"wake"}},
		{name: "duplicate powered on", message: darwinHasPoweredOn},
		{name: "second sleep", message: darwinWillSleep, argument: 5, want: []string{"sleep:enter", "sleep:return", "ack:5"}},
		{name: "restart discards old sleep", restart: true, message: darwinHasPoweredOn},
		{name: "restarted sleep", message: darwinWillSleep, argument: 6, want: []string{"sleep:enter", "sleep:return", "ack:6"}},
		{name: "wake without will power on", message: darwinHasPoweredOn, want: []string{"wake"}},
		{name: "panic finalization", message: darwinWillSleep, argument: 7, panics: true, want: []string{"sleep:enter", "ack:7"}},
		{name: "duplicate after panic", message: darwinWillSleep, argument: 8, want: []string{"ack:8"}},
	} {
		t.Run(step.name, func(t *testing.T) {
			trace = nil
			token = step.argument
			panicOnSleep = step.panics
			if step.restart {
				l.resetCycle()
			}
			var recovered any
			func() {
				defer func() { recovered = recover() }()
				invokeDarwinCallback(handle, step.message, step.argument)
			}()
			if step.panics && recovered != sentinel {
				t.Errorf("panic was not propagated: %v", recovered)
			}
			if !step.panics && recovered != nil {
				t.Fatalf("unexpected panic: %v", recovered)
			}
			if !reflect.DeepEqual(trace, step.want) {
				t.Errorf("trace = %v, want %v", trace, step.want)
			}
		})
	}
	// A second callback identity must never redirect the first listener's ack.
	other := newDarwinIOKitListener(NewPowerBroadcaster()).(*darwinCGOListener)
	otherHandle := cgo.NewHandle(other)
	defer otherHandle.Delete()
	var otherAck int64
	other.ack = func(arg int64) { otherAck = arg }
	invokeDarwinCallback(otherHandle, darwinCanSleep, 9)
	invokeDarwinCallback(handle, darwinCanSleep, 10)
	if otherAck != 9 || !acked[10] || acked[9] {
		t.Errorf("callback identity crossed listeners: other=%d first=%v", otherAck, acked)
	}
}
