package processstart

import (
	"errors"
	"math"
	"testing"
	"time"
)

func TestDarwinProcessStartParsesKernelIdentity(t *testing.T) {
	const pid = 4312
	want := time.Date(2026, time.August, 26, 17, 24, 51, 654321000, time.UTC)

	got, ok := darwinProcessStart(pid, func(queriedPID int) (darwinProcessInfo, error) {
		if queriedPID != pid {
			t.Fatalf("queried pid = %d, want %d", queriedPID, pid)
		}
		return darwinProcessInfo{
			pid:       pid,
			startSec:  want.Unix(),
			startUsec: int64(want.Nanosecond() / int(time.Microsecond)),
		}, nil
	})
	if !ok {
		t.Fatal("darwinProcessStart rejected valid kernel identity")
	}
	if !got.Equal(want) {
		t.Fatalf("darwinProcessStart = %v, want %v", got, want)
	}
	if got.Location() != time.UTC {
		t.Fatalf("darwinProcessStart location = %v, want UTC", got.Location())
	}
}

func TestDarwinProcessStartRejectsInvalidPIDBeforeRead(t *testing.T) {
	type testCase struct {
		name string
		pid  int
	}
	tests := []testCase{
		{name: "negative", pid: -1},
		{name: "zero", pid: 0},
	}
	if math.MaxInt > math.MaxInt32 {
		largerThanPIDT := int64(math.MaxInt32) + 1
		tests = append(tests, testCase{name: "larger than pid_t", pid: int(largerThanPIDT)})
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			called := false
			if got, ok := darwinProcessStart(tt.pid, func(int) (darwinProcessInfo, error) {
				called = true
				return darwinProcessInfo{}, nil
			}); ok || !got.IsZero() {
				t.Fatalf("darwinProcessStart(%d) = %v, %v; want zero, false", tt.pid, got, ok)
			}
			if called {
				t.Fatalf("darwinProcessStart(%d) read a kernel record", tt.pid)
			}
		})
	}
}

func TestDarwinProcessStartRejectsUnusableKernelIdentity(t *testing.T) {
	const pid = 4312
	tests := []struct {
		name string
		info darwinProcessInfo
		err  error
	}{
		{name: "read error", err: errors.New("sysctl failed")},
		{name: "missing record", info: darwinProcessInfo{}},
		{name: "different pid", info: darwinProcessInfo{pid: pid + 1, startSec: 1, startUsec: 2}},
		{name: "zero seconds", info: darwinProcessInfo{pid: pid, startUsec: 2}},
		{name: "negative seconds", info: darwinProcessInfo{pid: pid, startSec: -1, startUsec: 2}},
		{name: "negative microseconds", info: darwinProcessInfo{pid: pid, startSec: 1, startUsec: -1}},
		{name: "overflow microseconds", info: darwinProcessInfo{pid: pid, startSec: 1, startUsec: 1_000_000}},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, ok := darwinProcessStart(pid, func(int) (darwinProcessInfo, error) {
				return tt.info, tt.err
			})
			if ok || !got.IsZero() {
				t.Fatalf("darwinProcessStart = %v, %v; want zero, false", got, ok)
			}
		})
	}
}
