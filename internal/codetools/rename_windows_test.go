//go:build windows

package codetools

import (
	"errors"
	"fmt"
	"reflect"
	"syscall"
	"testing"
	"time"
)

func TestRenameReplacingWithRetriesTransientErrorsThenSucceeds(t *testing.T) {
	wantErrs := []error{
		fmt.Errorf("held by scanner: %w", syscall.Errno(5)),
		fmt.Errorf("held by indexer: %w", syscall.Errno(32)),
		nil,
	}
	var calls int
	var sleeps []time.Duration
	err := renameReplacingWith("old", "new", func(oldPath, newPath string) error {
		if oldPath != "old" || newPath != "new" {
			t.Fatalf("rename paths = %q, %q", oldPath, newPath)
		}
		err := wantErrs[calls]
		calls++
		return err
	}, func(delay time.Duration) {
		sleeps = append(sleeps, delay)
	})
	if err != nil {
		t.Fatalf("renameReplacingWith() error = %v", err)
	}
	if calls != 3 {
		t.Fatalf("rename calls = %d, want 3", calls)
	}
	if want := []time.Duration{time.Millisecond, 2 * time.Millisecond}; !reflect.DeepEqual(sleeps, want) {
		t.Fatalf("sleeps = %v, want %v", sleeps, want)
	}
}

func TestRenameReplacingWithStopsAfterBoundedTransientRetries(t *testing.T) {
	wantErr := fmt.Errorf("still held: %w", syscall.Errno(32))
	var calls int
	var sleeps []time.Duration
	err := renameReplacingWith("old", "new", func(string, string) error {
		calls++
		return wantErr
	}, func(delay time.Duration) {
		sleeps = append(sleeps, delay)
	})
	if !errors.Is(err, syscall.Errno(32)) || err != wantErr {
		t.Fatalf("error = %v, want final transient error %v", err, wantErr)
	}
	if calls != 6 {
		t.Fatalf("rename calls = %d, want 6", calls)
	}
	if want := []time.Duration{time.Millisecond, 2 * time.Millisecond, 4 * time.Millisecond, 8 * time.Millisecond, 16 * time.Millisecond}; !reflect.DeepEqual(sleeps, want) {
		t.Fatalf("sleeps = %v, want %v", sleeps, want)
	}
}

func TestRenameReplacingWithReturnsPermanentErrorImmediately(t *testing.T) {
	wantErr := errors.New("permission policy denied rename")
	var calls int
	var sleeps []time.Duration
	err := renameReplacingWith("old", "new", func(string, string) error {
		calls++
		return wantErr
	}, func(delay time.Duration) {
		sleeps = append(sleeps, delay)
	})
	if err != wantErr {
		t.Fatalf("error = %v, want %v", err, wantErr)
	}
	if calls != 1 {
		t.Fatalf("rename calls = %d, want 1", calls)
	}
	if len(sleeps) != 0 {
		t.Fatalf("sleeps = %v, want none", sleeps)
	}
}

func TestTransientWindowsRenameErrorRecognizesOnlyWrappedRetryableErrnos(t *testing.T) {
	tests := []struct {
		name string
		err  error
		want bool
	}{
		{name: "access denied", err: fmt.Errorf("rename: %w", syscall.Errno(5)), want: true},
		{name: "sharing violation", err: fmt.Errorf("rename: %w", syscall.Errno(32)), want: true},
		{name: "file not found", err: fmt.Errorf("rename: %w", syscall.Errno(2)), want: false},
		{name: "ordinary error", err: errors.New("rename failed"), want: false},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := transientWindowsRenameError(tt.err); got != tt.want {
				t.Fatalf("transientWindowsRenameError(%v) = %v, want %v", tt.err, got, tt.want)
			}
		})
	}
}
