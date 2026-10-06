package main

import (
	"context"
	"errors"
	"reflect"
	"strings"
	"testing"
	"time"
)

type fakeLaunchctl struct {
	calls [][]string
	// respond returns (output, err) for a call; nil means success.
	respond func(args []string) ([]byte, error)
}

func (f *fakeLaunchctl) agent() launchdAgent {
	return launchdAgent{
		uid:   501,
		sleep: func(time.Duration) {},
		run: func(_ context.Context, args ...string) ([]byte, error) {
			f.calls = append(f.calls, append([]string(nil), args...))
			if f.respond != nil {
				return f.respond(args)
			}
			return nil, nil
		},
	}
}

func TestLaunchdModernBootstrapArgvSequence(t *testing.T) {
	f := &fakeLaunchctl{respond: func(args []string) ([]byte, error) {
		if args[0] == "bootout" {
			return []byte("Boot-out failed: 3: No such process"), errors.New("exit status 3")
		}
		return nil, nil
	}}
	if err := f.agent().Bootstrap(context.Background(), "com.fak.x", "/p/com.fak.x.plist", true); err != nil {
		t.Fatalf("Bootstrap: %v", err)
	}
	want := [][]string{
		{"bootout", "gui/501/com.fak.x"},
		{"enable", "gui/501/com.fak.x"},
		{"bootstrap", "gui/501", "/p/com.fak.x.plist"},
		{"kickstart", "-k", "gui/501/com.fak.x"},
	}
	if !reflect.DeepEqual(f.calls, want) {
		t.Fatalf("argv sequence:\n got %q\nwant %q", f.calls, want)
	}
}

func TestLaunchdModernBootstrapFailureReturned(t *testing.T) {
	f := &fakeLaunchctl{respond: func(args []string) ([]byte, error) {
		if args[0] == "bootstrap" {
			return []byte("Bootstrap failed: 5: Input/output error"), errors.New("exit status 5")
		}
		return nil, nil
	}}
	err := f.agent().Bootstrap(context.Background(), "com.fak.x", "/p/x.plist", true)
	if err == nil || !strings.Contains(err.Error(), "Input/output error") || !strings.Contains(err.Error(), "bootstrap gui/501 /p/x.plist") {
		t.Fatalf("want surfaced bootstrap error with output, got %v", err)
	}
	// One retry, and no kickstart after a failed bootstrap.
	want := [][]string{
		{"bootout", "gui/501/com.fak.x"},
		{"enable", "gui/501/com.fak.x"},
		{"bootstrap", "gui/501", "/p/x.plist"},
		{"bootstrap", "gui/501", "/p/x.plist"},
	}
	if !reflect.DeepEqual(f.calls, want) {
		t.Fatalf("argv sequence:\n got %q\nwant %q", f.calls, want)
	}
}

func TestLaunchdModernBootstrapRetrySucceeds(t *testing.T) {
	n := 0
	f := &fakeLaunchctl{respond: func(args []string) ([]byte, error) {
		if args[0] == "bootstrap" {
			n++
			if n == 1 {
				return []byte("Bootstrap failed: 5: Input/output error"), errors.New("exit status 5")
			}
		}
		return nil, nil
	}}
	if err := f.agent().Bootstrap(context.Background(), "l", "/p", false); err != nil {
		t.Fatalf("transient bootstrap failure should be retried: %v", err)
	}
	if n != 2 {
		t.Fatalf("bootstrap attempts = %d, want 2", n)
	}
}

func TestLaunchdModernBootoutRealFailureReturned(t *testing.T) {
	f := &fakeLaunchctl{respond: func(args []string) ([]byte, error) {
		return []byte("Boot-out failed: 1: Operation not permitted"), errors.New("exit status 1")
	}}
	a := f.agent()
	if err := a.Bootout(context.Background(), "l"); err == nil {
		t.Fatal("Bootout should surface a non-not-loaded failure")
	}
	if err := a.Bootstrap(context.Background(), "l", "/p", false); err == nil {
		t.Fatal("Bootstrap should stop on a real bootout failure")
	}
}

func TestLaunchdModernLoaded(t *testing.T) {
	for _, tc := range []struct {
		name    string
		out     string
		err     error
		loaded  bool
		wantErr bool
	}{
		{"loaded", "gui/501/l = {\n state = running\n}", nil, true, false},
		{"missing", "Could not find service \"l\" in domain for port", errors.New("exit status 113"), false, false},
		{"other", "boom", errors.New("exit status 1"), false, true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			f := &fakeLaunchctl{respond: func([]string) ([]byte, error) { return []byte(tc.out), tc.err }}
			got, err := f.agent().Loaded(context.Background(), "l")
			if got != tc.loaded || (err != nil) != tc.wantErr {
				t.Fatalf("Loaded = %v, %v; want %v, err=%v", got, err, tc.loaded, tc.wantErr)
			}
			if want := [][]string{{"print", "gui/501/l"}}; !reflect.DeepEqual(f.calls, want) {
				t.Fatalf("argv: got %q want %q", f.calls, want)
			}
		})
	}
}
