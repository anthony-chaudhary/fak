//go:build windows

package main

import (
	"github.com/anthony-chaudhary/fak/internal/hostresurrect"
	"os"
	"testing"
)

func TestLaunchHostSessionPlatformQueuesBeforeInteractiveBroker(t *testing.T) {
	dir := t.TempDir()
	t.Setenv("FAK_HOST_RELAUNCH_DIR", dir)
	old := runInteractiveBrokerTask
	called := false
	runInteractiveBrokerTask = func(task string) error {
		called = true
		if task != "FakHostRelaunchBroker" {
			t.Fatalf("task=%q", task)
		}
		return os.ErrNotExist
	}
	t.Cleanup(func() { runInteractiveBrokerTask = old })
	req := hostresurrect.Request{Schema: hostresurrect.Schema, EventID: "evt", Session: "g1", CWD: `C:\work`, Command: []string{"claude"}, ResumeHandle: "g1"}
	if _, err := launchHostSessionPlatform(req); err != nil {
		t.Fatal(err)
	}
	if !called {
		t.Fatal("broker not signaled")
	}
	pending, err := hostresurrect.Pending(dir)
	if err != nil || len(pending) != 1 {
		t.Fatalf("pending=%v err=%v", pending, err)
	}
	got, err := hostresurrect.ReadQueued(pending[0])
	if err != nil || got.Session != "g1" {
		t.Fatalf("got=%+v err=%v", got, err)
	}
}
