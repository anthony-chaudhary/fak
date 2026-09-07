package sysproc

import (
	"context"
	"testing"
)

func TestNilSafety(t *testing.T) {
	// None of the configuration helpers should panic when passed a nil command.
	ConfigureBackground(nil)
	ConfigureDetached(nil)
	ConfigureProcessGroup(nil)
}

func TestCommand(t *testing.T) {
	cmd := Command("echo", "test")
	if cmd == nil {
		t.Fatal("Command returned nil")
	}
	assertCommandConfigured(t, cmd)
}

func TestCommandContext(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	cmd := CommandContext(ctx, "echo", "test")
	if cmd == nil {
		t.Fatal("CommandContext returned nil")
	}
	assertCommandConfigured(t, cmd)
}
