package parentwatch

import (
	"context"
	"os"
	"os/exec"
	"testing"
	"time"
)

func TestParentAlive(t *testing.T) {
	if !ParentAlive(os.Getpid()) {
		t.Fatal("ParentAlive(self) = false, want true")
	}
	if ParentAlive(0) || ParentAlive(-1) {
		t.Fatal("ParentAlive(non-positive pid) = true, want false")
	}

	cmd := exec.Command("sh", "-c", "exit 0")
	if err := cmd.Start(); err != nil {
		t.Fatal(err)
	}
	pid := cmd.Process.Pid
	if err := cmd.Wait(); err != nil {
		t.Fatal(err)
	}
	if ParentAlive(pid) {
		t.Fatalf("ParentAlive(reaped pid %d) = true, want false", pid)
	}
}

func TestWatchLiveParent(t *testing.T) {
	cmd := exec.Command("sleep", "30")
	if err := cmd.Start(); err != nil {
		t.Fatal(err)
	}
	pid := cmd.Process.Pid

	ctx, stop := Watch(context.Background(), pid)
	defer stop()

	select {
	case <-ctx.Done():
		t.Fatal("ctx canceled while parent alive")
	case <-time.After(300 * time.Millisecond):
	}

	if err := cmd.Process.Kill(); err != nil {
		t.Fatal(err)
	}
	_ = cmd.Wait()

	select {
	case <-ctx.Done():
	case <-time.After(3 * time.Second):
		t.Fatal("ctx not canceled after parent death")
	}
}

func TestWatchAlreadyDeadParent(t *testing.T) {
	cmd := exec.Command("sh", "-c", "exit 0")
	if err := cmd.Start(); err != nil {
		t.Fatal(err)
	}
	pid := cmd.Process.Pid
	if err := cmd.Wait(); err != nil {
		t.Fatal(err)
	}

	ctx, stop := Watch(context.Background(), pid)
	defer stop()

	select {
	case <-ctx.Done():
	case <-time.After(500 * time.Millisecond):
		t.Fatal("ctx not canceled for already-dead parent")
	}
}

func TestWatchNoParent(t *testing.T) {
	parent := context.Background()
	ctx, stop := Watch(parent, 1)
	if ctx != parent {
		t.Fatal("Watch(pid<=1) returned a different context")
	}
	if ctx.Err() != nil {
		t.Fatal("Watch(pid<=1) context already canceled")
	}
	stop()
	stop()

	canceled, cancel := context.WithCancel(context.Background())
	cancel()
	ctx2, stop2 := Watch(canceled, 0)
	if ctx2 != canceled {
		t.Fatal("Watch(parentPid<=1) did not return parent")
	}
	if ctx2.Err() == nil {
		t.Fatal("returned parent ctx not canceled")
	}
	stop2()
	stop2()
}

func TestStopIdempotent(t *testing.T) {
	cmd := exec.Command("sleep", "30")
	if err := cmd.Start(); err != nil {
		t.Fatal(err)
	}
	defer func() {
		_ = cmd.Process.Kill()
		_ = cmd.Wait()
	}()

	_, stop := Watch(context.Background(), cmd.Process.Pid)
	stop()
	stop()
}
