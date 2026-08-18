package childprocess

import (
	"errors"
	"os/exec"
	"runtime"
	"testing"
)

func TestExitCode(t *testing.T) {
	if got := ExitCode(nil, 127); got != 0 {
		t.Fatalf("nil = %d", got)
	}
	if got := ExitCode(errors.New("launch"), 127); got != 127 {
		t.Fatalf("launch = %d", got)
	}
	if runtime.GOOS == "windows" {
		return
	}
	err := exec.Command("sh", "-c", "exit 7").Run()
	if got := ExitCode(err, 127); got != 7 {
		t.Fatalf("child exit = %d", got)
	}
}
