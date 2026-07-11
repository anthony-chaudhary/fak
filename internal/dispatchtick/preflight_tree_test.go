package dispatchtick

import (
	"strings"
	"testing"
)

func TestEvaluatePreflightRefusesPoisonedTree(t *testing.T) {
	in := PreflightInput{MaxWorkers: 4, Host: HostCheck{Safe: true}, Tree: TreeCheck{Poisoned: true, Package: "github.com/x/broken"}, Account: AccountCheck{Available: true}, Kernel: KernelCheck{Alive: IntPtr(0), Target: IntPtr(0)}}
	got := EvaluatePreflight(in)
	if got.OK || got.Verdict != PreflightRefuseTreePoison || !strings.Contains(got.Reason, "github.com/x/broken") {
		t.Fatalf("got=%+v", got)
	}
}
func TestEvaluatePreflightGreenTreeContinues(t *testing.T) {
	in := PreflightInput{MaxWorkers: 4, Host: HostCheck{Safe: true}, Account: AccountCheck{Available: true}, Kernel: KernelCheck{Alive: IntPtr(0), Target: IntPtr(0)}}
	got := EvaluatePreflight(in)
	if got.Verdict == PreflightRefuseTreePoison {
		t.Fatalf("got=%+v", got)
	}
}
