package compute

import (
	"errors"
	"fmt"
	"io/fs"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func fusedCount(op, mode string) uint64 {
	for _, c := range FusedCounts() {
		if c.Op == op && c.Mode == mode {
			return c.Count
		}
	}
	return 0
}

// fak-test:runtime fast est=2ms lane=default
func TestFusedShaderMissingIsATypedInitWarning(t *testing.T) {
	dir := t.TempDir()
	if err := os.WriteFile(filepath.Join(dir, "rmsnorm_q4k_matmul2.spv"), []byte{0x03, 0x02, 0x23, 0x07}, 0o600); err != nil {
		t.Fatal(err)
	}
	present, missing := probeFusedShaders(dir, os.Stat)
	if !present[FusedOpQ4KRMSNormMatMul2] || present[FusedOpQ4KSwiGLUMatMulAdd] {
		t.Fatalf("present=%v, want only the rmsnorm fixture", present)
	}
	if len(missing) != 1 || missing[0].Op != FusedOpQ4KSwiGLUMatMulAdd {
		t.Fatalf("missing=%v, want the swiglu shader", missing)
	}
	if !errors.Is(missing[0], ErrFusedShaderMissing) || !errors.Is(missing[0], fs.ErrNotExist) {
		t.Fatalf("warning %v does not carry ErrFusedShaderMissing and fs.ErrNotExist", missing[0])
	}

	prev := FusedShaderWarnings()
	t.Cleanup(func() { recordFusedShaderWarnings(prev, func(string, ...any) {}) })
	var logged []string
	recordFusedShaderWarnings(missing, func(format string, args ...any) { logged = append(logged, fmt.Sprintf(format, args...)) })
	if len(logged) != 1 || !strings.Contains(logged[0], FusedOpQ4KSwiGLUMatMulAdd) {
		t.Fatalf("init log=%q, want one warning naming the op", logged)
	}
	if got := FusedShaderWarnings(); len(got) != 1 || got[0].Op != FusedOpQ4KSwiGLUMatMulAdd {
		t.Fatalf("recorded warnings=%v", got)
	}
}

// fak-test:runtime fast est=1ms lane=default
func TestFusedCounterSplitsFusedFromComposedPerOp(t *testing.T) {
	DeclareFusedOps()
	fused, composed := fusedCount(FusedOpQ4KSwiGLUMatMulAdd, "fused"), fusedCount(FusedOpQ4KSwiGLUMatMulAdd, "composed")
	other := fusedCount(FusedOpGDNQ8InProj, "composed")
	CountFusedDispatch(FusedOpQ4KSwiGLUMatMulAdd, false)
	CountFusedDispatch(FusedOpQ4KSwiGLUMatMulAdd, false)
	CountFusedDispatch(FusedOpQ4KSwiGLUMatMulAdd, true)
	CountFusedDispatch("not_a_fused_op", true)
	if got := fusedCount(FusedOpQ4KSwiGLUMatMulAdd, "composed"); got != composed+2 {
		t.Fatalf("composed=%d, want %d", got, composed+2)
	}
	if got := fusedCount(FusedOpQ4KSwiGLUMatMulAdd, "fused"); got != fused+1 {
		t.Fatalf("fused=%d, want %d", got, fused+1)
	}
	if got := fusedCount(FusedOpGDNQ8InProj, "composed"); got != other {
		t.Fatalf("another op moved: %d -> %d", other, got)
	}
	if n := len(FusedCounts()); n != 2*len(fusedOps) {
		t.Fatalf("series=%d, want %d (closed op set, both modes)", n, 2*len(fusedOps))
	}
}

// fak-test:runtime fast est=1ms lane=default
func TestFusedCounterDefaultOffFusionsReportOptIn(t *testing.T) {
	t.Setenv("FAK_VULKAN_Q4K_FUSION", "")
	t.Setenv("FAK_VULKAN_Q4K_ARM", "fused")
	for _, arm := range DefaultOffFusions() {
		if strings.HasPrefix(arm.Op, "q4k_") && !arm.Enabled {
			t.Fatalf("%s: FAK_VULKAN_Q4K_ARM=fused not reported as enabled", arm.Op)
		}
	}
	t.Setenv("FAK_VULKAN_Q4K_ARM", "scalar")
	for _, arm := range DefaultOffFusions() {
		if strings.HasPrefix(arm.Op, "q4k_") && arm.Enabled {
			t.Fatalf("%s: scalar reported as enabled", arm.Op)
		}
	}
}
