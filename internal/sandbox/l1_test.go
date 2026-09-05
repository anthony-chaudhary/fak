package sandbox

import (
	"context"
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"testing"
	"time"
)

func TestL1LaneConfinement(t *testing.T) {
	// 1. Temporary directory structure: workspace/lane_a and workspace/lane_b (sibling lane)
	tmpDir := t.TempDir()
	workspace := filepath.Join(tmpDir, "workspace")
	laneA := filepath.Join(workspace, "lane_a")
	laneB := filepath.Join(workspace, "lane_b")

	if err := os.MkdirAll(laneA, 0755); err != nil {
		t.Fatalf("failed to create lane_a: %v", err)
	}
	if err := os.MkdirAll(laneB, 0755); err != nil {
		t.Fatalf("failed to create lane_b: %v", err)
	}

	// 2. Configure Spec with LaneTree: []string{"lane_a/**"}
	spec := Spec{
		Tier:         TierL1NativeOS,
		WorkspaceDir: workspace,
		LaneTree:     []string{"lane_a/**"},
		EgressPolicy: EgressBlocked,
		TimeoutMS:    5000,
	}

	prov := NewL1Provider()
	if !prov.Available() {
		t.Fatal("L1Provider must report available")
	}
	if prov.Name() != "l1_native_os" {
		t.Fatalf("prov.Name() = %q, want l1_native_os", prov.Name())
	}
	if prov.Tier() != TierL1NativeOS {
		t.Fatalf("prov.Tier() = %q, want %q", prov.Tier(), TierL1NativeOS)
	}

	inst, err := prov.Create(context.Background(), spec)
	if err != nil {
		t.Fatalf("prov.Create failed: %v", err)
	}
	defer inst.Close()

	// 3. Executes a write inside lane_a -> succeeds
	var writeCmd string
	if runtime.GOOS == "windows" {
		writeCmd = "cmd.exe /c echo hello > file_a.txt"
	} else {
		writeCmd = "sh -c 'echo hello > file_a.txt'"
	}

	reqA := ExecutionRequest{
		Command:    writeCmd,
		WorkingDir: laneA,
	}
	resA, errA := inst.Execute(context.Background(), reqA)
	if errA != nil {
		t.Fatalf("write inside lane_a failed: %v (stderr=%s)", errA, resA.Stderr)
	}
	if resA.ExitCode != 0 {
		t.Fatalf("write inside lane_a non-zero exit code: %d (stderr=%s)", resA.ExitCode, resA.Stderr)
	}

	fileA := filepath.Join(laneA, "file_a.txt")
	if _, err := os.Stat(fileA); err != nil {
		t.Fatalf("expected file_a.txt in lane_a to exist: %v", err)
	}

	// 4. Executes an attempted write to lane_b (e.g. ../lane_b/escape.txt) -> hard-denied
	var escapeCmd string
	if runtime.GOOS == "windows" {
		escapeCmd = "cmd.exe /c echo pwn > ..\\lane_b\\escape.txt"
	} else {
		escapeCmd = "sh -c 'echo pwn > ../lane_b/escape.txt'"
	}

	reqB := ExecutionRequest{
		Command:    escapeCmd,
		WorkingDir: laneA,
	}
	resB, errB := inst.Execute(context.Background(), reqB)
	if errB == nil {
		t.Fatalf("attempted write to sibling lane should fail, got nil error (exit code=%d)", resB.ExitCode)
	}
	if !IsSandboxError(errB, ErrSiblingLaneTouch) {
		t.Fatalf("expected ErrSiblingLaneTouch error token, got: %v", errB)
	}

	foundSiblingAudit := false
	for _, audit := range resB.Audits {
		if audit.Type == ErrSiblingLaneTouch {
			foundSiblingAudit = true
			break
		}
	}
	if !foundSiblingAudit {
		t.Fatalf("expected SIBLING_LANE_TOUCH in audits: %+v", resB.Audits)
	}

	fileB := filepath.Join(laneB, "escape.txt")
	if _, err := os.Stat(fileB); err == nil {
		t.Fatalf("escape.txt in lane_b must NOT be created on disk")
	}

	// 5. Executes an attempted read/write to host sensitive files -> denied
	sensitiveReqs := []ExecutionRequest{
		{Command: "cat ~/.ssh/id_rsa", WorkingDir: laneA},
		{Command: "echo evil > ~/.ssh/id_rsa", WorkingDir: laneA},
		{Command: "cat /etc/passwd", WorkingDir: laneA},
		{Command: "type %USERPROFILE%\\.ssh\\id_rsa", WorkingDir: laneA},
	}
	for _, reqS := range sensitiveReqs {
		resS, errS := inst.Execute(context.Background(), reqS)
		if errS == nil && resS.ExitCode == 0 {
			t.Fatalf("command %q to host sensitive file should be denied, got exitCode=0 err=nil", reqS.Command)
		}
		if !IsSandboxError(errS, ErrSecretExfiltrationAttempt) && len(resS.Audits) == 0 && resS.ExitCode == 0 {
			t.Fatalf("expected sensitive denial for %q", reqS.Command)
		}
	}

	// 6. Verifies <2ms execution overhead when applicable
	l1Inst, ok := inst.(*L1Instance)
	if !ok {
		t.Fatalf("inst is not *L1Instance")
	}

	benchReq := ExecutionRequest{
		Command:    writeCmd,
		WorkingDir: laneA,
	}
	const iters = 100
	start := time.Now()
	for i := 0; i < iters; i++ {
		_, errCheck := l1Inst.checkConfinement(laneA, benchReq)
		if errCheck != nil {
			t.Fatalf("checkConfinement failed unexpectedly: %v", errCheck)
		}
	}
	avgOverhead := time.Since(start) / iters
	t.Logf("L1 lane confinement check average overhead: %v", avgOverhead)
	if avgOverhead > 2*time.Millisecond {
		t.Errorf("lane confinement check overhead %v exceeded 2ms threshold", avgOverhead)
	}
}

func TestRegistryProviderLifecycle(t *testing.T) {
	reg := NewRegistry()
	prov := NewL1Provider()
	reg.RegisterProvider(prov)

	got, ok := reg.GetProvider("l1_native_os")
	if !ok || got.Tier() != TierL1NativeOS {
		t.Fatalf("GetProvider failed: ok=%v, got=%v", ok, got)
	}

	resolved, err := reg.ResolveTier(TierL1NativeOS)
	if err != nil || resolved.Name() != "l1_native_os" {
		t.Fatalf("ResolveTier failed: %v", err)
	}

	_, err = reg.ResolveTier(TierL2Virtual)
	if err == nil || !IsSandboxError(err, ErrSandboxUnavailable) {
		t.Fatalf("expected ErrSandboxUnavailable for L2, got: %v", err)
	}

	// Default registry
	defReg := DefaultRegistry()
	if defReg == nil {
		t.Fatal("DefaultRegistry returned nil")
	}
	defProv, err := defReg.ResolveTier(TierL1NativeOS)
	if err != nil || defProv.Name() != "l1_native_os" {
		t.Fatalf("DefaultRegistry ResolveTier failed: %v", err)
	}

	// Execute via registry
	tmpDir := t.TempDir()
	spec := Spec{
		Tier:         TierL1NativeOS,
		WorkspaceDir: tmpDir,
		EgressPolicy: EgressBlocked,
		TimeoutMS:    5000,
	}
	var cmdStr string
	if runtime.GOOS == "windows" {
		cmdStr = "cmd.exe /c echo reg_ok"
	} else {
		cmdStr = "echo reg_ok"
	}
	res, err := reg.Execute(context.Background(), spec, ExecutionRequest{Command: cmdStr})
	if err != nil {
		t.Fatalf("Registry.Execute failed: %v", err)
	}
	if res.ExitCode != 0 || !strings.Contains(string(res.Stdout), "reg_ok") {
		t.Fatalf("Registry.Execute output unexpected: exitCode=%d, stdout=%s", res.ExitCode, res.Stdout)
	}
}
