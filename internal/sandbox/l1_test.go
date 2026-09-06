package sandbox

import (
	"context"
	"encoding/json"
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/abi"
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

// ---------------------------------------------------------------------------
// L1 BENCHMARK SUITE: PROFILE VALIDATION, ISOLATION & CAPABILITIES
// ---------------------------------------------------------------------------

// BenchmarkL1ProfileValidation measures L1 sandbox specification validation,
// capability envelope checks, and configuration serialization overhead.
func BenchmarkL1ProfileValidation(b *testing.B) {
	validSpec := Spec{
		Tier:             TierL1NativeOS,
		Rootfs:           "/var/lib/fak/rootfs/alpine",
		WorkspaceDir:     "/work/repo",
		LaneTree:         []string{"internal/sandbox/**", "pkg/abi/**"},
		ReadOnlyPaths:    []string{"/usr", "/lib", "/bin"},
		WritablePaths:    []string{"/tmp", "/work/repo/scratch"},
		MemoryLimitBytes: 512 * 1024 * 1024,
		CPULimitPercent:  80,
		FuelLimit:        1000000,
		TimeoutMS:        10000,
		Env:              []string{"LANG=C.UTF-8", "FAK_SANDBOX=1"},
		EgressPolicy:     EgressBlocked,
		Capabilities:     []abi.Capability{"fs.read", "fs.write", "proc.exec"},
	}

	minimalSpec := Spec{
		Tier:         TierL1NativeOS,
		WorkspaceDir: "/work/repo",
		EgressPolicy: EgressBlocked,
	}

	invalidSpec := Spec{
		Tier:         "invalid_tier",
		WorkspaceDir: "",
		EgressPolicy: "",
	}

	b.Run("ValidFullSpec", func(b *testing.B) {
		b.ReportAllocs()
		for i := 0; i < b.N; i++ {
			if err := validSpec.Validate(); err != nil {
				b.Fatal(err)
			}
		}
	})

	b.Run("ValidMinimalSpec", func(b *testing.B) {
		b.ReportAllocs()
		for i := 0; i < b.N; i++ {
			if err := minimalSpec.Validate(); err != nil {
				b.Fatal(err)
			}
		}
	})

	b.Run("InvalidRejection", func(b *testing.B) {
		b.ReportAllocs()
		for i := 0; i < b.N; i++ {
			_ = invalidSpec.Validate()
		}
	})

	b.Run("JSONRoundtrip", func(b *testing.B) {
		b.ReportAllocs()
		data, err := json.Marshal(validSpec)
		if err != nil {
			b.Fatal(err)
		}
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			var decoded Spec
			if err := json.Unmarshal(data, &decoded); err != nil {
				b.Fatal(err)
			}
		}
	})
}

// BenchmarkL1ExecutionIsolation measures L1 isolation enforcement, confinement verification,
// boundary escape refusal latency, and command execution overhead.
func BenchmarkL1ExecutionIsolation(b *testing.B) {
	tmpDir := b.TempDir()
	workspace := filepath.Join(tmpDir, "workspace")
	laneA := filepath.Join(workspace, "lane_a")
	laneB := filepath.Join(workspace, "lane_b")
	_ = os.MkdirAll(laneA, 0755)
	_ = os.MkdirAll(laneB, 0755)

	spec := Spec{
		Tier:         TierL1NativeOS,
		WorkspaceDir: workspace,
		LaneTree:     []string{"lane_a/**"},
		EgressPolicy: EgressBlocked,
		TimeoutMS:    5000,
	}

	prov := NewL1Provider()
	inst, err := prov.Create(context.Background(), spec)
	if err != nil {
		b.Fatalf("prov.Create failed: %v", err)
	}
	defer inst.Close()

	l1Inst, ok := inst.(*L1Instance)
	if !ok {
		b.Fatalf("inst is not *L1Instance")
	}

	var validCmd string
	if runtime.GOOS == "windows" {
		validCmd = "cmd.exe /c echo hello > file_a.txt"
	} else {
		validCmd = "sh -c 'echo hello > file_a.txt'"
	}
	reqValid := ExecutionRequest{
		Command:    validCmd,
		WorkingDir: laneA,
	}

	reqEscape := ExecutionRequest{
		Command:    "cat ../../../etc/shadow",
		WorkingDir: laneA,
	}

	reqSibling := ExecutionRequest{
		Command:    "echo pwn > ../lane_b/escape.txt",
		WorkingDir: laneA,
	}

	b.Run("ConfinementCheck_AllowedInLane", func(b *testing.B) {
		b.ReportAllocs()
		for i := 0; i < b.N; i++ {
			audit, err := l1Inst.checkConfinement(laneA, reqValid)
			if err != nil || audit != nil {
				b.Fatalf("unexpected refusal: %v", err)
			}
		}
	})

	b.Run("ConfinementCheck_RefuseWorkspaceEscape", func(b *testing.B) {
		b.ReportAllocs()
		for i := 0; i < b.N; i++ {
			_, err := l1Inst.checkConfinement(laneA, reqEscape)
			if err == nil {
				b.Fatal("expected escape refusal, got nil")
			}
		}
	})

	b.Run("ConfinementCheck_RefuseSiblingLaneTouch", func(b *testing.B) {
		b.ReportAllocs()
		for i := 0; i < b.N; i++ {
			_, err := l1Inst.checkConfinement(laneA, reqSibling)
			if err == nil {
				b.Fatal("expected sibling lane refusal, got nil")
			}
		}
	})

	b.Run("CommandPreparation", func(b *testing.B) {
		b.ReportAllocs()
		ctx := context.Background()
		for i := 0; i < b.N; i++ {
			cmd := prepareCommand(ctx, reqValid)
			if cmd == nil {
				b.Fatal("cmd is nil")
			}
		}
	})

	b.Run("InstanceLifecycleCreateClose", func(b *testing.B) {
		b.ReportAllocs()
		ctx := context.Background()
		for i := 0; i < b.N; i++ {
			createdInst, err := prov.Create(ctx, spec)
			if err != nil {
				b.Fatalf("prov.Create failed: %v", err)
			}
			if err := createdInst.Close(); err != nil {
				b.Fatalf("inst.Close failed: %v", err)
			}
		}
	})

	b.Run("EndToEndProcessExecution", func(b *testing.B) {
		b.ReportAllocs()
		ctx := context.Background()
		var echoCmd string
		if runtime.GOOS == "windows" {
			echoCmd = "cmd.exe /c echo isolation_bench_ok"
		} else {
			echoCmd = "echo isolation_bench_ok"
		}
		req := ExecutionRequest{
			Command:    echoCmd,
			WorkingDir: laneA,
		}
		for i := 0; i < b.N; i++ {
			res, err := inst.Execute(ctx, req)
			if err != nil {
				b.Fatalf("inst.Execute failed: %v", err)
			}
			if res.ExitCode != 0 {
				b.Fatalf("res.ExitCode = %d, want 0", res.ExitCode)
			}
		}
	})
}

// BenchmarkL1CapabilityRestriction measures capability envelope checks, sensitive path detection,
// lane glob matching, candidate path parsing, and output normalization.
func BenchmarkL1CapabilityRestriction(b *testing.B) {
	b.Run("EnvelopeCapabilityCheck", func(b *testing.B) {
		b.ReportAllocs()
		envelope := CapabilityEnvelope{
			Capabilities: []abi.Capability{
				"fs.read", "fs.write", "proc.exec", "sys.info", "net.dial", "ipc.shm",
			},
		}
		presentCap := abi.Capability("proc.exec")
		missingCap := abi.Capability("kernel.module")
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			if !envelope.HasCapability(presentCap) {
				b.Fatal("expected presentCap to be found")
			}
			if envelope.HasCapability(missingCap) {
				b.Fatal("expected missingCap to be absent")
			}
		}
	})

	b.Run("SensitivePathInspection", func(b *testing.B) {
		b.ReportAllocs()
		paths := []string{
			"src/main.go",
			"~/.ssh/id_rsa",
			"/etc/passwd",
			".aws/credentials",
			"C:\\Windows\\System32\\config\\SAM",
			".bash_history",
			"docs/README.md",
		}
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			for _, p := range paths {
				_, _ = isSensitiveHostPath(p)
			}
		}
	})

	b.Run("LaneTreeGlobMatching", func(b *testing.B) {
		b.ReportAllocs()
		laneTree := []string{
			"internal/sandbox/**",
			"platform/installer/*",
			"tools/dgxbridge/**",
			"pkg/abi/types.go",
		}
		testPaths := []string{
			"internal/sandbox/l1_engine.go",
			"platform/installer/usb.go",
			"platform/installer/sub/deep.go",
			"pkg/abi/types.go",
			"cmd/fak/main.go",
		}
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			for _, p := range testPaths {
				_ = isLaneAllowed(laneTree, p)
			}
		}
	})

	b.Run("CandidatePathExtraction", func(b *testing.B) {
		b.ReportAllocs()
		command := "go test -v -coverprofile=coverage.out ./internal/sandbox > test.log 2>&1"
		argv := []string{"--workdir", "C:\\work\\fak\\internal\\sandbox", "<", "input.dat"}
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			paths := extractCandidatePaths(command, argv)
			if len(paths) == 0 {
				b.Fatal("expected extracted paths")
			}
		}
	})

	b.Run("OutputNormalization", func(b *testing.B) {
		b.ReportAllocs()
		rawOutput := []byte("\x1b[32mPASS\x1b[0m: test completed in C:\\work\\fak\\workspace\\test_lane\r\n\x1b[31;1mERROR\x1b[0m: none   \r\n")
		ws := "C:\\work\\fak\\workspace"
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			normalized := NormalizeOutput(rawOutput, ws)
			if len(normalized) == 0 {
				b.Fatal("expected normalized output")
			}
		}
	})
}
