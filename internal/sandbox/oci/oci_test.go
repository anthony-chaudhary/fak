package oci

import (
	"context"
	"encoding/json"
	"errors"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/abi"
	"github.com/anthony-chaudhary/fak/internal/agent"
	"github.com/anthony-chaudhary/fak/internal/refutil"
	"github.com/anthony-chaudhary/fak/internal/sandbox"
)

func TestOCIBundleGeneration(t *testing.T) {
	bundleDir := t.TempDir()
	wsDir := t.TempDir()

	spec := sandbox.Spec{
		Tier:             sandbox.TierL2Virtual,
		WorkspaceDir:     wsDir,
		EgressPolicy:     sandbox.EgressBlocked,
		MemoryLimitBytes: 256 * 1024 * 1024,
		CPULimitPercent:  50,
		ReadOnlyPaths:    []string{"/etc/custom_config"},
		WritablePaths:    []string{"/var/log/app"},
	}
	req := sandbox.ExecutionRequest{
		Command: "python -m script.py --arg1 testval",
	}

	err := GenerateBundle(spec, req, bundleDir)
	if err != nil {
		t.Fatalf("GenerateBundle returned error: %v", err)
	}

	configPath := filepath.Join(bundleDir, "config.json")
	data, err := os.ReadFile(configPath)
	if err != nil {
		t.Fatalf("failed to read config.json: %v", err)
	}

	var ociSpec Spec
	if err := json.Unmarshal(data, &ociSpec); err != nil {
		t.Fatalf("failed to unmarshal config.json into Spec: %v", err)
	}

	// 1. ociVersion
	if ociSpec.OCIVersion != "1.0.2" {
		t.Errorf("OCIVersion = %q, want '1.0.2'", ociSpec.OCIVersion)
	}

	// 2. Root
	if ociSpec.Root == nil {
		t.Fatal("Root is nil")
	}
	if !ociSpec.Root.Readonly {
		t.Errorf("Root.Readonly = false, want true")
	}
	if ociSpec.Root.Path != "rootfs" {
		t.Errorf("Root.Path = %q, want 'rootfs'", ociSpec.Root.Path)
	}

	// 3. Process
	if ociSpec.Process == nil {
		t.Fatal("Process is nil")
	}
	if ociSpec.Process.Terminal {
		t.Errorf("Process.Terminal = true, want false")
	}
	if ociSpec.Process.User.UID != 1000 || ociSpec.Process.User.GID != 1000 {
		t.Errorf("Process.User = %+v, want UID=1000, GID=1000", ociSpec.Process.User)
	}
	expectedArgs := []string{"python", "-m", "script.py", "--arg1", "testval"}
	if len(ociSpec.Process.Args) != len(expectedArgs) {
		t.Fatalf("Process.Args = %v, want %v", ociSpec.Process.Args, expectedArgs)
	}
	for i, a := range expectedArgs {
		if ociSpec.Process.Args[i] != a {
			t.Errorf("Process.Args[%d] = %q, want %q", i, ociSpec.Process.Args[i], a)
		}
	}
	if ociSpec.Process.Cwd != "/workspace" {
		t.Errorf("Process.Cwd = %q, want '/workspace'", ociSpec.Process.Cwd)
	}

	// 4. Capabilities (minimal set, no CAP_SYS_ADMIN)
	if ociSpec.Process.Capabilities == nil {
		t.Fatal("Process.Capabilities is nil")
	}
	for _, capName := range ociSpec.Process.Capabilities.Bounding {
		if capName == "CAP_SYS_ADMIN" {
			t.Errorf("bounding capabilities contains CAP_SYS_ADMIN")
		}
	}
	for _, capName := range ociSpec.Process.Capabilities.Effective {
		if capName == "CAP_SYS_ADMIN" {
			t.Errorf("effective capabilities contains CAP_SYS_ADMIN")
		}
	}
	for _, capName := range ociSpec.Process.Capabilities.Permitted {
		if capName == "CAP_SYS_ADMIN" {
			t.Errorf("permitted capabilities contains CAP_SYS_ADMIN")
		}
	}
	if len(ociSpec.Process.Capabilities.Bounding) == 0 {
		t.Errorf("bounding capabilities is empty")
	}

	// 5. Mounts
	mountMap := make(map[string]Mount)
	for _, m := range ociSpec.Mounts {
		mountMap[m.Destination] = m
	}

	requiredMounts := []string{"/proc", "/dev/pts", "/dev/shm", "/dev/mqueue", "/sys", "/tmp", "/workspace"}
	for _, rm := range requiredMounts {
		if _, ok := mountMap[rm]; !ok {
			t.Errorf("missing required mount destination: %q", rm)
		}
	}

	wsMount, ok := mountMap["/workspace"]
	if !ok {
		t.Fatal("missing /workspace mount")
	}
	if wsMount.Source != filepath.Clean(wsDir) {
		t.Errorf("workspace mount source = %q, want %q", wsMount.Source, filepath.Clean(wsDir))
	}
	hasRbind := false
	for _, opt := range wsMount.Options {
		if opt == "rbind" {
			hasRbind = true
		}
	}
	if !hasRbind {
		t.Errorf("workspace mount options missing 'rbind': %v", wsMount.Options)
	}

	// ReadOnlyPaths and WritablePaths mounts
	roMount, ok := mountMap["/etc/custom_config"]
	if !ok {
		t.Errorf("missing read-only mount /etc/custom_config")
	} else {
		hasRo := false
		for _, opt := range roMount.Options {
			if opt == "ro" {
				hasRo = true
			}
		}
		if !hasRo {
			t.Errorf("/etc/custom_config mount missing 'ro' option: %v", roMount.Options)
		}
	}

	wpMount, ok := mountMap["/var/log/app"]
	if !ok {
		t.Errorf("missing writable tmpfs mount /var/log/app")
	} else if wpMount.Type != "tmpfs" {
		t.Errorf("/var/log/app mount type = %q, want 'tmpfs'", wpMount.Type)
	}

	// 6. Linux Namespaces & Resources
	if ociSpec.Linux == nil {
		t.Fatal("Linux configuration is nil")
	}
	nsMap := make(map[string]bool)
	for _, ns := range ociSpec.Linux.Namespaces {
		nsMap[ns.Type] = true
	}
	for _, reqNS := range []string{"pid", "network", "ipc", "uts", "mount", "user"} {
		if !nsMap[reqNS] {
			t.Errorf("missing linux namespace: %q", reqNS)
		}
	}

	if ociSpec.Linux.Resources == nil || ociSpec.Linux.Resources.Memory == nil {
		t.Fatal("Linux.Resources.Memory is nil")
	}
	if ociSpec.Linux.Resources.Memory.Limit == nil || *ociSpec.Linux.Resources.Memory.Limit != 256*1024*1024 {
		t.Errorf("memory limit = %v, want %d", ociSpec.Linux.Resources.Memory.Limit, 256*1024*1024)
	}

	// 7. Rootfs directories existence
	rootfsDir := filepath.Join(bundleDir, "rootfs")
	for _, sub := range []string{"proc", "dev/pts", "dev/shm", "dev/mqueue", "sys", "tmp", "workspace"} {
		path := filepath.Join(rootfsDir, sub)
		if fi, err := os.Stat(path); err != nil || !fi.IsDir() {
			t.Errorf("expected rootfs mount point directory %q to exist", sub)
		}
	}
}

func TestRunscProvider(t *testing.T) {
	// 1. Verify auto-registration in default sandbox registry
	p, ok := sandbox.GetProvider("runsc")
	if !ok {
		t.Fatal("runsc provider not found in DefaultRegistry()")
	}
	if p.Name() != "runsc" {
		t.Errorf("provider Name = %q, want 'runsc'", p.Name())
	}
	if p.Tier() != sandbox.TierL2Virtual {
		t.Errorf("provider Tier = %q, want %q", p.Tier(), sandbox.TierL2Virtual)
	}

	spec := sandbox.Spec{
		Tier:         sandbox.TierL2Virtual,
		WorkspaceDir: t.TempDir(),
		EgressPolicy: sandbox.EgressBlocked,
	}

	// 2. Test absent runsc behavior: Available() returns false, Create() returns ErrSandboxUnavailable
	absentProvider := NewRunscProviderWithLookPath(func(name string) (string, error) {
		return "", errors.New("executable file not found in %PATH%")
	})
	if absentProvider.Available() {
		t.Errorf("Available() = true for absent runsc, want false")
	}

	_, err := absentProvider.Create(context.Background(), spec)
	if err == nil {
		t.Fatal("Create() succeeded when runsc is absent, want error")
	}
	if !sandbox.IsSandboxError(err, sandbox.ErrSandboxUnavailable) {
		t.Fatalf("Create() error = %v, want token %q", err, sandbox.ErrSandboxUnavailable)
	}

	// 3. Test graceful step-down ladder behavior:
	// When L2 virtual sandbox is unavailable, resolve and fall back to L1 native OS.
	l1Provider, err := sandbox.DefaultRegistry().ResolveTier(sandbox.TierL1NativeOS)
	if err != nil {
		t.Fatalf("failed to step down to TierL1NativeOS: %v", err)
	}
	if !l1Provider.Available() {
		t.Fatal("L1 native OS provider should be available")
	}

	// 4. Test present runsc behavior: Available() returns true, Create() returns active Instance
	presentProvider := NewRunscProviderWithLookPath(func(name string) (string, error) {
		return "/usr/bin/runsc", nil
	})
	if !presentProvider.Available() {
		t.Errorf("Available() = false for present runsc, want true")
	}

	inst, err := presentProvider.Create(context.Background(), spec)
	if err != nil {
		t.Fatalf("Create() error with present runsc: %v", err)
	}
	defer inst.Close()

	if inst.Spec().Tier != sandbox.TierL2Virtual {
		t.Errorf("instance Tier = %q, want %q", inst.Spec().Tier, sandbox.TierL2Virtual)
	}
}

func TestMCPToolExecutionIntegration(t *testing.T) {
	tmpWS := t.TempDir()
	agent.SetSandboxWorkspace(tmpWS)
	defer agent.SetSandboxWorkspace("")

	cat, err := agent.ArmMCPTools()
	if err != nil {
		t.Fatalf("ArmMCPTools failed: %v", err)
	}
	defer agent.DisarmMCPTools()

	// Verify all 4 sandbox tools exist in catalog
	catalogNames := make(map[string]bool)
	for _, tool := range cat {
		catalogNames[tool.Function.Name] = true
	}
	for _, expected := range []string{"sandbox_exec", "sandbox_read", "sandbox_write", "sandbox_reset"} {
		if !catalogNames[expected] {
			t.Errorf("expected tool %q in ArmMCPTools catalog", expected)
		}
	}

	ctx := context.Background()
	eng := abi.Engine("inprocess_mcp")
	if eng == nil {
		t.Fatal("inprocess_mcp engine is nil")
	}

	putBytes := func(b []byte) abi.Ref {
		return abi.Ref{
			Kind:   abi.RefInline,
			Inline: b,
			Len:    int64(len(b)),
		}
	}

	// 1. sandbox_write
	writeArgs, _ := json.Marshal(map[string]any{
		"path":    "src/hello.go",
		"content": "package main\n\nfunc main() {}\n",
	})
	cWrite := &abi.ToolCall{
		Tool: "sandbox_write",
		Args: putBytes(writeArgs),
	}
	resWrite, err := eng.Complete(ctx, cWrite)
	if err != nil {
		t.Fatalf("sandbox_write Complete error: %v", err)
	}
	if resWrite == nil || resWrite.Status != abi.StatusOK {
		t.Fatalf("sandbox_write returned non-OK status: %+v", resWrite)
	}

	writtenPath := filepath.Join(tmpWS, "src", "hello.go")
	contentBytes, err := os.ReadFile(writtenPath)
	if err != nil {
		t.Fatalf("failed to read written file from disk: %v", err)
	}
	if string(contentBytes) != "package main\n\nfunc main() {}\n" {
		t.Fatalf("file content mismatch: %q", string(contentBytes))
	}

	// 2. sandbox_read
	readArgs, _ := json.Marshal(map[string]any{
		"path": "src/hello.go",
	})
	cRead := &abi.ToolCall{
		Tool: "sandbox_read",
		Args: putBytes(readArgs),
	}
	resRead, err := eng.Complete(ctx, cRead)
	if err != nil {
		t.Fatalf("sandbox_read Complete error: %v", err)
	}
	var readBody map[string]any
	if err := json.Unmarshal(refutil.Bytes(ctx, resRead.Payload), &readBody); err != nil {
		t.Fatalf("sandbox_read unmarshal error: %v", err)
	}
	if readBody["content"] != "package main\n\nfunc main() {}\n" {
		t.Fatalf("sandbox_read content mismatch: %v", readBody["content"])
	}

	// 3. sandbox_read with offset and limit
	sliceArgs, _ := json.Marshal(map[string]any{
		"path":   "src/hello.go",
		"offset": 3,
		"limit":  1,
	})
	cSlice := &abi.ToolCall{
		Tool: "sandbox_read",
		Args: putBytes(sliceArgs),
	}
	resSlice, err := eng.Complete(ctx, cSlice)
	if err != nil {
		t.Fatalf("sandbox_read slice Complete error: %v", err)
	}
	var sliceBody map[string]any
	_ = json.Unmarshal(refutil.Bytes(ctx, resSlice.Payload), &sliceBody)
	if sliceBody["content"] != "func main() {}" {
		t.Fatalf("sandbox_read slice content = %q, want 'func main() {}'", sliceBody["content"])
	}

	// 4. Confinement path escape check
	escapeArgs, _ := json.Marshal(map[string]any{
		"path": "../../outside.txt",
	})
	cEscape := &abi.ToolCall{
		Tool: "sandbox_read",
		Args: putBytes(escapeArgs),
	}
	resEscape, _ := eng.Complete(ctx, cEscape)
	if resEscape == nil || resEscape.Status != abi.StatusError {
		t.Fatalf("sandbox_read traversal escape should return StatusError: %+v", resEscape)
	}

	// 5. sandbox_reset
	cReset := &abi.ToolCall{
		Tool: "sandbox_reset",
		Args: putBytes([]byte("{}")),
	}
	resReset, err := eng.Complete(ctx, cReset)
	if err != nil {
		t.Fatalf("sandbox_reset Complete error: %v", err)
	}
	if resReset == nil || resReset.Status != abi.StatusOK {
		t.Fatalf("sandbox_reset returned non-OK status: %+v", resReset)
	}

	// 6. sandbox_exec
	execArgs, _ := json.Marshal(map[string]any{
		"command": "echo mcp_sandbox_exec_test",
	})
	cExec := &abi.ToolCall{
		Tool: "sandbox_exec",
		Args: putBytes(execArgs),
	}
	resExec, err := eng.Complete(ctx, cExec)
	if err != nil {
		t.Fatalf("sandbox_exec Complete error: %v", err)
	}
	if resExec == nil || resExec.Status != abi.StatusOK {
		t.Fatalf("sandbox_exec returned non-OK status: %+v", resExec)
	}
	var execBody map[string]any
	if err := json.Unmarshal(refutil.Bytes(ctx, resExec.Payload), &execBody); err != nil {
		t.Fatalf("sandbox_exec unmarshal error: %v", err)
	}
	stdoutStr, _ := execBody["stdout"].(string)
	if !strings.Contains(stdoutStr, "mcp_sandbox_exec_test") {
		t.Fatalf("sandbox_exec stdout = %q, want it to contain 'mcp_sandbox_exec_test'", stdoutStr)
	}
}
