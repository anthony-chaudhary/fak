package doshook

import (
	"bytes"
	"io"
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"testing"
)

// BenchmarkRepoRootEnv measures workspace root resolution from the environment map.
func BenchmarkRepoRootEnv(b *testing.B) {
	env := map[string]string{
		"CLAUDE_PROJECT_DIR": "/workspace/fak",
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		root := RepoRoot(env)
		if root != "/workspace/fak" {
			b.Fatalf("unexpected root: %s", root)
		}
	}
}

// BenchmarkRepoRootDefault measures fallback directory traversal when environment is nil.
func BenchmarkRepoRootDefault(b *testing.B) {
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		root := defaultRepoRoot()
		if root == "" {
			b.Fatal("unexpected empty root")
		}
	}
}

// BenchmarkNativeBinaryHit measures lookup latency when the provisioned dos-hook binary exists.
func BenchmarkNativeBinaryHit(b *testing.B) {
	root := b.TempDir()
	binDir := filepath.Join(root, "tools", ".bin")
	if err := os.MkdirAll(binDir, 0o755); err != nil {
		b.Fatalf("mkdir: %v", err)
	}
	name := "dos-hook"
	if runtime.GOOS == "windows" {
		name = "dos-hook.exe"
	}
	binPath := filepath.Join(binDir, name)
	if err := os.WriteFile(binPath, []byte(""), 0o755); err != nil {
		b.Fatalf("write: %v", err)
	}

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		p := NativeBinary(root)
		if p == "" {
			b.Fatal("expected native binary path")
		}
	}
}

// BenchmarkNativeBinaryMiss measures lookup overhead when no candidate binary exists.
func BenchmarkNativeBinaryMiss(b *testing.B) {
	root := filepath.Join(b.TempDir(), "nonexistent")
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		p := NativeBinary(root)
		if p != "" {
			b.Fatalf("expected empty binary path, got: %s", p)
		}
	}
}

// BenchmarkRunHookNativeSuccess measures hook dispatch on the native fast path.
func BenchmarkRunHookNativeSuccess(b *testing.B) {
	native := filepath.Join("tools", ".bin", "dos-hook")
	payload := []byte(`{"tool_name":"Read","tool_input":{"file_path":"main.go"}}`)
	expectedOut := []byte(`{"hookSpecificOutput":{"permissionDecision":"allow"}}`)
	runner := func(argv []string, stdin []byte) (int, []byte, error) {
		return 0, expectedOut, nil
	}

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		out, err := RunHook("pretool", ".", payload, native, runner)
		if err != nil || len(out) == 0 {
			b.Fatalf("RunHook failed: %v", err)
		}
	}
}

// BenchmarkRunHookFallbackToPython measures hook dispatch when native execution fails and falls back to Python.
func BenchmarkRunHookFallbackToPython(b *testing.B) {
	native := filepath.Join("tools", ".bin", "dos-hook")
	payload := []byte(`{"tool_name":"Bash","tool_input":{"command":"git status"}}`)
	expectedOut := []byte(`{"hookSpecificOutput":{"permissionDecision":"allow"}}`)
	runner := func(argv []string, stdin []byte) (int, []byte, error) {
		if len(argv) > 0 && strings.Contains(argv[0], "dos-hook") {
			return 3, nil, nil
		}
		return 0, expectedOut, nil
	}

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		out, err := RunHook("pretool", ".", payload, native, runner)
		if err != nil || len(out) == 0 {
			b.Fatalf("RunHook fallback failed: %v", err)
		}
	}
}

// BenchmarkRunHookPythonDirect measures hook dispatch when native binary is absent.
func BenchmarkRunHookPythonDirect(b *testing.B) {
	payload := []byte(`{"tool_name":"Glob","tool_input":{"pattern":"*.go"}}`)
	expectedOut := []byte(`{"hookSpecificOutput":{"permissionDecision":"allow"}}`)
	runner := func(argv []string, stdin []byte) (int, []byte, error) {
		return 0, expectedOut, nil
	}

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		out, err := RunHook("pretool", ".", payload, "", runner)
		if err != nil || len(out) == 0 {
			b.Fatalf("RunHook direct python failed: %v", err)
		}
	}
}

// BenchmarkRunHookPayloadScaling measures hook execution throughput across varying payload sizes.
func BenchmarkRunHookPayloadScaling(b *testing.B) {
	sizes := []struct {
		name string
		size int
	}{
		{"64B", 64},
		{"1KB", 1024},
		{"64KB", 64 * 1024},
	}

	for _, tc := range sizes {
		payload := bytes.Repeat([]byte("a"), tc.size)
		runner := func(argv []string, stdin []byte) (int, []byte, error) {
			return 0, stdin, nil
		}
		b.Run(tc.name, func(b *testing.B) {
			b.SetBytes(int64(tc.size))
			b.ReportAllocs()
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				out, err := RunHook("pretool", ".", payload, "dos-hook", runner)
				if err != nil || len(out) != tc.size {
					b.Fatalf("RunHook payload scaling failed: %v", err)
				}
			}
		})
	}
}

// BenchmarkRunNative measures end-to-end Run lifecycle on the native execution path.
func BenchmarkRunNative(b *testing.B) {
	root := b.TempDir()
	binDir := filepath.Join(root, "tools", ".bin")
	if err := os.MkdirAll(binDir, 0o755); err != nil {
		b.Fatalf("mkdir: %v", err)
	}
	name := "dos-hook"
	if runtime.GOOS == "windows" {
		name = "dos-hook.exe"
	}
	if err := os.WriteFile(filepath.Join(binDir, name), []byte(""), 0o755); err != nil {
		b.Fatalf("write: %v", err)
	}

	argv := []string{"pretool", "--workspace", root}
	payload := []byte(`{"tool_name":"Edit","tool_input":{"file_path":"doc.go"}}`)
	expectedOut := []byte(`{"decision":"allow"}`)
	runner := func(argv []string, stdin []byte) (int, []byte, error) {
		return 0, expectedOut, nil
	}
	env := map[string]string{"CLAUDE_PROJECT_DIR": root}

	var stdout bytes.Buffer
	r := bytes.NewReader(payload)

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		r.Reset(payload)
		stdout.Reset()
		rc := Run(argv, r, &stdout, io.Discard, env, runner)
		if rc != 0 || stdout.Len() == 0 {
			b.Fatalf("unexpected Run outcome: rc=%d, stdoutLen=%d", rc, stdout.Len())
		}
	}
}

// BenchmarkRunPythonFallback measures end-to-end Run lifecycle when falling back to Python.
func BenchmarkRunPythonFallback(b *testing.B) {
	argv := []string{"posttool", "--workspace", "."}
	payload := []byte(`{"tool_name":"Bash","tool_input":{"command":"make build"}}`)
	expectedOut := []byte(`{"decision":"allow"}`)
	runner := func(argv []string, stdin []byte) (int, []byte, error) {
		if len(argv) > 0 && strings.Contains(argv[0], "dos-hook") {
			return 1, nil, nil
		}
		return 0, expectedOut, nil
	}
	env := map[string]string{"CLAUDE_PROJECT_DIR": b.TempDir()}

	var stdout bytes.Buffer
	r := bytes.NewReader(payload)

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		r.Reset(payload)
		stdout.Reset()
		rc := Run(argv, r, &stdout, io.Discard, env, runner)
		if rc != 0 || stdout.Len() == 0 {
			b.Fatalf("unexpected Run outcome: rc=%d, stdoutLen=%d", rc, stdout.Len())
		}
	}
}

// BenchmarkRunArgParsing measures CLI flag extraction and workspace argument parsing.
func BenchmarkRunArgParsing(b *testing.B) {
	cases := []struct {
		name string
		argv []string
	}{
		{"BareVerb", []string{"pretool"}},
		{"WorkspaceSpace", []string{"pretool", "--workspace", "custom/ws"}},
		{"WorkspaceEquals", []string{"pretool", "--workspace=custom/ws"}},
		{"TrailingWorkspace", []string{"--workspace", "custom/ws", "posttool"}},
	}

	payload := []byte(`{}`)
	runner := func(argv []string, stdin []byte) (int, []byte, error) {
		return 0, nil, nil
	}
	env := map[string]string{"CLAUDE_PROJECT_DIR": "/ws"}
	r := bytes.NewReader(payload)

	for _, tc := range cases {
		b.Run(tc.name, func(b *testing.B) {
			b.ReportAllocs()
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				r.Reset(payload)
				rc := Run(tc.argv, r, io.Discard, io.Discard, env, runner)
				if rc != 0 {
					b.Fatalf("unexpected rc: %d", rc)
				}
			}
		})
	}
}

// BenchmarkMain measures top-level Main entry point throughput.
func BenchmarkMain(b *testing.B) {
	argv := []string{"pretool", "--workspace=."}
	payload := []byte(`{"tool_name":"Read","tool_input":{"file_path":"README.md"}}`)
	expectedOut := []byte(`{"decision":"allow"}`)
	runner := func(argv []string, stdin []byte) (int, []byte, error) {
		return 0, expectedOut, nil
	}
	env := map[string]string{"CLAUDE_PROJECT_DIR": "/ws"}

	var stdout bytes.Buffer
	r := bytes.NewReader(payload)

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		r.Reset(payload)
		stdout.Reset()
		rc := Main(argv, r, &stdout, io.Discard, env, runner)
		if rc != 0 || stdout.Len() == 0 {
			b.Fatalf("unexpected Main outcome: rc=%d, stdoutLen=%d", rc, stdout.Len())
		}
	}
}
