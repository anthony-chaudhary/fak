package main

import (
	"bytes"
	"context"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

func TestServeHarnessConfigWritesAreExplicit(t *testing.T) {
	const helperEnv = "FAK_TEST_SERVE_GUARD_OPTIN_HELPER"
	if mode := os.Getenv(helperEnv); mode != "" {
		args := []string{"--mock", "--addr", os.Getenv("FAK_TEST_SERVE_ADDR"), "--session-state", "off", "--keep-awake", "off"}
		switch mode {
		case "opencode":
			args = append(args, "--opencode")
		case "codex":
			args = append(args, "--codex", "--codex-config-path", os.Getenv("FAK_TEST_CODEX_CONFIG"))
		case "claude":
			args = append(args, "--claude")
		case "write-opencode":
			args = append(args, "--write-opencode-config", "--model", "explicit-opencode-model")
		case "write-codex":
			args = append(args, "--write-codex-config", "--codex-config-path", os.Getenv("FAK_TEST_CODEX_CONFIG"), "--model", "explicit-codex-model")
		default:
			os.Exit(97)
		}
		cmdServe(args)
		return
	}

	workspace := t.TempDir()
	stateDir := t.TempDir()
	opencodePath := filepath.Join(workspace, "opencode.json")
	codexPath := filepath.Join(workspace, ".codex", "config.toml")
	claudePath := filepath.Join(workspace, ".claude", "settings.json")
	for _, path := range []string{codexPath, claudePath} {
		if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
			t.Fatal(err)
		}
	}
	original := map[string][]byte{
		opencodePath: []byte("{\n  \"sentinel\": \"opencode-original\"\n}\n"),
		codexPath:    []byte("sentinel = \"codex-original\"\n"),
		claudePath:   []byte("{\n  \"sentinel\": \"claude-original\"\n}\n"),
	}
	writeOriginals := func(t *testing.T) {
		t.Helper()
		for path, body := range original {
			if err := os.WriteFile(path, body, 0o644); err != nil {
				t.Fatal(err)
			}
		}
	}
	writeOriginals(t)

	run := func(t *testing.T, mode string, waitForServer bool) {
		t.Helper()
		addr := reserveServeStartupAddr(t)
		ctx, cancel := context.WithTimeout(context.Background(), 15*time.Second)
		defer cancel()
		cmd := exec.CommandContext(ctx, os.Args[0], "-test.run=^TestServeHarnessConfigWritesAreExplicit$")
		cmd.Dir = workspace
		cmd.Env = append(os.Environ(),
			helperEnv+"="+mode,
			"FAK_TEST_SERVE_ADDR="+addr,
			"FAK_TEST_CODEX_CONFIG="+codexPath,
			"FAK_SESSION_REGISTRY="+filepath.Join(stateDir, "sessions.json"),
			"CODEX_HOME="+filepath.Dir(codexPath),
			"HOME="+workspace,
			"USERPROFILE="+workspace,
			"XDG_CONFIG_HOME="+filepath.Join(workspace, ".config"),
		)
		if err := cmd.Start(); err != nil {
			t.Fatal(err)
		}
		if waitForServer {
			_ = readLiveFeatureCatalog(t, "http://"+addr+"/v1/fak/features", 12*time.Second)
			cancel()
		}
		err := cmd.Wait()
		if !waitForServer && err != nil {
			t.Fatalf("serve helper %s failed: %v", mode, err)
		}
	}

	for _, harness := range []string{"opencode", "codex", "claude"} {
		t.Run(harness+" ordinary launch does not write", func(t *testing.T) {
			writeOriginals(t)
			run(t, harness, true)
			for path, before := range original {
				after, err := os.ReadFile(path)
				if err != nil {
					t.Fatal(err)
				}
				if !bytes.Equal(after, before) {
					t.Fatalf("serve --%s mutated persistent config %s\nbefore: %s\nafter: %s", harness, path, before, after)
				}
			}
		})
	}

	for _, tc := range []struct {
		name string
		mode string
		path string
		want string
	}{
		{name: "OpenCode explicit writer", mode: "write-opencode", path: opencodePath, want: "explicit-opencode-model"},
		{name: "Codex explicit writer", mode: "write-codex", path: codexPath, want: "[model_providers.fak]"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			writeOriginals(t)
			run(t, tc.mode, false)
			after, err := os.ReadFile(tc.path)
			if err != nil {
				t.Fatal(err)
			}
			if bytes.Equal(after, original[tc.path]) || !strings.Contains(string(after), tc.want) {
				t.Fatalf("%s did not explicitly update %s: %s", tc.mode, tc.path, after)
			}
		})
	}
}
