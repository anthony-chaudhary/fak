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

func TestServePiDoesNotWritePersistentConfig(t *testing.T) {
	const helperEnv = "FAK_TEST_SERVE_PI_NO_WRITE_HELPER"
	if os.Getenv(helperEnv) == "1" {
		cmdServe([]string{
			"--mock",
			"--pi",
			"--pi-config-path", os.Getenv("FAK_TEST_SERVE_PI_MODELS"),
			"--addr", os.Getenv("FAK_TEST_SERVE_PI_ADDR"),
			"--session-state", "off",
			"--keep-awake", "off",
		})
		return
	}

	piHome := t.TempDir()
	modelsPath := filepath.Join(piHome, "models.json")
	settingsPath := filepath.Join(piHome, "settings.json")
	modelsBefore := []byte("{\n  \"providers\": [],\n  \"sentinel\": \"serve-models-original\"\n}\n")
	settingsBefore := []byte("{\n  \"defaultProvider\": \"operator\",\n  \"defaultModel\": \"operator-model\",\n  \"sentinel\": \"serve-settings-original\"\n}\n")
	if err := os.WriteFile(modelsPath, modelsBefore, 0o644); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(settingsPath, settingsBefore, 0o644); err != nil {
		t.Fatal(err)
	}

	addr := reserveServeStartupAddr(t)
	stateDir := t.TempDir()
	ctx, cancel := context.WithTimeout(context.Background(), 15*time.Second)
	defer cancel()
	cmd := exec.CommandContext(ctx, os.Args[0], "-test.run=^TestServePiDoesNotWritePersistentConfig$")
	cmd.Env = append(os.Environ(),
		helperEnv+"=1",
		"FAK_TEST_SERVE_PI_ADDR="+addr,
		"FAK_TEST_SERVE_PI_MODELS="+modelsPath,
		"PI_CODING_AGENT_DIR="+piHome,
		"FAK_SESSION_REGISTRY="+filepath.Join(stateDir, "sessions.json"),
		"HOME="+stateDir,
		"USERPROFILE="+stateDir,
		"XDG_CONFIG_HOME="+filepath.Join(stateDir, ".config"),
	)
	if err := cmd.Start(); err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		if cmd.ProcessState == nil || !cmd.ProcessState.Exited() {
			_ = cmd.Process.Kill()
			_ = cmd.Wait()
		}
	})

	_ = readLiveFeatureCatalog(t, "http://"+addr+"/v1/fak/features", 12*time.Second)
	cancel()
	_ = cmd.Wait()

	for path, before := range map[string][]byte{modelsPath: modelsBefore, settingsPath: settingsBefore} {
		after, err := os.ReadFile(path)
		if err != nil {
			t.Fatalf("read Pi config after serve --pi: %v", err)
		}
		if !bytes.Equal(after, before) {
			t.Errorf("ordinary fak serve --pi mutated %s\nbefore: %s\nafter:  %s", filepath.Base(path), before, after)
		}
	}
}

func TestServePiConfigPrintsSnippet(t *testing.T) {
	isolatePiHome(t)
	var buf bytes.Buffer
	sf := &serveFlags{
		addr:  strPtr("127.0.0.1:8080"),
		model: strPtr("qwen38:27b-q4"),
	}

	runServePiConfig(sf, &buf, false)
	out := buf.String()

	if !strings.Contains(out, "http://127.0.0.1:8080/v1") {
		t.Errorf("expected baseURL in output: %s", out)
	}
	if !strings.Contains(out, "qwen38:27b-q4") {
		t.Errorf("expected model in output: %s", out)
	}
	if !strings.Contains(out, `"openai-completions"`) {
		t.Errorf("expected openai-completions in output: %s", out)
	}
	if !strings.Contains(out, `"supportsDeveloperRole": false`) {
		t.Errorf("expected supportsDeveloperRole: false in output: %s", out)
	}
}

func TestServePiConfigWritesFile(t *testing.T) {
	isolatePiHome(t)
	ws := t.TempDir()
	configPath := filepath.Join(ws, "models.json")

	var buf bytes.Buffer
	sf := &serveFlags{
		addr:         strPtr("127.0.0.1:9000"),
		model:        strPtr("custom-model"),
		piConfigPath: strPtr(configPath),
	}

	runServePiConfig(sf, &buf, true)

	data, err := os.ReadFile(configPath)
	if err != nil {
		t.Fatalf("failed to read models.json: %v", err)
	}

	content := string(data)
	if !strings.Contains(content, "http://127.0.0.1:9000/v1") {
		t.Errorf("expected baseURL in file: %s", content)
	}
	if !strings.Contains(content, "custom-model") {
		t.Errorf("expected model in file: %s", content)
	}
	if !strings.Contains(content, `"openai-completions"`) {
		t.Errorf("expected api in file: %s", content)
	}
}
