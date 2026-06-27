package benchcli

import (
	"os"
	"path/filepath"
	"testing"
)

func TestReadHFConfigDerivesHeadDim(t *testing.T) {
	dir := t.TempDir()
	body := []byte(`{
		"model_type": "qwen2",
		"hidden_size": 1536,
		"num_attention_heads": 12,
		"num_key_value_heads": 2,
		"num_hidden_layers": 28,
		"intermediate_size": 8960,
		"vocab_size": 151936
	}`)
	if err := os.WriteFile(filepath.Join(dir, "config.json"), body, 0o644); err != nil {
		t.Fatalf("write config: %v", err)
	}
	cfg, err := ReadHFConfig(dir)
	if err != nil {
		t.Fatalf("ReadHFConfig: %v", err)
	}
	if cfg.ModelType != "qwen2" {
		t.Fatalf("ModelType = %q, want qwen2", cfg.ModelType)
	}
	// head_dim absent in the JSON -> derived as hidden_size/num_attention_heads.
	if cfg.HeadDim != 128 {
		t.Fatalf("HeadDim = %d, want 1536/12 = 128", cfg.HeadDim)
	}
}

func TestReadHFConfigKeepsExplicitHeadDim(t *testing.T) {
	dir := t.TempDir()
	body := []byte(`{"hidden_size":1536,"num_attention_heads":12,"head_dim":64}`)
	if err := os.WriteFile(filepath.Join(dir, "config.json"), body, 0o644); err != nil {
		t.Fatalf("write config: %v", err)
	}
	cfg, err := ReadHFConfig(dir)
	if err != nil {
		t.Fatalf("ReadHFConfig: %v", err)
	}
	if cfg.HeadDim != 64 {
		t.Fatalf("HeadDim = %d, want explicit 64 (not derived 128)", cfg.HeadDim)
	}
}

func TestReadHFConfigMissingFileErrors(t *testing.T) {
	if _, err := ReadHFConfig(t.TempDir()); err == nil {
		t.Fatal("ReadHFConfig on a dir with no config.json: want error, got nil")
	}
}

func TestWriteFileCreatesParentDirs(t *testing.T) {
	root := t.TempDir()
	path := filepath.Join(root, "nested", "deep", "report.json")
	want := []byte(`{"ok":true}`)
	if err := WriteFile(path, want); err != nil {
		t.Fatalf("WriteFile: %v", err)
	}
	got, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("read back: %v", err)
	}
	if string(got) != string(want) {
		t.Fatalf("content = %q, want %q", got, want)
	}
}

func TestWriteFileBareFilename(t *testing.T) {
	// A bare filename has Dir "." — WriteFile must skip mkdir and still write it.
	dir := t.TempDir()
	wd, err := os.Getwd()
	if err != nil {
		t.Fatalf("getwd: %v", err)
	}
	t.Cleanup(func() { _ = os.Chdir(wd) })
	if err := os.Chdir(dir); err != nil {
		t.Fatalf("chdir: %v", err)
	}
	if err := WriteFile("bare.json", []byte("x")); err != nil {
		t.Fatalf("WriteFile bare: %v", err)
	}
	if _, err := os.Stat(filepath.Join(dir, "bare.json")); err != nil {
		t.Fatalf("bare file not written: %v", err)
	}
}
