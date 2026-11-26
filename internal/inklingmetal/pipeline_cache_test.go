package inklingmetal

import (
	"os"
	"path/filepath"
	"testing"
)

func TestPipelineCacheDirDisableSpellings(t *testing.T) {
	for _, v := range []string{"", "off", "OFF", "Off", " off ", "0", " 0\t", "false", "FALSE", "False", "none", "NONE", "  None\n", "   "} {
		t.Setenv(PipelineCacheEnv, v)
		if got := pipelineCacheDir(); got != "" {
			t.Errorf("%s=%q: pipelineCacheDir() = %q, want disabled", PipelineCacheEnv, v, got)
		}
	}
}

func TestPipelineCacheDirExplicitPath(t *testing.T) {
	for _, v := range []string{t.TempDir(), "relative/cache", "/abs/inkling", "offline"} {
		t.Setenv(PipelineCacheEnv, v)
		if got := pipelineCacheDir(); got != v {
			t.Errorf("%s=%q: pipelineCacheDir() = %q, want the path", PipelineCacheEnv, v, got)
		}
	}
}

func TestPipelineCacheDirDefaultsUnderUserCacheDir(t *testing.T) {
	home := t.TempDir()
	t.Setenv("HOME", home)
	t.Setenv("XDG_CACHE_HOME", filepath.Join(home, ".cache"))
	t.Setenv("LocalAppData", filepath.Join(home, "AppData", "Local"))
	t.Setenv(PipelineCacheEnv, "x")
	if err := os.Unsetenv(PipelineCacheEnv); err != nil {
		t.Fatal(err)
	}
	base, err := os.UserCacheDir()
	if err != nil {
		t.Fatal(err)
	}
	want := filepath.Join(base, "fak", "inkling-metal-pipelines")
	if got := pipelineCacheDir(); got != want {
		t.Fatalf("pipelineCacheDir() = %q, want %q", got, want)
	}
}
