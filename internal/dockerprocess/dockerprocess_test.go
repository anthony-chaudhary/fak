package dockerprocess

import (
	"context"
	"path/filepath"
	"reflect"
	"runtime"
	"strings"
	"testing"
)

func TestComposeCommandPinsDockerAndCopiesProcessSettings(t *testing.T) {
	env := []string{"A=1"}
	args := []string{"-f", "compose.yml", "up", "-d"}
	cmd := composeCommand(context.Background(), "/work", env, args...)
	base := strings.TrimSuffix(strings.ToLower(filepath.Base(cmd.Path)), ".exe")
	if base != "docker" {
		t.Fatalf("Compose executable = %q, want docker", cmd.Path)
	}
	if got, want := cmd.Args, []string{"docker", "compose", "-f", "compose.yml", "up", "-d"}; !reflect.DeepEqual(got, want) {
		t.Fatalf("Compose argv = %#v, want %#v", got, want)
	}
	if cmd.Dir != "/work" || !reflect.DeepEqual(cmd.Env, []string{"A=1"}) {
		t.Fatalf("Compose process settings = dir %q env %#v", cmd.Dir, cmd.Env)
	}
	env[0], args[0] = "MUTATED=1", "MUTATED"
	if cmd.Env[0] != "A=1" || cmd.Args[2] != "-f" {
		t.Fatalf("Compose command retained caller-owned slices: argv=%#v env=%#v", cmd.Args, cmd.Env)
	}
}

func TestAvailableRequiresDockerOnPath(t *testing.T) {
	t.Setenv("PATH", t.TempDir())
	if runtime.GOOS == "windows" {
		t.Setenv("PATHEXT", ".EXE")
	}
	if Available() {
		t.Fatal("Available reported Docker in an empty PATH")
	}
}
