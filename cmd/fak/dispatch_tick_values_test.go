package main

import (
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/accounts"
)

// TestEnvMapDropsWindowsDrivePseudoVars pins the seam that made every accounts-launch spawn
// fail-close on Windows consoles carrying a per-drive current-directory pseudo-variable:
// os.Environ() surfaces "=C:=C:\..." whose NAME is empty once split on the first '='. The
// spawn broker treats an empty name as EMPTY_ENV_NAME and denies the whole launch, so the
// map-building seam must drop such entries rather than hand an empty key downstream. This
// mirrors toolprocgate.EnvFromStrings (c275b86f9b5) and envMapFromStrings' i>0 guard.
func TestEnvMapDropsWindowsDrivePseudoVars(t *testing.T) {
	got := envMap([]string{"=C:=C:\\x", "PATH=y"})
	if len(got) != 1 {
		t.Fatalf("envMap kept %d entries, want exactly 1 (the empty-name pseudo-var must be dropped): %v", len(got), got)
	}
	if got["PATH"] != "y" {
		t.Fatalf("envMap[PATH] = %q, want \"y\"", got["PATH"])
	}
	if _, present := got[""]; present {
		t.Fatal("envMap kept an empty-named entry; the spawn broker would deny the launch with EMPTY_ENV_NAME")
	}
}

// TestAccountsLaunchBrokerAdmitsSeatEnvWithDrivePseudoVar is the launch-phase witness: a seat
// environment composed from a base slice carrying the Windows per-drive pseudo-variable must
// reach the spawn broker WITHOUT an empty-named EnvVar, so the broker admits the launch
// instead of failing closed with EMPTY_ENV_NAME (#14268).
func TestAccountsLaunchBrokerAdmitsSeatEnvWithDrivePseudoVar(t *testing.T) {
	base := []string{"=C:=C:\\x", "PATH=/usr/bin:/bin"}
	home := accounts.Home{Name: "sub", Dir: "/tmp/.claude-sub"}
	env, _ := launchSeatEnv(base, home)
	a := newLaunchBrokerAttemptDeclaring("accounts_launch", "claude",
		[]string{"claude", "-p", "hello"}, envMap(env), t.TempDir(), nil)
	for _, kv := range a.Spawn.Env {
		if strings.TrimSpace(kv.Name) == "" {
			t.Fatalf("SpawnAttempt.Env carries an empty-named variable (%#v); normalizeEnv would deny with EMPTY_ENV_NAME", kv)
		}
	}
	grant := defaultLaunchSpawnBroker(a)
	if !grant.Allow {
		t.Fatalf("spawn broker denied the launch with %q; a per-drive pseudo-variable in the ambient env must not fail-close accounts launch", grant.Reason)
	}
}
