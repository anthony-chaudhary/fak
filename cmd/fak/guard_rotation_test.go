package main

import (
	"github.com/anthony-chaudhary/fak/internal/accounts"
	"strings"
	"testing"
)

func TestGuardRotateModeDefaults(t *testing.T) {
	cases := []struct {
		raw              string
		set, interactive bool
		want             string
	}{{"", false, true, "off"}, {"", false, false, "auto"}, {"auto", true, true, "auto"}, {"off", true, false, "off"}, {"seat-b", true, true, "seat-b"}}
	for _, c := range cases {
		got, err := normalizeGuardRotateMode(c.raw, c.set, c.interactive)
		if err != nil || got != c.want {
			t.Fatalf("%+v got %q %v", c, got, err)
		}
	}
}
func TestGuardNextRotationDistinctAndFailLoud(t *testing.T) {
	ready := func(name, key string) accounts.Home {
		return accounts.Home{Name: name, Dir: "/" + name, Identity: accounts.Identity{Exists: true, HasCreds: true, AccountUUID: key}}
	}
	reg := accounts.Registry{Homes: []accounts.Home{ready("a", "acct-a"), ready("b", "acct-b")}}
	r, ok := guardNextRotation(reg, "a", "auto", "stale", "CLAUDE_CONFIG_DIR")
	if !ok || r.Seat != "b" || r.Dir != "/b" {
		t.Fatalf("rotation=%+v ok=%v", r, ok)
	}
	if !strings.Contains(guardRotationBanner(r), "account b (stale)") {
		t.Fatalf("banner=%q", guardRotationBanner(r))
	}
	one := accounts.Registry{Homes: []accounts.Home{ready("a", "acct-a")}}
	if _, ok := guardNextRotation(one, "a", "auto", "stale", "CLAUDE_CONFIG_DIR"); ok {
		t.Fatal("one bucket re-handed walled account")
	}
	if _, ok := guardNextRotation(reg, "a", "off", "stale", "CLAUDE_CONFIG_DIR"); ok {
		t.Fatal("rotate=off changed behavior")
	}
}
func TestGuardNextRotationExplicitCodexSeat(t *testing.T) {
	ready := func(name, key string) accounts.Home {
		return accounts.Home{Name: name, Dir: "/codex-" + name, Identity: accounts.Identity{Exists: true, HasCreds: true, AccountUUID: key}}
	}
	reg := accounts.Registry{Homes: []accounts.Home{ready("chatgpt-a", "acct-a"), ready("chatgpt-b", "acct-b")}}
	r, ok := guardNextRotation(reg, "chatgpt-a", "chatgpt-b", "walled", "CODEX_HOME")
	if !ok || r.EnvKey != "CODEX_HOME" || r.Seat != "chatgpt-b" {
		t.Fatalf("rotation=%+v ok=%v", r, ok)
	}
}

func TestGuardApplyRotationRepointsChildEnvironment(t *testing.T) {
	cmd := []string{"codex", "exec"}
	gotCmd, env := guardApplyRotation(cmd, [][2]string{{"CODEX_HOME", "/old"}, {"X", "1"}}, guardRotation{Seat: "b", Dir: "/new", EnvKey: "CODEX_HOME"})
	if strings.Join(gotCmd, " ") != "codex exec" || len(env) != 2 || env[0] != [2]string{"CODEX_HOME", "/new"} {
		t.Fatalf("command=%v env=%v", gotCmd, env)
	}
}
