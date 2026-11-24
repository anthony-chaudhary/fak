package main

import (
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"testing"
)

// A proxying serve must lift the 60 s planner and 90 s write defaults, which otherwise cut a
// buffered agent turn off before a local engine finishes it, without overriding an operator.
func TestEnsureServeProxyTimeoutFloors(t *testing.T) {
	upstream := "http://127.0.0.1:1"
	empty := ""
	floor := strconv.Itoa(serveProxyTimeoutFloorS)
	vars := []string{"FAK_PLANNER_TIMEOUT_S", "FAK_HTTP_WRITE_TIMEOUT_S"}

	tests := []struct {
		name     string
		sf       *serveFlags
		preset   string
		wantEach string
	}{
		{name: "base-url proxy raises unset defaults", sf: &serveFlags{baseURL: &upstream}, wantEach: floor},
		{name: "operator value wins", sf: &serveFlags{baseURL: &upstream}, preset: "900", wantEach: "900"},
		{name: "explicit zero opt-out wins", sf: &serveFlags{baseURL: &upstream}, preset: "0", wantEach: "0"},
		{name: "no upstream leaves defaults", sf: &serveFlags{baseURL: &empty}, wantEach: ""},
		{name: "nil flags leave defaults", sf: nil, wantEach: ""},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			for _, v := range vars {
				t.Setenv(v, tt.preset)
			}
			ensureServeProxyTimeoutFloors(tt.sf)
			for _, v := range vars {
				if got := os.Getenv(v); got != tt.wantEach {
					t.Fatalf("%s = %q, want %q", v, got, tt.wantEach)
				}
			}
		})
	}

	t.Run("replica upstream counts as proxy", func(t *testing.T) {
		for _, v := range vars {
			t.Setenv(v, "")
		}
		sf := &serveFlags{baseURL: &empty}
		if err := sf.replicaBaseURLs.Set("http://127.0.0.1:2"); err != nil {
			t.Fatalf("set replica: %v", err)
		}
		ensureServeProxyTimeoutFloors(sf)
		for _, v := range vars {
			if got := os.Getenv(v); got != floor {
				t.Fatalf("%s = %q, want %q", v, got, floor)
			}
		}
	})
}

// The floors only help if they are in the environment before gateway.New builds the proxy
// planners and before the listener reads its write deadline.
func TestServeBuildGatewayAppliesProxyTimeoutFloorsBeforeGatewayNew(t *testing.T) {
	root := repoRootFromTest(t)
	body, err := os.ReadFile(filepath.Join(root, "cmd", "fak", "serve.go"))
	if err != nil {
		t.Fatalf("read serve.go: %v", err)
	}
	src := string(body)
	fn := strings.Index(src, "func (rt *serveRuntime) buildGateway(")
	floors := strings.Index(src, "ensureServeProxyTimeoutFloors(sf)")
	build := strings.Index(src, "gateway.New(gateway.Config{")
	if fn < 0 || floors < 0 || build < 0 {
		t.Fatalf("serve.go markers missing: buildGateway=%d floors=%d gateway.New=%d", fn, floors, build)
	}
	if !(fn < floors && floors < build) {
		t.Fatalf("ensureServeProxyTimeoutFloors must run inside buildGateway before gateway.New (buildGateway=%d floors=%d gateway.New=%d)", fn, floors, build)
	}
}
