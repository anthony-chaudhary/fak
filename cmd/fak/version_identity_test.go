package main

import (
	"bytes"
	"encoding/json"
	"runtime/debug"
	"strings"
	"testing"
)

// TestBuildIdentityStamped is the witness for epic #2218 gap G2 (R1): a binary built
// WITH VCS provenance must expose a machine-readable commit + dirty bit, so a fleet
// monitor / wave-admission skew witness can tell one running copy from another.
func TestBuildIdentityStamped(t *testing.T) {
	bi := &debug.BuildInfo{
		Main: debug.Module{Version: "(devel)"},
		Settings: []debug.BuildSetting{
			{Key: "vcs.revision", Value: "abcdef0123456789abcdef0123456789abcdef01"},
			{Key: "vcs.modified", Value: "true"},
			{Key: "vcs.time", Value: "2026-07-04T12:00:00Z"},
		},
	}
	id := buildIdentity(bi)

	if !id.Stamped {
		t.Fatalf("Stamped = false, want true for a VCS-stamped build")
	}
	if id.Commit != "abcdef0123456789abcdef0123456789abcdef01" {
		t.Errorf("Commit = %q, want the full vcs.revision", id.Commit)
	}
	if id.CommitShort != "abcdef012345" {
		t.Errorf("CommitShort = %q, want first 12 hex", id.CommitShort)
	}
	if !id.Dirty {
		t.Errorf("Dirty = false, want true (vcs.modified=true)")
	}
	if id.CommitTime != "2026-07-04T12:00:00Z" {
		t.Errorf("CommitTime = %q, want the vcs.time", id.CommitTime)
	}

	// The JSON surface must carry commit + dirty (the issue's explicit ask).
	var buf bytes.Buffer
	if err := writeVersionJSON(&buf, id); err != nil {
		t.Fatalf("writeVersionJSON: %v", err)
	}
	var got map[string]any
	if err := json.Unmarshal(buf.Bytes(), &got); err != nil {
		t.Fatalf("emitted JSON does not parse: %v\n%s", err, buf.String())
	}
	if got["commit"] != "abcdef0123456789abcdef0123456789abcdef01" {
		t.Errorf("json commit = %v, want full revision", got["commit"])
	}
	if got["dirty"] != true {
		t.Errorf("json dirty = %v, want true", got["dirty"])
	}
	if got["stamped"] != true {
		t.Errorf("json stamped = %v, want true", got["stamped"])
	}
}

// TestBuildIdentityUnstamped pins the witnessed failure mode: a binary built WITHOUT
// VCS provenance (the "no VCS stamp" case from the host on 2026-07-01) must report
// stamped=false and an empty commit rather than fabricate an identity. A machine
// consumer reads stamped=false as "commit unknown", never as agreement.
func TestBuildIdentityUnstamped(t *testing.T) {
	bi := &debug.BuildInfo{Main: debug.Module{Version: "v0.37.0"}}
	id := buildIdentity(bi)

	if id.Stamped {
		t.Errorf("Stamped = true, want false for a build with no vcs.* settings")
	}
	if id.Commit != "" || id.CommitShort != "" {
		t.Errorf("Commit=%q CommitShort=%q, want both empty when unstamped", id.Commit, id.CommitShort)
	}
	if id.Dirty {
		t.Errorf("Dirty = true, want false when unstamped")
	}
	if id.ModuleVersion != "v0.37.0" {
		t.Errorf("ModuleVersion = %q, want the go-install module version", id.ModuleVersion)
	}

	// A nil BuildInfo (ReadBuildInfo reported ok=false) is unstamped, not a panic.
	nilID := buildIdentity(nil)
	if nilID.Stamped || nilID.Commit != "" {
		t.Errorf("buildIdentity(nil) = %+v, want unstamped/empty commit", nilID)
	}
}

func TestVersionJSONRequested(t *testing.T) {
	for _, tc := range []struct {
		args []string
		want bool
	}{
		{[]string{"--json"}, true},
		{[]string{"-json"}, true},
		{[]string{"modules", "--json"}, true},
		{nil, false},
		{[]string{"--jsonx"}, false},
	} {
		if got := versionJSONRequested(tc.args); got != tc.want {
			t.Errorf("versionJSONRequested(%v) = %v, want %v", tc.args, got, tc.want)
		}
	}
}

// Sanity: the pure identity always carries a runtime toolchain string, so a consumer
// never sees a wholly empty record even on an unstamped build.
func TestBuildIdentityAlwaysHasRuntime(t *testing.T) {
	id := buildIdentity(nil)
	if !strings.HasPrefix(id.Go, "go") {
		t.Errorf("Go = %q, want a runtime.Version() string", id.Go)
	}
	if id.OS == "" || id.Arch == "" {
		t.Errorf("OS/Arch empty (%q/%q), want the runtime GOOS/GOARCH", id.OS, id.Arch)
	}
}

func TestBuildIdentityUsesInjectedCommitWhenGoVCSStampMissing(t *testing.T) {
	old := appversion.BuildCommit
	appversion.BuildCommit = "0123456789abcdef0123456789abcdef01234567"
	t.Cleanup(func() { appversion.BuildCommit = old })

	id := buildIdentity(&debug.BuildInfo{})
	if !id.Stamped || id.Commit != appversion.BuildCommit || id.CommitShort != "0123456789ab" {
		t.Fatalf("injected commit identity = %+v, want stamped %s", id, appversion.BuildCommit)
	}
}

func TestBuildIdentityPrefersGoVCSStampOverInjectedCommit(t *testing.T) {
	old := appversion.BuildCommit
	appversion.BuildCommit = "0123456789abcdef0123456789abcdef01234567"
	t.Cleanup(func() { appversion.BuildCommit = old })

	const vcs = "abcdef0123456789abcdef0123456789abcdef01"
	id := buildIdentity(&debug.BuildInfo{Settings: []debug.BuildSetting{{Key: "vcs.revision", Value: vcs}}})
	if id.Commit != vcs {
		t.Fatalf("Commit = %q, want Go vcs.revision %q to win over injected fallback", id.Commit, vcs)
	}
}
