package policy

import (
	"os"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/abi"
)

// TestMountViewExampleManifestParses is the load-path half of the #2577 witness:
// the example manifest under examples/ must decode through the real ParseManifest
// (which runs DisallowUnknownFields), proving mount_view is a recognized stanza.
func TestMountViewExampleManifestParses(t *testing.T) {
	b, err := os.ReadFile("../../examples/mountview-policy.json")
	if err != nil {
		t.Fatalf("read example manifest: %v", err)
	}
	m, err := ParseManifest(b)
	if err != nil {
		t.Fatalf("parse example manifest: %v", err)
	}
	if len(m.MountView) == 0 {
		t.Fatal("example manifest declared no mount_view")
	}
}

// TestMountViewRefusesOutOfViewPath is the enforcement half of the #2577 witness:
// an in-view read passes, an out-of-view read is DEFAULT_DENY, a write into a
// read-only subtree is POLICY_BLOCK, and an empty view admits (feature off).
func TestMountViewRefusesOutOfViewPath(t *testing.T) {
	view := []MountRule{{Path: "src", Mode: "rw"}, {Path: "docs", Mode: "ro"}}

	if code, ok := MountViewRefusal(view, "src/main.go", false); !ok {
		t.Fatalf("in-view read refused, cited %s", abi.ReasonName(code))
	}

	code, ok := MountViewRefusal(view, "secrets/id_rsa", false)
	if ok {
		t.Fatal("out-of-view read admitted")
	}
	if code != abi.ReasonDefaultDeny {
		t.Fatalf("out-of-view read cited %s, want DEFAULT_DENY", abi.ReasonName(code))
	}

	code, ok = MountViewRefusal(view, "docs/x.md", true)
	if ok {
		t.Fatal("write into read-only subtree admitted")
	}
	if code != abi.ReasonPolicyBlock {
		t.Fatalf("read-only write cited %s, want POLICY_BLOCK", abi.ReasonName(code))
	}

	// A read of the same read-only path is fine.
	if _, ok := MountViewRefusal(view, "docs/x.md", false); !ok {
		t.Fatal("read of in-view read-only path refused")
	}

	// Empty view = no view configured = feature off: everything visible.
	if _, ok := MountViewRefusal(nil, "anything/at/all", true); !ok {
		t.Fatal("empty view should admit (feature off)")
	}
}

// TestMountViewWitnessFromManifest ties the refusal to the manifest itself: a
// path outside every subtree the example declares does not exist to the agent,
// driven purely by the loaded policy — no model in the loop.
func TestMountViewWitnessFromManifest(t *testing.T) {
	b, err := os.ReadFile("../../examples/mountview-policy.json")
	if err != nil {
		t.Fatalf("read example manifest: %v", err)
	}
	m, err := ParseManifest(b)
	if err != nil {
		t.Fatalf("parse example manifest: %v", err)
	}
	code, ok := MountViewRefusal(m.MountView, "etc/shadow", false)
	if ok {
		t.Fatal("out-of-view path admitted by example manifest")
	}
	if code != abi.ReasonDefaultDeny {
		t.Fatalf("out-of-view path cited %s, want DEFAULT_DENY", abi.ReasonName(code))
	}
	// And an in-view read the example declares passes.
	if _, ok := MountViewRefusal(m.MountView, "src/lib.go", false); !ok {
		t.Fatal("in-view read refused by example manifest")
	}
}
