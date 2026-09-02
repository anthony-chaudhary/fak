// Tests for harnessinspect's inspection report: the per-asset control
// classification (mandatory > locked > re-resolvable) and detail fallback
// chain, Inspect's sorting and lock-field projection, and the Render text
// contract that operators read. Fixtures are constructed in-memory; no lock
// file is read and main() is never invoked.
package harnessinspect

import (
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/harnesscompose"
	"github.com/anthony-chaudhary/fak/internal/harnessresolve"
	"github.com/anthony-chaudhary/fak/internal/stackresolve"
)

func TestInspectAssetControlAndDetail(t *testing.T) {
	cases := []struct {
		name        string
		asset       harnesscompose.EffectiveAsset
		wantCap     string
		wantControl string
		wantDetail  string
		wantGrants  []string
		wantDenies  []string
	}{
		{
			name:        "mandatory beats every other control",
			asset:       harnesscompose.EffectiveAsset{Kind: "model", ID: "qwen", Value: "8b", Locked: true, Mandatory: true, Source: "manifest"},
			wantCap:     "model:qwen",
			wantControl: "mandatory",
			wantDetail:  "8b",
		},
		{
			name:        "locked by source",
			asset:       harnesscompose.EffectiveAsset{Kind: "policy", ID: "ro", Ref: "policy.json", Source: "manifest", Locked: true},
			wantCap:     "policy:ro",
			wantControl: "locked by source",
			wantDetail:  "policy.json",
		},
		{
			name:        "default control is re-resolvable",
			asset:       harnesscompose.EffectiveAsset{Kind: "memory", ID: "kv", Boundary: "4GiB", Source: "layer"},
			wantCap:     "memory:kv",
			wantControl: "changeable by re-resolve",
			wantDetail:  "4GiB",
		},
		{
			name:        "detail falls back value then ref then boundary",
			asset:       harnesscompose.EffectiveAsset{Kind: "tool", ID: "web", Ref: "", Boundary: "allowlist", Source: "layer"},
			wantCap:     "tool:web",
			wantControl: "changeable by re-resolve",
			wantDetail:  "allowlist",
		},
		{
			name:        "grants and denies pass through",
			asset:       harnesscompose.EffectiveAsset{Kind: "tool", ID: "fs", Value: "sandbox", Source: "manifest", Grants: []string{"read"}, Denies: []string{"write", "exec"}},
			wantCap:     "tool:fs",
			wantControl: "changeable by re-resolve",
			wantDetail:  "sandbox",
			wantGrants:  []string{"read"},
			wantDenies:  []string{"write", "exec"},
		},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			got := inspectAsset(tc.asset)
			if got.Capability != tc.wantCap {
				t.Errorf("Capability = %q, want %q", got.Capability, tc.wantCap)
			}
			if got.Control != tc.wantControl {
				t.Errorf("Control = %q, want %q", got.Control, tc.wantControl)
			}
			if got.Detail != tc.wantDetail {
				t.Errorf("Detail = %q, want %q", got.Detail, tc.wantDetail)
			}
			if got.Source != tc.asset.Source {
				t.Errorf("Source = %q, want %q", got.Source, tc.asset.Source)
			}
			if strings.Join(got.Grants, ",") != strings.Join(tc.wantGrants, ",") {
				t.Errorf("Grants = %v, want %v", got.Grants, tc.wantGrants)
			}
			if strings.Join(got.Denies, ",") != strings.Join(tc.wantDenies, ",") {
				t.Errorf("Denies = %v, want %v", got.Denies, tc.wantDenies)
			}
		})
	}
}

func inspectFixture(t *testing.T, lockPath string) (Report, harnessresolve.Lock) {
	t.Helper()
	lock := harnessresolve.Lock{
		Schema:      harnessresolve.LockSchema,
		ID:          "agent-harness",
		Environment: harnessresolve.Environment{OS: "darwin", Arch: "arm64", Contract: "contract-a"},
		Budget:      harnessresolve.Budget{ContextTokens: 100000, MemoryMiB: 64, Workers: 2},
		Components: []harnessresolve.LockedComponent{
			{ID: "zeta-core", Version: "v2", Source: "manifest", Reason: "root selection", Provides: []string{"caps.z"}},
			{ID: "alpha-core", Version: "v1", Source: "manifest", Reason: "root selection", Provides: []string{"caps.a"}},
		},
		Assets: []harnesscompose.EffectiveAsset{
			{Kind: "memory", ID: "kv", Boundary: "4GiB", Source: "layer"},
			{Kind: "model", ID: "qwen", Value: "8b", Mandatory: true, Source: "manifest"},
		},
		Decisions: []stackresolve.Decision{{From: "manifest", Chosen: "alpha-core"}, {From: "manifest", Chosen: "zeta-core"}},
	}
	return Inspect(lock, lockPath), lock
}

func TestInspectProjectsAndSortsLock(t *testing.T) {
	report, lock := inspectFixture(t, "run/agent.lock.json")
	if report.Schema != Schema {
		t.Fatalf("Schema = %q, want %q", report.Schema, Schema)
	}
	if !report.Verified {
		t.Error("Verified = false, want true")
	}
	if report.LockID != lock.ID {
		t.Errorf("LockID = %q, want %q", report.LockID, lock.ID)
	}
	if report.Environment != lock.Environment || report.Budget != lock.Budget {
		t.Errorf("environment/budget not copied from the lock: %+v", report)
	}
	if report.Decisions != 2 {
		t.Errorf("Decisions = %d, want 2", report.Decisions)
	}
	if len(report.Components) != 2 || report.Components[0].ID != "alpha-core" || report.Components[1].ID != "zeta-core" {
		t.Fatalf("components not sorted by ID: %+v", report.Components)
	}
	if report.Components[0].Version != "v1" || report.Components[0].Reason != "root selection" || strings.Join(report.Components[0].Provides, ",") != "caps.a" {
		t.Errorf("component projection lost lock fields: %+v", report.Components[0])
	}
	if len(report.Assets) != 2 || report.Assets[0].Capability != "memory:kv" || report.Assets[1].Capability != "model:qwen" {
		t.Fatalf("assets not sorted by capability: %+v", report.Assets)
	}
	if len(report.Controls) != 3 {
		t.Fatalf("Controls = %v, want three control lines", report.Controls)
	}
	if !strings.Contains(report.Controls[0], "run/agent.lock.json") {
		t.Errorf("Controls[0] does not name the lock path: %q", report.Controls[0])
	}
}

func TestInspectQuotesLockPathWithSpaces(t *testing.T) {
	report, _ := inspectFixture(t, "my locks/agent.lock.json")
	if !strings.Contains(report.Controls[0], `"my locks/agent.lock.json"`) {
		t.Errorf("lock path with a space was not quoted: %q", report.Controls[0])
	}
}

func TestRenderSections(t *testing.T) {
	report, _ := inspectFixture(t, "run/agent.lock.json")
	out := Render(report)
	for _, want := range []string{
		"HARNESS INSPECT | VERIFIED",
		"lock: agent-harness",
		"environment: darwin/arm64 | contract contract-a",
		"budget: context=100000 tokens | memory=64 MiB | workers=2",
		"components (2):",
		"- alpha-core@v1 | from manifest | root selection | provides caps.a",
		"- zeta-core@v2 | from manifest | root selection | provides caps.z",
		"effective capabilities (2):",
		"- memory:kv | from layer | changeable by re-resolve | 4GiB",
		"- model:qwen | from manifest | mandatory | 8b",
		"resolver decisions: 2",
		"controls:",
	} {
		if !strings.Contains(out, want) {
			t.Fatalf("Render output missing %q:\n%s", want, out)
		}
	}
	// Sections appear in operator reading order.
	order := []string{"lock: agent-harness", "components (2):", "effective capabilities (2):", "controls:"}
	last := -1
	for _, marker := range order {
		at := strings.Index(out, marker)
		if at <= last {
			t.Fatalf("Render section %q out of order (at %d, previous %d):\n%s", marker, at, last, out)
		}
		last = at
	}
}

func TestRenderGrantsAndDenies(t *testing.T) {
	report := Report{
		Schema: Schema, Verified: true, LockID: "l",
		Assets: []Asset{{Capability: "tool:fs", Source: "manifest", Control: "mandatory", Detail: "sandbox", Grants: []string{"read"}, Denies: []string{"write"}}},
	}
	out := Render(report)
	if !strings.Contains(out, "- tool:fs | from manifest | mandatory | sandbox | grants read | denies write") {
		t.Fatalf("Render dropped grants/denies detail:\n%s", out)
	}
}
