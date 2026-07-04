package livecodebench

import (
	"strings"
	"testing"
)

func TestBuildPreflightNeverAllowsResultClaim(t *testing.T) {
	// Every reachable host state -- fully blocked through fully ready -- must
	// keep result_claim_allowed false. A preflight measures readiness, not the
	// benchmark.
	cases := []PreflightProbe{
		{},
		{UvPresent: true},
		{UvPresent: true, PythonPresent: true, PythonVersion: "3.11.6"},
		{UvPresent: true, PythonPresent: true, PythonVersion: "3.11.6", DatasetChecked: true, DatasetReachable: true},
		{
			UvPresent: true, PythonPresent: true, PythonVersion: "3.11.6",
			DatasetChecked: true, DatasetReachable: true,
			GatewayChecked: true, GatewayReachable: true,
			SandboxAvailable: true,
		},
	}
	for i, probe := range cases {
		p := BuildPreflight(PreflightInput{Probe: probe})
		if p.ResultClaimAllowed {
			t.Fatalf("case %d: preflight must never allow a result claim", i)
		}
		if p.Schema != PreflightSchema {
			t.Fatalf("case %d: schema = %q", i, p.Schema)
		}
		if strings.TrimSpace(p.ClaimBoundary) == "" {
			t.Fatalf("case %d: claim boundary must be recorded", i)
		}
	}
}

func TestBuildPreflightStatusMatrix(t *testing.T) {
	ready := PreflightProbe{
		UvPresent: true, PythonPresent: true, PythonVersion: "3.11.6",
		DatasetChecked: true, DatasetReachable: true, DatasetURL: "https://huggingface.co/datasets/livecodebench/code_generation_lite",
		GatewayChecked: true, GatewayReachable: true, GatewayURL: "http://localhost:18080/v1/models",
		SandboxAvailable: true,
	}

	tests := []struct {
		name       string
		probe      PreflightProbe
		wantStatus string
		wantReason string // first blocking reason, "" if none
	}{
		{
			name:       "nothing present blocks on uv first",
			probe:      PreflightProbe{},
			wantStatus: PreflightBlocked,
			wantReason: ReasonUvMissing,
		},
		{
			name: "python not 3.11 blocks",
			probe: func() PreflightProbe {
				p := ready
				p.PythonVersion = "3.10.2"
				return p
			}(),
			wantStatus: PreflightBlocked,
			wantReason: ReasonPython311Missing,
		},
		{
			name: "dataset unreachable blocks",
			probe: func() PreflightProbe {
				p := ready
				p.DatasetReachable = false
				return p
			}(),
			wantStatus: PreflightBlocked,
			wantReason: ReasonDatasetUnreach,
		},
		{
			name: "gateway unreachable blocks",
			probe: func() PreflightProbe {
				p := ready
				p.GatewayReachable = false
				return p
			}(),
			wantStatus: PreflightBlocked,
			wantReason: ReasonGatewayUnreach,
		},
		{
			name: "sandbox missing blocks",
			probe: func() PreflightProbe {
				p := ready
				p.SandboxAvailable = false
				return p
			}(),
			wantStatus: PreflightBlocked,
			wantReason: ReasonSandboxUnavail,
		},
		{
			name:       "fully ready",
			probe:      ready,
			wantStatus: PreflightReady,
			wantReason: "",
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			p := BuildPreflight(PreflightInput{Probe: tt.probe})
			if p.Status != tt.wantStatus {
				t.Fatalf("status = %q, want %q (gates: %+v)", p.Status, tt.wantStatus, p.Gates)
			}
			if tt.wantReason == "" {
				if len(p.BlockingReasons) != 0 {
					t.Fatalf("expected no blocking reasons, got %+v", p.BlockingReasons)
				}
			} else if len(p.BlockingReasons) == 0 || p.BlockingReasons[0] != tt.wantReason {
				t.Fatalf("first blocking reason = %+v, want %q", p.BlockingReasons, tt.wantReason)
			}
			if strings.TrimSpace(p.NextAction) == "" {
				t.Fatal("next action must not be empty")
			}
		})
	}
}

func TestBuildPreflightDistinctFailingGates(t *testing.T) {
	// Acceptance: missing sandbox / dataset / gateway must each show as a
	// distinct failing gate (not collapsed into one generic reason).
	p := BuildPreflight(PreflightInput{Probe: PreflightProbe{
		UvPresent: true, PythonPresent: true, PythonVersion: "3.11.6",
	}})
	byName := map[string]PreflightGate{}
	for _, g := range p.Gates {
		byName[g.Name] = g
	}
	for _, name := range []string{"hf_dataset_reachable", "fak_gateway_reachable", "sandbox_available"} {
		g, ok := byName[name]
		if !ok {
			t.Fatalf("missing gate %q", name)
		}
		if g.OK {
			t.Fatalf("gate %q should be failing", name)
		}
	}
	wantReasons := map[string]bool{ReasonDatasetUnreach: true, ReasonGatewayUnreach: true, ReasonSandboxUnavail: true}
	for _, r := range p.BlockingReasons {
		delete(wantReasons, r)
	}
	if len(wantReasons) != 0 {
		t.Fatalf("blocking reasons missing distinct entries: %+v (got %+v)", wantReasons, p.BlockingReasons)
	}
}

func TestBuildPreflightCarriesContext(t *testing.T) {
	p := BuildPreflight(PreflightInput{
		GeneratedAt: "2026-07-04T00:00:00Z",
		Issue:       "#2111",
		Probe:       PreflightProbe{},
	})
	if p.Issue != "#2111" || p.GeneratedAt != "2026-07-04T00:00:00Z" {
		t.Fatalf("context not carried: issue=%q generated_at=%q", p.Issue, p.GeneratedAt)
	}
}
