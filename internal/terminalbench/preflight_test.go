package terminalbench

import (
	"strings"
	"testing"
)

func TestBuildRehearsalPreflightNeverAllowsResultClaim(t *testing.T) {
	// Every reachable host state — fully blocked through fully ready — must keep
	// result_claim_allowed false. A preflight measures readiness, not the bench.
	cases := []PreflightProbe{
		{},
		{HarborPresent: true},
		{HarborPresent: true, DockerEngineUp: true},
		{HarborPresent: true, DockerEngineUp: true, OpenAIKeyPresent: true},
		{HarborPresent: true, DockerEngineUp: true, OpenAIKeyPresent: true, GatewayChecked: true, GatewayReachable: true},
	}
	for i, probe := range cases {
		p := BuildRehearsalPreflight(RehearsalPreflightInput{Probe: probe})
		if p.ResultClaimAllowed {
			t.Fatalf("case %d: preflight must never allow a result claim", i)
		}
		if p.Schema != RehearsalPreflightSchema {
			t.Fatalf("case %d: schema = %q", i, p.Schema)
		}
		if p.EvidenceClass != "REHEARSAL_PREFLIGHT" {
			t.Fatalf("case %d: evidence class = %q", i, p.EvidenceClass)
		}
		if p.Dataset != OfficialTerminalBench21Dataset {
			t.Fatalf("case %d: dataset = %q", i, p.Dataset)
		}
		if len(p.KnownDependencies) == 0 {
			t.Fatalf("case %d: known dependencies must be recorded", i)
		}
	}
}

func TestBuildRehearsalPreflightStatusMatrix(t *testing.T) {
	tests := []struct {
		name       string
		probe      PreflightProbe
		wantStatus string
		wantReason string // first blocking reason, "" if none
		oracle     bool
		raw        bool
		fak        bool
	}{
		{
			name:       "harbor missing blocks oracle",
			probe:      PreflightProbe{},
			wantStatus: PreflightBlocked,
			wantReason: ReasonHarborMissing,
		},
		{
			name:       "docker down blocks oracle",
			probe:      PreflightProbe{HarborPresent: true},
			wantStatus: PreflightBlocked,
			wantReason: ReasonDockerEngineDown,
		},
		{
			name:       "oracle ready paid blocked without key",
			probe:      PreflightProbe{HarborPresent: true, DockerEngineUp: true},
			wantStatus: PreflightOracleReadyPaidWait,
			wantReason: ReasonOpenAIKeyMissing,
			oracle:     true,
		},
		{
			name:       "raw ready fak gateway unreachable",
			probe:      PreflightProbe{HarborPresent: true, DockerEngineUp: true, OpenAIKeyPresent: true, GatewayChecked: true, GatewayReachable: false},
			wantStatus: PreflightRawReadyFakWait,
			wantReason: ReasonFakGatewayUnreach,
			oracle:     true,
			raw:        true,
		},
		{
			name:       "fully ready",
			probe:      PreflightProbe{HarborPresent: true, DockerEngineUp: true, OpenAIKeyPresent: true, GatewayChecked: true, GatewayReachable: true},
			wantStatus: PreflightReady,
			wantReason: "",
			oracle:     true,
			raw:        true,
			fak:        true,
		},
		{
			name:       "ready to attempt raw with gateway unprobed leaves fak unverified",
			probe:      PreflightProbe{HarborPresent: true, DockerEngineUp: true, OpenAIKeyPresent: true},
			wantStatus: PreflightReady,
			wantReason: "",
			oracle:     true,
			raw:        true,
			fak:        false,
		},
		{
			name:       "oracle artifact required but missing blocks paid even with key",
			probe:      PreflightProbe{HarborPresent: true, DockerEngineUp: true, OpenAIKeyPresent: true, OracleArtifactRequired: true},
			wantStatus: PreflightOracleReadyPaidWait,
			wantReason: ReasonOracleArtifactMissing,
			oracle:     true,
			raw:        false,
			fak:        false,
		},
		{
			name:       "oracle artifact present unblocks paid",
			probe:      PreflightProbe{HarborPresent: true, DockerEngineUp: true, OpenAIKeyPresent: true, OracleArtifactRequired: true, OracleArtifactPresent: true},
			wantStatus: PreflightReady,
			wantReason: "",
			oracle:     true,
			raw:        true,
			fak:        false,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			p := BuildRehearsalPreflight(RehearsalPreflightInput{Probe: tt.probe})
			if p.Status != tt.wantStatus {
				t.Fatalf("status = %q, want %q", p.Status, tt.wantStatus)
			}
			if p.OracleSmokeReady != tt.oracle || p.RawPaidSmokeReady != tt.raw || p.FakPaidSmokeReady != tt.fak {
				t.Fatalf("readiness oracle=%t raw=%t fak=%t, want %t/%t/%t",
					p.OracleSmokeReady, p.RawPaidSmokeReady, p.FakPaidSmokeReady, tt.oracle, tt.raw, tt.fak)
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

func TestBuildRehearsalPreflightCarriesContext(t *testing.T) {
	p := BuildRehearsalPreflight(RehearsalPreflightInput{
		GeneratedAt:      "2026-06-26T00:00:00Z",
		Probe:            PreflightProbe{HarborPresent: true, HarborVersion: "0.15.0", DockerEngineUp: false, DockerDetail: "engine not reachable"},
		Dataset:          OfficialTerminalBench21Dataset,
		Issue:            "#900",
		OfficialContract: "experiments/agent-live/terminalbench-official-run-contract-20260626.json",
		SubmissionPacket: "docs/benchmarks/TERMINAL-BENCH-2.1-SUBMISSION-PACKET.md",
		CandidateTaskIDs: []string{"go-cli-help-benign", "python-config-fix-danger-after-tests"},
	})
	if p.Issue != "#900" || p.Dataset != OfficialTerminalBench21Dataset {
		t.Fatalf("context not carried: issue=%q dataset=%q", p.Issue, p.Dataset)
	}
	if p.OfficialContract == "" || p.SubmissionPacket == "" || len(p.CandidateTaskIDs) != 2 {
		t.Fatalf("campaign links/task ids not carried: %+v", p)
	}
	md := RenderRehearsalPreflightMarkdown(p)
	for _, want := range []string{
		"Terminal-Bench 2.1 Rehearsal Preflight",
		"Result claim allowed: `false`",
		ReasonDockerEngineDown,
		"harbor 0.15.0",
		"#900",
		"Known dependencies before any result claim",
	} {
		if !strings.Contains(md, want) {
			t.Fatalf("markdown missing %q:\n%s", want, md)
		}
	}
}
