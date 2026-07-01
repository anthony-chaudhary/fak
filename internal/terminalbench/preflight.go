package terminalbench

import (
	"fmt"
	"strings"
)

// RehearsalPreflightSchema identifies the host-readiness preflight artifact that
// gates the Terminal-Bench 2.1 raw/fak rehearsal (#900).
const RehearsalPreflightSchema = "fak.terminalbench21-rehearsal-preflight.v1"

// Closed-vocabulary preflight status values. They describe how far this host can
// get toward the rehearsal, never a benchmark result.
const (
	PreflightBlocked             = "BLOCKED_PREFLIGHT"
	PreflightOracleReadyPaidWait = "ORACLE_READY_PAID_BLOCKED"
	PreflightRawReadyFakWait     = "RAW_READY_FAK_GATEWAY_BLOCKED"
	PreflightReady               = "READY_TO_ATTEMPT_PAID_RUN"
)

// Closed-vocabulary blocking reasons.
const (
	ReasonHarborMissing         = "HARBOR_NOT_INSTALLED"
	ReasonDockerEngineDown      = "DOCKER_ENGINE_DOWN"
	ReasonOracleArtifactMissing = "ORACLE_SMOKE_ARTIFACT_MISSING"
	ReasonOpenAIKeyMissing      = "OPENAI_API_KEY_MISSING"
	ReasonFakGatewayUnreach     = "FAK_GATEWAY_UNREACHABLE"
)

// PreflightProbe is the observed host state. The caller (cmd/terminalbench)
// probes the live host; the classifier below is pure so it is fully
// unit-testable without Harbor, Docker, a key, or a network.
type PreflightProbe struct {
	HarborPresent    bool
	HarborVersion    string
	DockerEngineUp   bool
	DockerDetail     string
	OpenAIKeyPresent bool
	GatewayChecked   bool
	GatewayReachable bool
	GatewayURL       string
	// OracleArtifactRequired enforces acceptance criterion #2: the official
	// oracle smoke artifact must exist before any paid model rehearsal. When
	// required but absent, the paid arms stay blocked even on an otherwise
	// ready host.
	OracleArtifactRequired bool
	OracleArtifactPresent  bool
	OracleArtifactPath     string
}

// RehearsalPreflightInput carries the probe plus the campaign context the
// artifact records (dataset, contract/packet links, candidate task ids).
type RehearsalPreflightInput struct {
	GeneratedAt      string
	Probe            PreflightProbe
	Dataset          string
	Issue            string
	OfficialContract string
	SubmissionPacket string
	CandidateTaskIDs []string
}

// PreflightGate is one host capability the rehearsal needs.
type PreflightGate struct {
	Name   string `json:"name"`
	OK     bool   `json:"ok"`
	Detail string `json:"detail,omitempty"`
}

// RehearsalPreflight is the result-claim-gated host-readiness artifact.
type RehearsalPreflight struct {
	Schema             string          `json:"schema"`
	GeneratedAt        string          `json:"generated_at"`
	Benchmark          string          `json:"benchmark"`
	Issue              string          `json:"issue,omitempty"`
	Dataset            string          `json:"dataset,omitempty"`
	Status             string          `json:"status"`
	EvidenceClass      string          `json:"evidence_class"`
	ClaimBoundary      string          `json:"claim_boundary"`
	OracleSmokeReady   bool            `json:"oracle_smoke_ready"`
	RawPaidSmokeReady  bool            `json:"raw_paid_smoke_ready"`
	FakPaidSmokeReady  bool            `json:"fak_paid_smoke_ready"`
	Gates              []PreflightGate `json:"gates"`
	BlockingReasons    []string        `json:"blocking_reasons"`
	NextAction         string          `json:"next_action"`
	OfficialContract   string          `json:"official_run_contract,omitempty"`
	SubmissionPacket   string          `json:"submission_packet,omitempty"`
	CandidateTaskIDs   []string        `json:"candidate_task_ids,omitempty"`
	KnownDependencies  []string        `json:"known_dependencies,omitempty"`
	ResultClaimAllowed bool            `json:"result_claim_allowed"`
}

// BuildRehearsalPreflight classifies a probe into the gated preflight artifact.
// It never allows a result claim: a preflight measures host readiness, not the
// benchmark.
func BuildRehearsalPreflight(in RehearsalPreflightInput) RehearsalPreflight {
	p := in.Probe
	dataset := strings.TrimSpace(in.Dataset)
	if dataset == "" {
		dataset = OfficialTerminalBench21Dataset
	}
	oracleReady := p.HarborPresent && p.DockerEngineUp
	oracleArtifactBlocked := p.OracleArtifactRequired && !p.OracleArtifactPresent
	rawReady := oracleReady && p.OpenAIKeyPresent && !oracleArtifactBlocked
	gatewayBlocked := p.GatewayChecked && !p.GatewayReachable
	fakReady := rawReady && p.GatewayChecked && p.GatewayReachable

	var reasons []string
	if !p.HarborPresent {
		reasons = append(reasons, ReasonHarborMissing)
	}
	if !p.DockerEngineUp {
		reasons = append(reasons, ReasonDockerEngineDown)
	}
	// Oracle-before-paid: surface the missing oracle artifact ahead of the key,
	// because the operator must run the oracle smoke first to produce it.
	if oracleReady && oracleArtifactBlocked {
		reasons = append(reasons, ReasonOracleArtifactMissing)
	}
	if !p.OpenAIKeyPresent {
		reasons = append(reasons, ReasonOpenAIKeyMissing)
	}
	if gatewayBlocked {
		reasons = append(reasons, ReasonFakGatewayUnreach)
	}

	status := PreflightReady
	switch {
	case !oracleReady:
		status = PreflightBlocked
	case !rawReady:
		status = PreflightOracleReadyPaidWait
	case gatewayBlocked:
		status = PreflightRawReadyFakWait
	}

	gates := []PreflightGate{
		{Name: "harbor_present", OK: p.HarborPresent, Detail: harborDetail(p)},
		{Name: "docker_engine_up", OK: p.DockerEngineUp, Detail: dockerDetail(p)},
		{Name: "oracle_smoke_artifact", OK: !oracleArtifactBlocked, Detail: oracleArtifactDetail(p)},
		{Name: "openai_api_key_present", OK: p.OpenAIKeyPresent, Detail: keyDetail(p.OpenAIKeyPresent)},
		{Name: "fak_gateway_reachable", OK: p.GatewayChecked && p.GatewayReachable, Detail: gatewayDetail(p)},
	}

	return RehearsalPreflight{
		Schema:            RehearsalPreflightSchema,
		GeneratedAt:       strings.TrimSpace(in.GeneratedAt),
		Benchmark:         "Terminal-Bench 2.1 raw/fak rehearsal preflight",
		Issue:             strings.TrimSpace(in.Issue),
		Dataset:           dataset,
		Status:            status,
		EvidenceClass:     "REHEARSAL_PREFLIGHT",
		ClaimBoundary:     "Host-readiness preflight only: probes whether this host can attempt the Terminal-Bench 2.1 raw/fak rehearsal (Harbor, Docker engine, OPENAI_API_KEY, fak gateway). It is never a benchmark result; result_claim_allowed stays false. The oracle smoke and the credentialed raw-vs-fak compare remain the result-bearing artifacts.",
		OracleSmokeReady:  oracleReady,
		RawPaidSmokeReady: rawReady,
		FakPaidSmokeReady: fakReady,
		Gates:             gates,
		BlockingReasons:   reasons,
		NextAction:        preflightNextAction(reasons),
		OfficialContract:  strings.TrimSpace(in.OfficialContract),
		SubmissionPacket:  strings.TrimSpace(in.SubmissionPacket),
		CandidateTaskIDs:  in.CandidateTaskIDs,
		KnownDependencies: []string{
			"a running Docker engine for every Harbor task container (oracle, raw, and fak arms)",
			"OPENAI_API_KEY in the rehearsal shell for the credentialed raw baseline (#900)",
			"a running, reachable fak gateway so the fak arm routes through it (the client-facing /v1/responses inbound route shipped in #925; the remaining requirement is that the gateway is up on this host)",
			"explicit paid-spend authority before the credentialed raw/fak smoke pair",
			"the official Harbor grader output as the sole authority for any pass-rate number",
		},
		ResultClaimAllowed: false,
	}
}

// RenderRehearsalPreflightMarkdown renders the preflight as human-readable text.
func RenderRehearsalPreflightMarkdown(p RehearsalPreflight) string {
	var b strings.Builder
	fmt.Fprintf(&b, "# Terminal-Bench 2.1 Rehearsal Preflight\n\n")
	fmt.Fprintf(&b, "- Generated: `%s`\n", p.GeneratedAt)
	fmt.Fprintf(&b, "- Benchmark: `%s`\n", p.Benchmark)
	if p.Issue != "" {
		fmt.Fprintf(&b, "- Issue: `%s`\n", p.Issue)
	}
	if p.Dataset != "" {
		fmt.Fprintf(&b, "- Dataset: `%s`\n", p.Dataset)
	}
	fmt.Fprintf(&b, "- Status: `%s`\n", p.Status)
	fmt.Fprintf(&b, "- Evidence class: `%s`\n", p.EvidenceClass)
	fmt.Fprintf(&b, "- Result claim allowed: `%t`\n", p.ResultClaimAllowed)
	fmt.Fprintf(&b, "- Oracle smoke ready: `%t`\n", p.OracleSmokeReady)
	fmt.Fprintf(&b, "- Raw paid smoke ready: `%t`\n", p.RawPaidSmokeReady)
	fmt.Fprintf(&b, "- fak paid smoke ready: `%t`\n", p.FakPaidSmokeReady)
	fmt.Fprintf(&b, "- Boundary: %s\n\n", p.ClaimBoundary)

	fmt.Fprintf(&b, "## Host gates\n\n")
	fmt.Fprintf(&b, "| Gate | OK | Detail |\n")
	fmt.Fprintf(&b, "|---|:---:|---|\n")
	for _, g := range p.Gates {
		mark := "no"
		if g.OK {
			mark = "yes"
		}
		fmt.Fprintf(&b, "| `%s` | %s | %s |\n", g.Name, mark, mdCell(g.Detail))
	}

	if len(p.BlockingReasons) > 0 {
		fmt.Fprintf(&b, "\n## Blocking reasons\n\n")
		for _, r := range p.BlockingReasons {
			fmt.Fprintf(&b, "- `%s`\n", r)
		}
	}

	fmt.Fprintf(&b, "\n- Next action: %s\n", p.NextAction)

	if p.OfficialContract != "" {
		fmt.Fprintf(&b, "- Official-run contract: `%s`\n", p.OfficialContract)
	}
	if p.SubmissionPacket != "" {
		fmt.Fprintf(&b, "- Submission packet: `%s`\n", p.SubmissionPacket)
	}
	if len(p.CandidateTaskIDs) > 0 {
		fmt.Fprintf(&b, "- Candidate task ids: `%s`\n", strings.Join(p.CandidateTaskIDs, ", "))
	}

	if len(p.KnownDependencies) > 0 {
		fmt.Fprintf(&b, "\n## Known dependencies before any result claim\n\n")
		for _, d := range p.KnownDependencies {
			fmt.Fprintf(&b, "- %s\n", d)
		}
	}
	return b.String()
}

func preflightNextAction(reasons []string) string {
	if len(reasons) == 0 {
		return "run the official Harbor oracle smoke, then the bounded raw and fak paid smoke on the fixed task slice, then the full 2.1 rehearsal"
	}
	switch reasons[0] {
	case ReasonHarborMissing:
		return "install Harbor (uv tool install harbor), then re-run the preflight"
	case ReasonDockerEngineDown:
		return "start the Docker engine, then re-run the preflight and the official oracle smoke before any paid run"
	case ReasonOracleArtifactMissing:
		return "run the official Harbor oracle smoke and check in its artifact before any paid raw/fak run"
	case ReasonOpenAIKeyMissing:
		return "run the oracle smoke first, then set OPENAI_API_KEY in the rehearsal shell for the bounded raw paid smoke"
	case ReasonFakGatewayUnreach:
		return "start the fak gateway (the /v1/responses inbound route is shipped, #925) and confirm it is reachable, then re-check the fak arm"
	default:
		return "resolve the blocking reasons above, then re-run the preflight"
	}
}

func harborDetail(p PreflightProbe) string {
	if !p.HarborPresent {
		return "harbor not found on PATH"
	}
	if v := strings.TrimSpace(p.HarborVersion); v != "" {
		return "harbor " + v
	}
	return "harbor present"
}

func dockerDetail(p PreflightProbe) string {
	if d := strings.TrimSpace(p.DockerDetail); d != "" {
		return d
	}
	if p.DockerEngineUp {
		return "docker engine reachable"
	}
	return "docker engine not reachable"
}

func oracleArtifactDetail(p PreflightProbe) string {
	path := strings.TrimSpace(p.OracleArtifactPath)
	switch {
	case !p.OracleArtifactRequired:
		return "not required (pass --oracle-artifact to enforce oracle-before-paid)"
	case p.OracleArtifactPresent:
		if path != "" {
			return "present at " + path
		}
		return "present"
	default:
		if path != "" {
			return "required but missing: " + path
		}
		return "required but missing"
	}
}

func keyDetail(present bool) string {
	if present {
		return "OPENAI_API_KEY set"
	}
	return "OPENAI_API_KEY not set in this shell"
}

func gatewayDetail(p PreflightProbe) string {
	url := strings.TrimSpace(p.GatewayURL)
	switch {
	case !p.GatewayChecked:
		if url != "" {
			return "not probed (" + url + "); pass --probe-gateway to check"
		}
		return "not probed; pass --fak-gateway and --probe-gateway to check"
	case p.GatewayReachable:
		return "reachable at " + url
	default:
		return "unreachable at " + url
	}
}
