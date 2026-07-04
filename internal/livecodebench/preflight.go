package livecodebench

import (
	"strings"
)

// PreflightSchema identifies the host-readiness preflight artifact that gates
// a LiveCodeBench run (#2111, part of the LiveCodeBench epic #2085).
const PreflightSchema = "fak.livecodebench-preflight.v1"

// Closed-vocabulary preflight status values. They describe whether this host
// can attempt a LiveCodeBench run, never a benchmark result.
const (
	PreflightBlocked = "BLOCKED_PREFLIGHT"
	PreflightReady   = "READY"
)

// Closed-vocabulary blocking reasons.
const (
	ReasonUvMissing        = "UV_NOT_INSTALLED"
	ReasonPython311Missing = "PYTHON311_NOT_FOUND"
	ReasonDatasetUnreach   = "HF_DATASET_UNREACHABLE"
	ReasonGatewayUnreach   = "FAK_GATEWAY_UNREACHABLE"
	ReasonSandboxUnavail   = "SANDBOX_UNAVAILABLE"
)

// PreflightProbe is the observed host state. The caller (cmd/livecodebench)
// probes the live host; the classifier below is pure so it is fully
// unit-testable without uv, Python, a network, or a sandbox.
type PreflightProbe struct {
	UvPresent     bool
	UvVersion     string
	PythonPresent bool
	// PythonVersion must report a 3.11.x string to satisfy the gate; the
	// official LiveCodeBench runner pins Python 3.11.
	PythonVersion string

	DatasetChecked   bool
	DatasetReachable bool
	DatasetURL       string

	GatewayChecked   bool
	GatewayReachable bool
	GatewayURL       string

	SandboxAvailable bool
	SandboxDetail    string
}

// PreflightGate is one host capability the run needs.
type PreflightGate struct {
	Name   string `json:"name"`
	OK     bool   `json:"ok"`
	Detail string `json:"detail,omitempty"`
}

// Preflight is the result-claim-gated host-readiness artifact.
type Preflight struct {
	Schema             string          `json:"schema"`
	GeneratedAt        string          `json:"generated_at,omitempty"`
	Issue              string          `json:"issue,omitempty"`
	Status             string          `json:"status"`
	Gates              []PreflightGate `json:"gates"`
	BlockingReasons    []string        `json:"blocking_reasons"`
	NextAction         string          `json:"next_action"`
	ResultClaimAllowed bool            `json:"result_claim_allowed"`
	ClaimBoundary      string          `json:"claim_boundary"`
}

// PreflightInput carries the probe plus the campaign context the artifact
// records.
type PreflightInput struct {
	GeneratedAt string
	Issue       string
	Probe       PreflightProbe
}

// BuildPreflight classifies a probe into the gated preflight artifact. It
// never allows a result claim: a preflight measures host readiness, not the
// benchmark.
func BuildPreflight(in PreflightInput) Preflight {
	p := in.Probe
	pyOK := p.PythonPresent && isPython311(p.PythonVersion)
	datasetOK := p.DatasetChecked && p.DatasetReachable
	gatewayOK := p.GatewayChecked && p.GatewayReachable
	sandboxOK := p.SandboxAvailable

	var reasons []string
	if !p.UvPresent {
		reasons = append(reasons, ReasonUvMissing)
	}
	if !pyOK {
		reasons = append(reasons, ReasonPython311Missing)
	}
	if !datasetOK {
		reasons = append(reasons, ReasonDatasetUnreach)
	}
	if !gatewayOK {
		reasons = append(reasons, ReasonGatewayUnreach)
	}
	if !sandboxOK {
		reasons = append(reasons, ReasonSandboxUnavail)
	}

	status := PreflightReady
	if len(reasons) > 0 {
		status = PreflightBlocked
	}

	gates := []PreflightGate{
		{Name: "uv_present", OK: p.UvPresent, Detail: uvDetail(p)},
		{Name: "python311_present", OK: pyOK, Detail: pythonDetail(p)},
		{Name: "hf_dataset_reachable", OK: datasetOK, Detail: datasetDetail(p)},
		{Name: "fak_gateway_reachable", OK: gatewayOK, Detail: gatewayDetail(p)},
		{Name: "sandbox_available", OK: sandboxOK, Detail: sandboxDetail(p)},
	}

	return Preflight{
		Schema:          PreflightSchema,
		GeneratedAt:     strings.TrimSpace(in.GeneratedAt),
		Issue:           strings.TrimSpace(in.Issue),
		Status:          status,
		Gates:           gates,
		BlockingReasons: reasons,
		NextAction:      preflightNextAction(reasons),
		// A preflight probes host readiness only; it never carries a
		// benchmark number, so this stays false regardless of status.
		ResultClaimAllowed: false,
		ClaimBoundary:      "Host-readiness preflight only: probes whether this host can attempt a LiveCodeBench run (uv, Python 3.11, the HF dataset, the fak gateway /models route, a sandbox). It is never a benchmark result; result_claim_allowed stays false. The official lcb_runner grading remains the sole result-bearing authority.",
	}
}

func isPython311(version string) bool {
	v := strings.TrimSpace(version)
	return v == "3.11" || strings.HasPrefix(v, "3.11.")
}

func preflightNextAction(reasons []string) string {
	if len(reasons) == 0 {
		return "run the official LiveCodeBench evaluation via lcb_runner, then export and grade with the custom evaluator"
	}
	switch reasons[0] {
	case ReasonUvMissing:
		return "install uv (see https://docs.astral.sh/uv/), then re-run the preflight"
	case ReasonPython311Missing:
		return "install Python 3.11 (uv python install 3.11), then re-run the preflight"
	case ReasonDatasetUnreach:
		return "check network access to the HF LiveCodeBench dataset, then re-run the preflight"
	case ReasonGatewayUnreach:
		return "start the fak gateway and confirm /models is reachable, then re-run the preflight"
	case ReasonSandboxUnavail:
		return "make a sandbox available for code execution scenarios, then re-run the preflight"
	default:
		return "resolve the blocking reasons above, then re-run the preflight"
	}
}

func uvDetail(p PreflightProbe) string {
	if !p.UvPresent {
		return "uv not found on PATH"
	}
	if v := strings.TrimSpace(p.UvVersion); v != "" {
		return "uv " + v
	}
	return "uv present"
}

func pythonDetail(p PreflightProbe) string {
	v := strings.TrimSpace(p.PythonVersion)
	switch {
	case !p.PythonPresent:
		return "python not found on PATH"
	case v == "":
		return "python present but version unknown"
	case isPython311(v):
		return "python " + v
	default:
		return "python " + v + " found, want 3.11.x"
	}
}

func datasetDetail(p PreflightProbe) string {
	url := strings.TrimSpace(p.DatasetURL)
	switch {
	case !p.DatasetChecked:
		if url != "" {
			return "not probed (" + url + "); pass --probe-dataset to check"
		}
		return "not probed; pass --probe-dataset to check"
	case p.DatasetReachable:
		return "reachable at " + url
	default:
		return "unreachable at " + url
	}
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

func sandboxDetail(p PreflightProbe) string {
	if d := strings.TrimSpace(p.SandboxDetail); d != "" {
		return d
	}
	if p.SandboxAvailable {
		return "sandbox available"
	}
	return "no sandbox available"
}
