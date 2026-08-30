package runtimecap

import (
	"fmt"
	"strings"
)

const ExecutionModeReceiptSchema = "fak-execution-mode-receipt/1"

const (
	ExecutionModeLocalAccelerator   = "local_accelerator"
	ExecutionModeLocalCPUDegraded   = "local_cpu_degraded"
	ExecutionModeRemoteBacked       = "remote_backed"
	ExecutionModeOfflineControlMock = "offline_control_mock"
	ExecutionModeOfflineModelBacked = "offline_model_backed"
	ExecutionModeControlOnly        = "control_only"
	ExecutionModeRefused            = "refused"
)

const (
	ExecutionHealthReady       = "ready"
	ExecutionHealthDegraded    = "degraded"
	ExecutionHealthOffline     = "offline"
	ExecutionHealthControlOnly = "control_only"
	ExecutionHealthRefused     = "refused"
	ExecutionHealthUnwitnessed = "unwitnessed"
)

const (
	EvidenceObserved    = "observed"
	EvidenceFixture     = "fixture"
	EvidenceUnknown     = "unknown"
	EvidenceUnwitnessed = "unwitnessed"
)

type ExecutionIdentity struct {
	Engine   string `json:"engine"`
	Backend  string `json:"backend"`
	Device   string `json:"device"`
	Provider string `json:"provider"`
	Model    string `json:"model"`
}

type ExecutionModeBinary struct {
	GOOS      string   `json:"goos"`
	GOARCH    string   `json:"goarch"`
	Runnable  bool     `json:"runnable"`
	BuildTags []string `json:"build_tags"`
}

type ExecutionModeView struct {
	Mode     string `json:"mode"`
	Health   string `json:"health"`
	Evidence string `json:"evidence"`
}

type ExecutionModeWitness struct {
	Status        string `json:"status"`
	Source        string `json:"source"`
	Certification string `json:"certification"`
}

type ExecutionModeTransition struct {
	From    []string `json:"from"`
	To      string   `json:"to"`
	Trigger string   `json:"trigger"`
}

type ExecutionModeReceipt struct {
	Schema              string                    `json:"schema"`
	Mode                string                    `json:"mode"`
	Health              string                    `json:"health"`
	ControlPlaneOwner   string                    `json:"control_plane_owner"`
	Binary              ExecutionModeBinary       `json:"binary"`
	Status              ExecutionModeView         `json:"status"`
	Audit               ExecutionModeView         `json:"audit"`
	ModelBacked         bool                      `json:"model_backed"`
	NativePerformance   bool                      `json:"native_performance"`
	Identity            ExecutionIdentity         `json:"identity"`
	FallbackReason      *Reason                   `json:"fallback_reason"`
	Offline             string                    `json:"offline"`
	Egress              string                    `json:"egress"`
	OperatingEnvelope   string                    `json:"operating_envelope"`
	PrerequisiteWitness string                    `json:"prerequisite_witness"`
	Witness             ExecutionModeWitness      `json:"witness"`
	Transitions         []ExecutionModeTransition `json:"transitions"`
	Valid               bool                      `json:"valid"`
	ValidationReason    *Reason                   `json:"validation_reason"`
}

type ExecutionModeOptions struct {
	Mode                string
	Health              string
	Engine              string
	Backend             string
	Device              string
	Provider            string
	Model               string
	FallbackReason      *Reason
	Offline             string
	Egress              string
	OperatingEnvelope   string
	PrerequisiteWitness string
	WitnessStatus       string
	WitnessSource       string
	Certification       string
	NativePerformance   bool
	StatusMode          string
	StatusHealth        string
	StatusEvidence      string
	AuditMode           string
	AuditHealth         string
	AuditEvidence       string
	GOOS                string
	GOARCH              string
	BinaryRunnable      bool
	BuildTags           []string
}

func ExecutionModeReceiptFromReport(report Report) ExecutionModeReceipt {
	mode := ExecutionModeControlOnly
	health := ExecutionHealthControlOnly
	opts := ExecutionModeOptions{
		Engine:              report.ModelExecution.Engine,
		Backend:             report.ModelExecution.Backend,
		FallbackReason:      report.ModelExecution.Reason,
		Offline:             EvidenceUnknown,
		Egress:              EvidenceUnknown,
		OperatingEnvelope:   EvidenceUnknown,
		PrerequisiteWitness: EvidenceUnknown,
		WitnessStatus:       EvidenceObserved,
		WitnessSource:       "runtime-capabilities",
		Certification:       "pre-payload-observation",
		StatusEvidence:      EvidenceUnwitnessed,
		AuditEvidence:       EvidenceUnwitnessed,
		GOOS:                report.GOOS, GOARCH: report.GOARCH, BinaryRunnable: report.BinaryRunnable, BuildTags: report.BuildTags,
	}

	exec := report.ModelExecution
	switch {
	case !exec.Runnable:
		mode, health = ExecutionModeRefused, ExecutionHealthRefused
	case exec.RemotePlacement != nil:
		mode, health = ExecutionModeRemoteBacked, ExecutionHealthReady
		opts.Provider = exec.RemotePlacement.Provider
		opts.Model = exec.RemotePlacement.Model
		opts.Egress = exec.RemotePlacement.Egress.State
		opts.PrerequisiteWitness = exec.RemotePlacement.Reachability.State
		opts.OperatingEnvelope = fmt.Sprintf("timeout_ms=%d,retry_ceiling=%d,budget_microusd=%d", exec.RemotePlacement.TimeoutMilliseconds, exec.RemotePlacement.RetryCeiling, exec.RemotePlacement.BudgetMicroUSD)
		opts.FallbackReason = exec.RemotePlacement.LocalFailureTrigger
	case exec.LocalCPUDegraded != nil:
		mode, health = ExecutionModeLocalCPUDegraded, ExecutionHealthDegraded
		opts.Model = exec.LocalCPUDegraded.Model
		opts.OperatingEnvelope = exec.LocalCPUDegraded.EnvelopeID
		opts.PrerequisiteWitness = exec.LocalCPUDegraded.QualityEquivalence
		opts.FallbackReason = exec.LocalCPUDegraded.Reason
		opts.NativePerformance = true
	case exec.Backend != "" && exec.Backend != "cpu-ref":
		mode, health = ExecutionModeLocalAccelerator, ExecutionHealthReady
		opts.Device = EvidenceUnknown
		opts.NativePerformance = true
	case exec.Backend == "cpu-ref" && exec.CPUEnvelope != "":
		mode, health = ExecutionModeOfflineModelBacked, ExecutionHealthOffline
		opts.Offline = "full"
		opts.Egress = "denied"
		opts.OperatingEnvelope = exec.CPUEnvelope
		opts.NativePerformance = true
	default:
		mode, health = ExecutionModeControlOnly, ExecutionHealthControlOnly
	}
	opts.Mode, opts.Health = mode, health
	if opts.FallbackReason == nil && (mode == ExecutionModeLocalAccelerator || mode == ExecutionModeOfflineModelBacked) {
		opts.FallbackReason = &Reason{Code: "not_applicable", Detail: "the selected execution mode did not enter through a fallback transition"}
	}
	return NewExecutionModeReceipt(opts)
}

func NewExecutionModeReceipt(opts ExecutionModeOptions) ExecutionModeReceipt {
	r := ExecutionModeReceipt{
		Schema:            ExecutionModeReceiptSchema,
		Mode:              strings.TrimSpace(opts.Mode),
		Health:            strings.TrimSpace(opts.Health),
		ControlPlaneOwner: "fak-local",
		Binary:            ExecutionModeBinary{GOOS: explicit(opts.GOOS), GOARCH: explicit(opts.GOARCH), Runnable: opts.BinaryRunnable, BuildTags: append([]string(nil), opts.BuildTags...)},
		Status: ExecutionModeView{
			Mode: defaultViewValue(opts.StatusMode, opts.Mode), Health: defaultViewValue(opts.StatusHealth, opts.Health), Evidence: explicit(opts.StatusEvidence),
		},
		Audit: ExecutionModeView{
			Mode: defaultViewValue(opts.AuditMode, opts.Mode), Health: defaultViewValue(opts.AuditHealth, opts.Health), Evidence: explicit(opts.AuditEvidence),
		},
		NativePerformance: opts.NativePerformance,
		Identity: ExecutionIdentity{
			Engine: explicit(opts.Engine), Backend: explicit(opts.Backend), Device: explicit(opts.Device),
			Provider: explicit(opts.Provider), Model: explicit(opts.Model),
		},
		FallbackReason: opts.FallbackReason,
		Offline:        explicit(opts.Offline), Egress: explicit(opts.Egress),
		OperatingEnvelope: explicit(opts.OperatingEnvelope), PrerequisiteWitness: explicit(opts.PrerequisiteWitness),
		Witness:     ExecutionModeWitness{Status: explicit(opts.WitnessStatus), Source: explicit(opts.WitnessSource), Certification: explicit(opts.Certification)},
		Transitions: executionModeTransitions(),
		Valid:       true,
	}
	r.ModelBacked = r.Mode == ExecutionModeLocalAccelerator || r.Mode == ExecutionModeLocalCPUDegraded || r.Mode == ExecutionModeRemoteBacked || r.Mode == ExecutionModeOfflineModelBacked
	if reason := validateExecutionModeReceipt(r); reason != nil {
		r.Valid = false
		r.Health = ExecutionHealthRefused
		r.Status.Health = ExecutionHealthRefused
		r.Audit.Health = ExecutionHealthRefused
		r.ValidationReason = reason
	}
	return r
}

func explicit(value string) string {
	if value = strings.TrimSpace(value); value != "" {
		return value
	}
	return EvidenceUnknown
}

func defaultViewValue(value, fallback string) string {
	if value = strings.TrimSpace(value); value != "" {
		return value
	}
	return strings.TrimSpace(fallback)
}

func validateExecutionModeReceipt(r ExecutionModeReceipt) *Reason {
	validModes := map[string]bool{
		ExecutionModeLocalAccelerator: true, ExecutionModeLocalCPUDegraded: true, ExecutionModeRemoteBacked: true,
		ExecutionModeOfflineControlMock: true, ExecutionModeOfflineModelBacked: true, ExecutionModeControlOnly: true, ExecutionModeRefused: true,
	}
	if !validModes[r.Mode] {
		return &Reason{Code: "execution_mode_unknown", Detail: "execution mode is outside fak-execution-mode-receipt/1", Remediation: "use one of the seven closed execution modes"}
	}
	if r.Status.Evidence == EvidenceObserved && r.Audit.Evidence == EvidenceObserved &&
		(r.Status.Mode != r.Audit.Mode || r.Status.Health != r.Audit.Health || r.Status.Mode != r.Mode || r.Status.Health != r.Health) {
		return &Reason{Code: "execution_view_mismatch", Detail: "independently observed health/status and audit execution views disagree", Remediation: "reconcile the observed transition before claiming readiness"}
	}
	if r.ModelBacked {
		if r.Identity.Engine == EvidenceUnknown || r.Identity.Model == EvidenceUnknown {
			return &Reason{Code: "model_identity_unwitnessed", Detail: "model-backed modes require exact engine and model evidence", Remediation: "name the actual engine and model artifact"}
		}
		if r.Mode == ExecutionModeRemoteBacked {
			if r.Identity.Provider == EvidenceUnknown {
				return &Reason{Code: "remote_provider_unwitnessed", Detail: "remote-backed mode requires exact provider evidence", Remediation: "name the actual remote provider"}
			}
		} else if r.Identity.Backend == EvidenceUnknown {
			return &Reason{Code: "local_backend_unwitnessed", Detail: "local model-backed mode requires exact backend evidence", Remediation: "name the actual local backend"}
		}
	}
	localNativeMode := r.Mode == ExecutionModeLocalAccelerator || r.Mode == ExecutionModeLocalCPUDegraded || r.Mode == ExecutionModeOfflineModelBacked
	if (r.NativePerformance || localNativeMode) && r.Identity.Engine != "fak-native" {
		return &Reason{Code: "native_engine_substitution", Detail: "native/performance receipt did not execute with fak-native", Remediation: "refuse the claim or execute end-to-end inside fak-native"}
	}
	return nil
}

func executionModeTransitions() []ExecutionModeTransition {
	return []ExecutionModeTransition{
		{From: []string{ExecutionModeControlOnly, ExecutionModeOfflineControlMock}, To: ExecutionModeLocalAccelerator, Trigger: "local_accelerator_admitted"},
		{From: []string{ExecutionModeLocalAccelerator}, To: ExecutionModeLocalCPUDegraded, Trigger: "explicit_cpu_fallback_admitted"},
		{From: []string{ExecutionModeLocalAccelerator, ExecutionModeLocalCPUDegraded}, To: ExecutionModeRemoteBacked, Trigger: "explicit_remote_fallback_admitted"},
		{From: []string{ExecutionModeControlOnly}, To: ExecutionModeOfflineControlMock, Trigger: "offline_mock_selected"},
		{From: []string{ExecutionModeControlOnly, ExecutionModeOfflineControlMock}, To: ExecutionModeOfflineModelBacked, Trigger: "offline_model_admitted"},
		{From: []string{ExecutionModeLocalAccelerator, ExecutionModeLocalCPUDegraded, ExecutionModeRemoteBacked, ExecutionModeOfflineControlMock, ExecutionModeOfflineModelBacked, ExecutionModeControlOnly}, To: ExecutionModeRefused, Trigger: "policy_or_capability_refusal"},
	}
}

func IsExecutionMode(mode string) bool {
	switch mode {
	case ExecutionModeLocalAccelerator, ExecutionModeLocalCPUDegraded, ExecutionModeRemoteBacked, ExecutionModeOfflineControlMock, ExecutionModeOfflineModelBacked, ExecutionModeControlOnly, ExecutionModeRefused:
		return true
	default:
		return false
	}
}

func ExecutionModeFixture(mode string) ExecutionModeReceipt {
	base := ExecutionModeOptions{Mode: mode, WitnessStatus: EvidenceFixture, WitnessSource: "deterministic-fixture", Certification: EvidenceUnwitnessed, Offline: EvidenceUnknown, Egress: EvidenceUnknown, OperatingEnvelope: EvidenceUnwitnessed, PrerequisiteWitness: EvidenceUnwitnessed, StatusEvidence: EvidenceUnwitnessed, AuditEvidence: EvidenceUnwitnessed}
	switch mode {
	case ExecutionModeLocalAccelerator:
		base.Health, base.Engine, base.Backend, base.Device, base.Model, base.NativePerformance = ExecutionHealthReady, "fak-native", "vulkan", "fixture-device", "fixture-qwen3.8", true
		base.FallbackReason = &Reason{Code: "not_applicable", Detail: "the fixture describes a direct accelerator selection"}
	case ExecutionModeLocalCPUDegraded:
		base.Health, base.Engine, base.Backend, base.Model, base.NativePerformance = ExecutionHealthDegraded, "fak-native", "cpu-ref", "fixture-qwen3.8", true
		base.FallbackReason = &Reason{Code: "fixture_accelerator_unavailable", Detail: "deterministic fixture only"}
	case ExecutionModeRemoteBacked:
		base.Health, base.Engine, base.Backend, base.Provider, base.Model = ExecutionHealthReady, "fixture-remote-engine", "remote:fixture", "fixture-provider", "fixture-qwen3.8"
		base.FallbackReason = &Reason{Code: "fixture_local_unavailable", Detail: "deterministic fixture only"}
	case ExecutionModeOfflineControlMock:
		base.Health, base.Engine, base.Backend, base.Offline, base.Egress = ExecutionHealthOffline, "mock", "mock", "full", "denied"
	case ExecutionModeOfflineModelBacked:
		base.Health, base.Engine, base.Backend, base.Model, base.Offline, base.Egress, base.NativePerformance = ExecutionHealthOffline, "fak-native", "cpu-ref", "fixture-qwen3.8", "full", "denied", true
		base.FallbackReason = &Reason{Code: "not_applicable", Detail: "the fixture describes a directly selected offline model"}
	case ExecutionModeControlOnly:
		base.Health = ExecutionHealthControlOnly
	case ExecutionModeRefused:
		base.Health = ExecutionHealthRefused
		base.FallbackReason = &Reason{Code: "fixture_refusal", Detail: "deterministic fixture only"}
	}
	return NewExecutionModeReceipt(base)
}
