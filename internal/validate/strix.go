package validate

import (
	"bufio"
	"context"
	"errors"
	"fmt"
	"io"
	"os"
	"path"
	"path/filepath"
	"sort"
	"strconv"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/amdgpu"
)

const publicFakModulePath = "github.com/anthony-chaudhary/fak"

// isPublicValidationModule keeps runtime qualification separate from companion
// control-plane code. A backend name in a factory path does not change a device.
func isPublicValidationModule(root string) bool {
	f, err := os.Open(filepath.Join(root, "go.mod"))
	if err != nil {
		return false
	}
	defer f.Close()

	scanner := bufio.NewScanner(f)
	for scanner.Scan() {
		fields := strings.Fields(scanner.Text())
		if len(fields) != 2 || fields[0] != "module" {
			continue
		}
		modulePath := fields[1]
		if unquoted, err := strconv.Unquote(modulePath); err == nil {
			modulePath = unquoted
		}
		return modulePath == publicFakModulePath
	}
	return false
}

// isGPURelatedValidation reports whether any changed paths affect GPU/compute subsystems.
func isGPURelatedValidation(mine []string) bool {
	gpuRoots := []string{
		"internal/amdgpu",
		"internal/compute",
		"internal/roofline",
		"cmd/fak/validate_acceptance",
		"cmd/fak/validate_acceptance.go",
		"cmd/fak/validate_acceptance_test.go",
	}
	// internal/model is deliberately NOT a blanket GPU root: it is the largest
	// package in the tree, and most of it (model geometry, quant math, routing,
	// loaders, the V4.1 forward) is pure-CPU and host-independent. Only the
	// device-surface files under internal/model justify physical hardware, and
	// those are selected by the framework-marker test below.
	const modelRoot = "internal/model"
	gpuKeywords := []string{
		"strix",
		"halo",
		"vulkan",
		"gfx115",
	}
	// modelGPUMarkers name the device frameworks/backends that can appear in an
	// internal/model path and mean the file carries GPU execution surface. A
	// pure-CPU model file (llm.go, v41_forward.go, v4_topk_partial.go) matches
	// none of these and must not demand physical Strix silicon.
	modelGPUMarkers := []string{
		"metal",
		"vulkan",
		"gpudirect",
		"cuda",
		"rocm",
		"hip",
		"amd",
		"gfx",
	}
	for _, p := range mine {
		norm := strings.ReplaceAll(p, "\\", "/")
		if strings.HasPrefix(norm, "/") || strings.Contains(norm, ":") {
			return true
		}
		for _, part := range strings.Split(norm, "/") {
			if part == ".." {
				return true
			}
		}
		norm = path.Clean(norm)
		if isCUDAOnlyComputeValidationPath(norm) {
			continue
		}
		lower := strings.ToLower(norm)
		for _, root := range gpuRoots {
			if norm == root || strings.HasPrefix(norm, root+"/") {
				return true
			}
		}
		// The loose keyword test applies only to source/kernel artifacts. A
		// documentation or data path that merely mentions a GPU keyword must
		// not force physical Strix hardware validation on a docs-only land.
		if !isSourcePath(norm) {
			continue
		}
		// The keyword arm skips two classes that never carry device surface:
		//   - test files (`_test.go`): a test ships no device code; tests under
		//     a GPU root already triggered via the root match above, and model
		//     device tests still trigger via the marker arm below.
		//   - the private factory/control-plane tree (`platform/`): it carries
		//     no device surface by boundary rule; device HAL lives in public
		//     internal/. Its component names (e.g. a `strix` or `halo` package)
		//     describe what it manages, not code that runs on the device.
		if keywordArmApplies(norm) {
			for _, kw := range gpuKeywords {
				if strings.Contains(lower, kw) {
					return true
				}
			}
		}
		if norm == modelRoot || strings.HasPrefix(norm, modelRoot+"/") {
			for _, marker := range modelGPUMarkers {
				if strings.Contains(lower, marker) {
					return true
				}
			}
		}
	}
	return false
}

func isCUDAOnlyComputeValidationPath(norm string) bool {
	const computeRoot = "internal/compute/"
	if !strings.HasPrefix(norm, computeRoot) {
		return false
	}
	rel := strings.TrimPrefix(norm, computeRoot)
	// Keep cross-backend artifacts under CUDA namespaces hardware-relevant.
	for _, part := range strings.FieldsFunc(strings.ToLower(rel), func(r rune) bool {
		return r < 'a' || r > 'z'
	}) {
		for _, marker := range []string{"strix", "halo", "vulkan", "rocm", "hip", "amd", "gfx", "metal"} {
			if strings.HasPrefix(part, marker) {
				return false
			}
		}
	}
	if rel == "testdata/cuda_device_context.cpp" {
		return true
	}
	switch path.Ext(rel) {
	case ".go", ".h", ".cu", ".cuh":
	default:
		return false
	}
	if strings.HasPrefix(rel, "cuda/") {
		return true
	}
	if strings.Contains(rel, "/") {
		return false
	}
	return rel == "cuda.go" || strings.HasPrefix(rel, "cuda_") || strings.HasSuffix(rel, "_cuda.go")
}

// keywordArmApplies reports whether the loose GPU-keyword substring test may
// fire for a source path. Tests, private factory/control-plane trees, and host
// Go sources in internal/validate are excluded. That package orchestrates checks,
// so validator maintenance stays host-testable; device payloads remain eligible.
func keywordArmApplies(norm string) bool {
	if strings.HasSuffix(strings.ToLower(norm), "_test.go") {
		return false
	}
	if path.Dir(norm) == "internal/validate" && strings.EqualFold(path.Ext(norm), ".go") {
		return false
	}
	if norm == "platform" || strings.HasPrefix(norm, "platform/") {
		return false
	}
	for _, root := range []string{"cmd/fak-flow", "cmd/fak-server", "cmd/fak-strix", "cmd/fak-sync"} {
		if norm == root || strings.HasPrefix(norm, root+"/") {
			return false
		}
	}
	return true
}

// sourcePathExtensions are the artifact kinds that can carry GPU/compute
// implementation or device payloads and therefore justify the keyword-based
// hardware-validation trigger. Documentation (`.md`), JSON, and other data
// paths are excluded so a docs-only change cannot demand physical hardware.
var sourcePathExtensions = map[string]bool{
	".go":    true,
	".s":     true,
	".asm":   true,
	".c":     true,
	".cc":    true,
	".cpp":   true,
	".cxx":   true,
	".h":     true,
	".hh":    true,
	".hpp":   true,
	".cu":    true,
	".cuh":   true,
	".hip":   true,
	".cl":    true,
	".spv":   true,
	".comp":  true,
	".metal": true,
	".rs":    true,
	".bin":   true,
	".fw":    true,
	".elf":   true,
}

// isSourcePath reports whether a repo-relative path names an implementation
// artifact or device payload rather than documentation, configuration, or data.
func isSourcePath(norm string) bool {
	name := norm
	if idx := strings.LastIndex(norm, "/"); idx >= 0 {
		name = norm[idx+1:]
	}
	dot := strings.LastIndex(name, ".")
	if dot < 0 {
		return false
	}
	return sourcePathExtensions[strings.ToLower(name[dot:])]
}

var (
	discoverStrixTargetFn           = amdgpu.DiscoverStrixTarget
	newStrixControllerAuthorityFn   = amdgpu.NewStrixControllerAuthority
	strixControllerAuthorityValidFn = func(authority amdgpu.StrixControllerAuthority) bool {
		return authority.Epoch() != "" &&
			authority.ObservedRevision() != "" &&
			authority.ExecutableSHA256() != "" &&
			authority.ExecutableBytes() > 0 &&
			!authority.AdmittedAt().IsZero()
	}
	runStrixValidationFn         = amdgpu.RunStrixValidation
	buildStrixCandidateArchiveFn = amdgpu.BuildStrixCandidateArchive
)

// HardwareQualification records missing evidence, not a physical execution pass.
// Software validation can finish while these backend-specific obligations remain.
type HardwareQualification struct {
	Status   string   `json:"status"`
	Backends []string `json:"pending_backends"`
	Detail   string   `json:"detail"`
}

// hardwareQualificationBackends selects evidence by changed implementation,
// rather than treating every compute path as an AMD appliance workload. Unknown
// production compute code remains shared; unmarked Go tests do not ship kernels.
// The separate, conservative isGPURelatedValidation predicate is retained for
// stale-receipt interval checks, where narrowing relevance could admit old proof.
func hardwareQualificationBackends(mine []string) []string {
	backends := make(map[string]bool)
	for _, changed := range mine {
		norm := strings.ReplaceAll(changed, "\\", "/")
		if strings.HasPrefix(norm, "/") || strings.Contains(norm, ":") || strings.Contains("/"+norm+"/", "/../") {
			backends["shared"] = true
			continue
		}
		norm = path.Clean(norm)
		if !isSourcePath(norm) {
			continue
		}
		if isCUDAOnlyComputeValidationPath(norm) {
			backends["cuda"] = true
			continue
		}
		compute := strings.HasPrefix(norm, "internal/compute/")
		model := strings.HasPrefix(norm, "internal/model/")
		amd := strings.HasPrefix(norm, "internal/amdgpu/")
		if !compute && !model && !amd && !keywordArmApplies(norm) {
			continue
		}
		matched := false
		// Native device-language extensions retain their backend even when the
		// filename has no framework prefix; bridge markers can add another one.
		switch strings.ToLower(path.Ext(norm)) {
		case ".cu", ".cuh":
			backends["cuda"], matched = true, true
		case ".metal":
			backends["metal"], matched = true, true
		case ".hip":
			backends["rocm"], matched = true, true
		case ".comp", ".spv":
			backends["amd-vulkan"], matched = true, true
		}
		for _, token := range strings.FieldsFunc(strings.ToLower(norm), func(r rune) bool {
			return !(r >= 'a' && r <= 'z' || r >= '0' && r <= '9')
		}) {
			backend := ""
			switch {
			case token == "amd64":
				// amd64 is a CPU ISA, not an AMD device backend.
			case strings.HasPrefix(token, "cuda"), strings.HasPrefix(token, "gpudirect"):
				if compute || model {
					backend = "cuda"
				}
			case strings.HasPrefix(token, "metal"):
				if compute || model || path.Ext(norm) == ".metal" {
					backend = "metal"
				}
			case strings.HasPrefix(token, "hip"), strings.HasPrefix(token, "rocm"):
				backend = "rocm"
			case strings.HasPrefix(token, "vulkan"), strings.HasPrefix(token, "strix"), token == "halo", token == "amd", token == "amdgpu", strings.HasPrefix(token, "gfx"):
				backend = "amd-vulkan"
			}
			if backend != "" {
				backends[backend], matched = true, true
			}
		}
		if amd {
			backends["amd-vulkan"], matched = true, true
		}
		if !matched && ((compute && !strings.HasSuffix(norm, "_test.go")) || strings.HasPrefix(norm, "internal/roofline/") || strings.HasPrefix(norm, "cmd/fak/validate_acceptance")) {
			backends["shared"] = true
		}
	}
	result := make([]string, 0, len(backends))
	for backend := range backends {
		result = append(result, backend)
	}
	sort.Strings(result)
	return result
}

func recordPendingHardware(res *validateResult, root string, mine []string) {
	if !isPublicValidationModule(root) {
		return
	}
	backends := hardwareQualificationBackends(mine)
	if len(backends) == 0 {
		return
	}
	res.HardwareQualification = &HardwareQualification{
		Status: "PENDING_HARDWARE", Backends: backends,
		Detail: "Software checks do not qualify physical execution or performance. Strix Halo appliances are retired; use an independently authorized, currently admitted target for the selected backend and return source-bound evidence.",
	}
}

func runValidateHardwarePhase(ctx context.Context, stdout, stderr io.Writer, res *validateResult, recorder *validateRecorder, root string, opts validateOptions, mine []string) {
	recordPendingHardware(res, root, mine)
	useReceipt := opts.strix || strings.TrimSpace(opts.strixReceipt) != ""
	if q := res.HardwareQualification; q != nil {
		for _, backend := range q.Backends {
			useReceipt = useReceipt || backend == "amd-vulkan" || backend == "shared"
		}
	}
	// An ambient Strix receipt is relevant only to its backend. Explicit
	// requests remain fail closed even when no changed path selects that backend.
	if receiptPath := hwReceiptPath(opts.strixReceipt); useReceipt && receiptPath != "" {
		// Supplying a receipt is an explicit qualification request, even for a
		// host-only delta. Missing, stale and invalid evidence must fail closed.
		// Archive binding proves which source was tested, not which kernels
		// executed. Keep broader qualification pending: argmax or SSD-restart
		// evidence cannot discharge arbitrary changed GPU operations.
		admitHWReceipt(ctx, res, recorder, root, receiptPath, mine)
		return
	}
	if err := executeStrixValidationPhase(ctx, stdout, stderr, res, recorder, root, opts.strix, opts.strixHost, opts.subkernels, opts.ablate, mine); err != nil {
		res.OK = false
	}
}

// executeStrixValidationPhase runs physical validation on Strix Halo during fak
// validate. It is a thin orchestrator over four stages: controller-authority
// acquisition, target probe, options build, and receipt evaluation.
func executeStrixValidationPhase(
	ctx context.Context,
	stdout, stderr io.Writer,
	res *validateResult,
	recorder *validateRecorder,
	root string,
	explicitStrix bool,
	hostOverride string,
	subkernelsArg string,
	ablateArg string,
	mine []string,
) error {
	// Runtime paths select qualification obligations, never permission to wake
	// or dispatch a retired appliance. Explicit requests retain every authority,
	// candidate-binding and receipt gate below.
	if !explicitStrix {
		recordPendingHardware(res, root, mine)
		recorder.skip("strix_validation", "no implicit hardware dispatch; backend qualification is reported separately")
		res.SkippedPhases = append(res.SkippedPhases, "strix_validation")
		return nil
	}

	phase := recorder.start("strix_validation")

	authority, err := acquireStrixControllerAuthority(ctx, root, res, phase, mine)
	if err != nil {
		return err
	}
	// Keep the opaque authority alive for the full phase. A later receipt-binding
	// leaf can consume it without reconstructing or serializing private evidence.
	_ = authority

	target, err := probeStrixTarget(ctx, explicitStrix, hostOverride, res, phase)
	if err != nil {
		return err
	}

	opts, err := buildStrixValidationOpts(ctx, root, target, res, subkernelsArg, ablateArg, mine, phase)
	if err != nil {
		return err
	}

	return evaluateStrixReceipt(ctx, opts, target, res, phase)
}

// acquireStrixControllerAuthority obtains and validates the opaque controller
// authority required before any physical Strix work. On refusal it records the
// ciPreflight failure, finishes the phase, and wraps the refusal for Run.
func acquireStrixControllerAuthority(
	ctx context.Context,
	root string,
	res *validateResult,
	phase validateActivePhase,
	mine []string,
) (amdgpu.StrixControllerAuthority, error) {
	authority, err := newStrixControllerAuthorityFn(ctx, root)
	if err == nil && !strixControllerAuthorityValidFn(authority) {
		err = errors.New("controller authority constructor returned unusable authority")
	}
	if err != nil {
		code := "CONTROLLER_AUTHORITY_UNAVAILABLE"
		recovery := ""
		var refusal interface {
			Code() string
			Recovery() string
		}
		if errors.As(err, &refusal) {
			code = refusal.Code()
			recovery = refusal.Recovery()
		}
		detail := fmt.Sprintf("strix hardware validation required for relevant changes but controller authority refused: %s (observed_revision=%s executable_sha256=%s)",
			code, authority.ObservedRevision(), authority.ExecutableSHA256())
		var cleanupOwner interface{ RetryCleanup() error }
		if errors.As(err, &cleanupOwner) {
			if cleanupErr := cleanupOwner.RetryCleanup(); cleanupErr != nil {
				detail += "; cleanup_retry=pending"
			} else {
				detail += "; cleanup_retry=complete"
			}
		}
		if recovery != "" {
			detail += fmt.Sprintf("; recovery: %s", recovery)
		}
		phase.finish(fmt.Errorf("strix controller authority refused: %s", code))
		res.Failures = append(res.Failures, ciPreflightFailure{
			Step:   "strix-controller-authority",
			Detail: detail,
			Files:  mine,
		})
		res.OK = false
		return authority, fmt.Errorf("strix controller authority refused: %s: %w", code, err)
	}
	return authority, nil
}

// probeStrixTarget discovers the appliance under a short probe budget: implicit
// runs get a fast 1.5s probe, explicit --strix gets 5s. An unreachable target
// records the pending-hardware-evidence failure and stops the phase.
func probeStrixTarget(
	ctx context.Context,
	explicitStrix bool,
	hostOverride string,
	res *validateResult,
	phase validateActivePhase,
) (*amdgpu.StrixTarget, error) {
	// In auto mode, probe with a fast timeout (1.5s)
	probeTimeout := 1500 * time.Millisecond
	if explicitStrix {
		probeTimeout = 5 * time.Second
	}

	probeCtx, cancel := context.WithTimeout(ctx, probeTimeout)
	target, err := discoverStrixTargetFn(probeCtx, hostOverride)
	cancel()

	if err != nil || target == nil || !target.Reachable {
		unreachErr := err
		if unreachErr == nil {
			if target != nil && target.Error != "" {
				unreachErr = fmt.Errorf("%s", target.Error)
			} else {
				unreachErr = fmt.Errorf("appliance unreachable or target not discovered")
			}
		}
		detail := fmt.Sprintf("strix hardware validation required for relevant changes but appliance is unreachable (pending hardware evidence): %v", unreachErr)
		if explicitStrix {
			detail = fmt.Sprintf("strix hardware validation required for relevant changes but appliance is unreachable (pending hardware evidence): explicit --strix demanded: %v", unreachErr)
		}
		phase.finish(fmt.Errorf("strix validation required but appliance unreachable: %w", unreachErr))
		var failFiles []string
		if hostOverride != "" {
			failFiles = []string{hostOverride}
		} else if target != nil && target.Host != "" {
			failFiles = []string{target.Host}
		}
		res.Failures = append(res.Failures, ciPreflightFailure{
			Step:   "strix-validation",
			Detail: detail,
			Files:  failFiles,
		})
		res.OK = false
		return nil, fmt.Errorf("strix unreachable: %w", unreachErr)
	}
	return target, nil
}

// buildStrixValidationOpts resolves the subkernel/ablation selections, builds
// the exact candidate archive, and sizes the run timeout from the admission
// budget plus the per-execution TERM/kill-after envelope.
func buildStrixValidationOpts(
	ctx context.Context,
	root string,
	target *amdgpu.StrixTarget,
	res *validateResult,
	subkernelsArg string,
	ablateArg string,
	mine []string,
	phase validateActivePhase,
) (amdgpu.StrixValidationOpts, error) {
	// Prepare validation options
	skList := amdgpu.ValidationSubkernelSelection(subkernelsArg)

	runAblations := false
	var abList []string
	if ablateArg != "" {
		runAblations = true
		if ablateArg == "all" {
			abList = []string{"cpu_vs_vulkan_gpu", "fused_vs_discrete_norm_matmul"}
		} else {
			abList = strings.Split(ablateArg, ",")
		}
	}

	candidate, archiveErr := buildStrixCandidateArchiveFn(ctx, root, res.Tip, mine)
	if archiveErr != nil {
		phase.finish(archiveErr)
		res.OK = false
		res.Failures = append(res.Failures, ciPreflightFailure{Step: "strix-validation", Detail: fmt.Sprintf("exact candidate archive failed: %v", archiveErr), Files: mine})
		return amdgpu.StrixValidationOpts{}, fmt.Errorf("strix candidate archive: %w", archiveErr)
	}
	admissionTimeout := 30 * time.Second
	runCount := len(skList) + len(abList)
	// Reserve a fresh-build budget plus the complete worst-case admission and
	// TERM/kill-after envelope for every sequential device execution.
	totalTimeout := 3*time.Minute + time.Duration(runCount)*(admissionTimeout+65*time.Second)

	return amdgpu.StrixValidationOpts{
		Host:                 target.Host,
		RunSubkernels:        true,
		Subkernels:           skList,
		RunAblations:         runAblations,
		Ablations:            abList,
		GitRef:               res.Ref,
		GitTip:               res.Tip,
		Command:              "fak validate --strix",
		Timeout:              totalTimeout,
		RequireSourceBinding: true,
		CandidateArchive:     candidate.Bytes,
		SourceArchiveSHA256:  candidate.SourceArchiveSHA256,
		AdmissionTimeout:     admissionTimeout,
	}, nil
}

// evaluateStrixReceipt runs the physical validation and enforces the receipt
// invariants in order: non-nil receipt, Validate() integrity, PASS verdict,
// credit eligibility, and no trailing execution error. Each violated invariant
// records its own ciPreflight failure entry.
func evaluateStrixReceipt(
	ctx context.Context,
	opts amdgpu.StrixValidationOpts,
	target *amdgpu.StrixTarget,
	res *validateResult,
	phase validateActivePhase,
) error {
	receipt, valErr := runStrixValidationFn(ctx, opts)
	phase.finish(valErr)

	if receipt == nil {
		res.OK = false
		detail := "strix validation returned nil receipt"
		if valErr != nil {
			detail = fmt.Sprintf("strix validation failed: %v", valErr)
		}
		var files []string
		if target != nil && target.Host != "" {
			files = []string{target.Host}
		}
		res.Failures = append(res.Failures, ciPreflightFailure{
			Step:   "strix-validation",
			Detail: detail,
			Files:  files,
		})
		if valErr == nil {
			valErr = fmt.Errorf("strix validation returned nil receipt")
		}
		return valErr
	}

	res.StrixValidation = receipt
	if err := receipt.Validate(); err != nil {
		receipt.Verdict = "FAIL"
		receipt.Verified = false
		res.OK = false
		res.Failures = append(res.Failures, ciPreflightFailure{
			Step:   "strix-validation",
			Detail: fmt.Sprintf("receipt invariant validation failed: %v", err),
			Files:  []string{target.Host},
		})
		return fmt.Errorf("receipt invariant validation failed: %w", err)
	}

	if receipt.Verdict != "PASS" {
		res.OK = false
		if len(receipt.Failures) == 0 {
			res.Failures = append(res.Failures, ciPreflightFailure{
				Step:   "strix-validation",
				Detail: fmt.Sprintf("strix validation verdict: %s", receipt.Verdict),
				Files:  []string{target.Host},
			})
		} else {
			for _, f := range receipt.Failures {
				res.Failures = append(res.Failures, ciPreflightFailure{
					Step:   "strix-validation",
					Detail: f,
					Files:  []string{target.Host},
				})
			}
		}
		return fmt.Errorf("strix validation failed with verdict: %s", receipt.Verdict)
	}
	if !receipt.CreditEligible() {
		res.OK = false
		res.Failures = append(res.Failures, ciPreflightFailure{
			Step:   "strix-validation",
			Detail: "receipt is integrity-readable but not eligible for current physical Strix credit",
			Files:  []string{target.Host},
		})
		return fmt.Errorf("strix validation receipt is not credit eligible")
	}

	if valErr != nil {
		res.OK = false
		res.Failures = append(res.Failures, ciPreflightFailure{
			Step:   "strix-validation",
			Detail: fmt.Sprintf("strix validation execution error: %v", valErr),
			Files:  []string{target.Host},
		})
		return valErr
	}

	return nil
}
