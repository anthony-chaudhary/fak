package devcmd

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"os/exec"
	"path/filepath"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/amdgpu"
)

var (
	runStrixValidationFn         = amdgpu.RunStrixValidation
	buildStrixCandidateArchiveFn = amdgpu.BuildStrixCandidateArchive
	gitRevParseFn                = defaultGitRevParse
	gitStatusFn                  = defaultGitStatus
)

type stringSliceFlag []string

func (s *stringSliceFlag) String() string { return strings.Join(*s, ",") }
func (s *stringSliceFlag) Set(val string) error {
	*s = append(*s, val)
	return nil
}

func defaultGitRevParse(ctx context.Context, dir string, args ...string) (string, error) {
	if dir == "" {
		dir = "."
	}
	cmd := exec.CommandContext(ctx, "git", append([]string{"-C", dir, "rev-parse"}, args...)...)
	out, err := cmd.Output()
	if err != nil {
		return "", err
	}
	return strings.TrimSpace(string(out)), nil
}

func defaultGitStatus(ctx context.Context, dir string) (string, error) {
	if dir == "" {
		dir = "."
	}
	cmd := exec.CommandContext(ctx, "git", "-C", dir, "status", "--porcelain")
	out, err := cmd.Output()
	if err != nil {
		return "", err
	}
	return strings.TrimSpace(string(out)), nil
}

// IsValidFullGitTip accepts only Git's full SHA-1 object identity.
func IsValidFullGitTip(s string) bool {
	s = strings.TrimSpace(s)
	if len(s) != 40 {
		return false
	}
	for _, c := range s {
		if !((c >= '0' && c <= '9') || (c >= 'a' && c <= 'f') || (c >= 'A' && c <= 'F')) {
			return false
		}
	}
	return true
}

type resolvedStrixCandidate struct {
	tip     string
	archive amdgpu.StrixCandidateArchive
}

func validateMinePaths(mine []string) error {
	seen := make(map[string]struct{}, len(mine))
	for _, raw := range mine {
		cleanRaw := strings.TrimSpace(raw)
		if cleanRaw == "" {
			return fmt.Errorf("empty --mine overlay path rejected")
		}
		if isCrossPlatformAbsolutePath(cleanRaw) {
			return fmt.Errorf("absolute --mine overlay path rejected: %q", raw)
		}
		norm := filepath.ToSlash(filepath.Clean(filepath.FromSlash(cleanRaw)))
		if norm == "." || norm == ".." || strings.HasPrefix(norm, "../") || strings.Contains(norm, "/../") {
			return fmt.Errorf("unsafe --mine overlay path rejected: %q", raw)
		}
		key := strings.ToLower(norm)
		if _, ok := seen[key]; ok {
			return fmt.Errorf("duplicate --mine overlay path rejected: %q", raw)
		}
		seen[key] = struct{}{}
	}
	return nil
}

func isCrossPlatformAbsolutePath(path string) bool {
	if filepath.IsAbs(path) || filepath.VolumeName(path) != "" || strings.HasPrefix(path, "/") || strings.HasPrefix(path, "\\") {
		return true
	}
	// filepath follows the controller OS. Recognize Windows drive syntax even
	// when the command is built or tested on Linux/WSL.
	return len(path) >= 2 && ((path[0] >= 'a' && path[0] <= 'z') || (path[0] >= 'A' && path[0] <= 'Z')) && path[1] == ':'
}

func resolveStrixCandidate(ctx context.Context, candidateDir, explicitTip string, mine []string, committedOnly bool) (resolvedStrixCandidate, error) {
	root, err := gitRevParseFn(ctx, candidateDir, "--show-toplevel")
	if err != nil || strings.TrimSpace(root) == "" {
		return resolvedStrixCandidate{}, fmt.Errorf("resolve repository root for %q: %w", candidateDir, err)
	}
	root, err = filepath.Abs(strings.TrimSpace(root))
	if err != nil {
		return resolvedStrixCandidate{}, fmt.Errorf("canonicalize repository root: %w", err)
	}
	root = filepath.Clean(root)

	tip, err := gitRevParseFn(ctx, root, "--verify", "HEAD^{commit}")
	if err != nil || !IsValidFullGitTip(tip) {
		return resolvedStrixCandidate{}, fmt.Errorf("resolve full HEAD commit: got %q: %w", strings.TrimSpace(tip), err)
	}
	tip = strings.TrimSpace(tip)
	if explicitTip != "" {
		if !IsValidFullGitTip(explicitTip) {
			return resolvedStrixCandidate{}, fmt.Errorf("invalid or abbreviated --git-tip %q: must be the full 40-hex HEAD commit", explicitTip)
		}
		if !strings.EqualFold(strings.TrimSpace(explicitTip), tip) {
			return resolvedStrixCandidate{}, fmt.Errorf("--git-tip %s does not match resolved HEAD %s", explicitTip, tip)
		}
	}

	if len(mine) == 0 {
		if !committedOnly {
			return resolvedStrixCandidate{}, fmt.Errorf("candidate ownership is required: provide repeatable --mine PATH or --committed-only")
		}
		status, statusErr := gitStatusFn(ctx, root)
		if statusErr != nil {
			return resolvedStrixCandidate{}, fmt.Errorf("verify clean worktree for --committed-only: %w", statusErr)
		}
		if strings.TrimSpace(status) != "" {
			return resolvedStrixCandidate{}, fmt.Errorf("--committed-only requires a clean worktree")
		}
	} else if committedOnly {
		return resolvedStrixCandidate{}, fmt.Errorf("--committed-only cannot be combined with --mine")
	}
	if err := validateMinePaths(mine); err != nil {
		return resolvedStrixCandidate{}, err
	}

	archive, err := buildStrixCandidateArchiveFn(ctx, root, tip, append([]string(nil), mine...))
	if err != nil {
		return resolvedStrixCandidate{}, fmt.Errorf("build exact candidate archive: %w", err)
	}
	if len(archive.Bytes) == 0 {
		return resolvedStrixCandidate{}, fmt.Errorf("canonical candidate builder returned an empty archive")
	}
	if !isCanonicalSHA256(archive.SourceArchiveSHA256) {
		return resolvedStrixCandidate{}, fmt.Errorf("canonical candidate builder returned invalid source archive digest %q", archive.SourceArchiveSHA256)
	}
	h := sha256.Sum256(archive.Bytes)
	computed := "sha256:" + hex.EncodeToString(h[:])
	if archive.SourceArchiveSHA256 != computed {
		return resolvedStrixCandidate{}, fmt.Errorf("candidate archive/digest disagreement: computed %s, got %s", computed, archive.SourceArchiveSHA256)
	}
	return resolvedStrixCandidate{tip: tip, archive: archive}, nil
}

func isCanonicalSHA256(digest string) bool {
	if len(digest) != len("sha256:")+64 || !strings.HasPrefix(digest, "sha256:") {
		return false
	}
	for _, c := range digest[len("sha256:"):] {
		if !((c >= '0' && c <= '9') || (c >= 'a' && c <= 'f')) {
			return false
		}
	}
	return true
}

func validateCurrentStrixReceipt(receipt *amdgpu.StrixValidationReceipt, tip, sourceArchiveSHA256 string) error {
	if receipt == nil {
		return fmt.Errorf("receipt is nil")
	}
	if err := receipt.Validate(); err != nil {
		return fmt.Errorf("receipt invariant validation failed: %w", err)
	}
	if !receipt.CreditEligible() {
		return fmt.Errorf("receipt is integrity-readable but not eligible for current v2 physical Strix credit")
	}
	if !strings.EqualFold(strings.TrimSpace(receipt.Provenance.GitTip), strings.TrimSpace(tip)) {
		return fmt.Errorf("receipt GitTip %q does not match candidate %q", receipt.Provenance.GitTip, tip)
	}
	if receipt.Provenance.GitRef != sourceArchiveSHA256 {
		return fmt.Errorf("receipt GitRef %q does not match candidate source archive digest %q", receipt.Provenance.GitRef, sourceArchiveSHA256)
	}
	if !strings.EqualFold(strings.TrimSpace(receipt.Provenance.SourceArchiveSHA256), strings.TrimSpace(sourceArchiveSHA256)) {
		return fmt.Errorf("receipt source archive digest %q does not match candidate %q", receipt.Provenance.SourceArchiveSHA256, sourceArchiveSHA256)
	}
	return nil
}

func emitFailReceipt(w io.Writer, host, gitTip, gitRef string, argv []string, err error) {
	receipt := amdgpu.NewStrixValidationReceipt(amdgpu.StrixTarget{
		Mode: "ssh", Host: host, Reachable: false, TargetISA: "gfx1151", ComputeUnits: 40,
		DiscoveredAt: time.Now().UTC().Format(time.RFC3339),
	}, gitRef, gitTip, "fak-dev amd-strix-validate "+strings.Join(argv, " "))
	receipt.Verdict = "FAIL"
	receipt.Verified = false
	receipt.Failures = append(receipt.Failures, err.Error())
	receipt.Digest, _ = receipt.ComputeDigest()
	data, _ := json.MarshalIndent(receipt, "", "  ")
	fmt.Fprintln(w, string(data))
}

// RunAMDStrixValidate executes physical validation on the Strix Halo appliance.
func RunAMDStrixValidate(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("amd-strix-validate", flag.ContinueOnError)
	fs.SetOutput(stderr)
	host := fs.String("host", "", "target Strix Halo host")
	subkernels := fs.String("subkernels", "all", "sub-kernels to test")
	ablate := fs.String("ablate", "all", "ablation arms to run")
	asJSON := fs.Bool("json", false, "emit receipt as JSON")
	timeoutSec := fs.Int("timeout", 45, "total timeout in seconds")
	admissionTimeoutSec := fs.Int("admission-timeout", 10, "hardware admission timeout in seconds")
	gitTip := fs.String("git-tip", "", "expected full candidate HEAD")
	var minePaths stringSliceFlag
	fs.Var(&minePaths, "mine", "repeatable owned path relative to repository root")
	candidateDir := fs.String("candidate-dir", ".", "path within the candidate checkout")
	committedOnly := fs.Bool("committed-only", false, "use committed HEAD only (requires clean worktree)")
	if err := fs.Parse(argv); err != nil {
		return 2
	}
	if fs.NArg() > 0 {
		err := fmt.Errorf("positional arguments rejected: %v (use repeatable --mine PATH)", fs.Args())
		fmt.Fprintf(stderr, "amd-strix-validate: %v\n", err)
		if *asJSON {
			emitFailReceipt(stdout, *host, *gitTip, "", argv, err)
		}
		return 1
	}
	if *timeoutSec <= 0 {
		fmt.Fprintf(stderr, "amd-strix-validate: invalid total timeout %d: must be > 0\n", *timeoutSec)
		return 1
	}
	if *admissionTimeoutSec <= 0 || *admissionTimeoutSec > 60 || *admissionTimeoutSec >= *timeoutSec {
		fmt.Fprintf(stderr, "amd-strix-validate: invalid admission timeout %d: must be > 0, <= 60, and < total timeout %d\n", *admissionTimeoutSec, *timeoutSec)
		return 1
	}

	ctx, cancel := context.WithTimeout(context.Background(), time.Duration(*timeoutSec)*time.Second)
	defer cancel()
	candidate, err := resolveStrixCandidate(ctx, *candidateDir, *gitTip, minePaths, *committedOnly)
	if err != nil {
		fmt.Fprintf(stderr, "amd-strix-validate: candidate archive validation failed: %v\n", err)
		if *asJSON {
			emitFailReceipt(stdout, *host, *gitTip, "", argv, fmt.Errorf("candidate archive error: %w", err))
		}
		return 1
	}

	runSK := *subkernels != "none" && *subkernels != ""
	runAB := *ablate != "none" && *ablate != ""
	var skList, abList []string
	if runSK && *subkernels != "all" {
		skList = strings.Split(*subkernels, ",")
	}
	if runAB && *ablate != "all" {
		abList = strings.Split(*ablate, ",")
	}
	opts := amdgpu.StrixValidationOpts{
		Host: *host, RunSubkernels: runSK, Subkernels: skList, RunAblations: runAB, Ablations: abList,
		GitRef: candidate.archive.SourceArchiveSHA256, GitTip: candidate.tip,
		Command: "fak-dev amd-strix-validate " + strings.Join(argv, " "),
		Timeout: time.Duration(*timeoutSec) * time.Second, RequireSourceBinding: true,
		CandidateArchive: candidate.archive.Bytes, SourceArchiveSHA256: candidate.archive.SourceArchiveSHA256,
		AdmissionTimeout: time.Duration(*admissionTimeoutSec) * time.Second,
	}
	if !*asJSON {
		fmt.Fprintf(stderr, "==> Validating exact candidate on AMD Strix Halo target %s...\n", *host)
	}
	receipt, runErr := runStrixValidationFn(ctx, opts)
	if receipt == nil {
		fmt.Fprintf(stderr, "amd-strix-validate: validation failed: %v\n", runErr)
		if *asJSON {
			emitFailReceipt(stdout, *host, candidate.tip, candidate.archive.SourceArchiveSHA256, argv, fmt.Errorf("validation failed: %v", runErr))
		}
		return 1
	}
	receiptErr := validateCurrentStrixReceipt(receipt, candidate.tip, candidate.archive.SourceArchiveSHA256)
	if *asJSON {
		data, _ := json.MarshalIndent(receipt, "", "  ")
		fmt.Fprintln(stdout, string(data))
		if runErr != nil {
			fmt.Fprintf(stderr, "amd-strix-validate: validation failed: %v\n", runErr)
			return 1
		}
		if receiptErr != nil {
			fmt.Fprintf(stderr, "amd-strix-validate: receipt validation failed: %v\n", receiptErr)
			return 1
		}
		return 0
	}
	renderAMDStrixReceipt(stdout, receipt, receiptErr == nil)
	if runErr != nil {
		fmt.Fprintf(stderr, "amd-strix-validate: validation failed: %v\n", runErr)
		return 1
	}
	if receiptErr != nil {
		fmt.Fprintf(stderr, "amd-strix-validate: receipt validation failed: %v\n", receiptErr)
		return 1
	}
	return 0
}

func renderAMDStrixReceipt(w io.Writer, receipt *amdgpu.StrixValidationReceipt, credit bool) {
	fmt.Fprintln(w, "\n================================================================================")
	fmt.Fprintln(w, "AMD Strix Halo Physical Validation Receipt")
	fmt.Fprintln(w, "================================================================================")
	verdict := receipt.Verdict
	if receipt.Verdict == "PASS" && !credit {
		verdict = "PASS (historical/non-credit)"
	}
	fmt.Fprintf(w, "Verdict:     %s\n", verdict)
	fmt.Fprintf(w, "Target:      %s (%s)\n", receipt.Target.Host, receipt.Target.Mode)
	fmt.Fprintf(w, "CPU Model:   %s\n", receipt.Target.CPUModel)
	fmt.Fprintf(w, "GPU Model:   %s (%s, %d CUs)\n", receipt.Target.GPUName, receipt.Target.TargetISA, receipt.Target.ComputeUnits)
	fmt.Fprintf(w, "Memory:      %.1f GiB UMA Aperture (Total: %.1f GiB)\n", float64(receipt.Target.UMABufferBytes)/(1024*1024*1024), float64(receipt.Target.TotalRAMBytes)/(1024*1024*1024))
	fmt.Fprintf(w, "DPM Level:   %s (Watchdog: %d)\n", receipt.Target.DPMLevel, receipt.Target.LockupTimeout)
	fmt.Fprintf(w, "Digest:      %s\n", receipt.Digest)
	if len(receipt.Failures) > 0 {
		fmt.Fprintf(w, "--------------------------------------------------------------------------------\nFailures (%d):\n", len(receipt.Failures))
		for _, failure := range receipt.Failures {
			fmt.Fprintf(w, "  * %s\n", failure)
		}
	}
	fmt.Fprintf(w, "--------------------------------------------------------------------------------\nSub-Kernels Tested (%d):\n", len(receipt.Subkernels))
	for _, sk := range receipt.Subkernels {
		parity := "PASSED"
		if !sk.Parity.Passed {
			parity = "FAIL"
		}
		fmt.Fprintf(w, "  * %-25s [%s] %5d µs  (parity: %s, cosine: %.6f)\n",
			sk.Name, sk.Status, sk.DurationUS, parity, sk.Parity.LogitCosineSimilarity)
	}
	fmt.Fprintf(w, "--------------------------------------------------------------------------------\nAblation Arms Evaluated (%d):\n", len(receipt.Ablations))
	for _, ab := range receipt.Ablations {
		fmt.Fprintf(w, "  * %-28s [%s] speedup: %6.1fx (baseline: %d µs, candidate: %d µs)\n", ab.Feature, ab.Verdict, ab.Speedup, ab.BaselineArm.LatencyUS, ab.CandidateArm.LatencyUS)
	}
	fmt.Fprintln(w, "================================================================================")
}

// RunAMDStrixProbe inspects and reports live hardware facts from the Strix Halo appliance.
func RunAMDStrixProbe(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("amd-strix-probe", flag.ContinueOnError)
	fs.SetOutput(stderr)
	host := fs.String("host", "", "target Strix Halo host")
	asJSON := fs.Bool("json", false, "emit facts as JSON")
	if err := fs.Parse(argv); err != nil {
		return 2
	}
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	target, err := amdgpu.DiscoverStrixTarget(ctx, *host)
	if err != nil || target == nil || !target.Reachable {
		fmt.Fprintf(stderr, "amd-strix-probe: cannot reach Strix Halo appliance: %v\n", err)
		return 1
	}
	if *asJSON {
		data, _ := json.MarshalIndent(target, "", "  ")
		fmt.Fprintln(stdout, string(data))
		return 0
	}
	fmt.Fprintf(stdout, "Strix Halo Appliance Status:\n")
	fmt.Fprintf(stdout, "  Host:         %s (%s, latency: %.1f ms)\n", target.Host, target.Mode, target.LatencyMS)
	fmt.Fprintf(stdout, "  CPU:          %s\n", target.CPUModel)
	fmt.Fprintf(stdout, "  GPU:          %s (%s, %d CUs)\n", target.GPUName, target.TargetISA, target.ComputeUnits)
	fmt.Fprintf(stdout, "  RAM:          %.1f GiB UMA (Total: %.1f GiB)\n", float64(target.UMABufferBytes)/(1024*1024*1024), float64(target.TotalRAMBytes)/(1024*1024*1024))
	fmt.Fprintf(stdout, "  DPM:          %s\n", target.DPMLevel)
	fmt.Fprintf(stdout, "  Lockup TO:    %d\n", target.LockupTimeout)
	fmt.Fprintf(stdout, "  Vulkan ICD:   %s\n", target.VulkanICD)
	return 0
}
