package devcmd

import (
	"archive/tar"
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"os"
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
	osLstatFn                    = os.Lstat
	osReadFileFn                 = os.ReadFile
)

type stringSliceFlag []string

func (s *stringSliceFlag) String() string {
	return strings.Join(*s, ",")
}

func (s *stringSliceFlag) Set(val string) error {
	*s = append(*s, val)
	return nil
}

func defaultGitRevParse(ctx context.Context, dir string, args ...string) (string, error) {
	if dir == "" {
		dir = "."
	}
	cmdArgs := append([]string{"-C", dir, "rev-parse"}, args...)
	cmd := exec.CommandContext(ctx, "git", cmdArgs...)
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

// StrixCandidateArchive represents an immutable source bundle binding a base commit plus overlay.
type StrixCandidateArchive struct {
	BaseCommit            string            `json:"base_commit"`
	ArchiveBytes          []byte            `json:"archive_bytes,omitempty"`
	ArchiveSHA256         string            `json:"archive_sha256"`
	OverlayManifestSHA256 string            `json:"overlay_manifest_sha256"`
	OverlayFiles          map[string][]byte `json:"overlay_files,omitempty"`
}

// IsValidFullGitTip reports whether s is a valid 40-hex (or 64-hex) Git commit hash.
func IsValidFullGitTip(s string) bool {
	s = strings.TrimSpace(s)
	if len(s) != 40 && len(s) != 64 {
		return false
	}
	for _, c := range s {
		if !((c >= '0' && c <= '9') || (c >= 'a' && c <= 'f') || (c >= 'A' && c <= 'F')) {
			return false
		}
	}
	return true
}

// BuildStrixCandidateArchiveFromPaths builds a candidate archive reading overlay files from disk under rootDir by delegating to the canonical amdgpu helper.
func BuildStrixCandidateArchiveFromPaths(baseCommit, rootDir string, overlayPaths []string) (*StrixCandidateArchive, error) {
	cleanTip := strings.TrimSpace(baseCommit)
	if !IsValidFullGitTip(cleanTip) {
		return nil, fmt.Errorf("invalid or abbreviated Git tip %q: must be full 40-hex commit hash", baseCommit)
	}
	if buildStrixCandidateArchiveFn == nil {
		return nil, fmt.Errorf("candidate archive builder unavailable")
	}
	built, err := buildStrixCandidateArchiveFn(context.Background(), rootDir, cleanTip, overlayPaths)
	if err != nil {
		return nil, err
	}
	digest := strings.TrimPrefix(built.SourceArchiveSHA256, "sha256:")
	return &StrixCandidateArchive{
		BaseCommit:            cleanTip,
		ArchiveBytes:          built.Bytes,
		ArchiveSHA256:         digest,
		OverlayManifestSHA256: digest,
	}, nil
}

// LoadOrBuildCandidateArchive loads an existing archive or builds one from candidate state.
func LoadOrBuildCandidateArchive(gitTip, archivePath, archiveDigest string, minePaths []string, committedOnly bool, overlayDigest, candidateDir string) (*StrixCandidateArchive, error) {
	cleanTip := strings.TrimSpace(gitTip)

	if archivePath != "" {
		if len(minePaths) > 0 || committedOnly {
			return nil, fmt.Errorf("conflicting options: cannot specify both --archive and --mine/--committed-only")
		}
		data, err := osReadFileFn(archivePath)
		if err != nil {
			return nil, fmt.Errorf("missing candidate archive: cannot read %q: %w", archivePath, err)
		}
		if len(data) == 0 {
			return nil, fmt.Errorf("missing candidate archive: archive file %q is empty", archivePath)
		}

		hash := sha256.Sum256(data)
		calcSHA256 := hex.EncodeToString(hash[:])

		tr := tar.NewReader(bytes.NewReader(data))
		var tarBaseCommit string
		for {
			hdr, err := tr.Next()
			if err != nil {
				break
			}
			if hdr.Name == ".strix-base-commit" {
				b, err := io.ReadAll(tr)
				if err == nil {
					tarBaseCommit = strings.TrimSpace(string(b))
				}
				break
			}
		}

		var jsonMeta struct {
			BaseCommit string `json:"base_commit"`
		}
		if tarBaseCommit != "" {
			if !IsValidFullGitTip(tarBaseCommit) {
				return nil, fmt.Errorf("invalid or abbreviated Git tip %q in candidate archive: must be full 40-hex commit hash", tarBaseCommit)
			}
			if cleanTip != "" && !strings.EqualFold(cleanTip, tarBaseCommit) {
				return nil, fmt.Errorf("archive base commit %s does not match GitTip %s", tarBaseCommit, cleanTip)
			}
			if cleanTip == "" {
				cleanTip = tarBaseCommit
			}
		} else if json.Unmarshal(data, &jsonMeta) == nil && jsonMeta.BaseCommit != "" {
			if !IsValidFullGitTip(jsonMeta.BaseCommit) {
				return nil, fmt.Errorf("invalid or abbreviated Git tip %q in candidate archive: must be full 40-hex commit hash", jsonMeta.BaseCommit)
			}
			if cleanTip != "" && !strings.EqualFold(cleanTip, jsonMeta.BaseCommit) {
				return nil, fmt.Errorf("archive base commit %s does not match GitTip %s", jsonMeta.BaseCommit, cleanTip)
			}
			if cleanTip == "" {
				cleanTip = jsonMeta.BaseCommit
			}
		}

		if cleanTip == "" {
			canonicalRoot, _ := gitRevParseFn(context.Background(), candidateDir, "--show-toplevel")
			if canonicalRoot != "" {
				cleanTip, _ = gitRevParseFn(context.Background(), canonicalRoot, "--verify", "HEAD^{commit}")
			}
		}
		if cleanTip == "" {
			return nil, fmt.Errorf("candidate archive provided without explicit GitTip and archive carries no base_commit metadata")
		}
		if !IsValidFullGitTip(cleanTip) {
			return nil, fmt.Errorf("invalid or abbreviated Git tip %q: must be full 40-hex commit hash", cleanTip)
		}

		if archiveDigest != "" {
			expected := strings.ToLower(strings.TrimPrefix(strings.TrimSpace(archiveDigest), "sha256:"))
			if strings.ToLower(calcSHA256) != expected {
				return nil, fmt.Errorf("archive/digest disagreement: computed sha256:%s != expected %s", calcSHA256, archiveDigest)
			}
		}

		return &StrixCandidateArchive{
			BaseCommit:            cleanTip,
			ArchiveBytes:          data,
			ArchiveSHA256:         calcSHA256,
			OverlayManifestSHA256: calcSHA256,
		}, nil
	}

	// No archivePath specified -> Build from local candidate checkout
	canonicalRoot, err := gitRevParseFn(context.Background(), candidateDir, "--show-toplevel")
	if err != nil || strings.TrimSpace(canonicalRoot) == "" {
		return nil, fmt.Errorf("missing or unresolvable Git tip or root directory for %q: %w", candidateDir, err)
	}
	canonicalRoot = strings.TrimSpace(canonicalRoot)

	if cleanTip == "" {
		resolved, err := gitRevParseFn(context.Background(), canonicalRoot, "--verify", "HEAD^{commit}")
		if err != nil || !IsValidFullGitTip(resolved) {
			return nil, fmt.Errorf("missing or unresolvable Git tip: candidate checkout must provide full 40-hex commit (got %q, err: %v)", resolved, err)
		}
		cleanTip = resolved
	} else if !IsValidFullGitTip(cleanTip) {
		return nil, fmt.Errorf("invalid or abbreviated Git tip %q: must be full 40-hex commit hash", cleanTip)
	}

	if len(minePaths) == 0 {
		if !committedOnly {
			return nil, fmt.Errorf("no candidate archive or --mine overlay specified: must supply explicit --mine PATH or enable --committed-only mode")
		}
		status, err := gitStatusFn(context.Background(), canonicalRoot)
		if err != nil {
			return nil, fmt.Errorf("failed to verify clean worktree for committed-only mode: %w", err)
		}
		if strings.TrimSpace(status) != "" {
			return nil, fmt.Errorf("worktree is dirty: committed-only mode requires clean worktree")
		}
	} else {
		if committedOnly {
			return nil, fmt.Errorf("conflicting options: cannot specify both --mine overlays and --committed-only mode")
		}
	}

	fileBaseDir := canonicalRoot
	if candidateDir != "" && candidateDir != "." {
		fileBaseDir = candidateDir
	}

	seenPaths := make(map[string]struct{}, len(minePaths))
	for _, rawPath := range minePaths {
		rawPath = strings.TrimSpace(rawPath)
		if rawPath == "" {
			return nil, fmt.Errorf("empty --mine overlay path rejected")
		}
		if filepath.IsAbs(rawPath) || strings.HasPrefix(rawPath, "/") || strings.HasPrefix(rawPath, "\\") || (len(rawPath) > 1 && rawPath[1] == ':') {
			return nil, fmt.Errorf("absolute overlay path rejected: %q must be relative to repository root", rawPath)
		}
		norm := filepath.ToSlash(filepath.Clean(filepath.FromSlash(rawPath)))
		if strings.HasPrefix(norm, "../") || strings.Contains(norm, "/../") || norm == ".." || norm == "." {
			return nil, fmt.Errorf("overlay path traversal rejected: %q escapes root", rawPath)
		}
		if _, exists := seenPaths[norm]; exists {
			return nil, fmt.Errorf("duplicate overlay path rejected: %q specified multiple times", rawPath)
		}
		seenPaths[norm] = struct{}{}

		cleanPath := filepath.Clean(filepath.FromSlash(rawPath))
		fullPath := filepath.Join(fileBaseDir, cleanPath)
		fi, err := osLstatFn(fullPath)
		if err != nil {
			return nil, fmt.Errorf("overlay file unreadable or missing: %w", err)
		}
		if fi.Mode()&os.ModeSymlink != 0 {
			return nil, fmt.Errorf("symlink overlay rejected: %q is a symlink", rawPath)
		}
		if fi.IsDir() {
			return nil, fmt.Errorf("directory overlay rejected: %q is a directory, must be a regular file", rawPath)
		}
		if !fi.Mode().IsRegular() {
			return nil, fmt.Errorf("invalid overlay file mode: %q must be a regular file (mode: %s)", rawPath, fi.Mode())
		}
		realPath, err := filepath.EvalSymlinks(fullPath)
		if err == nil {
			rel, relErr := filepath.Rel(fileBaseDir, realPath)
			if relErr != nil || strings.HasPrefix(rel, "..") || filepath.IsAbs(rel) {
				return nil, fmt.Errorf("symlink escape rejected: %q resolves outside root (%s)", rawPath, realPath)
			}
		}
	}

	candArchive, err := BuildStrixCandidateArchiveFromPaths(cleanTip, fileBaseDir, minePaths)
	if err != nil {
		return nil, err
	}

	if overlayDigest != "" {
		expected := strings.ToLower(strings.TrimPrefix(strings.TrimSpace(overlayDigest), "sha256:"))
		if strings.ToLower(candArchive.ArchiveSHA256) != expected {
			return nil, fmt.Errorf("overlay manifest digest disagreement: computed sha256:%s != expected %s", candArchive.ArchiveSHA256, overlayDigest)
		}
	}

	if archiveDigest != "" {
		expected := strings.ToLower(strings.TrimPrefix(strings.TrimSpace(archiveDigest), "sha256:"))
		if strings.ToLower(candArchive.ArchiveSHA256) != expected {
			return nil, fmt.Errorf("archive/digest disagreement: computed sha256:%s != expected %s", candArchive.ArchiveSHA256, archiveDigest)
		}
	}

	if len(candArchive.ArchiveBytes) == 0 || strings.TrimSpace(candArchive.ArchiveSHA256) == "" {
		return nil, fmt.Errorf("candidate archive invalid: empty archive bytes or digest")
	}

	return candArchive, nil
}

type candidateArchiveContextKey struct{}
type admissionTimeoutContextKey struct{}

// WithCandidateArchive returns a context carrying the StrixCandidateArchive.
func WithCandidateArchive(ctx context.Context, archive *StrixCandidateArchive) context.Context {
	return context.WithValue(ctx, candidateArchiveContextKey{}, archive)
}

// CandidateArchiveFromContext retrieves the StrixCandidateArchive from context if present.
func CandidateArchiveFromContext(ctx context.Context) (*StrixCandidateArchive, bool) {
	a, ok := ctx.Value(candidateArchiveContextKey{}).(*StrixCandidateArchive)
	return a, ok
}

// WithAdmissionTimeout returns a context carrying the admission timeout duration.
func WithAdmissionTimeout(ctx context.Context, timeout time.Duration) context.Context {
	return context.WithValue(ctx, admissionTimeoutContextKey{}, timeout)
}

// AdmissionTimeoutFromContext retrieves the admission timeout duration from context if present.
func AdmissionTimeoutFromContext(ctx context.Context) (time.Duration, bool) {
	d, ok := ctx.Value(admissionTimeoutContextKey{}).(time.Duration)
	return d, ok
}

func isCurrentValidPass(receipt *amdgpu.StrixValidationReceipt, expectedTip, expectedRef string) (bool, string) {
	if receipt == nil {
		return false, "receipt is nil"
	}
	if strings.TrimSpace(expectedTip) == "" {
		return false, "missing required expected GitTip"
	}
	if strings.TrimSpace(expectedRef) == "" {
		return false, "missing required expected archive digest"
	}
	if receipt.Verdict != "PASS" {
		return false, fmt.Sprintf("verdict is %s (want PASS)", receipt.Verdict)
	}
	if !receipt.Verified {
		return false, "receipt is not verified (verified=false)"
	}
	if len(receipt.Failures) > 0 {
		return false, fmt.Sprintf("receipt contains failures: %s", strings.Join(receipt.Failures, "; "))
	}
	// Historical v1 receipt rejection: must carry v2 schema
	if receipt.Schema != amdgpu.StrixValidationSchemaV2 {
		return false, fmt.Sprintf("historical or non-credit schema %q: current PASS requires v2 (%s)", receipt.Schema, amdgpu.StrixValidationSchemaV2)
	}
	// Historical or unbound receipt rejection: must carry non-empty source binding tokens
	cleanTip := strings.TrimSpace(receipt.Provenance.GitTip)
	if cleanTip == "" {
		return false, "historical or unbound receipt: GitTip is empty"
	}
	if !IsValidFullGitTip(cleanTip) {
		return false, fmt.Sprintf("invalid or abbreviated GitTip in receipt: %q", receipt.Provenance.GitTip)
	}
	cleanRef := strings.TrimSpace(receipt.Provenance.GitRef)
	if cleanRef == "" {
		return false, "historical or unbound receipt: GitRef (archive digest) is empty"
	}
	if !strings.HasPrefix(strings.ToLower(cleanRef), "sha256:") {
		return false, fmt.Sprintf("invalid GitRef in receipt (must have sha256: prefix): %q", receipt.Provenance.GitRef)
	}
	if !strings.EqualFold(cleanTip, strings.TrimSpace(expectedTip)) {
		return false, fmt.Sprintf("receipt GitTip %s does not match expected %s", receipt.Provenance.GitTip, expectedTip)
	}
	cleanExpected := strings.TrimPrefix(strings.ToLower(strings.TrimSpace(expectedRef)), "sha256:")
	cleanRefDigest := strings.TrimPrefix(strings.ToLower(cleanRef), "sha256:")
	if cleanRefDigest != cleanExpected {
		return false, fmt.Sprintf("receipt GitRef %s does not match expected archive digest %s", receipt.Provenance.GitRef, expectedRef)
	}
	cleanSourceArchive := strings.TrimPrefix(strings.ToLower(strings.TrimSpace(receipt.Provenance.SourceArchiveSHA256)), "sha256:")
	if cleanSourceArchive == "" {
		return false, "historical or unbound receipt: SourceArchiveSHA256 is empty"
	}
	if cleanSourceArchive != cleanExpected {
		return false, fmt.Sprintf("receipt SourceArchiveSHA256 %s does not match expected archive digest %s", receipt.Provenance.SourceArchiveSHA256, expectedRef)
	}
	// Execution completeness: if subkernels were selected/executed, none can be SKIPPED or FAIL
	if len(receipt.Subkernels) == 0 && len(receipt.Ablations) == 0 {
		return false, "missing execution evidence: receipt contains zero subkernels and zero ablations"
	}
	if err := receipt.Validate(); err != nil {
		return false, fmt.Sprintf("receipt invariant validation failed: %v", err)
	}
	if receipt.SelectedCount > 0 && receipt.ExecutedCount == 0 {
		return false, "subkernels were selected but zero were executed"
	}
	if receipt.SelectedSubkernels > 0 && receipt.ExecutedSubkernels == 0 {
		return false, "subkernels were selected but zero were executed"
	}
	if len(receipt.Subkernels) > 0 && !receipt.CreditEligible() {
		return false, "receipt is not credit eligible for physical Strix Halo parity"
	}
	for _, sk := range receipt.Subkernels {
		if sk.Status != "PASS" {
			return false, fmt.Sprintf("subkernel %q status is %s (want PASS)", sk.Name, sk.Status)
		}
		if !sk.Parity.Passed {
			return false, fmt.Sprintf("subkernel %q parity not passed", sk.Name)
		}
		if sk.DurationUS <= 0 {
			return false, fmt.Sprintf("subkernel %q has non-positive duration (%d µs)", sk.Name, sk.DurationUS)
		}
	}
	for _, ab := range receipt.Ablations {
		if ab.Verdict == "REGRESSION" {
			return false, fmt.Sprintf("ablation %q suffered regression (speedup=%.2fx)", ab.Feature, ab.Speedup)
		}
		if ab.Verdict != "VERIFIED_LIFT" && ab.Verdict != "PARITY_MATCH" {
			return false, fmt.Sprintf("ablation %q verdict is %s (want VERIFIED_LIFT or PARITY_MATCH)", ab.Feature, ab.Verdict)
		}
		if ab.BaselineArm.LatencyUS <= 0 || ab.CandidateArm.LatencyUS <= 0 {
			return false, fmt.Sprintf("ablation %q has non-positive latency (baseline: %d µs, candidate: %d µs)", ab.Feature, ab.BaselineArm.LatencyUS, ab.CandidateArm.LatencyUS)
		}
	}
	return true, ""
}

func emitFailReceipt(w io.Writer, host, gitTip, gitRef string, argv []string, err error) {
	receipt := amdgpu.NewStrixValidationReceipt(
		amdgpu.StrixTarget{
			Mode:         "ssh",
			Host:         host,
			Reachable:    false,
			TargetISA:    "gfx1151",
			ComputeUnits: 40,
			DiscoveredAt: time.Now().UTC().Format(time.RFC3339),
		},
		gitRef,
		gitTip,
		"fak-dev amd-strix-validate "+strings.Join(argv, " "),
	)
	receipt.Verdict = "FAIL"
	receipt.Verified = false
	receipt.Failures = append(receipt.Failures, err.Error())
	digest, _ := receipt.ComputeDigest()
	receipt.Digest = digest
	data, _ := json.MarshalIndent(receipt, "", "  ")
	fmt.Fprintln(w, string(data))
}

// RunAMDStrixValidate executes physical sub-kernel validation and ablation sweeps on the Strix Halo appliance.
func RunAMDStrixValidate(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("amd-strix-validate", flag.ContinueOnError)
	fs.SetOutput(stderr)

	host := fs.String("host", "", "target Strix Halo host (default: strix1 or FAK_STRIX_HOST)")
	subkernels := fs.String("subkernels", "all", "sub-kernels to test (comma-separated, 'all', or 'none')")
	ablate := fs.String("ablate", "all", "ablation arms to run (comma-separated, 'all', or 'none')")
	asJSON := fs.Bool("json", false, "emit receipt as JSON")
	timeoutSec := fs.Int("timeout", 45, "timeout in seconds")
	admissionTimeoutSec := fs.Int("admission-timeout", 10, "hardware admission timeout in seconds (must be > 0 and < timeout)")
	gitTip := fs.String("git-tip", "", "candidate base commit (full 40-hex hash; default resolved from git rev-parse --verify HEAD^{commit})")
	archivePath := fs.String("archive", "", "path to pre-built candidate archive file or JSON manifest")
	archiveDigest := fs.String("archive-digest", "", "expected SHA256 digest of candidate archive (sha256:...)")
	var minePaths stringSliceFlag
	fs.Var(&minePaths, "mine", "repeatable explicit owned overlay path relative to repository root")
	overlay := fs.String("overlay", "", "comma-separated owned overlay file paths (alias for --mine)")
	overlayDigest := fs.String("overlay-digest", "", "expected SHA256 digest of overlay manifest (sha256:...)")
	candidateDir := fs.String("candidate-dir", ".", "path to local candidate checkout directory")
	committedOnly := fs.Bool("committed-only", false, "committed-only candidate mode (requires clean worktree)")

	if err := fs.Parse(argv); err != nil {
		return 2
	}

	if fs.NArg() > 0 {
		err := fmt.Errorf("positional overlays or arguments rejected: %v (use repeatable --mine PATH instead)", fs.Args())
		fmt.Fprintf(stderr, "amd-strix-validate: %v\n", err)
		if *asJSON {
			emitFailReceipt(stdout, *host, *gitTip, "", argv, err)
		}
		return 1
	}

	if *timeoutSec <= 0 {
		err := fmt.Errorf("invalid command timeout (%d): must be > 0", *timeoutSec)
		fmt.Fprintf(stderr, "amd-strix-validate: %v\n", err)
		if *asJSON {
			emitFailReceipt(stdout, *host, *gitTip, "", argv, err)
		}
		return 1
	}

	if *admissionTimeoutSec <= 0 || *admissionTimeoutSec >= *timeoutSec {
		err := fmt.Errorf("invalid admission timeout (%d): must be > 0 and < total command timeout (%d)", *admissionTimeoutSec, *timeoutSec)
		fmt.Fprintf(stderr, "amd-strix-validate: %v\n", err)
		if *asJSON {
			emitFailReceipt(stdout, *host, *gitTip, "", argv, err)
		}
		return 1
	}

	if *overlay != "" {
		for _, p := range strings.Split(*overlay, ",") {
			p = strings.TrimSpace(p)
			if p != "" {
				minePaths = append(minePaths, p)
			}
		}
	}

	candArchive, archiveErr := LoadOrBuildCandidateArchive(
		*gitTip,
		*archivePath,
		*archiveDigest,
		minePaths,
		*committedOnly,
		*overlayDigest,
		*candidateDir,
	)
	if archiveErr != nil {
		fmt.Fprintf(stderr, "amd-strix-validate: candidate archive validation failed: %v\n", archiveErr)
		if *asJSON {
			emitFailReceipt(stdout, *host, *gitTip, "", argv, fmt.Errorf("candidate archive error: %w", archiveErr))
		}
		return 1
	}

	runSK := *subkernels != "none" && *subkernels != ""
	runAB := *ablate != "none" && *ablate != ""

	var skList []string
	if runSK && *subkernels != "all" {
		skList = strings.Split(*subkernels, ",")
	}

	var abList []string
	if runAB && *ablate != "all" {
		abList = strings.Split(*ablate, ",")
	}

	ctx, cancel := context.WithTimeout(context.Background(), time.Duration(*timeoutSec)*time.Second)
	defer cancel()

	archiveDigestStr := "sha256:" + strings.TrimPrefix(candArchive.ArchiveSHA256, "sha256:")

	opts := amdgpu.StrixValidationOpts{
		Host:                 *host,
		RunSubkernels:        runSK,
		Subkernels:           skList,
		RunAblations:         runAB,
		Ablations:            abList,
		GitRef:               archiveDigestStr,
		GitTip:               candArchive.BaseCommit,
		Command:              "fak-dev amd-strix-validate " + strings.Join(argv, " "),
		Timeout:              time.Duration(*timeoutSec) * time.Second,
		RequireSourceBinding: true,
		CandidateArchive:     candArchive.ArchiveBytes,
		SourceArchiveSHA256:  archiveDigestStr,
		AdmissionTimeout:     time.Duration(*admissionTimeoutSec) * time.Second,
	}

	ctx = amdgpu.WithSourceBinding(ctx, opts.GitTip, opts.GitRef)
	ctx = WithCandidateArchive(ctx, candArchive)
	ctx = WithAdmissionTimeout(ctx, time.Duration(*admissionTimeoutSec)*time.Second)

	if !*asJSON {
		fmt.Fprintf(stderr, "==> Probing AMD Strix Halo appliance at %s...\n", *host)
	}

	receipt, err := runStrixValidationFn(ctx, opts)
	if receipt == nil {
		fmt.Fprintf(stderr, "amd-strix-validate: validation failed: %v\n", err)
		if *asJSON {
			emitFailReceipt(stdout, *host, candArchive.BaseCommit, archiveDigestStr, argv, fmt.Errorf("validation failed: %v", err))
		}
		return 1
	}

	passOK, passReason := isCurrentValidPass(receipt, candArchive.BaseCommit, archiveDigestStr)

	if *asJSON {
		data, _ := json.MarshalIndent(receipt, "", "  ")
		fmt.Fprintln(stdout, string(data))
		if err != nil {
			fmt.Fprintf(stderr, "amd-strix-validate: validation failed: %v\n", err)
			return 1
		}
		if !passOK {
			fmt.Fprintf(stderr, "amd-strix-validate: receipt validation failed: %s\n", passReason)
			return 1
		}
		return 0
	}

	// Render human-readable summary
	fmt.Fprintf(stdout, "\n================================================================================\n")
	fmt.Fprintf(stdout, "AMD Strix Halo Physical Validation Receipt\n")
	fmt.Fprintf(stdout, "================================================================================\n")
	verdictStr := receipt.Verdict
	if receipt.Verdict == "PASS" && !passOK {
		verdictStr = "PASS (historical/non-credit)"
	}
	fmt.Fprintf(stdout, "Verdict:     %s\n", verdictStr)
	fmt.Fprintf(stdout, "Target:      %s (%s)\n", receipt.Target.Host, receipt.Target.Mode)
	fmt.Fprintf(stdout, "CPU Model:   %s\n", receipt.Target.CPUModel)
	fmt.Fprintf(stdout, "GPU Model:   %s (%s, %d CUs)\n", receipt.Target.GPUName, receipt.Target.TargetISA, receipt.Target.ComputeUnits)
	fmt.Fprintf(stdout, "Memory:      %.1f GiB UMA Aperture (Total: %.1f GiB)\n",
		float64(receipt.Target.UMABufferBytes)/(1024*1024*1024),
		float64(receipt.Target.TotalRAMBytes)/(1024*1024*1024))
	fmt.Fprintf(stdout, "DPM Level:   %s (Watchdog: %d)\n", receipt.Target.DPMLevel, receipt.Target.LockupTimeout)
	fmt.Fprintf(stdout, "Digest:      %s\n", receipt.Digest)
	if len(receipt.Failures) > 0 {
		fmt.Fprintf(stdout, "--------------------------------------------------------------------------------\n")
		fmt.Fprintf(stdout, "Failures (%d):\n", len(receipt.Failures))
		for _, f := range receipt.Failures {
			fmt.Fprintf(stdout, "  * %s\n", f)
		}
	}
	fmt.Fprintf(stdout, "--------------------------------------------------------------------------------\n")
	fmt.Fprintf(stdout, "Sub-Kernels Tested (%d):\n", len(receipt.Subkernels))
	for _, sk := range receipt.Subkernels {
		parity := "PASSED"
		if !sk.Parity.Passed {
			parity = "FAIL"
		}
		fmt.Fprintf(stdout, "  * %-25s [%s] %5d µs  (parity: %s, cosine: %.6f)\n",
			sk.Name, sk.Status, sk.DurationUS, parity, sk.Parity.LogitCosineSimilarity)
	}
	fmt.Fprintf(stdout, "--------------------------------------------------------------------------------\n")
	fmt.Fprintf(stdout, "Ablation Arms Evaluated (%d):\n", len(receipt.Ablations))
	for _, ab := range receipt.Ablations {
		fmt.Fprintf(stdout, "  * %-28s [%s] speedup: %6.1fx (baseline: %d µs, candidate: %d µs)\n",
			ab.Feature, ab.Verdict, ab.Speedup, ab.BaselineArm.LatencyUS, ab.CandidateArm.LatencyUS)
	}
	fmt.Fprintf(stdout, "================================================================================\n\n")

	if err != nil {
		fmt.Fprintf(stderr, "amd-strix-validate: validation failed: %v\n", err)
		return 1
	}

	if !passOK {
		fmt.Fprintf(stderr, "amd-strix-validate: receipt validation failed: %s\n", passReason)
		return 1
	}
	return 0
}

// RunAMDStrixProbe inspects and reports live hardware facts from the Strix Halo appliance.
func RunAMDStrixProbe(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("amd-strix-probe", flag.ContinueOnError)
	fs.SetOutput(stderr)

	host := fs.String("host", "", "target Strix Halo host (default: strix1 or FAK_STRIX_HOST)")
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
	fmt.Fprintf(stdout, "  RAM:          %.1f GiB UMA (Total: %.1f GiB)\n",
		float64(target.UMABufferBytes)/(1024*1024*1024),
		float64(target.TotalRAMBytes)/(1024*1024*1024))
	fmt.Fprintf(stdout, "  DPM:          %s\n", target.DPMLevel)
	fmt.Fprintf(stdout, "  Lockup TO:    %d\n", target.LockupTimeout)
	fmt.Fprintf(stdout, "  Vulkan ICD:   %s\n", target.VulkanICD)
	return 0
}
