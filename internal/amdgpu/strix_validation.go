package amdgpu

import (
	"archive/tar"
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"io"
	"os"
	"os/exec"
	"path"
	"path/filepath"
	"regexp"
	"sort"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/windowgate"
)

type sourceBindingKey struct{}

type SourceBinding struct {
	GitTip              string
	GitRef              string
	SourceArchiveSHA256 string
	BinarySHA256        string
	ShaderBundleSHA256  string
	BuildCommandSHA256  string
	WorkDir             string
	AdmissionWait       time.Duration
}

func WithSourceBinding(ctx context.Context, gitTip, gitRef string) context.Context {
	return context.WithValue(ctx, sourceBindingKey{}, SourceBinding{GitTip: strings.TrimSpace(gitTip), GitRef: strings.TrimSpace(gitRef)})
}

func withPreparedSourceBinding(ctx context.Context, sb SourceBinding) context.Context {
	return context.WithValue(ctx, sourceBindingKey{}, sb)
}

func SourceBindingFromContext(ctx context.Context) (SourceBinding, bool) {
	sb, ok := ctx.Value(sourceBindingKey{}).(SourceBinding)
	return sb, ok
}

type StrixCandidateArchive struct {
	Bytes               []byte
	SourceArchiveSHA256 string
}

type StrixValidationOpts struct {
	Host                 string        `json:"host,omitempty"`
	Subkernels           []string      `json:"subkernels,omitempty"`
	Ablations            []string      `json:"ablations,omitempty"`
	RunSubkernels        bool          `json:"run_subkernels"`
	RunAblations         bool          `json:"run_ablations"`
	GitRef               string        `json:"git_ref,omitempty"`
	GitTip               string        `json:"git_tip,omitempty"`
	Command              string        `json:"command,omitempty"`
	Timeout              time.Duration `json:"timeout,omitempty"`
	RequireSourceBinding bool          `json:"require_source_binding,omitempty"`
	CandidateArchive     []byte        `json:"-"`
	SourceArchiveSHA256  string        `json:"source_archive_sha256,omitempty"`
	AdmissionTimeout     time.Duration `json:"admission_timeout,omitempty"`
}

type tarEntry struct {
	name string
	mode int64
	kind byte
	link string
	body []byte
}

var safeStrixWorkspaceRE = regexp.MustCompile(`^/tmp/fak-strix-validation\.[A-Za-z0-9_-]+$`)

func validateStrixWorkspace(work string) error {
	if !safeStrixWorkspaceRE.MatchString(work) {
		return fmt.Errorf("amdgpu: unsafe target workspace %q", work)
	}
	return nil
}

func digestBytes(b []byte) string { h := sha256.Sum256(b); return "sha256:" + hex.EncodeToString(h[:]) }

// BuildStrixCandidateArchive creates a deterministic archive of baseTip plus exactly ownedPaths from root.
func BuildStrixCandidateArchive(ctx context.Context, root, baseTip string, ownedPaths []string) (StrixCandidateArchive, error) {
	if !fullGitTipRE.MatchString(strings.TrimSpace(baseTip)) {
		return StrixCandidateArchive{}, fmt.Errorf("amdgpu: full 40-hex base GitTip is required")
	}
	cmd := exec.CommandContext(ctx, "git", "-C", root, "archive", "--format=tar", baseTip)
	base, err := cmd.Output()
	if err != nil {
		return StrixCandidateArchive{}, fmt.Errorf("amdgpu: archive base candidate: %w", err)
	}
	entries := map[string]tarEntry{}
	tr := tar.NewReader(bytes.NewReader(base))
	for {
		h, er := tr.Next()
		if er == io.EOF {
			break
		}
		if er != nil {
			return StrixCandidateArchive{}, fmt.Errorf("amdgpu: read base archive: %w", er)
		}
		name := strings.TrimSuffix(filepath.ToSlash(filepath.Clean(h.Name)), "/")
		if name == "." || name == "" || h.Typeflag == tar.TypeDir {
			continue
		}
		if h.Typeflag != tar.TypeReg && h.Typeflag != tar.TypeRegA && h.Typeflag != tar.TypeSymlink && h.Typeflag != tar.TypeLink {
			continue
		}
		if err := rejectArchiveLinkEntry(h); err != nil {
			return StrixCandidateArchive{}, err
		}
		body, er := io.ReadAll(tr)
		if er != nil {
			return StrixCandidateArchive{}, er
		}
		entries[name] = tarEntry{name: name, mode: h.Mode, kind: h.Typeflag, link: h.Linkname, body: body}
	}
	for _, raw := range ownedPaths {
		rel := strings.TrimPrefix(filepath.ToSlash(filepath.Clean(raw)), "./")
		if rel == "." || rel == "" || strings.HasPrefix(rel, "../") || filepath.IsAbs(raw) {
			return StrixCandidateArchive{}, fmt.Errorf("amdgpu: unsafe candidate overlay path %q", raw)
		}
		for name := range entries {
			if name == rel || strings.HasPrefix(name, rel+"/") {
				delete(entries, name)
			}
		}
		path := filepath.Join(root, filepath.FromSlash(rel))
		info, er := os.Lstat(path)
		if os.IsNotExist(er) {
			continue
		}
		if er != nil {
			return StrixCandidateArchive{}, fmt.Errorf("amdgpu: inspect overlay %s: %w", rel, er)
		}
		add := func(full, name string, fi os.FileInfo) error {
			name = filepath.ToSlash(name)
			if fi.Mode()&os.ModeSymlink != 0 {
				return fmt.Errorf("candidate overlay symlink %q is not allowed", name)
			}
			if !fi.Mode().IsRegular() {
				return nil
			}
			body, e := os.ReadFile(full)
			if e != nil {
				return e
			}
			mode := int64(fi.Mode().Perm())
			entries[name] = tarEntry{name: name, mode: mode, kind: tar.TypeReg, body: body}
			return nil
		}
		if info.IsDir() {
			er = filepath.Walk(path, func(full string, fi os.FileInfo, walkErr error) error {
				if walkErr != nil {
					return walkErr
				}
				if fi.IsDir() {
					return nil
				}
				child, e := filepath.Rel(root, full)
				if e != nil {
					return e
				}
				return add(full, child, fi)
			})
		} else {
			er = add(path, rel, info)
		}
		if er != nil {
			return StrixCandidateArchive{}, fmt.Errorf("amdgpu: overlay %s: %w", rel, er)
		}
	}
	names := make([]string, 0, len(entries))
	for name := range entries {
		names = append(names, name)
	}
	sort.Strings(names)
	var out bytes.Buffer
	tw := tar.NewWriter(&out)
	for _, name := range names {
		e := entries[name]
		h := &tar.Header{Name: e.name, Mode: e.mode, Size: int64(len(e.body)), Typeflag: e.kind, Linkname: e.link, ModTime: time.Unix(0, 0).UTC(), Uid: 0, Gid: 0}
		if e.kind == tar.TypeSymlink {
			h.Size = 0
		}
		if err := tw.WriteHeader(h); err != nil {
			return StrixCandidateArchive{}, err
		}
		if len(e.body) > 0 {
			if _, err := tw.Write(e.body); err != nil {
				return StrixCandidateArchive{}, err
			}
		}
	}
	if err := tw.Close(); err != nil {
		return StrixCandidateArchive{}, err
	}
	return StrixCandidateArchive{Bytes: out.Bytes(), SourceArchiveSHA256: digestBytes(out.Bytes())}, nil
}

func rejectArchiveLinkEntry(h *tar.Header) error {
	if h != nil && (h.Typeflag == tar.TypeSymlink || h.Typeflag == tar.TypeLink) {
		return fmt.Errorf("amdgpu: candidate archive link %q -> %q is not allowed", h.Name, h.Linkname)
	}
	return nil
}

func validateCandidateArchiveBytes(data []byte) error {
	tr := tar.NewReader(bytes.NewReader(data))
	seen := map[string]bool{}
	count := 0
	for {
		h, err := tr.Next()
		if err == io.EOF {
			break
		}
		if err != nil {
			return fmt.Errorf("amdgpu: invalid candidate archive: %w", err)
		}
		name := strings.ReplaceAll(h.Name, "\\", "/")
		clean := path.Clean(name)
		if name == "" || strings.HasPrefix(name, "/") || clean == ".." || strings.HasPrefix(clean, "../") || clean != strings.TrimSuffix(name, "/") {
			return fmt.Errorf("amdgpu: unsafe candidate archive path %q", h.Name)
		}
		if seen[clean] {
			return fmt.Errorf("amdgpu: duplicate candidate archive path %q", clean)
		}
		seen[clean] = true
		if err := rejectArchiveLinkEntry(h); err != nil {
			return err
		}
		if h.Typeflag != tar.TypeReg && h.Typeflag != tar.TypeRegA && h.Typeflag != tar.TypeDir {
			return fmt.Errorf("amdgpu: unsupported candidate archive entry %q type %d", h.Name, h.Typeflag)
		}
		count++
	}
	if count == 0 {
		return fmt.Errorf("amdgpu: candidate archive is empty")
	}
	return nil
}

func shellQuote(s string) string { return "'" + strings.ReplaceAll(s, "'", "'\"'\"'") + "'" }

func buildStrixAdmissionCommand(target *StrixTarget, sb SourceBinding, testCmd string) string {
	leaseWait := int64(sb.AdmissionWait / time.Second)
	if leaseWait <= 0 {
		leaseWait = 30
	}
	verify := `test "sha256:$(sha256sum ` + shellQuote(sb.WorkDir+"/build/compute.test") + ` | awk '{print $1}')" = ` + shellQuote(sb.BinarySHA256) + ` && test "sha256:$(find ` + shellQuote(sb.WorkDir+"/build/spirv") + ` -type f -name '*.spv' -print0 | sort -z | xargs -0 sha256sum | sha256sum | awk '{print $1}')" = ` + shellQuote(sb.ShaderBundleSHA256)
	deviceCmd := `timeout --signal=TERM --kill-after=5s 60s bash -c ` + shellQuote(testCmd)
	semantic := verify + "; " + deviceCmd
	inner := `echo FAK_STRIX_ADMISSION_ACQUIRED=1; trap 'echo FAK_STRIX_ADMISSION_RELEASED=1' EXIT; ` + verify + ` || exit 1; echo FAK_STRIX_ARTIFACT_REHASH=1; echo FAK_STRIX_ENGINE=fak-native/vulkan; echo FAK_STRIX_DEVICE_TIMEOUT_MS=60000; echo FAK_STRIX_DEVICE=` + shellQuote(target.GPUName+"|"+target.TargetISA) + `; ` + deviceCmd
	return fmt.Sprintf(`set +e; echo FAK_STRIX_COMMAND_SHA256=%s; lease="${FAK_GPU_LEASE:-/tmp/fak-gpu.lease}"; echo FAK_STRIX_LEASE_SHA256=sha256:$(printf %%s "$lease" | sha256sum | awk '{print $1}'); flock -w %d -x "$lease" bash -c %s; rc=$?; echo FAK_STRIX_EXIT=$rc; exit $rc`, digestBytes([]byte(semantic)), leaseWait, shellQuote(inner))
}

func runStrixTargetCommand(ctx context.Context, target *StrixTarget, command string, stdin []byte) ([]byte, error) {
	var cmd *exec.Cmd
	if target.Mode == "local" {
		cmd = exec.CommandContext(ctx, "bash", "-c", command)
	} else {
		cmd = exec.CommandContext(ctx, "ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10", target.Host, command)
	}
	if stdin != nil {
		cmd.Stdin = bytes.NewReader(stdin)
	}
	windowgate.ConfigureBackgroundCommand(cmd)
	return cmd.CombinedOutput()
}

var verifySourceBindingFn = VerifySourceBinding

func VerifySourceBinding(ctx context.Context, target *StrixTarget, gitTip, gitRef string) error {
	if target == nil || !target.Reachable {
		return fmt.Errorf("target is nil or unreachable")
	}
	if gitTip == "" && gitRef == "" {
		return nil
	}
	dir := os.Getenv("FAK_STRIX_DIR")
	if dir == "" {
		dir = "/var/lib/fak/repo"
	}
	out, err := runStrixTargetCommand(ctx, target, "cd "+shellQuote(dir)+" && git rev-parse HEAD", nil)
	if err != nil {
		return fmt.Errorf("failed to resolve git commit on target: %v (output: %s)", err, truncateOutput(string(out), 150))
	}
	actual := strings.TrimSpace(string(out))
	if actual == "" {
		return fmt.Errorf("target returned empty git commit")
	}
	if gitTip != "" && !strings.HasPrefix(actual, gitTip) && !strings.HasPrefix(gitTip, actual) {
		return fmt.Errorf("source binding mismatch: target HEAD %s does not match GitTip %s", actual, gitTip)
	}
	return nil
}

type stagedStrixCandidate struct {
	SourceBinding SourceBinding
	Trace         []string
}

var stageStrixCandidateFn = stageStrixCandidate
var cleanupStrixCandidateFn = cleanupStrixCandidate

const strixBuildCommand = `set -eu
test "$(sha256sum archive.tar | awk '{print $1}')" = "$1"
echo FAK_STRIX_TRACE=verify_archive
mkdir src build build/spirv
tar -xf archive.tar -C src
echo FAK_STRIX_TRACE=extract_fresh
for shader in src/internal/compute/shaders/*.comp; do name=$(basename "$shader" .comp); glslc -O --target-env=vulkan1.2 -fshader-stage=comp "$shader" -o "build/spirv/$name.spv"; done
echo FAK_STRIX_TRACE=build_spirv
c++ -O3 -std=c++17 -fPIC -c src/internal/compute/vulkan_shim.cpp -o build/vulkan_shim.o
ar rcs build/libfakvulkan.a build/vulkan_shim.o
(cd src && CGO_ENABLED=1 CGO_LDFLAGS="-L$PWD/../build" go test -c -tags vulkan -o ../build/compute.test ./internal/compute)
echo FAK_STRIX_TRACE=build_compute
echo FAK_STRIX_BINARY_SHA256=sha256:$(sha256sum build/compute.test | awk '{print $1}')
echo FAK_STRIX_SHADER_SHA256=sha256:$(find build/spirv -type f -name '*.spv' -print0 | sort -z | xargs -0 sha256sum | sha256sum | awk '{print $1}')`

func stageStrixCandidate(ctx context.Context, target *StrixTarget, opts StrixValidationOpts) (stagedStrixCandidate, error) {
	wait := opts.AdmissionTimeout
	if wait <= 0 {
		wait = 30 * time.Second
	}
	if wait > 60*time.Second {
		return stagedStrixCandidate{}, fmt.Errorf("amdgpu: admission timeout exceeds 60s")
	}
	if !fullGitTipRE.MatchString(opts.GitTip) || !sha256RE.MatchString(opts.SourceArchiveSHA256) || len(opts.CandidateArchive) == 0 || digestBytes(opts.CandidateArchive) != opts.SourceArchiveSHA256 {
		return stagedStrixCandidate{}, fmt.Errorf("amdgpu: incomplete or mismatched exact candidate archive")
	}
	if err := validateCandidateArchiveBytes(opts.CandidateArchive); err != nil {
		return stagedStrixCandidate{}, err
	}
	out, err := runStrixTargetCommand(ctx, target, "mktemp -d /tmp/fak-strix-validation.XXXXXX", nil)
	if err != nil {
		return stagedStrixCandidate{}, fmt.Errorf("amdgpu: allocate fresh target workspace: %w", err)
	}
	work := strings.TrimSpace(string(out))
	if err := validateStrixWorkspace(work); err != nil {
		return stagedStrixCandidate{}, err
	}
	keepWorkspace := false
	defer func() {
		if !keepWorkspace {
			cleanupStrixCandidateFresh(target, work)
		}
	}()
	if _, err = runStrixTargetCommand(ctx, target, "umask 077; cat > "+shellQuote(work+"/archive.tar"), opts.CandidateArchive); err != nil {
		return stagedStrixCandidate{}, fmt.Errorf("amdgpu: stage candidate archive: %w", err)
	}
	cmd := "cd " + shellQuote(work) + " && bash -c " + shellQuote(strixBuildCommand) + " -- " + shellQuote(strings.TrimPrefix(opts.SourceArchiveSHA256, "sha256:"))
	buildOut, err := runStrixTargetCommand(ctx, target, cmd, nil)
	if err != nil {
		return stagedStrixCandidate{}, fmt.Errorf("amdgpu: fresh candidate build failed: %v (%s)", err, truncateOutput(string(buildOut), 500))
	}
	bin := markerValue(string(buildOut), "FAK_STRIX_BINARY_SHA256=")
	shader := markerValue(string(buildOut), "FAK_STRIX_SHADER_SHA256=")
	if !sha256RE.MatchString(bin) || !sha256RE.MatchString(shader) {
		return stagedStrixCandidate{}, fmt.Errorf("amdgpu: build omitted binary or shader digest")
	}
	keepWorkspace = true
	return stagedStrixCandidate{SourceBinding: SourceBinding{GitTip: opts.GitTip, GitRef: opts.GitRef, SourceArchiveSHA256: opts.SourceArchiveSHA256, BinarySHA256: bin, ShaderBundleSHA256: shader, BuildCommandSHA256: digestBytes([]byte(strixBuildCommand)), WorkDir: work, AdmissionWait: wait}, Trace: []string{"stage_archive", "verify_archive", "extract_fresh", "build_spirv", "build_compute", "hash_artifacts"}}, nil
}

func cleanupStrixCandidateFresh(target *StrixTarget, work string) bool {
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()
	return cleanupStrixCandidateFn(ctx, target, work)
}

func cleanupStrixCandidate(ctx context.Context, target *StrixTarget, work string) bool {
	if validateStrixWorkspace(work) != nil {
		return false
	}
	out, err := runStrixTargetCommand(ctx, target, "rm -rf -- "+shellQuote(work)+" && echo FAK_STRIX_CLEANUP=1", nil)
	return err == nil && strings.Contains(string(out), "FAK_STRIX_CLEANUP=1")
}

func markerValue(out, prefix string) string {
	for _, line := range strings.Split(out, "\n") {
		line = strings.TrimSpace(line)
		if strings.HasPrefix(line, prefix) {
			return strings.TrimSpace(strings.TrimPrefix(line, prefix))
		}
	}
	return ""
}

func failedReceipt(opts StrixValidationOpts, target StrixTarget, message string, err error) *StrixValidationReceipt {
	r := NewStrixValidationReceipt(target, opts.GitRef, opts.GitTip, opts.Command)
	if opts.RunSubkernels {
		if specs, selectErr := FilterSubkernelSpecs(opts.Subkernels); selectErr == nil {
			r.SelectedCount = len(specs)
			r.SelectedSubkernels = len(specs)
		}
	}
	r.Verdict = "FAIL"
	r.Verified = false
	r.Failures = append(r.Failures, message)
	r.Digest, _ = r.ComputeDigest()
	return r
}

func RunStrixValidation(ctx context.Context, opts StrixValidationOpts) (*StrixValidationReceipt, error) {
	if opts.Timeout > 0 {
		var cancel context.CancelFunc
		ctx, cancel = context.WithTimeout(ctx, opts.Timeout)
		defer cancel()
	}
	var specs []SubkernelSpec
	selectedAblations := 0
	var err error
	if opts.RunSubkernels {
		specs, err = FilterSubkernelSpecs(opts.Subkernels)
		if err != nil || len(specs) == 0 {
			if err == nil {
				err = fmt.Errorf("amdgpu: zero subkernels selected")
			}
			return failedReceipt(opts, StrixTarget{Host: opts.Host}, err.Error(), err), err
		}
	}
	if opts.RunAblations {
		selectedAblations, err = validateAblationSelectors(opts.Ablations)
		if err != nil {
			return failedReceipt(opts, StrixTarget{Host: opts.Host}, err.Error(), err), err
		}
	}
	target, err := DiscoverStrixTarget(ctx, opts.Host)
	if err != nil || target == nil || !target.Reachable {
		if err == nil {
			err = fmt.Errorf("target unavailable")
		}
		return failedReceipt(opts, StrixTarget{Host: opts.Host}, "strix halo target unreachable: "+err.Error(), err), err
	}
	receipt := NewStrixValidationReceipt(*target, opts.GitRef, opts.GitTip, opts.Command)
	receipt.SelectedCount = len(specs)
	receipt.SelectedSubkernels = len(specs)
	receipt.SelectedAblations = selectedAblations
	if opts.RequireSourceBinding && strings.TrimSpace(opts.GitTip) == "" {
		err = fmt.Errorf("amdgpu: source binding required but GitTip is missing")
		return failedReceipt(opts, *target, err.Error(), err), err
	}
	if !opts.RequireSourceBinding {
		err = fmt.Errorf("amdgpu: v2 validation requires source binding")
		return failedReceipt(opts, *target, err.Error(), err), err
	}
	if opts.AdmissionTimeout > 60*time.Second {
		err = fmt.Errorf("amdgpu: admission timeout exceeds 60s")
		return failedReceipt(opts, *target, err.Error(), err), err
	}
	staged, err := stageStrixCandidateFn(ctx, target, opts)
	if err != nil {
		return failedReceipt(opts, *target, err.Error(), err), err
	}
	cleanupDone := false
	cleanup := func() {
		if cleanupDone {
			return
		}
		cleanupDone = true
		cleanupCtx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
		defer cancel()
		receipt.Provenance.CleanupObserved = cleanupStrixCandidateFn(cleanupCtx, target, staged.SourceBinding.WorkDir)
		receipt.Provenance.Trace = append(receipt.Provenance.Trace, "cleanup")
	}
	defer cleanup()
	ctx = withPreparedSourceBinding(ctx, staged.SourceBinding)
	receipt.Provenance.SourceArchiveSHA256 = staged.SourceBinding.SourceArchiveSHA256
	receipt.Provenance.BinarySHA256 = staged.SourceBinding.BinarySHA256
	receipt.Provenance.ShaderBundleSHA256 = staged.SourceBinding.ShaderBundleSHA256
	receipt.Provenance.BuildCommandSHA256 = staged.SourceBinding.BuildCommandSHA256
	receipt.Provenance.EngineIdentity = "fak-native/vulkan"
	receipt.Provenance.Trace = append(receipt.Provenance.Trace, staged.Trace...)
	var validationErr error
	if !opts.RunSubkernels && !opts.RunAblations {
		validationErr = fmt.Errorf("amdgpu: neither subkernels nor ablations requested")
		receipt.Failures = append(receipt.Failures, validationErr.Error())
	}
	if opts.RunSubkernels {
		results, runErr := RunSubkernelTests(ctx, target, opts.Subkernels)
		receipt.Subkernels = results
		receipt.ExecutedCount = len(results)
		receipt.ExecutedSubkernels = len(results)
		if runErr != nil {
			validationErr = runErr
			receipt.Failures = append(receipt.Failures, runErr.Error())
		}
		for _, r := range results {
			if r.Status != "PASS" {
				receipt.Failures = append(receipt.Failures, fmt.Sprintf("subkernel %q did not pass", r.Name))
			}
		}
	}
	if opts.RunAblations {
		results, runErr := RunAblationTests(ctx, target, opts.Ablations)
		receipt.Ablations = results
		receipt.ExecutedAblations = len(results)
		if runErr != nil {
			validationErr = runErr
			receipt.Failures = append(receipt.Failures, runErr.Error())
		}
		if len(results) == 0 {
			receipt.Failures = append(receipt.Failures, "ablations requested but none executed")
		}
		for _, r := range results {
			if r.Verdict != "VERIFIED_LIFT" && r.Verdict != "PARITY_MATCH" {
				receipt.Failures = append(receipt.Failures, fmt.Sprintf("ablation %q did not produce supported evidence", r.Feature))
			}
		}
	}
	cleanup()
	if !receipt.Provenance.CleanupObserved {
		receipt.Failures = append(receipt.Failures, "fresh target workspace cleanup not observed")
	}
	receipt.Verdict = "PASS"
	receipt.Verified = len(receipt.Failures) == 0
	if !receipt.Verified {
		receipt.Verdict = "FAIL"
		if validationErr == nil {
			validationErr = fmt.Errorf("amdgpu: validation evidence incomplete")
		}
	}
	receipt.Provenance.ExecutionManifestSHA256 = executionManifestDigest(receipt)
	receipt.Digest, _ = receipt.ComputeDigest()
	if receipt.Verified {
		if e := receipt.Validate(); e != nil {
			receipt.Verified = false
			receipt.Verdict = "FAIL"
			receipt.Failures = append(receipt.Failures, e.Error())
			receipt.Digest, _ = receipt.ComputeDigest()
			validationErr = e
		}
	}
	return receipt, validationErr
}
