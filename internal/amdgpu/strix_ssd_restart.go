package amdgpu

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"regexp"
	"strconv"
	"strings"
	"time"
)

const (
	StrixSSDRestartSchema = "fak.strix.ssd-restart/v1"
	ssdRestartMarker      = "FAK_VULKAN_WARM_DISK_RECEIPT="
	ssdRestartTestName    = "TestInKernelWarmPrefixDiskAcrossProcessesVulkan"
	ssdRestartOutputLimit = 64 << 10
)

type StrixSSDRestartOpts struct {
	Host                 string
	GitRef               string
	GitTip               string
	Command              string
	Timeout              time.Duration
	AdmissionTimeout     time.Duration
	RequireSourceBinding bool
	CandidateArchive     []byte
	SourceArchiveSHA256  string
}

type StrixSSDRestartFailure struct {
	Code    string `json:"code"`
	Message string `json:"message"`
}

type StrixSSDRestartReceipt struct {
	Schema       string                         `json:"schema"`
	Timestamp    string                         `json:"timestamp"`
	Verdict      string                         `json:"verdict"`
	Hardware     bool                           `json:"hardware"`
	EvidenceRung string                         `json:"evidence_rung,omitempty"`
	Target       StrixTarget                    `json:"target"`
	Provenance   StrixProvenance                `json:"provenance"`
	Execution    StrixExecutionEvidence         `json:"execution"`
	Device       *StrixVulkanDeviceObservation  `json:"device_observation,omitempty"`
	Filesystem   *StrixSSDFileSystemObservation `json:"filesystem_observation,omitempty"`
	Witness      *StrixSSDRestartWitness        `json:"witness,omitempty"`
	Failures     []StrixSSDRestartFailure       `json:"failures,omitempty"`
	Digest       string                         `json:"digest"`
	Verified     bool                           `json:"verified"`
}

type StrixSSDFileSystemObservation struct {
	Workspace         string `json:"workspace"`
	MountTarget       string `json:"mount_target"`
	Source            string `json:"source"`
	FSType            string `json:"fs_type"`
	MajorMinor        string `json:"major_minor"`
	SourcePath        string `json:"source_path"`
	SourceType        string `json:"source_type"`
	ParentDisk        string `json:"parent_disk"`
	ParentType        string `json:"parent_type"`
	Transport         string `json:"transport"`
	Rotational        bool   `json:"rotational"`
	RotationObserved  bool   `json:"rotation_observed"`
	ObservationSHA256 string `json:"observation_sha256"`
}

type StrixVulkanDeviceObservation struct {
	Name          string `json:"name"`
	DeviceType    string `json:"device_type"`
	VendorID      string `json:"vendor_id"`
	DriverID      string `json:"driver_id"`
	DriverName    string `json:"driver_name"`
	DriverInfo    string `json:"driver_info"`
	ISA           string `json:"isa"`
	SummarySHA256 string `json:"summary_sha256"`
	SummaryBytes  int    `json:"summary_bytes"`
}

type StrixSSDRestartTransfer struct {
	ComputeDispatches uint64 `json:"compute_dispatches"`
	DispatchSubmits   uint64 `json:"dispatch_submits"`
	H2DCount          uint64 `json:"h2d_count"`
	H2DBytes          uint64 `json:"h2d_bytes"`
	D2HCount          uint64 `json:"d2h_count"`
	D2HBytes          uint64 `json:"d2h_bytes"`
}

type StrixSSDRestartChild struct {
	Schema                   string                  `json:"schema"`
	Phase                    string                  `json:"phase"`
	PID                      int                     `json:"pid"`
	SourceRevision           string                  `json:"source_revision"`
	SourceArchiveSHA256      string                  `json:"source_archive_sha256"`
	SourceModified           bool                    `json:"source_modified"`
	ExecutableSHA256         string                  `json:"executable_sha256"`
	ModelIdentity            string                  `json:"model_identity"`
	Backend                  string                  `json:"backend"`
	Device                   string                  `json:"device"`
	Driver                   string                  `json:"driver"`
	Runtime                  string                  `json:"runtime"`
	DiskOutcome              string                  `json:"disk_outcome"`
	DiskTier                 string                  `json:"disk_tier"`
	WriteBytes               int64                   `json:"write_bytes,omitempty"`
	ReadBytes                int64                   `json:"read_bytes,omitempty"`
	Prefilled                int                     `json:"prefilled_tokens"`
	Stable                   int                     `json:"stable_tokens"`
	Matched                  int                     `json:"matched_tokens"`
	Generated                []int                   `json:"generated_tokens"`
	LogitsDigest             string                  `json:"logits_digest"`
	ContinuationLogitsDigest string                  `json:"continuation_logits_digest"`
	Transfers                StrixSSDRestartTransfer `json:"transfers"`
}

type StrixSSDRestartWitness struct {
	Schema                  string               `json:"schema"`
	PID                     int                  `json:"pid"`
	CacheMode               string               `json:"cache_mode"`
	SourceRevision          string               `json:"source_revision"`
	SourceArchiveSHA256     string               `json:"source_archive_sha256"`
	SourceModified          bool                 `json:"source_modified"`
	ExecutableSHA256        string               `json:"executable_sha256"`
	ModelIdentity           string               `json:"model_identity"`
	ExpectedKVTransferBytes uint64               `json:"expected_kv_transfer_bytes"`
	ExpectedColdD2HBytes    uint64               `json:"expected_cold_d2h_bytes"`
	ExpectedKVTransferCount uint64               `json:"expected_kv_transfer_count"`
	Cold                    StrixSSDRestartChild `json:"cold"`
	Warm                    StrixSSDRestartChild `json:"warm"`
	TokenParity             bool                 `json:"token_parity"`
	LogitParity             bool                 `json:"logit_parity"`
	ContinuationLogitParity bool                 `json:"continuation_logit_parity"`
}

type StrixSSDRestartError struct {
	Code string
	Err  error
}

func (e *StrixSSDRestartError) Error() string {
	return "amdgpu: native SSD restart " + e.Code + ": " + e.Err.Error()
}
func (e *StrixSSDRestartError) Unwrap() error { return e.Err }

func (r *StrixSSDRestartReceipt) ComputeDigest() (string, error) {
	copyReceipt := *r
	copyReceipt.Digest = ""
	raw, err := json.Marshal(copyReceipt)
	if err != nil {
		return "", err
	}
	return digestBytes(raw), nil
}

func (r *StrixSSDRestartReceipt) Validate() error {
	if r.Schema != StrixSSDRestartSchema || (r.Verdict != "PASS" && r.Verdict != "FAIL") {
		return errors.New("invalid native SSD restart receipt schema or verdict")
	}
	want, err := r.ComputeDigest()
	if err != nil || r.Digest != want {
		return errors.New("native SSD restart receipt digest mismatch")
	}
	if r.Verdict == "FAIL" {
		if r.Verified || r.Hardware || r.EvidenceRung != "" || len(r.Failures) == 0 {
			return errors.New("failed native SSD restart receipt claims evidence")
		}
		return nil
	}
	if !r.Verified || !r.Hardware || r.EvidenceRung != "HW_WITNESSED" || len(r.Failures) != 0 || !r.Provenance.CleanupObserved || r.Device == nil || r.Filesystem == nil || r.Witness == nil {
		return errors.New("passing native SSD restart receipt is incomplete")
	}
	if !fullGitTipRE.MatchString(r.Provenance.GitTip) || !sha256RE.MatchString(r.Provenance.SourceArchiveSHA256) || !sha256RE.MatchString(r.Provenance.BinarySHA256) || !sha256RE.MatchString(r.Provenance.ShaderBundleSHA256) || !sha256RE.MatchString(r.Provenance.BuildCommandSHA256) || !sha256RE.MatchString(r.Provenance.ExecutionManifestSHA256) || r.Provenance.EngineIdentity != "fak-native/vulkan" || strings.TrimSpace(r.Provenance.Command) == "" {
		return errors.New("passing native SSD restart receipt has incomplete provenance")
	}
	if !r.Target.Reachable || strings.ToLower(r.Target.TargetISA) != "gfx1151" || (!strings.Contains(strings.ToLower(r.Target.GPUName), "8060s") && !strings.Contains(strings.ToLower(r.Target.GPUName), "strix") && !strings.Contains(strings.ToLower(r.Target.GPUName), "gfx1151")) {
		return errors.New("passing native SSD restart receipt lacks a reachable Strix Halo target")
	}
	if err := r.Device.Validate(); err != nil || r.Target.GPUName != r.Device.Name || r.Target.TargetISA != r.Device.ISA {
		return errors.New("passing native SSD restart receipt is not bound to the observed Vulkan device")
	}
	if err := r.Filesystem.Validate(); err != nil {
		return errors.New("passing native SSD restart receipt is not bound to a physical SSD workspace")
	}
	if err := validExecutionEvidence(r.Execution); err != nil {
		return err
	}
	if r.Execution.SourceArchiveSHA256 != r.Provenance.SourceArchiveSHA256 || r.Execution.BinarySHA256 != r.Provenance.BinarySHA256 || r.Execution.ShaderBundleSHA256 != r.Provenance.ShaderBundleSHA256 || r.Execution.DeviceIdentity != r.Target.GPUName+"|"+r.Target.TargetISA || r.Execution.EngineIdentity != r.Provenance.EngineIdentity {
		return errors.New("native SSD restart execution contradicts provenance")
	}
	if r.Provenance.ExecutionManifestSHA256 != strixSSDRestartExecutionManifestDigest(r) {
		return errors.New("native SSD restart execution manifest digest mismatch")
	}
	return validateStrixSSDRestartWitness(r.Witness, r.Provenance.GitTip, r.Provenance.SourceArchiveSHA256, r.Provenance.BinarySHA256, r.Target.GPUName)
}

const strixSSDRestartBuildCommand = `set -eu
test "$(sha256sum archive.tar | awk '{print $1}')" = "$1"
echo FAK_STRIX_TRACE=verify_archive
mkdir src build build/spirv
tar -xf archive.tar -C src
echo FAK_STRIX_TRACE=extract_fresh
for shader in src/internal/compute/shaders/*.comp; do name=$(basename "$shader" .comp); glslc -O --target-env=vulkan1.2 -fshader-stage=comp "$shader" -o "build/spirv/$name.spv"; done
echo FAK_STRIX_TRACE=build_spirv
c++ -O3 -std=c++17 -fPIC -c src/internal/compute/vulkan_shim.cpp -o build/vulkan_shim.o
ar rcs build/libfakvulkan.a build/vulkan_shim.o
(cd src && CGO_ENABLED=1 CGO_LDFLAGS="-L$PWD/../build" go test -c -buildvcs=false -tags vulkan -ldflags "-X github.com/anthony-chaudhary/fak/internal/agent.vulkanWarmDiskSourceRevisionStamp=$2 -X github.com/anthony-chaudhary/fak/internal/agent.vulkanWarmDiskSourceArchiveSHA256Stamp=$1" -o ../build/agent.test ./internal/agent)
echo FAK_STRIX_TRACE=build_agent
echo FAK_STRIX_BINARY_SHA256=sha256:$(sha256sum build/agent.test | awk '{print $1}')
echo FAK_STRIX_SHADER_SHA256=sha256:$(find build/spirv -type f -name '*.spv' -print0 | sort -z | xargs -0 sha256sum | sha256sum | awk '{print $1}')`

type stagedStrixSSDRestart struct {
	stagedStrixCandidate
	Filesystem StrixSSDFileSystemObservation
}

func stageStrixSSDRestart(ctx context.Context, target *StrixTarget, opts StrixSSDRestartOpts) (stagedStrixSSDRestart, error) {
	wait := opts.AdmissionTimeout
	if wait <= 0 {
		wait = 30 * time.Second
	}
	if wait > 60*time.Second || !fullGitTipRE.MatchString(opts.GitTip) || !sha256RE.MatchString(opts.SourceArchiveSHA256) || len(opts.CandidateArchive) == 0 || digestBytes(opts.CandidateArchive) != opts.SourceArchiveSHA256 {
		return stagedStrixSSDRestart{}, errors.New("amdgpu: incomplete native SSD restart admission")
	}
	if err := validateCandidateArchiveBytes(opts.CandidateArchive); err != nil {
		return stagedStrixSSDRestart{}, err
	}
	out, err := runStrixTargetCommand(ctx, target, "mktemp -d /tmp/fak-strix-validation.XXXXXX", nil)
	if err != nil {
		return stagedStrixSSDRestart{}, fmt.Errorf("amdgpu: allocate fresh target workspace: %w", err)
	}
	work := strings.TrimSpace(string(out))
	if err := validateStrixWorkspace(work); err != nil {
		return stagedStrixSSDRestart{}, err
	}
	keep := false
	defer func() {
		if !keep {
			cleanupStrixCandidateFresh(target, work)
		}
	}()
	filesystem, err := observeStrixSSDFileSystem(ctx, target, work)
	if err != nil {
		return stagedStrixSSDRestart{}, fmt.Errorf("amdgpu: verify native SSD workspace: %w", err)
	}
	if _, err = runStrixTargetCommand(ctx, target, "umask 077; mkdir "+shellQuote(work+"/tmp")+" && cat > "+shellQuote(work+"/archive.tar"), opts.CandidateArchive); err != nil {
		return stagedStrixSSDRestart{}, fmt.Errorf("amdgpu: stage candidate archive: %w", err)
	}
	archiveHex := strings.TrimPrefix(opts.SourceArchiveSHA256, "sha256:")
	cmd := "cd " + shellQuote(work) + " && bash -c " + shellQuote(strixSSDRestartBuildCommand) + " -- " + shellQuote(archiveHex) + " " + shellQuote(opts.GitTip)
	buildOut, err := runStrixTargetCommand(ctx, target, cmd, nil)
	if err != nil {
		return stagedStrixSSDRestart{}, fmt.Errorf("amdgpu: fresh native SSD restart build failed: %v (%s)", err, truncateOutput(string(buildOut), 500))
	}
	bin := markerValue(string(buildOut), "FAK_STRIX_BINARY_SHA256=")
	shader := markerValue(string(buildOut), "FAK_STRIX_SHADER_SHA256=")
	if !sha256RE.MatchString(bin) || !sha256RE.MatchString(shader) {
		return stagedStrixSSDRestart{}, errors.New("amdgpu: native SSD restart build omitted artifact digests")
	}
	keep = true
	return stagedStrixSSDRestart{stagedStrixCandidate: stagedStrixCandidate{SourceBinding: SourceBinding{GitTip: opts.GitTip, GitRef: opts.GitRef, SourceArchiveSHA256: opts.SourceArchiveSHA256, BinarySHA256: bin, ShaderBundleSHA256: shader, BuildCommandSHA256: digestBytes([]byte(strixSSDRestartBuildCommand)), WorkDir: work, AdmissionWait: wait}, Trace: []string{"observe_ssd_filesystem", "stage_archive", "verify_archive", "extract_fresh", "build_spirv", "build_agent", "hash_artifacts"}}, Filesystem: filesystem}, nil
}

func buildStrixSSDRestartAdmissionCommand(target *StrixTarget, sb SourceBinding) string {
	wait := int64(sb.AdmissionWait / time.Second)
	if wait <= 0 {
		wait = 30
	}
	verify := `test "sha256:$(sha256sum ` + shellQuote(sb.WorkDir+"/build/agent.test") + ` | cut -f1 -d' ')" = ` + shellQuote(sb.BinarySHA256) + ` && test "sha256:$(cd ` + shellQuote(sb.WorkDir) + ` && find build/spirv -type f -name '*.spv' -print0 | sort -z | xargs -0 sha256sum | sha256sum | cut -f1 -d' ')" = ` + shellQuote(sb.ShaderBundleSHA256)
	testCmd := `TMPDIR=` + shellQuote(sb.WorkDir+"/tmp") + ` FAK_VULKAN_REQUIRE_DEVICE=1 FAK_VULKAN_DISPATCH_PROFILE=1 FAK_VULKAN_EXPECT_DEVICE=` + shellQuote(target.GPUName) + ` FAK_VULKAN_SPIRV=` + shellQuote(sb.WorkDir+"/build/spirv") + ` ` + shellQuote(sb.WorkDir+"/build/agent.test") + ` -test.run=^` + ssdRestartTestName + `$ -test.count=1 -fak-vulkan-warm-disk-restart=true`
	deviceCmd := `timeout --signal=TERM --kill-after=5s 60s bash -c ` + shellQuote(testCmd)
	semantic := verify + "; " + deviceCmd
	inner := `echo FAK_STRIX_ADMISSION_ACQUIRED=1; trap 'echo FAK_STRIX_ADMISSION_RELEASED=1' EXIT; ` + verify + ` || exit 1; echo FAK_STRIX_ARTIFACT_REHASH=1; echo FAK_STRIX_ENGINE=fak-native/vulkan; echo FAK_STRIX_DEVICE_TIMEOUT_MS=60000; echo FAK_STRIX_DEVICE=` + shellQuote(target.GPUName+"|"+target.TargetISA) + `; ` + deviceCmd
	return fmt.Sprintf(`set +e; echo FAK_STRIX_COMMAND_SHA256=%s; lease="${FAK_GPU_LEASE:-/tmp/fak-gpu.lease}"; echo FAK_STRIX_LEASE_SHA256=sha256:$(printf %%s "$lease" | sha256sum | awk '{print $1}'); flock -w %d -x "$lease" bash -c %s; rc=$?; echo FAK_STRIX_EXIT=$rc; exit 0`, digestBytes([]byte(semantic)), wait, shellQuote(inner))
}

func newStrixSSDRestartReceipt(opts StrixSSDRestartOpts, target StrixTarget) *StrixSSDRestartReceipt {
	return &StrixSSDRestartReceipt{Schema: StrixSSDRestartSchema, Timestamp: time.Now().UTC().Format(time.RFC3339), Verdict: "FAIL", Target: target, Provenance: StrixProvenance{GitRef: opts.GitRef, GitTip: opts.GitTip, Command: opts.Command, GeneratedBy: "fak-dev amd-strix-validate", Transport: target.Mode}}
}

func finishStrixSSDRestart(r *StrixSSDRestartReceipt, code string, err error) (*StrixSSDRestartReceipt, error) {
	if err != nil {
		r.Failures = append(r.Failures, StrixSSDRestartFailure{Code: code, Message: truncateOutput(err.Error(), 500)})
		if code != "CLEANUP_FAILED" && r.Provenance.SourceArchiveSHA256 != "" && !r.Provenance.CleanupObserved {
			r.Failures = append(r.Failures, StrixSSDRestartFailure{Code: "CLEANUP_FAILED", Message: "fresh target workspace cleanup was not observed"})
		}
		r.Verdict, r.Verified, r.Hardware, r.EvidenceRung = "FAIL", false, false, ""
	}
	r.Digest, _ = r.ComputeDigest()
	if err == nil {
		return r, nil
	}
	return r, &StrixSSDRestartError{Code: code, Err: err}
}

func RunStrixSSDRestart(ctx context.Context, opts StrixSSDRestartOpts) (*StrixSSDRestartReceipt, error) {
	if opts.Host != "" {
		if err := validateStrixSSHDestination(opts.Host); err != nil {
			return finishStrixSSDRestart(newStrixSSDRestartReceipt(opts, StrixTarget{Host: opts.Host}), "INVALID_HOST", err)
		}
	}
	if !opts.RequireSourceBinding || !fullGitTipRE.MatchString(opts.GitTip) || !sha256RE.MatchString(opts.SourceArchiveSHA256) {
		return finishStrixSSDRestart(newStrixSSDRestartReceipt(opts, StrixTarget{Host: opts.Host}), "SOURCE_BINDING_REQUIRED", errors.New("full source revision and exact archive digest are required"))
	}
	if opts.Timeout > 0 {
		var cancel context.CancelFunc
		ctx, cancel = context.WithTimeout(ctx, opts.Timeout)
		defer cancel()
	}
	target, err := DiscoverStrixTarget(ctx, opts.Host)
	if err != nil || target == nil || !target.Reachable {
		if err == nil {
			err = errors.New("target unavailable")
		}
		return finishStrixSSDRestart(newStrixSSDRestartReceipt(opts, StrixTarget{Host: opts.Host}), "TARGET_UNREACHABLE", err)
	}
	r := newStrixSSDRestartReceipt(opts, *target)
	staged, err := stageStrixSSDRestart(ctx, target, opts)
	if err != nil {
		return finishStrixSSDRestart(r, "STAGE_FAILED", err)
	}
	r.Provenance.SourceArchiveSHA256 = staged.SourceBinding.SourceArchiveSHA256
	r.Provenance.BinarySHA256 = staged.SourceBinding.BinarySHA256
	r.Provenance.ShaderBundleSHA256 = staged.SourceBinding.ShaderBundleSHA256
	r.Provenance.BuildCommandSHA256 = staged.SourceBinding.BuildCommandSHA256
	r.Provenance.EngineIdentity = "fak-native/vulkan"
	r.Provenance.Trace = append(r.Provenance.Trace, staged.Trace...)
	r.Filesystem = &staged.Filesystem
	cleaned := false
	cleanup := func() {
		if cleaned {
			return
		}
		cleaned = true
		cleanupCtx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
		defer cancel()
		r.Provenance.CleanupObserved = cleanupStrixCandidateFn(cleanupCtx, target, staged.SourceBinding.WorkDir)
		r.Provenance.Trace = append(r.Provenance.Trace, "cleanup")
	}
	defer cleanup()
	device, err := observeStrixVulkanDevice(ctx, target)
	if err != nil {
		cleanup()
		return finishStrixSSDRestart(r, "VULKAN_DEVICE_OBSERVATION_FAILED", err)
	}
	r.Device = &device
	target.GPUName, target.TargetISA = device.Name, device.ISA
	r.Target.GPUName, r.Target.TargetISA = device.Name, device.ISA
	r.Provenance.Trace = append(r.Provenance.Trace, "observe_vulkan_device")
	out, runErr := runStrixTargetCommand(ctx, target, buildStrixSSDRestartAdmissionCommand(target, staged.SourceBinding), nil)
	r.Execution = executionEvidenceFromOutput(string(out), staged.SourceBinding)
	if len(out) > ssdRestartOutputLimit {
		cleanup()
		return finishStrixSSDRestart(r, "OUTPUT_TOO_LARGE", fmt.Errorf("device output is %d bytes", len(out)))
	}
	if runErr != nil {
		cleanup()
		return finishStrixSSDRestart(r, "TRANSPORT_FAILED", runErr)
	}
	if r.Execution.ExitCode == nil || *r.Execution.ExitCode != 0 {
		cleanup()
		return finishStrixSSDRestart(r, "EXECUTION_FAILED", errors.New(safeStrixSSDRestartFailure(out, r.Execution.ExitCode)))
	}
	if err := validExecutionEvidence(r.Execution); err != nil {
		cleanup()
		return finishStrixSSDRestart(r, "EXECUTION_EVIDENCE_INVALID", err)
	}
	witness, err := parseStrixSSDRestartWitness(out)
	if err == nil {
		err = validateStrixSSDRestartWitness(witness, opts.GitTip, opts.SourceArchiveSHA256, staged.SourceBinding.BinarySHA256, target.GPUName)
	}
	if err != nil {
		cleanup()
		return finishStrixSSDRestart(r, "WITNESS_INVALID", err)
	}
	r.Witness = witness
	cleanup()
	if !r.Provenance.CleanupObserved {
		return finishStrixSSDRestart(r, "CLEANUP_FAILED", errors.New("fresh target workspace cleanup was not observed"))
	}
	r.Verdict, r.Verified, r.Hardware, r.EvidenceRung = "PASS", true, true, "HW_WITNESSED"
	r.Provenance.ExecutionManifestSHA256 = strixSSDRestartExecutionManifestDigest(r)
	r.Digest, _ = r.ComputeDigest()
	if err := r.Validate(); err != nil {
		return finishStrixSSDRestart(r, "RECEIPT_INVALID", err)
	}
	return r, nil
}

func strixSSDRestartExecutionManifestDigest(r *StrixSSDRestartReceipt) string {
	manifest := struct {
		GitTip          string                         `json:"git_tip"`
		Source          string                         `json:"source_archive_sha256"`
		Binary          string                         `json:"binary_sha256"`
		Shader          string                         `json:"shader_bundle_sha256"`
		Build           string                         `json:"build_command_sha256"`
		Command         string                         `json:"command_sha256"`
		Lease           string                         `json:"lease_path_sha256"`
		Device          string                         `json:"device_identity"`
		Acquired        bool                           `json:"acquired"`
		Released        bool                           `json:"released"`
		CleanupObserved bool                           `json:"cleanup_observed"`
		ObservedDevice  *StrixVulkanDeviceObservation  `json:"observed_device"`
		Filesystem      *StrixSSDFileSystemObservation `json:"filesystem"`
		Witness         *StrixSSDRestartWitness        `json:"witness"`
	}{r.Provenance.GitTip, r.Provenance.SourceArchiveSHA256, r.Provenance.BinarySHA256, r.Provenance.ShaderBundleSHA256, r.Provenance.BuildCommandSHA256, r.Execution.CommandSHA256, r.Execution.LeasePathSHA256, r.Execution.DeviceIdentity, r.Execution.Acquired, r.Execution.Released, r.Provenance.CleanupObserved, r.Device, r.Filesystem, r.Witness}
	raw, _ := json.Marshal(manifest)
	return digestBytes(raw)
}

const strixSSDFileSystemScript = `set +e
work="$1"
mount=$(findmnt --target "$work" --noheadings --pairs --output TARGET,SOURCE,FSTYPE,MAJ:MIN 2>/dev/null)
rc=$?
source=$(findmnt --target "$work" --noheadings --raw --output SOURCE 2>/dev/null)
source=${source%%[*}
block=""
parent=""
if [ "$rc" -eq 0 ] && [ -n "$source" ]; then
  block=$(lsblk --nodeps --noheadings --pairs --output PATH,KNAME,TYPE,ROTA,PKNAME,TRAN "$source" 2>/dev/null)
  block_rc=$?
  pk=$(lsblk --nodeps --noheadings --raw --output PKNAME "$source" 2>/dev/null | head -n 1)
  disk="$source"
  if [ -n "$pk" ]; then disk="/dev/$pk"; fi
  parent=$(lsblk --nodeps --noheadings --pairs --output PATH,KNAME,TYPE,ROTA,PKNAME,TRAN "$disk" 2>/dev/null)
  parent_rc=$?
else
  block_rc=1
  parent_rc=1
fi
printf 'FAK_STRIX_FS_MOUNT=%s\n' "$mount"
printf 'FAK_STRIX_FS_BLOCK=%s\n' "$block"
printf 'FAK_STRIX_FS_PARENT=%s\n' "$parent"
printf 'FAK_STRIX_FS_EXIT=%d,%d,%d\n' "$rc" "$block_rc" "$parent_rc"
exit 0`

var strixPairsRE = regexp.MustCompile(`([A-Z][A-Z0-9:]*)="([^"\\]*)"`)

func observeStrixSSDFileSystem(ctx context.Context, target *StrixTarget, workspace string) (StrixSSDFileSystemObservation, error) {
	if err := validateStrixWorkspace(workspace); err != nil {
		return StrixSSDFileSystemObservation{}, err
	}
	command := "bash -c " + shellQuote(strixSSDFileSystemScript) + " -- " + shellQuote(workspace)
	out, err := runStrixTargetCommand(ctx, target, command, nil)
	if err != nil {
		return StrixSSDFileSystemObservation{}, errors.New("pinned SSD filesystem observation transport failed")
	}
	return ParseStrixSSDFileSystemObservation(workspace, out)
}

func ParseStrixSSDFileSystemObservation(workspace string, out []byte) (StrixSSDFileSystemObservation, error) {
	if err := validateStrixWorkspace(workspace); err != nil {
		return StrixSSDFileSystemObservation{}, err
	}
	if len(out) == 0 || len(out) > 8<<10 {
		return StrixSSDFileSystemObservation{}, errors.New("SSD filesystem observation is empty or exceeds its bound")
	}
	if markerValue(string(out), "FAK_STRIX_FS_EXIT=") != "0,0,0" {
		return StrixSSDFileSystemObservation{}, errors.New("findmnt or lsblk could not resolve the workspace backing device")
	}
	mountLine := markerValue(string(out), "FAK_STRIX_FS_MOUNT=")
	blockLine := markerValue(string(out), "FAK_STRIX_FS_BLOCK=")
	parentLine := markerValue(string(out), "FAK_STRIX_FS_PARENT=")
	if strings.Count(string(out), "FAK_STRIX_FS_MOUNT=") != 1 || strings.Count(string(out), "FAK_STRIX_FS_BLOCK=") != 1 || strings.Count(string(out), "FAK_STRIX_FS_PARENT=") != 1 || strings.Count(string(out), "FAK_STRIX_FS_EXIT=") != 1 {
		return StrixSSDFileSystemObservation{}, errors.New("SSD filesystem observation markers are missing or ambiguous")
	}
	mount, err := parseStrixPairs(mountLine)
	if err != nil {
		return StrixSSDFileSystemObservation{}, err
	}
	block, err := parseStrixPairs(blockLine)
	if err != nil {
		return StrixSSDFileSystemObservation{}, err
	}
	parent, err := parseStrixPairs(parentLine)
	if err != nil {
		return StrixSSDFileSystemObservation{}, err
	}
	rotation, err := strconv.Atoi(parent["ROTA"])
	if err != nil || (rotation != 0 && rotation != 1) {
		return StrixSSDFileSystemObservation{}, errors.New("SSD filesystem rotation observation is invalid")
	}
	observation := StrixSSDFileSystemObservation{
		Workspace: workspace, MountTarget: mount["TARGET"], Source: mount["SOURCE"], FSType: mount["FSTYPE"], MajorMinor: mount["MAJ:MIN"],
		SourcePath: block["PATH"], SourceType: block["TYPE"], ParentDisk: parent["PATH"], ParentType: parent["TYPE"], Transport: parent["TRAN"],
		Rotational: rotation == 1, RotationObserved: true, ObservationSHA256: digestBytes([]byte(mountLine + "\n" + blockLine + "\n" + parentLine)),
	}
	if err := observation.Validate(); err != nil {
		return StrixSSDFileSystemObservation{}, err
	}
	return observation, nil
}

func parseStrixPairs(line string) (map[string]string, error) {
	fields := map[string]string{}
	for _, match := range strixPairsRE.FindAllStringSubmatch(line, -1) {
		if _, duplicate := fields[match[1]]; duplicate {
			return nil, fmt.Errorf("SSD filesystem observation repeats %s", match[1])
		}
		fields[match[1]] = match[2]
	}
	if len(fields) == 0 || strings.TrimSpace(strixPairsRE.ReplaceAllString(line, "")) != "" {
		return nil, errors.New("SSD filesystem observation has malformed key/value fields")
	}
	return fields, nil
}

func (f StrixSSDFileSystemObservation) Validate() error {
	fsType := strings.ToLower(f.FSType)
	denied := map[string]bool{"tmpfs": true, "ramfs": true, "devtmpfs": true, "overlay": true, "squashfs": true}
	workspaceOnMount := f.Workspace == f.MountTarget || f.MountTarget == "/" && strings.HasPrefix(f.Workspace, "/") || strings.HasPrefix(f.Workspace, strings.TrimSuffix(f.MountTarget, "/")+"/")
	sourceType := strings.ToLower(f.SourceType)
	if validateStrixWorkspace(f.Workspace) != nil || f.MountTarget == "" || !workspaceOnMount || denied[fsType] || !strings.HasPrefix(f.Source, "/dev/") || f.MajorMinor == "" || f.SourcePath == "" || !strings.HasPrefix(f.SourcePath, "/dev/") || (sourceType != "part" && sourceType != "disk" && sourceType != "lvm" && sourceType != "crypt") {
		return errors.New("workspace is not on a supported block-backed filesystem")
	}
	if f.ParentType != "disk" || !strings.HasPrefix(f.ParentDisk, "/dev/nvme") || strings.ToLower(f.Transport) != "nvme" || !f.RotationObserved || f.Rotational || !sha256RE.MatchString(f.ObservationSHA256) {
		return errors.New("workspace backing disk is not measured nonrotating NVMe")
	}
	return nil
}

const strixVulkanSummaryScript = `set +e
timeout --signal=TERM --kill-after=2s 10s vulkaninfo --summary 2>&1 | head -c 32769
rc=${PIPESTATUS[0]}
printf '\nFAK_STRIX_VULKANINFO_EXIT=%d\n' "$rc"
exit 0`

func observeStrixVulkanDevice(ctx context.Context, target *StrixTarget) (StrixVulkanDeviceObservation, error) {
	out, err := runStrixTargetCommand(ctx, target, "bash -c "+shellQuote(strixVulkanSummaryScript), nil)
	if err != nil {
		return StrixVulkanDeviceObservation{}, errors.New("pinned Vulkan device observation transport failed")
	}
	if len(out) > 33<<10 {
		return StrixVulkanDeviceObservation{}, errors.New("vulkaninfo summary exceeded the observation bound")
	}
	rawExit := markerValue(string(out), "FAK_STRIX_VULKANINFO_EXIT=")
	exitCode, parseErr := strconv.Atoi(rawExit)
	if parseErr != nil {
		return StrixVulkanDeviceObservation{}, errors.New("vulkaninfo exit marker missing")
	}
	marker := []byte("\nFAK_STRIX_VULKANINFO_EXIT=" + rawExit)
	idx := bytes.LastIndex(out, marker)
	if idx < 0 || bytes.Count(out, []byte("FAK_STRIX_VULKANINFO_EXIT=")) != 1 {
		return StrixVulkanDeviceObservation{}, errors.New("vulkaninfo exit marker is malformed")
	}
	summary := bytes.TrimSpace(out[:idx])
	if exitCode != 0 {
		switch exitCode {
		case 127:
			return StrixVulkanDeviceObservation{}, errors.New("vulkaninfo is unavailable; install the target's native vulkan-tools package and retry")
		case 124, 137:
			return StrixVulkanDeviceObservation{}, errors.New("vulkaninfo summary timed out")
		case 141:
			return StrixVulkanDeviceObservation{}, errors.New("vulkaninfo summary exceeded the observation bound")
		default:
			return StrixVulkanDeviceObservation{}, fmt.Errorf("vulkaninfo summary failed with exit code %d", exitCode)
		}
	}
	return ParseStrixVulkanInfoSummary(summary)
}

func ParseStrixVulkanInfoSummary(summary []byte) (StrixVulkanDeviceObservation, error) {
	if len(summary) == 0 || len(summary) > 32<<10 {
		return StrixVulkanDeviceObservation{}, errors.New("vulkaninfo summary is empty or exceeds the observation bound")
	}
	type fields map[string]string
	var devices []fields
	var current fields
	for _, raw := range bytes.Split(summary, []byte{'\n'}) {
		line := strings.TrimSpace(string(raw))
		if isVulkanSummaryGPUHeader(line) {
			current = fields{}
			devices = append(devices, current)
			continue
		}
		if current == nil {
			continue
		}
		parts := strings.SplitN(line, "=", 2)
		if len(parts) != 2 {
			continue
		}
		key, value := strings.TrimSpace(parts[0]), strings.TrimSpace(parts[1])
		switch key {
		case "deviceName", "deviceType", "vendorID", "driverID", "driverName", "driverInfo":
			if _, duplicate := current[key]; duplicate {
				return StrixVulkanDeviceObservation{}, fmt.Errorf("vulkaninfo summary repeats %s", key)
			}
			current[key] = value
		}
	}
	if len(devices) != 1 {
		return StrixVulkanDeviceObservation{}, fmt.Errorf("vulkaninfo reported %d devices; exactly one is required", len(devices))
	}
	d := devices[0]
	observation := StrixVulkanDeviceObservation{Name: d["deviceName"], DeviceType: d["deviceType"], VendorID: d["vendorID"], DriverID: d["driverID"], DriverName: d["driverName"], DriverInfo: d["driverInfo"], ISA: "gfx1151", SummarySHA256: digestBytes(summary), SummaryBytes: len(summary)}
	if err := observation.Validate(); err != nil {
		return StrixVulkanDeviceObservation{}, err
	}
	return observation, nil
}

func (d StrixVulkanDeviceObservation) Validate() error {
	name := strings.ToLower(d.Name)
	driver := strings.ToLower(strings.Join([]string{d.DriverID, d.DriverName, d.DriverInfo}, "|"))
	typeName := strings.ToUpper(d.DeviceType)
	vendor := strings.ToLower(strings.TrimSpace(d.VendorID))
	identity := name + "|" + driver
	if d.Name == "" || d.DriverID == "" || d.DriverName == "" || d.DriverInfo == "" || typeName != "PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU" || (!strings.HasSuffix(vendor, "1002") && vendor != "amd") || !strings.Contains(name, "amd") || (!strings.Contains(name, "gfx1151") && !strings.Contains(name, "strix")) || !strings.Contains(driver, "radv") || strings.Contains(identity, "llvmpipe") || strings.Contains(identity, "lavapipe") || strings.Contains(identity, "software") || strings.Contains(typeName, "CPU") {
		return errors.New("vulkaninfo did not identify one physical integrated RADV Strix Halo GPU")
	}
	if d.ISA != "gfx1151" || !sha256RE.MatchString(d.SummarySHA256) || d.SummaryBytes <= 0 || d.SummaryBytes > 32<<10 {
		return errors.New("Vulkan device observation provenance is incomplete")
	}
	return nil
}

func isVulkanSummaryGPUHeader(line string) bool {
	if len(line) < 5 || !strings.HasPrefix(line, "GPU") || line[len(line)-1] != ':' {
		return false
	}
	for _, c := range line[3 : len(line)-1] {
		if c < '0' || c > '9' {
			return false
		}
	}
	return true
}

func parseStrixSSDRestartWitness(out []byte) (*StrixSSDRestartWitness, error) {
	var payload []byte
	for _, line := range bytes.Split(out, []byte{'\n'}) {
		line = bytes.TrimSpace(line)
		if !bytes.HasPrefix(line, []byte(ssdRestartMarker)) {
			continue
		}
		if payload != nil {
			return nil, errors.New("multiple native SSD restart receipts")
		}
		payload = bytes.TrimPrefix(line, []byte(ssdRestartMarker))
	}
	if len(payload) == 0 {
		return nil, errors.New("native SSD restart receipt marker missing")
	}
	if err := rejectDuplicateJSONFields(payload); err != nil {
		return nil, err
	}
	var witness StrixSSDRestartWitness
	dec := json.NewDecoder(bytes.NewReader(payload))
	dec.DisallowUnknownFields()
	if err := dec.Decode(&witness); err != nil {
		return nil, fmt.Errorf("decode native SSD restart receipt: %w", err)
	}
	if err := dec.Decode(&struct{}{}); err != io.EOF {
		return nil, errors.New("native SSD restart receipt has trailing JSON")
	}
	return &witness, nil
}

func validateStrixSSDRestartWitness(w *StrixSSDRestartWitness, revision, archiveDigest, binaryDigest, targetDevice string) error {
	if w == nil {
		return errors.New("native SSD restart witness is nil")
	}
	archive := strings.TrimPrefix(archiveDigest, "sha256:")
	binary := strings.TrimPrefix(binaryDigest, "sha256:")
	if w.Schema != "fak-vulkan-warm-disk-restart/1" || w.PID <= 0 || w.CacheMode != "isolated-default" || w.SourceRevision != revision || !isLowerHex(w.SourceRevision, 40) || w.SourceArchiveSHA256 != archive || !isLowerHex(w.SourceArchiveSHA256, 64) || w.SourceModified || w.ExecutableSHA256 != binary || !isLowerHex(w.ExecutableSHA256, 64) || !strings.HasPrefix(w.ModelIdentity, "sha256:") || !isLowerHex(strings.TrimPrefix(w.ModelIdentity, "sha256:"), 64) {
		return errors.New("native SSD restart parent identity mismatch")
	}
	if !w.TokenParity || !w.LogitParity || !w.ContinuationLogitParity || w.ExpectedKVTransferBytes == 0 || w.ExpectedColdD2HBytes == 0 || w.ExpectedKVTransferCount == 0 {
		return errors.New("native SSD restart parity or transfer expectation missing")
	}
	for phase, child := range map[string]StrixSSDRestartChild{"cold": w.Cold, "warm": w.Warm} {
		if child.Schema != w.Schema || child.Phase != phase || child.PID <= 0 || child.PID == w.PID || child.SourceRevision != revision || child.SourceArchiveSHA256 != archive || child.SourceModified || child.ExecutableSHA256 != binary || child.ModelIdentity != w.ModelIdentity || child.Backend != "vulkan" || child.Device != targetDevice || child.Driver == "" || child.Runtime == "" || child.DiskTier != "local_ssd_l3" || child.Stable <= 0 || child.Matched < child.Stable || len(child.Generated) != 4 || !isLowerHex(child.LogitsDigest, 64) || !isLowerHex(child.ContinuationLogitsDigest, 64) {
			return fmt.Errorf("native SSD restart %s child identity mismatch", phase)
		}
	}
	if w.Cold.PID == w.Warm.PID || w.Cold.Device != w.Warm.Device || w.Cold.Driver != w.Warm.Driver || w.Cold.Runtime != w.Warm.Runtime || w.Cold.Stable != w.Warm.Stable || !equalSSDTokenIDs(w.Cold.Generated, w.Warm.Generated) || w.Cold.LogitsDigest != w.Warm.LogitsDigest || w.Cold.ContinuationLogitsDigest != w.Warm.ContinuationLogitsDigest {
		return errors.New("native SSD restart cross-process parity mismatch")
	}
	if w.Cold.DiskOutcome != "persisted" || w.Cold.WriteBytes <= 0 || w.Cold.Prefilled != w.Cold.Stable || w.Warm.DiskOutcome != "restored" || w.Warm.ReadBytes <= 0 || w.Warm.Prefilled != 0 {
		return errors.New("native SSD restart disk reuse mismatch")
	}
	cold, warm := w.Cold.Transfers, w.Warm.Transfers
	if cold.D2HBytes != w.ExpectedColdD2HBytes || cold.D2HCount != w.ExpectedKVTransferCount+1 || cold.ComputeDispatches == 0 || cold.DispatchSubmits == 0 || warm.H2DBytes != w.ExpectedKVTransferBytes || warm.H2DCount != w.ExpectedKVTransferCount || warm.D2HBytes != 0 || warm.D2HCount != 0 || warm.ComputeDispatches != 0 || warm.DispatchSubmits == 0 {
		return errors.New("native SSD restart transfer counters mismatch")
	}
	return nil
}

func rejectDuplicateJSONFields(raw []byte) error {
	dec := json.NewDecoder(bytes.NewReader(raw))
	var walk func() error
	walk = func() error {
		token, err := dec.Token()
		if err != nil {
			return err
		}
		delim, ok := token.(json.Delim)
		if !ok {
			return nil
		}
		switch delim {
		case '{':
			seen := map[string]struct{}{}
			for dec.More() {
				keyToken, err := dec.Token()
				if err != nil {
					return err
				}
				key, ok := keyToken.(string)
				if !ok {
					return errors.New("native SSD restart JSON object has a non-string key")
				}
				if _, exists := seen[key]; exists {
					return fmt.Errorf("native SSD restart JSON contains duplicate field %q", key)
				}
				seen[key] = struct{}{}
				if err := walk(); err != nil {
					return err
				}
			}
			_, err = dec.Token()
			return err
		case '[':
			for dec.More() {
				if err := walk(); err != nil {
					return err
				}
			}
			_, err = dec.Token()
			return err
		default:
			return errors.New("native SSD restart JSON has an unexpected delimiter")
		}
	}
	if err := walk(); err != nil {
		return err
	}
	if _, err := dec.Token(); err != io.EOF {
		return errors.New("native SSD restart JSON has trailing content")
	}
	return nil
}

func safeStrixSSDRestartFailure(out []byte, exitCode *int) string {
	for _, line := range bytes.Split(out, []byte{'\n'}) {
		text := strings.TrimSpace(string(line))
		if strings.Contains(text, "inkernel_warm_disk_vulkan_test.go:") {
			return truncateOutput(text, 400)
		}
	}
	if exitCode == nil {
		return "native SSD restart device exit marker missing"
	}
	return fmt.Sprintf("native SSD restart device test exited with code %d", *exitCode)
}

func equalSSDTokenIDs(a, b []int) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}
