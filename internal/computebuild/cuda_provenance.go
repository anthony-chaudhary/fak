package computebuild

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"strconv"
	"strings"
	"time"
)

// ErrCUDAArchMissing reports that a gencode target derived from cuda_arch.txt
// has no matching image in the built artifact. A build that claims an
// architecture without carrying its SASS or PTX image fails with this sentinel.
var ErrCUDAArchMissing = errors.New("computebuild: requested CUDA architecture is missing from the built artifact")

// ErrCUDAProvenanceInvalid reports that the CUDA provenance block could not be
// observed: cuobjdump is missing beside the selected nvcc, a listing or the
// nvcc version banner is unparseable, or the build-time cuBLAS header cannot
// be read. The build never writes a CUDA block it could not observe.
var ErrCUDAProvenanceInvalid = errors.New("computebuild: CUDA build provenance could not be observed")

// Artifact kinds carried by CUDABuildProvenance.ArtifactKind.
const (
	CUDAProvenanceArtifactBinary  = "binary"
	CUDAProvenanceArtifactLibrary = "libfakcuda"
)

// CUDANVCCIdentity binds the nvcc driver that compiled the artifact to its
// version banner and content digest. Executable is retained without any host
// path, following BuildToolIdentity.
type CUDANVCCIdentity struct {
	Version string `json:"version"`
	BuildToolIdentity
}

// CUDABuildProvenance carries the observed evidence that a successful CUDA
// build actually contains the images cuda_arch.txt requested. It is attached
// to the receipt only on success; failed receipts keep the base schema and
// omit the block entirely.
type CUDABuildProvenance struct {
	ArtifactKind   string           `json:"artifact_kind"`
	ArtifactSHA256 string           `json:"artifact_sha256"`
	RequestedArchs []string         `json:"requested_archs"`
	SASSArchs      []string         `json:"sass_archs"`
	PTXArchs       []string         `json:"ptx_archs"`
	NVCC           CUDANVCCIdentity `json:"nvcc"`
	CUBLASVersion  string           `json:"cublas_version"`
}

// cudaBuildReceiptV1Wire is the explicit successor envelope for CUDA receipts.
// The common JSON type keeps the block behind json:"-" so historical decoders
// and the Vulkan v2 unknown-field rejection never see the field.
type cudaBuildReceiptV1Wire struct {
	computeBuildReceiptV2Wire
	CUDA *CUDABuildProvenance `json:"cuda,omitempty"`
}

// DecodeStrictCUDABuildReceipt decodes a fak.cuda-build-receipt.v1 receipt with
// unknown-field and trailing-data rejection. The schema must be the CUDA
// successor and the block must be present; anything else is not this schema.
func DecodeStrictCUDABuildReceipt(raw []byte) (ComputeBuildReceipt, error) {
	dec := json.NewDecoder(bytes.NewReader(raw))
	dec.DisallowUnknownFields()
	var wire cudaBuildReceiptV1Wire
	if err := dec.Decode(&wire); err != nil {
		return ComputeBuildReceipt{}, fmt.Errorf("decode CUDA build receipt: %w", err)
	}
	if err := dec.Decode(&struct{}{}); err != io.EOF {
		return ComputeBuildReceipt{}, fmt.Errorf("decode CUDA build receipt: trailing JSON data")
	}
	receipt := ComputeBuildReceipt(wire.computeBuildReceiptV2Wire)
	receipt.CUDA = wire.CUDA
	if receipt.Schema != CUDABuildReceiptSchema {
		return ComputeBuildReceipt{}, fmt.Errorf("CUDA build receipt schema mismatch: want %s", CUDABuildReceiptSchema)
	}
	if receipt.CUDA == nil {
		return ComputeBuildReceipt{}, fmt.Errorf("CUDA build receipt requires the cuda block")
	}
	return receipt, nil
}

// cudaCommandRunner runs one external command. RunCmd is the production seam;
// tests substitute a stub so cuobjdump and nvcc never need a real toolkit.
type cudaCommandRunner func(ctx context.Context, name string, args []string, dir string, env []string, stdout, stderr io.Writer) error

// observeCUDABuildProvenance reads the built artifact back with cuobjdump and
// binds it to the requested gencode targets, the nvcc driver, and the
// build-time cuBLAS header. It fails closed: any step that cannot be observed
// returns an error instead of a partial block.
func observeCUDABuildProvenance(ctx context.Context, tc *Toolchain, repoRoot, arch, artifactPath, artifactKind string, run cudaCommandRunner) (*CUDABuildProvenance, error) {
	if artifactKind != CUDAProvenanceArtifactBinary && artifactKind != CUDAProvenanceArtifactLibrary {
		return nil, fmt.Errorf("%w: unknown artifact kind %q", ErrCUDAProvenanceInvalid, artifactKind)
	}
	if tc == nil || tc.NVCC == "" {
		return nil, fmt.Errorf("%w: no selected nvcc", ErrCUDAProvenanceInvalid)
	}
	if !FileExists(artifactPath) {
		return nil, fmt.Errorf("%w: artifact %s does not exist", ErrCUDAProvenanceInvalid, artifactPath)
	}
	absArtifact, err := filepath.Abs(artifactPath)
	if err != nil {
		return nil, fmt.Errorf("%w: %v", ErrCUDAProvenanceInvalid, err)
	}
	artifactSHA, _, err := stableRegularFileSHA256(absArtifact, "CUDA artifact")
	if err != nil {
		return nil, fmt.Errorf("%w: %v", ErrCUDAProvenanceInvalid, err)
	}

	gencodeArgs, _, err := LoadCUDAArchMatrix(repoRoot, arch)
	if err != nil {
		return nil, fmt.Errorf("%w: %v", ErrCUDAProvenanceInvalid, err)
	}
	requested, err := requestedCUDAGencodeTargets(gencodeArgs)
	if err != nil {
		return nil, err
	}

	tool, err := cudaImageAuditTool(tc)
	if err != nil {
		return nil, fmt.Errorf("%w: %v", ErrCUDAProvenanceInvalid, err)
	}
	sass, err := cudaProvenanceListing(ctx, run, tool, "--list-elf", absArtifact)
	if err != nil {
		return nil, err
	}
	ptx, err := cudaProvenanceListing(ctx, run, tool, "--list-ptx", absArtifact)
	if err != nil {
		return nil, err
	}
	if missing := checkRequestedCUDAImages(requested, sass, ptx); len(missing) > 0 {
		return nil, fmt.Errorf("%w: %s", ErrCUDAArchMissing, strings.Join(missing, "; "))
	}

	nvcc, err := observeCUDANVCC(ctx, tc, run)
	if err != nil {
		return nil, err
	}
	cublas, err := cublasHeaderVersion(tc)
	if err != nil {
		return nil, err
	}
	return &CUDABuildProvenance{
		ArtifactKind:   artifactKind,
		ArtifactSHA256: artifactSHA,
		RequestedArchs: requested,
		SASSArchs:      sass,
		PTXArchs:       ptx,
		NVCC:           nvcc,
		CUBLASVersion:  cublas,
	}, nil
}

// attachCUDABuildProvenance observes the artifact through the production
// command seam and switches the receipt to the CUDA successor schema. It is
// all-or-nothing: on error the receipt keeps the base schema and no block.
func attachCUDABuildProvenance(ctx context.Context, receipt *ComputeBuildReceipt, tc *Toolchain, repoRoot, arch, artifactPath, artifactKind string, stdout io.Writer) error {
	provenance, err := observeCUDABuildProvenance(ctx, tc, repoRoot, arch, artifactPath, artifactKind, RunCmd)
	if err != nil {
		return err
	}
	receipt.CUDA = provenance
	receipt.Schema = CUDABuildReceiptSchema
	if stdout != nil {
		fmt.Fprintf(stdout, "[cuda] provenance: %s sha256=%.12s requested=%v SASS=%v PTX=%v\n",
			artifactKind, provenance.ArtifactSHA256, provenance.RequestedArchs, provenance.SASSArchs, provenance.PTXArchs)
	}
	return nil
}

// requestedCUDAGencodeTargets extracts the deduplicated, numerically sorted
// image targets (the code= values) from ParseCUDAArchMatrix gencode arguments.
func requestedCUDAGencodeTargets(gencodeArgs []string) ([]string, error) {
	set := make(map[string]bool)
	var targets []string
	for i := 1; i < len(gencodeArgs); i += 2 {
		flag := gencodeArgs[i]
		_, code, ok := strings.Cut(flag, ",code=")
		if !ok || (!strings.HasPrefix(code, "sm_") && !strings.HasPrefix(code, "compute_")) {
			return nil, fmt.Errorf("%w: invalid gencode declaration %q", ErrCUDAProvenanceInvalid, flag)
		}
		if !set[code] {
			set[code] = true
			targets = append(targets, code)
		}
	}
	if len(targets) == 0 {
		return nil, fmt.Errorf("%w: no gencode targets declared", ErrCUDAProvenanceInvalid)
	}
	sortCUDAImageArchitectures(targets)
	return targets, nil
}

// cudaProvenanceListing captures one bounded cuobjdump listing through the
// injected command seam and parses its image architectures.
func cudaProvenanceListing(ctx context.Context, run cudaCommandRunner, tool, flag, archive string) ([]string, error) {
	ctx, cancel := context.WithTimeout(ctx, 30*time.Second)
	defer cancel()
	out := &cudaImageOutput{limit: cudaImageListingLimit}
	diagnostic := &cudaImageOutput{limit: 4096}
	if err := run(ctx, tool, []string{flag, archive}, "", nil, out, diagnostic); err != nil {
		return nil, fmt.Errorf("%w: cuobjdump %s: %v (%s)", ErrCUDAProvenanceInvalid, flag, err, strings.TrimSpace(string(diagnostic.data)))
	}
	if out.exceeded || diagnostic.exceeded {
		return nil, fmt.Errorf("%w: cuobjdump %s output exceeds limit", ErrCUDAProvenanceInvalid, flag)
	}
	kind := "ELF"
	if flag == "--list-ptx" {
		kind = "PTX"
	}
	return parseCUDAProvenanceListing(out.data, kind)
}

// parseCUDAProvenanceListing extracts the image architectures from one
// cuobjdump listing. SASS entries keep their sm_<cc> identity with the
// optional 'a' or 'f' suffix; PTX entries are recorded as compute_<cc> because
// that is the requested PTX target form. Every list is sorted by numeric
// compute capability, then suffix.
func parseCUDAProvenanceListing(data []byte, kind string) ([]string, error) {
	set := make(map[string]bool)
	scanner := bufio.NewScanner(strings.NewReader(string(data)))
	for scanner.Scan() {
		line := strings.TrimSpace(scanner.Text())
		if line == "" || strings.HasPrefix(line, "File archived in ") || strings.HasPrefix(line, "cuobjdump info ") || (strings.HasPrefix(line, "member ") && strings.HasSuffix(line, ":")) {
			continue
		}
		parts := cudaImageListing.FindStringSubmatch(line)
		if parts == nil || parts[1] != kind || (kind == "ELF" && parts[4] != "cubin") || (kind == "PTX" && parts[4] != "ptx") {
			return nil, fmt.Errorf("%w: malformed %s listing line %q", ErrCUDAProvenanceInvalid, kind, line)
		}
		arch := parts[3]
		if kind == "PTX" {
			arch = "compute_" + strings.TrimPrefix(arch, "sm_")
		}
		set[arch] = true
	}
	if err := scanner.Err(); err != nil {
		return nil, fmt.Errorf("%w: reading %s listing: %v", ErrCUDAProvenanceInvalid, kind, err)
	}
	arches := make([]string, 0, len(set))
	for arch := range set {
		arches = append(arches, arch)
	}
	sortCUDAImageArchitectures(arches)
	return arches, nil
}

// checkRequestedCUDAImages reports each requested image target absent from the
// observed listings, classed as SASS (sm_*) or PTX (compute_*).
func checkRequestedCUDAImages(requested, sass, ptx []string) []string {
	haveSASS := make(map[string]bool, len(sass))
	for _, arch := range sass {
		haveSASS[arch] = true
	}
	havePTX := make(map[string]bool, len(ptx))
	for _, arch := range ptx {
		havePTX[arch] = true
	}
	var missing []string
	for _, target := range requested {
		switch {
		case strings.HasPrefix(target, "sm_"):
			if !haveSASS[target] {
				missing = append(missing, "missing SASS image "+target)
			}
		case strings.HasPrefix(target, "compute_"):
			if !havePTX[target] {
				missing = append(missing, "missing PTX image "+target)
			}
		default:
			missing = append(missing, "unclassified requested target "+target)
		}
	}
	return missing
}

// observeCUDANVCC records the selected nvcc driver's version banner and
// executable digest. The digest covers the exact bytes invoked; the
// executable name is retained without any host path.
func observeCUDANVCC(ctx context.Context, tc *Toolchain, run cudaCommandRunner) (CUDANVCCIdentity, error) {
	nvcc, err := exec.LookPath(tc.NVCC)
	if err != nil {
		return CUDANVCCIdentity{}, fmt.Errorf("%w: resolve nvcc: %v", ErrCUDAProvenanceInvalid, err)
	}
	absNVCC, err := filepath.Abs(nvcc)
	if err != nil {
		return CUDANVCCIdentity{}, fmt.Errorf("%w: %v", ErrCUDAProvenanceInvalid, err)
	}
	sha, _, err := stableRegularFileSHA256(absNVCC, "nvcc executable")
	if err != nil {
		return CUDANVCCIdentity{}, fmt.Errorf("%w: %v", ErrCUDAProvenanceInvalid, err)
	}
	var banner, diagnostic bytes.Buffer
	if err := run(ctx, absNVCC, []string{"--version"}, "", nil, &banner, &diagnostic); err != nil {
		return CUDANVCCIdentity{}, fmt.Errorf("%w: nvcc --version: %v (%s)", ErrCUDAProvenanceInvalid, err, strings.TrimSpace(diagnostic.String()))
	}
	version, err := parseCUDAVersion(banner.String())
	if err != nil {
		return CUDANVCCIdentity{}, err
	}
	return CUDANVCCIdentity{
		Version:           version,
		BuildToolIdentity: BuildToolIdentity{Role: "nvcc", Executable: filepath.Base(absNVCC), SHA256: sha},
	}, nil
}

var cudaVersionBanner = regexp.MustCompile(`Cuda compilation tools, release (\d+\.\d+(?:\.\d+)?)`)

// parseCUDAVersion extracts the toolkit release from an `nvcc --version` banner.
func parseCUDAVersion(banner string) (string, error) {
	match := cudaVersionBanner.FindStringSubmatch(banner)
	if match == nil {
		return "", fmt.Errorf("%w: unparseable nvcc --version banner", ErrCUDAProvenanceInvalid)
	}
	return match[1], nil
}

var (
	cublasPackedVersion = regexp.MustCompile(`(?m)^#define\s+CUBLASAPI_VERSION\s+(\d+)`)
	cublasVerMajor      = regexp.MustCompile(`(?m)^#define\s+CUBLAS_VER_MAJOR\s+(\d+)`)
	cublasVerMinor      = regexp.MustCompile(`(?m)^#define\s+CUBLAS_VER_MINOR\s+(\d+)`)
	cublasVerPatch      = regexp.MustCompile(`(?m)^#define\s+CUBLAS_VER_PATCH\s+(\d+)`)
	cublasVerBuild      = regexp.MustCompile(`(?m)^#define\s+CUBLAS_VER_BUILD\s+(\d+)`)
)

// cublasHeaderVersion reads the build-time cuBLAS version from cublas_api.h
// under the toolchain's CUDA include directories, falling back to the toolkit
// include directory derived from CUDAHome or the selected nvcc. This is the
// header the kernels were compiled against, not the runtime-loaded library.
func cublasHeaderVersion(tc *Toolchain) (string, error) {
	dirs := append([]string(nil), tc.CUDAInc...)
	addDir := func(dir string) {
		if dir == "" {
			return
		}
		for _, existing := range dirs {
			if filepath.Clean(existing) == filepath.Clean(dir) {
				return
			}
		}
		dirs = append(dirs, dir)
	}
	if tc.CUDAHome != "" {
		addDir(filepath.Join(tc.CUDAHome, "include"))
	}
	if nvcc, err := exec.LookPath(tc.NVCC); err == nil {
		addDir(filepath.Join(filepath.Dir(nvcc), "..", "include"))
	}
	for _, dir := range dirs {
		header := filepath.Join(dir, "cublas_api.h")
		if !FileExists(header) {
			continue
		}
		data, err := os.ReadFile(header)
		if err != nil {
			return "", fmt.Errorf("%w: read cublas_api.h: %v", ErrCUDAProvenanceInvalid, err)
		}
		version, ok := parseCUBLASHeaderVersion(string(data))
		if !ok {
			return "", fmt.Errorf("%w: cublas_api.h at %s declares no version", ErrCUDAProvenanceInvalid, header)
		}
		return version, nil
	}
	return "", fmt.Errorf("%w: cublas_api.h not found in CUDA include dirs", ErrCUDAProvenanceInvalid)
}

// parseCUBLASHeaderVersion prefers the packed CUBLASAPI_VERSION macro and falls
// back to the CUBLAS_VER_* component macros older headers use.
func parseCUBLASHeaderVersion(header string) (string, bool) {
	if match := cublasPackedVersion.FindStringSubmatch(header); match != nil {
		packed, err := strconv.Atoi(match[1])
		if err != nil {
			return "", false
		}
		return fmt.Sprintf("%d.%d.%d", packed/10000, (packed/100)%100, packed%100), true
	}
	major := cublasVerMajor.FindStringSubmatch(header)
	minor := cublasVerMinor.FindStringSubmatch(header)
	if major == nil || minor == nil {
		return "", false
	}
	version := major[1] + "." + minor[1]
	if patch := cublasVerPatch.FindStringSubmatch(header); patch != nil {
		version += "." + patch[1]
		if build := cublasVerBuild.FindStringSubmatch(header); build != nil {
			version += "." + build[1]
		}
	}
	return version, true
}
