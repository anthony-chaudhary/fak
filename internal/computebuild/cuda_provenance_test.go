package computebuild

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"io"
	"os"
	"path/filepath"
	"runtime"
	"slices"
	"strings"
	"testing"
)

// fak-test:runtime fast est=10ms
// fak-test:justify why=contract when=changed:internal/computebuild/**
func TestParseCUDAVersion(t *testing.T) {
	const banner = "nvcc: NVIDIA (R) Cuda compiler driver\nCopyright (c) 2005-2024 NVIDIA Corporation\nAll rights reserved.\n\nCuda compilation tools, release 12.6, V12.6.85\nBuilt on Tue_Oct__9_23:50:17_2024\nCuda compilel tools, release 12.6, V12.6.85\n"
	cases := []struct {
		name    string
		banner  string
		want    string
		wantErr bool
	}{
		{"release", "Cuda compilation tools, release 12.6, V12.6.85\n", "12.6", false},
		{"three-part", "junk before\nCuda compilation tools, release 13.4.1, V13.4.99\n", "13.4.1", false},
		{"full banner", banner, "12.6", false},
		{"empty", "", "", true},
		{"host compiler banner", "synthetic CUDA host compiler 1\n", "", true},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			got, err := parseCUDAVersion(tc.banner)
			if tc.wantErr {
				if err == nil || !errors.Is(err, ErrCUDAProvenanceInvalid) {
					t.Fatalf("got %q err=%v; want ErrCUDAProvenanceInvalid", got, err)
				}
				return
			}
			if err != nil || got != tc.want {
				t.Fatalf("got %q err=%v; want %q", got, err, tc.want)
			}
		})
	}
}

// fak-test:runtime fast est=10ms
// fak-test:justify why=contract when=changed:internal/computebuild/**
func TestRequestedCUDAGencodeTargets(t *testing.T) {
	root := t.TempDir()
	writeFixtureFile(t, filepath.Join(root, "internal", "compute", "cuda_arch.txt"), "sm_80\nsm_90\nsm_120\n")
	gencode, _, err := LoadCUDAArchMatrix(root, "")
	if err != nil {
		t.Fatal(err)
	}
	requested, err := requestedCUDAGencodeTargets(gencode)
	if err != nil {
		t.Fatal(err)
	}
	want := []string{"sm_80", "sm_90", "compute_120", "sm_120"}
	if !slices.Equal(requested, want) {
		t.Fatalf("requested=%v want=%v", requested, want)
	}

	single, _, err := LoadCUDAArchMatrix(root, "sm_120")
	if err != nil {
		t.Fatal(err)
	}
	targets, err := requestedCUDAGencodeTargets(single)
	if err != nil || !slices.Equal(targets, []string{"sm_120"}) {
		t.Fatalf("single arch targets=%v err=%v", targets, err)
	}

	dupRoot := t.TempDir()
	writeFixtureFile(t, filepath.Join(dupRoot, "internal", "compute", "cuda_arch.txt"), "sm_80\nsm_80\nsm_90\n")
	dupGencode, _, err := LoadCUDAArchMatrix(dupRoot, "")
	if err != nil {
		t.Fatal(err)
	}
	dup, err := requestedCUDAGencodeTargets(dupGencode)
	// Numeric capability first; ties break on the full string, so the
	// compute_90 PTX target sorts ahead of its sm_90 SASS sibling.
	if err != nil || !slices.Equal(dup, []string{"sm_80", "compute_90", "sm_90"}) {
		t.Fatalf("dedupe targets=%v err=%v", dup, err)
	}

	if _, err := requestedCUDAGencodeTargets([]string{"-gencode", "arch=compute_90"}); !errors.Is(err, ErrCUDAProvenanceInvalid) {
		t.Fatalf("malformed gencode err=%v", err)
	}
	if _, err := requestedCUDAGencodeTargets(nil); !errors.Is(err, ErrCUDAProvenanceInvalid) {
		t.Fatalf("empty gencode err=%v", err)
	}
}

// fak-test:runtime fast est=10ms
// fak-test:justify why=contract when=changed:internal/computebuild/**
func TestParseCUDAProvenanceListing(t *testing.T) {
	const elf = "member libfakcuda.a:k.o:\nELF file 1: k.sm_100f.cubin\nELF file 2: k.sm_100a.cubin\nELF file 3: k.sm_100.cubin\nELF file 4: k.sm_9.cubin\nELF file 5: k.sm_10.cubin\nELF file 6: k.sm_100.cubin\n"
	sass, err := parseCUDAProvenanceListing([]byte(elf), "ELF")
	if err != nil {
		t.Fatal(err)
	}
	wantSASS := []string{"sm_9", "sm_10", "sm_100", "sm_100a", "sm_100f"}
	if !slices.Equal(sass, wantSASS) {
		t.Fatalf("sass=%v want=%v", sass, wantSASS)
	}

	const ptx = "PTX file 1: k.sm_120.ptx\nPTX file 2: k.sm_90.ptx\n"
	ptxArchs, err := parseCUDAProvenanceListing([]byte(ptx), "PTX")
	if err != nil {
		t.Fatal(err)
	}
	if !slices.Equal(ptxArchs, []string{"compute_90", "compute_120"}) {
		t.Fatalf("ptx=%v want=[compute_90 compute_120]", ptxArchs)
	}

	empty, err := parseCUDAProvenanceListing([]byte(""), "PTX")
	if err != nil || len(empty) != 0 {
		t.Fatalf("empty listing=%v err=%v", empty, err)
	}

	for name, tc := range map[string]struct{ raw, kind string }{
		"garbage":       {"random text\n", "ELF"},
		"kind mismatch": {"ELF file 1: k.sm_100.cubin\n", "PTX"},
		"wrong ext":     {"PTX file 1: k.sm_100.cubin\n", "PTX"},
		"bad arch":      {"ELF file 1: k.x86_64.cubin\n", "ELF"},
	} {
		if _, err := parseCUDAProvenanceListing([]byte(tc.raw), tc.kind); !errors.Is(err, ErrCUDAProvenanceInvalid) {
			t.Fatalf("%s: accepted %q (err=%v)", name, tc.raw, err)
		}
	}
}

// fak-test:runtime fast est=10ms
// fak-test:justify why=contract when=changed:internal/computebuild/**
func TestCheckRequestedCUDAImages(t *testing.T) {
	requested := []string{"sm_80", "sm_90", "compute_90"}
	if missing := checkRequestedCUDAImages(requested, []string{"sm_80", "sm_90"}, []string{"compute_90"}); len(missing) != 0 {
		t.Fatalf("complete coverage reported missing=%v", missing)
	}
	missing := checkRequestedCUDAImages(requested, []string{"sm_80"}, []string{})
	if len(missing) != 2 || !strings.Contains(missing[0], "SASS image sm_90") || !strings.Contains(missing[1], "PTX image compute_90") {
		t.Fatalf("missing=%v", missing)
	}
}

// fak-test:runtime fast est=10ms
// fak-test:justify why=contract when=changed:internal/computebuild/**
func TestParseCUBLASHeaderVersion(t *testing.T) {
	cases := []struct {
		header string
		want   string
		ok     bool
	}{
		{"#define CUBLASAPI_VERSION 120503\n", "12.5.3", true},
		{"#define CUBLAS_VER_MAJOR 12\n#define CUBLAS_VER_MINOR 9\n#define CUBLAS_VER_PATCH 1\n#define CUBLAS_VER_BUILD 2\n", "12.9.1.2", true},
		{"#define CUBLAS_VER_MAJOR 12\n#define CUBLAS_VER_MINOR 9\n", "12.9", true},
		{"packed wins\n#define CUBLASAPI_VERSION 120900\n#define CUBLAS_VER_MAJOR 1\n#define CUBLAS_VER_MINOR 2\n", "12.9.0", true},
		{"no macros\n#define OTHER 1\n", "", false},
	}
	for _, tc := range cases {
		got, ok := parseCUBLASHeaderVersion(tc.header)
		if ok != tc.ok || got != tc.want {
			t.Fatalf("header %q -> (%q,%v) want (%q,%v)", tc.header, got, ok, tc.want, tc.ok)
		}
	}
}

// fak-test:runtime fast est=10ms
// fak-test:justify why=contract when=changed:internal/computebuild/**
func TestCUBLASHeaderVersionSearch(t *testing.T) {
	const header = "#define CUBLASAPI_VERSION 120901\n"
	incDir := t.TempDir()
	writeFixtureFile(t, filepath.Join(incDir, "cublas_api.h"), header)

	tc := &Toolchain{NVCC: "nvcc-not-on-path-but-unused", CUDAInc: []string{incDir}}
	version, err := cublasHeaderVersion(tc)
	if err != nil || version != "12.9.1" { //boundarylint:ignore CHANGE_DETECTOR_TEST decode contract: CUBLASAPI_VERSION macro 120901 decodes to major.minor.patch; fixed encoding, not a growable total
		t.Fatalf("explicit include dir: (%q,%v)", version, err)
	}

	homeDir := t.TempDir()
	writeFixtureFile(t, filepath.Join(homeDir, "include", "cublas_api.h"), "#define CUBLASAPI_VERSION 120600\n")
	fallback := &Toolchain{NVCC: "nvcc-not-on-path-but-unused", CUDAHome: homeDir}
	version, err = cublasHeaderVersion(fallback)
	if err != nil || version != "12.6.0" { //boundarylint:ignore CHANGE_DETECTOR_TEST decode contract: CUBLASAPI_VERSION macro 120600 decodes to major.minor.patch; fixed encoding, not a growable total
		t.Fatalf("CUDAHome fallback: (%q,%v)", version, err)
	}

	nowhere := &Toolchain{NVCC: "nvcc-not-on-path-but-unused", CUDAInc: []string{t.TempDir()}, CUDAHome: t.TempDir()}
	if _, err := cublasHeaderVersion(nowhere); !errors.Is(err, ErrCUDAProvenanceInvalid) {
		t.Fatalf("absent header err=%v", err)
	}

	noMacros := t.TempDir()
	writeFixtureFile(t, filepath.Join(noMacros, "cublas_api.h"), "// no version macros\n")
	if _, err := cublasHeaderVersion(&Toolchain{NVCC: "nvcc-not-on-path-but-unused", CUDAInc: []string{noMacros}}); !errors.Is(err, ErrCUDAProvenanceInvalid) {
		t.Fatalf("macro-less header err=%v", err)
	}
}

// fak-test:runtime fast est=10ms
// fak-test:justify why=contract when=changed:internal/computebuild/**
func TestDecodeStrictCUDABuildReceipt(t *testing.T) {
	receipt := ComputeBuildReceipt{
		Schema:      CUDABuildReceiptSchema,
		Backend:     "cuda",
		Command:     "build",
		Outcome:     "success",
		ReceiptPath: "/tmp/receipt.json",
		Phases: []ComputeBuildPhase{{
			Name: "provenance", Outcome: "success", ElapsedMS: 10,
		}},
		Artifact: &BuildArtifact{Path: "/tmp/fak", SHA256: strings.Repeat("0", 64)},
		CUDA: &CUDABuildProvenance{
			ArtifactKind:   CUDAProvenanceArtifactBinary,
			ArtifactSHA256: strings.Repeat("1", 64),
			RequestedArchs: []string{"sm_120"},
			SASSArchs:      []string{"sm_120"},
			PTXArchs:       []string{},
			NVCC:           CUDANVCCIdentity{Version: "12.6", BuildToolIdentity: BuildToolIdentity{Role: "nvcc", Executable: "nvcc", SHA256: strings.Repeat("2", 64)}},
			CUBLASVersion:  "12.6.0",
		},
	}
	raw, err := json.Marshal(receipt)
	if err != nil {
		t.Fatal(err)
	}
	if !bytes.Contains(raw, []byte(`"cuda":`)) {
		t.Fatalf("success receipt lost the cuda block: %s", raw)
	}
	decoded, err := DecodeStrictCUDABuildReceipt(raw)
	if err != nil {
		t.Fatal(err)
	}
	if decoded.Schema != CUDABuildReceiptSchema || decoded.CUDA == nil ||
		decoded.CUDA.ArtifactKind != CUDAProvenanceArtifactBinary ||
		!slices.Equal(decoded.CUDA.RequestedArchs, []string{"sm_120"}) ||
		decoded.CUDA.NVCC.Version != "12.6" || decoded.CUDA.CUBLASVersion != "12.6.0" { //boundarylint:ignore CHANGE_DETECTOR_TEST round-trip echo: the strict decoder must return the fixture receipt's nvcc/cublas versions verbatim
		t.Fatalf("round trip: %+v", decoded)
	}

	unknown := bytes.Replace(raw, []byte(`{"schema"`), []byte(`{"bogus_field":1,"schema"`), 1)
	if _, err := DecodeStrictCUDABuildReceipt(unknown); err == nil {
		t.Fatal("unknown field accepted")
	}
	trailing := append(append([]byte{}, raw...), []byte(" {}")...)
	if _, err := DecodeStrictCUDABuildReceipt(trailing); err == nil {
		t.Fatal("trailing data accepted")
	}
	wrongSchema := bytes.Replace(raw, []byte(CUDABuildReceiptSchema), []byte(ComputeBuildReceiptSchema), 1)
	if _, err := DecodeStrictCUDABuildReceipt(wrongSchema); err == nil {
		t.Fatal("base schema accepted as CUDA successor")
	}
	noBlock := `{"schema":"` + CUDABuildReceiptSchema + `","backend":"cuda","command":"build","outcome":"success","receipt_path":"/tmp/r.json","phases":[]}`
	if _, err := DecodeStrictCUDABuildReceipt([]byte(noBlock)); err == nil {
		t.Fatal("receipt without the cuda block accepted")
	}
}

// fak-test:runtime fast est=10ms
// fak-test:justify why=contract when=changed:internal/computebuild/**
func TestCUDABuildReceiptWireSelector(t *testing.T) {
	base := ComputeBuildReceipt{
		Schema: ComputeBuildReceiptSchema, Backend: "cuda", Command: "build",
		Outcome: "failed", Error: "boom", ReceiptPath: "/tmp/r.json",
	}
	failedRaw, err := json.Marshal(base)
	if err != nil {
		t.Fatal(err)
	}
	if bytes.Contains(failedRaw, []byte(`"cuda":`)) {
		t.Fatalf("failed receipt leaked the cuda block: %s", failedRaw)
	}
	if !bytes.Contains(failedRaw, []byte(ComputeBuildReceiptSchema)) {
		t.Fatalf("failed receipt lost its base schema: %s", failedRaw)
	}

	success := base
	success.Schema = CUDABuildReceiptSchema
	success.Outcome = "success"
	success.Error = ""
	success.CUDA = &CUDABuildProvenance{ArtifactKind: CUDAProvenanceArtifactLibrary, ArtifactSHA256: strings.Repeat("3", 64)}
	successRaw, err := json.Marshal(success)
	if err != nil {
		t.Fatal(err)
	}
	if !bytes.Contains(successRaw, []byte(CUDABuildReceiptSchema)) || !bytes.Contains(successRaw, []byte(`"cuda":`)) {
		t.Fatalf("success receipt missed schema or block: %s", successRaw)
	}
}

type cudaProvenanceFixture struct {
	repoRoot  string
	binDir    string
	nvcc      string
	artifact  string
	include   string
	toolchain *Toolchain
}

// newCUDAProvenanceFixture lays out a GPU-free repo with an nvcc/cuobjdump
// sibling pair (copies of the test binary, never executed because every call
// goes through the injected runner seam) and a built artifact to observe.
func newCUDAProvenanceFixture(t *testing.T, archs string) *cudaProvenanceFixture {
	t.Helper()
	repoRoot := t.TempDir()
	writeFixtureFile(t, filepath.Join(repoRoot, "internal", "compute", "cuda_arch.txt"), archs)
	include := filepath.Join(repoRoot, "toolkit", "include")
	writeFixtureFile(t, filepath.Join(include, "cublas_api.h"), "#define CUBLASAPI_VERSION 120600\n")

	self, err := os.Executable()
	if err != nil {
		t.Fatal(err)
	}
	raw, err := os.ReadFile(self)
	if err != nil {
		t.Fatal(err)
	}
	binDir := filepath.Join(repoRoot, "bin")
	if err := os.MkdirAll(binDir, 0o755); err != nil {
		t.Fatal(err)
	}
	suffix := ""
	if runtime.GOOS == "windows" {
		suffix = ".exe"
	}
	nvcc := filepath.Join(binDir, "nvcc"+suffix)
	cuobjdump := filepath.Join(binDir, "cuobjdump"+suffix)
	for _, tool := range []string{nvcc, cuobjdump} {
		if err := os.WriteFile(tool, raw, 0o755); err != nil {
			t.Fatal(err)
		}
	}
	artifact := filepath.Join(repoRoot, "built-fak"+suffix)
	if err := os.WriteFile(artifact, []byte("observed-artifact-bytes"), 0o600); err != nil {
		t.Fatal(err)
	}
	return &cudaProvenanceFixture{
		repoRoot:  repoRoot,
		binDir:    binDir,
		nvcc:      nvcc,
		artifact:  artifact,
		include:   include,
		toolchain: &Toolchain{NVCC: nvcc, CUDAInc: []string{include}, CUDAHome: filepath.Join(repoRoot, "toolkit")},
	}
}

// stubCUDARunner answers cuobjdump listings and the nvcc version banner from
// the table; anything else is a runner error.
func stubCUDARunner(listings map[string]string, version string, failTool string) cudaCommandRunner {
	return func(ctx context.Context, name string, args []string, dir string, env []string, stdout, stderr io.Writer) error {
		if failTool != "" && slices.Contains(args, failTool) {
			return errors.New("fixture command failure")
		}
		switch {
		case len(args) > 0 && args[0] == "--version":
			_, _ = io.WriteString(stdout, version)
		case len(args) > 0 && (args[0] == "--list-elf" || args[0] == "--list-ptx"):
			_, _ = io.WriteString(stdout, listings[args[0]])
		default:
			return errors.New("unexpected fixture invocation: " + strings.Join(args, " "))
		}
		return nil
	}
}

// fak-test:runtime fast est=10ms
// fak-test:justify why=regression when=changed:internal/computebuild/**
func TestObserveCUDABuildProvenanceStubbed(t *testing.T) {
	fx := newCUDAProvenanceFixture(t, "sm_80\nsm_90\nsm_120\n")
	sha, err := sha256OfFile(t, fx.artifact)
	if err != nil {
		t.Fatal(err)
	}
	run := stubCUDARunner(map[string]string{
		"--list-elf": "ELF file 1: k.sm_80.cubin\nELF file 2: k.sm_90.cubin\nELF file 3: k.sm_120.cubin\n",
		"--list-ptx": "PTX file 1: k.sm_120.ptx\n",
	}, "Cuda compilation tools, release 12.6, V12.6.85\n", "")
	prov, err := observeCUDABuildProvenance(context.Background(), fx.toolchain, fx.repoRoot, "", fx.artifact, CUDAProvenanceArtifactBinary, run)
	if err != nil {
		t.Fatal(err)
	}
	if prov.ArtifactKind != CUDAProvenanceArtifactBinary || prov.ArtifactSHA256 != sha {
		t.Fatalf("artifact identity: %+v", prov)
	}
	if !slices.Equal(prov.RequestedArchs, []string{"sm_80", "sm_90", "compute_120", "sm_120"}) {
		t.Fatalf("requested=%v", prov.RequestedArchs)
	}
	if !slices.Equal(prov.SASSArchs, []string{"sm_80", "sm_90", "sm_120"}) || !slices.Equal(prov.PTXArchs, []string{"compute_120"}) {
		t.Fatalf("observed: SASS=%v PTX=%v", prov.SASSArchs, prov.PTXArchs)
	}
	if prov.NVCC.Version != "12.6" || prov.NVCC.Role != "nvcc" || prov.NVCC.Executable != "nvcc"+execSuffix() || len(prov.NVCC.SHA256) != 64 { //boundarylint:ignore CHANGE_DETECTOR_TEST fixture echo of the observed toolchain version + SHA-256 digests are exactly 64 hexadecimal characters
		t.Fatalf("nvcc identity: %+v", prov.NVCC)
	}
	if strings.Contains(prov.NVCC.Executable, string(filepath.Separator)) {
		t.Fatalf("nvcc executable carries a host path: %+v", prov.NVCC)
	}
	if prov.CUBLASVersion != "12.6.0" { //boundarylint:ignore CHANGE_DETECTOR_TEST fixture echo of the observed cublas header version
		t.Fatalf("cublas=%q", prov.CUBLASVersion)
	}
}

// fak-test:runtime fast est=10ms
// fak-test:justify why=regression when=changed:internal/computebuild/**
func TestObserveCUDABuildProvenanceFailures(t *testing.T) {
	t.Run("missing SASS arch", func(t *testing.T) {
		fx := newCUDAProvenanceFixture(t, "sm_80\nsm_120\n")
		run := stubCUDARunner(map[string]string{
			"--list-elf": "ELF file 1: k.sm_80.cubin\n",
			"--list-ptx": "PTX file 1: k.sm_120.ptx\n",
		}, "Cuda compilation tools, release 12.6, V12.6.85\n", "")
		_, err := observeCUDABuildProvenance(context.Background(), fx.toolchain, fx.repoRoot, "", fx.artifact, CUDAProvenanceArtifactBinary, run)
		if !errors.Is(err, ErrCUDAArchMissing) || !strings.Contains(err.Error(), "sm_120") {
			t.Fatalf("err=%v", err)
		}
	})
	t.Run("missing PTX target", func(t *testing.T) {
		fx := newCUDAProvenanceFixture(t, "sm_80\nsm_120\n")
		run := stubCUDARunner(map[string]string{
			"--list-elf": "ELF file 1: k.sm_80.cubin\nELF file 2: k.sm_120.cubin\n",
			"--list-ptx": "",
		}, "Cuda compilation tools, release 12.6, V12.6.85\n", "")
		_, err := observeCUDABuildProvenance(context.Background(), fx.toolchain, fx.repoRoot, "", fx.artifact, CUDAProvenanceArtifactBinary, run)
		if !errors.Is(err, ErrCUDAArchMissing) || !strings.Contains(err.Error(), "compute_120") {
			t.Fatalf("err=%v", err)
		}
	})
	t.Run("cuobjdump missing", func(t *testing.T) {
		fx := newCUDAProvenanceFixture(t, "sm_120\n")
		if err := os.Remove(filepath.Join(fx.binDir, "cuobjdump"+execSuffix())); err != nil {
			t.Fatal(err)
		}
		_, err := observeCUDABuildProvenance(context.Background(), fx.toolchain, fx.repoRoot, "", fx.artifact, CUDAProvenanceArtifactBinary, stubCUDARunner(nil, "", ""))
		if !errors.Is(err, ErrCUDAProvenanceInvalid) || errors.Is(err, ErrCUDAArchMissing) {
			t.Fatalf("err=%v", err)
		}
	})
	t.Run("garbage listing", func(t *testing.T) {
		fx := newCUDAProvenanceFixture(t, "sm_120\n")
		run := stubCUDARunner(map[string]string{"--list-elf": "totally unparseable\n", "--list-ptx": ""}, "Cuda compilation tools, release 12.6, V12.6.85\n", "")
		_, err := observeCUDABuildProvenance(context.Background(), fx.toolchain, fx.repoRoot, "", fx.artifact, CUDAProvenanceArtifactBinary, run)
		if !errors.Is(err, ErrCUDAProvenanceInvalid) {
			t.Fatalf("err=%v", err)
		}
	})
	t.Run("runner failure", func(t *testing.T) {
		fx := newCUDAProvenanceFixture(t, "sm_120\n")
		_, err := observeCUDABuildProvenance(context.Background(), fx.toolchain, fx.repoRoot, "", fx.artifact, CUDAProvenanceArtifactBinary, stubCUDARunner(nil, "", "--list-elf"))
		if !errors.Is(err, ErrCUDAProvenanceInvalid) {
			t.Fatalf("err=%v", err)
		}
	})
	t.Run("unparseable nvcc banner", func(t *testing.T) {
		fx := newCUDAProvenanceFixture(t, "sm_120\n")
		run := stubCUDARunner(map[string]string{"--list-elf": "ELF file 1: k.sm_120.cubin\n", "--list-ptx": "PTX file 1: k.sm_120.ptx\n"}, "synthetic CUDA host compiler 1\n", "")
		_, err := observeCUDABuildProvenance(context.Background(), fx.toolchain, fx.repoRoot, "", fx.artifact, CUDAProvenanceArtifactBinary, run)
		if !errors.Is(err, ErrCUDAProvenanceInvalid) {
			t.Fatalf("err=%v", err)
		}
	})
	t.Run("missing artifact", func(t *testing.T) {
		fx := newCUDAProvenanceFixture(t, "sm_120\n")
		_, err := observeCUDABuildProvenance(context.Background(), fx.toolchain, fx.repoRoot, "", filepath.Join(fx.repoRoot, "nope"), CUDAProvenanceArtifactBinary, stubCUDARunner(nil, "", ""))
		if !errors.Is(err, ErrCUDAProvenanceInvalid) {
			t.Fatalf("err=%v", err)
		}
	})
}

// fak-test:runtime fast est=10ms
// fak-test:justify why=regression when=changed:internal/computebuild/**
func TestAttachCUDABuildProvenanceClearsOnFailure(t *testing.T) {
	fx := newCUDAProvenanceFixture(t, "sm_120\n")
	run := stubCUDARunner(map[string]string{
		"--list-elf": "ELF file 1: k.sm_120.cubin\n",
		"--list-ptx": "PTX file 1: k.sm_120.ptx\n",
	}, "Cuda compilation tools, release 12.6, V12.6.85\n", "")
	receipt := &ComputeBuildReceipt{Schema: ComputeBuildReceiptSchema, Backend: "cuda", Command: "binary", Outcome: "success"}
	attached, err := observeCUDABuildProvenance(context.Background(), fx.toolchain, fx.repoRoot, "", fx.artifact, CUDAProvenanceArtifactBinary, run)
	if err != nil {
		t.Fatal(err)
	}
	receipt.CUDA = attached
	receipt.Schema = CUDABuildReceiptSchema

	tracker := &receiptTracker{receipt: receipt}
	tracker.fail(errors.New("smoke failed after attach"), 1)
	if tracker.receipt.Outcome != "failed" || tracker.receipt.CUDA != nil {
		t.Fatalf("failure retained the provenance block: %+v", tracker.receipt)
	}
	if tracker.receipt.Schema != ComputeBuildReceiptSchema {
		t.Fatalf("failure kept successor schema %q", tracker.receipt.Schema)
	}
	raw, err := json.Marshal(tracker.receipt)
	if err != nil {
		t.Fatal(err)
	}
	if bytes.Contains(raw, []byte(`"cuda":`)) {
		t.Fatalf("failed receipt serialized the cuda block: %s", raw)
	}
}

func execSuffix() string {
	if runtime.GOOS == "windows" {
		return ".exe"
	}
	return ""
}

func sha256OfFile(t *testing.T, path string) (string, error) {
	t.Helper()
	data, err := os.ReadFile(path)
	if err != nil {
		return "", err
	}
	sum := sha256.Sum256(data)
	return hex.EncodeToString(sum[:]), nil
}
