package devcmd

import (
	"bytes"
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"runtime"
	"strconv"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/computebuild"
)

// CUDABuildResult represents the structured JSON output for RunBuildCUDA.
type CUDABuildResult struct {
	Schema     string `json:"schema"`
	Command    string `json:"command"`
	Arch       string `json:"arch,omitempty"`
	Success    bool   `json:"success"`
	OutBin     string `json:"out_bin,omitempty"`
	Error      string `json:"error,omitempty"`
	DurationMS int64  `json:"duration_ms"`
}

// RunBuildCUDA parses CLI arguments and runs CUDA build/test/check tasks via computebuild.RunCUDA.
func RunBuildCUDA(stdout, stderr io.Writer, argv []string) int {
	return runBuildCUDA(stdout, stderr, argv, computebuild.RunCUDA)
}

func runBuildCUDA(stdout, stderr io.Writer, argv []string, run func(context.Context, *computebuild.CUDAConfig) error) int {
	return runBuildCUDAWith(stdout, stderr, argv, run, computebuild.RunCmd)
}

func runBuildCUDAWith(stdout, stderr io.Writer, argv []string, run func(context.Context, *computebuild.CUDAConfig) error, command func(context.Context, string, []string, string, []string, io.Writer, io.Writer) error) int {
	var sub string
	var flagArgs []string
	if len(argv) > 0 && !strings.HasPrefix(argv[0], "-") {
		sub = argv[0]
		flagArgs = argv[1:]
	} else {
		flagArgs = argv
	}
	if sub == "first-light" {
		return runCUDAFirstLight(stdout, stderr, flagArgs)
	}
	if sub == "legacy" {
		return runCUDALegacy(stdout, stderr, flagArgs, run, command)
	}

	fs := flag.NewFlagSet("build-cuda", flag.ContinueOnError)
	fs.SetOutput(stderr)

	arch := fs.String("arch", os.Getenv("FAK_CUDA_ARCH"), "target CUDA architecture (defaults to FAK_CUDA_ARCH, empty for all)")
	cudaHome := fs.String("cuda-home", "", "path to CUDA Toolkit root directory")
	nccl := fs.Bool("nccl", os.Getenv("FAK_CUDA_NCCL") == "1", "compile with NCCL collectives enabled (-tags cuda,nccl)")
	ncclHome := fs.String("nccl-home", os.Getenv("NCCL_HOME"), "optional path to NCCL installation directory")
	jsonOut := fs.Bool("json", false, "emit machine-readable JSON output")
	receiptFlag := fs.String("receipt", ".fak/cuda-build-receipt.json", "path to write durable JSON build receipt")
	smokeFlag := fs.Bool("smoke", true, "run shift-left smoke verification on compiled binary")
	repoRoot := fs.String("repo-root", "", "repository root directory")
	pkgDir := fs.String("pkg-dir", "", "compute package directory")
	outPkg := fs.String("out-pkg", "", "target Go package for binary build")
	outBin := fs.String("out-bin", "", "output binary path")
	fs.StringVar(outBin, "o", "", "output binary path (shorthand)")
	fs.StringVar(outBin, "out", "", "output binary path (alias)")
	benchModel := fs.String("bench-model", "", "model directory for bench command")
	benchSteps := fs.Int("bench-steps", 0, "decode steps for bench command")
	skipSign := fs.Bool("skip-sign", false, "skip Authenticode code signing on Windows")
	signCert := fs.String("sign-cert", "", "SHA-1 thumbprint for Windows certificate store")
	signPFX := fs.String("sign-pfx", "", "path to .pfx certificate file")
	signPassword := fs.String("sign-password", "", "password for .pfx certificate")
	timestampURL := fs.String("timestamp-url", "", "RFC-3161 timestamp server URL")
	nvcc := fs.String("nvcc", "", "path to nvcc compiler")
	cxx := fs.String("cxx", os.Getenv("FAK_NVCC_CCBIN"), "C++ compiler executable")
	cc := fs.String("cc", "", "C compiler executable")
	cmdFlag := fs.String("cmd", "", "subcommand mode (check, build, test, bench, binary)")
	hf := fs.String("hf", os.Getenv("FAK_CUDA_ACCEPT_HF"), "HuggingFace model directory for native model acceptance")
	tokensDefault := 12
	if value := os.Getenv("FAK_CUDA_ACCEPT_TOKENS"); value != "" {
		tokensDefault, _ = strconv.Atoi(value)
	}
	tokens := fs.Int("tokens", tokensDefault, "greedy tokens for model acceptance (1..4096)")

	fs.Usage = func() {
		fmt.Fprintln(stderr, "usage: fak-dev build-cuda [subcommand] [flags] [pkg] [out]")
		fmt.Fprintln(stderr, "subcommands:")
		fmt.Fprintln(stderr, "  first-light check headers, build gpucheck and require native HF model parity (--plan prints preparation only)")
		fmt.Fprintln(stderr, "  check      run GPU-free ABI parity, arch matrix, and header syntax checks")
		fmt.Fprintln(stderr, "  build      compile CUDA kernels to libfakcuda.a and build compute package")
		fmt.Fprintln(stderr, "  test       run CUDA HAL device integration tests")
		fmt.Fprintln(stderr, "  bench      run decode microbenchmarks against model weights")
		fmt.Fprintln(stderr, "  binary     build full binary with CUDA backend enabled")
		fmt.Fprintln(stderr, "  accept-model build gpucheck and require native model parity")
		fmt.Fprintln(stderr, "  accept     require model parity and all isolated CUDA family witnesses")
		fmt.Fprintln(stderr, "flags:")
		fs.PrintDefaults()
	}

	if err := fs.Parse(flagArgs); err != nil {
		if err == flag.ErrHelp {
			return 0
		}
		return 2
	}

	if sub == "" {
		if *cmdFlag != "" {
			sub = *cmdFlag
		} else if fs.NArg() > 0 {
			sub = fs.Arg(0)
		}
	}

	sub = strings.ToLower(strings.TrimSpace(sub))
	if sub == "" || sub == "help" {
		fs.Usage()
		if sub == "help" {
			return 0
		}
		return 2
	}

	switch sub {
	case "check", "build", "test", "bench", "binary", "accept-model", "accept":
	default:
		fmt.Fprintf(stderr, "build-cuda: unknown subcommand %q (expected check, build, test, bench, binary)\n", sub)
		return 2
	}

	cfg := &computebuild.CUDAConfig{
		Command:            sub,
		RepoRoot:           *repoRoot,
		PkgDir:             *pkgDir,
		Arch:               *arch,
		EnableNCCL:         *nccl,
		NCCLHome:           *ncclHome,
		OutPkg:             *outPkg,
		OutBin:             *outBin,
		ReceiptPath:        *receiptFlag,
		Smoke:              *smokeFlag,
		SkipSmoke:          !*smokeFlag,
		BenchModelDir:      *benchModel,
		BenchDecodeSteps:   *benchSteps,
		SkipSign:           *skipSign,
		SignCertThumbprint: *signCert,
		SignPFX:            *signPFX,
		SignPFXPassword:    *signPassword,
		TimestampURL:       *timestampURL,
		Stdout:             stdout,
		Stderr:             stderr,
	}

	// For binary subcommand, allow trailing positional args: [pkg] [out]
	posArgs := fs.Args()
	if len(posArgs) > 0 && posArgs[0] == sub {
		posArgs = posArgs[1:]
	}
	if sub == "binary" {
		if cfg.OutPkg == "" && len(posArgs) > 0 {
			cfg.OutPkg = posArgs[0]
		}
		if cfg.OutBin == "" && len(posArgs) > 1 {
			cfg.OutBin = posArgs[1]
		}
	}

	// Customize toolchain if explicit overrides were passed
	if *cudaHome != "" || *nvcc != "" || *cxx != "" || *cc != "" || *ncclHome != "" {
		tc, _ := computebuild.DiscoverToolchain()
		if tc == nil {
			tc = &computebuild.Toolchain{IsWindows: runtime.GOOS == "windows"}
		}
		if *cudaHome != "" {
			tc.CUDAHome = *cudaHome
			nvccName := "nvcc"
			if tc.IsWindows {
				nvccName += ".exe"
			}
			tc.NVCC = filepath.Join(*cudaHome, "bin", nvccName)
			tc.CUDAInc = existingCUDAPaths(*cudaHome, "include", "targets/x86_64-linux/include")
			if tc.IsWindows {
				tc.CUDALib = []string{filepath.Join(*cudaHome, "lib", "x64")}
			} else {
				tc.CUDALib = existingCUDAPaths(*cudaHome, "lib64", "lib", "targets/x86_64-linux/lib")
			}
		}
		if *nvcc != "" {
			tc.NVCC = *nvcc
		}
		if *cxx != "" {
			tc.CXX = *cxx
		}
		if *cc != "" {
			tc.CC = *cc
		}
		if *ncclHome != "" {
			tc.CUDAInc = append(tc.CUDAInc, existingCUDAPaths(*ncclHome, "include", "usr/include")...)
			tc.CUDALib = append(tc.CUDALib, existingCUDAPaths(*ncclHome, "lib64", "lib", "usr/lib/x86_64-linux-gnu", "usr/lib")...)
		}
		cfg.Toolchain = tc
	}
	if sub == "accept-model" || sub == "accept" {
		return runCUDABuildAcceptance(stdout, stderr, cfg, *hf, *tokens, sub == "accept", run, command)
	}

	if *jsonOut {
		cfg.Stdout = io.Discard
	}

	start := time.Now()
	err := run(context.Background(), cfg)
	dur := time.Since(start).Milliseconds()

	if *jsonOut {
		res := CUDABuildResult{
			Schema:     "fak.cuda-build.v1",
			Command:    sub,
			Arch:       cfg.Arch,
			Success:    err == nil,
			OutBin:     cfg.OutBin,
			DurationMS: dur,
		}
		if err != nil {
			res.Error = err.Error()
		}
		_ = json.NewEncoder(stdout).Encode(res)
		if err != nil {
			return 1
		}
		return 0
	}

	if err != nil {
		fmt.Fprintf(stderr, "build-cuda: %v\n", err)
		return 1
	}

	return 0
}

func existingCUDAPaths(root string, relative ...string) []string {
	var paths []string
	for _, name := range relative {
		path := filepath.Join(root, filepath.FromSlash(name))
		if info, err := os.Stat(path); err == nil && info.IsDir() {
			paths = append(paths, path)
		}
	}
	return paths
}

type cudaAcceptanceFamily struct {
	Name    string   `json:"name"`
	Package string   `json:"package"`
	Tests   []string `json:"tests"`
	Bench   string   `json:"-"`
	Outcome string   `json:"outcome"`
}

func cudaAcceptanceFamilies() []cudaAcceptanceFamily {
	return []cudaAcceptanceFamily{
		{Name: "base-forward", Package: "./internal/compute/", Tests: []string{"TestCUDAEvictMiddleSpanEqualsNeverSaw"}},
		{Name: "async-hostxfer", Package: "./internal/compute/", Tests: []string{"TestCUDAAsyncArgmaxParityAndResidency"}, Bench: "BenchmarkCUDA(Sync|Async)Decode"},
		{Name: "cuda-graph", Package: "./internal/compute/", Tests: []string{"TestCUDAGraphDecodeParity", "TestCUDAGraphCompileCapGated"}, Bench: "BenchmarkCUDADecode(NoCapture|Capture)"},
		{Name: "fp16", Package: "./internal/compute/", Tests: []string{"TestCUDAMatMulF16ApproxMatchesRef", "TestCUDABatchedMatMulF16ApproxMatchesRef"}, Bench: "BenchmarkCUDABatchedMatMulF(32|16)"},
		{Name: "quant", Package: "./internal/compute/", Tests: []string{"TestCUDAQ8MatMulApproxMatchesRef", "TestCUDAQ8BatchedMatMulApproxMatchesRef", "TestCUDAQ4KMatMulApproxMatchesRef", "TestCUDAQ4KBatchedMatMulApproxMatchesRef", "TestCUDAQuantVRAMWitness"}, Bench: "BenchmarkCUDA(BatchedMatMulF32|Q8BatchedMatMul|Q4KBatchedMatMul)"},
		{Name: "flash-attention", Package: "./internal/compute/", Tests: []string{"TestCUDAFlashAttentionMatchesRef", "TestCUDAForwardMatchesRef"}, Bench: "BenchmarkCUDA(FlashAttention|NaiveAttention)"},
		{Name: "glm-forward", Package: "./internal/model/", Tests: []string{"TestCUDAGLMMoeDsaBackendForward"}},
		{Name: "glm-cpu-offload", Package: "./internal/model/", Tests: []string{"TestCUDAGLMDsaCPUOffloadHybrid"}},
		{Name: "glm-index-selection", Package: "./internal/model/", Tests: []string{"TestCUDAGLMMoeDsaIndexSelectMatches"}},
		{Name: "awq", Package: "./internal/compute/", Tests: []string{"TestCUDAAWQMatMulApproxMatchesRef", "TestCUDAAWQBatchedMatMulApproxMatchesRef"}, Bench: "BenchmarkCUDA(BatchedMatMulF32|AWQBatchedMatMul)"},
	}
}

func runCUDABuildAcceptance(stdout, stderr io.Writer, cfg *computebuild.CUDAConfig, hf string, tokens int, families bool, build func(context.Context, *computebuild.CUDAConfig) error, command func(context.Context, string, []string, string, []string, io.Writer, io.Writer) error) int {
	if hf == "" || tokens < 1 || tokens > 4096 {
		return 2
	}
	root, err := computebuild.ResolveRepoRoot(cfg.RepoRoot)
	if err != nil {
		fmt.Fprintln(stderr, err)
		return 2
	}
	cfg.RepoRoot = root
	if !filepath.IsAbs(hf) {
		hf = filepath.Join(root, hf)
	}
	for _, name := range []string{"config.json", "model.safetensors"} {
		info, err := os.Stat(filepath.Join(hf, name))
		if err != nil || !info.Mode().IsRegular() || info.Size() == 0 {
			fmt.Fprintf(stderr, "CUDA acceptance requires nonempty model input %s\n", name)
			return 2
		}
	}
	if cfg.OutPkg == "" {
		cfg.OutPkg = "./cmd/gpucheck"
	}
	if cfg.OutBin == "" {
		cfg.OutBin = filepath.Join(root, "tools", ".bin", "gpucheck")
		if runtime.GOOS == "windows" {
			cfg.OutBin += ".exe"
		}
	} else if !filepath.IsAbs(cfg.OutBin) {
		cfg.OutBin = filepath.Join(root, cfg.OutBin)
	}
	if err := os.MkdirAll(filepath.Dir(cfg.OutBin), 0755); err != nil {
		fmt.Fprintln(stderr, err)
		return 1
	}
	path := cfg.ReceiptPath + ".accept.json"
	if !filepath.IsAbs(path) {
		path = filepath.Join(root, path)
	}
	if filepath.Clean(path) == filepath.Clean(cfg.OutBin) {
		return 2
	}
	if err := writeJSONAtomic(path, map[string]any{"schema": "fak.cuda-acceptance/1", "success": false, "outcome": "pending"}); err != nil {
		fmt.Fprintln(stderr, err)
		return 1
	}
	cfg.Command, cfg.Stdout, cfg.Stderr = "binary", stderr, stderr
	if err := build(context.Background(), cfg); err != nil {
		fmt.Fprintln(stderr, err)
		return 1
	}
	if cfg.Toolchain == nil {
		fmt.Fprintln(stderr, "CUDA acceptance build did not resolve its toolchain")
		return 1
	}
	env := computebuild.MergeEnviron(os.Environ(), computebuild.SynthesizeCUDACgoEnv(cfg.Toolchain, cfg.PkgDir, cfg.EnableNCCL, nil))
	env = computebuild.MergeEnviron(env, cfg.Toolchain.VsDevEnv)
	var parity cudaFirstLightOutput
	if err := command(context.Background(), cfg.OutBin, []string{"-hf", hf, "-backend", "cuda", "-n", strconv.Itoa(tokens), "-json", "-require-native"}, root, env, &parity, stderr); err != nil {
		fmt.Fprintln(stderr, err)
		return 1
	}
	arch := cfg.Arch
	if arch == "" {
		var identity struct {
			ObservedTier string `json:"observed_tier"`
		}
		if err := json.Unmarshal(parity.Bytes(), &identity); err != nil {
			return 1
		}
		arch = identity.ObservedTier
	}
	if !strings.HasPrefix(arch, "sm_") {
		arch = "sm_" + arch
	}
	if _, _, err := computebuild.LoadCUDAArchMatrix(root, arch); err != nil {
		fmt.Fprintln(stderr, err)
		return 1
	}
	if err := validateCUDAFirstLightAcceptance(parity.Bytes(), arch, tokens); err != nil {
		fmt.Fprintln(stderr, err)
		return 1
	}
	if !families {
		if err := writeJSONAtomic(path, json.RawMessage(parity.Bytes())); err != nil {
			fmt.Fprintln(stderr, err)
			return 1
		}
		_, err := stdout.Write(parity.Bytes())
		if err != nil {
			return 1
		}
		return 0
	}
	results := cudaAcceptanceFamilies()
	code := 0
	tags := "cuda"
	if cfg.EnableNCCL {
		tags += ",nccl"
	}
	for i := range results {
		family := &results[i]
		var output cudaFirstLightOutput
		args := []string{"test", "-json", "-tags", tags, "-count=1", "-run", "^(" + strings.Join(family.Tests, "|") + ")$", family.Package}
		err := command(context.Background(), "go", args, root, env, &output, stderr)
		family.Outcome = "failed"
		if err == nil {
			err = validateCUDAAcceptanceEvents(output.Bytes(), family.Tests)
		}
		if err != nil {
			code = 1
			fmt.Fprintf(stderr, "%s: %v\n", family.Name, err)
		} else {
			family.Outcome = "passed"
		}
		_, _ = stderr.Write(output.Bytes())
		if err == nil && family.Bench != "" {
			// Existing family tools report benchmark observations separately from correctness.
			if err := command(context.Background(), "go", []string{"test", "-tags", tags, "-count=1", "-run", "^$", "-bench", "^(" + family.Bench + ")$", "-benchtime=2s", family.Package}, root, env, stderr, stderr); err != nil {
				fmt.Fprintf(stderr, "%s benchmark: %v\n", family.Name, err)
			}
		}
	}
	result := struct {
		Schema   string                 `json:"schema"`
		Success  bool                   `json:"success"`
		Model    json.RawMessage        `json:"model"`
		Families []cudaAcceptanceFamily `json:"families"`
	}{"fak.cuda-acceptance/1", code == 0, parity.Bytes(), results}
	if err := writeJSONAtomic(path, result); err != nil {
		fmt.Fprintln(stderr, err)
		return 1
	}
	if err := json.NewEncoder(stdout).Encode(result); err != nil {
		return 1
	}
	return code
}

func validateCUDAAcceptanceEvents(data []byte, expected []string) error {
	passed := make(map[string]bool)
	decoder := json.NewDecoder(bytes.NewReader(data))
	for {
		var event struct{ Action, Test string }
		if err := decoder.Decode(&event); err != nil {
			if err == io.EOF {
				break
			}
			return err
		}
		if event.Action == "skip" || event.Action == "fail" {
			return fmt.Errorf("CUDA_ACCEPTANCE_UNVERIFIED: %s %s", event.Action, event.Test)
		}
		if event.Action == "pass" {
			passed[event.Test] = true
		}
	}
	for _, name := range expected {
		if !passed[name] {
			return fmt.Errorf("CUDA_ACCEPTANCE_MISSING: %s", name)
		}
	}
	return nil
}
