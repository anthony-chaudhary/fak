package devcmd

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/computebuild"
)

const cudaFirstLightSchema = "fak.cuda-first-light/1"

type cudaFirstLightStage struct {
	Name      string `json:"name"`
	Outcome   string `json:"outcome"`
	ElapsedMS int64  `json:"elapsed_ms"`
	Error     string `json:"error,omitempty"`
}

type cudaFirstLightReceipt struct {
	Schema            string                `json:"schema"`
	Outcome           string                `json:"outcome"`
	Plan              bool                  `json:"plan"`
	Scope             string                `json:"scope"`
	RepoRoot          string                `json:"repo_root"`
	RequestedArch     string                `json:"requested_arch"`
	RequestedTokens   int                   `json:"requested_tokens"`
	Timeout           string                `json:"timeout"`
	ModelDirectory    string                `json:"model_directory"`
	Binary            string                `json:"binary"`
	BinarySHA256      string                `json:"binary_sha256,omitempty"`
	ReceiptPath       string                `json:"receipt_path"`
	StartedAt         time.Time             `json:"started_at"`
	FinishedAt        time.Time             `json:"finished_at"`
	NativeModelParity bool                  `json:"native_model_parity"`
	HardwareQualified bool                  `json:"hardware_qualified"`
	Stages            []cudaFirstLightStage `json:"stages"`
	Acceptance        json.RawMessage       `json:"acceptance,omitempty"`
	Pending           []string              `json:"pending"`
	Error             string                `json:"error,omitempty"`
}

type cudaFirstLightDeps struct {
	discover func() (*computebuild.Toolchain, error)
	build    func(context.Context, *computebuild.CUDAConfig) error
	run      func(context.Context, string, []string, string, []string, io.Writer, io.Writer) error
}

func runCUDAFirstLight(stdout, stderr io.Writer, args []string) int {
	return runCUDAFirstLightWith(stdout, stderr, args, cudaFirstLightDeps{
		discover: computebuild.DiscoverToolchain, build: computebuild.RunCUDA, run: computebuild.RunCmd,
	})
}

func runCUDAFirstLightWith(stdout, stderr io.Writer, args []string, deps cudaFirstLightDeps) int {
	fs := flag.NewFlagSet("build-cuda first-light", flag.ContinueOnError)
	fs.SetOutput(stderr)
	rootFlag := fs.String("repo-root", "", "public repository root (default: discover from current directory)")
	arch := fs.String("arch", "", "exact CUDA device architecture to witness, e.g. sm_120")
	hf := fs.String("hf", "", "unsharded HuggingFace directory containing config.json and model.safetensors")
	tokens := fs.Int("tokens", 12, "greedy tokens compared by gpucheck (1..4096)")
	timeout := fs.Duration("timeout", 20*time.Minute, "deadline shared by header, build and acceptance (maximum 2h)")
	receiptFlag := fs.String("receipt", ".fak/cuda-first-light.json", "durable result path, relative to --repo-root")
	binaryFlag := fs.String("out-bin", ".fak/bin/cuda-first-light-gpucheck", "gpucheck output path, relative to --repo-root")
	plan := fs.Bool("plan", false, "write a plan receipt without toolchain discovery, compilation or device execution")
	jsonOut := fs.Bool("json", false, "emit the result as JSON")
	fs.Usage = func() {
		fmt.Fprintln(stderr, "usage: fak-dev build-cuda first-light --repo-root ROOT --arch sm_120 --hf DIRECTORY [--plan] [--json]")
		fmt.Fprintln(stderr, "Runs header syntax, native gpucheck build and strict HF model parity. GGUF serving, kernel-family acceptance and reference performance remain pending.")
		fs.PrintDefaults()
	}
	if err := fs.Parse(args); err != nil {
		if errors.Is(err, flag.ErrHelp) {
			return 0
		}
		return 2
	}
	if fs.NArg() != 0 || *arch == "" || *hf == "" || *tokens < 1 || *tokens > 4096 || *timeout <= 0 || *timeout > 2*time.Hour || strings.TrimSpace(*receiptFlag) == "" || strings.TrimSpace(*binaryFlag) == "" {
		fs.Usage()
		return 2
	}
	root := *rootFlag
	var err error
	if root == "" {
		root, err = computebuild.ResolveRepoRoot("")
	} else {
		root, err = filepath.Abs(root)
	}
	if err != nil {
		fmt.Fprintln(stderr, err)
		return 2
	}
	target := *arch
	if !strings.HasPrefix(target, "sm_") {
		target = "sm_" + target
	}
	if runtime.GOOS == "windows" && *binaryFlag == ".fak/bin/cuda-first-light-gpucheck" {
		*binaryFlag += ".exe"
	}
	r := cudaFirstLightReceipt{
		Schema: cudaFirstLightSchema, Outcome: "planned", Plan: *plan, Scope: "native-hf-model-parity",
		RepoRoot: root, RequestedArch: target, RequestedTokens: *tokens, Timeout: timeout.String(), ModelDirectory: rootedBuildPath(root, *hf),
		Binary: rootedBuildPath(root, *binaryFlag), ReceiptPath: rootedBuildPath(root, *receiptFlag),
		StartedAt: time.Now().UTC(),
		Stages:    []cudaFirstLightStage{{Name: "preflight", Outcome: "pending"}, {Name: "header", Outcome: "pending"}, {Name: "build", Outcome: "pending"}, {Name: "accept", Outcome: "pending"}},
		Pending:   []string{"gguf-serving-coherence", "kernel-family-acceptance", "same-card-reference-abba", "perf-receipt"},
	}
	finish := func(code int, cause error) int {
		r.FinishedAt = time.Now().UTC()
		if cause != nil {
			r.Outcome = "failed"
			r.Error = cause.Error()
		}
		if err := writeJSONAtomic(r.ReceiptPath, r); err != nil {
			fmt.Fprintf(stderr, "first-light receipt: %v\n", err)
			return 1
		}
		if *jsonOut {
			_ = json.NewEncoder(stdout).Encode(r)
		} else {
			fmt.Fprintf(stdout, "CUDA first-light %s: native_model_parity=%t hardware_qualified=false receipt=%s\n", r.Outcome, r.NativeModelParity, r.ReceiptPath)
		}
		return code
	}
	// Never let a receipt overwrite a model input or the executable it describes.
	for _, input := range []string{r.Binary, filepath.Join(r.ModelDirectory, "config.json"), filepath.Join(r.ModelDirectory, "model.safetensors")} {
		if sameCanonicalPath(r.ReceiptPath, input) {
			fmt.Fprintln(stderr, "first-light: receipt must differ from binary and model inputs")
			return 2
		}
	}
	for _, input := range []string{filepath.Join(r.ModelDirectory, "config.json"), filepath.Join(r.ModelDirectory, "model.safetensors"), filepath.Join(root, "go.mod"), filepath.Join(root, "internal", "compute", "cuda_backend.h"), filepath.Join(root, "internal", "compute", "cuda_arch.txt")} {
		for _, output := range []string{r.Binary, r.ReceiptPath, r.ReceiptPath + ".build.json"} {
			if sameCanonicalPath(output, input) {
				fmt.Fprintln(stderr, "first-light: output must differ from model and source inputs")
				return 2
			}
		}
	}
	if sameCanonicalPath(r.Binary, r.ReceiptPath+".build.json") {
		fmt.Fprintln(stderr, "first-light: binary must differ from build receipt")
		return 2
	}
	if *plan {
		return finish(0, nil)
	}
	r.Outcome = "running"
	if err := writeJSONAtomic(r.ReceiptPath, r); err != nil {
		fmt.Fprintln(stderr, err)
		return 1
	}
	ctx, cancel := context.WithTimeout(context.Background(), *timeout)
	defer cancel()
	stage := func(i int, operation func() error) error {
		start := time.Now()
		err := operation()
		r.Stages[i].ElapsedMS = time.Since(start).Milliseconds()
		r.Stages[i].Outcome = "passed"
		if err != nil {
			r.Stages[i].Outcome = "failed"
			r.Stages[i].Error = err.Error()
		}
		return err
	}
	var tc *computebuild.Toolchain
	if err := stage(0, func() error {
		if _, _, err := computebuild.LoadCUDAArchMatrix(root, target); err != nil {
			return err
		}
		for _, input := range []string{filepath.Join(root, "go.mod"), filepath.Join(r.ModelDirectory, "config.json"), filepath.Join(r.ModelDirectory, "model.safetensors")} {
			info, err := os.Stat(input)
			if err != nil {
				return err
			}
			if !info.Mode().IsRegular() || info.Size() == 0 {
				return fmt.Errorf("first-light input is not a nonempty regular file: %s", input)
			}
		}
		var err error
		tc, err = deps.discover()
		if err != nil {
			return err
		}
		if tc == nil || tc.NVCC == "" || tc.CC == "" {
			return errors.New("first-light requires nvcc and a C compiler")
		}
		return ctx.Err()
	}); err != nil {
		return finish(1, err)
	}
	env := os.Environ()
	if len(tc.VsDevEnv) != 0 {
		env = computebuild.MergeEnviron(env, tc.VsDevEnv)
	}
	if err := stage(1, func() error {
		return deps.run(ctx, tc.CC, []string{"-x", "c", "-std=c11", "-fsyntax-only", "-Wall", "-Werror", filepath.Join(root, "internal", "compute", "cuda_backend.h")}, root, env, stderr, stderr)
	}); err != nil {
		return finish(1, err)
	}
	if err := stage(2, func() error {
		if err := os.MkdirAll(filepath.Dir(r.Binary), 0o755); err != nil {
			return err
		}
		cfg := &computebuild.CUDAConfig{
			Command: "binary", RepoRoot: root, PkgDir: filepath.Join(root, "internal", "compute"), Arch: target,
			OutPkg: "./cmd/gpucheck", OutBin: r.Binary, Toolchain: tc, SkipSmoke: true, SkipSign: true,
			ReceiptPath: r.ReceiptPath + ".build.json", Stdout: stderr, Stderr: stderr,
		}
		if err := deps.build(ctx, cfg); err != nil {
			return err
		}
		artifact, err := computebuild.InspectArtifact(r.Binary)
		if err != nil {
			return err
		}
		if artifact.SizeBytes == 0 || len(artifact.SHA256) != 64 {
			return errors.New("first-light build requires a nonempty binary with a readable SHA256 digest")
		}
		r.BinarySHA256 = artifact.SHA256
		return nil
	}); err != nil {
		return finish(1, err)
	}
	if err := stage(3, func() error {
		var output cudaFirstLightOutput
		runErr := deps.run(ctx, r.Binary, []string{"-hf", r.ModelDirectory, "-backend", "cuda", "-n", fmt.Sprint(*tokens), "-json", "-require-native"}, root, env, &output, stderr)
		if json.Valid(output.Bytes()) {
			r.Acceptance = append(json.RawMessage(nil), output.Bytes()...)
		}
		if runErr != nil {
			return runErr
		}
		if len(r.Acceptance) == 0 {
			return errors.New("first-light acceptance is not one JSON object")
		}
		return validateCUDAFirstLightAcceptance(r.Acceptance, target, *tokens)
	}); err != nil {
		return finish(1, err)
	}
	r.NativeModelParity = true
	r.Outcome = "passed"
	return finish(0, nil)
}

type cudaFirstLightOutput struct{ buffer bytes.Buffer }

func (w *cudaFirstLightOutput) Bytes() []byte { return w.buffer.Bytes() }

func (w *cudaFirstLightOutput) Write(p []byte) (int, error) {
	if w.buffer.Len()+len(p) > 1<<20 {
		return 0, errors.New("first-light acceptance exceeds 1 MiB")
	}
	return w.buffer.Write(p)
}

func validateCUDAFirstLightAcceptance(data []byte, arch string, tokens int) error {
	var receipt struct {
		Schema           string `json:"schema"`
		Engine           string `json:"engine"`
		Pass             *bool  `json:"pass"`
		RequestedBackend string `json:"requested_backend"`
		ObservedBackend  string `json:"observed_backend"`
		ObservedTier     string `json:"observed_tier"`
		DeviceMemory     *bool  `json:"device_memory"`
		BackendObserved  *bool  `json:"backend_observed"`
		FallbackActive   *bool  `json:"fallback_active"`
		RequestedTokens  int    `json:"requested_tokens"`
		ComparedTokens   int    `json:"compared_tokens"`
		ReferenceTokens  []int  `json:"reference_tokens"`
		DeviceTokens     []int  `json:"device_tokens"`
	}
	if err := json.Unmarshal(data, &receipt); err != nil {
		return err
	}
	if receipt.Schema != "fak.gpucheck/1" || receipt.Engine != "fak-native" || receipt.Pass == nil || !*receipt.Pass ||
		receipt.RequestedBackend != "cuda" || receipt.ObservedBackend != "cuda" || receipt.ObservedTier != arch ||
		receipt.DeviceMemory == nil || !*receipt.DeviceMemory || receipt.BackendObserved == nil || !*receipt.BackendObserved ||
		receipt.FallbackActive == nil || *receipt.FallbackActive || receipt.RequestedTokens != tokens ||
		receipt.ComparedTokens < 1 || receipt.ComparedTokens > tokens || len(receipt.ReferenceTokens) != receipt.ComparedTokens || len(receipt.DeviceTokens) != receipt.ComparedTokens {
		return errors.New("first-light native CUDA identity or parity receipt refused")
	}
	for i, id := range receipt.ReferenceTokens {
		if id < 0 || receipt.DeviceTokens[i] != id {
			return errors.New("first-light token streams differ or contain an invalid token")
		}
	}
	return nil
}
