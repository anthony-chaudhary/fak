package devcmd

import (
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/amdgpu"
)

var (
	runStrixHILFn = amdgpu.RunStrixHIL
)

// RunAMDStrixHIL executes the comprehensive hardware-in-the-loop evaluation on AMD Strix Halo.
func RunAMDStrixHIL(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("amd-strix-hil", flag.ContinueOnError)
	fs.SetOutput(stderr)

	host := fs.String("host", "", "target Strix Halo host (default auto-discover: strix-agent / strix1)")
	subkernels := fs.String("subkernels", "all", "sub-kernels to test ('all', 'none', or comma-separated list)")
	ablate := fs.String("ablate", "all", "ablation arms to evaluate ('all', 'none', or comma-separated list)")
	runInfer := fs.Bool("infer", true, "run live model serving inference check on the appliance")
	model := fs.String("model", amdgpu.DefaultHILModel, "model name to query for inference check")
	prompt := fs.String("prompt", "What is 2 + 2? Reply with just the number.", "prompt for live inference check")
	key := fs.String("key", "", "gateway authentication key (optional, auto-discovered from target)")
	asJSON := fs.Bool("json", false, "emit complete HIL receipt as JSON")
	timeoutSec := fs.Int("timeout", 180, "total execution timeout in seconds")
	admissionTimeoutSec := fs.Int("admission-timeout", 25, "hardware admission timeout in seconds")
	var minePaths stringSliceFlag
	fs.Var(&minePaths, "mine", "repeatable owned path relative to repository root for candidate overlay")
	candidateDir := fs.String("candidate-dir", ".", "path within the candidate checkout")
	committedOnly := fs.Bool("committed-only", false, "use committed HEAD only (requires clean worktree)")

	if err := fs.Parse(argv); err != nil {
		return 2
	}
	if fs.NArg() > 0 {
		fmt.Fprintf(stderr, "amd-strix-hil: positional arguments rejected: %v (use repeatable --mine PATH)\n", fs.Args())
		return 1
	}
	if *timeoutSec <= 0 {
		fmt.Fprintf(stderr, "amd-strix-hil: invalid timeout %d: must be > 0\n", *timeoutSec)
		return 1
	}

	ctx, cancel := context.WithTimeout(context.Background(), time.Duration(*timeoutSec)*time.Second)
	defer cancel()

	runSK := *subkernels != "none" && *subkernels != ""
	runAB := *ablate != "none" && *ablate != ""
	runVal := runSK || runAB

	var skList, abList []string
	if runSK && *subkernels != "all" {
		skList = strings.Split(*subkernels, ",")
	}
	if runAB && *ablate != "all" {
		abList = strings.Split(*ablate, ",")
	}

	opts := amdgpu.StrixHILOpts{
		Host:             *host,
		RunValidation:    runVal,
		RunSubkernels:    runSK,
		Subkernels:       skList,
		RunAblations:     runAB,
		Ablations:        abList,
		RunInference:     *runInfer,
		InferenceModel:   *model,
		InferencePrompt:  *prompt,
		GatewayKey:       *key,
		Command:          "fak-dev amd-strix-hil " + strings.Join(argv, " "),
		Timeout:          time.Duration(*timeoutSec) * time.Second,
		AdmissionTimeout: time.Duration(*admissionTimeoutSec) * time.Second,
	}

	if runVal {
		candidate, err := resolveStrixCandidate(ctx, *candidateDir, "", minePaths, *committedOnly)
		if err != nil {
			fmt.Fprintf(stderr, "amd-strix-hil: candidate archive validation failed: %v\n", err)
			return 1
		}
		opts.RequireSourceBinding = true
		opts.GitRef = candidate.archive.SourceArchiveSHA256
		opts.GitTip = candidate.tip
		opts.CandidateArchive = candidate.archive.Bytes
		opts.SourceArchiveSHA256 = candidate.archive.SourceArchiveSHA256
	}

	if !*asJSON {
		fmt.Fprintf(stderr, "==> Running Hardware-In-The-Loop (HIL) evaluation on AMD Strix Halo...\n")
	}

	receipt, err := runStrixHILFn(ctx, opts)
	if receipt == nil {
		fmt.Fprintf(stderr, "amd-strix-hil: execution failed: %v\n", err)
		return 1
	}

	if *asJSON {
		data, _ := json.MarshalIndent(receipt, "", "  ")
		fmt.Fprintln(stdout, string(data))
		if err != nil || !receipt.Verified {
			return 1
		}
		return 0
	}

	renderStrixHILReceipt(stdout, receipt)
	if err != nil || !receipt.Verified {
		return 1
	}
	return 0
}

func renderStrixHILReceipt(w io.Writer, r *amdgpu.StrixHILReceipt) {
	fmt.Fprintln(w, "\n================================================================================")
	fmt.Fprintln(w, "AMD Strix Halo Hardware-In-The-Loop (HIL) Validation Receipt")
	fmt.Fprintln(w, "================================================================================")
	fmt.Fprintf(w, "Verdict:      %s (verified: %v)\n", r.Verdict, r.Verified)
	fmt.Fprintf(w, "Target:       %s (%s)\n", r.Target.Host, r.Target.Mode)
	fmt.Fprintf(w, "CPU Model:    %s\n", r.Target.CPUModel)
	fmt.Fprintf(w, "GPU Model:    %s (%s, %d CUs)\n", r.Target.GPUName, r.Target.TargetISA, r.Target.ComputeUnits)
	fmt.Fprintf(w, "Memory:       %.1f GiB UMA Aperture (Total: %.1f GiB)\n",
		float64(r.Target.UMABufferBytes)/(1024*1024*1024), float64(r.Target.TotalRAMBytes)/(1024*1024*1024))
	fmt.Fprintf(w, "DPM Level:    %s (Watchdog: %d)\n", r.Target.DPMLevel, r.Target.LockupTimeout)
	fmt.Fprintf(w, "Digest:       %s\n", r.Digest)

	if r.InferenceWitness != nil {
		fmt.Fprintln(w, "--------------------------------------------------------------------------------")
		fmt.Fprintln(w, "Live Model Serving Inference Witness:")
		fmt.Fprintf(w, "  Endpoint:     %s\n", r.InferenceWitness.Endpoint)
		fmt.Fprintf(w, "  Model:        %s\n", r.InferenceWitness.Model)
		fmt.Fprintf(w, "  Prompt:       %q\n", r.InferenceWitness.Prompt)
		fmt.Fprintf(w, "  Response:     %q\n", r.InferenceWitness.Response)
		fmt.Fprintf(w, "  Tokens:       %d prompt + %d completion = %d total\n",
			r.InferenceWitness.PromptTokens, r.InferenceWitness.CompletionTokens, r.InferenceWitness.TotalTokens)
		fmt.Fprintf(w, "  Throughput:   %.1f tok/s (latency: %.1f ms)\n",
			r.InferenceWitness.TokensPerSec, r.InferenceWitness.LatencyMS)
		fmt.Fprintf(w, "  Witness:      VERIFIED (model served natively on Strix Halo APU)\n")
	}

	if r.ValidationReceipt != nil {
		if len(r.ValidationReceipt.Subkernels) > 0 {
			fmt.Fprintln(w, "--------------------------------------------------------------------------------")
			fmt.Fprintf(w, "Physical Sub-Kernels Verified (%d):\n", len(r.ValidationReceipt.Subkernels))
			for _, sk := range r.ValidationReceipt.Subkernels {
				parity := "PASSED"
				if !sk.Parity.Passed {
					parity = "FAIL"
				}
				fmt.Fprintf(w, "  * %-25s [%s] %6d µs (parity: %s, cosine: %.6f)\n",
					sk.Name, sk.Status, sk.DurationUS, parity, sk.Parity.LogitCosineSimilarity)
			}
		}
		if len(r.ValidationReceipt.Ablations) > 0 {
			fmt.Fprintln(w, "--------------------------------------------------------------------------------")
			fmt.Fprintf(w, "Differential Ablation Arms (%d):\n", len(r.ValidationReceipt.Ablations))
			for _, ab := range r.ValidationReceipt.Ablations {
				fmt.Fprintf(w, "  * %-28s [%s] speedup: %6.1fx (baseline: %d µs, candidate: %d µs)\n",
					ab.Feature, ab.Verdict, ab.Speedup, ab.BaselineArm.LatencyUS, ab.CandidateArm.LatencyUS)
			}
		}
	}

	if len(r.Failures) > 0 {
		fmt.Fprintln(w, "--------------------------------------------------------------------------------")
		fmt.Fprintf(w, "Failures (%d):\n", len(r.Failures))
		for _, f := range r.Failures {
			fmt.Fprintf(w, "  * %s\n", f)
		}
	}
	fmt.Fprintln(w, "================================================================================")
}
