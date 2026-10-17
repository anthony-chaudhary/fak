package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"os"
	"runtime"
	"runtime/debug"
	"sort"
	"strconv"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/compute"
	computestrix "github.com/anthony-chaudhary/fak/internal/compute/strix"
	"github.com/anthony-chaudhary/fak/internal/logitoracle"
	fakmodel "github.com/anthony-chaudhary/fak/internal/model"
)

const computeTraceLogitsUsage = "usage: fak compute-trace logits dump --gguf <model.gguf> --prompts <prompts.json> --out <dir> [--teacher <ref-dump>] | compare --ref <dump> --cand <dump> [--json]"

func runComputeTraceLogits(stdout, stderr io.Writer, args []string) int {
	if len(args) == 0 {
		fmt.Fprintln(stderr, computeTraceLogitsUsage)
		return 2
	}
	switch args[0] {
	case "dump":
		return runComputeTraceLogitsDump(stdout, stderr, args[1:])
	case "compare":
		return runComputeTraceLogitsCompare(stdout, stderr, args[1:])
	default:
		fmt.Fprintln(stderr, computeTraceLogitsUsage)
		return 2
	}
}

// runComputeTraceLogitsDump loads the model once through the serve loader and
// records full logit vectors for raw-token prompts. The process environment
// selects the configuration under test; it is captured in the manifest.
func runComputeTraceLogitsDump(stdout, stderr io.Writer, args []string) int {
	fs := flag.NewFlagSet("compute-trace logits dump", flag.ContinueOnError)
	fs.SetOutput(stderr)
	gguf := fs.String("gguf", "", "model checkpoint (required)")
	promptsPath := fs.String("prompts", "", "JSON list of {name, ids | text, positions}; text is encoded raw, with no chat template (required)")
	out := fs.String("out", "", "dump directory (required)")
	label := fs.String("label", "", "configuration label recorded in the manifest")
	backendName := fs.String("backend", "vulkan", "compute backend (cpu for the host reference)")
	positions := fs.Int("positions", 64, "logit vectors per prompt when the prompt sets none")
	chunk := fs.Int("prefill-chunk", 512, "prefill panel size; earlier panels skip logits, as the native planner does")
	teacher := fs.String("teacher", "", "reference dump whose prompt ids and fed tokens this run consumes")
	topK := fs.Int("topk", 5, "ranked tokens recorded per position")
	if err := fs.Parse(args); err != nil {
		return 2
	}
	if *gguf == "" || *promptsPath == "" || *out == "" || *positions < 1 || *chunk < 1 || *topK < 1 {
		fmt.Fprintln(stderr, computeTraceLogitsUsage)
		return 2
	}
	specs, err := readLogitPromptSpecs(*promptsPath)
	if err != nil {
		fmt.Fprintln(stderr, err)
		return 1
	}
	var teacherManifest *logitoracle.Manifest
	if *teacher != "" {
		if teacherManifest, err = logitoracle.ReadManifest(*teacher); err != nil {
			fmt.Fprintln(stderr, err)
			return 1
		}
	}
	backend, err := resolveServeChatBackend(*backendName)
	if err != nil {
		fmt.Fprintln(stderr, err)
		return 1
	}
	m, q4k, _, _ := loadServeInKernelModel(*gguf, backend, false, 0, nil, 1, nil)
	if m == nil {
		fmt.Fprintf(stderr, "fak compute-trace logits: failed to load %q\n", *gguf)
		return 1
	}
	defer m.CloseWeights()
	tok, haveTok := resolveServeTokenizer("", *gguf)
	var decode func([]int) string
	if haveTok && tok != nil {
		decode = func(ids []int) string {
			s, _ := tok.Decode(ids)
			return s
		}
	}
	for i := range specs {
		if len(specs[i].IDs) > 0 {
			continue
		}
		if !haveTok || tok == nil {
			fmt.Fprintf(stderr, "prompt %q has text but the checkpoint has no tokenizer\n", specs[i].Name)
			return 1
		}
		if specs[i].IDs, err = tok.Encode(specs[i].Text); err != nil {
			fmt.Fprintf(stderr, "prompt %q: %v\n", specs[i].Name, err)
			return 1
		}
	}
	halo, _, _ := computestrix.DetectPhysicalGFX1151("", "")
	newSession := func() (logitoracle.Session, func(), error) {
		var s *fakmodel.Session
		if backend != nil {
			var sessionErr error
			if s, sessionErr = m.NewBackendSessionChecked(backend); sessionErr != nil {
				return nil, nil, sessionErr
			}
		} else {
			s = m.NewSession()
		}
		if halo && backend != nil {
			s.SetExecutionPolicy(fakmodel.ExecutionPolicyDeviceOnly)
		}
		s.Quant, s.Q4K = true, q4k
		return s, s.Close, nil
	}
	identity := logitDumpIdentity(*gguf, backend, q4k, halo)
	manifest, err := logitoracle.Dump(specs, logitoracle.DumpOptions{
		Dir: *out, Label: *label, PrefillChunk: *chunk, DefaultPositions: *positions, TopK: *topK,
		Teacher: teacherManifest, NewSession: newSession, Decode: decode,
		Counters: logitRouteCounters(backend), Identity: identity, Env: logitDumpEnv(os.Environ()),
	})
	if err != nil {
		fmt.Fprintln(stderr, err)
		return 1
	}
	addLogitVerdicts(manifest, backend)
	if err := manifest.Write(); err != nil {
		fmt.Fprintln(stderr, err)
		return 1
	}
	for _, p := range manifest.Prompts {
		fmt.Fprintf(stdout, "prompt=%s prompt_tokens=%d positions=%d forced=%t prefill_ms=%.1f decode_ms=%.1f counters=%v\n",
			p.Name, len(p.PromptIDs), p.Positions, p.Forced, float64(p.PrefillNS)/1e6, float64(p.DecodeNS)/1e6, p.Counters)
	}
	fmt.Fprintf(stdout, "wrote %s vocab=%d prompts=%d\n", manifest.Dir(), manifest.Vocab, len(manifest.Prompts))
	return 0
}

func readLogitPromptSpecs(path string) ([]logitoracle.PromptSpec, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	var specs []logitoracle.PromptSpec
	if err := json.Unmarshal(data, &specs); err != nil {
		return nil, fmt.Errorf("%s: %w", path, err)
	}
	if len(specs) == 0 {
		return nil, logitoracle.ErrNoPrompts
	}
	for _, s := range specs {
		if len(s.IDs) == 0 && s.Text == "" {
			return nil, fmt.Errorf("prompt %q needs ids or text", s.Name)
		}
	}
	return specs, nil
}

// logitDumpEnv keeps the configuration-selecting variables and drops anything
// that names a credential.
func logitDumpEnv(environ []string) map[string]string {
	out := map[string]string{}
	for _, kv := range environ {
		k, v, ok := strings.Cut(kv, "=")
		if !ok {
			continue
		}
		upper := strings.ToUpper(k)
		if !strings.HasPrefix(upper, "FAK_") && !strings.HasPrefix(upper, "VK_") && !strings.HasPrefix(upper, "RADV_") &&
			upper != "GOMAXPROCS" && upper != "GOMEMLIMIT" {
			continue
		}
		if strings.Contains(upper, "KEY") || strings.Contains(upper, "TOKEN") || strings.Contains(upper, "SECRET") ||
			strings.Contains(upper, "PASSWORD") || strings.Contains(upper, "CREDENTIAL") {
			continue
		}
		out[k] = v
	}
	return out
}

func logitDumpIdentity(gguf string, backend compute.Backend, q4k, halo bool) map[string]string {
	id := map[string]string{"model_path": gguf, "q4k": strconv.FormatBool(q4k), "halo": strconv.FormatBool(halo),
		"goos": runtime.GOOS, "goarch": runtime.GOARCH, "backend": "cpu"}
	if info, err := os.Stat(gguf); err == nil {
		id["model_bytes"] = strconv.FormatInt(info.Size(), 10)
	}
	if backend != nil {
		id["backend"], id["device"] = backend.Name(), backend.Tier()
	}
	if bi, ok := debug.ReadBuildInfo(); ok {
		for _, s := range bi.Settings {
			switch s.Key {
			case "vcs.revision", "vcs.modified", "-tags":
				id["build_"+strings.TrimPrefix(s.Key, "-")] = s.Value
			}
		}
	}
	return id
}

type logitGDNTiledCounters interface {
	VulkanDebugGDNTiledPrefillProfileSnapshot() (tiledCalls, scalarCalls int64)
}

type logitGDNVerifyCounters interface {
	VulkanDebugGDNVerifyTiledProfileSnapshot() (verifyCalls, scalarCalls int64)
}

type logitGDNVectorCounters interface {
	VulkanDebugGDNProfileSnapshot() (vectorCalls, scalarCalls int64)
}

type logitGDNOptOut interface {
	VulkanDebugGDNOptOutCalls() int64
}

type logitGDNVerdicts interface {
	VulkanDebugGDNVerifyTiledVerdict() (admitted bool, deviation float32, checked bool)
	VulkanDebugGDNTiledVerdict() (admitted bool, deviation float32)
}

// logitRouteCounters samples the GDN route and mm_cm1 counters so each dump
// names the kernels that actually ran in prefill and in decode.
func logitRouteCounters(backend compute.Backend) func() map[string]int64 {
	return func() map[string]int64 {
		c := map[string]int64{}
		if b, ok := backend.(logitGDNTiledCounters); ok {
			c["gdn_register_resident_calls"], c["gdn_stock_calls"] = b.VulkanDebugGDNTiledPrefillProfileSnapshot()
		}
		if b, ok := backend.(logitGDNVerifyCounters); ok {
			c["gdn_verify_tiled_calls"], c["gdn_verify_width_stock_calls"] = b.VulkanDebugGDNVerifyTiledProfileSnapshot()
		}
		if b, ok := backend.(logitGDNOptOut); ok {
			c["gdn_optout_calls"] = b.VulkanDebugGDNOptOutCalls()
		}
		if b, ok := backend.(logitGDNVectorCounters); ok {
			c["gdn_device_calls"], c["gdn_host_scalar_calls"] = b.VulkanDebugGDNProfileSnapshot()
		}
		if backend != nil && backend.Name() == "vulkan" {
			s := compute.VulkanMMCM1Snapshot()
			c["mm_cm1_gemm"], c["mm_cm1_disabled"], c["mm_cm1_below_threshold"] = int64(s.GEMM), int64(s.Disabled), int64(s.BelowThreshold)
		}
		return c
	}
}

func addLogitVerdicts(m *logitoracle.Manifest, backend compute.Backend) {
	b, ok := backend.(logitGDNVerdicts)
	if !ok {
		return
	}
	if m.Identity == nil {
		m.Identity = map[string]string{}
	}
	admitted, dev, checked := b.VulkanDebugGDNVerifyTiledVerdict()
	m.Identity["gdn_verify_selfcheck"] = fmt.Sprintf("admitted=%t checked=%t deviation=%g", admitted, checked, dev)
	admitted, dev = b.VulkanDebugGDNTiledVerdict()
	m.Identity["gdn_tiled_selfcheck"] = fmt.Sprintf("admitted=%t deviation=%g", admitted, dev)
}

func runComputeTraceLogitsCompare(stdout, stderr io.Writer, args []string) int {
	fs := flag.NewFlagSet("compute-trace logits compare", flag.ContinueOnError)
	fs.SetOutput(stderr)
	ref := fs.String("ref", "", "reference dump directory or manifest (required)")
	cand := fs.String("cand", "", "candidate dump directory or manifest (required)")
	th := logitoracle.FlipThresholds
	fs.Float64Var(&th.CosineMin, "cosine-min", th.CosineMin, "minimum cosine at every compared position")
	fs.Float64Var(&th.RelL2Max, "rel-l2-max", th.RelL2Max, "maximum relative L2 at every compared position")
	fs.Float64Var(&th.TieEpsilon, "tie-epsilon", th.TieEpsilon, "logit gap below which a greedy mismatch is a tie in both distributions")
	fs.IntVar(&th.TopK, "topk", th.TopK, "k for top-k overlap")
	jsonOut := fs.Bool("json", false, "print the fak.logit-oracle.compare/v1 report as JSON")
	reportOut := fs.String("out", "", "also write the JSON report to this path")
	if err := fs.Parse(args); err != nil {
		return 2
	}
	if *ref == "" || *cand == "" {
		fmt.Fprintln(stderr, computeTraceLogitsUsage)
		return 2
	}
	rm, err := logitoracle.ReadManifest(*ref)
	if err != nil {
		fmt.Fprintln(stderr, err)
		return 1
	}
	cm, err := logitoracle.ReadManifest(*cand)
	if err != nil {
		fmt.Fprintln(stderr, err)
		return 1
	}
	rep, err := logitoracle.CompareManifests(rm, cm, th)
	if err != nil {
		fmt.Fprintln(stderr, err)
		return 1
	}
	data, err := json.MarshalIndent(rep, "", "  ")
	if err != nil {
		fmt.Fprintln(stderr, err)
		return 1
	}
	if *reportOut != "" {
		if err := os.WriteFile(*reportOut, append(data, '\n'), 0o644); err != nil {
			fmt.Fprintln(stderr, err)
			return 1
		}
	}
	if *jsonOut {
		fmt.Fprintln(stdout, string(data))
	} else {
		writeLogitCompareText(stdout, rep, rm, cm)
	}
	if rep.Verdict != logitoracle.VerdictPass {
		return 1
	}
	return 0
}

func writeLogitCompareText(w io.Writer, rep logitoracle.Report, rm, cm *logitoracle.Manifest) {
	for _, p := range rep.Prompts {
		fmt.Fprintf(w, "prompt=%s compared=%d excluded=%d cos_min=%.8f rel_l2_max=%.3g rel_l2_med=%.3g kl_max=%.3g top1=%d/%d top%d_overlap_mean=%.3f greedy=%s gate=%t strict=%t\n",
			p.Name, p.Compared, p.Excluded, p.CosineMin, p.RelL2Max, p.RelL2Median, p.KLMax, p.Top1Agree, p.Compared,
			rep.Thresholds.TopK, p.TopKOverlapMean, p.Greedy, p.GatePass, p.StrictPass)
		mismatches := append([]logitoracle.PositionStats(nil), p.Mismatches...)
		sort.Slice(mismatches, func(i, j int) bool { return mismatches[i].Position < mismatches[j].Position })
		for _, st := range mismatches {
			fmt.Fprintf(w, "  mismatch position=%d ref_top1=%d%s cand_top1=%d%s ref_gap=%.4f cand_gap=%.4f ref_top2_margin=%.4f tie=%t\n",
				st.Position, st.RefTop1, logitPiece(rm, p.Name, st.Position, st.RefTop1), st.CandTop1,
				logitPiece(cm, p.Name, st.Position, st.CandTop1), st.RefGap, st.CandGap, st.RefTop2Margin, st.Tie)
		}
	}
	fmt.Fprintf(w, "verdict=%s strict_verdict=%s greedy=%s cosine_min=%g rel_l2_max=%g tie_epsilon=%g\n",
		rep.Verdict, rep.StrictVerdict, rep.Greedy, rep.Thresholds.CosineMin, rep.Thresholds.RelL2Max, rep.Thresholds.TieEpsilon)
}

func logitPiece(m *logitoracle.Manifest, prompt string, position, id int) string {
	p, err := m.Prompt(prompt)
	if err != nil || position >= len(p.Top) {
		return ""
	}
	t := p.Top[position]
	for i, tid := range t.IDs {
		if tid == id && i < len(t.Pieces) {
			return strconv.Quote(t.Pieces[i])
		}
	}
	return ""
}
