// serve_fit_auto.go implements `fak serve --fit auto` (T04, ticket
// model-fit-auto-planner-20261023-04): the joint model-fit search over context x
// KV precision x slots scored against the stated workload and SLO. It reuses the
// exact header-derived inputs the --plan-json artifact sizes (allocation-free,
// pre-load), so the reported numbers are the admission numbers.
//
// Pin precedence: any explicitly set sizing flag fixes its axis exactly; the
// search never moves a pinned axis, and an infeasible pin refuses with
// PIN_INFEASIBLE instead of being silently changed. A refusal on a live boot
// exits 2; a chosen row is applied to the UNPINNED axes only.
package main

import (
	"fmt"
	"os"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/compute"
	"github.com/anthony-chaudhary/fak/internal/fitplan"
	"github.com/anthony-chaudhary/fak/internal/ggufload"
)

// serveFitAutoRequest is the operator-stated load and SLO the T04 search gates
// against, with defaults filled: unstated workload reads as the documented
// 2048/512 single-stream point, and zero SLO means unconstrained.
type serveFitAutoRequest struct {
	Enabled  bool
	Workload fitplan.SearchWorkload
	SLO      fitplan.SearchSLO
}

// serveFitAutoRequestFromFlags assembles the request from the --workload-* and
// --slo-* flags, applying the documented defaults for unset values.
func serveFitAutoRequestFromFlags(fit string, prompt, output, concurrency int, tpotMS, ttftMS float64) serveFitAutoRequest {
	req := serveFitAutoRequest{Enabled: fit == "auto"}
	if !req.Enabled {
		return req
	}
	req.Workload = fitplan.SearchWorkload{
		PromptTokens: prompt,
		OutputTokens: output,
		Concurrency:  concurrency,
	}
	if req.Workload.PromptTokens <= 0 {
		req.Workload.PromptTokens = fitplan.DefaultPromptTokens
	}
	if req.Workload.OutputTokens <= 0 {
		req.Workload.OutputTokens = fitplan.DefaultOutputTokens
	}
	if req.Workload.Concurrency <= 0 {
		req.Workload.Concurrency = 1
	}
	req.SLO = fitplan.SearchSLO{TPOTMS: tpotMS, TTFTMS: ttftMS}
	return req
}

// validateServeFitFlags refuses unknown --fit tokens and negative workload/SLO
// values at startup, before any pool probe.
func validateServeFitFlags(fit string, prompt, output, concurrency int, tpotMS, ttftMS float64) error {
	switch fit {
	case "", "off", "auto":
	default:
		return fmt.Errorf("--fit %q is invalid (want off or auto)", fit)
	}
	if prompt < 0 || output < 0 || concurrency < 0 {
		return fmt.Errorf("--workload-* values must be >= 0 (got prompt=%d output=%d concurrency=%d)", prompt, output, concurrency)
	}
	if tpotMS < 0 || ttftMS < 0 {
		return fmt.Errorf("--slo-tpot-ms/--slo-ttft-ms must be >= 0 (got tpot=%v ttft=%v)", tpotMS, ttftMS)
	}
	return nil
}

// serveFitAutoPins derives the search pins from the EXPLICITLY set sizing flags:
// a pin fixes its axis exactly, so the search may not move it. Unset flags stay
// open for the search to choose.
func serveFitAutoPins(sf *serveFlags) fitplan.SearchPins {
	var pins fitplan.SearchPins
	if sf.isExplicitFlag("native-context-tokens") && sf.nativeContextTokens != nil && *sf.nativeContextTokens > 0 {
		pins.ContextTokens = *sf.nativeContextTokens
	}
	if sf.isExplicitFlag("kv-precision") && sf.kvPrecision != nil {
		switch strings.ToLower(strings.TrimSpace(*sf.kvPrecision)) {
		case "q8_0", "q8":
			pins.KVPrecision = "q8"
		case "f32":
			pins.KVPrecision = "f32"
		}
	}
	return pins
}

// buildServeSizingFitAuto runs the T04 joint search on the SAME header-derived
// inputs the --plan-json artifact sizes: the selected arm's weights plan, the
// budgets the context resolves against, the two KV tiers the engine can store,
// and the calibration row for this epoch/arm. Allocation-free and pre-load.
func buildServeSizingFitAuto(ws *ggufload.WeightSource, be compute.Backend, cpuOffloadExperts bool, requestedNativeContextTokens int, model string, req serveFitAutoRequest, pins fitplan.SearchPins) (*fitplan.SearchReport, error) {
	pq2Checkpoint := servePQ2Artifact(ggufload.ClassifyTensorQuant(ws.File.Tensors))
	useMetal := be == nil && !pq2Checkpoint && serveMetalAvailable()
	selectedArm := resolveServeNativeContextLoadArm(ws, be, cpuOffloadExperts, useMetal)
	cpuOffloadActive := false
	if be != nil && cpuOffloadExperts {
		if ok, err := serveArtifactCPUOffloadExperts(ws); ok && err == nil {
			cpuOffloadActive = true
		}
	}
	weights, fit, err := serveNativeContextSizingInputs(ws, be, cpuOffloadExperts, useMetal, 1, model)
	if err != nil {
		return nil, err
	}
	resolution, _, _, err := resolveServeNativeContextBudget(ws, be, weights, fit, requestedNativeContextTokens)
	if err != nil {
		return nil, err
	}
	armLabel := serveSizingArmForSelected(be, cpuOffloadActive, useMetal, selectedArm)
	if pq2Checkpoint && be == nil && selectedArm == serveLoadArmResidentQ4K {
		armLabel = "cpu-resident-pq2_0"
	}
	embed, q4kOpts := serveFitPlanEmbed(ws, be, selectedArm, cpuOffloadActive, model, fit)
	m, err := ws.FitPlanModel(embed, q4kOpts...)
	if err != nil {
		return nil, err
	}
	cfg, cfgErr := ws.File.Config()
	if cfgErr != nil {
		return nil, cfgErr
	}
	cur := serveKVPrecision()
	other := compute.KVPrecisionQ8
	if cur == compute.KVPrecisionQ8 {
		other = compute.KVPrecisionF32
	}
	// Label each tier by the storage the geometry realizes, mirroring
	// buildServeSizingFitPlan: a model may force F32 KV whatever the knob asks.
	curCSC, otherCSC := cfg.ContextSizeConfigWithPrecision(cur), cfg.ContextSizeConfigWithPrecision(other)
	tiers := []fitplan.KVTier{
		{Precision: curCSC.KV.Precision.String(), Context: curCSC},
		{Precision: otherCSC.KV.Precision.String(), Context: otherCSC},
	}
	hostTotal, hostFree, hostKnown := compute.HostSystemMemoryInfo()
	hostFit := serveHostFitBudgetFromReported(hostTotal, hostFree, hostKnown, nil)
	if be == nil {
		hostFit = fit
	}
	hw := serveFitPlanHardware(be, fit, hostFit)
	hw.Peaks, _ = fitplan.DevicePeaks(serveFitPlanArch(be))
	rep := fitplan.Search(fitplan.SearchInput{
		Model:           m,
		Weights:         weights,
		Hardware:        hw,
		Calibration:     resolveServeCalibration(be, m, armLabel),
		Tiers:           tiers,
		ResolvedContext: resolution.ResolvedTokens,
		CurrentKV:       curCSC.KV.Precision.String(),
		Workload:        req.Workload,
		SLO:             req.SLO,
		Pins:            pins,
	})
	return &rep, nil
}

// applyServeFitAutoChoice writes the chosen row back onto the boot config for
// every axis the operator did NOT pin: the explicit flag wins, the search fills
// the rest. Pinned axes are left byte-for-byte as the operator set them.
func applyServeFitAutoChoice(sf *serveFlags, rep *fitplan.SearchReport, pins fitplan.SearchPins) {
	ch := rep.Chosen
	if ch == nil || sf == nil {
		return
	}
	if pins.ContextTokens == 0 && sf.nativeContextTokens != nil {
		*sf.nativeContextTokens = ch.ContextTokens
	}
	if pins.KVPrecision == "" && sf.kvPrecision != nil {
		switch strings.ToLower(ch.KVPrecision) {
		case "q8":
			*sf.kvPrecision = "q8_0"
		case "f32":
			*sf.kvPrecision = "f32"
		}
	}
}

// printServeFitAutoSummary is the compact stderr receipt a live boot prints:
// the chosen row (or the typed refusal) plus the top alternatives.
func printServeFitAutoSummary(w *os.File, rep *fitplan.SearchReport, req serveFitAutoRequest) {
	wl := req.Workload
	if rep.Chosen != nil {
		ch := rep.Chosen
		fmt.Fprintf(w, "fak serve --fit auto: chosen kv=%s ctx=%d slots=%d B*=%d ~%.1f tok/s ttft=%.2fs (workload %d/%d x%d)\n",
			ch.KVPrecision, ch.ContextTokens, ch.Slots, ch.BStar, ch.AggregateTokS, ch.TTFTSeconds, wl.PromptTokens, wl.OutputTokens, wl.Concurrency)
		for i, a := range rep.Alternatives {
			fmt.Fprintf(w, "  alt %d: kv=%s ctx=%d slots=%d B*=%d ~%.1f tok/s\n", i+1, a.KVPrecision, a.ContextTokens, a.Slots, a.BStar, a.AggregateTokS)
		}
		return
	}
	if rep.Refusal != nil {
		fmt.Fprintf(w, "fak serve --fit auto: REFUSED %s (workload %d/%d x%d)\n", rep.Refusal, wl.PromptTokens, wl.OutputTokens, wl.Concurrency)
	} else {
		fmt.Fprintf(w, "fak serve --fit auto: status=%s omitted=%v (workload %d/%d x%d)\n", rep.Status, rep.Omitted, wl.PromptTokens, wl.OutputTokens, wl.Concurrency)
	}
}

// runServeFitAutoLive is the live-boot hook: resolve the backend and open the
// header the way --plan-json does, run the search, print the receipt, and apply
// the choice to the unpinned axes. Exit policy keeps the T01 fail-open split:
// a TYPED REFUSAL exits 2 — a live boot never proceeds on a configuration the
// planner refused; an UNPROBEABLE pool (StatusUnknown, no Refusal) warns and
// lets the boot proceed on today's sizing, exactly as an unknown pool does
// everywhere else in the planner.
func runServeFitAutoLive(sf *serveFlags) {
	if strings.TrimSpace(derefString(sf.ggufPath)) == "" {
		writeConfigBail(os.Stderr, configBail{
			Verb:    "fak serve",
			Reason:  bailWeightsRequired,
			Summary: "--fit auto requires --gguf WEIGHTS (the search is header-derived)",
			Knobs: []bailKnob{
				bailFlag("gguf", "").want("a GGUF path, an hf:// URI, or a registry alias"),
			},
			Check: "fak ls   # the locally cached models --gguf can name",
		})
		os.Exit(2)
	}
	be, err := resolveServeChatBackend(*sf.backendName)
	if err != nil {
		writeBackendUnavailableBail(os.Stderr, "fak serve", *sf.backendName)
		os.Exit(2)
	}
	ws, err := ggufload.OpenWeights(*sf.ggufPath)
	if err != nil {
		fmt.Fprintln(os.Stderr, "fak serve: --fit auto:", err)
		os.Exit(1)
	}
	defer ws.Close()
	req := serveFitAutoRequestFromFlags(*sf.fit, *sf.workloadPromptTokens, *sf.workloadOutputTokens, *sf.workloadConcurrency, *sf.sloTPOTMS, *sf.sloTTFTMS)
	pins := serveFitAutoPins(sf)
	rep, err := buildServeSizingFitAuto(ws, be, *sf.cpuOffloadExperts, *sf.nativeContextTokens, *sf.ggufPath, req, pins)
	if err != nil {
		fmt.Fprintln(os.Stderr, "fak serve: --fit auto:", err)
		os.Exit(1)
	}
	printServeFitAutoSummary(os.Stderr, rep, req)
	switch {
	case rep.Status == fitplan.StatusRefused:
		fmt.Fprintf(os.Stderr, "fak serve: --fit auto refused: %v\n", rep.Refusal)
		os.Exit(2)
	case rep.Status == fitplan.StatusUnknown:
		fmt.Fprintf(os.Stderr, "fak serve: --fit auto: pool unprobeable (%v); proceeding on today's sizing\n", rep.Omitted)
	default:
		applyServeFitAutoChoice(sf, rep, pins)
	}
}
