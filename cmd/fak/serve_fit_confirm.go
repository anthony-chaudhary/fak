package main

// Host-side two-phase fit confirm (ticket T03):
//
// Item 1 — post-load fit_confirm record. buildServeFitConfirm buckets the committed
// memory plan into the per-class rows (weights / kv / compute-staging / ring /
// transient) and prices them against the POST-LOAD capacity reading; the record is
// attached to the gateway load profile (JSON on /debug/vars) and emitted on one
// startup log line by withServeGGUFMemoryProfile.
//
// Item 2 — first-N-token decode confirm. The gateway warmup runs one tagged probe
// completion after the boot warmup succeeds (gateway.Server.runDecodeConfirm,
// default N = 32, opt-out FAK_DECODE_CONFIRM_TOKENS=0). installServeDecodeConfirmSink
// wires its result into recordServeDecodeConfirm, which compares the measured
// ms/step against the fitplan nominal for the same layout and writes the measured
// point into the local calibration store as a SILICON-tier row (a ReceiptDecode
// whose TierHint marks the fitted row fitplan.TierSilicon), so the next plan prices
// this machine's own engine instead of a foreign calibrated line.

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/compute"
	"github.com/anthony-chaudhary/fak/internal/fitplan"
	"github.com/anthony-chaudhary/fak/internal/gateway"
	"github.com/anthony-chaudhary/fak/internal/ggufload"
)

// serveConfirmClassFor buckets one memory-plan row into a fit_confirm class. The
// Detail label wins over the class where it names a more specific bucket: the
// v41 forward-transient scratch is a scratchpad-class row but a TRANSIENT charge,
// and the activated-expert ring rows are weights-class bytes that the operator
// reads as the RING. Everything else lands in compute/staging (scratchpads,
// activations, offload staging, DDR transit).
func serveConfirmClassFor(d compute.MemoryDemand) fitplan.ConfirmClass {
	detail := strings.ToLower(d.Detail)
	switch {
	case strings.Contains(detail, "transient"):
		return fitplan.ConfirmTransient
	case strings.Contains(detail, "expert-ring"):
		return fitplan.ConfirmRing
	case d.Class == compute.MemoryKVCache:
		return fitplan.ConfirmKV
	case d.Class == compute.MemoryWeights:
		return fitplan.ConfirmWeights
	default:
		return fitplan.ConfirmComputeStaging
	}
}

// serveConfirmPlannedByClass sums one scope's plan rows per confirm class.
func serveConfirmPlannedByClass(plan compute.MemoryPlan, scope compute.MemoryScope) map[fitplan.ConfirmClass]int64 {
	out := make(map[fitplan.ConfirmClass]int64)
	for _, d := range plan {
		if d.Bytes <= 0 || d.ScopeOrDefault() != scope {
			continue
		}
		out[serveConfirmClassFor(d)] += d.Bytes
	}
	return out
}

// buildServeFitConfirm computes the post-load two-phase fit confirmation for the
// committed plan: per scope, the planned charge per class against the post-load
// capacity reading, plus unaccounted = total - free_after - self. It returns nil
// when there is no plan to confirm or the record fails validation (the load
// proceeds either way: the confirm is evidence, not a gate).
func buildServeFitConfirm(plan compute.MemoryPlan, be compute.Backend, ggufPath string) *gateway.ModelLoadFitConfirm {
	if len(plan) == 0 {
		return nil
	}
	deviceTotal, deviceFree, deviceKnown := int64(0), int64(-1), false
	hostTotal, hostFree, hostKnown := int64(0), int64(-1), false
	if be != nil {
		deviceTotal, deviceFree, deviceKnown = compute.DeviceMemoryInfo(be)
		hostTotal, hostFree, hostKnown = compute.HostMemoryInfo(be)
	} else {
		hostTotal, hostFree, hostKnown = compute.HostSystemMemoryInfo()
	}
	backendName := ""
	if be != nil {
		backendName = be.Name()
	}
	rec := fitplan.BuildFitConfirm(filepath.Base(ggufPath), backendName,
		fitplan.ConfirmInput{
			Scope:     "device",
			Total:     deviceTotal,
			FreeAfter: deviceFree,
			Known:     deviceKnown,
			FreeKnown: deviceKnown && deviceFree >= 0,
			Planned:   serveConfirmPlannedByClass(plan, compute.MemoryScopeDevice),
		},
		fitplan.ConfirmInput{
			Scope:     "host",
			Total:     hostTotal,
			FreeAfter: hostFree,
			Known:     hostKnown,
			FreeKnown: hostKnown && hostFree >= 0,
			Planned:   serveConfirmPlannedByClass(plan, compute.MemoryScopeHost),
		},
	)
	if err := rec.Validate(); err != nil {
		fmt.Fprintf(os.Stderr, "fak serve: fit_confirm record failed validation: %v\n", err)
		return nil
	}
	out := &gateway.ModelLoadFitConfirm{Schema: rec.Schema, Model: rec.Model, Backend: rec.Backend}
	for _, s := range rec.Scopes {
		row := gateway.ModelLoadFitScopeRow{
			Scope:            s.Scope,
			TotalBytes:       s.TotalBytes,
			FreeAfterLoad:    s.FreeAfterLoad,
			FreeKnown:        s.FreeKnown,
			UnaccountedBytes: s.UnaccountedBytes,
		}
		for _, c := range s.Classes {
			row.Classes = append(row.Classes, gateway.ModelLoadFitClassRow{Class: string(c.Class), PlannedBytes: c.PlannedBytes, AllocatedBytes: c.AllocatedBytes})
		}
		out.Scopes = append(out.Scopes, row)
	}
	return out
}

// attachServeFitConfirm builds and attaches the post-load fit_confirm record to
// the load profile and emits the one startup log line. It is the single call
// site withServeGGUFMemoryProfile makes, so every device arm that carries a
// memory plan confirms exactly once per boot.
func attachServeFitConfirm(p *gateway.ModelLoadProfile, plan compute.MemoryPlan, be compute.Backend, ggufPath string) {
	if p == nil {
		return
	}
	rec := buildServeFitConfirm(plan, be, ggufPath)
	if rec == nil {
		return
	}
	p.FitConfirm = rec
	line := fitplanTextOf(rec)
	fmt.Fprintln(os.Stderr, "fak serve: "+line)
	p.Messages = append(p.Messages, gateway.StartupMessage{Source: "model-load", Kind: "fit-confirm", Level: "info", Text: line})
}

// fitplanTextOf renders the gateway mirror back into the canonical log-line form
// without importing fitplan into the gateway package.
func fitplanTextOf(rec *gateway.ModelLoadFitConfirm) string {
	var b strings.Builder
	b.WriteString("fit_confirm")
	if rec.Model != "" {
		fmt.Fprintf(&b, " model=%s", rec.Model)
	}
	if rec.Backend != "" {
		fmt.Fprintf(&b, " backend=%s", rec.Backend)
	}
	for _, s := range rec.Scopes {
		fmt.Fprintf(&b, " | %s:", s.Scope)
		if !s.FreeKnown {
			b.WriteString(" unprobed")
			continue
		}
		fmt.Fprintf(&b, " total=%d free_after=%d", s.TotalBytes, s.FreeAfterLoad)
		for _, c := range s.Classes {
			fmt.Fprintf(&b, " %s=%d", c.Class, c.AllocatedBytes)
		}
		fmt.Fprintf(&b, " unaccounted=%d", s.UnaccountedBytes)
	}
	return b.String()
}

// ---------------------------------------------------------------------------
// Item 2: the first-N-token decode confirm -> SILICON calibration row.
// ---------------------------------------------------------------------------

// installServeDecodeConfirmSink wires the gateway's warmup-time decode confirm
// into the calibration-store writer. It must run before ArmWarmupGate so the
// sink is live when RunWarmup reaches the probe. A serve with no local in-kernel
// model never arms the warmup gate, so the sink is never invoked.
func (rt *serveRuntime) installServeDecodeConfirmSink(sf *serveFlags) {
	if rt.srv == nil || rt.chatBackend == nil || strings.TrimSpace(*sf.ggufPath) == "" {
		return
	}
	rt.srv.SetDecodeConfirmSink(func(res gateway.DecodeConfirmResult) {
		rt.recordServeDecodeConfirm(res, *sf.ggufPath, *sf.cpuOffloadExperts)
	})
}

// serveDecodeConfirmArmLabel names the kernel path the loaded model actually
// ran, from the load profile's mode label: the same closed vocabulary the
// sizing path keys its calibration rows by.
func serveDecodeConfirmArmLabel(mode string, cpuOffloadExperts bool) string {
	switch mode {
	case "gguf-resident-q4k-device":
		if cpuOffloadExperts {
			return "device-resident-q4k-cpu-offload-experts"
		}
		return "device-resident-q4k"
	case "gguf-lean-q8-device":
		return "device-lean-q8"
	case "gguf-f32-device":
		return "device-f32"
	case "gguf-resident-q4k-host":
		return "cpu-resident-q4k-cpu-offload-experts"
	case "gguf-resident-q4k":
		return "cpu-resident-q4k"
	default:
		return "serve-" + mode
	}
}

// recordServeDecodeConfirm compares the measured ms/step against the fitplan
// nominal for the committed layout and writes the measured point into the local
// calibration store as a SILICON-tier row. Every failure path logs and returns:
// the confirm is evidence, never a boot gate, and warmup already proved the
// backend decodes.
func (rt *serveRuntime) recordServeDecodeConfirm(res gateway.DecodeConfirmResult, ggufPath string, cpuOffloadExperts bool) {
	be := rt.chatBackend
	if res.Tokens < 2 || res.MsPerStep <= 0 {
		return
	}
	arm := ""
	if rt.lastLoadProfile != nil {
		arm = serveDecodeConfirmArmLabel(rt.lastLoadProfile.Mode, cpuOffloadExperts)
	}
	if arm == "" {
		arm = "serve-decode-confirm"
	}
	nominalMS, bytesPerStep, chosen, key, quantFamily, ok := rt.serveDecodeConfirmNominal(ggufPath, be, arm, cpuOffloadExperts)
	if !ok {
		fmt.Fprintf(os.Stderr, "fak serve: decode_confirm: no fitplan nominal available (%s); measurement not recorded\n", chosen)
		return
	}
	gapPct := (res.MsPerStep - nominalMS) / nominalMS * 100
	receipt := fitplan.CalibrationReceipt{
		Key:          key,
		Kind:         fitplan.ReceiptDecode,
		Label:        fmt.Sprintf("decode-confirm n=%d gap=%.1f%%", res.Tokens, gapPct),
		TierHint:     fitplan.TierSilicon,
		BytesPerStep: float64(bytesPerStep),
		StepSeconds:  res.MsPerStep / 1000,
		MeasuredAt:   time.Now().UTC().Format(time.RFC3339),
		Provenance:   fitplan.Provenance{Path: "fak-serve-decode-confirm", SHA256: serveDecodeConfirmSHA256(ggufPath, quantFamily, arm, res)},
	}
	if err := receipt.Validate(); err != nil {
		fmt.Fprintf(os.Stderr, "fak serve: decode_confirm receipt invalid: %v\n", err)
		return
	}
	path := fitPlanCalibrationPath()
	_, added, err := ingestFitPlanCalibration(path, []fitplan.CalibrationReceipt{receipt})
	if err != nil {
		fmt.Fprintf(os.Stderr, "fak serve: decode_confirm: write calibration store %s: %v\n", path, err)
		return
	}
	state := "recorded"
	if added == 0 {
		state = "already present"
	}
	fmt.Fprintf(os.Stderr, "fak serve: decode_confirm n=%d measured=%.1fms/step nominal=%.1fms/step gap=%+.1f%% -> %s (%s)\n",
		res.Tokens, res.MsPerStep, nominalMS, gapPct, path, state)
}

// serveDecodeConfirmNominal re-prices the committed layout through the SAME
// planner the sizing path uses and returns its nominal ms/step, the predicted
// bytes one step streams, and the calibration key the measured point belongs on.
// Inputs captured at boot (the sizing budget, the resolved context) are exact;
// the model facts are re-derived from the GGUF header with the arm's embedding
// storage. ok=false names why the nominal is unavailable (chosen states the
// report status/refusal).
func (rt *serveRuntime) serveDecodeConfirmNominal(ggufPath string, be compute.Backend, arm string, cpuOffloadExperts bool) (nominalMS float64, bytesPerStep int64, chosen string, key fitplan.CalibrationKey, quantFamily string, ok bool) {
	key = fitplan.CalibrationKey{
		Epoch:       fitPlanHardwareEpoch(be).Digest(),
		Backend:     fitPlanEngineBackend(be),
		KernelPath:  arm,
		BatchBucket: fitplan.BatchBucket(fitplan.DefaultWorkload().Concurrency),
	}
	ws, err := ggufload.OpenWeights(ggufPath)
	if err != nil {
		return 0, 0, "open weights: " + err.Error(), key, "", false
	}
	defer ws.Close()
	embed, q4kOpts := serveFitPlanEmbed(ws, be, serveDecodeConfirmLoadArm(arm), cpuOffloadExperts, ggufPath, rt.sizingFit)
	m, err := ws.FitPlanModel(embed, q4kOpts...)
	if err != nil {
		return 0, 0, "model facts: " + err.Error(), key, "", false
	}
	rep := buildServeSizingFitPlan(ws, be, serveFitPlanSizing{
		weights:    rt.contextWeights,
		embed:      embed,
		q4kOpts:    q4kOpts,
		sizingFit:  rt.sizingFit,
		resolution: rt.nativeContext,
		arm:        arm,
	})
	if rep == nil {
		return 0, 0, "no report", key, m.QuantFamily, false
	}
	if rep.Chosen == nil || rep.Chosen.DecodeTokSPerSeq <= 0 {
		why := string(rep.Status)
		if rep.Refusal != nil {
			why += " refusal=" + string(rep.Refusal.Reason)
		}
		return 0, 0, why, key, m.QuantFamily, false
	}
	nominalMS = 1000 / rep.Chosen.DecodeTokSPerSeq
	return nominalMS, rep.Chosen.DecodeBytesPerStep, "ok", key, m.QuantFamily, true
}

// serveDecodeConfirmLoadArm maps the kernel-path label back to the load arm whose
// embedding storage the model facts must be priced with.
func serveDecodeConfirmLoadArm(arm string) serveLoadArm {
	switch arm {
	case "device-lean-q8":
		return serveLoadArmQuantProfileQ8
	case "device-f32":
		return serveLoadArmF32
	default:
		return serveLoadArmResidentQ4K
	}
}

// serveDecodeConfirmSHA256 stamps the receipt's provenance from the measurement
// itself: two boots measuring the same thing dedupe, a different model or a
// different step rate does not.
func serveDecodeConfirmSHA256(ggufPath, quantFamily, arm string, res gateway.DecodeConfirmResult) string {
	raw, _ := json.Marshal(struct {
		Model       string  `json:"model"`
		QuantFamily string  `json:"quant_family"`
		Arm         string  `json:"arm"`
		Tokens      int     `json:"tokens"`
		MsPerStep   float64 `json:"ms_per_step"`
	}{filepath.Base(ggufPath), quantFamily, arm, res.Tokens, res.MsPerStep})
	sum := sha256.Sum256(raw)
	return hex.EncodeToString(sum[:])
}
