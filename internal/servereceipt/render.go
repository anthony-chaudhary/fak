package servereceipt

import (
	"bytes"
	"encoding/json"
	"fmt"
	"strconv"
	"strings"
)

// comma inserts thousands separators for readable receipt prose (md output
// only; JSON carries raw integers).
func comma(n int64) string {
	s := strconv.FormatInt(n, 10)
	neg := strings.HasPrefix(s, "-")
	if neg {
		s = s[1:]
	}
	var b strings.Builder
	start := len(s) % 3
	if start == 0 {
		start = 3
	}
	b.WriteString(s[:start])
	for i := start; i < len(s); i += 3 {
		b.WriteByte(',')
		b.WriteString(s[i : i+3])
	}
	if neg {
		return "-" + b.String()
	}
	return b.String()
}

// RenderJSON encodes the receipt as indented JSON (deterministic: struct field
// order, no wall-clock).
func RenderJSON(r Receipt) ([]byte, error) {
	var buf bytes.Buffer
	enc := json.NewEncoder(&buf)
	enc.SetIndent("", "  ")
	if err := enc.Encode(r); err != nil {
		return nil, err
	}
	return buf.Bytes(), nil
}

// RenderMD renders the receipt verdict-first, with the full arithmetic block,
// warnings, provenance, and the witness checklist.
func RenderMD(r Receipt) string {
	var b strings.Builder
	fmt.Fprintf(&b, "# fak serve-receipt — %s\n\n", r.Model)

	verdict := fmt.Sprintf("**Verdict: %s**", r.Verdict)
	if r.Reason != "" {
		verdict += " — " + r.Reason
	}
	if r.Verdict == VerdictTight {
		verdict += "\n\nTIGHT: the layout fits but the requested max-model-len does not fit in the KV budget — lower max-model-len, add GPUs, or accept cache pressure (exit 0, marked)."
	}
	if r.Verdict == VerdictOffloadRequired {
		verdict += "\n\nOFFLOAD_REQUIRED: the device KV pool alone does not cover the request — the plan assumes the host KV pool is attached (vLLM KV offloading or equivalent). Decode reads miss to host at the miss-rate priced in the curve block; routed-expert weights must stay on device (exit 0, marked)."
	}
	b.WriteString(verdict + "\n\n")

	fmt.Fprintf(&b, "GPU: %s (vram %g GB, sm %s, fp4_native %t, nvlink %t%s) · config: tp %d × pp %d = %d GPUs, max_num_seqs %d, gpu_mem_util %g, activation_reserve_gb %g, max_model_len %s",
		r.GPU.Key, r.GPU.VRAMGB, r.GPU.SM, r.GPU.FP4Native, r.GPU.NVLink, pcieSuffix(r.GPU.PCIe),
		r.Config.TP, r.Config.PP, r.Arithmetic.GPUsTotal, r.Config.MaxNumSeqs,
		r.Config.GPUMemUtil, r.Config.ActivationReserveGB, modelLenLabel(r.Config.MaxModelLen))
	if dcp := r.Arithmetic.DCP; dcp > 1 {
		fmt.Fprintf(&b, ", dcp %d", dcp)
	}
	fmt.Fprintf(&b, ", kv_variant %s", r.Arithmetic.KVVariant)
	if host := r.Arithmetic.HostPoolBytes; host > 0 {
		fmt.Fprintf(&b, ", host_kv_pool %s GiB (usability %g, host↔device %g GB/s)",
			comma(host/GibBytes), r.Arithmetic.HostPoolUsability, r.Arithmetic.HostBandwidthGBs)
	}
	b.WriteString("\n\n")

	b.WriteString("## Arithmetic\n\n")
	fmt.Fprintf(&b, "units: %s\n\n", r.Arithmetic.Units)
	ar := r.Arithmetic
	fmt.Fprintf(&b, "- gpus = tp %d × pp %d = %d\n", r.Config.TP, r.Config.PP, ar.GPUsTotal)
	fmt.Fprintf(&b, "- weights_per_gpu = ceil(%s B / %d) = %s B\n", comma(r.Facts.WeightsBytes), ar.GPUsTotal, comma(ar.WeightsPerGPUBytes))
	fmt.Fprintf(&b, "- kv_bytes_per_token_per_gpu = %s = %s B (kv.tp_replicated=%t → %s)\n",
		ar.KVFormula, comma(ar.KVBytesPerTokenPerGPU), r.Facts.KV.TPReplicated, tpDivideLabel(r.Facts.KV.TPReplicated))
	fmt.Fprintf(&b, "- state_per_gpu_per_seq = %s B / tp %d = %s B (state_tp_sharded=%t)\n",
		comma(r.Facts.StatePerSeqBytes), r.Config.TP, comma(ar.StatePerGPUPerSeqBytes), r.Facts.StateTPSharded)
	fmt.Fprintf(&b, "- vram_per_gpu = %g GiB = %s B × util %g = %s B\n", r.GPU.VRAMGB, comma(ar.VRAMBytesPerGPU), r.Config.GPUMemUtil, comma(ar.UtilBudgetBytes))
	fmt.Fprintf(&b, "- usable_per_gpu = %s − %s = %s B\n", comma(ar.UtilBudgetBytes), comma(ar.WeightsPerGPUBytes), comma(ar.UsablePerGPUBytes))
	fmt.Fprintf(&b, "- reserved_state = %s B × max_num_seqs %d = %s B\n", comma(ar.StatePerGPUPerSeqBytes), r.Config.MaxNumSeqs, comma(ar.ReservedStateBytes))
	fmt.Fprintf(&b, "- activation_reserve = %g GB = %s B\n", r.Config.ActivationReserveGB, comma(ar.ActivationReserveBytes))
	fmt.Fprintf(&b, "- max_cached_tokens = max(0, floor((%s − %s − %s) / %s)) = %s\n",
		comma(ar.UsablePerGPUBytes), comma(ar.ReservedStateBytes), comma(ar.ActivationReserveBytes),
		comma(ar.KVBytesPerTokenPerGPU), comma(ar.MaxCachedTokens))
	if ar.PhysicalFormula != "" {
		if ar.KVVariantFormula != "" {
			fmt.Fprintf(&b, "- kv_variant %s: %s → %s B/token/GPU\n", ar.KVVariant, ar.KVVariantFormula, comma(ar.KVBytesPerTokenPerGPU))
		}
		if ar.DCP > 1 {
			fmt.Fprintf(&b, "- kv_effective_per_token_per_gpu = %s ÷ dcp %d = %s B\n",
				comma(ar.KVBytesPerTokenPerGPU), ar.DCP, comma(ar.KVEffectivePerTokenGPU))
		}
		fmt.Fprintf(&b, "- device_pool = %s − %s − %s = %s B (clamped ≥ 0)\n",
			comma(ar.UsablePerGPUBytes), comma(ar.ReservedStateBytes), comma(ar.ActivationReserveBytes), comma(ar.DevicePoolBytes))
		if ar.HostPoolBytes > 0 {
			fmt.Fprintf(&b, "- host_pool = %s GiB × usability %g ÷ dcp %d = %s B usable per rank\n",
				comma(ar.HostPoolBytes/GibBytes), ar.HostPoolUsability, ar.DCP, comma(ar.UsableHostBytes))
		}
		fmt.Fprintf(&b, "- physical_tokens = %s = %s\n", ar.PhysicalFormula, comma(ar.PhysicalTokens))
	}
	if ar.ExpertBytesPerExpert > 0 {
		estLabel := ""
		if ar.ExpertEstimated {
			estLabel = " (estimated from the weights share; no measured expert bytes in facts)"
		}
		fmt.Fprintf(&b, "- expert offload check: bytes/expert %s B%s; union min(routed %d, seqs %d × per_tok %d) = %d experts = %s B → ~%.1f s/forward at %g GB/s\n",
			comma(ar.ExpertBytesPerExpert), estLabel, r.Facts.RoutedExperts, r.Config.MaxNumSeqs, r.Facts.ExpertsPerToken,
			ar.ExpertUnionExperts, comma(ar.ExpertUnionBytes), ar.ExpertSecondsPerForward, ar.HostBandwidthGBs)
	}
	fmt.Fprintf(&b, "- requested max-model-len: %s → verdict %s", modelLenLabel(ar.RequestedMaxModelLen), r.Verdict)
	if ar.RequestedMaxModelLen > 0 {
		fmt.Fprintf(&b, " (headroom %s tokens)", comma(ar.HeadroomTokens))
	}
	b.WriteString("\n\n")

	b.WriteString("## Warnings\n\n")
	if len(r.Warnings) == 0 {
		b.WriteString("(none)\n\n")
	} else {
		for _, w := range r.Warnings {
			fmt.Fprintf(&b, "- %s\n", w)
		}
		b.WriteString("\n")
	}

	b.WriteString("## Provenance\n\n")
	fmt.Fprintf(&b, "- observed_at: %s\n", r.ObservedAt)
	if len(r.SourceURLs) == 0 {
		b.WriteString("- sources: (none cited)\n")
	} else {
		b.WriteString("- sources:\n")
		for _, u := range r.SourceURLs {
			fmt.Fprintf(&b, "  - %s\n", u)
		}
	}
	if len(r.Facts.Notes) > 0 {
		b.WriteString("- notes:\n")
		for _, n := range r.Facts.Notes {
			fmt.Fprintf(&b, "  - %s\n", n)
		}
	}
	b.WriteString("\n")

	if r.Curve != nil {
		b.WriteString("## Performance curve (ESTIMATE)\n\n")
		fmt.Fprintf(&b, "%s\n\n", r.Curve.Anchor)
		b.WriteString("capacity sweep (host KV pool → physical tokens; all other flags held at the receipt's values):\n\n")
		for _, row := range r.Curve.Capacity {
			fmt.Fprintf(&b, "- host %g GiB → %s physical tokens\n", row.HostGiB, comma(row.PhysicalTokens))
		}
		b.WriteString("\nmiss-rate estimate (per token, at the receipt's kv/dcp bytes and host bandwidth):\n\n")
		for _, row := range r.Curve.MissLatency {
			fmt.Fprintf(&b, "- miss %g → +%.3g µs/token\n", row.MissRate, row.ExtraLatencyUS)
		}
		fmt.Fprintf(&b, "\n%s\n\n", r.Curve.PenaltyNote)
	}

	b.WriteString("## Witness checklist\n\n")
	for i, w := range r.WitnessChecklist {
		fmt.Fprintf(&b, "%d. %s\n", i+1, w)
	}
	return b.String()
}

func pcieSuffix(pcie string) string {
	if pcie == "" {
		return ""
	}
	return ", pcie " + pcie
}

func tpDivideLabel(replicated bool) string {
	if replicated {
		return "no TP divide"
	}
	return "divided by TP"
}

func modelLenLabel(n int) string {
	if n <= 0 {
		return "(none)"
	}
	return comma(int64(n))
}
