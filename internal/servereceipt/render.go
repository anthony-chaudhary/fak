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
	b.WriteString(verdict + "\n\n")

	fmt.Fprintf(&b, "GPU: %s (vram %g GB, sm %s, fp4_native %t, nvlink %t%s) · config: tp %d × pp %d = %d GPUs, max_num_seqs %d, gpu_mem_util %g, activation_reserve_gb %g, max_model_len %s\n\n",
		r.GPU.Key, r.GPU.VRAMGB, r.GPU.SM, r.GPU.FP4Native, r.GPU.NVLink, pcieSuffix(r.GPU.PCIe),
		r.Config.TP, r.Config.PP, r.Arithmetic.GPUsTotal, r.Config.MaxNumSeqs,
		r.Config.GPUMemUtil, r.Config.ActivationReserveGB, modelLenLabel(r.Config.MaxModelLen))

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
