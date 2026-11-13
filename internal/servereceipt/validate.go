package servereceipt

import (
	"bytes"
	"encoding/json"
	"fmt"
	"os"
	"reflect"
	"strings"
)

// ValidateDoc is the validate output (fak.serve-receipt.validate/1): ok plus
// one row per field that fails to reconcile.
type ValidateDoc struct {
	Schema  string         `json:"schema"`
	OK      bool           `json:"ok"`
	Checked int            `json:"checked"`
	Diffs   []ValidateDiff `json:"diffs"`
}

// ValidateDiff is one field-by-field disagreement, values rendered as strings
// so numeric and textual fields share one shape.
type ValidateDiff struct {
	Field      string `json:"field"`
	Receipt    string `json:"receipt"`
	Recomputed string `json:"recomputed"`
}

// LoadReceipt reads a receipt document from disk (schema-checked).
func LoadReceipt(path string) (Receipt, error) {
	raw, err := os.ReadFile(path)
	if err != nil {
		return Receipt{}, fmt.Errorf("read receipt: %w", err)
	}
	var r Receipt
	if err := json.Unmarshal(raw, &r); err != nil {
		return Receipt{}, fmt.Errorf("parse receipt %s: %w", path, err)
	}
	if r.Schema != ReceiptSchema {
		return Receipt{}, fmt.Errorf("receipt %s: schema %q, want %q", path, r.Schema, ReceiptSchema)
	}
	if r.Facts.Schema != FactsSchema {
		return Receipt{}, fmt.Errorf("receipt %s: embedded facts schema %q, want %q", path, r.Facts.Schema, FactsSchema)
	}
	return r, nil
}

// Validate recomputes EVERY rendered value in the receipt from its embedded
// facts, config, and GPU profile, and returns one diff per disagreement.
//
// Recomputed surfaces: the top-level model/verdict/reason, the built-in
// profile cross-check (a receipt whose gpu.key names a built-in profile must
// carry that profile's exact fields), the whole arithmetic block (including
// the units note and KV formula strings), and the warnings list.
func Validate(r Receipt) ValidateDoc {
	doc := ValidateDoc{Schema: ValidateSchema, OK: true, Diffs: []ValidateDiff{}}
	add := func(field string, receipt, recomputed any) {
		rs, ns := renderVal(receipt), renderVal(recomputed)
		if rs != ns {
			doc.OK = false
			doc.Diffs = append(doc.Diffs, ValidateDiff{Field: field, Receipt: rs, Recomputed: ns})
		}
		doc.Checked++
	}

	add("schema", r.Schema, ReceiptSchema)
	add("model", r.Model, r.Facts.Model)

	// Built-in profile cross-check: the receipt must not silently re-price a
	// catalog card with edited hardware numbers. Custom (@file.json) profiles
	// have no catalog row and are taken as-is for the recomputation.
	if builtin, ok := LookupProfile(r.GPU.Key); ok {
		add("gpu.vram_gb", r.GPU.VRAMGB, builtin.VRAMGB)
		add("gpu.sm", r.GPU.SM, builtin.SM)
		add("gpu.fp4_native", r.GPU.FP4Native, builtin.FP4Native)
		add("gpu.nvlink", r.GPU.NVLink, builtin.NVLink)
		add("gpu.pcie", r.GPU.PCIe, builtin.PCIe)
	}

	recomputed, err := Plan(r.Facts, r.GPU, r.Config)
	if err != nil {
		doc.OK = false
		doc.Diffs = append(doc.Diffs, ValidateDiff{
			Field:      "plan",
			Receipt:    "(receipt as stored)",
			Recomputed: "recomputation refused: " + err.Error(),
		})
		doc.Checked++
		return doc
	}

	add("verdict", r.Verdict, recomputed.Verdict)
	add("reason", r.Reason, recomputed.Reason)
	add("gpu.key", r.GPU.Key, recomputed.GPU.Key)

	ra, na := r.Arithmetic, recomputed.Arithmetic
	add("arithmetic.units", ra.Units, na.Units)
	add("arithmetic.gpus_total", ra.GPUsTotal, na.GPUsTotal)
	add("arithmetic.weights_per_gpu_bytes", ra.WeightsPerGPUBytes, na.WeightsPerGPUBytes)
	add("arithmetic.kv_bytes_per_token_per_gpu", ra.KVBytesPerTokenPerGPU, na.KVBytesPerTokenPerGPU)
	add("arithmetic.kv_formula", ra.KVFormula, na.KVFormula)
	add("arithmetic.state_per_gpu_per_seq_bytes", ra.StatePerGPUPerSeqBytes, na.StatePerGPUPerSeqBytes)
	add("arithmetic.vram_bytes_per_gpu", ra.VRAMBytesPerGPU, na.VRAMBytesPerGPU)
	add("arithmetic.util_budget_bytes", ra.UtilBudgetBytes, na.UtilBudgetBytes)
	add("arithmetic.usable_per_gpu_bytes", ra.UsablePerGPUBytes, na.UsablePerGPUBytes)
	add("arithmetic.reserved_state_bytes", ra.ReservedStateBytes, na.ReservedStateBytes)
	add("arithmetic.activation_reserve_bytes", ra.ActivationReserveBytes, na.ActivationReserveBytes)
	add("arithmetic.max_cached_tokens", ra.MaxCachedTokens, na.MaxCachedTokens)
	add("arithmetic.requested_max_model_len", ra.RequestedMaxModelLen, na.RequestedMaxModelLen)
	add("arithmetic.headroom_tokens", ra.HeadroomTokens, na.HeadroomTokens)

	// Offload-aware fields (issue #14438) are checked only on receipts that
	// carry the new physical_formula marker: v1 receipts predate these fields,
	// and the recomputation of a v1 config fills the defaults, which would
	// otherwise read as spurious diffs against a v1-shaped receipt.
	if ra.PhysicalFormula != "" {
		add("arithmetic.kv_variant", ra.KVVariant, na.KVVariant)
		add("arithmetic.kv_variant_formula", ra.KVVariantFormula, na.KVVariantFormula)
		add("arithmetic.dcp", ra.DCP, na.DCP)
		add("arithmetic.kv_effective_per_token_per_gpu", ra.KVEffectivePerTokenGPU, na.KVEffectivePerTokenGPU)
		add("arithmetic.device_pool_bytes", ra.DevicePoolBytes, na.DevicePoolBytes)
		add("arithmetic.host_pool_bytes", ra.HostPoolBytes, na.HostPoolBytes)
		add("arithmetic.host_pool_usability", ra.HostPoolUsability, na.HostPoolUsability)
		add("arithmetic.usable_host_bytes", ra.UsableHostBytes, na.UsableHostBytes)
		add("arithmetic.host_bandwidth_gbs", ra.HostBandwidthGBs, na.HostBandwidthGBs)
		add("arithmetic.physical_tokens", ra.PhysicalTokens, na.PhysicalTokens)
		add("arithmetic.physical_formula", ra.PhysicalFormula, na.PhysicalFormula)
		if ra.ExpertBytesPerExpert > 0 || ra.ExpertUnionExperts > 0 || ra.ExpertUnionBytes > 0 || ra.ExpertSecondsPerForward > 0 || ra.ExpertEstimated {
			add("arithmetic.expert_bytes_per_expert", ra.ExpertBytesPerExpert, na.ExpertBytesPerExpert)
			add("arithmetic.expert_union_experts", ra.ExpertUnionExperts, na.ExpertUnionExperts)
			add("arithmetic.expert_union_bytes", ra.ExpertUnionBytes, na.ExpertUnionBytes)
			add("arithmetic.expert_seconds_per_forward", ra.ExpertSecondsPerForward, na.ExpertSecondsPerForward)
			add("arithmetic.expert_estimated", ra.ExpertEstimated, na.ExpertEstimated)
		}
	}

	// The curve block is recomputed only when the receipt carries one; a
	// stored block that the config no longer asks for (or vice versa) is
	// itself a diff.
	if r.Curve != nil || recomputed.Curve != nil {
		add("curve.present", r.Curve != nil, recomputed.Curve != nil)
		if r.Curve != nil && recomputed.Curve != nil {
			add("curve.anchor", r.Curve.Anchor, recomputed.Curve.Anchor)
			add("curve.penalty_note", r.Curve.PenaltyNote, recomputed.Curve.PenaltyNote)
			add("curve.capacity_rows", len(r.Curve.Capacity), len(recomputed.Curve.Capacity))
			for i := range r.Curve.Capacity {
				if i >= len(recomputed.Curve.Capacity) {
					break
				}
				add(fmt.Sprintf("curve.capacity[%d].host_gib", i), r.Curve.Capacity[i].HostGiB, recomputed.Curve.Capacity[i].HostGiB)
				add(fmt.Sprintf("curve.capacity[%d].physical_tokens", i), r.Curve.Capacity[i].PhysicalTokens, recomputed.Curve.Capacity[i].PhysicalTokens)
			}
			add("curve.miss_rows", len(r.Curve.MissLatency), len(recomputed.Curve.MissLatency))
			for i := range r.Curve.MissLatency {
				if i >= len(recomputed.Curve.MissLatency) {
					break
				}
				add(fmt.Sprintf("curve.miss_latency[%d].miss_rate", i), r.Curve.MissLatency[i].MissRate, recomputed.Curve.MissLatency[i].MissRate)
				add(fmt.Sprintf("curve.miss_latency[%d].extra_latency_us", i), r.Curve.MissLatency[i].ExtraLatencyUS, recomputed.Curve.MissLatency[i].ExtraLatencyUS)
			}
		}
	}

	add("warnings", r.Warnings, recomputed.Warnings)
	add("observed_at", r.ObservedAt, recomputed.ObservedAt)
	add("source_urls", r.SourceURLs, recomputed.SourceURLs)
	add("witness_checklist", r.WitnessChecklist, recomputed.WitnessChecklist)

	return doc
}

// renderVal renders a compared value the way the diff prints it.
func renderVal(v any) string {
	switch t := v.(type) {
	case string:
		return t
	case Verdict:
		return string(t)
	default:
		if rv := reflect.ValueOf(v); rv.Kind() == reflect.Slice {
			parts := make([]string, 0, rv.Len())
			for i := 0; i < rv.Len(); i++ {
				parts = append(parts, renderVal(rv.Index(i).Interface()))
			}
			return "[" + strings.Join(parts, ", ") + "]"
		}
		return fmt.Sprint(v)
	}
}

// RenderValidateMD renders the validate doc verdict-first with the
// field-by-field diff.
func RenderValidateMD(doc ValidateDoc) string {
	var b strings.Builder
	if doc.OK {
		fmt.Fprintf(&b, "# fak serve-receipt validate\n\n**OK** — receipt reconciles; %d fields recomputed from the embedded facts+config.\n", doc.Checked)
		return b.String()
	}
	fmt.Fprintf(&b, "# fak serve-receipt validate\n\n**MISMATCH** — %d of %d recomputed fields disagree with the receipt:\n\n", len(doc.Diffs), doc.Checked)
	for _, d := range doc.Diffs {
		fmt.Fprintf(&b, "- %s: receipt=%s recomputed=%s\n", d.Field, d.Receipt, d.Recomputed)
	}
	return b.String()
}

// RenderValidateJSON encodes the validate doc as indented JSON.
func RenderValidateJSON(doc ValidateDoc) ([]byte, error) {
	var buf bytes.Buffer
	enc := json.NewEncoder(&buf)
	enc.SetIndent("", "  ")
	if err := enc.Encode(doc); err != nil {
		return nil, err
	}
	return buf.Bytes(), nil
}
