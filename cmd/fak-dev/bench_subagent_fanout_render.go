package main

import (
	"fmt"
	"io"
	"strings"
)

func renderPrettyReceipt(w io.Writer, r *SubagentFanoutReceipt) {
	fmt.Fprintln(w, strings.Repeat("=", 120))
	fmt.Fprintln(w, "APPLES-TO-APPLES SUBAGENT FANOUT BENCHMARK HARNESS (Issue #6036 & Issue #12325)")
	fmt.Fprintf(w, "Model: %s | Quantization: %s | Memory Fraction: %.2f\n", r.Model, r.Quantization, r.MemoryFraction)
	fmt.Fprintf(w, "Timestamp: %s | Host: %s\n", r.Timestamp, r.HostOS)
	fmt.Fprintln(w, strings.Repeat("-", 120))
	fmt.Fprintf(w, "%-6s | %-26s | %8s | %10s | %10s | %10s | %10s | %10s | %14s\n",
		"Fanout", "Arm", "HitRate", "TTFT p50", "TTFT p95", "TTFT p99", "ITL Mean", "ITL Jitter", "Throughput")
	fmt.Fprintln(w, strings.Repeat("-", 120))

	currentN := -1
	for _, res := range r.Results {
		if currentN != -1 && currentN != res.FanoutN {
			fmt.Fprintln(w, strings.Repeat("-", 120))
		}
		currentN = res.FanoutN

		armLabel := res.Arm
		switch res.Arm {
		case ArmNoReuse:
			armLabel = "No-reuse baseline"
		case ArmSGLang:
			armLabel = "SGLang RadixAttention"
		case ArmVLLM:
			armLabel = "vLLM Prefix Caching"
		case ArmFAK:
			armLabel = "FAK native engine"
		case ArmLLamaCPP:
			armLabel = "llama.cpp llama-server"
		}

		if res.Error != "" {
			fmt.Fprintf(w, "N=%-4d | %-26s | %8s | %s\n", res.FanoutN, armLabel, "ERR", res.Error)
			// A serve-completeness refusal is the difference between "the arm
			// erred" and "the control never ran". Spell out the structured
			// reason and the observed capacity so a reader cannot mistake a
			// broken reference for a measured null result (issue #13134).
			if sc := res.ServeCompleteness; sc != nil && !sc.ControlArmServed {
				fmt.Fprintf(w, "       %-26s | serve-completeness: %s (per-slot=%d, slots=%d, requested=%d, evidence_gap=%v)\n",
					"", sc.Refusal, sc.ObservedPerSlotTokens, sc.ObservedTotalSlots, sc.RequestedTotalTokens, sc.EvidenceGap)
			}
			continue
		}

		fmt.Fprintf(w, "N=%-4d | %-26s | %7.1f%% | %8.1f ms | %8.1f ms | %8.1f ms | %8.1f ms | %8.1f ms | %10.1f tok/s\n",
			res.FanoutN,
			armLabel,
			res.PrefixHitRate*100.0,
			res.TTFT.P50,
			res.TTFT.P95,
			res.TTFT.P99,
			res.ITL.MeanMs,
			res.ITL.JitterMs,
			res.DecodeThroughputTokPerSec,
		)
		if res.ReuseObservationSource != "" {
			fmt.Fprintf(w, "       %-26s | %s/%s: %s\n", "", res.RequestShape, res.Schedule, renderCellReuse(res))
		}
		if sc := res.ServeCompleteness; sc != nil && sc.CapacityDeclared {
			fmt.Fprintf(w, "       %-26s | capacity DECLARED %d tokens/slot (unobserved; primary: %s)\n", "", sc.DeclaredPerSlotTokens, sc.PrimaryProbe)
		}
	}

	fmt.Fprintln(w, strings.Repeat("=", 120))
	if r.UnifiedThroughput != nil {
		ut := r.UnifiedThroughput
		verdict := "no SLO floor set"
		if ut.Met != nil {
			if *ut.Met {
				verdict = fmt.Sprintf("SLO MET (floor %.1f tok/s)", ut.MinTokPerSec)
			} else {
				verdict = fmt.Sprintf("SLO MISSED (floor %.1f tok/s)", ut.MinTokPerSec)
			}
		}
		fmt.Fprintf(w, "Unified throughput (measured, issue #13076): %.1f tok/s | %d tokens / %.2fs | %s\n",
			ut.TokPerSec, ut.GeneratedTokens, ut.WallSeconds, verdict)
	}
	if len(r.ReuseDivergenceArms) > 0 {
		fmt.Fprintf(w, "REUSE DIVERGENCE (analytic reuse unobserved by server): %s\n", strings.Join(r.ReuseDivergenceArms, ", "))
	}
	fmt.Fprintf(w, "Issue #6036 Contract Audit: %s\n", contractStatusLabel(r.Contract))
	fmt.Fprintf(w, "  - All 4 Arms Present: %v\n", r.Contract.AllFourArmsPresent)
	fmt.Fprintf(w, "  - Identical Weights & Quantization: %v (%s / %s)\n", r.Contract.IdenticalWeightsEnforced && r.Contract.IdenticalQuantization, r.Model, r.Quantization)
	fmt.Fprintf(w, "  - Fixed Memory Fraction 0.85: %v (observed %.2f)\n", r.Contract.FixedMemoryFractionVerified, r.Contract.ObservedMemoryFraction)
	fmt.Fprintf(w, "  - Fanout Sweep Verified [1,4,8,16,32]: %v (scope %s)\n", r.Contract.FanoutSweepVerified, r.Contract.SweepScope)
	if r.Contract.Partial {
		fmt.Fprintln(w, "  - PARTIAL: smoke subset of the canonical sweep; not a full contract run")
	}
	fmt.Fprintf(w, "  - Captured TTFT (p50/p95/p99) & ITL Jitter: %v\n", r.Contract.CapturedTTFTDistributions && r.Contract.CapturedITLJitter)

	if len(r.Contract.Violations) > 0 {
		fmt.Fprintln(w, "Contract Violations:")
		for _, v := range r.Contract.Violations {
			fmt.Fprintf(w, "  [!] %s\n", v)
		}
	}
	fmt.Fprintln(w, strings.Repeat("=", 120))
}

func renderCellReuse(res FanoutArmResult) string {
	sc := res.ReuseScrape
	if sc == nil {
		return "observed reuse UNOBSERVED"
	}
	mult := "multiplier unobserved"
	if res.ObservedReuseMultiplier > 0 {
		mult = fmt.Sprintf("reuse multiplier %.2fx", res.ObservedReuseMultiplier)
	}
	return fmt.Sprintf("observed reuse %d tokens (%s delta %d->%d, offered %d, hit %.1f%%), %s",
		sc.Delta, sc.Counter, sc.Before, sc.After, sc.OfferedTokens, res.ObservedHitRate*100.0, mult)
}

func contractStatusLabel(c ContractValidation) string {
	if c.Partial && len(c.Violations) == 0 {
		return "PARTIAL (smoke subset; not a full contract run)"
	}
	if c.Compliant {
		return "PASS (Compliant)"
	}
	return "FAIL (Non-Compliant)"
}
