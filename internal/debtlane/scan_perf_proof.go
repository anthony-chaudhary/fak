package debtlane

import (
	"fmt"
	"strings"
)

// AuditPerformanceProof evaluates whether a performance lane has fresh, fak-native, workload-bound,
// and quality-constrained performance proof evidence. Returns typed findings.
func AuditPerformanceProof(lane *DebtLane) []FindingProvenance {
	if lane == nil {
		return nil
	}
	// Non-performance lanes stay clean regardless of proof presence.
	if !lane.IsPerformanceLane() {
		return nil
	}

	surface := string(classifySurface(lane.UnitOfWork))
	proof := lane.PerformanceProof
	if proof == nil && lane.Evidence.PerformanceProof != nil {
		proof = lane.Evidence.PerformanceProof
	}

	makeFinding := func(reason, msg string) FindingProvenance {
		lane.PerfProofReason = reason
		lane.Evidence.PerfProofReason = reason
		return FindingProvenance{
			Dimension: string(DimStalePerfProof),
			Surface:   surface,
			Lane:      lane.Lane,
			Path:      lane.UnitOfWork,
			Severity:  "critical",
			Message:   fmt.Sprintf("stale performance proof [%s]: %s", reason, msg),
		}
	}

	// 1. Missing performance proof
	if proof == nil {
		hist := lane.HistoricalProofs
		if len(hist) == 0 && len(lane.Evidence.HistoricalProofs) > 0 {
			hist = lane.Evidence.HistoricalProofs
		}
		if len(hist) > 0 {
			for _, h := range hist {
				scopeDesc := h.IncompatibleScope
				if scopeDesc == "" {
					scopeDesc = "historical"
				}
				return []FindingProvenance{makeFinding(ReasonIncompatibleScope,
					fmt.Sprintf("historical evidence preserved under explicit incompatible scope %q; current native performance proof required", scopeDesc))}
			}
		}

		if lane.CurrentRevision != "" || lane.RequiredWorkload != "" || lane.RequiredQuality != "" ||
			(lane.Evidence.Integrated && !lane.Evidence.Benchmarked && !lane.Evidence.Dogfooded) {
			return []FindingProvenance{makeFinding(ReasonMissingPerfProof,
				"performance-critical unit lacks a workload- and quality-bound performance authority record")}
		}
		return nil
	}

	// 2. Historical evidence with explicit incompatible scope: preserve, but flag as incompatible
	if proof.IncompatibleScope != "" || proof.IsHistorical {
		alreadyPreserved := false
		for _, h := range lane.HistoricalProofs {
			if h.Revision == proof.Revision && h.Artifact == proof.Artifact {
				alreadyPreserved = true
				break
			}
		}
		if !alreadyPreserved {
			lane.HistoricalProofs = append(lane.HistoricalProofs, *proof)
			lane.Evidence.HistoricalProofs = append(lane.Evidence.HistoricalProofs, *proof)
		}
		scopeDesc := proof.IncompatibleScope
		if scopeDesc == "" {
			scopeDesc = "historical"
		}
		return []FindingProvenance{makeFinding(ReasonIncompatibleScope,
			fmt.Sprintf("historical evidence preserved under explicit incompatible scope %q; current native performance proof required", scopeDesc))}
	}

	// 3. Engine check: must be fak-native, NOT reference engine (e.g. llama.cpp)
	eng := strings.ToLower(strings.TrimSpace(proof.Engine))
	if eng == "" || isReferenceEngine(eng) {
		engName := proof.Engine
		if engName == "" {
			engName = "unspecified"
		}
		return []FindingProvenance{makeFinding(ReasonReferenceEngine,
			fmt.Sprintf("reference engine %q cannot satisfy native performance proof; fak-native execution required", engName))}
	}

	// 4. Revision check: must match current revision
	currentRev := strings.TrimSpace(lane.CurrentRevision)
	proofRev := strings.TrimSpace(proof.Revision)
	if currentRev != "" {
		if proofRev == "" {
			return []FindingProvenance{makeFinding(ReasonMismatchedRevision,
				fmt.Sprintf("authority record missing revision; current code revision is %s", currentRev))}
		}
		if !revisionsMatch(proofRev, currentRev) {
			return []FindingProvenance{makeFinding(ReasonMismatchedRevision,
				fmt.Sprintf("authority revision %s does not match current code revision %s", proofRev, currentRev))}
		}
	} else if proofRev == "" {
		return []FindingProvenance{makeFinding(ReasonMismatchedRevision,
			"authority record missing source revision binding")}
	}

	// 5. Workload check: must be bound and non-empty
	workload := strings.TrimSpace(proof.Workload)
	if workload == "" {
		return []FindingProvenance{makeFinding(ReasonMismatchedWorkload,
			"authority record lacks bound workload specification")}
	}
	reqWorkload := strings.TrimSpace(lane.RequiredWorkload)
	if reqWorkload != "" && !workloadsMatch(workload, reqWorkload) {
		return []FindingProvenance{makeFinding(ReasonMismatchedWorkload,
			fmt.Sprintf("measured workload %q does not match required workload %q", workload, reqWorkload))}
	}

	// 6. Quality envelope check: must be bound and non-empty
	quality := strings.TrimSpace(proof.QualityEnvelope)
	if quality == "" {
		return []FindingProvenance{makeFinding(ReasonMismatchedQualityEnvelope,
			"authority record lacks bound quality envelope specification")}
	}
	reqQuality := strings.TrimSpace(lane.RequiredQuality)
	if reqQuality != "" && !qualitiesMatch(quality, reqQuality) {
		return []FindingProvenance{makeFinding(ReasonMismatchedQualityEnvelope,
			fmt.Sprintf("measured quality envelope %q does not match required quality %q", quality, reqQuality))}
	}

	// Clean!
	lane.PerfProofReason = ""
	lane.Evidence.PerfProofReason = ""
	return nil
}

func isReferenceEngine(eng string) bool {
	lower := strings.ToLower(strings.TrimSpace(eng))
	if lower == "llama.cpp" || lower == "llamacpp" || lower == "reference" ||
		lower == "vllm" || lower == "mlx" || lower == "external" || lower == "ollama" {
		return true
	}
	if strings.Contains(lower, "fak") || lower == "native" || lower == "in-kernel" || lower == "inkernel" {
		return false
	}
	return true
}

func revisionsMatch(proofRev, currentRev string) bool {
	p := strings.ToLower(strings.TrimSpace(proofRev))
	c := strings.ToLower(strings.TrimSpace(currentRev))
	if p == c {
		return true
	}
	if len(p) >= 7 && strings.HasPrefix(c, p) {
		return true
	}
	if len(c) >= 7 && strings.HasPrefix(p, c) {
		return true
	}
	if strings.Contains(p, "+g") && strings.Contains(c, "+g") {
		pParts := strings.Split(p, "+g")
		cParts := strings.Split(c, "+g")
		if len(pParts) == 2 && len(cParts) == 2 {
			pSha := pParts[1]
			cSha := cParts[1]
			if (len(pSha) >= 7 && strings.HasPrefix(cSha, pSha)) || (len(cSha) >= 7 && strings.HasPrefix(pSha, cSha)) {
				return pParts[0] == cParts[0]
			}
		}
	}
	return false
}

func workloadsMatch(measured, required string) bool {
	m := strings.ToLower(strings.TrimSpace(measured))
	r := strings.ToLower(strings.TrimSpace(required))
	if m == r {
		return true
	}
	return strings.Contains(m, r) || strings.Contains(r, m)
}

func qualitiesMatch(measured, required string) bool {
	m := strings.ToLower(strings.TrimSpace(measured))
	r := strings.ToLower(strings.TrimSpace(required))
	if m == r {
		return true
	}
	return strings.Contains(m, r) || strings.Contains(r, m)
}
