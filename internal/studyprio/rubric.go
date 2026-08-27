package studyprio

import (
	"sort"
)

const (
	gateBoundedSource = "bounded-source-coverage"
	gateSemanticMerge = "semantic-merge-integrity"
	gateNative        = "fak-native-execution"
	gateQwen38        = "qwen3.8-default"
	gateWitness       = "end-to-end-witness"
	gateNoFallback    = "no-llama.cpp-fallback"
)

var rubric = Rubric{
	Version: RubricVersion, Minimum: 0, Maximum: 5,
	Dimensions: []RubricDimension{
		{"product_centrality", 4, "directness of the user-visible P1-P4 outcome on the fak kernel path"},
		{"fak_native_qwen3_8_impact", 5, "expected improvement to fak-native Qwen3.8 execution; reference engines earn no impact"},
		{"end_to_end_value", 4, "net-true value on the real path"},
		{"evidence_strength", 3, "specificity and reproducibility of source and fak evidence"},
		{"recurrence", 2, "frequency of the tax on supported workloads"},
		{"dependency_unlock", 3, "central blocked outcomes enabled"},
		{"implementation_cost", -3, "bounded engineering and integration effort"},
		{"hardware_witness_cost", -2, "accelerator and operating cost for the witness"},
		{"compatibility_risk", -3, "format, kernel, platform, quality, and rollback risk"},
		{"duplication_conflict_penalty", -3, "overlap with shipped/open work or conflicting inputs"},
	},
	RequiredGates: []string{gateBoundedSource, gateSemanticMerge, gateNative, gateQwen38, gateWitness, gateNoFallback},
	TieBreaks:     []string{"dependencies before dependents", "weighted score descending among ready candidates", "centrality Core > Enabling > Stewardship > Peripheral", "candidate id ascending"},
}

func dimensionValues(d Dimensions) []int {
	return []int{d.ProductCentrality, d.FakNativeQwen38Impact, d.EndToEndValue, d.EvidenceStrength, d.Recurrence, d.DependencyUnlock, d.ImplementationCost, d.HardwareWitnessCost, d.CompatibilityRisk, d.DuplicationConflictPenalty}
}
func score(d Dimensions) int {
	total := 0
	for i, v := range dimensionValues(d) {
		total += v * rubric.Dimensions[i].Weight
	}
	return total
}
func validateDimensions(d Dimensions) error {
	for i, v := range dimensionValues(d) {
		if v < rubric.Minimum || v > rubric.Maximum {
			return invalidf("dimension %s=%d outside [%d,%d]", rubric.Dimensions[i].Name, v, rubric.Minimum, rubric.Maximum)
		}
	}
	return nil
}

func validateRequiredGates(c Candidate) error {
	seen := map[string]bool{}
	for _, g := range c.HardGates {
		if g.Name == "" || g.Evidence == "" || seen[g.Name] {
			return invalidf("candidate %s has incomplete or duplicate hard gate", c.ID)
		}
		seen[g.Name] = true
		if !g.Pass {
			return invalidf("candidate %s fails hard gate %s", c.ID, g.Name)
		}
	}
	for _, name := range rubric.RequiredGates {
		if !seen[name] {
			return invalidf("candidate %s is missing hard gate %s", c.ID, name)
		}
	}
	if len(seen) != len(rubric.RequiredGates) {
		return invalidf("candidate %s has unknown hard gate", c.ID)
	}
	return nil
}

func buildQueue(candidates []Candidate) ([]QueueEntry, error) {
	byID := map[string]Candidate{}
	indegree := map[string]int{}
	dependents := map[string][]string{}
	for _, c := range candidates {
		if c.ID == "" {
			return nil, invalidf("candidate id is empty")
		}
		if _, ok := byID[c.ID]; ok {
			return nil, invalidf("duplicate candidate %s", c.ID)
		}
		if err := validateRequiredGates(c); err != nil {
			return nil, err
		}
		byID[c.ID] = c
		indegree[c.ID] = len(c.Dependencies)
	}
	for _, c := range candidates {
		seen := map[string]bool{}
		for _, d := range c.Dependencies {
			if d == c.ID || seen[d] {
				return nil, invalidf("candidate %s has invalid dependency %s", c.ID, d)
			}
			seen[d] = true
			if _, ok := byID[d]; !ok {
				return nil, invalidf("candidate %s has missing dependency %s", c.ID, d)
			}
			dependents[d] = append(dependents[d], c.ID)
		}
	}
	ready := []Candidate{}
	for id, n := range indegree {
		if n == 0 {
			ready = append(ready, byID[id])
		}
	}
	out := []QueueEntry{}
	for len(ready) > 0 {
		sort.Slice(ready, func(i, j int) bool { return candidateLess(ready[i], ready[j]) })
		c := ready[0]
		ready = ready[1:]
		out = append(out, QueueEntry{len(out) + 1, c.ID, c.Score, c.Category, c.Horizon, append([]string(nil), c.Dependencies...)})
		for _, id := range dependents[c.ID] {
			indegree[id]--
			if indegree[id] == 0 {
				ready = append(ready, byID[id])
			}
		}
	}
	if len(out) != len(candidates) {
		return nil, invalidf("candidate dependency cycle")
	}
	return out, nil
}
func candidateLess(a, b Candidate) bool {
	if a.Score != b.Score {
		return a.Score > b.Score
	}
	if centralityRank(a.Centrality) != centralityRank(b.Centrality) {
		return centralityRank(a.Centrality) > centralityRank(b.Centrality)
	}
	return a.ID < b.ID
}
func centralityRank(v string) int {
	switch v {
	case "Core":
		return 4
	case "Enabling":
		return 3
	case "Stewardship":
		return 2
	case "Peripheral":
		return 1
	}
	return 0
}

func sensitivity(candidates []Candidate, queue []QueueEntry) (SensitivityExample, error) {
	baselineFirst := ""
	if len(queue) > 0 {
		baselineFirst = queue[0].CandidateID
	}
	var ir Candidate
	found := false
	for _, candidate := range candidates {
		if candidate.ID == "native-vllm-ir" {
			ir = candidate
			found = true
			break
		}
	}
	if !found {
		return SensitivityExample{}, invalidf("sensitivity candidate missing")
	}
	implementationAdjusted := ir.Score - 3
	hardwareAdjusted := implementationAdjusted - 2
	compatibilityAdjusted := hardwareAdjusted - 3
	steps := []SensitivityStep{
		{Dimension: "implementation_cost", From: ir.Dimensions.ImplementationCost, To: ir.Dimensions.ImplementationCost + 1, AdjustedScore: implementationAdjusted, QueueFirst: baselineFirst, Explanation: "raising implementation cost by one applies the -3 weight; vllm-ir remains first"},
		{Dimension: "hardware_witness_cost", From: ir.Dimensions.HardwareWitnessCost, To: ir.Dimensions.HardwareWitnessCost + 1, AdjustedScore: hardwareAdjusted, QueueFirst: baselineFirst, Explanation: "also raising hardware witness cost by one applies another -2; vllm-ir remains first by one point"},
		{Dimension: "compatibility_risk", From: ir.Dimensions.CompatibilityRisk, To: ir.Dimensions.CompatibilityRisk + 1, AdjustedScore: compatibilityAdjusted, QueueFirst: "allocator-fragmentation", Explanation: "also raising compatibility risk by one applies another -3 and flips the deterministic queue to allocator fragmentation"},
	}
	return SensitivityExample{CandidateID: ir.ID, BaselineScore: ir.Score, Steps: steps}, nil
}
