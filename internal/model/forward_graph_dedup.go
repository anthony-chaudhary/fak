package model

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"sort"
)

// GraphDedupOutcome records how one function was handled during bottom-up deduplication.
type GraphDedupOutcome struct {
	Function       string `json:"function"`
	Representative string `json:"representative"`
	Action         string `json:"action"` // "representative", "merged", "fenced", "retained"
	Reason         string `json:"reason,omitempty"`
}

// GraphDedupReceipt is the deterministic witness for bottom-up graph function deduplication.
type GraphDedupReceipt struct {
	Outcomes          []GraphDedupOutcome `json:"outcomes"`
	CascadedCalls     int                 `json:"cascaded_calls"`
	MergedFunctions   int                 `json:"merged_functions"`
	RetainedFunctions int                 `json:"retained_functions"`
	Digest            string              `json:"digest"`
}

// DeduplicateGraphFunctions merges structurally equivalent graph functions bottom-up.
// Leaves are deduplicated first, cascading representative rewrites into callers so that
// higher-level callers that differ only in callee names become equivalent and merge.
// The lexicographically smallest function name is selected as representative.
func DeduplicateGraphFunctions(program GraphInlineProgram) (GraphInlineProgram, GraphDedupReceipt, error) {
	functions, referenced, err := validateDedupProgram(program)
	if err != nil {
		return GraphInlineProgram{}, GraphDedupReceipt{}, err
	}

	// Compute dependency depth (level) for bottom-up processing.
	depths := make(map[string]int)
	visiting := make(map[string]bool)

	var computeDepth func(string) int
	computeDepth = func(name string) int {
		if d, ok := depths[name]; ok {
			return d
		}
		if visiting[name] {
			return 0 // break cycles defensively
		}
		visiting[name] = true
		defer func() { visiting[name] = false }()

		fn := functions[name]
		maxCalleeDepth := -1
		for _, inst := range fn.Instructions {
			if inst.Call != "" {
				cd := computeDepth(inst.Call)
				if cd > maxCalleeDepth {
					maxCalleeDepth = cd
				}
			}
		}
		depth := maxCalleeDepth + 1
		depths[name] = depth
		return depth
	}

	maxDepth := 0
	for name := range functions {
		d := computeDepth(name)
		if d > maxDepth {
			maxDepth = d
		}
	}

	// Group functions by level
	levelGroups := make(map[int][]string)
	for name, d := range depths {
		levelGroups[d] = append(levelGroups[d], name)
	}

	var receipt GraphDedupReceipt
	aliases := make(map[string]string)
	retained := make(map[string]bool)
	for name := range functions {
		retained[name] = true
	}

	isFenced := func(fn GraphInlineFunction) bool {
		return fn.Name == program.Entry || fn.Exported || fn.External || fn.IndirectCallable || referenced[fn.Name]
	}

	// Process levels bottom-up (0, 1, ..., maxDepth)
	for level := 0; level <= maxDepth; level++ {
		names := levelGroups[level]
		sort.Strings(names)

		// Find equivalence classes among retained functions at this level
		processed := make(map[string]bool)
		for i := 0; i < len(names); i++ {
			f1Name := names[i]
			if !retained[f1Name] || processed[f1Name] {
				continue
			}

			var cluster []string
			cluster = append(cluster, f1Name)
			processed[f1Name] = true

			for j := i + 1; j < len(names); j++ {
				f2Name := names[j]
				if !retained[f2Name] || processed[f2Name] {
					continue
				}
				if areFunctionsStructurallyEquivalent(functions[f1Name], functions[f2Name], aliases) {
					cluster = append(cluster, f2Name)
					processed[f2Name] = true
				}
			}

			if len(cluster) <= 1 {
				continue
			}

			sort.Strings(cluster)
			repIndex := 0
			for idx, cName := range cluster {
				if isFenced(functions[cName]) {
					repIndex = idx
					break
				}
			}
			rep := cluster[repIndex]
			receipt.Outcomes = append(receipt.Outcomes, GraphDedupOutcome{
				Function:       rep,
				Representative: rep,
				Action:         "representative",
			})

			for idx, cName := range cluster {
				if idx == repIndex {
					continue
				}
				if isFenced(functions[cName]) {
					receipt.Outcomes = append(receipt.Outcomes, GraphDedupOutcome{
						Function:       cName,
						Representative: rep,
						Action:         "fenced",
						Reason:         "abi boundary preserved",
					})
					continue
				}

				aliases[cName] = rep
				delete(retained, cName)
				receipt.MergedFunctions++
				receipt.Outcomes = append(receipt.Outcomes, GraphDedupOutcome{
					Function:       cName,
					Representative: rep,
					Action:         "merged",
				})
			}

			for callerName, fn := range functions {
				if !retained[callerName] {
					continue
				}
				for k := range fn.Instructions {
					inst := &fn.Instructions[k]
					if targetRep, exists := aliases[inst.Call]; exists {
						inst.Call = targetRep
						receipt.CascadedCalls++
					}
				}
				functions[callerName] = fn
			}
		}
	}

	var remainingFns []GraphInlineFunction
	for name, fn := range functions {
		if retained[name] {
			remainingFns = append(remainingFns, fn)
		}
	}
	sort.Slice(remainingFns, func(i, j int) bool {
		if remainingFns[i].Name == program.Entry {
			return true
		}
		if remainingFns[j].Name == program.Entry {
			return false
		}
		return remainingFns[i].Name < remainingFns[j].Name
	})

	outProgram := GraphInlineProgram{
		Entry:     program.Entry,
		Functions: remainingFns,
	}

	receipt.RetainedFunctions = len(remainingFns)
	sort.Slice(receipt.Outcomes, func(i, j int) bool {
		return receipt.Outcomes[i].Function < receipt.Outcomes[j].Function
	})

	encoded, err := json.Marshal(outProgram)
	if err != nil {
		return GraphInlineProgram{}, receipt, fmt.Errorf("serialize dedup program: %w", err)
	}
	hash := sha256.Sum256(encoded)
	receipt.Digest = hex.EncodeToString(hash[:])

	return outProgram, receipt, nil
}

func areFunctionsStructurallyEquivalent(f1, f2 GraphInlineFunction, aliases map[string]string) bool {
	if len(f1.Arguments) != len(f2.Arguments) {
		return false
	}
	if len(f1.Results) != len(f2.Results) {
		return false
	}
	if len(f1.ReturnValues) != len(f2.ReturnValues) {
		return false
	}
	if len(f1.Instructions) != len(f2.Instructions) {
		return false
	}

	resolveCall := func(call string) string {
		for {
			if target, ok := aliases[call]; ok {
				call = target
			} else {
				return call
			}
		}
	}

	val1 := make(map[string]string, len(f1.Arguments)+len(f1.Instructions))
	val2 := make(map[string]string, len(f2.Arguments)+len(f2.Instructions))

	for i, arg := range f1.Arguments {
		val1[arg] = fmt.Sprintf("$arg_%d", i)
	}
	for i, arg := range f2.Arguments {
		val2[arg] = fmt.Sprintf("$arg_%d", i)
	}

	for i := range f1.Instructions {
		inst1 := f1.Instructions[i]
		inst2 := f2.Instructions[i]

		if inst1.Operation != inst2.Operation {
			return false
		}
		if inst1.Value != inst2.Value {
			return false
		}
		if resolveCall(inst1.Call) != resolveCall(inst2.Call) {
			return false
		}
		if inst1.Reference != inst2.Reference {
			return false
		}
		if len(inst1.Arguments) != len(inst2.Arguments) {
			return false
		}
		for aIdx := range inst1.Arguments {
			norm1, ok1 := val1[inst1.Arguments[aIdx]]
			norm2, ok2 := val2[inst2.Arguments[aIdx]]
			if !ok1 || !ok2 || norm1 != norm2 {
				return false
			}
		}
		if len(inst1.Results) != len(inst2.Results) {
			return false
		}
		for rIdx := range inst1.Results {
			canonical := fmt.Sprintf("$res_%d_%d", i, rIdx)
			val1[inst1.Results[rIdx]] = canonical
			val2[inst2.Results[rIdx]] = canonical
		}
	}

	for i := range f1.ReturnValues {
		norm1, ok1 := val1[f1.ReturnValues[i]]
		norm2, ok2 := val2[f2.ReturnValues[i]]
		if !ok1 || !ok2 || norm1 != norm2 {
			return false
		}
	}

	return true
}

func validateDedupProgram(program GraphInlineProgram) (map[string]GraphInlineFunction, map[string]bool, error) {
	functions := make(map[string]GraphInlineFunction, len(program.Functions))
	referenced := make(map[string]bool)

	for idx := range program.Functions {
		target := program.Functions[idx]
		if target.Name == "" {
			return nil, nil, fmt.Errorf("function name is empty")
		}
		if _, seen := functions[target.Name]; seen {
			return nil, nil, fmt.Errorf("duplicate function %q", target.Name)
		}
		functions[target.Name] = cloneGraphInlineFunction(target)
		for _, op := range target.Instructions {
			if op.Reference != "" {
				referenced[op.Reference] = true
			}
		}
	}

	if _, ok := functions[program.Entry]; !ok {
		return nil, nil, fmt.Errorf("entry function %q is missing", program.Entry)
	}

	for scopeName, item := range functions {
		for _, op := range item.Instructions {
			if op.Call != "" && functions[op.Call].Name == "" {
				return nil, nil, fmt.Errorf("function %q calls missing function %q", scopeName, op.Call)
			}
			if op.Reference != "" && functions[op.Reference].Name == "" {
				return nil, nil, fmt.Errorf("function %q references missing function %q", scopeName, op.Reference)
			}
		}
	}
	return functions, referenced, nil
}
