package studyprio

import (
	"encoding/json"
	"fmt"
	"os"
	"reflect"
	"sort"
)

func ReadLedger(path string) (Ledger, error) { var v Ledger; err := readJSON(path, &v); return v, err }
func ReadSummary(path string) (Summary, error) {
	var v Summary
	err := readJSON(path, &v)
	return v, err
}
func readJSON(path string, target any) error {
	if path == "" {
		return invalidf("artifact path empty")
	}
	b, err := os.ReadFile(path)
	if err != nil {
		return fmt.Errorf("studyprio: read %s: %w", path, err)
	}
	if err := json.Unmarshal(b, target); err != nil {
		return invalidf("decode %s: %v", path, err)
	}
	return nil
}
func ValidateFiles(opts ValidateOptions) error {
	actualL, err := ReadLedger(opts.LedgerPath)
	if err != nil {
		return err
	}
	actualS, err := ReadSummary(opts.SummaryPath)
	if err != nil {
		return err
	}
	wantL, wantS, err := Build(BuildOptions{opts.SourceLedgerPath})
	if err != nil {
		return err
	}
	if !reflect.DeepEqual(actualL, wantL) {
		return invalidf("ledger differs from deterministic source rebuild")
	}
	if !reflect.DeepEqual(actualS, wantS) {
		return invalidf("summary differs from deterministic source rebuild")
	}
	return nil
}

func Validate(l Ledger) error {
	if l.Schema != Schema || !reflect.DeepEqual(l.Rubric, rubric) {
		return invalidf("schema or rubric version mismatch")
	}
	if l.Source.Path == "" || l.Source.SHA256 == "" || l.Source.Schema != sourceSchema || l.Source.SourceRevision == "" || l.Source.Cutoff == "" || l.Source.UncoveredCount != 5 {
		return invalidf("source receipt incomplete")
	}
	if len(l.Candidates) != 2 {
		return invalidf("bounded scope requires 2 candidates")
	}
	byID := map[string]Candidate{}
	coverage := map[string]string{}
	for _, c := range l.Candidates {
		if _, ok := byID[c.ID]; ok {
			return invalidf("duplicate candidate %s", c.ID)
		}
		byID[c.ID] = c
		if c.Title == "" || c.Category == "" || c.Horizon == "" || centralityRank(c.Centrality) == 0 {
			return invalidf("candidate %s classification incomplete", c.ID)
		}
		if err := validateDimensions(c.Dimensions); err != nil {
			return err
		}
		if c.Score != score(c.Dimensions) {
			return invalidf("candidate %s score mismatch", c.ID)
		}
		if err := validateRequiredGates(c); err != nil {
			return err
		}
		if err := validateContract(c); err != nil {
			return err
		}
		for _, m := range c.SourceMappings {
			if m.ClusterID == "" || m.Mechanism == "" || m.Signal == "" || m.Rule == "" || m.MembersSHA256 == "" || m.EvidenceSHA256 == "" {
				return invalidf("candidate %s mapping incomplete", c.ID)
			}
			if owner, ok := coverage[m.ClusterID]; ok {
				return invalidf("source %s mapped twice by %s and %s", m.ClusterID, owner, c.ID)
			}
			coverage[m.ClusterID] = c.ID
		}
	}
	if err := validateMerge(byID, coverage); err != nil {
		return err
	}
	q, err := buildQueue(l.Candidates)
	if err != nil {
		return err
	}
	if !reflect.DeepEqual(q, l.Queue) {
		return invalidf("queue is not deterministic dependency order")
	}
	s, err := sensitivity(l.Candidates, l.Queue)
	if err != nil {
		return err
	}
	if !reflect.DeepEqual(s, l.Sensitivity) {
		return invalidf("sensitivity mismatch")
	}
	return nil
}
func validateContract(c Candidate) error {
	if c.Frame.For == "" || c.Frame.Problem == "" || c.Frame.Today == "" || c.Frame.BetterBecause == "" || c.Frame.Witness == "" {
		return invalidf("candidate %s frame incomplete", c.ID)
	}
	checks := []P1P4Check{c.P1P4.P1Context, c.P1P4.P2NetValue, c.P1P4.P3Adaptation, c.P1P4.P4Operations}
	for i, x := range checks {
		if x.Reason == "" || (x.Status != "advanced" && x.Status != "preserved" && x.Status != "not-applicable") {
			return invalidf("candidate %s P%d invalid", c.ID, i+1)
		}
	}
	if c.Witness.Artifact == "" || c.Witness.Command == "" || c.Witness.PassCondition == "" || c.Witness.Engine != "fak-native" || c.Witness.Model != "Qwen3.8" {
		return invalidf("candidate %s witness is not fak-native Qwen3.8", c.ID)
	}
	if c.Execution.Engine != "fak-native" || c.Execution.DefaultModel != "Qwen3.8" || c.Execution.LlamaCPPUse != "reference-or-borrow-evidence-only" || c.Execution.FallbackAllowed {
		return invalidf("candidate %s native contract violated", c.ID)
	}
	return nil
}
func validateMerge(byID map[string]Candidate, coverage map[string]string) error {
	if len(coverage) != 5 {
		return invalidf("source coverage count %d", len(coverage))
	}
	for _, id := range requiredSourceClusters {
		if _, ok := coverage[id]; !ok {
			return invalidf("source %s unmapped", id)
		}
	}
	ir, ok := byID["native-vllm-ir"]
	if !ok || ir.MergeJustification == "" || len(ir.SourceMappings) != 4 {
		return invalidf("vllm-ir semantic merge invalid")
	}
	actual := []string{}
	for _, m := range ir.SourceMappings {
		actual = append(actual, m.ClusterID)
	}
	sort.Strings(actual)
	want := append([]string(nil), vllmIRClusters...)
	sort.Strings(want)
	if !reflect.DeepEqual(actual, want) {
		return invalidf("vllm-ir merge inputs changed")
	}
	a, ok := byID["allocator-fragmentation"]
	if !ok || len(a.SourceMappings) != 1 || a.SourceMappings[0].ClusterID != requiredSourceClusters[4] || a.MergeJustification != "" {
		return invalidf("allocator fragmentation must remain distinct")
	}
	return nil
}
func invalidf(format string, args ...any) error {
	return fmt.Errorf("%w: %s", ErrInvalid, fmt.Sprintf(format, args...))
}
