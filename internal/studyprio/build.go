package studyprio

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"sort"

	"github.com/anthony-chaudhary/fak/internal/stablejson"
)

var requiredSourceClusters = []string{"architecture_runtime:body:vllm-ir", "architecture_runtime:label:vllm-ir", "architecture_runtime:title:vllm-ir", "kernels_compilation:label:vllm-ir", "memory_residency:body:allocator-fragmentation"}
var vllmIRClusters = requiredSourceClusters[:4]

func Build(opts BuildOptions) (Ledger, Summary, error) {
	source, b, err := readSourceLedger(opts.SourceLedgerPath)
	if err != nil {
		return Ledger{}, Summary{}, err
	}
	inputs, err := uncoveredInputs(source)
	if err != nil {
		return Ledger{}, Summary{}, err
	}
	candidates, err := buildCandidates(inputs)
	if err != nil {
		return Ledger{}, Summary{}, err
	}
	queue, err := buildQueue(candidates)
	if err != nil {
		return Ledger{}, Summary{}, err
	}
	example, err := sensitivity(candidates, queue)
	if err != nil {
		return Ledger{}, Summary{}, err
	}
	ledger := Ledger{Schema: Schema, Rubric: rubric, Source: SourceReceipt{filepath.ToSlash(opts.SourceLedgerPath), digest(b), source.Schema, source.SourceRevision, source.Cutoff, len(inputs)}, Candidates: candidates, Queue: queue, Sensitivity: example}
	if err := Validate(ledger); err != nil {
		return Ledger{}, Summary{}, err
	}
	lb, err := MarshalLedger(ledger)
	if err != nil {
		return Ledger{}, Summary{}, err
	}
	summary := Summary{Schema: SummarySchema, RubricVersion: RubricVersion, SourceLedgerSHA256: ledger.Source.SHA256, PriorityLedgerSHA256: digest(lb), SourceClusterCount: len(inputs), CandidateCount: len(candidates), QueueCount: len(queue)}
	for _, q := range queue {
		summary.QueueCandidateIDs = append(summary.QueueCandidateIDs, q.CandidateID)
	}
	return ledger, summary, nil
}

func buildCandidates(inputs map[string]sourceJoin) ([]Candidate, error) {
	ir, err := mappingsFor(inputs, vllmIRClusters)
	if err != nil {
		return nil, err
	}
	alloc, err := mappingsFor(inputs, requiredSourceClusters[4:])
	if err != nil {
		return nil, err
	}
	candidates := []Candidate{
		{ID: "native-vllm-ir", Title: "Lower fak-native Qwen3.8 execution through a stable vLLM-IR-shaped optimization seam", Category: "architecture-runtime", Horizon: "now", Centrality: "Core", SourceMappings: ir, MergeJustification: "the three architecture-runtime rules and the kernels-compilation label name one vLLM-IR boundary between model execution and compiled kernel lowering; one candidate retains all four rule-specific mappings while avoiding four incompatible IR contracts", HardGates: passingGates("four retained vLLM-IR inputs; allocator fragmentation remains distinct", "all four inputs normalize to one execution-to-kernel IR boundary with distinct cluster/rule/checksum mappings", "execution, lowering, scheduling, cache, memory, and kernels remain fak-owned", "default implementation and quality envelope use Qwen3.8", "fak-native serve receipt covers quality, latency, memory, counters, and rollback", "llama.cpp is reference/borrow evidence only, never fallback"), Dimensions: Dimensions{5, 5, 5, 4, 4, 5, 4, 4, 3, 1}, Dependencies: []string{}, Frame: ValueFrame{"operators running Qwen3.8 through fak-native inference", "runtime and kernel optimizations lack one stable inspectable IR seam", "add one-off execution or kernel paths and use external engines only as references", "one owned IR contract unlocks repeated fak-native graph and kernel optimizations while preserving product ownership", "a Qwen3.8 fak-native serve receipt names the IR lowering, matches accepted-token quality, reports net-true latency/memory, and proves rollback"}, P1P4: P1P4{P1P4Check{"preserved", "the IR seam must not duplicate prompt, KV, or scheduler state"}, P1P4Check{"advanced", "net-true evidence includes lowering, verification, and recovery cost"}, P1P4Check{"advanced", "a versioned bounded IR supports replaceable transforms and rollback"}, P1P4Check{"advanced", "the real serve receipt exposes selection, counters, refusal, and rollback"}}, Witness: Witness{"fak-native Qwen3.8 serve receipt", "fak bench native --model Qwen3.8 --engine fak-native --receipt <path>", "equal accepted-token quality with net-true improvement, named engine, IR counters, and rollback", "fak-native", "Qwen3.8"}, Execution: nativeExecutionContract()},
		{ID: "allocator-fragmentation", Title: "Measure and bound allocator fragmentation on the fak-native Qwen3.8 serve path", Category: "memory-residency", Horizon: "next", Centrality: "Core", SourceMappings: alloc, HardGates: passingGates("allocator input is retained once outside vLLM-IR", "fragmentation is a residency/lifetime mechanism, not an IR or lowering spelling", "allocation, residency, scheduling, cache, and recovery remain fak-owned", "default fragmentation envelope uses Qwen3.8", "long-running receipt reports reserved/live bytes, quality, throughput, and recovery", "llama.cpp is reference/borrow evidence only, never fallback"), Dimensions: Dimensions{5, 5, 4, 4, 4, 3, 3, 5, 3, 0}, Dependencies: []string{}, Frame: ValueFrame{"operators sustaining variable-shape Qwen3.8 workloads in fak", "fragmentation can strand accelerator memory and cause late admission failures while live tensors fit", "treat OOM or shrinking capacity as aggregate pressure without a fragmentation ratio", "separate reserved, live, reusable, and stranded bytes can drive an owned mitigation without conflating IR with lifetime", "a fak-native Qwen3.8 churn receipt compares reserved/live bytes, fragmentation, throughput, quality, admission, and recovery"}, P1P4: P1P4{P1P4Check{"preserved", "accounting must not evict useful KV state or duplicate residency"}, P1P4Check{"advanced", "witness counts overhead, reclaimed capacity, throughput, quality, and retries"}, P1P4Check{"advanced", "allocator policy stays behind a bounded reversible seam"}, P1P4Check{"advanced", "operators receive ratios, thresholds, refusal, and recovery"}}, Witness: Witness{"fak-native Qwen3.8 allocator-churn receipt", "fak bench native --model Qwen3.8 --engine fak-native --allocator-churn --receipt <path>", "bounded fragmentation with equal quality, net-true capacity/throughput, and recovery", "fak-native", "Qwen3.8"}, Execution: nativeExecutionContract()},
	}
	for i := range candidates {
		candidates[i].Score = score(candidates[i].Dimensions)
	}
	sort.Slice(candidates, func(i, j int) bool { return candidates[i].ID < candidates[j].ID })
	return candidates, nil
}

func mappingsFor(inputs map[string]sourceJoin, ids []string) ([]SourceMapping, error) {
	out := make([]SourceMapping, 0, len(ids))
	for _, id := range ids {
		j, ok := inputs[id]
		if !ok {
			return nil, invalidf("required uncovered source cluster %s missing", id)
		}
		out = append(out, SourceMapping{j.ClusterID, j.Mechanism, j.Signal, j.Rule, j.MembersSHA256, j.Evidence.Digest})
	}
	return out, nil
}
func passingGates(evidence ...string) []HardGate {
	out := make([]HardGate, len(rubric.RequiredGates))
	for i, n := range rubric.RequiredGates {
		out[i] = HardGate{n, true, evidence[i]}
	}
	return out
}
func nativeExecutionContract() ExecutionContract {
	return ExecutionContract{"fak-native", "Qwen3.8", "reference-or-borrow-evidence-only", false}
}
func readSourceLedger(path string) (sourceLedger, []byte, error) {
	if path == "" {
		return sourceLedger{}, nil, invalidf("source path empty")
	}
	b, err := os.ReadFile(path)
	if err != nil {
		return sourceLedger{}, nil, fmt.Errorf("studyprio: read source: %w", err)
	}
	var s sourceLedger
	if err := json.Unmarshal(b, &s); err != nil {
		return s, nil, invalidf("decode source: %v", err)
	}
	if s.Schema != sourceSchema {
		return s, nil, invalidf("source schema %q", s.Schema)
	}
	if s.SourceRevision == "" || s.Cutoff == "" {
		return s, nil, invalidf("source receipt incomplete")
	}
	return s, b, nil
}
func uncoveredInputs(s sourceLedger) (map[string]sourceJoin, error) {
	out := map[string]sourceJoin{}
	for _, j := range s.Joins {
		if !j.Actionable || j.Disposition != "uncovered" {
			continue
		}
		if j.ClusterID == "" || j.Mechanism == "" || j.Signal == "" || j.Rule == "" || j.MembersSHA256 == "" || j.Evidence.Digest == "" {
			return nil, invalidf("uncovered source %q incomplete", j.ClusterID)
		}
		if _, ok := out[j.ClusterID]; ok {
			return nil, invalidf("duplicate uncovered source %s", j.ClusterID)
		}
		out[j.ClusterID] = j
	}
	if len(out) != len(requiredSourceClusters) {
		return nil, invalidf("bounded scope requires 5 uncovered sources, found %d", len(out))
	}
	for _, id := range requiredSourceClusters {
		if _, ok := out[id]; !ok {
			return nil, invalidf("required uncovered source %s missing", id)
		}
	}
	return out, nil
}
func MarshalLedger(v Ledger) ([]byte, error)   { return stablejson.Marshal(v) }
func MarshalSummary(v Summary) ([]byte, error) { return stablejson.Marshal(v) }
func digest(b []byte) string                   { s := sha256.Sum256(b); return "sha256:" + hex.EncodeToString(s[:]) }
