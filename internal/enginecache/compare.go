package enginecache

import (
	"context"
	"net/http"
	"net/http/httptest"
	"time"

	"github.com/anthony-chaudhary/fak/internal/cachemeta"
)

// ComparisonArm is one independently reportable engine-cache invalidation arm.
// Available only means this checkout can execute the arm; it is never evidence
// that an unavailable external engine was measured.
type ComparisonArm struct {
	Name        string        `json:"name"`
	Kind        string        `json:"kind"`
	Integration bool          `json:"integration"`
	Available   bool          `json:"available"`
	Correct     bool          `json:"correct"`
	Latency     time.Duration `json:"latency"`
	Requests    int           `json:"requests"`
	Bytes       int64         `json:"bytes"`
	CostUSD     float64       `json:"cost_usd"`
	Note        string        `json:"note,omitempty"`
}

// ComparisonResult describes the same invalidation workload across native,
// no-feature, external, and first-class integration arms.
type ComparisonResult struct {
	Workload string          `json:"workload"`
	Arms     []ComparisonArm `json:"arms"`
}

// CompareLocal witnesses only the deterministic local arms. Real vLLM,
// SGLang, and LMCache measurements belong in the external benchmark runner.
func CompareLocal() ComparisonResult {
	dirs := comparisonDirectives()
	result := ComparisonResult{
		Workload: "invalidate one quarantined KV span and its dependent attention index before reuse",
		Arms: []ComparisonArm{
			{Name: "no invalidation", Kind: "baseline", Available: true, Correct: false, Note: "zero-work tuned baseline leaves both poisoned objects reusable"},
			{Name: "vLLM", Kind: "external", Note: "standalone serving-engine reset arm; requires a real vLLM endpoint"},
			{Name: "SGLang", Kind: "external", Note: "standalone serving-engine flush arm; requires a real SGLang endpoint"},
			{Name: "LMCache", Kind: "external", Note: "strongest practical distributed KV-cache alternative; requires real LMCache storage and serving"},
			{Name: "fak + vLLM", Kind: "integration", Integration: true, Note: "first-class --engine-cache-engine=vllm arm; requires a real vLLM endpoint"},
			{Name: "fak + SGLang", Kind: "integration", Integration: true, Note: "first-class --engine-cache-engine=sglang arm; requires a real SGLang endpoint"},
		},
	}

	var requests int
	var bytes int64
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		requests++
		bytes += r.ContentLength
		w.WriteHeader(http.StatusOK)
	}))
	defer server.Close()

	start := time.Now()
	witnessed, err := (Client{Engine: EngineVLLM, BaseURL: server.URL}).Invalidate(context.Background(), dirs)
	elapsed := time.Since(start)
	correct := err == nil && requests == 1 && witnessed.StatusCode == http.StatusOK && witnessed.Directives == len(dirs) && witnessed.Scope == ScopeWholePrefixCache
	result.Arms = append([]ComparisonArm{{
		Name: "fak native invalidation planner + adapter", Kind: "native", Available: true,
		Correct: correct, Latency: elapsed, Requests: requests, Bytes: bytes,
		Note: "in-process governance plus a loopback HTTP witness; not a real engine result",
	}}, result.Arms...)
	return result
}

func comparisonDirectives() []cachemeta.ExternalInvalidationDirective {
	kv := cachemeta.FromKVPrefix(
		cachemeta.KVPrefix{Tokens: []int{1, 2, 3}, ModelID: "bench-model", TokenizerID: "bench-tokenizer"},
		cachemeta.WithResidency(cachemeta.TierProvider, "vllm", "bench-lease"),
		cachemeta.WithAdmission(cachemeta.AdmissionQuarantine, "benchmark-referee"),
		cachemeta.WithDeletionCertificate(cachemeta.DeletionCertificate{Schema: "fak.deletioncert/v1", Subject: "bench-span", Digest: "bench-cert"}),
		cachemeta.WithLabel("provider", "vllm"),
		cachemeta.WithLabel("engine", "vllm"),
	)
	governance := cachemeta.GovernanceFromEntry(kv)
	return []cachemeta.ExternalInvalidationDirective{
		{Kind: cachemeta.ExternalInvalidateKVSpan, Entry: kv.ID, Plane: kv.Plane, Residency: kv.Residency, Provider: "vllm", Engine: "vllm", Reason: "quarantined", Governance: governance},
		{Kind: cachemeta.ExternalInvalidateAttentionIndex, Entry: kv.ID, Plane: cachemeta.PlaneAttentionIndex, Residency: kv.Residency, Provider: "vllm", Engine: "vllm", Reason: "dependent_on_quarantined_kv", Governance: governance},
	}
}
