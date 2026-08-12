package adjudicator

import (
	"context"
	"runtime"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/abi"
)

const (
	ComparisonSchema = "fak-policy-adjudication-comparison/1"
	// DefaultCompareIterations is the corpus-repeat count CompareLocal falls back to
	// when the caller passes a non-positive iterations value.
	DefaultCompareIterations = 10000
)

type ComparisonCase struct {
	Name     string          `json:"name"`
	Call     abi.ToolCall    `json:"-"`
	WantKind abi.VerdictKind `json:"want_kind"`
}

type ComparisonArm struct {
	Name              string  `json:"name"`
	Class             string  `json:"class"`
	Available         bool    `json:"available"`
	Calls             int     `json:"calls,omitempty"`
	Correct           int     `json:"correct,omitempty"`
	Correctness       float64 `json:"correctness,omitempty"`
	ElapsedNS         int64   `json:"elapsed_ns,omitempty"`
	ElapsedPerCallNS  float64 `json:"elapsed_per_call_ns,omitempty"`
	PeakRSSBytes      int64   `json:"peak_rss_bytes,omitempty"`
	TotalCostUSD      float64 `json:"total_cost_usd,omitempty"`
	UnavailableReason string  `json:"unavailable_reason,omitempty"`
}

type ComparisonReport struct {
	Schema   string          `json:"schema"`
	Workload string          `json:"workload"`
	GOOS     string          `json:"goos"`
	GOARCH   string          `json:"goarch"`
	Arms     []ComparisonArm `json:"arms"`
	Complete bool            `json:"complete"`
	Verdict  string          `json:"verdict"`
}

func comparisonInlineCall(tool, args string) abi.ToolCall {
	return abi.ToolCall{Tool: tool, Args: abi.Ref{Kind: abi.RefInline, Inline: []byte(args)}}
}

// ComparisonCorpus is a frozen structural-policy fixture shared by the native
// and tuned no-feature arms. Live external arms must consume an equivalent
// serialized input set rather than a hand-translated, easier policy.
func ComparisonCorpus() []ComparisonCase {
	return []ComparisonCase{
		{Name: "explicit-allow", Call: comparisonInlineCall("search_kb", `{}`), WantKind: abi.VerdictAllow},
		{Name: "prefix-allow", Call: comparisonInlineCall("read_ticket", `{"id":"42"}`), WantKind: abi.VerdictAllow},
		{Name: "explicit-deny", Call: comparisonInlineCall("refund_payment", `{"id":"42"}`), WantKind: abi.VerdictDeny},
		{Name: "default-deny", Call: comparisonInlineCall("unknown_tool", `{}`), WantKind: abi.VerdictDeny},
		{Name: "self-modify", Call: comparisonInlineCall("write_file", `{"path":"internal/adjudicator/decide.go"}`), WantKind: abi.VerdictDeny},
	}
}

func comparisonPolicy() Policy {
	return Policy{
		Allow:           map[string]bool{"search_kb": true},
		AllowPrefix:     []string{"read_"},
		Deny:            map[string]abi.ReasonCode{"refund_payment": abi.ReasonPolicyBlock},
		SelfModifyGlobs: []string{"internal/adjudicator"},
	}
}

// CompareLocal measures the real native seam and a tuned direct-lookup
// no-engine baseline. OPA and Cedar remain unavailable until their real
// processes evaluate the same policy/corpus with process metrics captured.
func CompareLocal(iterations int) ComparisonReport {
	if iterations <= 0 {
		iterations = DefaultCompareIterations
	}
	corpus := ComparisonCorpus()
	policy := comparisonPolicy()
	native := New(policy)
	ctx := context.Background()

	nativeCorrect := 0
	start := time.Now()
	for i := 0; i < iterations; i++ {
		for j := range corpus {
			if native.Adjudicate(ctx, &corpus[j].Call).Kind == corpus[j].WantKind {
				nativeCorrect++
			}
		}
	}
	nativeElapsed := time.Since(start)

	baseline := tunedBaselineLookup(policy, corpus)
	baselineCorrect := 0
	start = time.Now()
	for i := 0; i < iterations; i++ {
		for j := range corpus {
			if baseline[corpus[j].Call.Tool] == corpus[j].WantKind {
				baselineCorrect++
			}
		}
	}
	baselineElapsed := time.Since(start)
	total := iterations * len(corpus)

	return ComparisonReport{
		Schema:   ComparisonSchema,
		Workload: "identical structural policy and call corpus; live completion fixes policy semantics, process lifetime, warmup, concurrency, and correctness oracle",
		GOOS:     runtime.GOOS, GOARCH: runtime.GOARCH,
		Arms: []ComparisonArm{
			localComparisonArm("fak native adjudicator", "native", total, nativeCorrect, nativeElapsed),
			localComparisonArm("direct allow/deny lookup", "tuned_baseline", total, baselineCorrect, baselineElapsed),
			unavailableComparisonArm("OPA/Rego", "next_best", "requires real OPA evaluation of the equivalent policy with process latency, RSS, and correctness captured"),
			unavailableComparisonArm("Cedar", "next_best", "requires real Cedar evaluation of the equivalent policy with process latency, RSS, and correctness captured"),
		},
		Complete: false,
		Verdict:  "local structural correctness and call overhead only; no net-true policy-engine winner until real OPA and Cedar arms report equivalent semantics, latency, resources, and total cost",
	}
}

// tunedBaselineLookup precomputes the direct allow/deny answer for every tool in the
// corpus, derived from the SAME Policy the native arm evaluates. Deriving it means the
// baseline can never silently drift from comparisonPolicy() the way a hand-maintained
// switch does — an added Allow entry or AllowPrefix is picked up here for free, so a
// policy edit can no longer make the baseline look wrong instead of the arm.
//
// The derivation happens ONCE, OUTSIDE the timed loop, and that placement is the whole
// point: this arm is the no-engine floor the native seam is measured against, so its
// hot path has to stay a bare map hit. Resolving the policy per call would put map
// construction and prefix scanning inside the measurement and stop it being a floor at
// all.
func tunedBaselineLookup(p Policy, corpus []ComparisonCase) map[string]abi.VerdictKind {
	m := make(map[string]abi.VerdictKind, len(corpus))
	for i := range corpus {
		tool := corpus[i].Call.Tool
		if _, done := m[tool]; !done {
			m[tool] = directLookupKind(p, tool)
		}
	}
	return m
}

func directLookupKind(p Policy, tool string) abi.VerdictKind {
	if p.Allow[tool] {
		return abi.VerdictAllow
	}
	for _, prefix := range p.AllowPrefix {
		if strings.HasPrefix(tool, prefix) {
			return abi.VerdictAllow
		}
	}
	return abi.VerdictDeny
}

func localComparisonArm(name, class string, calls, correct int, elapsed time.Duration) ComparisonArm {
	return ComparisonArm{Name: name, Class: class, Available: true, Calls: calls, Correct: correct,
		Correctness: float64(correct) / float64(calls), ElapsedNS: elapsed.Nanoseconds(),
		ElapsedPerCallNS: float64(elapsed.Nanoseconds()) / float64(calls)}
}

func unavailableComparisonArm(name, class, reason string) ComparisonArm {
	return ComparisonArm{Name: name, Class: class, Available: false, UnavailableReason: reason}
}
