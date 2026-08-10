package adjudicator

import (
	"context"
	"runtime"
	"time"

	"github.com/anthony-chaudhary/fak/internal/abi"
)

const ComparisonSchema = "fak-policy-adjudication-comparison/1"

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
		iterations = 10000
	}
	corpus := ComparisonCorpus()
	native := New(comparisonPolicy())
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

	baselineCorrect := 0
	start = time.Now()
	for i := 0; i < iterations; i++ {
		for j := range corpus {
			if directLookupKind(corpus[j].Call.Tool) == corpus[j].WantKind {
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

func directLookupKind(tool string) abi.VerdictKind {
	switch tool {
	case "search_kb", "read_ticket":
		return abi.VerdictAllow
	default:
		return abi.VerdictDeny
	}
}

func localComparisonArm(name, class string, calls, correct int, elapsed time.Duration) ComparisonArm {
	return ComparisonArm{Name: name, Class: class, Available: true, Calls: calls, Correct: correct,
		Correctness: float64(correct) / float64(calls), ElapsedNS: elapsed.Nanoseconds(),
		ElapsedPerCallNS: float64(elapsed.Nanoseconds()) / float64(calls)}
}

func unavailableComparisonArm(name, class, reason string) ComparisonArm {
	return ComparisonArm{Name: name, Class: class, Available: false, UnavailableReason: reason}
}
