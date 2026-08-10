package ctxmmu

import (
	"runtime"
	"strings"
	"time"
)

const ComparisonSchema = "fak-context-memory-comparison/1"

type ComparisonCase struct {
	Name      string `json:"name"`
	Body      []byte `json:"-"`
	WantAdmit bool   `json:"want_admit"`
}

type ComparisonArm struct {
	Name               string  `json:"name"`
	Class              string  `json:"class"`
	Available          bool    `json:"available"`
	Writes             int     `json:"writes,omitempty"`
	Correct            int     `json:"correct,omitempty"`
	Correctness        float64 `json:"correctness,omitempty"`
	ElapsedNS          int64   `json:"elapsed_ns,omitempty"`
	ElapsedPerWriteNS  float64 `json:"elapsed_per_write_ns,omitempty"`
	RetainedFactRecall float64 `json:"retained_fact_recall,omitempty"`
	PeakRSSBytes       int64   `json:"peak_rss_bytes,omitempty"`
	TotalCostUSD       float64 `json:"total_cost_usd,omitempty"`
	UnavailableReason  string  `json:"unavailable_reason,omitempty"`
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

// ComparisonCorpus exercises the native durable-memory write seam. The full
// live workload must also test read-back and end-task recall; these local cases
// cannot establish memory-system quality by themselves.
func ComparisonCorpus() []ComparisonCase {
	return []ComparisonCase{
		{Name: "explicit-fact", Body: []byte("The account renewal date is 2030-04-17, confirmed by the operator."), WantAdmit: true},
		{Name: "oversize", Body: makeOversizeComparisonBody(), WantAdmit: false},
		{Name: "secret", Body: []byte("prod api key is sk-" + strings.Repeat("a", 16)), WantAdmit: false},
	}
}

func makeOversizeComparisonBody() []byte {
	body := make([]byte, MemoryWriteMaxBytes+1)
	for i := range body {
		body[i] = 'x'
	}
	return body
}

// CompareLocal runs the real structural memory-write gate and a tuned
// full-history/no-memory-management baseline over identical bytes. Every
// first-class external memory integration remains a separate unavailable arm.
func CompareLocal(iterations int) ComparisonReport {
	if iterations <= 0 {
		iterations = 10000
	}
	corpus := ComparisonCorpus()
	native := NewMemoryWriteAdjudicator()

	nativeCorrect := 0
	start := time.Now()
	for i := 0; i < iterations; i++ {
		for _, c := range corpus {
			if native.AdmitWrite(c.Body).Admit == c.WantAdmit {
				nativeCorrect++
			}
		}
	}
	nativeElapsed := time.Since(start)

	baselineCorrect := 0
	start = time.Now()
	for i := 0; i < iterations; i++ {
		for _, c := range corpus {
			if true == c.WantAdmit {
				baselineCorrect++
			}
		}
	}
	baselineElapsed := time.Since(start)
	total := iterations * len(corpus)

	return ComparisonReport{
		Schema:   ComparisonSchema,
		Workload: "identical candidate memory writes locally; live completion adds identical long-horizon tasks, read-back queries, model, token budget, and grader",
		GOOS:     runtime.GOOS, GOARCH: runtime.GOARCH,
		Arms: []ComparisonArm{
			localComparisonArm("fak native context-memory gate", "native", total, nativeCorrect, nativeElapsed),
			localComparisonArm("retain full history without memory management", "tuned_baseline", total, baselineCorrect, baselineElapsed),
			unavailableComparisonArm("Letta", "next_best", "requires real long-horizon Letta memory-block and archival-memory read/write runs"),
			unavailableComparisonArm("fak + mem0", "first_class_integration", "requires real mem0 write/read-back runs through the fak memory integration contract"),
			unavailableComparisonArm("fak + Letta", "first_class_integration", "requires real Letta write/read-back runs through the fak memory integration contract"),
			unavailableComparisonArm("fak + Zep/Graphiti", "first_class_integration", "requires real temporal-memory write/read-back runs through the fak memory integration contract"),
			unavailableComparisonArm("fak + LangMem/LangGraph memory", "first_class_integration", "requires real manage_memory/search_memory runs through the fak integration contract"),
		},
		Complete: false,
		Verdict:  "local write-admission behavior and overhead only; no net-true memory-system winner until all live arms report task success, retained-fact recall, latency, tokens/resources, and total cost",
	}
}

func localComparisonArm(name, class string, writes, correct int, elapsed time.Duration) ComparisonArm {
	return ComparisonArm{Name: name, Class: class, Available: true, Writes: writes, Correct: correct,
		Correctness: float64(correct) / float64(writes), ElapsedNS: elapsed.Nanoseconds(),
		ElapsedPerWriteNS: float64(elapsed.Nanoseconds()) / float64(writes)}
}

func unavailableComparisonArm(name, class, reason string) ComparisonArm {
	return ComparisonArm{Name: name, Class: class, Available: false, UnavailableReason: reason}
}
