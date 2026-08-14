// Package speedab grades pinned Claude fast-versus-standard experiments.
package speedab

import (
	"fmt"
	"math"
	"sort"
	"strings"
)

const Schema = "fak.speed-ab.v1"

type Run struct {
	ID              string    `json:"id"`
	WorkClass       string    `json:"work_class"`
	Speed           string    `json:"speed"`
	Model           string    `json:"model"`
	Account         string    `json:"account"`
	Revision        string    `json:"revision"`
	DurationSeconds float64   `json:"duration_seconds"`
	ToolCalls       int       `json:"tool_calls"`
	TurnaroundMS    []float64 `json:"turnaround_ms"`
	Quality         string    `json:"quality"`
	Witness         string    `json:"witness"`
}

type Manifest struct {
	Schema string `json:"schema"`
	Runs   []Run  `json:"runs"`
}

type Arm struct {
	WorkClass        string  `json:"work_class"`
	Speed            string  `json:"speed"`
	Model            string  `json:"model"`
	Revision         string  `json:"revision"`
	Samples          int     `json:"samples"`
	P50MS            float64 `json:"p50_ms"`
	P90MS            float64 `json:"p90_ms"`
	ToolCallsPerHour float64 `json:"tool_calls_per_hour"`
	Quality          string  `json:"quality"`
	Witness          string  `json:"witness"`
}

type Comparison struct {
	WorkClass string  `json:"work_class"`
	Fast      Arm     `json:"fast"`
	Standard  Arm     `json:"standard"`
	P50Delta  float64 `json:"p50_delta_percent"`
	P90Delta  float64 `json:"p90_delta_percent"`
	ToolDelta float64 `json:"tool_calls_hour_delta_percent"`
	Verdict   string  `json:"verdict"`
	Reason    string  `json:"reason"`
}

type Report struct {
	Schema      string       `json:"schema"`
	Verdict     string       `json:"verdict"`
	Reason      string       `json:"reason,omitempty"`
	Comparisons []Comparison `json:"comparisons,omitempty"`
}

func Grade(m Manifest) Report {
	if m.Schema != Schema {
		return Report{Schema: Schema, Verdict: "NOT_YET", Reason: "schema mismatch"}
	}
	byClass := map[string]map[string]Run{}
	for _, r := range m.Runs {
		class := strings.TrimSpace(r.WorkClass)
		speed := strings.ToLower(strings.TrimSpace(r.Speed))
		if class == "" || (speed != "fast" && speed != "standard") {
			return Report{Schema: Schema, Verdict: "NOT_YET", Reason: "every run needs work_class and fast|standard speed"}
		}
		if r.Model == "" || r.Account == "" || r.Revision == "" || r.DurationSeconds <= 0 || len(r.TurnaroundMS) < 2 || r.Witness == "" || (r.Quality != "pass" && r.Quality != "fail") {
			return Report{Schema: Schema, Verdict: "NOT_YET", Reason: "run evidence incomplete or turnaround undersampled"}
		}
		if byClass[class] == nil {
			byClass[class] = map[string]Run{}
		}
		if _, duplicate := byClass[class][speed]; duplicate {
			return Report{Schema: Schema, Verdict: "NOT_YET", Reason: "duplicate arm for work class"}
		}
		byClass[class][speed] = r
	}
	classes := make([]string, 0, len(byClass))
	for class := range byClass {
		classes = append(classes, class)
	}
	sort.Strings(classes)
	report := Report{Schema: Schema, Verdict: "NET_TRUE"}
	for _, class := range classes {
		fast, fok := byClass[class]["fast"]
		standard, sok := byClass[class]["standard"]
		if !fok || !sok {
			return Report{Schema: Schema, Verdict: "NOT_YET", Reason: "missing fast or standard arm"}
		}
		if fast.Model != standard.Model || fast.Account != standard.Account || fast.Revision != standard.Revision {
			return Report{Schema: Schema, Verdict: "NOT_YET", Reason: "arms are not pinned to identical model, account, and revision"}
		}
		fa, sa := arm(fast), arm(standard)
		c := Comparison{WorkClass: class, Fast: fa, Standard: sa, P50Delta: delta(sa.P50MS, fa.P50MS), P90Delta: delta(sa.P90MS, fa.P90MS), ToolDelta: delta(sa.ToolCallsPerHour, fa.ToolCallsPerHour)}
		switch {
		case fa.Quality != "pass" || sa.Quality != "pass":
			c.Verdict, c.Reason = "NOT_YET", "quality witness failed"
		case c.P50Delta >= 0 || c.P90Delta >= 0:
			c.Verdict, c.Reason = "NOT_YET", "fast did not improve both latency quantiles"
		case strings.Contains(strings.ToLower(class), "grind") && c.ToolDelta <= 0:
			c.Verdict, c.Reason = "NOT_YET", "fast did not improve grind tool throughput"
		default:
			c.Verdict, c.Reason = "NET_TRUE", "faster against pinned standard with quality preserved"
		}
		if c.Verdict != "NET_TRUE" {
			report.Verdict = "NOT_YET"
		}
		report.Comparisons = append(report.Comparisons, c)
	}
	if len(report.Comparisons) == 0 {
		report.Verdict, report.Reason = "NOT_YET", "no comparisons"
	}
	return report
}

func arm(r Run) Arm {
	return Arm{WorkClass: r.WorkClass, Speed: r.Speed, Model: r.Model, Revision: r.Revision, Samples: len(r.TurnaroundMS), P50MS: quantile(r.TurnaroundMS, .5), P90MS: quantile(r.TurnaroundMS, .9), ToolCallsPerHour: float64(r.ToolCalls) * 3600 / r.DurationSeconds, Quality: r.Quality, Witness: r.Witness}
}
func quantile(v []float64, q float64) float64 {
	x := append([]float64(nil), v...)
	sort.Float64s(x)
	i := int(math.Ceil(q*float64(len(x)))) - 1
	if i < 0 {
		i = 0
	}
	return x[i]
}
func delta(base, candidate float64) float64 {
	if base == 0 {
		return 0
	}
	return (candidate - base) * 100 / base
}
func Validate(m Manifest) error {
	if r := Grade(m); r.Verdict == "NOT_YET" && len(r.Comparisons) == 0 {
		return fmt.Errorf("%s", r.Reason)
	}
	return nil
}
