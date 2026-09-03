package overtonscore

import (
	"encoding/json"
	"os"
	"path/filepath"

	"github.com/anthony-chaudhary/fak/pkg/scorecard"
)

// Schema is the schema identifier for overton baseline payloads.
const Schema = "fak-overton-baseline-debt/1"

// Key is the canonical key for overton baseline evaluations.
const Key = "overton_points"

// Disposition represents the normality band disposition of a metric observation.
type Disposition string

const (
	DispositionOrthodoxClean        Disposition = "orthodox_clean"
	DispositionAcceptedTemporary    Disposition = "accepted_temporary"
	DispositionAccidentalUnaccepted Disposition = "accidental_unaccepted"
)

// WindowBand defines the nested thresholds for normality evaluation.
type WindowBand struct {
	OrthodoxMin   float64 `json:"orthodox_min"`
	OrthodoxMax   float64 `json:"orthodox_max"`
	AcceptableMin float64 `json:"acceptable_min"`
	AcceptableMax float64 `json:"acceptable_max"`
	ElevatedMin   float64 `json:"elevated_min"`
	ElevatedMax   float64 `json:"elevated_max"`
}

// MetricEvaluation holds the evaluation result of a single metric against its window band.
type MetricEvaluation struct {
	Subsystem   string      `json:"subsystem"`
	Metric      string      `json:"metric"`
	Observed    float64     `json:"observed"`
	Band        WindowBand  `json:"band"`
	Disposition Disposition `json:"disposition"`
	Points      int         `json:"points"`
	Deficit     float64     `json:"deficit"`
	Surplus     float64     `json:"surplus"`
}

// Report captures the consolidated overton baseline evaluation.
type Report struct {
	Schema        string             `json:"schema"`
	Score         float64            `json:"score"`
	Grade         string             `json:"grade"`
	OvertonPoints int                `json:"overton_points"`
	Pressure      float64            `json:"pressure"`
	Slack         float64            `json:"slack"`
	Dispositions  map[string]int     `json:"dispositions"`
	Evaluations   []MetricEvaluation `json:"evaluations"`
}

// MetricSpec describes an input metric to be evaluated.
type MetricSpec struct {
	Subsystem string     `json:"subsystem"`
	Metric    string     `json:"metric"`
	Observed  float64    `json:"observed"`
	Band      WindowBand `json:"band"`
}

// Ready reports that the leaf is wired.
func Ready() bool { return true }

// EvaluateMetric evaluates an observed metric value against the given WindowBand.
//
// Rules:
// - If within [OrthodoxMin, OrthodoxMax]: orthodox_clean, Points: 0, Deficit: 0, Surplus is distance from bounds.
// - If within [AcceptableMin, AcceptableMax]: orthodox_clean / acceptable, Points: 0, Deficit: 0.
// - If within [ElevatedMin, ElevatedMax]: accepted_temporary, Points: 1, Deficit: > 0.
// - Outside: accidental_unaccepted, Points: 2, Deficit: >> 0.
func EvaluateMetric(subsystem, metric string, observed float64, band WindowBand) MetricEvaluation {
	ev := MetricEvaluation{
		Subsystem: subsystem,
		Metric:    metric,
		Observed:  observed,
		Band:      band,
	}

	if observed >= band.OrthodoxMin && observed <= band.OrthodoxMax {
		ev.Disposition = DispositionOrthodoxClean
		ev.Points = 0
		ev.Deficit = 0.0
		dMin := observed - band.OrthodoxMin
		dMax := band.OrthodoxMax - observed
		if dMin < dMax {
			ev.Surplus = dMin
		} else {
			ev.Surplus = dMax
		}
		return ev
	}

	if observed >= band.AcceptableMin && observed <= band.AcceptableMax {
		ev.Disposition = DispositionOrthodoxClean
		ev.Points = 0
		ev.Deficit = 0.0
		ev.Surplus = 0.0
		return ev
	}

	if observed >= band.ElevatedMin && observed <= band.ElevatedMax {
		ev.Disposition = DispositionAcceptedTemporary
		ev.Points = 1
		ev.Surplus = 0.0
		if observed < band.AcceptableMin {
			ev.Deficit = band.AcceptableMin - observed
		} else {
			ev.Deficit = observed - band.AcceptableMax
		}
		return ev
	}

	// Accidental unaccepted: observed is outside [ElevatedMin, ElevatedMax]
	ev.Disposition = DispositionAccidentalUnaccepted
	ev.Points = 2
	ev.Surplus = 0.0
	if observed < band.ElevatedMin {
		dElevated := band.ElevatedMin - observed
		dAcceptable := band.AcceptableMin - band.ElevatedMin
		ev.Deficit = dAcceptable + 2.0*dElevated
	} else {
		dElevated := observed - band.ElevatedMax
		dAcceptable := band.ElevatedMax - band.AcceptableMax
		ev.Deficit = dAcceptable + 2.0*dElevated
	}
	return ev
}

// FoldReport consolidates metric evaluations into a Report.
func FoldReport(evaluations []MetricEvaluation) Report {
	rep := Report{
		Schema:      Schema,
		Evaluations: evaluations,
		Dispositions: map[string]int{
			string(DispositionOrthodoxClean):        0,
			string(DispositionAcceptedTemporary):    0,
			string(DispositionAccidentalUnaccepted): 0,
		},
	}

	var totalDeficit float64
	var totalSurplus float64
	var scoreSum float64

	for _, ev := range evaluations {
		rep.OvertonPoints += ev.Points
		rep.Dispositions[string(ev.Disposition)]++
		totalDeficit += ev.Deficit
		totalSurplus += ev.Surplus

		switch ev.Disposition {
		case DispositionOrthodoxClean:
			scoreSum += 100.0
		case DispositionAcceptedTemporary:
			scoreSum += 60.0
		case DispositionAccidentalUnaccepted:
			scoreSum += 0.0
		}
	}

	if len(evaluations) > 0 {
		rep.Score = scorecard.Round1(scoreSum / float64(len(evaluations)))
	} else {
		rep.Score = 100.0
	}

	rep.Grade = scorecard.GradeStd(rep.Score)
	rep.Pressure = scorecard.Round3(totalDeficit)
	rep.Slack = scorecard.Round3(totalSurplus)

	return rep
}

func defaultBaselineSpecs() []MetricSpec {
	return []MetricSpec{
		// compute
		{
			Subsystem: "compute",
			Metric:    "compute_latency_ms",
			Observed:  2.5,
			Band: WindowBand{
				OrthodoxMin:   1.0,
				OrthodoxMax:   5.0,
				AcceptableMin: 0.5,
				AcceptableMax: 10.0,
				ElevatedMin:   0.1,
				ElevatedMax:   20.0,
			},
		},
		{
			Subsystem: "compute",
			Metric:    "utilization_ratio",
			Observed:  0.75,
			Band: WindowBand{
				OrthodoxMin:   0.60,
				OrthodoxMax:   0.85,
				AcceptableMin: 0.40,
				AcceptableMax: 0.90,
				ElevatedMin:   0.20,
				ElevatedMax:   0.98,
			},
		},
		// vcache
		{
			Subsystem: "vcache",
			Metric:    "hit_rate",
			Observed:  0.92,
			Band: WindowBand{
				OrthodoxMin:   0.85,
				OrthodoxMax:   1.00,
				AcceptableMin: 0.70,
				AcceptableMax: 1.00,
				ElevatedMin:   0.50,
				ElevatedMax:   1.00,
			},
		},
		{
			Subsystem: "vcache",
			Metric:    "churn_rate",
			Observed:  10.0,
			Band: WindowBand{
				OrthodoxMin:   0.0,
				OrthodoxMax:   20.0,
				AcceptableMin: 0.0,
				AcceptableMax: 40.0,
				ElevatedMin:   0.0,
				ElevatedMax:   80.0,
			},
		},
		// transport
		{
			Subsystem: "transport",
			Metric:    "p99_roundtrip_ms",
			Observed:  8.0,
			Band: WindowBand{
				OrthodoxMin:   2.0,
				OrthodoxMax:   15.0,
				AcceptableMin: 1.0,
				AcceptableMax: 30.0,
				ElevatedMin:   0.5,
				ElevatedMax:   60.0,
			},
		},
		{
			Subsystem: "transport",
			Metric:    "packet_drop_ratio",
			Observed:  0.0005,
			Band: WindowBand{
				OrthodoxMin:   0.0,
				OrthodoxMax:   0.001,
				AcceptableMin: 0.0,
				AcceptableMax: 0.005,
				ElevatedMin:   0.0,
				ElevatedMax:   0.02,
			},
		},
		// rules
		{
			Subsystem: "rules",
			Metric:    "eval_duration_us",
			Observed:  25.0,
			Band: WindowBand{
				OrthodoxMin:   5.0,
				OrthodoxMax:   50.0,
				AcceptableMin: 2.0,
				AcceptableMax: 100.0,
				ElevatedMin:   1.0,
				ElevatedMax:   250.0,
			},
		},
		{
			Subsystem: "rules",
			Metric:    "conflict_rate",
			Observed:  0.005,
			Band: WindowBand{
				OrthodoxMin:   0.0,
				OrthodoxMax:   0.01,
				AcceptableMin: 0.0,
				AcceptableMax: 0.03,
				ElevatedMin:   0.0,
				ElevatedMax:   0.08,
			},
		},
		// traces
		{
			Subsystem: "traces",
			Metric:    "retention_span_hours",
			Observed:  72.0,
			Band: WindowBand{
				OrthodoxMin:   24.0,
				OrthodoxMax:   168.0,
				AcceptableMin: 12.0,
				AcceptableMax: 336.0,
				ElevatedMin:   6.0,
				ElevatedMax:   720.0,
			},
		},
		{
			Subsystem: "traces",
			Metric:    "export_lag_ms",
			Observed:  120.0,
			Band: WindowBand{
				OrthodoxMin:   0.0,
				OrthodoxMax:   500.0,
				AcceptableMin: 0.0,
				AcceptableMax: 1500.0,
				ElevatedMin:   0.0,
				ElevatedMax:   5000.0,
			},
		},
	}
}

// Build evaluates core baseline metrics across standard subsystems and folds into Report.
func Build(workspace string) Report {
	specs := defaultBaselineSpecs()
	if workspace != "" {
		cfgPath := filepath.Join(workspace, "overton.json")
		if raw, err := os.ReadFile(cfgPath); err == nil {
			var parsed []MetricSpec
			if err := json.Unmarshal(raw, &parsed); err == nil && len(parsed) > 0 {
				specs = parsed
			}
		}
	}

	evaluations := make([]MetricEvaluation, 0, len(specs))
	for _, s := range specs {
		ev := EvaluateMetric(s.Subsystem, s.Metric, s.Observed, s.Band)
		evaluations = append(evaluations, ev)
	}

	return FoldReport(evaluations)
}
