package metrics

import (
	"encoding/json"
	"math"
)

const NegationPressureSchema = "fak.negation_pressure.v1"

type NegationPressureBucket struct {
	Bucket           string  `json:"bucket"`
	Pressure         float64 `json:"pressure"`
	NegativePassRate float64 `json:"negative_pass_rate"`
	PositivePassRate float64 `json:"positive_pass_rate"`
	FramingDelta     float64 `json:"framing_delta"`
	SamplesPerArm    int     `json:"samples_per_arm"`
}

type NegationPressureReport struct {
	Schema                      string                   `json:"schema"`
	Provenance                  string                   `json:"provenance"`
	Buckets                     []NegationPressureBucket `json:"buckets"`
	NegativePressureCorrelation float64                  `json:"negative_pressure_correlation"`
	PositivePressureCorrelation float64                  `json:"positive_pressure_correlation"`
	SignPinned                  bool                     `json:"sign_pinned"`
}

func FoldNegationPressure(provenance string, buckets []NegationPressureBucket) NegationPressureReport {
	pressure := make([]float64, 0, len(buckets))
	negative := make([]float64, 0, len(buckets))
	positive := make([]float64, 0, len(buckets))
	rows := append([]NegationPressureBucket(nil), buckets...)
	for i := range rows {
		rows[i].FramingDelta = rows[i].PositivePassRate - rows[i].NegativePassRate
		pressure = append(pressure, rows[i].Pressure)
		negative = append(negative, rows[i].NegativePassRate)
		positive = append(positive, rows[i].PositivePassRate)
	}
	negCorr := correlation(pressure, negative)
	posCorr := correlation(pressure, positive)
	return NegationPressureReport{Schema: NegationPressureSchema, Provenance: provenance, Buckets: rows, NegativePressureCorrelation: negCorr, PositivePressureCorrelation: posCorr, SignPinned: negCorr < 0}
}

func (r NegationPressureReport) JSON() []byte { b, _ := json.MarshalIndent(r, "", "  "); return b }

func correlation(x, y []float64) float64 {
	if len(x) != len(y) || len(x) == 0 {
		return 0
	}
	var sx, sy float64
	for i := range x {
		sx += x[i]
		sy += y[i]
	}
	mx, my := sx/float64(len(x)), sy/float64(len(y))
	var num, dx, dy float64
	for i := range x {
		a, b := x[i]-mx, y[i]-my
		num += a * b
		dx += a * a
		dy += b * b
	}
	if dx == 0 || dy == 0 {
		return 0
	}
	return num / math.Sqrt(dx*dy)
}
