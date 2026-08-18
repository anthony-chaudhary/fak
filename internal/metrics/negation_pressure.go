package metrics

import (
	"encoding/json"

	"github.com/anthony-chaudhary/fak/internal/mathx"
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
	negCorr := mathx.Pearson(pressure, negative)
	posCorr := mathx.Pearson(pressure, positive)
	return NegationPressureReport{Schema: NegationPressureSchema, Provenance: provenance, Buckets: rows, NegativePressureCorrelation: negCorr, PositivePressureCorrelation: posCorr, SignPinned: negCorr < 0}
}

func (r NegationPressureReport) JSON() []byte { b, _ := json.MarshalIndent(r, "", "  "); return b }
