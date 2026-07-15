package bench

import "github.com/anthony-chaudhary/fak/internal/metrics"

const NegationPressureModeledProvenance = "MODELED / OFFLINE PROXY"

type NegationPressureObservation struct {
	ItemID   string  `json:"item_id"`
	Bucket   string  `json:"bucket"`
	Pressure float64 `json:"pressure"`
	Arm      string  `json:"arm"`
	Passed   bool    `json:"passed"`
}

// RunNegationPressureProbe expands the held-out matched pairs across fixed
// pressure buckets. The unit path remains deterministic and explicitly modeled.
func RunNegationPressureProbe(items []NegatedQAItem) ([]NegationPressureObservation, metrics.NegationPressureReport) {
	buckets := []struct {
		name                             string
		pressure                         float64
		negativePenalty, positivePenalty float64
	}{
		{"low", 0.20, 0.03, 0.01}, {"mid", 0.60, 0.14, 0.04}, {"near-budget", 0.92, 0.31, 0.08},
	}
	var observations []NegationPressureObservation
	rows := make([]metrics.NegationPressureBucket, 0, len(buckets))
	for _, bucket := range buckets {
		negPass, posPass := 0, 0
		for i, item := range items {
			// Stable threshold dithering prevents fractional modeled rates from
			// becoming a claim about a live model while retaining item-level rows.
			negFailure := item.HighFailure + bucket.negativePenalty
			posFailure := item.LowFailure + bucket.positivePenalty
			negOK := float64((i*37+11)%100)/100 >= negFailure
			posOK := float64((i*53+29)%100)/100 >= posFailure
			if negOK {
				negPass++
			}
			if posOK {
				posPass++
			}
			observations = append(observations, NegationPressureObservation{item.ID, bucket.name, bucket.pressure, "negative", negOK}, NegationPressureObservation{item.ID, bucket.name, bucket.pressure, "positive", posOK})
		}
		rows = append(rows, metrics.NegationPressureBucket{Bucket: bucket.name, Pressure: bucket.pressure, NegativePassRate: float64(negPass) / float64(len(items)), PositivePassRate: float64(posPass) / float64(len(items)), SamplesPerArm: len(items)})
	}
	return observations, metrics.FoldNegationPressure(NegationPressureModeledProvenance, rows)
}
