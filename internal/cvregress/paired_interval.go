package cvregress

import (
	"errors"
	"math"
	"math/rand"
	"sort"
)

type PairedInterval struct {
	Mean, Lower, Upper, Confidence float64
	Samples, Resamples             int
	Seed                           int64
	Conclusive                     bool
	Finding                        string
}

func PairedConfidenceInterval(baseline, candidate []float64, confidence float64, resamples int, seed int64) (PairedInterval, error) {
	if len(baseline) != len(candidate) || len(baseline) < 2 {
		return PairedInterval{}, errors.New("paired interval requires equal samples >=2")
	}
	if confidence <= 0 || confidence >= 1 || resamples < 100 {
		return PairedInterval{}, errors.New("invalid confidence/resamples")
	}
	d := make([]float64, len(baseline))
	for i := range d {
		if math.IsNaN(baseline[i]) || math.IsNaN(candidate[i]) {
			return PairedInterval{}, errors.New("non-finite score")
		}
		d[i] = candidate[i] - baseline[i]
	}
	rng := rand.New(rand.NewSource(seed))
	means := make([]float64, resamples)
	for r := range means {
		var sum float64
		for range d {
			sum += d[rng.Intn(len(d))]
		}
		means[r] = sum / float64(len(d))
	}
	sort.Float64s(means)
	alpha := (1 - confidence) / 2
	lo := means[int(alpha*float64(resamples))]
	hi := means[int((1-alpha)*float64(resamples))-1]
	var sum float64
	for _, v := range d {
		sum += v
	}
	return PairedInterval{Mean: sum / float64(len(d)), Lower: lo, Upper: hi, Confidence: confidence, Samples: len(d), Resamples: resamples, Seed: seed, Conclusive: true, Finding: "paired deterministic bootstrap"}, nil
}
