// sampling_topk_test.go — pins the device sampling-truncation family
// (sampling_topk.go/.cu) against the closed contract of the host
// applyCPUSamplingTruncation chain in internal/agent/inkernel_sampling.go:
//
//   (a) topK < vocab: the kept set equals the host total order — probability
//       descending, ties broken by ascending index (the probSorter tieByIndex
//       order) — with bit-equal surviving mass;
//   (b) topP cutoff mid-run: the post-truncation mass equals
//       applyCPUSamplingTruncation's returned mass, bit for bit;
//   (c) a NaN-logit row and a zero-mass refusal return the
//       ErrNonFiniteLogits-equivalent sentinel (errors.Is).
//
// The host chain is unexported; the oracles below transcribe its documented
// closed contract (total-order top-k + minimal-prefix nucleus + strict-CDF
// draw) independently, so this test compares two independent implementations
// of one contract. Rows are wide-vocabulary (>= the cluster sampling gate)
// because that is the regime the device path serves. CPU/in-memory evidence
// only: no GPU or throughput claim.

package compute_test

import (
	"errors"
	"math"
	"sort"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/compute"
)

const clusterMinVocab = 1 << 16 // mirrors agent.clusterSamplingMinVocabulary

// hostTotalOrder returns the indices of probs in the host total order:
// probability descending, ties by ascending index — the probSorter
// tieByIndex=true order the host truncations sort on.
func hostTotalOrder(probs []float64) []int {
	order := make([]int, len(probs))
	for i := range order {
		order[i] = i
	}
	sort.SliceStable(order, func(a, b int) bool {
		pa, pb := probs[order[a]], probs[order[b]]
		if pa != pb {
			return pa > pb
		}
		return order[a] < order[b]
	})
	return order
}

// oracleTopKTruncate mirrors the host topKTruncate closed contract: keep the k
// highest under the total order, zero the rest, return the surviving mass
// summed in ascending index order.
func oracleTopKTruncate(probs []float64, k int) float64 {
	order := hostTotalOrder(probs)
	kept := make(map[int]bool, k)
	for _, idx := range order[:k] {
		kept[idx] = true
	}
	var mass float64
	for i, p := range probs {
		if kept[i] {
			mass += p
		} else {
			probs[i] = 0
		}
	}
	return mass
}

// oracleNucleusTruncate mirrors the host nucleusTruncateFull closed contract:
// keep the smallest prefix of the total order whose mass reaches topP*sum
// (rank 0 always kept), zero the tail, return the index-order surviving mass.
func oracleNucleusTruncate(probs []float64, sum, topP float64) float64 {
	order := hostTotalOrder(probs)
	target := topP * sum
	var cum float64
	cutoff := len(order)
	for rank, idx := range order {
		if rank > 0 && cum >= target {
			cutoff = rank
			break
		}
		cum += probs[idx]
	}
	for _, idx := range order[cutoff:] {
		probs[idx] = 0
	}
	var mass float64
	for _, p := range probs {
		mass += p
	}
	return mass
}

// oracleCPUTruncation mirrors the host applyCPUSamplingTruncation pipeline:
// top-k first, then top-p on the post-top-k mass.
func oracleCPUTruncation(probs []float64, sum, topP float64, topK int) float64 {
	if topK > 0 && topK < len(probs) {
		sum = oracleTopKTruncate(probs, topK)
	}
	if topP > 0 && topP < 1 {
		sum = oracleNucleusTruncate(probs, sum, topP)
	}
	return sum
}

// wideTieRow builds a deterministic wide-vocabulary weight row: a strictly
// decreasing base with an engineered 200-wide tie plateau at value base[350]
// spanning indices [300,500), straddling any top-k boundary in that range.
func wideTieRow() ([]float64, float64) {
	probs := make([]float64, clusterMinVocab)
	for i := 0; i < clusterMinVocab; i++ {
		probs[i] = float64(1_000_000-i) * 1e-9
	}
	plateau := probs[350]
	for i := 300; i < 500; i++ {
		probs[i] = plateau
	}
	var sum float64
	for _, p := range probs {
		sum += p
	}
	return probs, sum
}

// geometricRow builds a wide-vocabulary row with distinct decaying weights
// (no ties), where a topP cut lands mid-row.
func geometricRow() ([]float64, float64) {
	probs := make([]float64, clusterMinVocab)
	for i := 0; i < clusterMinVocab; i++ {
		probs[i] = math.Exp(-float64(i) / 500.0)
	}
	var sum float64
	for _, p := range probs {
		sum += p
	}
	return probs, sum
}

func assertRowsMatch(t *testing.T, got, want []float64) {
	t.Helper()
	for i := range got {
		if got[i] != want[i] {
			t.Fatalf("row diverges at %d: device %v, host oracle %v", i, got[i], want[i])
		}
	}
}

func TestDeviceTopKTruncationMatchesHostOracle(t *testing.T) {
	t.Run("topK ties broken by index order", func(t *testing.T) {
		const k = 400
		probs, sum := wideTieRow()
		host := append([]float64(nil), probs...)
		wantMass := oracleCPUTruncation(host, sum, 0, k)

		gotMass, err := compute.DeviceSamplingTruncation(probs, sum, 0, k)
		if err != nil {
			t.Fatalf("device truncation refused a finite row: %v", err)
		}
		if gotMass != wantMass {
			t.Fatalf("surviving mass %v != host oracle %v", gotMass, wantMass)
		}
		assertRowsMatch(t, probs, host)
		// Pin the tie-break directly: the plateau spans ranks 300..499, so the
		// kept plateau members must be exactly indices [300,400).
		for i := 300; i < 500; i++ {
			if (probs[i] > 0) != (i < 400) {
				t.Fatalf("tie-break: index %d kept=%v, want %v", i, probs[i] > 0, i < 400)
			}
		}
	})

	t.Run("topP cutoff mid-run", func(t *testing.T) {
		probs, sum := geometricRow()
		const topP = 0.7
		host := append([]float64(nil), probs...)
		wantMass := oracleCPUTruncation(host, sum, topP, 0)

		gotMass, err := compute.DeviceSamplingTruncation(probs, sum, topP, 0)
		if err != nil {
			t.Fatalf("device truncation refused a finite row: %v", err)
		}
		if gotMass != wantMass {
			t.Fatalf("post-truncation mass %v != host oracle %v", gotMass, wantMass)
		}
		assertRowsMatch(t, probs, host)
		nonzero := 0
		for _, p := range host {
			if p > 0 {
				nonzero++
			}
		}
		if nonzero <= 1 || nonzero >= clusterMinVocab {
			t.Fatalf("fixture did not cut mid-run: %d survivors of %d", nonzero, clusterMinVocab)
		}
	})

	t.Run("topK then topP combined", func(t *testing.T) {
		probs, sum := wideTieRow()
		const topP, k = 0.9, 1024
		host := append([]float64(nil), probs...)
		wantMass := oracleCPUTruncation(host, sum, topP, k)

		gotMass, err := compute.DeviceSamplingTruncation(probs, sum, topP, k)
		if err != nil {
			t.Fatalf("device truncation refused a finite row: %v", err)
		}
		if gotMass != wantMass {
			t.Fatalf("post-truncation mass %v != host oracle %v", gotMass, wantMass)
		}
		assertRowsMatch(t, probs, host)
	})

	t.Run("topP near one keeps almost all", func(t *testing.T) {
		probs, sum := geometricRow()
		const topP = 1 - 1e-12
		host := append([]float64(nil), probs...)
		wantMass := oracleCPUTruncation(host, sum, topP, 0)

		gotMass, err := compute.DeviceSamplingTruncation(probs, sum, topP, 0)
		if err != nil {
			t.Fatalf("device truncation refused a finite row: %v", err)
		}
		if gotMass != wantMass {
			t.Fatalf("post-truncation mass %v != host oracle %v", gotMass, wantMass)
		}
		assertRowsMatch(t, probs, host)
	})

	t.Run("NaN row refused", func(t *testing.T) {
		probs, sum := wideTieRow()
		probs[1234] = math.NaN()
		_, err := compute.DeviceSamplingTruncation(probs, sum, 0.5, 1024)
		if !errors.Is(err, compute.ErrNonFiniteLogits) {
			t.Fatalf("want ErrNonFiniteLogits sentinel, got %v", err)
		}
	})

	t.Run("+Inf row refused", func(t *testing.T) {
		probs, sum := wideTieRow()
		probs[1234] = math.Inf(1)
		_, err := compute.DeviceSamplingTruncation(probs, sum, 0.5, 1024)
		if !errors.Is(err, compute.ErrNonFiniteLogits) {
			t.Fatalf("want ErrNonFiniteLogits sentinel, got %v", err)
		}
	})

	t.Run("zero-mass row refused", func(t *testing.T) {
		probs := make([]float64, clusterMinVocab) // all-zero weights: no finite logit survived
		_, err := compute.DeviceSamplingTruncation(probs, 0, 0.5, 1024)
		if !errors.Is(err, compute.ErrNonFiniteLogits) {
			t.Fatalf("want ErrNonFiniteLogits sentinel for zero mass, got %v", err)
		}
	})

	t.Run("draw strict CDF boundary", func(t *testing.T) {
		row := []float64{0, 0.5, 0.5}
		if got := compute.DrawToken(row, 1, 0); got != 1 {
			t.Fatalf("coin 0: want token 1 (zero-mass tokens never selected), got %d", got)
		}
		if got := compute.DrawToken(row, 1, 0.5); got != 2 {
			t.Fatalf("coin on CDF boundary: want token 2 (boundary belongs to next interval), got %d", got)
		}
		if got := compute.DrawToken(row, 1, 0.4999999); got != 1 {
			t.Fatalf("coin below boundary: want token 1, got %d", got)
		}
		if got := compute.DrawToken([]float64{0, 0, 0.25}, 0.25, 0.9999999); got != 2 {
			t.Fatalf("exhausted row: want last nonzero token 2, got %d", got)
		}
	})

	t.Run("repetition penalties match host formula", func(t *testing.T) {
		logits := make([]float32, clusterMinVocab)
		for i := range logits {
			logits[i] = float32(i%7 - 3)
		}
		counts := make([]int32, clusterMinVocab)
		counts[10] = 3
		counts[1<<15] = 1
		counts[4096] = 0 // skipped: count <= 0
		const freq, pres = 0.7, 0.3
		host := append([]float32(nil), logits...)
		for tok, c := range counts {
			if tok >= len(host) || c <= 0 {
				continue
			}
			host[tok] -= float32(freq*float64(c) + pres)
		}
		compute.ApplyRepetitionPenalties(logits, freq, pres, counts)
		for i := range logits {
			if logits[i] != host[i] {
				t.Fatalf("logit[%d] = %v, host formula %v", i, logits[i], host[i])
			}
		}
		// Zero-penalty no-op: a fresh row is untouched.
		fresh := append([]float32(nil), logits...)
		compute.ApplyRepetitionPenalties(fresh, 0, 0, counts)
		for i := range fresh {
			if fresh[i] != logits[i] {
				t.Fatalf("zero-penalty no-op altered logit[%d]", i)
			}
		}
	})
}
