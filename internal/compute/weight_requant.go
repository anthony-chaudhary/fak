package compute

import (
	"sort"
	"strings"
	"sync"
)

// WeightRequant aggregates weight uploads a backend stored in a wider dtype than the
// file carried (for example a raw i-quant expanded to Q8_0 because no native kernel is
// built). FileBytes is the packed source size, ResidentBytes the size actually kept.
type WeightRequant struct {
	From          string `json:"from"`
	To            string `json:"to"`
	Tensors       int64  `json:"tensors"`
	FileBytes     int64  `json:"file_bytes"`
	ResidentBytes int64  `json:"resident_bytes"`
}

// Label is the closed "<from>-><to>" spelling used on receipts and perf lines.
func (r WeightRequant) Label() string { return r.From + "->" + r.To }

type weightRequantKey struct{ from, to Dtype }

var weightRequants struct {
	mu   sync.Mutex
	rows map[weightRequantKey]*WeightRequant
}

// RecordWeightRequant counts one weight upload stored as to instead of its file dtype
// from. It is process-wide: a serve holds one model, and the ledger is the evidence that
// the requantization actually ran, not a prediction.
func RecordWeightRequant(from, to Dtype, fileBytes, residentBytes int) {
	if from == to || fileBytes <= 0 || residentBytes <= 0 {
		return
	}
	weightRequants.mu.Lock()
	defer weightRequants.mu.Unlock()
	if weightRequants.rows == nil {
		weightRequants.rows = map[weightRequantKey]*WeightRequant{}
	}
	k := weightRequantKey{from, to}
	r := weightRequants.rows[k]
	if r == nil {
		r = &WeightRequant{From: WeightDtypeName(from), To: WeightDtypeName(to)}
		weightRequants.rows[k] = r
	}
	r.Tensors++
	r.FileBytes += int64(fileBytes)
	r.ResidentBytes += int64(residentBytes)
}

// WeightDtypeName is dt in the GGUF spelling ("IQ3_XXS") receipts use.
func WeightDtypeName(dt Dtype) string { return strings.ToUpper(dt.String()) }

// WeightRequantSnapshot returns the recorded requantizations ordered by label.
func WeightRequantSnapshot() []WeightRequant {
	weightRequants.mu.Lock()
	out := make([]WeightRequant, 0, len(weightRequants.rows))
	for _, r := range weightRequants.rows {
		out = append(out, *r)
	}
	weightRequants.mu.Unlock()
	sort.Slice(out, func(i, j int) bool { return out[i].Label() < out[j].Label() })
	return out
}

// ResetWeightRequantsForTest clears the process-wide ledger.
func ResetWeightRequantsForTest() {
	weightRequants.mu.Lock()
	weightRequants.rows = nil
	weightRequants.mu.Unlock()
}

// WeightRequantizer is the optional backend probe for a weight dtype the backend admits
// but stores in a wider representation at Upload.
type WeightRequantizer interface {
	RequantizesWeightDtype(dt Dtype) (Dtype, bool)
}

// BackendRequantizesWeightDtype reports the dtype be would store dt as when that differs
// from dt. A backend without the probe is reported as storing dt natively.
func BackendRequantizesWeightDtype(be Backend, dt Dtype) (Dtype, bool) {
	r, ok := be.(WeightRequantizer)
	if !ok {
		return dt, false
	}
	return r.RequantizesWeightDtype(dt)
}

// Q8ResidentBytes is the resident size of an [out,in] Q8_0 weight with block-size
// float32 scales.
func Q8ResidentBytes(out, in, block int) int64 {
	if out <= 0 || in <= 0 || block <= 0 {
		return 0
	}
	n := int64(out) * int64(in)
	return n + 4*((n+int64(block)-1)/int64(block))
}
