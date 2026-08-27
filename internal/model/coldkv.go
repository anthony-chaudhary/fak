package model

import (
	"fmt"
	"math"
	"time"
)

// ColdKVRow stores one cold row as groupwise Q4 plus sparse residual recovery.
type ColdKVRow struct {
	Quantized     KVQuant4  `json:"-"`
	ResidualIndex []uint32  `json:"residual_index,omitempty"`
	ResidualValue []float32 `json:"residual_value,omitempty"`
}

// ColdKVWindow keeps recent rows exact while compressing older rows.
type ColdKVWindow struct {
	Cold     []ColdKVRow `json:"cold"`
	Hot      [][]float32 `json:"hot"`
	RowWidth int         `json:"row_width"`
}

type ColdKVReceipt struct {
	Schema                  string        `json:"schema"`
	Engine                  string        `json:"engine"`
	Rows                    int           `json:"rows"`
	ColdRows                int           `json:"cold_rows"`
	HotRows                 int           `json:"hot_rows"`
	OriginalBytes           int64         `json:"original_bytes"`
	CompressedBytes         int64         `json:"compressed_bytes"`
	ReadBytesAvoided        int64         `json:"read_bytes_avoided"`
	BytesAvoidedPerAccepted float64       `json:"bytes_avoided_per_accepted_token"`
	ResidualValues          int           `json:"residual_values"`
	MaxAbsoluteError        float32       `json:"max_absolute_error"`
	EncodeLatency           time.Duration `json:"encode_latency"`
	QualityConstraint       string        `json:"quality_constraint"`
	Rollback                string        `json:"rollback"`
}

// CompressColdKV applies Q4 only outside the exact hot window. Quantization errors
// above residualThreshold are stored sparsely and exactly recovered.
func CompressColdKV(rows [][]float32, hotRows, acceptedTokens int, residualThreshold float32) (ColdKVWindow, ColdKVReceipt, error) {
	if hotRows < 0 || hotRows > len(rows) || acceptedTokens < 0 || residualThreshold < 0 {
		return ColdKVWindow{}, ColdKVReceipt{}, fmt.Errorf("model: invalid cold KV envelope")
	}
	width := 0
	if len(rows) > 0 {
		width = len(rows[0])
	}
	for _, row := range rows {
		if len(row) != width {
			return ColdKVWindow{}, ColdKVReceipt{}, fmt.Errorf("model: ragged cold KV rows")
		}
	}
	start := time.Now()
	split := len(rows) - hotRows
	w := ColdKVWindow{Cold: make([]ColdKVRow, split), Hot: make([][]float32, hotRows), RowWidth: width}
	r := ColdKVReceipt{Schema: "fak-cold-kv-receipt/1", Engine: "fak-native", Rows: len(rows), ColdRows: split, HotRows: hotRows, OriginalBytes: int64(len(rows) * width * 4), QualityConstraint: "hot rows bit-exact; cold max error bounded by residual threshold", Rollback: "retain all rows at FP32"}
	for i := 0; i < split; i++ {
		q := QuantizeKV4(rows[i])
		decoded := q.Dequantize()
		cr := ColdKVRow{Quantized: q}
		for j, want := range rows[i] {
			err := want - decoded[j]
			if float32(math.Abs(float64(err))) > residualThreshold {
				cr.ResidualIndex = append(cr.ResidualIndex, uint32(j))
				cr.ResidualValue = append(cr.ResidualValue, err)
			}
		}
		w.Cold[i] = cr
		r.CompressedBytes += int64(q.Bytes() + len(cr.ResidualIndex)*4 + len(cr.ResidualValue)*4)
		r.ResidualValues += len(cr.ResidualIndex)
	}
	for i := 0; i < hotRows; i++ {
		w.Hot[i] = append([]float32(nil), rows[split+i]...)
		r.CompressedBytes += int64(width * 4)
	}
	restored := w.Decompress()
	for i := range rows {
		for j := range rows[i] {
			e := float32(math.Abs(float64(rows[i][j] - restored[i][j])))
			if e > r.MaxAbsoluteError {
				r.MaxAbsoluteError = e
			}
		}
	}
	r.ReadBytesAvoided = r.OriginalBytes - r.CompressedBytes
	if r.ReadBytesAvoided < 0 {
		r.ReadBytesAvoided = 0
	}
	if acceptedTokens > 0 {
		r.BytesAvoidedPerAccepted = float64(r.ReadBytesAvoided) / float64(acceptedTokens)
	}
	r.EncodeLatency = time.Since(start)
	return w, r, nil
}

func (w ColdKVWindow) Decompress() [][]float32 {
	out := make([][]float32, 0, len(w.Cold)+len(w.Hot))
	for _, cr := range w.Cold {
		row := cr.Quantized.Dequantize()
		for i, idx := range cr.ResidualIndex {
			if int(idx) < len(row) {
				row[idx] += cr.ResidualValue[i]
			}
		}
		out = append(out, row)
	}
	for _, row := range w.Hot {
		out = append(out, append([]float32(nil), row...))
	}
	return out
}
