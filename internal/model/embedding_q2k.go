package model

import (
	"errors"
	"fmt"
)

// ErrPackedEmbeddingWholeTableRefused is returned or panicked when whole-table expansion
// is attempted on a packed Q2_K embedding table.
var ErrPackedEmbeddingWholeTableRefused = errors.New("model: embedRows refused on packed Q2_K embedding: whole-table expansion of packed store is not supported")

// Q2KEmbedding holds a model-owned, immutable raw Q2_K embedding table
// used for on-demand row gathering without full F32 expansion.
type Q2KEmbedding struct {
	raw    []byte
	vocab  int
	hidden int
}

// NewQ2KEmbedding constructs a validated Q2KEmbedding instance.
// It creates an owned copy of data to guarantee immutability against caller mutation.
func NewQ2KEmbedding(data []byte, vocab, hidden int) (*Q2KEmbedding, error) {
	if vocab <= 0 || hidden <= 0 {
		return nil, fmt.Errorf("model: invalid embedding dimensions: vocab=%d, hidden=%d", vocab, hidden)
	}
	if hidden%qkK != 0 {
		return nil, fmt.Errorf("model: embedding hidden dimension %d is not divisible by %d", hidden, qkK)
	}
	wantBytes := int64(vocab) * int64(hidden/qkK) * q2kBlockBytes
	if int64(len(data)) != wantBytes {
		return nil, fmt.Errorf("model: embedding payload size mismatch: got %d bytes, want %d", len(data), wantBytes)
	}
	raw := append([]byte(nil), data...)
	return &Q2KEmbedding{
		raw:    raw,
		vocab:  vocab,
		hidden: hidden,
	}, nil
}

// Vocab returns the vocabulary size (number of token rows).
func (q *Q2KEmbedding) Vocab() int {
	if q == nil {
		return 0
	}
	return q.vocab
}

// Hidden returns the hidden dimension size (row width in elements).
func (q *Q2KEmbedding) Hidden() int {
	if q == nil {
		return 0
	}
	return q.hidden
}

// Bytes returns the total byte length of the packed embedding table.
func (q *Q2KEmbedding) Bytes() int {
	if q == nil {
		return 0
	}
	return len(q.raw)
}

// GatherRow dequantizes the superblock row for tokenID into dst (len >= hidden),
// optionally multiplying by scale if scale != 0 and scale != 1.0.
func (q *Q2KEmbedding) GatherRow(tokenID int, dst []float32, scale float32) error {
	if q == nil {
		return fmt.Errorf("model: Q2KEmbedding is nil")
	}
	if tokenID < 0 || tokenID >= q.vocab {
		return fmt.Errorf("model: token ID %d out of range [0, %d)", tokenID, q.vocab)
	}
	if len(dst) < q.hidden {
		return fmt.Errorf("model: destination buffer len %d < hidden %d", len(dst), q.hidden)
	}
	nBlocks := q.hidden / qkK
	rowBytes := nBlocks * q2kBlockBytes
	rowStart := tokenID * rowBytes
	rowData := q.raw[rowStart : rowStart+rowBytes]
	for b := 0; b < nBlocks; b++ {
		blk := rowData[b*q2kBlockBytes : (b+1)*q2kBlockBytes]
		q2kDequantSuperBlock(dst[b*qkK:(b+1)*qkK], blk)
	}
	if scale != 0 && scale != 1.0 {
		for i := 0; i < q.hidden; i++ {
			dst[i] *= scale
		}
	}
	return nil
}

// GatherRows dequantizes exactly the requested token rows into a compact row-major
// panel. Input order and repetitions are preserved; no unrequested vocabulary row
// is materialized. The returned slice is owned by the caller.
func (q *Q2KEmbedding) GatherRows(tokenIDs []int, scale float32) ([]float32, error) {
	if q == nil {
		return nil, fmt.Errorf("model: Q2KEmbedding is nil")
	}
	if len(tokenIDs) > 0 && q.hidden > int(^uint(0)>>1)/len(tokenIDs) {
		return nil, fmt.Errorf("model: Q2K embedding row panel size overflows int")
	}
	dst := make([]float32, len(tokenIDs)*q.hidden)
	for row, tokenID := range tokenIDs {
		if err := q.GatherRow(tokenID, dst[row*q.hidden:(row+1)*q.hidden], scale); err != nil {
			return nil, err
		}
	}
	return dst, nil
}

// DequantizeTable dequantizes all rows into a float32 slice of length vocab * hidden.
func (q *Q2KEmbedding) DequantizeTable() ([]float32, error) {
	if q == nil {
		return nil, fmt.Errorf("model: Q2KEmbedding is nil")
	}
	dst := make([]float32, q.vocab*q.hidden)
	for tokenID := 0; tokenID < q.vocab; tokenID++ {
		if err := q.GatherRow(tokenID, dst[tokenID*q.hidden:(tokenID+1)*q.hidden], 1.0); err != nil {
			return nil, err
		}
	}
	return dst, nil
}
