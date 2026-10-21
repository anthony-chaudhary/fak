package model

import (
	"errors"
	"fmt"
	"math"
)

// Adapted from DeepInfer at 774ce244; see third_party/deepinfer-v41-admission/NOTICE.

// ErrV41EngramDerivation refuses a converter-supplied Engram layout whose table
// primes, pad id or hash multipliers disagree with what DeepInfer derives from
// the published config.
var ErrV41EngramDerivation = errors.New("model: V4.1 Engram layout disagrees with its derivation")

const (
	v41PublishedEngramVocabSize = 16000000
	v41PublishedEngramMaxNgram  = 4
	v41PublishedEngramHeads     = 8
	v41EngramRawPadTokenID      = 2
)

var v41PublishedEngramLayerIDs = []int{1, 14}

// ValidateV41PublishedEngramLayout checks a layout with the published
// V4.1-Flash Engram geometry against its derivation. Neither GGUF dialect
// carries the table vocabulary, so the published one is assumed and any other
// geometry is admitted unchecked.
func ValidateV41PublishedEngramLayout(layerIDs []int, l V41EngramLayout) error {
	if !v41EqualInts(layerIDs, v41PublishedEngramLayerIDs) ||
		l.MaxNgramSize != v41PublishedEngramMaxNgram || l.HeadsPerNgram != v41PublishedEngramHeads {
		return nil
	}
	return validateV41EngramDerived(l, v41PublishedEngramVocabSize)
}

func validateV41EngramDerived(l V41EngramLayout, vocab int) error {
	want, err := deriveV41EngramPrimes(vocab, l.HeadsPerNgram, l.MaxNgramSize, len(l.Primes))
	if err != nil {
		return err
	}
	for layer := range want {
		if len(l.Primes[layer]) != len(want[layer]) {
			return fmt.Errorf("%w: layer %d has %d primes, want %d", ErrV41EngramDerivation, layer, len(l.Primes[layer]), len(want[layer]))
		}
		for col, p := range want[layer] {
			if l.Primes[layer][col] != p {
				return fmt.Errorf("%w: layer %d prime[%d]=%d, derived %d", ErrV41EngramDerivation, layer, col, l.Primes[layer][col], p)
			}
		}
	}
	if len(l.TokenMap) <= v41EngramRawPadTokenID || l.TokenMap[v41EngramRawPadTokenID] != l.PadID {
		return fmt.Errorf("%w: pad id %d is not token_map[%d]", ErrV41EngramDerivation, l.PadID, v41EngramRawPadTokenID)
	}
	if l.CompressedVocab == 0 {
		return fmt.Errorf("%w: empty compressed vocabulary", ErrV41EngramDerivation)
	}
	limit := 2 * max(1, (uint64(math.MaxInt64)/uint64(l.CompressedVocab))/2)
	for layer, ms := range l.Multipliers {
		for i, m := range ms {
			if m&1 == 0 || m >= limit {
				return fmt.Errorf("%w: layer %d multiplier[%d]=%d outside odd [1,%d)", ErrV41EngramDerivation, layer, i, m, limit)
			}
		}
	}
	return nil
}

// deriveV41EngramPrimes returns, per layer, the [ngram][head]-flattened table
// sizes: for each ngram table the search restarts at vocab-1, takes the next
// prime per head, and skips any prime already used by an earlier layer or head.
func deriveV41EngramPrimes(vocab, heads, maxNgram, layers int) ([][]uint32, error) {
	if vocab < 1 || heads < 1 || maxNgram < 2 || layers < 1 {
		return nil, fmt.Errorf("%w: derivation geometry vocab=%d heads=%d max_ngram=%d layers=%d", ErrV41EngramDerivation, vocab, heads, maxNgram, layers)
	}
	seen := make(map[uint64]bool, layers*(maxNgram-1)*heads)
	out := make([][]uint32, layers)
	for layer := range out {
		out[layer] = make([]uint32, 0, (maxNgram-1)*heads)
		for ng := 0; ng < maxNgram-1; ng++ {
			start := uint64(vocab) - 1
			for h := 0; h < heads; h++ {
				p := v41EngramNextPrime(start)
				for seen[p] {
					p = v41EngramNextPrime(p)
				}
				if p > math.MaxUint32 {
					return nil, fmt.Errorf("%w: derived prime %d overflows uint32", ErrV41EngramDerivation, p)
				}
				seen[p] = true
				out[layer] = append(out[layer], uint32(p))
				start = p
			}
		}
	}
	return out, nil
}

func v41EngramNextPrime(n uint64) uint64 {
	for c := n + 1; ; c++ {
		if v41EngramIsPrime(c) {
			return c
		}
	}
}

func v41EngramIsPrime(n uint64) bool {
	if n < 2 {
		return false
	}
	if n%2 == 0 {
		return n == 2
	}
	for d := uint64(3); d*d <= n; d += 2 {
		if n%d == 0 {
			return false
		}
	}
	return true
}
