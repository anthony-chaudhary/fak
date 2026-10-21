package model

import (
	"errors"
	"math"
	"testing"
)

// fak-test:justify why=contract when=changed:internal/model/**
// fak-test:runtime fast est=5ms lane=default
func TestV41EngramPrimeDerivation(t *testing.T) {
	primes, err := deriveV41EngramPrimes(16000000, 8, 4, 2)
	if err != nil {
		t.Fatal(err)
	}
	want := []struct {
		sum                  uint64
		first, trigram, last uint32
	}{{384006168, 16000057, 16000189, 16000463}, {384016682, 16000477, 16000667, 16000889}}
	prev := uint32(0)
	for layer, w := range want {
		p := primes[layer]
		if len(p) != 24 {
			t.Fatalf("layer %d: %d columns, want 24", layer, len(p))
		}
		var sum uint64
		for _, v := range p {
			if v <= prev {
				t.Fatalf("layer %d: prime %d not above %d", layer, v, prev)
			}
			prev = v
			sum += uint64(v)
		}
		if sum != w.sum || p[0] != w.first || p[8] != w.trigram || p[23] != w.last {
			t.Errorf("layer %d: sum=%d first=%d [8]=%d last=%d, want %d %d %d %d", layer, sum, p[0], p[8], p[23], w.sum, w.first, w.trigram, w.last)
		}
	}
	if p, err := deriveV41EngramPrimes(17, 2, 2, 1); err != nil || p[0][0] != 17 || p[0][1] != 19 {
		t.Errorf("prime vocab: %v %v, want [[17 19]]", p, err)
	}
	if _, err := deriveV41EngramPrimes(0, 8, 4, 2); !errors.Is(err, ErrV41EngramDerivation) {
		t.Errorf("zero vocab: err = %v, want ErrV41EngramDerivation", err)
	}
}

func v41DerivedEngramLayout(t *testing.T) V41EngramLayout {
	t.Helper()
	primes, err := deriveV41EngramPrimes(16000000, 8, 4, 2)
	if err != nil {
		t.Fatal(err)
	}
	l := V41EngramLayout{
		TokenMap:        []uint32{0, 1, 2, 3, 4, 5, 6, 7},
		CompressedVocab: 99092,
		PadID:           2,
		Multipliers:     [][]uint64{{1, 3, 5, 7}, {9, 11, 13, 15}},
		Primes:          primes,
		MaxNgramSize:    4,
		HeadsPerNgram:   8,
	}
	for _, p := range primes {
		var rows uint32
		for _, v := range p {
			rows += v
		}
		l.Rows = append(l.Rows, rows)
	}
	return l
}

// fak-test:justify why=contract when=changed:internal/model/**
// fak-test:runtime fast est=5ms lane=default
func TestV41EngramDerivedLayoutRefuses(t *testing.T) {
	published := []int{1, 14}
	if err := ValidateV41PublishedEngramLayout(published, v41DerivedEngramLayout(t)); err != nil {
		t.Fatalf("derived layout refused: %v", err)
	}
	limit := 2 * ((uint64(math.MaxInt64) / 99092) / 2)
	edge := v41DerivedEngramLayout(t)
	edge.Multipliers[0][3] = limit - 1
	if err := ValidateV41PublishedEngramLayout(published, edge); err != nil {
		t.Fatalf("largest generated multiplier %d refused: %v", limit-1, err)
	}
	for _, tc := range []struct {
		name string
		edit func(*V41EngramLayout)
	}{
		{"head-major transposition", func(l *V41EngramLayout) {
			p := append([]uint32(nil), l.Primes[0]...)
			for ng := 0; ng < 3; ng++ {
				for h := 0; h < 8; h++ {
					l.Primes[0][h*3+ng] = p[ng*8+h]
				}
			}
		}},
		{"swap within an ngram", func(l *V41EngramLayout) { l.Primes[0][1], l.Primes[0][2] = l.Primes[0][2], l.Primes[0][1] }},
		{"swap across layers", func(l *V41EngramLayout) { l.Primes[0][0], l.Primes[1][0] = l.Primes[1][0], l.Primes[0][0] }},
		{"pad id not token_map[2]", func(l *V41EngramLayout) { l.PadID = 3 }},
		{"token map shorter than the raw pad", func(l *V41EngramLayout) { l.TokenMap = l.TokenMap[:2] }},
		{"even multiplier", func(l *V41EngramLayout) { l.Multipliers[1][0] = 10 }},
		{"odd multiplier past the generation limit", func(l *V41EngramLayout) { l.Multipliers[0][0] = limit + 1 }},
	} {
		l := v41DerivedEngramLayout(t)
		tc.edit(&l)
		if err := ValidateV41PublishedEngramLayout(published, l); !errors.Is(err, ErrV41EngramDerivation) {
			t.Errorf("%s: err = %v, want ErrV41EngramDerivation", tc.name, err)
		}
	}

	other := v41DerivedEngramLayout(t)
	other.Primes[0][0], other.PadID = 3, 5
	if err := ValidateV41PublishedEngramLayout([]int{1}, other); err != nil {
		t.Fatalf("non-published geometry checked: %v", err)
	}
}
