package compute

import (
	"math"
	"math/rand"
	"slices"
	"strings"
	"testing"
)

func TestQwen4ExpPLEFusedHashReference(t *testing.T) {
	spec := Qwen4ExpPLEHashSpec{
		NGramSize: 3, HeadsPerNGram: 3, BoundaryToken: 99,
		Multipliers:    []int64{math.MaxInt64 - 12, -0x61c8864680b583eb, 0x6a09e667f3bcc909},
		HeadVocabSizes: []int64{101, 103, 107, 109, 113, 127},
		HeadOffsets:    []int64{0, 101, 204, 311, 420, 533},
	}
	batch := Qwen4ExpPLEHashBatch{
		InputIDs:        []int64{7, 11, 99, 13, -5, 17, 19},
		SequenceLengths: []int{1, 4, 2},
		NGramContext: []int64{
			99, 99, // fresh one-token request
			3, 5, // two real predecessor tokens
			23, 99, // boundary in context blocks the older tap
		},
	}
	want := packedQwen4ExpPLERowIDsOracle(t, batch, spec)
	got, err := Qwen4ExpPLERowIDsReference(batch, spec)
	if err != nil {
		t.Fatal(err)
	}
	if !slices.Equal(got, want) {
		t.Fatalf("row ids differ\n got %v\nwant %v", got, want)
	}
}

func TestQwen4ExpPLEFusedHashRandomPackedParity(t *testing.T) {
	rng := rand.New(rand.NewSource(12079))
	for iteration := 0; iteration < 250; iteration++ {
		ngram := 2 + rng.Intn(4)
		headsPer := 1 + rng.Intn(4)
		heads := headsPer * (ngram - 1)
		requests := 1 + rng.Intn(5)
		lengths := make([]int, requests)
		tokens := 0
		for i := range lengths {
			lengths[i] = 1 + rng.Intn(7)
			tokens += lengths[i]
		}
		eos := int64(17)
		ids := make([]int64, tokens)
		for i := range ids {
			ids[i] = int64(rng.Intn(31) - 7)
			if rng.Intn(9) == 0 {
				ids[i] = eos
			}
		}
		context := make([]int64, requests*(ngram-1))
		for i := range context {
			context[i] = int64(rng.Intn(31) - 7)
			if rng.Intn(5) == 0 {
				context[i] = eos
			}
		}
		multipliers := make([]int64, ngram)
		for i := range multipliers {
			multipliers[i] = int64(rng.Uint64())
		}
		vocabs := make([]int64, heads)
		offsets := make([]int64, heads)
		var offset int64
		for i := range vocabs {
			vocabs[i] = int64(3 + rng.Intn(1009))
			offsets[i] = offset
			offset += vocabs[i]
		}
		spec := Qwen4ExpPLEHashSpec{
			NGramSize: ngram, HeadsPerNGram: headsPer, BoundaryToken: eos,
			Multipliers: multipliers, HeadVocabSizes: vocabs, HeadOffsets: offsets,
		}
		batch := Qwen4ExpPLEHashBatch{InputIDs: ids, SequenceLengths: lengths, NGramContext: context}
		got, err := Qwen4ExpPLERowIDsReference(batch, spec)
		if err != nil {
			t.Fatalf("iteration %d: %v", iteration, err)
		}
		want := packedQwen4ExpPLERowIDsOracle(t, batch, spec)
		if !slices.Equal(got, want) {
			t.Fatalf("iteration %d: row ids differ\n got %v\nwant %v", iteration, got, want)
		}
	}
}

func TestQwen4ExpPLEFusedHashRefusesInvalidGeometry(t *testing.T) {
	validSpec := Qwen4ExpPLEHashSpec{
		NGramSize: 2, HeadsPerNGram: 1, BoundaryToken: 0,
		Multipliers: []int64{3, 5}, HeadVocabSizes: []int64{7}, HeadOffsets: []int64{0},
	}
	validBatch := Qwen4ExpPLEHashBatch{InputIDs: []int64{1}, SequenceLengths: []int{1}, NGramContext: []int64{0}}
	tests := []struct {
		name string
		edit func(*Qwen4ExpPLEHashBatch, *Qwen4ExpPLEHashSpec)
		want string
	}{
		{"ngram", func(_ *Qwen4ExpPLEHashBatch, s *Qwen4ExpPLEHashSpec) { s.NGramSize = 1 }, "ngram size"},
		{"metadata", func(_ *Qwen4ExpPLEHashBatch, s *Qwen4ExpPLEHashSpec) { s.HeadOffsets = nil }, "metadata lengths"},
		{"vocab", func(_ *Qwen4ExpPLEHashBatch, s *Qwen4ExpPLEHashSpec) { s.HeadVocabSizes[0] = 0 }, "vocab size"},
		{"negative offset", func(_ *Qwen4ExpPLEHashBatch, s *Qwen4ExpPLEHashSpec) { s.HeadOffsets[0] = -1 }, "negative"},
		{"row overflow", func(_ *Qwen4ExpPLEHashBatch, s *Qwen4ExpPLEHashSpec) {
			s.HeadOffsets[0] = math.MaxInt64 - 5
			s.HeadVocabSizes[0] = 7
		}, "overflows"},
		{"length sum", func(b *Qwen4ExpPLEHashBatch, _ *Qwen4ExpPLEHashSpec) { b.SequenceLengths[0] = 2 }, "sum"},
		{"context", func(b *Qwen4ExpPLEHashBatch, _ *Qwen4ExpPLEHashSpec) { b.NGramContext = nil }, "context"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			batch := Qwen4ExpPLEHashBatch{
				InputIDs:        append([]int64(nil), validBatch.InputIDs...),
				SequenceLengths: append([]int(nil), validBatch.SequenceLengths...),
				NGramContext:    append([]int64(nil), validBatch.NGramContext...),
			}
			spec := Qwen4ExpPLEHashSpec{
				NGramSize: validSpec.NGramSize, HeadsPerNGram: validSpec.HeadsPerNGram,
				BoundaryToken:  validSpec.BoundaryToken,
				Multipliers:    append([]int64(nil), validSpec.Multipliers...),
				HeadVocabSizes: append([]int64(nil), validSpec.HeadVocabSizes...),
				HeadOffsets:    append([]int64(nil), validSpec.HeadOffsets...),
			}
			tt.edit(&batch, &spec)
			_, err := Qwen4ExpPLERowIDsReference(batch, spec)
			if err == nil || !strings.Contains(err.Error(), tt.want) {
				t.Fatalf("error = %v, want substring %q", err, tt.want)
			}
		})
	}
}

func TestQwen4ExpPLEFusedHashEmptyBatchIsAValidNoOp(t *testing.T) {
	spec := Qwen4ExpPLEHashSpec{
		NGramSize: 3, HeadsPerNGram: 1, BoundaryToken: 0,
		Multipliers: []int64{1, 3, 5}, HeadVocabSizes: []int64{7, 11}, HeadOffsets: []int64{0, 7},
	}
	got, err := Qwen4ExpPLERowIDsReference(Qwen4ExpPLEHashBatch{}, spec)
	if err != nil {
		t.Fatal(err)
	}
	if got == nil || len(got) != 0 {
		t.Fatalf("empty result = %#v, want non-nil empty", got)
	}
}

// packedQwen4ExpPLERowIDsOracle intentionally materializes the ragged windows
// and scans the complete crossed interval. It is structurally independent of
// the incremental virtual-window implementation under test.
func packedQwen4ExpPLERowIDsOracle(t *testing.T, batch Qwen4ExpPLEHashBatch, spec Qwen4ExpPLEHashSpec) []int64 {
	t.Helper()
	heads := spec.HeadsPerNGram * (spec.NGramSize - 1)
	rows := make([]int64, len(batch.InputIDs)*heads)
	base := 0
	for request, length := range batch.SequenceLengths {
		packed := make([]int64, spec.NGramSize-1+length)
		copy(packed, batch.NGramContext[request*(spec.NGramSize-1):(request+1)*(spec.NGramSize-1)])
		copy(packed[spec.NGramSize-1:], batch.InputIDs[base:base+length])
		for local := 0; local < length; local++ {
			token := base + local
			for order := 2; order <= spec.NGramSize; order++ {
				mixed := uint64(packed[spec.NGramSize-1+local]) * uint64(spec.Multipliers[0])
				for shift := 1; shift < order; shift++ {
					position := spec.NGramSize - 1 + local - shift
					valid := position >= 0
					for crossed := position; valid && crossed < spec.NGramSize-1+local; crossed++ {
						valid = packed[crossed] != spec.BoundaryToken
					}
					value := spec.BoundaryToken
					if valid {
						value = packed[position]
					}
					mixed ^= uint64(value) * uint64(spec.Multipliers[shift])
				}
				start := (order - 2) * spec.HeadsPerNGram
				for head := start; head < start+spec.HeadsPerNGram; head++ {
					rem := int64(mixed) % spec.HeadVocabSizes[head]
					if rem < 0 {
						rem += spec.HeadVocabSizes[head]
					}
					rows[token*heads+head] = rem + spec.HeadOffsets[head]
				}
			}
		}
		base += length
	}
	return rows
}
