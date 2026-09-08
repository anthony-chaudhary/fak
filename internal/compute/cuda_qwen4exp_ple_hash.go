package compute

import (
	"fmt"
	"math"
)

// Qwen4ExpPLEHashSpec is the immutable geometry and per-head metadata for one
// Qwen3.8-Flash-Next predictive-language-embedding (PLE) hash layer.
//
// The arithmetic contract is adapted from FreeToken's Torch oracle at
// python/freetoken/models/qwen4_exp/ple.py:428-478 in
// af71ba43206e124f5ff6419b47ee36c6e9981078 (Apache-2.0). The fused CUDA
// realization is separately attributed at its implementation site.
type Qwen4ExpPLEHashSpec struct {
	NGramSize      int
	HeadsPerNGram  int
	BoundaryToken  int64
	Multipliers    []int64
	HeadVocabSizes []int64
	HeadOffsets    []int64
}

// Qwen4ExpPLEHashBatch is one ragged forward. InputIDs concatenates requests
// in SequenceLengths order. NGramContext is row-major
// [len(SequenceLengths), NGramSize-1] and contains the tokens immediately
// before each request's first input token, oldest first.
type Qwen4ExpPLEHashBatch struct {
	InputIDs        []int64
	SequenceLengths []int
	NGramContext    []int64
}

// Qwen4ExpPLEHashDispatchReceipt describes only the prepared forward-dispatch
// interval. Uploads happen in PrepareQwen4ExpPLEHash and the result fence/read
// happens in ReadRows, so a qualifying receipt has one launch and zero bytes in
// both transfer directions.
type Qwen4ExpPLEHashDispatchReceipt struct {
	KernelLaunches    uint64
	HostToDeviceBytes uint64
	DeviceToHostBytes uint64
}

// Qwen4ExpPLEHashPlan is a prepared device-resident PLE hash operation. A plan
// may be dispatched repeatedly with stable pointers (including under a graph
// capture owned by a higher layer). Close must be called after the final use.
type Qwen4ExpPLEHashPlan interface {
	Dispatch() (Qwen4ExpPLEHashDispatchReceipt, error)
	ReadRows() ([]int64, error)
	Close() error
}

// Qwen4ExpPLEHashPreparer is the optional CUDA capability seam. Preparation
// validates geometry, builds the ragged prefix index, and uploads all inputs;
// Dispatch itself performs no allocation, upload, download, or host fence.
type Qwen4ExpPLEHashPreparer interface {
	PrepareQwen4ExpPLEHash(Qwen4ExpPLEHashBatch, Qwen4ExpPLEHashSpec) (Qwen4ExpPLEHashPlan, error)
}

// Qwen4ExpPLEHashError is a fail-closed validation or device-operation error.
type Qwen4ExpPLEHashError struct {
	Stage  string
	Reason string
}

func (e *Qwen4ExpPLEHashError) Error() string {
	return fmt.Sprintf("qwen4exp PLE hash %s: %s", e.Stage, e.Reason)
}

type qwen4ExpPLEHashGeometry struct {
	tokens     int
	requests   int
	contextLen int
	heads      int
	cuSeqLens  []int32
}

func validateQwen4ExpPLEHash(batch Qwen4ExpPLEHashBatch, spec Qwen4ExpPLEHashSpec) (qwen4ExpPLEHashGeometry, error) {
	fail := func(reason string) (qwen4ExpPLEHashGeometry, error) {
		return qwen4ExpPLEHashGeometry{}, &Qwen4ExpPLEHashError{Stage: "geometry", Reason: reason}
	}
	if spec.NGramSize < 2 {
		return fail("ngram size must be at least 2")
	}
	if len(spec.Multipliers) != spec.NGramSize {
		return fail(fmt.Sprintf("%d multipliers for ngram size %d", len(spec.Multipliers), spec.NGramSize))
	}
	if spec.HeadsPerNGram <= 0 {
		return fail("heads per ngram must be positive")
	}
	orders := spec.NGramSize - 1
	if spec.HeadsPerNGram > math.MaxInt/orders {
		return fail("head count overflows int")
	}
	heads := spec.HeadsPerNGram * orders
	if heads > 1024 {
		return fail(fmt.Sprintf("%d heads exceed the CUDA block limit 1024", heads))
	}
	if len(spec.HeadVocabSizes) != heads || len(spec.HeadOffsets) != heads {
		return fail(fmt.Sprintf("head metadata lengths vocab=%d offsets=%d, want %d", len(spec.HeadVocabSizes), len(spec.HeadOffsets), heads))
	}
	for head, vocab := range spec.HeadVocabSizes {
		if vocab <= 0 {
			return fail(fmt.Sprintf("head %d vocab size must be positive", head))
		}
		offset := spec.HeadOffsets[head]
		if offset < 0 {
			return fail(fmt.Sprintf("head %d offset %d is negative", head, offset))
		}
		if offset > math.MaxInt64-(vocab-1) {
			return fail(fmt.Sprintf("head %d row range overflows int64", head))
		}
	}

	tokens := len(batch.InputIDs)
	if tokens == 0 {
		if len(batch.SequenceLengths) != 0 || len(batch.NGramContext) != 0 {
			return fail("empty input requires empty sequence lengths and context")
		}
		return qwen4ExpPLEHashGeometry{contextLen: orders, heads: heads, cuSeqLens: []int32{0}}, nil
	}
	if tokens > math.MaxInt32 {
		return fail("token count overflows the CUDA int32 ABI")
	}
	if len(batch.SequenceLengths) == 0 {
		return fail("non-empty input requires at least one request")
	}
	if len(batch.SequenceLengths) > math.MaxInt32 {
		return fail("request count overflows the CUDA int32 ABI")
	}
	cu := make([]int32, len(batch.SequenceLengths)+1)
	total := 0
	for request, length := range batch.SequenceLengths {
		if length <= 0 {
			return fail(fmt.Sprintf("request %d length must be positive", request))
		}
		if total > math.MaxInt32-length {
			return fail("ragged prefix sum overflows the CUDA int32 ABI")
		}
		total += length
		cu[request+1] = int32(total)
	}
	if total != tokens {
		return fail(fmt.Sprintf("sequence lengths sum to %d, input has %d tokens", total, tokens))
	}
	if len(batch.SequenceLengths) > math.MaxInt/orders {
		return fail("context element count overflows int")
	}
	wantContext := len(batch.SequenceLengths) * orders
	if len(batch.NGramContext) != wantContext {
		return fail(fmt.Sprintf("context has %d elements, want %d", len(batch.NGramContext), wantContext))
	}
	if tokens > math.MaxInt/heads {
		return fail("output element count overflows int")
	}
	return qwen4ExpPLEHashGeometry{
		tokens: tokens, requests: len(batch.SequenceLengths), contextLen: orders,
		heads: heads, cuSeqLens: cu,
	}, nil
}

// Qwen4ExpPLERowIDsReference computes the exact CPU oracle for the fused CUDA
// operator. Multiplication wraps modulo 2^64 by converting both operands to
// uint64 before multiplying, XOR operates on those same bits, and the result is
// then interpreted as signed int64 for floored positive-divisor remainder.
func Qwen4ExpPLERowIDsReference(batch Qwen4ExpPLEHashBatch, spec Qwen4ExpPLEHashSpec) ([]int64, error) {
	geometry, err := validateQwen4ExpPLEHash(batch, spec)
	if err != nil {
		return nil, err
	}
	if geometry.tokens == 0 {
		return []int64{}, nil
	}
	rows := make([]int64, geometry.tokens*geometry.heads)
	request := 0
	for token := 0; token < geometry.tokens; token++ {
		for request+1 < len(geometry.cuSeqLens) && token >= int(geometry.cuSeqLens[request+1]) {
			request++
		}
		local := token - int(geometry.cuSeqLens[request])
		for order := 2; order <= spec.NGramSize; order++ {
			mixed := uint64(batch.InputIDs[token]) * uint64(spec.Multipliers[0])
			valid := true
			for shift := 1; shift < order; shift++ {
				column := geometry.contextLen + local - shift
				raw := spec.BoundaryToken
				switch {
				case column >= geometry.contextLen:
					// column >= contextLen is equivalent to local >= shift,
					// so token-shift remains inside this request.
					raw = batch.InputIDs[token-shift]
				case column >= 0:
					raw = batch.NGramContext[request*geometry.contextLen+column]
				}
				valid = valid && column >= 0 && raw != spec.BoundaryToken
				if !valid {
					raw = spec.BoundaryToken
				}
				mixed ^= uint64(raw) * uint64(spec.Multipliers[shift])
			}
			start := (order - 2) * spec.HeadsPerNGram
			for head := start; head < start+spec.HeadsPerNGram; head++ {
				vocab := spec.HeadVocabSizes[head]
				rem := int64(mixed) % vocab
				if rem < 0 {
					rem += vocab
				}
				rows[token*geometry.heads+head] = rem + spec.HeadOffsets[head]
			}
		}
	}
	return rows, nil
}
