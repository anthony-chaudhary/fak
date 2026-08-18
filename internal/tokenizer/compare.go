package tokenizer

import (
	"fmt"
	"sort"
	"strings"
	"time"
)

const ComparisonSchema = "fak-tokenizer-comparison/1"

type ComparisonInput struct {
	Name string
	Text string
}

type ComparisonArm struct {
	Name           string        `json:"name"`
	Owner          string        `json:"owner"`
	Integration    string        `json:"integration,omitempty"`
	Inputs         int           `json:"inputs"`
	Tokens         int           `json:"tokens"`
	Duration       time.Duration `json:"duration"`
	NSPerInput     int64         `json:"ns_per_input"`
	ExactMatches   int           `json:"exact_matches"`
	RoundTripMatch int           `json:"round_trip_matches"`
	Available      bool          `json:"available"`
	Reason         string        `json:"reason,omitempty"`
}

type ComparisonReport struct {
	Schema       string          `json:"schema"`
	Corpus       int             `json:"corpus"`
	Arms         []ComparisonArm `json:"arms"`
	Complete     bool            `json:"complete"`
	Pending      []string        `json:"pending,omitempty"`
	CorpusDigest string          `json:"corpus_digest"`
}

// ComparisonCorpus is a frozen, mixed-shape corpus shared by every tokenizer
// arm. The short fixture tokenizer used in tests intentionally covers only this
// bounded alphabet; live model-tokenizer comparisons use the same schema with a
// model-specific corpus and external witness.
func ComparisonCorpus() []ComparisonInput {
	return []ComparisonInput{
		{Name: "punctuation", Text: "."},
		{Name: "special_start", Text: "<|im_start|>"},
		{Name: "special_end", Text: "<|im_end|>"},
		{Name: "special_eot", Text: "<|endoftext|>"},
	}
}

// CompareLocal measures fak's production heap-based BPE merge against a tuned
// exhaustive adjacent-pair scan on the exact same pretokenized byte symbols.
// The exhaustive arm is a correctness oracle and incumbent algorithm, not an
// external tokenizer implementation.
func CompareLocal(tok *Tokenizer, inputs []ComparisonInput) ComparisonReport {
	report := ComparisonReport{
		Schema: ComparisonSchema, Corpus: len(inputs), CorpusDigest: comparisonDigest(inputs),
		Pending: []string{"llama_cpp_process_latency", "huggingface_process_latency", "peak_rss_bytes", "total_cost"},
	}
	want := make([][]int, len(inputs))
	for i, input := range inputs {
		ids, err := encodeNaive(tok, input.Text)
		if err != nil {
			report.Arms = append(report.Arms, ComparisonArm{Name: "naive_exhaustive", Owner: "incumbent", Inputs: len(inputs), Reason: err.Error()})
			return report
		}
		want[i] = ids
	}
	for _, arm := range []struct {
		name  string
		owner string
		run   func(string) ([]int, error)
	}{
		{name: "native_heap_bpe", owner: "fak", run: tok.Encode},
		{name: "naive_exhaustive", owner: "incumbent", run: func(text string) ([]int, error) { return encodeNaive(tok, text) }},
	} {
		result := ComparisonArm{Name: arm.name, Owner: arm.owner, Inputs: len(inputs), Available: true}
		start := time.Now()
		for i, input := range inputs {
			ids, err := arm.run(input.Text)
			if err != nil {
				result.Available = false
				result.Reason = err.Error()
				break
			}
			result.Tokens += len(ids)
			if equalIDs(ids, want[i]) {
				result.ExactMatches++
			}
			decoded, err := tok.Decode(ids)
			if err == nil && decoded == input.Text {
				result.RoundTripMatch++
			}
		}
		result.Duration = time.Since(start)
		if len(inputs) > 0 {
			result.NSPerInput = result.Duration.Nanoseconds() / int64(len(inputs))
		}
		report.Arms = append(report.Arms, result)
	}
	// External parity is separately witnessed by oracle_qwen_test.go and
	// oracle_ggml_test.go. Process-level latency/resource/cost remain pending.
	report.Arms = append(report.Arms,
		ComparisonArm{Name: "llama_cpp", Owner: "external", Integration: "llama.cpp", Inputs: len(inputs), Available: false, Reason: "live executable and model tokenizer required"},
		ComparisonArm{Name: "huggingface_tokenizers", Owner: "external", Integration: "huggingface/tokenizers", Inputs: len(inputs), Available: false, Reason: "live Python/Rust tokenizer and identical model artifact required"},
	)
	return report
}

func encodeNaive(tok *Tokenizer, text string) ([]int, error) {
	if tok == nil {
		return nil, fmt.Errorf("tokenizer comparison: nil tokenizer")
	}
	if tok.mergeRank == nil {
		return nil, ErrEncodeUnsupported
	}
	split := tok.split
	if split == nil {
		split = preTokenizeByteLevel
	}
	var ids []int
	for len(text) > 0 {
		if sp, ok := tok.matchAdded(text); ok {
			ids = append(ids, sp.id)
			text = text[len(sp.content):]
			continue
		}
		nextAdded := len(text)
		for _, sp := range tok.addedByContent {
			if i := strings.Index(text, sp.content); i >= 0 && i < nextAdded {
				nextAdded = i
			}
		}
		chunk := text[:nextAdded]
		text = text[nextAdded:]
		pieces := []string{chunk}
		if !tok.metaspace {
			pieces = split(chunk)
		}
		for _, piece := range pieces {
			encoded := byteLevelEncode(piece)
			if tok.metaspace {
				encoded = metaspaceEncode(piece)
			}
			pieceIDs, err := tok.encodePieceNaive(encoded)
			if err != nil {
				return nil, err
			}
			ids = append(ids, pieceIDs...)
		}
	}
	return ids, nil
}

func (t *Tokenizer) encodePieceNaive(piece string) ([]int, error) {
	if id, ok := t.tokenToID[piece]; ok {
		return []int{id}, nil
	}
	tokens := splitRunes(piece)
	for {
		bestRank := int(^uint(0) >> 1)
		bestPos := -1
		bestMerged := ""
		for i := 0; i+1 < len(tokens); i++ {
			merged := tokens[i] + tokens[i+1]
			if rank, ok := t.mergeRank[tokenPair{left: tokens[i], right: tokens[i+1]}]; ok && rank < bestRank {
				bestRank, bestPos, bestMerged = rank, i, merged
			}
		}
		if bestPos < 0 {
			break
		}
		tokens = append(tokens[:bestPos], append([]string{bestMerged}, tokens[bestPos+2:]...)...)
	}
	ids := make([]int, 0, len(tokens))
	for _, token := range tokens {
		id, ok := t.tokenToID[token]
		if !ok {
			return nil, fmt.Errorf("tokenizer comparison: token %q absent from vocabulary", token)
		}
		ids = append(ids, id)
	}
	return ids, nil
}

func splitRunes(text string) []string {
	out := make([]string, 0, len(text))
	for _, r := range text {
		out = append(out, string(r))
	}
	return out
}

func equalIDs(a, b []int) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}

func comparisonDigest(inputs []ComparisonInput) string {
	parts := make([]string, len(inputs))
	for i, input := range inputs {
		parts[i] = input.Name + "=" + input.Text
	}
	sort.Strings(parts)
	return fmt.Sprintf("fnv1a64:%016x", fnv64(parts))
}

func fnv64(parts []string) uint64 {
	const offset = uint64(14695981039346656037)
	const prime = uint64(1099511628211)
	h := offset
	for _, part := range parts {
		for i := 0; i < len(part); i++ {
			h ^= uint64(part[i])
			h *= prime
		}
		h ^= uint64('\n')
		h *= prime
	}
	return h
}
