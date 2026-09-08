package model

import (
	"math"
	"strings"
	"testing"
)

func TestLMHeadSampledRowFloor(t *testing.T) {
	const promptLen, batch = 4, 2
	cfg := Config{
		HiddenSize: 32, NumLayers: 2, NumHeads: 4, NumKVHeads: 2, HeadDim: 8,
		IntermediateSize: 64, VocabSize: 97, RMSNormEps: 1e-5, RopeTheta: 10000,
		TieWordEmbeddings: true, EOSTokenID: -1,
	}
	m := NewSynthetic(cfg)
	prompts := [][]int{{2, 5, 11, 23}, {3, 7, 13, 29}}

	// Forward is the explicit evaluation API: it intentionally projects every token row.
	fullEval := m.Forward(prompts[0])
	if got, want := len(fullEval.Logits), promptLen; got != want {
		t.Fatalf("Forward logits rows = %d, want full sequence %d", got, want)
	}
	for row := range fullEval.Logits {
		if got, want := len(fullEval.Logits[row]), cfg.VocabSize; got != want {
			t.Fatalf("Forward logits row %d width = %d, want vocab %d", row, got, want)
		}
	}

	// PrefillEach is the autoregressive API: for B prompts of P tokens it may expose only
	// B sampled terminal rows, never the B*P full-sequence panel.
	sampledPrefill := m.NewBatchSession(batch).PrefillEach(prompts)
	if got, want := len(sampledPrefill), batch; got != want {
		t.Fatalf("PrefillEach logits rows = %d, want sampled batch %d", got, want)
	}
	for row := range sampledPrefill {
		if got, want := len(sampledPrefill[row]), cfg.VocabSize; got != want {
			t.Fatalf("PrefillEach logits row %d width = %d, want vocab %d", row, got, want)
		}
	}

	const hiddenSize, vocab = 3, 5
	hidden := []float32{
		1, 2, 3,
		4, 5, 6,
		7, 8, 9,
		10, 11, 12,
	}
	head := []float32{
		1, 0, 0,
		0, 1, 0,
		0, 0, 1,
		1, 1, 0,
		0, 1, 1,
	}

	// Preserve the explicit evaluation path: opting into full-sequence mode reproduces
	// the historical [sequence,vocab] projection, including the original hidden backing.
	fullRows, err := selectLMHeadProjectionRows(nil, hidden, hiddenSize, nil, lmHeadProjectFullSequence)
	if err != nil {
		t.Fatal(err)
	}
	if &fullRows[0] != &hidden[0] {
		t.Fatal("full-sequence lm_head projection copied or sliced the input panel")
	}
	fullLogits := matMulBatch(head, fullRows, vocab, hiddenSize, len(fullRows)/hiddenSize)
	if got, want := len(fullLogits), 4*vocab; got != want {
		t.Fatalf("full-sequence logits length = %d, want %d", got, want)
	}

	// Autoregressive prefill admits only designated query rows. Their logits must remain
	// bit-identical to the corresponding rows of the full projection.
	sampledRows, err := selectLMHeadProjectionRows(nil, hidden, hiddenSize, []int{1, 3}, lmHeadProjectSampledRows)
	if err != nil {
		t.Fatal(err)
	}
	sampledLogits := matMulBatch(head, sampledRows, vocab, hiddenSize, len(sampledRows)/hiddenSize)
	if got, want := len(sampledLogits), 2*vocab; got != want {
		t.Fatalf("sampled logits length = %d, want %d", got, want)
	}
	for sampled, full := range []int{1, 3} {
		for v := 0; v < vocab; v++ {
			got := sampledLogits[sampled*vocab+v]
			want := fullLogits[full*vocab+v]
			if math.Float32bits(got) != math.Float32bits(want) {
				t.Fatalf("sampled row %d vocab %d = %v, full row %d = %v", sampled, v, got, full, want)
			}
		}
	}

	// The production failure geometry must select one row before vocabulary projection:
	// 8,192 hidden rows at a 248,320 vocabulary would be 3.79 GiB as fp16, while one
	// sampled row is below the issue's 10 MiB ceiling. This check never allocates the
	// forbidden full logits panel.
	const chunkRows, largeVocab = 8192, 248320
	largeHidden := make([]float32, chunkRows)
	oneRow, err := selectLMHeadProjectionRows(nil, largeHidden, 1, []int{chunkRows - 1}, lmHeadProjectSampledRows)
	if err != nil {
		t.Fatal(err)
	}
	if got, want := len(oneRow), 1; got != want {
		t.Fatalf("8K chunk selected hidden rows = %d, want %d", got, want)
	}
	if projectedBytes := int64(len(oneRow)) * largeVocab * 2; projectedBytes > 10<<20 {
		t.Fatalf("sampled fp16 logits allocation = %d bytes, want <= 10 MiB", projectedBytes)
	}

	for _, tc := range []struct {
		name       string
		hidden     []float32
		hiddenSize int
		rows       []int
		mode       lmHeadProjectionMode
		want       string
	}{
		{name: "missing sample plan", hidden: hidden, hiddenSize: hiddenSize, mode: lmHeadProjectSampledRows, want: "at least one row"},
		{name: "row out of range", hidden: hidden, hiddenSize: hiddenSize, rows: []int{4}, mode: lmHeadProjectSampledRows, want: "outside"},
		{name: "duplicate row", hidden: hidden, hiddenSize: hiddenSize, rows: []int{1, 1}, mode: lmHeadProjectSampledRows, want: "strictly increasing"},
		{name: "ambiguous full request", hidden: hidden, hiddenSize: hiddenSize, rows: []int{1}, mode: lmHeadProjectFullSequence, want: "cannot also specify"},
		{name: "malformed panel", hidden: hidden[:len(hidden)-1], hiddenSize: hiddenSize, mode: lmHeadProjectFullSequence, want: "not divisible"},
		{name: "unknown mode", hidden: hidden, hiddenSize: hiddenSize, mode: lmHeadProjectionMode(99), want: "unknown"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			if _, err := selectLMHeadProjectionRows(nil, tc.hidden, tc.hiddenSize, tc.rows, tc.mode); err == nil || !strings.Contains(err.Error(), tc.want) {
				t.Fatalf("error = %v, want containing %q", err, tc.want)
			}
		})
	}
}
