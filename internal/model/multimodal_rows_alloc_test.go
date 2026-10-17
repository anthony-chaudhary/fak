package model

import (
	"fmt"
	"math"
	"reflect"
	"testing"
)

// Frozen pre-change row assembly oracle from public eecb3d5.
func legacyMultimodalRows(m *Model, req MultimodalRequest) ([][]float32, MultimodalVerdict, error) {
	policy := req.Policy.withDefaults()
	verdict := MultimodalVerdict{Decision: MultimodalAllow, Mode: policy.Mode}
	if err := policy.valid(); err != nil {
		verdict.Decision = MultimodalDeny
		verdict.Reason = err.Error()
		return nil, verdict, fmt.Errorf("%w: %s", ErrMultimodalDenied, verdict.Reason)
	}
	if len(req.Parts) == 0 {
		verdict.Decision = MultimodalDeny
		verdict.Reason = "empty prompt"
		return nil, verdict, fmt.Errorf("%w: %s", ErrMultimodalDenied, verdict.Reason)
	}

	H := m.Cfg.HiddenSize
	if H <= 0 {
		verdict.Decision = MultimodalDeny
		verdict.Reason = "hidden size is unset"
		return nil, verdict, fmt.Errorf("%w: %s", ErrMultimodalDenied, verdict.Reason)
	}
	embed := m.embedRows()
	rows := make([][]float32, 0, len(req.Parts))
	for i, part := range req.Parts {
		hasText := len(part.TokenIDs) > 0
		hasImage := part.Image != nil
		if hasText == hasImage {
			verdict.Decision = MultimodalDeny
			verdict.Reason = fmt.Sprintf("part %d must contain exactly one of token ids or image embedding", i)
			return nil, verdict, fmt.Errorf("%w: %s", ErrMultimodalDenied, verdict.Reason)
		}
		if hasText {
			for _, id := range part.TokenIDs {
				if id < 0 || id >= m.Cfg.VocabSize {
					verdict.Decision = MultimodalDeny
					verdict.Reason = fmt.Sprintf("token id %d outside vocab size %d", id, m.Cfg.VocabSize)
					return nil, verdict, fmt.Errorf("%w: %s", ErrMultimodalDenied, verdict.Reason)
				}
				row := append([]float32(nil), embed[id*H:(id+1)*H]...)
				scaleEmbedInPlace(row, m.Cfg)
				rows = append(rows, row)
			}
			continue
		}
		if err := admitVisionEmbedding(part.Image, policy, &verdict, H); err != nil {
			return nil, verdict, err
		}
		for _, vec := range part.Image.Vectors {
			rows = append(rows, append([]float32(nil), vec...))
		}
	}
	if len(rows) == 0 {
		verdict.Decision = MultimodalDeny
		verdict.Reason = "empty prompt"
		return nil, verdict, fmt.Errorf("%w: %s", ErrMultimodalDenied, verdict.Reason)
	}
	if verdict.Images > 0 && policy.Mode != MultimodalModeQuarantine {
		verdict.Decision = MultimodalQuarantine
		verdict.Reason = "multimodal input requires explicit quarantine opt-in"
		verdict.QuarantineID = multimodalQuarantineID(req.Parts)
		return nil, verdict, fmt.Errorf("%w: %s", ErrMultimodalQuarantined, verdict.Reason)
	}
	return rows, verdict, nil
}

func checkMultimodalRowBits(t *testing.T, got, want [][]float32) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("row count %d != %d", len(got), len(want))
	}
	for i := range got {
		if len(got[i]) != len(want[i]) {
			t.Fatalf("row %d width mismatch", i)
		}
		for j, x := range got[i] {
			if math.Float32bits(x) != math.Float32bits(want[i][j]) {
				t.Fatalf("row %d col %d bits differ", i, j)
			}
		}
	}
}

func multimodalAllocationRequest(m *Model, n int) MultimodalRequest {
	return MultimodalRequest{Policy: MultimodalPolicy{Mode: MultimodalModeQuarantine, MaxEmbeddingTokens: 4096}, Parts: []MultimodalPart{
		{TokenIDs: []int{1, 2, 3}}, {Image: validVisionEmbedding(m, n)}, {TokenIDs: []int{4, 5}},
	}}
}

// fak-test:runtime fast est=100ms
func TestMultimodalRowSlabParityAndValidation(t *testing.T) {
	m := multimodalTestModel()
	valid := multimodalAllocationRequest(m, 4)
	cases := []struct {
		name string
		req  MultimodalRequest
	}{
		{"mixed", valid},
		{"text", MultimodalRequest{Parts: []MultimodalPart{{TokenIDs: []int{1, 2}}, {TokenIDs: []int{3}}}}},
		{"empty", MultimodalRequest{}},
		{"empty-part", MultimodalRequest{Parts: []MultimodalPart{{}}}},
		{"invalid-policy", MultimodalRequest{Policy: MultimodalPolicy{Mode: "invalid"}, Parts: valid.Parts}},
		{"quarantined", MultimodalRequest{Parts: valid.Parts}},
		{"token-before-image-error", MultimodalRequest{Parts: []MultimodalPart{{TokenIDs: []int{-1}}, {Image: &VisionEmbedding{}}}}},
		{"image-before-token-error", MultimodalRequest{Parts: []MultimodalPart{{Image: &VisionEmbedding{}}, {TokenIDs: []int{-1}}}}},
		{"valid-image-before-token-error", MultimodalRequest{Parts: []MultimodalPart{{Image: validVisionEmbedding(m, 1)}, {TokenIDs: []int{16}}}}},
		{"ambiguous-part", MultimodalRequest{Parts: []MultimodalPart{{TokenIDs: []int{1}, Image: validVisionEmbedding(m, 1)}}}},
		{"empty-image", MultimodalRequest{Parts: []MultimodalPart{{Image: validVisionEmbedding(m, 0)}}}},
	}
	short := validVisionEmbedding(m, 1)
	short.Vectors[0] = short.Vectors[0][:1]
	cases = append(cases, struct {
		name string
		req  MultimodalRequest
	}{"invalid-width", MultimodalRequest{Parts: []MultimodalPart{{Image: short}}}})
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			want, wv, we := legacyMultimodalRows(m, tc.req)
			got, gv, ge := m.prepareMultimodalRows(tc.req)
			if !reflect.DeepEqual(gv, wv) || fmt.Sprint(ge) != fmt.Sprint(we) {
				t.Fatalf("verdict/errors changed: got %+v %v want %+v %v", gv, ge, wv, we)
			}
			if (got == nil) != (want == nil) {
				t.Fatal("nil row contract changed")
			}
			checkMultimodalRowBits(t, got, want)
		})
	}
	// The same rows feed the actual small-model forward path.
	before, _, err := legacyMultimodalRows(m, valid)
	if err != nil {
		t.Fatal(err)
	}
	after, _, err := m.prepareMultimodalRows(valid)
	if err != nil {
		t.Fatal(err)
	}
	assertActivationsBitsEqual(t, m.forwardHiddenRows(after), m.forwardHiddenRows(before))
	t.Run("gemma-scaled-text", func(t *testing.T) {
		m := multimodalTestModel()
		m.Cfg.ModelType = "gemma"
		m.Cfg.EmbedScale = math.Sqrt(float64(m.Cfg.HiddenSize))
		req := multimodalAllocationRequest(m, 2)
		want, wv, we := legacyMultimodalRows(m, req)
		got, gv, ge := m.prepareMultimodalRows(req)
		if we != nil || ge != nil || !reflect.DeepEqual(gv, wv) {
			t.Fatalf("scaled assembly: got %+v %v want %+v %v", gv, ge, wv, we)
		}
		checkMultimodalRowBits(t, got, want)
		raw := m.embedRows()[m.Cfg.HiddenSize : 2*m.Cfg.HiddenSize]
		differs := false
		for i, x := range got[0] {
			if math.Float32bits(x) != math.Float32bits(raw[i]) {
				differs = true
			}
		}
		if !differs {
			t.Fatal("scaled fixture did not exercise nonidentity text scaling")
		}
		checkMultimodalRowBits(t, got[3:5], req.Parts[1].Image.Vectors)
	})
}

// fak-test:runtime fast est=100ms
func TestMultimodalRowSlabOwnership(t *testing.T) {
	m := multimodalTestModel()
	req := multimodalAllocationRequest(m, 3)
	// Verify non-canonical float bits are copied, not transformed, for image rows.
	req.Parts[1].Image.Vectors[0][0] = math.Float32frombits(0x80000000)
	req.Parts[1].Image.Vectors[1][0] = math.Float32frombits(0x7fc00017)
	want, _, err := legacyMultimodalRows(m, req)
	if err != nil {
		t.Fatal(err)
	}
	rows, _, err := m.prepareMultimodalRows(req)
	if err != nil {
		t.Fatal(err)
	}
	checkMultimodalRowBits(t, rows, want)
	for i, row := range rows {
		if cap(row) != len(row) {
			t.Fatalf("row %d has unsafe append capacity", i)
		}
		grown := append(row, 12345)
		grown[0] = 54321 // append must have detached this row, too.
		checkMultimodalRowBits(t, rows, want)
	}
	for i, row := range rows {
		old := row[0]
		row[0] = 12345
		for j := range rows {
			if i != j {
				checkMultimodalRowBits(t, [][]float32{rows[j]}, [][]float32{want[j]})
			}
		}
		row[0] = old
	}
	// Mutating caller-owned image storage cannot mutate output, and vice versa.
	source := req.Parts[1].Image.Vectors[2][0]
	rows[5][0] = 999
	if req.Parts[1].Image.Vectors[2][0] != source {
		t.Fatal("output aliases image input")
	}
	rows[5][0] = source
	req.Parts[1].Image.Vectors[2][0] = 777
	if rows[5][0] != source {
		t.Fatal("input aliases output")
	}
	// Text output likewise remains detached from the model embedding table.
	embed := m.embedRows()
	original := embed[m.Cfg.HiddenSize]
	rows[0][0] = 888
	if math.Float32bits(embed[m.Cfg.HiddenSize]) != math.Float32bits(original) {
		t.Fatal("output aliases text embeddings")
	}
}

var multimodalRowsSink [][]float32

// fak-test:runtime fast est=250ms
func TestMultimodalRowSlabAllocationReduction(t *testing.T) {
	m := multimodalTestModel()
	req := multimodalAllocationRequest(m, 1024)
	before := testing.AllocsPerRun(10, func() {
		rows, _, err := legacyMultimodalRows(m, req)
		if err != nil {
			panic(err)
		}
		multimodalRowsSink = rows
	})
	after := testing.AllocsPerRun(10, func() {
		rows, _, err := m.prepareMultimodalRows(req)
		if err != nil {
			panic(err)
		}
		multimodalRowsSink = rows
	})
	if after >= before/4 {
		t.Fatalf("row allocation reduction missing: baseline %.0f candidate %.0f", before, after)
	}
	t.Logf("synthetic row assembly allocations/op: baseline=%g candidate=%g", before, after)
}

func BenchmarkMultimodalRowAssembly(b *testing.B) {
	m := multimodalTestModel()
	for _, n := range []int{1, 1024, 2048} {
		req := multimodalAllocationRequest(m, n)
		for _, arm := range []string{"legacy", "slab"} {
			b.Run(fmt.Sprintf("%s/image_rows_%d", arm, n), func(b *testing.B) {
				b.ReportAllocs()
				b.ResetTimer()
				for i := 0; i < b.N; i++ {
					var rows [][]float32
					var err error
					if arm == "legacy" {
						rows, _, err = legacyMultimodalRows(m, req)
					} else {
						rows, _, err = m.prepareMultimodalRows(req)
					}
					if err != nil {
						b.Fatal(err)
					}
					multimodalRowsSink = rows
				}
			})
		}
	}
}
