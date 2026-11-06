package model

// DRC-K17 leaf 1 witnesses: the batched routed-expert prefill panel seam.
//
//   - the pure chunk planner (bounded windows over an expert group's rows,
// //     trailing singleton pulled back, remainder kept on the per-row stream);
//   - the scalar oracle vs an independent token-major reference: grouping and
//     chunking reorder the READS, never a token's arithmetic, so the weighted
//     outputs must agree bit-for-bit;
//   - the bind gates: no backend / non-device / missing panel methods / a CUDA
//     build without its own qualification all leave the callback nil, exactly as
//     v41RoutedLayerQualified gates the decode layer;
//   - end-to-end bit-exactness on the real full-geometry mixed-quant slate: the
//     panel arm (a qualified fake whose batched projections emulate per-row GEMVs
//     through the SAME staged k-quant weights) reproduces the historical per-pick
//     stream bit-for-bit while dispatching three batched calls per chunk instead
//     of three GEMVs per row;
//   - protocol discipline: a decline after a handled chunk surfaces, and a
//     selected panel failure closes the session typed, never replays on the host.
//
// [SW-VERIFIED]: every backend here forwards to cpu-ref (compute.Default()); this
// is a deterministic software contract, NOT a physical GPU qualification. No
// [HW-WITNESSED] criterion is claimed.

import (
	"errors"
	"fmt"
	"math"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/compute"
	"github.com/anthony-chaudhary/fak/internal/model/ffn"
)

// v41PanelRowsOf returns n placeholder rows in stable order for planner tests.
func v41PanelRowsOf(n int) []v41ExpertRow {
	rows := make([]v41ExpertRow, n)
	for i := range rows {
		rows[i] = v41ExpertRow{Token: i / 2, Slot: i % 2}
	}
	return rows
}

// fak-test:runtime fast est=10ms lane=default
func TestV41PrefillExpertPanelChunkPlan(t *testing.T) {
	cases := []struct {
		name    string
		n       int
		maxRows int
		sizes   []int // nil => no chunks planned
	}{
		{"empty group", 0, 64, nil},
		{"singleton group stays per-row", 1, 64, nil},
		{"pair at the floor", 2, 64, []int{2}},
		{"trailing singleton left behind", 3, 2, []int{2}},
		{"even split", 4, 2, []int{2, 2}},
		{"odd tail left behind", 5, 2, []int{2, 2}},
		{"window then pair", 5, 3, []int{3, 2}},
		{"three windows", 7, 3, []int{3, 2, 2}},
		{"pull back a full window", 9, 8, []int{7, 2}},
		{"just under the ceiling", 63, 64, []int{63}},
		{"exactly the ceiling", 64, 64, []int{64}},
		{"one over the ceiling", 65, 64, []int{63, 2}},
		{"two full windows", 128, 64, []int{64, 64}},
		{"windows plus pair", 130, 64, []int{64, 64, 2}},
		{"degenerate bound plans nothing", 4, 1, nil},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			all := v41PanelRowsOf(tc.n)
			got := v41PlanPrefillExpertPanelChunks(all, tc.maxRows)
			if tc.sizes == nil {
				if got != nil {
					t.Fatalf("planned %d chunks, want none", len(got))
				}
				return
			}
			if len(got) != len(tc.sizes) {
				t.Fatalf("planned sizes %v, want %v", v41PanelSizes(got), tc.sizes)
			}
			seen := 0
			for i, chunk := range got {
				if len(chunk) != tc.sizes[i] {
					t.Fatalf("chunk %d holds %d rows, want %d (sizes %v)", i, len(chunk), tc.sizes[i], v41PanelSizes(got))
				}
				if len(chunk) < 2 || len(chunk) > tc.maxRows {
					t.Fatalf("chunk %d holds %d rows outside the 2..%d bound", i, len(chunk), tc.maxRows)
				}
				for j, r := range chunk {
					if r != all[seen+j] {
						t.Fatalf("chunk %d row %d = %+v, want %+v (original order retained)", i, j, r, all[seen+j])
					}
				}
				seen += len(chunk)
			}
			if leftover := tc.n - seen; leftover < 0 || leftover > 1 {
				t.Fatalf("%d rows left behind, want at most one trailing singleton", leftover)
			}
		})
	}
	// The single-step helper reports done exactly when fewer than two rows remain,
	// and the pull-back guard keeps a two-row window intact (no one-row chunk).
	rows := v41PanelRowsOf(5)
	if last, done := v41NextPrefillExpertPanelChunk(rows, 4, 64); !done || last != 5 {
		t.Fatalf("remaining=1 reported (%d,%v), want (5,true)", last, done)
	}
	if last, done := v41NextPrefillExpertPanelChunk(rows, 5, 64); !done || last != 5 {
		t.Fatalf("remaining=0 reported (%d,%v), want (5,true)", last, done)
	}
	if last, done := v41NextPrefillExpertPanelChunk(v41PanelRowsOf(3), 0, 2); done || last != 2 {
		t.Fatalf("two-row window reported (%d,%v), want (2,false)", last, done)
	}
}

func v41PanelSizes(chunks [][]v41ExpertRow) []int {
	out := make([]int, len(chunks))
	for i, c := range chunks {
		out[i] = len(c)
	}
	return out
}

// v41PanelOracleValue is a deterministic matrix element with enough spread to
// cross a positive SwiGLU clamp limit.
func v41PanelOracleValue(expert, o, i, salt int) float32 {
	return float32((expert*31+o*7+i*3+salt*13)%23-11) / 4
}

// fak-test:runtime fast est=100ms lane=default
func TestV41PrefillExpertPanelOracleMatchesTokenMajor(t *testing.T) {
	const H, I = 8, 8
	baseCfg := Config{HiddenSize: H, MoEIntermediateSize: I}
	triple := func(expert int) ([]float32, []float32, []float32, error) {
		w1 := make([]float32, I*H)
		w3 := make([]float32, I*H)
		w2 := make([]float32, H*I)
		for o := 0; o < I; o++ {
			for i := 0; i < H; i++ {
				w1[o*H+i] = v41PanelOracleValue(expert, o, i, 1)
				w3[o*H+i] = v41PanelOracleValue(expert, o, i, 2)
				w2[o*I+i] = v41PanelOracleValue(expert, o, i, 3)
			}
		}
		return w1, w3, w2, nil
	}
	// Token-major reference: strict (token, slot) order and an expression written
	// independently of the grouped/oracle code, so a mis-keyed, dropped or
	// double-weighted row changes the sum observably.
	reference := func(xnByToken [][]float32, picks [][]routePick, cfg Config) [][]float32 {
		H, I := cfg.HiddenSize, cfg.MoEIntermediateSize
		out := make([][]float32, len(picks))
		for t := range out {
			out[t] = make([]float32, H)
		}
		for t, picks := range picks {
			for _, p := range picks {
				w1, w3, w2, err := triple(p.expert)
				if err != nil {
					panic(err)
				}
				gate := matRows(w1, xnByToken[t], I, H)
				up := matRows(w3, xnByToken[t], I, H)
				clampSwiGLUProjections(gate, up, float32(cfg.SwigluLimit))
				fused := make([]float32, I)
				for i := range fused {
					fused[i] = silu(gate[i]) * up[i]
				}
				y := matRows(w2, fused, H, I)
				if err := ffn.AddScaled(out[t], y, p.weight); err != nil {
					panic(err)
				}
			}
		}
		return out
	}
	xnOf := func(tokens int) [][]float32 {
		xn := make([][]float32, tokens)
		for t := range xn {
			xn[t] = make([]float32, H)
			for h := range xn[t] {
				xn[t][h] = float32((t*13+h*5)%11-5) / 8
			}
		}
		return xn
	}
	weightOf := func(t, k int) float32 { return 0.25 + 0.125*float32((t+k)%4) }
	cycling := func(tokens, experts, perToken int) [][]routePick {
		out := make([][]routePick, tokens)
		for t := 0; t < tokens; t++ {
			picks := make([]routePick, perToken)
			for k := 0; k < perToken; k++ {
				picks[k] = routePick{expert: (t*perToken + k) % experts, weight: weightOf(t, k)}
			}
			out[t] = picks
		}
		return out
	}
	cases := []struct {
		name  string
		picks [][]routePick
		limit float64
	}{
		{"cycling routes", cycling(6, 4, 3), 0},
		{"duplicate experts", [][]routePick{
			{{expert: 2, weight: 0.5}, {expert: 0, weight: 0.25}, {expert: 2, weight: 0.75}},
			{{expert: 0, weight: 1.0}},
			{{expert: 5, weight: 0.5}},
		}, 0},
		{"all groups singleton", [][]routePick{
			{{expert: 1, weight: 0.5}},
			{{expert: 2, weight: 0.5}},
			{{expert: 3, weight: 0.5}},
		}, 0},
		{"clamped projections", cycling(6, 4, 3), 2},
	}
	for _, n := range []int{63, 64, 65} {
		picks := make([][]routePick, n)
		for t := range picks {
			picks[t] = []routePick{{expert: 0, weight: weightOf(t, 0)}}
		}
		cases = append(cases, struct {
			name  string
			picks [][]routePick
			limit float64
		}{fmt.Sprintf("group of %d rows", n), picks, 0})
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			cfg := baseCfg
			cfg.SwigluLimit = tc.limit
			xn := xnOf(len(tc.picks))
			got, err := v41PrefillExpertPanelOracle(xn, tc.picks, cfg, triple)
			if err != nil {
				t.Fatalf("oracle: %v", err)
			}
			want := reference(xn, tc.picks, cfg)
			if len(got) != len(want) {
				t.Fatalf("oracle emitted %d token rows, want %d", len(got), len(want))
			}
			for tok := range want {
				for i := range want[tok] {
					if math.Float32bits(got[tok][i]) != math.Float32bits(want[tok][i]) {
						t.Fatalf("token %d component %d: grouped oracle %g != token-major reference %g",
							tok, i, got[tok][i], want[tok][i])
					}
				}
			}
		})
	}
}

// v41PrefillPanelFake is a DeviceMemory backend that implements
// compute.V41PrefillExpertPanelBackend and qualifies the panel for serving. Each
// batched projection is emulated as per-row GEMVs through the SAME staged
// k-quant weights the per-pick seam dispatches, so its results are bit-identical
// to the historical per-row stream by construction.
// One offer point chunk dispatches THREE batched projections (gate, up, down)
// in that order, so the fake groups its dispatches into triples of three to
// recover the chunk plan.
type v41PrefillPanelFake struct {
	*v41HalSeamBackend
	dispatches           int
	chunkRows            []int
	fail                 error
	declineAfterChunks   int
	declineQualification bool
}

func newV41PrefillPanelFake() *v41PrefillPanelFake {
	return &v41PrefillPanelFake{v41HalSeamBackend: &v41HalSeamBackend{Backend: compute.Default()}}
}

func (b *v41PrefillPanelFake) V41PrefillExpertPanelQualified() bool { return !b.declineQualification }

func (b *v41PrefillPanelFake) SupportsV41PrefillExpertPanel(weight compute.Tensor, rows int) bool {
	if b.declineAfterChunks > 0 && b.dispatches/3 >= b.declineAfterChunks {
		return false
	}
	_, err := compute.PlanV41PrefillExpertPanel(weight.Shape[0], weight.Shape[1], rows)
	return err == nil
}

func (b *v41PrefillPanelFake) V41PrefillExpertPanel(weight, input compute.Tensor, rows int) (compute.Tensor, error) {
	if b.fail != nil {
		return compute.Tensor{}, b.fail
	}
	out, in := weight.Shape[0], weight.Shape[1]
	data := b.Backend.Read(input)
	if len(data) != rows*in {
		return compute.Tensor{}, errors.New("model: panel fake input extent")
	}
	res := make([]float32, rows*out)
	for r := 0; r < rows; r++ {
		y := b.MatMul(weight, compute.NewF32(compute.Default(), []int{in}, data[r*in:(r+1)*in]))
		yr := b.Backend.Read(y)
		if len(yr) != out {
			return compute.Tensor{}, errors.New("model: panel fake row extent")
		}
		copy(res[r*out:(r+1)*out], yr)
	}
	b.dispatches++
	if b.dispatches%3 == 1 {
		b.chunkRows = append(b.chunkRows, rows)
	}
	return compute.NewF32(compute.Default(), []int{rows * out}, res), nil
}

// v41CudaUnqualifiedPanelBackend carries the panel methods but no qualification
// method and names itself cuda: the bind must stay nil, exactly as
// v41RoutedLayerQualified keeps the decode layer off an unqualified CUDA build.
type v41CudaUnqualifiedPanelBackend struct {
	*v41HalSeamBackend
}

func (b *v41CudaUnqualifiedPanelBackend) Name() string { return "cuda" }

func (b *v41CudaUnqualifiedPanelBackend) SupportsV41PrefillExpertPanel(compute.Tensor, int) bool {
	return true
}

func (b *v41CudaUnqualifiedPanelBackend) V41PrefillExpertPanel(compute.Tensor, compute.Tensor, int) (compute.Tensor, error) {
	return compute.Tensor{}, errors.New("unqualified cuda build must not dispatch the panel")
}

// fak-test:runtime fast est=100ms lane=default
func TestV41PrefillExpertPanelBindDeclines(t *testing.T) {
	m := v41MixedQuantExpertModel(t)
	cases := []struct {
		name    string
		backend compute.Backend
		bind    bool
	}{
		{"no backend", nil, false},
		{"non-device backend", compute.Default(), false},
		{"device backend without panel methods", &v41HalSeamBackend{Backend: compute.Default()}, false},
		{"cuda without qualifier", &v41CudaUnqualifiedPanelBackend{v41HalSeamBackend: &v41HalSeamBackend{Backend: compute.Default()}}, false},
		{"qualified fake", newV41PrefillPanelFake(), true},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			s := &Session{M: m, Backend: tc.backend, halW: map[string]compute.Tensor{}}
			if bound := s.v41ExpertPanelFunc() != nil; bound != tc.bind {
				t.Fatalf("panel callback bound=%v, want %v", bound, tc.bind)
			}
		})
	}
	// An explicit qualification decline keeps the historical stream even though
	// the backend carries the panel methods.
	declining := newV41PrefillPanelFake()
	declining.declineQualification = true
	s := &Session{M: m, Backend: declining, halW: map[string]compute.Tensor{}}
	if s.v41ExpertPanelFunc() != nil {
		t.Fatal("an explicitly unqualified backend bound the panel callback")
	}
}

// v41PanelE2EPicks builds the deterministic pick sets the end-to-end witness
// drives: eight six-row expert groups, and a ragged set with duplicate experts
// and singleton groups.
func v41PanelE2EPicks() ([][]routePick, [][]routePick) {
	cycling := make([][]routePick, 8)
	for t := 0; t < 8; t++ {
		picks := make([]routePick, 6)
		for k := 0; k < 6; k++ {
			picks[k] = routePick{expert: (t*6 + k) % 8, weight: 0.25 + 0.125*float32((t+k)%4)}
		}
		cycling[t] = picks
	}
	ragged := [][]routePick{
		{{expert: 0, weight: 0.5}, {expert: 0, weight: 0.25}, {expert: 1, weight: 0.75}},
		{{expert: 0, weight: 1.0}, {expert: 2, weight: 0.5}},
		{{expert: 3, weight: 0.5}},
		{{expert: 0, weight: 0.25}, {expert: 1, weight: 0.5}},
		{{expert: 4, weight: 0.5}},
	}
	return cycling, ragged
}

// fak-test:runtime medium est=5s lane=default
func TestV41PrefillExpertPanelEndToEndBitExact(t *testing.T) {
	setQ4KSDOTForTest(false)
	t.Cleanup(func() { setQ4KSDOTForTest(true) })
	v41PanelTestEnv(t, "FAK_V41_PREFILL_EXPERT_PANEL", "", false)

	m := v41PinnedSlateFixture(t, false)
	H := m.Cfg.HiddenSize
	cycling, ragged := v41PanelE2EPicks()
	subtests := []struct {
		name           string
		picks          [][]routePick
		wantPanels     int
		wantPanelSizes []int
	}{
		{"eight six-row groups", cycling, 8, []int{6, 6, 6, 6, 6, 6, 6, 6}},
		{"ragged groups with singletons", ragged, 2, []int{4, 2}},
	}
	for _, tc := range subtests {
		t.Run(tc.name, func(t *testing.T) {
			totalRows := 0
			for _, picks := range tc.picks {
				totalRows += len(picks)
			}
			inputs := make([][]float32, len(tc.picks))
			for token := range inputs {
				inputs[token] = make([]float32, H)
				for lane := range inputs[token] {
					inputs[token][lane] = float32((token*17+lane*3)%11-5) / 8
				}
			}
			// Per-pick arm: the historical grouped stream with the device gate/up
			// and down seams, no panel callback.
			rowBe := &v41HalSeamBackend{Backend: compute.Default()}
			rowSess := &Session{M: m, Backend: rowBe, halW: map[string]compute.Tensor{}}
			want := make([][]float32, len(tc.picks))
			if err := m.v41ContractRoutedGrouped(0, inputs, tc.picks, &v41ProjScratch{}, m.Cfg, want, rowSess.v41State()); err != nil {
				t.Fatalf("per-pick contraction: %v", err)
			}
			if rowBe.matmuls != 3*totalRows {
				t.Fatalf("per-pick MatMul count=%d, want %d (gate+up+down per row)", rowBe.matmuls, 3*totalRows)
			}
			// Panel arm: the qualified fake offers every multi-row group in bounded
			// panels; its batched projections emulate per-row GEMVs on the same
			// staged k-quant weights, so the logits must match bit-for-bit.
			fake := newV41PrefillPanelFake()
			panelSess := &Session{M: m, Backend: fake, halW: map[string]compute.Tensor{}}
			if panelSess.v41ExpertPanelFunc() == nil {
				t.Fatal("a qualified DeviceMemory session did not bind the panel callback")
			}
			got := make([][]float32, len(tc.picks))
			if err := m.v41ContractRoutedGrouped(0, inputs, tc.picks, &v41ProjScratch{}, m.Cfg, got, panelSess.v41State()); err != nil {
				t.Fatalf("panel contraction: %v", err)
			}
			if fake.dispatches != 3*tc.wantPanels {
				t.Fatalf("panel issued %d batched projections, want %d (three per chunk over %d chunks)",
					fake.dispatches, 3*tc.wantPanels, tc.wantPanels)
			}
			if !reflectIntsEqual(fake.chunkRows, tc.wantPanelSizes) {
				t.Fatalf("panel chunk sizes=%v, want %v", fake.chunkRows, tc.wantPanelSizes)
			}
			if fake.v41HalSeamBackend.matmuls != 3*totalRows {
				t.Fatalf("panel arm contracted %d rows, want %d (batched panels plus per-row singletons)",
					fake.v41HalSeamBackend.matmuls/3, totalRows)
			}
			for token := range want {
				for i := range want[token] {
					if math.Float32bits(got[token][i]) != math.Float32bits(want[token][i]) {
						t.Fatalf("token %d component %d: panel %g != per-pick %g (bit-exactness broken)",
							token, i, got[token][i], want[token][i])
					}
				}
			}
		})
	}
}

func reflectIntsEqual(a, b []int) bool {
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

// fak-test:runtime medium est=5s lane=default
func TestV41PrefillExpertPanelMidGroupDeclineSurfaces(t *testing.T) {
	setQ4KSDOTForTest(false)
	t.Cleanup(func() { setQ4KSDOTForTest(true) })
	v41PanelTestEnv(t, "FAK_V41_PREFILL_EXPERT_PANEL", "", false)

	m := v41PinnedSlateFixture(t, false)
	H := m.Cfg.HiddenSize
	// One expert with 66 rows plans two chunks ([0:64) then [64:66)), so a
	// decline after the first handled chunk is reachable.
	const rows = 66
	picks := make([][]routePick, rows)
	inputs := make([][]float32, rows)
	for t := range picks {
		picks[t] = []routePick{{expert: 0, weight: 0.25 + 0.001*float32(t)}}
		inputs[t] = make([]float32, H)
		for lane := range inputs[t] {
			inputs[t][lane] = float32((t*17+lane*3)%11-5) / 8
		}
	}
	fake := newV41PrefillPanelFake()
	fake.declineAfterChunks = 1
	s := &Session{M: m, Backend: fake, halW: map[string]compute.Tensor{}}
	got := make([][]float32, rows)
	err := m.v41ContractRoutedGrouped(0, inputs, picks, &v41ProjScratch{}, m.Cfg, got, s.v41State())
	var opErr *V41ExpertOperationError
	if !errors.As(err, &opErr) {
		t.Fatalf("mid-group decline surfaced %v, want a typed expert operation error", err)
	}
	if fake.dispatches != 3 {
		t.Fatalf("dispatched %d batched projections before the decline, want exactly one chunk (three projections)", fake.dispatches)
	}
}

// fak-test:runtime fast est=1s lane=default
func TestV41PrefillExpertPanelSelectedFailureSurfaces(t *testing.T) {
	setQ4KSDOTForTest(false)
	t.Cleanup(func() { setQ4KSDOTForTest(true) })
	v41PanelTestEnv(t, "FAK_V41_PREFILL_EXPERT_PANEL", "", false)

	m := v41PinnedSlateFixture(t, false)
	H := m.Cfg.HiddenSize
	picks := make([][]routePick, 4)
	inputs := make([][]float32, 4)
	for t := range picks {
		picks[t] = []routePick{{expert: 0, weight: 0.5}}
		inputs[t] = make([]float32, H)
		for lane := range inputs[t] {
			inputs[t][lane] = float32((t*17+lane*3)%11-5) / 8
		}
	}
	fake := newV41PrefillPanelFake()
	fake.fail = errors.New("panel kernel fault")
	s := &Session{M: m, Backend: fake, halW: map[string]compute.Tensor{}}
	got := make([][]float32, 4)
	err := m.v41ContractRoutedGrouped(0, inputs, picks, &v41ProjScratch{}, m.Cfg, got, s.v41State())
	var closed *BackendForwardOperationError
	if !errors.As(err, &closed) {
		t.Fatalf("selected panel failure surfaced %v, want a typed backend forward operation error", err)
	}
	if closed.Path != "v41-routed-expert-panel" {
		t.Fatalf("selected failure path=%q, want v41-routed-expert-panel", closed.Path)
	}
	if !s.BackendSessionClosed() {
		t.Fatal("selected panel failure did not close the backend session")
	}
	if fake.dispatches != 0 {
		t.Fatalf("failure replayed %d batched projections, want zero", fake.dispatches)
	}
}
