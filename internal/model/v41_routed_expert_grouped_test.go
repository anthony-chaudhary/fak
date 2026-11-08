package model

// v41_routed_expert_grouped_test.go — witness for the cold cohort's grouped GEMM arm
// (rank-4b): prompt-prefill rows batch into one BatchedMatMul per expert projection
// instead of per-row MatMul, consumed through the existing optional panel interfaces.
//
// [SW-VERIFIED]: the witness backend forwards to cpu-ref (compute.Default()); this is a
// deterministic software contract, NOT a physical GPU qualification. No [HW-WITNESSED]
// criterion is claimed. The control arm is the plain scalar witness with no panel
// interfaces; both arms must agree bit for bit (Float32bits) on every output row.

import (
	"bytes"
	"errors"
	"go/ast"
	"go/format"
	"go/parser"
	"go/token"
	"reflect"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/compute"
)

// v41GroupedCohortWitness adds the grouped arm's optional capabilities to the scalar
// cohort witness: counting BatchedMatMul over cpu-ref, CPU row-log gathers, the
// swiglu-chunk consumer and the down-row publication. Its deferred reader mirrors the
// native logical-verdict contract: a status with Records n contributes n consecutive
// verdicts. Capability refusal is not a fault; the flags select the degrade tiers.
type v41GroupedCohortWitness struct {
	*v41ScalarCohortWitness
	swigluChunk bool
	downRow     bool
	batched     []int
}

func newV41GroupedCohortWitness() *v41GroupedCohortWitness {
	return &v41GroupedCohortWitness{
		v41ScalarCohortWitness: &v41ScalarCohortWitness{v41RoutedLayerTestBackend: newV41RoutedLayerTestBackend(), enabled: true},
		swigluChunk:            true,
		downRow:                true,
	}
}

func (b *v41GroupedCohortWitness) BatchedMatMul(w, x compute.Tensor, p int) compute.Tensor {
	b.batched = append(b.batched, p)
	return b.v41HalSeamBackend.Backend.BatchedMatMul(w, x, p)
}

func (b *v41GroupedCohortWitness) V41RowLogAvailable() bool { return true }

func (b *v41GroupedCohortWitness) V41RowLogAlloc(rows, width int) (compute.Tensor, error) {
	if err := compute.ValidateV41RowLogAlloc(rows, width); err != nil {
		return compute.Tensor{}, err
	}
	return b.v41HalSeamBackend.Backend.Upload(compute.NewF32(compute.Default(), []int{rows, width}, make([]float32, rows*width)), compute.F32), nil
}

func (b *v41GroupedCohortWitness) V41RowLogCopy(dst compute.Tensor, dstRow int, src compute.Tensor, srcRow, rows int) error {
	if _, err := compute.ValidateV41RowLogCopy(dst, dstRow, src, srcRow, rows); err != nil {
		return err
	}
	d, s := b.v41HalSeamBackend.Backend.Read(dst), b.v41HalSeamBackend.Backend.Read(src)
	w := dst.Shape[1]
	copy(d[dstRow*w:(dstRow+rows)*w], s[srcRow*w:(srcRow+rows)*w])
	return nil
}

func (b *v41GroupedCohortWitness) V41RowLogGather(spans []compute.V41RowSpan, width int) (compute.Tensor, error) {
	total, err := compute.ValidateV41RowLogGather(spans, width)
	if err != nil {
		return compute.Tensor{}, err
	}
	out := make([]float32, total*width)
	next := 0
	for _, span := range spans {
		raw := b.v41HalSeamBackend.Backend.Read(span.Src)
		for r := 0; r < span.Rows; r++ {
			copy(out[next*width:(next+1)*width], raw[(span.Row+r)*width:(span.Row+r+1)*width])
			next++
		}
	}
	return b.v41HalSeamBackend.Backend.Upload(compute.NewF32(compute.Default(), []int{total, width}, out), compute.F32), nil
}

func (b *v41GroupedCohortWitness) SupportsV41RoutedSwiGLUPanelChunk(gate, up compute.Tensor, first, rows int) bool {
	if !b.swigluChunk || gate.Dtype != compute.F32 || gate.Layout != compute.RowMajor || gate.Quant != nil || len(gate.Shape) != 2 ||
		gate.Shape[0] != up.Shape[0] || gate.Shape[1] != up.Shape[1] {
		return false
	}
	_, err := compute.PlanV41RoutedSwiGLUPanelChunk(gate.Shape[0], gate.Shape[1], first, rows)
	return err == nil
}

// V41RoutedSwiGLUPanelChunkDeferred reproduces the per-row host activation exactly:
// BF16-rounded projections, the asymmetric limit clamp, then silu(gate)*up*weight
// rounded back to BF16. One status owns every row record.
func (b *v41GroupedCohortWitness) V41RoutedSwiGLUPanelChunkDeferred(gate, up compute.Tensor, first int, limit float32, weights []float32) (compute.Tensor, compute.V41DeferredStatus, error) {
	if !b.SupportsV41RoutedSwiGLUPanelChunk(gate, up, first, len(weights)) {
		return compute.Tensor{}, compute.V41DeferredStatus{}, errors.New("grouped witness: panel chunk geometry refused")
	}
	rows, width := len(weights), gate.Shape[1]
	gw, uw := b.v41HalSeamBackend.Backend.Read(gate), b.v41HalSeamBackend.Backend.Read(up)
	out := make([]float32, rows*width)
	var verdict error
	for r, weight := range weights {
		if fault := b.activationFaults[weight]; fault != nil {
			verdict = fault
		} else if failure := b.verdictFailure[weight]; failure != nil {
			verdict = failure
		}
		g, u := gw[(first+r)*width:(first+r+1)*width], uw[(first+r)*width:(first+r+1)*width]
		for i := range g {
			a, c := v41RoundBF16(g[i]), v41RoundBF16(u[i])
			if limit > 0 {
				a = min(a, limit)
				c = max(-limit, min(c, limit))
			}
			out[r*width+i] = v41RoundBF16(silu(a) * c * weight)
		}
	}
	t := b.v41HalSeamBackend.Backend.Upload(compute.NewF32(compute.Default(), []int{rows * width}, out), compute.F32)
	rec := b.v41HalSeamBackend.Backend.Upload(compute.NewF32(compute.Default(), []int{4 * rows}, make([]float32, 4*rows)), compute.F32)
	b.pending[rec.Buf()] = verdict
	st := compute.V41DeferredStatus{Kind: compute.V41DeferredRoutedSwiGLU, Count: width, Records: rows, Record: rec}
	return t, st, nil
}

func (b *v41GroupedCohortWitness) SupportsV41RoutedDownPanelRow(panel compute.Tensor, row int) bool {
	if !b.downRow || panel.Dtype != compute.F32 || panel.Layout != compute.RowMajor || panel.Quant != nil || len(panel.Shape) != 2 {
		return false
	}
	rows, width := panel.Shape[0], panel.Shape[1]
	return rows >= 1 && rows <= 64 && width >= 1 && row >= 0 && row < rows
}

// V41RoutedDownPanelRowDeferred publishes one panel row with the ordinary checked
// BF16 publication: finite/overflow verdict, integer rounding, RoutedDown site.
func (b *v41GroupedCohortWitness) V41RoutedDownPanelRowDeferred(panel compute.Tensor, row int) (compute.Tensor, compute.V41DeferredStatus, error) {
	if !b.SupportsV41RoutedDownPanelRow(panel, row) {
		return compute.Tensor{}, compute.V41DeferredStatus{}, errors.New("grouped witness: down panel row refused")
	}
	b.publishes++
	width := panel.Shape[1]
	raw := b.v41HalSeamBackend.Backend.Read(panel)
	values := append([]float32(nil), raw[row*width:(row+1)*width]...)
	var verdict error
	if index, bad := v41BF16PublishFault(values); bad || b.publishFaultAt[b.publishes] {
		verdict = &compute.BackendError{Backend: "test", Site: "V41RoutedDownPanelRow", Err: compute.ErrVulkanExecutionFailed,
			Recovered: &compute.V41BF16CheckedArithmeticError{Site: compute.V41BF16PublishRoutedDown, Index: index}}
	}
	for i, v := range values {
		values[i] = v41RoundBF16(v)
	}
	t := b.v41HalSeamBackend.Backend.Upload(compute.NewF32(compute.Default(), []int{len(values)}, values), compute.F32)
	st := b.record(verdict)
	st.Kind, st.Count, st.Site = compute.V41DeferredBF16Publish, len(values), compute.V41BF16PublishRoutedDown
	return t, st, nil
}

// The documented reader contract returns one verdict per record: a status with
// Records n contributes n consecutive verdicts, mirroring the native reader.
func (b *v41GroupedCohortWitness) V41ReadDeferred(st []compute.V41DeferredStatus, rows []compute.Tensor) ([]error, [][]float32, error) {
	b.fences++
	verdicts := make([]error, 0, len(st))
	for _, s := range st {
		verdict, ok := b.pending[s.Record.Buf()]
		if !ok {
			return nil, nil, errors.New("deferred status read twice or never issued")
		}
		delete(b.pending, s.Record.Buf())
		b.v41HalSeamBackend.Backend.Free(s.Record)
		for r := 0; r < compute.V41DeferredRecords(s); r++ {
			verdicts = append(verdicts, verdict)
		}
	}
	if b.readFailure != nil {
		return nil, nil, b.readFailure
	}
	values := make([][]float32, len(rows))
	for i, row := range rows {
		values[i] = append([]float32(nil), b.v41HalSeamBackend.Backend.Read(row)...)
	}
	return verdicts, values, nil
}

// The grouped arm batches every non-singleton down-schedule chunk of the cold cohort
// into one BatchedMatMul per projection, keeps every output row bit-identical to the
// per-row scalar control, and degrades tier by tier when a panel capability is absent.
// fak-test:runtime medium est=5s lane=default
func TestV41RoutedExpertGroupedGEMM(t *testing.T) {
	const H, I = 256, 256
	m := &Model{Cfg: Config{HiddenSize: H, MoEIntermediateSize: I}}
	gate := compute.NewQ4K(compute.Default(), []int{I, H}, buildRawQ4K(t, I, H, 0x13827))
	up := compute.NewQ4K(compute.Default(), []int{I, H}, buildRawQ4K(t, I, H, 0x13828))
	down := compute.NewQ4K(compute.Default(), []int{H, I}, buildRawQ4K(t, H, I, 0x13829))
	inputs := func(rows int) [][]float32 {
		out := make([][]float32, rows)
		for r := range out {
			out[r] = make([]float32, H)
			for i := range out[r] {
				out[r][i] = float32((r*31+i*7)%13-6) / 16
			}
		}
		return out
	}
	weights := func(rows int) []float32 {
		out := make([]float32, rows)
		for r := range out {
			out[r] = float32(r+1) / 8
		}
		return out
	}
	runControl := func(rows int) [][]float32 {
		b := &v41ScalarCohortWitness{v41RoutedLayerTestBackend: newV41RoutedLayerTestBackend(), enabled: true}
		s := &Session{M: m, Backend: b}
		var stage string
		var failed bool
		got, cause := s.v41RunExpertCohortRetained(b, gate, up, down, inputs(rows), weights(rows), 2, 0, &stage, &failed, nil, nil)
		if cause != nil || failed {
			t.Fatalf("control rows=%d: cause=%v failed=%t", rows, cause, failed)
		}
		if len(b.matmulInputs) != 3*rows || len(b.pending) != 0 {
			t.Fatalf("control rows=%d: matmuls=%d pending=%d, want %d/0", rows, len(b.matmulInputs), len(b.pending), 3*rows)
		}
		return got
	}
	runCandidate := func(rows int, swigluChunk, downRow bool) ([][]float32, *v41GroupedCohortWitness) {
		b := newV41GroupedCohortWitness()
		b.swigluChunk, b.downRow = swigluChunk, downRow
		s := &Session{M: m, Backend: b}
		var stage string
		var failed bool
		got, cause := s.v41RunExpertCohortRetained(b, gate, up, down, inputs(rows), weights(rows), 2, 0, &stage, &failed, nil, nil)
		if cause != nil || failed {
			t.Fatalf("candidate rows=%d chunk=%t downRow=%t: cause=%v failed=%t", rows, swigluChunk, downRow, cause, failed)
		}
		if len(b.pending) != 0 {
			t.Fatalf("candidate rows=%d leaked %d deferred statuses", rows, len(b.pending))
		}
		return got, b
	}
	check := func(name string, rows int, wantBatched []int, wantScalarMatmuls int) {
		want := runControl(rows)
		for _, tc := range []struct {
			name        string
			chunk, down bool
		}{
			{name: "swiglu_chunk_down_row", chunk: true, down: true},
			{name: "gather_activation_down_row", chunk: false, down: true},
			{name: "swiglu_chunk_gather_publication", chunk: true, down: false},
			{name: "gather_activation_gather_publication", chunk: false, down: false},
		} {
			t.Run(name+"/"+tc.name, func(t *testing.T) {
				got, b := runCandidate(rows, tc.chunk, tc.down)
				if len(got) != len(want) {
					t.Fatalf("rows=%d returned %d rows, want %d", rows, len(got), len(want))
				}
				for r := range got {
					if !v41SameBits(got[r], want[r]) {
						t.Fatalf("row %d bits differ from the scalar control", r)
					}
				}
				if !reflect.DeepEqual(b.batched, wantBatched) {
					t.Fatalf("batched extents=%v want %v", b.batched, wantBatched)
				}
				if len(b.matmulInputs) != wantScalarMatmuls {
					t.Fatalf("scalar MatMul count=%d want %d (singleton tail only)", len(b.matmulInputs), wantScalarMatmuls)
				}
			})
		}
	}
	// Chunks follow the down schedule: 8/4/2, singletons stay scalar.
	check("ten_rows", 10, []int{8, 8, 8, 2, 2, 2}, 0)
	check("fifteen_rows", 15, []int{8, 8, 8, 4, 4, 4, 2, 2, 2}, 3)
}

// v41GroupedGEMMAdmitted admits exactly the widths that stay bit-identical under
// BatchedMatMul: the pinned Q2_K token-panel tree, and Q4_K below the wide-row wave32
// threshold at or under the mm_cm1 matvec boundary. Everything else declines.
// fak-test:runtime fast est=100ms lane=default
func TestV41GroupedGEMMAdmission(t *testing.T) {
	be := compute.Default()
	q4k := compute.Tensor{Dtype: compute.Q4_K, Shape: []int{256, 256}}
	wide := compute.Tensor{Dtype: compute.Q4_K, Shape: []int{65536, 256}}
	q2k := compute.Tensor{Dtype: compute.Q2_K, Shape: []int{256, 256}}
	f32 := compute.Tensor{Dtype: compute.F32, Shape: []int{256, 256}}
	for _, tc := range []struct {
		w    compute.Tensor
		rows int
		want bool
	}{
		{q4k, 2, true}, {q4k, 4, true}, {q4k, 8, true},
		{q4k, 9, false}, {q4k, 32, false},
		{wide, 8, false},
		{q2k, 8, false}, // No token-panel backend: the pinned matvec tree is unavailable.
		{f32, 8, false},
	} {
		if got := v41GroupedGEMMAdmitted(be, tc.w, tc.rows); got != tc.want {
			t.Fatalf("admit(%v, rows=%d)=%t want %t", tc.w.Dtype, tc.rows, got, tc.want)
		}
	}
}

// This CPU source witness pins the grouped-GEMM admission predicate. It does not
// compile cgo or execute Vulkan.
// fak-test:runtime fast est=1ms lane=default
func TestV41RoutedExpertGroupedGEMMContract(t *testing.T) {
	t.Parallel()
	fset := token.NewFileSet()
	file, err := parser.ParseFile(fset, "v41_routed_expert_cohort.go", nil, 0)
	if err != nil {
		t.Fatal(err)
	}
	var found *ast.FuncDecl
	for _, decl := range file.Decls {
		if fn, ok := decl.(*ast.FuncDecl); ok && fn.Name.Name == "v41GroupedGEMMAdmitted" {
			found = fn
		}
	}
	if found == nil {
		t.Fatal("v41GroupedGEMMAdmitted missing from v41_routed_expert_cohort.go")
	}
	var got bytes.Buffer
	if err := format.Node(&got, fset, found); err != nil {
		t.Fatal(err)
	}
	// format.Node prints the declaration without its leading doc comment.
	const want = `func v41GroupedGEMMAdmitted(be compute.Backend, w compute.Tensor, rows int) bool {
	switch w.Dtype {
	case compute.Q2_K:
		op, ok := be.(compute.V41Q2KTokenPanelBackend)
		return ok && op.SupportsV41Q2KTokenPanel(w, rows)
	case compute.Q4_K:
		return len(w.Shape) == 2 && w.Shape[0] < 65536 && rows <= 8
	default:
		return false
	}
}`
	if got.String() != want {
		t.Fatalf("v41GroupedGEMMAdmitted changed its admission contract:\n%s\nwant:\n%s", &got, want)
	}
}
