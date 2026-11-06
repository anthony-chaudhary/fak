package compute

// fak-test:runtime fast est=5s

import (
	"math"
	"math/rand"
	"testing"
)

// TestGDNMarkListOrderAndSlotSelection pins the TensorFold-adapted segment-mark contract
// (zig/src/core/deltanet_kernels.zig:9 markList/slotKernel @9e8c368b0b4c2c7892a0c9f0f302d7c164016209):
// a mark list keeps at most 4 ascending non-zero rows <= total rows, refusing empty, late,
// repeated or fifth marks; tree state selects the narrowest slot kernel holding its peak;
// and recurrent state stores at segment marks so a resumed recurrence replays
// bit-identically to the uninterrupted one.
func TestGDNMarkListOrderAndSlotSelection(t *testing.T) {
	t.Run("mark admission order", func(t *testing.T) {
		if _, err := NewGDNMarkList(0); err != ErrGDNTotalRowsInvalid {
			t.Fatalf("NewGDNMarkList(0): got %v, want ErrGDNTotalRowsInvalid", err)
		}
		if _, err := NewGDNMarkList(-7); err != ErrGDNTotalRowsInvalid {
			t.Fatalf("NewGDNMarkList(-7): got %v, want ErrGDNTotalRowsInvalid", err)
		}

		m, err := NewGDNMarkList(64)
		if err != nil {
			t.Fatalf("NewGDNMarkList(64): %v", err)
		}
		for _, row := range []int{5, 17, 33, 48} {
			if err := m.Add(row); err != nil {
				t.Fatalf("Add(%d): %v", row, err)
			}
		}
		if m.Len() != 4 || m.Last() != 48 || m.TotalRows() != 64 {
			t.Fatalf("list state: Len=%d Last=%d TotalRows=%d, want 4/48/64", m.Len(), m.Last(), m.TotalRows())
		}
		want := []int{5, 17, 33, 48}
		for i, w := range want {
			if got := m.At(i); got != w {
				t.Fatalf("At(%d)=%d, want %d", i, got, w)
			}
		}
		if !m.Contains(33) || m.Contains(34) {
			t.Fatal("Contains disagrees with admitted marks")
		}
		// The fifth mark is refused regardless of content, and the list is unchanged.
		if err := m.Add(60); err != ErrGDNMarkCapacity {
			t.Fatalf("fifth Add(60): got %v, want ErrGDNMarkCapacity", err)
		}
		if m.Len() != 4 || m.Last() != 48 {
			t.Fatalf("list changed after refused fifth mark: Len=%d Last=%d", m.Len(), m.Last())
		}
		if err := m.Validate(); err != nil {
			t.Fatalf("Validate on well-formed list: %v", err)
		}

		l, _ := NewGDNMarkList(64)
		if err := l.Add(0); err != ErrGDNMarkZero {
			t.Fatalf("Add(0): got %v, want ErrGDNMarkZero", err)
		}
		if err := l.Add(-3); err != ErrGDNMarkZero {
			t.Fatalf("Add(-3): got %v, want ErrGDNMarkZero", err)
		}
		if err := l.Add(65); err != ErrGDNMarkLate {
			t.Fatalf("Add(65): got %v, want ErrGDNMarkLate", err)
		}
		if err := l.Add(64); err != nil {
			t.Fatalf("Add(64) boundary: %v", err)
		}
		if err := l.Add(64); err != ErrGDNMarkRepeated {
			t.Fatalf("repeated Add(64): got %v, want ErrGDNMarkRepeated", err)
		}
		if err := l.Add(10); err != ErrGDNMarkOutOfOrder {
			t.Fatalf("Add(10) behind 64: got %v, want ErrGDNMarkOutOfOrder", err)
		}
		if err := l.Validate(); err != nil {
			t.Fatalf("Validate after refusals: %v", err)
		}

		// Validate must catch corruption that bypassed Add.
		c, _ := NewGDNMarkList(64)
		c.Add(5)
		c.Add(9)
		c.marks[1] = 5
		if err := c.Validate(); err != ErrGDNMarkRepeated {
			t.Fatalf("Validate corrupted repeat: got %v, want ErrGDNMarkRepeated", err)
		}
		c.marks[1] = 3
		if err := c.Validate(); err != ErrGDNMarkOutOfOrder {
			t.Fatalf("Validate corrupted order: got %v, want ErrGDNMarkOutOfOrder", err)
		}
		c.marks[0] = 0
		if err := c.Validate(); err != ErrGDNMarkZero {
			t.Fatalf("Validate corrupted zero: got %v, want ErrGDNMarkZero", err)
		}
		c.marks[0] = 65
		if err := c.Validate(); err != ErrGDNMarkLate {
			t.Fatalf("Validate corrupted late: got %v, want ErrGDNMarkLate", err)
		}
	})

	t.Run("slot selection narrowest holding", func(t *testing.T) {
		ladder := GDNDefaultSlotKernels
		for i := 1; i < len(ladder); i++ {
			if ladder[i].Bits < ladder[i-1].Bits {
				t.Fatalf("ladder is not narrow-to-wide at %d", i)
			}
		}

		cases := []struct {
			peak float32
			want string
		}{
			{0, "f8e4m3"},
			{448, "f8e4m3"}, // exactly the E4M3 maximum
			{448.5, "f16"},  // just past it
			{449, "f16"},
			{65504, "f16"}, // exactly the binary16 maximum
			{65505, "bf16"},
			{1e6, "bf16"},
			{3.38e38, "bf16"},
			{3.39e38, "f32"}, // just past the bfloat16 maximum
			{math.MaxFloat32, "f32"},
			{-449, "f16"}, // sign is irrelevant
		}
		for _, tc := range cases {
			got, err := SelectGDNSlotKernel(tc.peak, ladder)
			if err != nil {
				t.Fatalf("peak=%g: %v", tc.peak, err)
			}
			if got.Name != tc.want {
				t.Fatalf("peak=%g: selected %s, want %s", tc.peak, got.Name, tc.want)
			}
		}

		// Narrowest-holding property over a sweep: the winner is the FIRST ladder entry
		// whose cap holds the peak, so every earlier entry provably does not.
		sweep := []float32{0, 1, 447, 448, 448.5, 449, 65503, 65504, 65505, 1e6, 1e30, 3.38e38, 3.39e38, math.MaxFloat32}
		for _, p := range sweep {
			k, err := SelectGDNSlotKernel(p, ladder)
			if err != nil {
				t.Fatalf("peak=%g: %v", p, err)
			}
			ap := math.Abs(float64(p))
			idx := -1
			for i, cand := range ladder {
				if ap <= float64(cand.PeakCap) {
					idx = i
					break
				}
			}
			if idx < 0 || ladder[idx].Name != k.Name {
				t.Fatalf("peak=%g: selected %s, first holding entry is %s", p, k.Name, ladder[idx].Name)
			}
			if idx > 0 && ap <= float64(ladder[idx-1].PeakCap) {
				t.Fatalf("peak=%g: narrower %s also holds it; selection is not narrowest", p, ladder[idx-1].Name)
			}
		}

		if _, err := SelectGDNSlotKernel(float32(math.NaN()), ladder); err != ErrGDNSlotNaN {
			t.Fatalf("NaN peak: got %v, want ErrGDNSlotNaN", err)
		}
		for _, inf := range []float32{float32(math.Inf(1)), float32(math.Inf(-1))} {
			if _, err := SelectGDNSlotKernel(inf, ladder); err != ErrGDNSlotOverflow {
				t.Fatalf("peak=%+v: got %v, want ErrGDNSlotOverflow", inf, err)
			}
		}
		if _, err := SelectGDNSlotKernel(1, nil); err != ErrGDNSlotEmptyLadder {
			t.Fatalf("empty ladder: got %v, want ErrGDNSlotEmptyLadder", err)
		}

		if got := GDNPeakOf([]float32{-2.5, 1, -0.5}); got != 2.5 {
			t.Fatalf("GDNPeakOf mixed signs = %g, want 2.5", got)
		}
		if got := GDNPeakOf(nil); got != 0 {
			t.Fatalf("GDNPeakOf(empty) = %g, want 0", got)
		}
		if got := GDNPeakOf([]float32{1, float32(math.NaN())}); !math.IsNaN(float64(got)) {
			t.Fatalf("GDNPeakOf with NaN = %g, want NaN", got)
		}
		st := make([]float32, 16)
		st[3], st[11] = 500, -500
		if k, err := GDNSelectSlotForState(st, ladder); err != nil || k.Name != "f16" {
			t.Fatalf("state peak 500: got %v/%v, want f16", k, err)
		}
		st[0] = float32(math.Inf(1))
		if _, err := GDNSelectSlotForState(st, ladder); err != ErrGDNSlotOverflow {
			t.Fatalf("state with +Inf: got %v, want ErrGDNSlotOverflow", err)
		}
		if k, err := GDNSelectSlotForState(make([]float32, 4), ladder); err != nil || k.Name != "f8e4m3" {
			t.Fatalf("all-zero state: got %v/%v, want f8e4m3", k, err)
		}
	})

	t.Run("marked recurrence store and restore", func(t *testing.T) {
		m, err := NewGDNMarkList(40)
		if err != nil {
			t.Fatal(err)
		}
		for _, row := range []int{10, 20, 30} {
			if err := m.Add(row); err != nil {
				t.Fatal(err)
			}
		}
		r := NewGDNMarkedRecurrence(m)

		fill := func(row int, st []float32) {
			for i := range st {
				st[i] = float32(row*100 + i)
			}
		}
		st := make([]float32, 8)

		buf := make([]float32, 8)
		if _, err := r.Restore(buf, 100); err != ErrGDNNoStoredState {
			t.Fatalf("Restore before any snapshot: got %v, want ErrGDNNoStoredState", err)
		}
		if _, _, err := r.SlotForLatest(GDNDefaultSlotKernels); err != ErrGDNNoStoredState {
			t.Fatalf("SlotForLatest before any snapshot: got %v, want ErrGDNNoStoredState", err)
		}

		row, err := r.Snapshot(fillRet(10, st))
		if err != nil || row != 10 {
			t.Fatalf("first Snapshot: row=%d err=%v, want 10/nil", row, err)
		}
		fill(99, st) // mutate the source: the store must keep row 10's bytes

		if row, err := r.Snapshot(fillRet(20, st)); err != nil || row != 20 {
			t.Fatalf("second Snapshot: row=%d err=%v, want 20/nil", row, err)
		}

		if got, err := r.Restore(buf, 15); err != nil || got != 10 {
			t.Fatalf("Restore(15): got %d/%v, want 10/nil", got, err)
		}
		for i := range buf {
			if buf[i] != float32(10*100+i) {
				t.Fatalf("Restore(15) buf[%d]=%g, want %d (source mutation leaked into the store)", i, buf[i], 10*100+i)
			}
		}
		if got, _ := r.Restore(buf, 10); got != 10 {
			t.Fatalf("Restore(10): got %d, want 10", got)
		}
		if got, _ := r.Restore(buf, 25); got != 20 {
			t.Fatalf("Restore(25): got %d, want 20", got)
		}
		if got, _ := r.Restore(buf, 39); got != 20 {
			t.Fatalf("Restore(39) with two stored: got %d, want 20", got)
		}

		if row, err := r.Snapshot(fillRet(30, st)); err != nil || row != 30 {
			t.Fatalf("third Snapshot: row=%d err=%v, want 30/nil", row, err)
		}
		if got, err := r.Restore(buf, 39); err != nil || got != 30 {
			t.Fatalf("Restore(39): got %d/%v, want 30/nil", got, err)
		}
		for i := range buf {
			if buf[i] != float32(30*100+i) {
				t.Fatalf("Restore(39) buf[%d]=%g, want %d", i, buf[i], 30*100+i)
			}
		}
		if _, err := r.Restore(buf, 9); err != ErrGDNNoStoredState {
			t.Fatalf("Restore(9): got %v, want ErrGDNNoStoredState", err)
		}
		if _, err := r.Restore(buf[:4], 30); err != ErrGDNStateShape {
			t.Fatalf("short restore buffer: got %v, want ErrGDNStateShape", err)
		}
		if _, err := r.Snapshot(st); err != ErrGDNNoPendingMark {
			t.Fatalf("Snapshot past last mark: got %v, want ErrGDNNoPendingMark", err)
		}
		if r.Stored() != 3 {
			t.Fatalf("Stored()=%d, want 3", r.Stored())
		}
		// Latest snapshot holds values up to 3007: past the 448 E4M3 cap, inside f16.
		if k, row, err := r.SlotForLatest(GDNDefaultSlotKernels); err != nil || k.Name != "f16" || row != 30 {
			t.Fatalf("SlotForLatest: got %v row=%d err=%v, want f16/30/nil", k, row, err)
		}
	})

	t.Run("recurrence checkpoint replay", func(t *testing.T) {
		const dim, totalRows = 128, 40
		m, err := NewGDNMarkList(totalRows)
		if err != nil {
			t.Fatal(err)
		}
		for _, row := range []int{10, 20, 30} {
			if err := m.Add(row); err != nil {
				t.Fatal(err)
			}
		}
		r := NewGDNMarkedRecurrence(m)

		rng := rand.New(rand.NewSource(20261103))
		type step struct {
			qn, kn, vh []float32
			beta, g    float32
		}
		steps := make([]step, 0, totalRows)
		for i := 0; i < totalRows; i++ {
			s := step{
				qn:   make([]float32, dim),
				kn:   make([]float32, dim),
				vh:   make([]float32, dim),
				beta: 0.25 + 0.5*rng.Float32(),
				g:    float32(math.Exp(-float64(rng.Float32()) * 0.2)),
			}
			for j := 0; j < dim; j++ {
				s.qn[j] = rng.Float32()*0.2 - 0.1
				s.kn[j] = rng.Float32()*0.2 - 0.1
				s.vh[j] = rng.Float32()*0.2 - 0.1
			}
			steps = append(steps, s)
		}

		runFrom := func(from, to int, st, od, kvmem, delta []float32) {
			for row := from; row <= to; row++ {
				s := steps[row-1]
				Wave32GatedDeltaNetStep(st, s.qn, s.kn, s.vh, s.beta, s.g, od, kvmem, delta)
			}
		}

		// Uninterrupted reference trajectory over all 40 rows.
		stA := make([]float32, dim*dim)
		odA := make([]float32, dim)
		kvmemA, deltaA := make([]float32, dim), make([]float32, dim)
		runFrom(1, totalRows, stA, odA, kvmemA, deltaA)

		// Checkpointed trajectory: same recurrence, state snapshotted at each mark.
		stB := make([]float32, dim*dim)
		odB := make([]float32, dim)
		kvmemB, deltaB := make([]float32, dim), make([]float32, dim)
		var snapped []int
		for row := 1; row <= 30; row++ {
			s := steps[row-1]
			Wave32GatedDeltaNetStep(stB, s.qn, s.kn, s.vh, s.beta, s.g, odB, kvmemB, deltaB)
			if m.Contains(row) {
				got, err := r.Snapshot(stB)
				if err != nil || got != row {
					t.Fatalf("Snapshot at row %d: got %d/%v", row, got, err)
				}
				snapped = append(snapped, got)
			}
		}
		if len(snapped) != 3 || snapped[0] != 10 || snapped[1] != 20 || snapped[2] != 30 {
			t.Fatalf("snapshotted rows %v, want [10 20 30]", snapped)
		}

		// Resume from the mark-30 checkpoint into a fresh state buffer and finish the
		// segment. If the stored state is anything other than the live state at row 30,
		// the final bytes diverge from the uninterrupted trajectory.
		stR := make([]float32, dim*dim)
		if got, err := r.Restore(stR, 30); err != nil || got != 30 {
			t.Fatalf("Restore(30): got %d/%v", got, err)
		}
		runFrom(31, totalRows, stR, odB, kvmemB, deltaB)

		if d := MaxAbsDelta(stR, stA); d != 0 {
			t.Fatalf("resumed recurrent state drifted from the uninterrupted trajectory: max|delta|=%g", d)
		}
		if d := MaxAbsDelta(odB, odA); d != 0 {
			t.Fatalf("resumed readout drifted from the uninterrupted trajectory: max|delta|=%g", d)
		}
		t.Logf("checkpoint replay: state snapshotted at %v, resume from row 30 reproduced all %d x %d state elements and the readout bit-identically",
			snapped, dim, dim)
	})
}

// fillRet fills st with the deterministic row pattern and returns it, so Snapshot calls
// read clearly at the call site.
func fillRet(row int, st []float32) []float32 {
	for i := range st {
		st[i] = float32(row*100 + i)
	}
	return st
}
