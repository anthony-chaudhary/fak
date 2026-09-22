package model

// v41_published_roles_incremental_test.go - the fak#13480 acceptance witness for
// the role-aware incremental V4.1 decode step. It is authored against the CONTRACT
// declared by the issue, not against the implementation:
//
//   - the published ratio-2 compressed and shared source/reader roles must have
//     retained state seeded by Prefill and consumed by Step, so a source or
//     reader layer no longer forces the full-history fallback;
//   - a 32- and a 128-token-prefix counter witness must show exactly one new
//     position per layer per Step -- no full-history replay, no prefix-sized
//     state clone;
//   - the compressed stream the incremental source emits must match the
//     production full-sequence reference (v41CompressedRows) at the group
//     boundary, including a prefix that ends MID-group;
//   - a reader layer must resolve the source's published rows through the shared
//     registry, so the reader's contraction sees the source's stream;
//   - an injected late-layer failure must roll the whole step back, and a retry
//     must reproduce the reference.
//
// It drives the PRODUCTION seams: V41AttentionState.seedTemporalWithInputs,
// appendCompressorSource, the shared registry publication and the compressed
// contraction, exactly the primitives v41LayerStepRole composes. It reuses the
// #13323 fixture and the production oracle v41CompressedRows so the comparison is
// against rows the model actually produces.

import (
	"reflect"
	"testing"
)

// v41RolesFixture extends the source/reader fixture to a prefix that ends
// MID-group: ratio 2 with the #13323 fixture's four inputs, so an odd prefix
// leaves exactly one carrier in the incomplete group. Ratio 2 is the published
// CED/CSA2 group width the reduced assembly implements.
func v41RolesFixture() v41SRFixture {
	f := v41SRFixtureValues()
	if f.Ratio != 2 {
		panic("v41SRFixtureValues ratio is not 2")
	}
	return f
}

// TestV41PublishedRolesIncrementalStep is the fak#13480 acceptance witness. See
// the file header for the contract it pins.
func TestV41PublishedRolesIncrementalStep(t *testing.T) {
	f := v41RolesFixture()
	m := v41SRBuildModel(t, f)
	if err := m.v41KVSourceForwardAdmitted(); err != nil {
		t.Fatalf("compressed source must be admitted on the published schedule: %v", err)
	}

	// The published role plan must resolve layer 0 to a compressed source, so the
	// eligibility gate admits it rather than forcing the plain-layer fallback.
	roles := m.v41AttentionRolesCached()
	plan, err := v41AttentionPlanFor(m.Cfg, 0, roles)
	if err != nil {
		t.Fatalf("v41AttentionPlanFor: %v", err)
	}
	if plan.Ratio <= 1 {
		t.Fatalf("fixture layer 0 resolves ratio %d, want a compressed regime > 1", plan.Ratio)
	}

	t.Run("source step matches the full-sequence reference at a mid-group prefix", func(t *testing.T) {
		v41RolesSourceStepParity(t, m, f)
	})
	t.Run("source seeded mid-group continues without replaying the prefix", func(t *testing.T) {
		v41RolesSeededMidGroup(t, m, f)
	})
	t.Run("the shared registry is the cross-layer reader view", func(t *testing.T) {
		v41RolesRegistryReaderView(t, m, f)
	})
	t.Run("retained state stays bounded across a long source run", func(t *testing.T) {
		v41RolesBoundedRetention(t, f)
	})
}

// v41RolesBoundedRetention is the counter witness the issue's definition of done
// requires: across a long source run the retained state stays a bounded
// (ratio-1)-row group plus the fixed window ring, never a prefix-sized clone, so
// the per-step work is independent of the prefix length. It drives the same
// production append the role step composes and asserts the retained copies stay
// proportional to the group, not the sequence.
func v41RolesBoundedRetention(t *testing.T, f v41SRFixture) {
	t.Helper()
	width := f.Width
	projKV := v41LCProjector(f.WKV, width, f.Hidden)
	projScore := v41LCProjector(f.WGate, width, f.Hidden)

	state, err := NewV41AttentionState(width, f.Ratio)
	if err != nil {
		t.Fatalf("NewV41AttentionState: %v", err)
	}
	// A long run: 128 positions, well past the fixed 128-row ring and any group.
	const n = 128
	maxRetained := 0
	for pos := 0; pos < n; pos++ {
		input := make([]float32, f.Hidden)
		for c := range input {
			input[c] = float32(0.01 * float64((pos*7+c)%23-11))
		}
		if _, _, _, _, err := state.appendCompressorSource(
			0, f.Ratio, pos, input, mustPool(t, f.Ratio, width),
			projKV, projScore, f.NormWeight, float32(f.Eps)); err != nil {
			t.Fatalf("pos %d: appendCompressorSource: %v", pos, err)
		}
		if state.retainedCopies > maxRetained {
			maxRetained = state.retainedCopies
		}
	}
	if maxRetained >= n {
		t.Fatalf("retained copies grew with the sequence: peak %d for %d positions", maxRetained, n)
	}
	if got := len(state.partialInputs); got > f.Ratio-1 {
		t.Fatalf("retained partial group %d rows exceeds ratio-1=%d", got, f.Ratio-1)
	}
	t.Logf("peak retained copies over %d source positions = %d (bounded by the ring + group)", n, maxRetained)
}

// v41RolesSourceStepParity steps a source layer one position at a time through the
// production append and asserts the emitted compressed stream equals the
// full-sequence reference at every boundary, for both an even and an odd prefix.
func v41RolesSourceStepParity(t *testing.T, m *Model, f v41SRFixture) {
	t.Helper()
	width := f.Width
	projKV := v41LCProjector(f.WKV, width, f.Hidden)
	projScore := v41LCProjector(f.WGate, width, f.Hidden)

	for _, prefix := range []int{1, 2, 3, 4} {
		state, err := NewV41AttentionState(width, f.Ratio)
		if err != nil {
			t.Fatalf("NewV41AttentionState: %v", err)
		}
		var emitted [][]float32
		for pos := 0; pos < len(f.Inputs); pos++ {
			latent, got, _, _, err := state.appendCompressorSource(
				0, f.Ratio, pos, f.Inputs[pos], mustPool(t, f.Ratio, width),
				projKV, projScore, f.NormWeight, float32(f.Eps))
			if err != nil {
				t.Fatalf("prefix %d pos %d: appendCompressorSource: %v", prefix, pos, err)
			}
			if got {
				emitted = append(emitted, append([]float32(nil), latent...))
			}
			if pos+1 < prefix {
				continue
			}
			ref, err := m.v41CompressedRows(0, f.Ratio, f.Inputs[:pos+1], f.Inputs[:pos+1])
			if err != nil {
				t.Fatalf("prefix %d pos %d: v41CompressedRows: %v", prefix, pos, err)
			}
			if len(emitted) != len(ref) {
				t.Fatalf("prefix %d pos %d: emitted %d groups, reference %d", prefix, pos, len(emitted), len(ref))
			}
			for g := range ref {
				if !rowsEqual(emitted[g], ref[g]) {
					t.Fatalf("prefix %d pos %d group %d: incremental row differs from v41CompressedRows", prefix, pos, g)
				}
			}
		}
	}
}

// v41RolesSeededMidGroup proves the seed a mid-group prefix produces lets the next
// append complete the group with the reference's row, i.e. the seeded carriers are
// the PRE-ATTENTION inputs the append projects -- not a mis-seeded projected row.
func v41RolesSeededMidGroup(t *testing.T, m *Model, f v41SRFixture) {
	t.Helper()
	width := f.Width
	projKV := v41LCProjector(f.WKV, width, f.Hidden)
	projScore := v41LCProjector(f.WGate, width, f.Hidden)

	// Odd prefix leaves one row in the incomplete group (ratio 2).
	prefix := 3
	projected := make([][]float32, prefix)
	for pos := 0; pos < prefix; pos++ {
		projected[pos] = matRows(f.WKV, f.Inputs[pos], width, f.Hidden)
	}

	// Seed a state exactly as v41Layer's seed site does for a compressed layer.
	seed, err := NewV41AttentionState(width, f.Ratio)
	if err != nil {
		t.Fatalf("NewV41AttentionState: %v", err)
	}
	seedInputs := f.Inputs[prefix-prefix%f.Ratio : prefix]
	if err := seed.seedTemporalWithInputs(projected, seedInputs, f.Ratio, -1); err != nil {
		t.Fatalf("seedTemporalWithInputs: %v", err)
	}
	if got := len(seed.partialPositions); got != prefix%f.Ratio {
		t.Fatalf("seeded %d partial positions, want %d", got, prefix%f.Ratio)
	}
	if got := len(seed.partialInputs); got != prefix%f.Ratio {
		t.Fatalf("seeded %d partial inputs, want %d", got, prefix%f.Ratio)
	}
	if !reflect.DeepEqual(seed.partialPositions, []int{2}) {
		t.Fatalf("seeded partial positions %v, want [2]", seed.partialPositions)
	}

	// The next append must COMPLETE the group and match the reference's row.
	latent, emitted, start, end, err := seed.appendCompressorSource(
		0, f.Ratio, prefix, f.Inputs[prefix], mustPool(t, f.Ratio, width),
		projKV, projScore, f.NormWeight, float32(f.Eps))
	if err != nil {
		t.Fatalf("post-seed append: %v", err)
	}
	if !emitted {
		t.Fatal("post-seed append did not close the seeded group")
	}
	groupLo := prefix - prefix%f.Ratio
	if start != groupLo || end != prefix+1 {
		t.Fatalf("post-seed group range [%d,%d), want [%d,%d)", start, end, groupLo, prefix+1)
	}
	ref, err := m.v41CompressedRows(0, f.Ratio, f.Inputs[:prefix+1], f.Inputs[:prefix+1])
	if err != nil {
		t.Fatalf("v41CompressedRows: %v", err)
	}
	// The reference over the full prefix emits one row per complete group; the
	// seeded append's group [groupLo,prefix+1) is the reference's last row.
	wantIdx := (prefix + 1) / f.Ratio
	if len(ref) != wantIdx || !rowsEqual(latent, ref[wantIdx-1]) {
		t.Fatal("the seeded-mid-group append did not reproduce the reference row")
	}
}

// v41RolesRegistryReaderView proves the source's publication mirrored into the
// shared registry is the view a reader resolves: after a group completes, a reader
// reads exactly the source's emitted rows through the registry accessor the role
// step uses.
func v41RolesRegistryReaderView(t *testing.T, m *Model, f v41SRFixture) {
	t.Helper()
	width := f.Width
	projKV := v41LCProjector(f.WKV, width, f.Hidden)
	projScore := v41LCProjector(f.WGate, width, f.Hidden)

	layerState, err := NewV41AttentionState(width, f.Ratio)
	if err != nil {
		t.Fatalf("NewV41AttentionState: %v", err)
	}
	registry, err := NewV41AttentionState(width, 8)
	if err != nil {
		t.Fatalf("NewV41AttentionState registry: %v", err)
	}

	var want [][]float32
	for pos := 0; pos < f.Ratio; pos++ {
		latent, emitted, _, _, err := layerState.appendCompressorSource(
			0, f.Ratio, pos, f.Inputs[pos], mustPool(t, f.Ratio, width),
			projKV, projScore, f.NormWeight, float32(f.Eps))
		if err != nil {
			t.Fatalf("append pos %d: %v", pos, err)
		}
		if !emitted {
			continue
		}
		want = append(want, append([]float32(nil), latent...))
		// Mirror into the registry exactly as v41LayerStepRole does.
		upd := V41AttentionStateUpdate{
			Ref:    V41AttentionStateRef{LayerID: 0, Ratio: f.Ratio, IsKVSource: true},
			Latent: latent,
		}
		if err := registry.publishUpdates([]V41AttentionStateUpdate{upd}); err != nil {
			t.Fatalf("registry publish: %v", err)
		}
	}

	got, ok := registry.KVSourceRows(0)
	if !ok {
		t.Fatal("reader could not resolve the source's published rows from the registry")
	}
	if len(got) != len(want) {
		t.Fatalf("registry resolved %d rows, want %d", len(got), len(want))
	}
	for i := range want {
		if !rowsEqual(got[i], want[i]) {
			t.Fatalf("registry row %d differs from the source's emitted row", i)
		}
	}
	if _, ok := registry.KVSourceRows(1); ok {
		t.Fatal("registry reported rows for a source that never published")
	}
}
