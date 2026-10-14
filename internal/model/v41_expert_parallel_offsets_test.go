package model

import (
	"reflect"
	"testing"
)

// This is the offset-table allocation loop from the prior owner-panel path,
// retained as a same-workload allocation control rather than a second engine.
func v41EPOwnerPanelLegacyOffsets(picks [][]routePick) [][]int {
	offsets := make([][]int, len(picks))
	for token, row := range picks {
		offsets[token] = make([]int, len(row))
	}
	return offsets
}

// fak-test:runtime fast est=1s lane=default
func TestV41EPOwnerPanelOffsetSlab(t *testing.T) {
	for _, lengths := range [][]int{nil, {0, 0, 0}, {1}, {3, 0, 2, 1}, {0, 2, 0, 3, 0}} {
		picks := make([][]routePick, len(lengths))
		total := 0
		for token, count := range lengths {
			picks[token] = make([]routePick, count)
			total += count
		}
		got := v41EPOwnerPanelOffsets(picks, total)
		want := v41EPOwnerPanelLegacyOffsets(picks)
		if !reflect.DeepEqual(got, want) {
			t.Fatalf("offset rows differ for pick counts %v: got %v want %v", lengths, got, want)
		}
		slabSlots := 0
		for token, row := range got {
			slabSlots += len(row)
			if len(row) != len(picks[token]) || cap(row) != len(picks[token]) {
				t.Fatalf("token %d offset span len/cap=%d/%d, picks=%d", token, len(row), cap(row), len(picks[token]))
			}
			for slot := range row {
				row[slot] = -1
				want[token][slot] = -1
			}
			// At exact capacity, an append must not overwrite the next row.
			extended := append(row, token+1)
			if extended[len(row)] != token+1 || !reflect.DeepEqual(got, want) {
				t.Fatal("offset append changed a neighboring token span")
			}
		}
		if slabSlots != total {
			t.Fatalf("slab describes %d slots, admitted pick sum is %d", slabSlots, total)
		}
	}
}

// fak-test:runtime fast est=1s lane=default
func TestV41EPOwnerPanelOffsetAllocations(t *testing.T) {
	picks := make([][]routePick, v41EPOwnerPanelTokens)
	total := 0
	for token := range picks {
		picks[token] = make([]routePick, V41RouterTopK)
		for slot := range picks[token] {
			picks[token][slot] = routePick{expert: slot, weight: float32(token+1) / 64}
		}
		total += len(picks[token])
	}
	var retained [][]int
	candidate := testing.AllocsPerRun(100, func() { retained = v41EPOwnerPanelOffsets(picks, total) })
	if len(retained) != len(picks) || len(retained[len(picks)-1]) != len(picks[len(picks)-1]) {
		t.Fatal("candidate allocation witness did not retain the complete panel")
	}
	legacy := testing.AllocsPerRun(100, func() { retained = v41EPOwnerPanelLegacyOffsets(picks) })
	if len(retained) != len(picks) || len(retained[len(picks)-1]) != len(picks[len(picks)-1]) {
		t.Fatal("legacy allocation witness did not retain the complete panel")
	}
	// One slice-header table and one exact integer backing slab, irrespective
	// of token count. The reference instead allocates each nonempty pick row.
	const offsetTables = 2
	if candidate > offsetTables || legacy-candidate < float64(len(picks)-1) {
		t.Fatalf("offset allocation reduction missing: candidate=%g legacy=%g, tokens=%d picks=%d", candidate, legacy, len(picks), total)
	}
	t.Logf("offset allocations: candidate=%g legacy=%g tokens=%d picks=%d", candidate, legacy, len(picks), total)
	for _, empty := range [][][]routePick{nil, make([][]routePick, v41EPOwnerPanelTokens)} {
		allocations := testing.AllocsPerRun(100, func() { retained = v41EPOwnerPanelOffsets(empty, 0) })
		maxTables := 0
		if len(empty) != 0 {
			maxTables = 1 // Only the row-header table needs storage.
		}
		if allocations > float64(maxTables) || len(retained) != len(empty) {
			t.Fatalf("zero-pick panel allocations=%g, header-table bound=%d", allocations, maxTables)
		}
		for _, row := range retained {
			if row == nil || len(row) != 0 || cap(row) != 0 {
				t.Fatal("zero-pick row lost its explicit empty span")
			}
		}
	}
}
