package modelperfobs_test

import (
	"os"
	"path/filepath"
	"reflect"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/laneadmit"
)

func TestModelperfobsLaneArbitratesRealTree(t *testing.T) {
	manifest, err := os.ReadFile(filepath.Join("..", "..", "dos.toml"))
	if err != nil {
		t.Fatalf("read dos.toml: %v", err)
	}
	taxonomy := laneadmit.ParseTaxonomy(manifest)
	wantTree := []string{"internal/modelperfobs/**"}
	if got := taxonomy.TreeFor("modelperfobs"); !reflect.DeepEqual(got, wantTree) {
		t.Fatalf("modelperfobs lane tree = %v, want %v", got, wantTree)
	}

	held := []laneadmit.Lease{{
		ID: "peer-modelperfobs", Lane: "modelperfobs", Tree: wantTree, Holder: "peer",
	}}
	overlap := laneadmit.Decide(
		laneadmit.Request{Surface: laneadmit.SurfaceManual, Lane: "modelperfobs", Holder: "owner"},
		held,
		taxonomy,
	)
	if overlap.Admit || overlap.Reason != laneadmit.ReasonCollisionRisk {
		t.Fatalf("same modelperfobs tree must refuse with %q, got %+v", laneadmit.ReasonCollisionRisk, overlap)
	}
	if len(overlap.Conflicts) != 1 || overlap.Conflicts[0].Kind != laneadmit.ConflictSameLane {
		t.Fatalf("same modelperfobs lane must report one same-lane conflict, got %+v", overlap.Conflicts)
	}

	disjoint := laneadmit.Decide(
		laneadmit.Request{Surface: laneadmit.SurfaceManual, Lane: "workflow", Holder: "owner"},
		held,
		taxonomy,
	)
	if !disjoint.Admit {
		t.Fatalf("disjoint workflow tree must admit, got %+v", disjoint)
	}
}
