package ultracodebench_test

import (
	"os"
	"path/filepath"
	"reflect"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/laneadmit"
)

func TestUltracodebenchLaneArbitratesRealTree(t *testing.T) {
	manifest, err := os.ReadFile(filepath.Join("..", "..", "dos.toml"))
	if err != nil {
		t.Fatalf("read dos.toml: %v", err)
	}
	taxonomy := laneadmit.ParseTaxonomy(manifest)
	wantTree := []string{"internal/ultracodebench/**"}
	if got := taxonomy.TreeFor("ultracodebench"); !reflect.DeepEqual(got, wantTree) {
		t.Fatalf("ultracodebench lane tree = %v, want %v", got, wantTree)
	}

	held := []laneadmit.Lease{{
		ID: "peer-ultracodebench", Lane: "ultracodebench", Tree: wantTree, Holder: "peer",
	}}
	overlap := laneadmit.Decide(
		laneadmit.Request{Surface: laneadmit.SurfaceManual, Lane: "ultracodebench", Holder: "owner"},
		held,
		taxonomy,
	)
	if overlap.Admit || overlap.Reason != laneadmit.ReasonCollisionRisk {
		t.Fatalf("same ultracodebench tree must refuse with %q, got %+v", laneadmit.ReasonCollisionRisk, overlap)
	}
	if len(overlap.Conflicts) != 1 || overlap.Conflicts[0].Kind != laneadmit.ConflictSameLane {
		t.Fatalf("same ultracodebench lane must report one same-lane conflict, got %+v", overlap.Conflicts)
	}

	disjoint := laneadmit.Decide(
		laneadmit.Request{Surface: laneadmit.SurfaceManual, Lane: "workflow", Holder: "owner"},
		held,
		taxonomy,
	)
	if !disjoint.Admit {
		t.Fatalf("disjoint workflow tree must admit, got %+v", disjoint)
	}

	for _, path := range []string{"cmd/fak/accounts_launch.go", "cmd/fak/orchestration_launch.go"} {
		if got := laneadmit.LaneForPath(path, taxonomy, laneadmit.GranLeaf); got != "cmd" {
			t.Errorf("%s resolves to lane %q, want shared cmd lane", path, got)
		}
	}
}
