package harnessinit

import (
	"context"
	"encoding/json"
	"os"
	"path/filepath"
	"reflect"
	"runtime"
	"strings"
	"testing"
)

func TestCrossDogfoodMatrix(t *testing.T) {
	repoRoot, err := filepath.Abs(filepath.Join("..", ".."))
	if err != nil {
		t.Fatal(err)
	}
	matrix, err := CrossDogfood(context.Background(), repoRoot)
	if err != nil {
		t.Fatal(err)
	}
	if matrix.Schema != CrossDogfoodSchema || matrix.Platform != runtime.GOOS+"/"+runtime.GOARCH {
		t.Fatalf("matrix header=%+v", matrix)
	}
	if matrix.Hosts != 3 || matrix.SubsystemsPerHost != 5 || matrix.DriftRefusals != 3 || len(matrix.Rows) != 3 {
		t.Fatalf("matrix envelope=%+v", matrix)
	}
	wantHosts := []string{"codex", "claude", "acme"}
	for i, row := range matrix.Rows {
		if row.Host != wantHosts[i] || row.GeneratedHost != row.Host {
			t.Fatalf("row[%d] identity=%+v", i, row)
		}
		profileJSON, _ := json.Marshal(row.Profile)
		guardJSON, _ := json.Marshal(row.LaunchBinding)
		if string(profileJSON) != string(guardJSON) {
			t.Fatalf("%s profile and guard plan disagree\nprofile=%s\nguard=%s", row.Host, profileJSON, guardJSON)
		}
		if row.Lock.ID == "" || row.Receipt.LockID != row.Lock.ID || len(row.ComponentGraph) < 3 {
			t.Fatalf("%s incomplete artifacts: %+v", row.Host, row)
		}
		if !strings.Contains(row.DriftRefusal, "stale") {
			t.Fatalf("%s drift refusal=%q", row.Host, row.DriftRefusal)
		}
	}

	raw, err := os.ReadFile(filepath.Join(repoRoot, "docs", "_witnesses", "issue-8227-harness-cross-dogfood.json"))
	if err != nil {
		t.Fatal(err)
	}
	var witness struct {
		PlatformWitnesses []struct {
			Platform string `json:"platform"`
			Status   string `json:"status"`
		} `json:"platform_witnesses"`
		Matrix CrossDogfoodMatrix `json:"matrix"`
	}
	if err := json.Unmarshal(raw, &witness); err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(witness.Matrix.Rows, matrix.Rows) {
		t.Fatal("committed machine readout is stale relative to the live resolved matrix")
	}
	platforms := map[string]string{}
	for _, row := range witness.PlatformWitnesses {
		platforms[row.Platform] = row.Status
	}
	if platforms["windows/amd64"] != "PASS" || platforms["linux/amd64"] != "PASS" {
		t.Fatalf("platform witnesses=%v", platforms)
	}
}
