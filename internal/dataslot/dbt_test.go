package dataslot

import (
	"os"
	"path/filepath"
	"reflect"
	"testing"
)

func TestReadDBTSemantics(t *testing.T) {
	dir := t.TempDir()
	manifestFile := filepath.Join(dir, "manifest.json")

	rawManifest := `{
  "nodes": {
    "model.my_project.stg_customers": {
      "name": "stg_customers",
      "package_name": "my_project",
      "resource_type": "model",
      "description": "Cleaned customer records",
      "depends_on": {
        "nodes": []
      },
      "columns": {
        "customer_id": { "name": "customer_id" },
        "email": { "name": "email" },
        "name": { "name": "name" }
      },
      "tags": ["staging", "core"]
    },
    "model.my_project.stg_orders": {
      "name": "stg_orders",
      "package_name": "my_project",
      "resource_type": "model",
      "description": "Cleaned order transactions",
      "depends_on": {
        "nodes": []
      },
      "columns": {
        "order_id": { "name": "order_id" },
        "customer_id": { "name": "customer_id" },
        "amount": { "name": "amount" }
      },
      "tags": ["staging"]
    },
    "model.my_project.fct_orders": {
      "name": "fct_orders",
      "package_name": "my_project",
      "resource_type": "model",
      "description": "Order transactions joined with customer attributes",
      "depends_on": {
        "nodes": [
          "model.my_project.stg_customers",
          "model.my_project.stg_orders"
        ]
      },
      "columns": {
        "order_id": { "name": "order_id" },
        "customer_id": { "name": "customer_id" },
        "order_total": { "name": "order_total" }
      },
      "tags": ["marts"]
    },
    "seed.my_project.raw_codes": {
      "name": "raw_codes",
      "resource_type": "seed"
    }
  }
}`

	if err := os.WriteFile(manifestFile, []byte(rawManifest), 0644); err != nil {
		t.Fatal(err)
	}

	receipt, err := ReadDBTSemantics(manifestFile)
	if err != nil {
		t.Fatalf("ReadDBTSemantics failed: %v", err)
	}

	if !receipt.RawSQLDormant {
		t.Errorf("expected RawSQLDormant=true")
	}
	if !receipt.ZeroNetwork {
		t.Errorf("expected ZeroNetwork=true")
	}
	if receipt.ModelCount != 3 {
		t.Errorf("expected 3 models, got %d", receipt.ModelCount)
	}
	if receipt.ArtifactSHA256 == "" {
		t.Errorf("expected non-empty artifact digest")
	}

	// 1. Lineage test for fct_orders
	upstream, downstream, ok := receipt.Lineage("fct_orders")
	if !ok {
		t.Fatal("expected Lineage ok for fct_orders")
	}
	wantUpstream := []string{"stg_customers", "stg_orders"}
	if !reflect.DeepEqual(upstream, wantUpstream) {
		t.Errorf("fct_orders upstream = %v, want %v", upstream, wantUpstream)
	}
	if len(downstream) != 0 {
		t.Errorf("fct_orders downstream should be empty, got %v", downstream)
	}

	// 2. Lineage test for stg_customers (downstream check)
	upCust, downCust, ok := receipt.Lineage("stg_customers")
	if !ok {
		t.Fatal("expected Lineage ok for stg_customers")
	}
	if len(upCust) != 0 {
		t.Errorf("stg_customers upstream should be empty, got %v", upCust)
	}
	wantDown := []string{"fct_orders"}
	if !reflect.DeepEqual(downCust, wantDown) {
		t.Errorf("stg_customers downstream = %v, want %v", downCust, wantDown)
	}

	// 3. Schema column reflection
	cols, err := receipt.QueryModelColumns("stg_customers")
	if err != nil {
		t.Fatalf("QueryModelColumns failed: %v", err)
	}
	wantCols := []string{"customer_id", "email", "name"}
	if !reflect.DeepEqual(cols, wantCols) {
		t.Errorf("stg_customers columns = %v, want %v", cols, wantCols)
	}

	// Nonexistent model
	if _, _, ok := receipt.Lineage("nonexistent"); ok {
		t.Errorf("expected ok=false for nonexistent model lineage")
	}
	if _, err := receipt.QueryModelColumns("nonexistent"); err == nil {
		t.Errorf("expected error querying columns for nonexistent model")
	}
}

func TestDetectDBTProject(t *testing.T) {
	dir := t.TempDir()
	if err := os.WriteFile(filepath.Join(dir, "dbt_project.yml"), []byte("name: 'my_analytics'\n"), 0644); err != nil {
		t.Fatal(err)
	}

	descs, err := Detect(dir)
	if err != nil {
		t.Fatal(err)
	}

	found := false
	for _, d := range descs {
		if d.Family == FamilyDBT && d.SourceArtifact == "dbt_project.yml" {
			found = true
			if d.Status != StatusReady {
				t.Errorf("expected status ready for dbt project descriptor")
			}
		}
	}
	if !found {
		t.Errorf("expected dbt project descriptor detected")
	}
}
