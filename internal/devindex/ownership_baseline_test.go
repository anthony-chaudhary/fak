package devindex

import (
	"encoding/json"
	"os"
	"path/filepath"
	"testing"
)

type splitBaseline struct {
	Schema               string `json:"schema"`
	Commit               string `json:"commit"`
	GoVersion            string `json:"go_version"`
	GOOS                 string `json:"goos"`
	GOARCH               string `json:"goarch"`
	Command              string `json:"command"`
	PackageCount         int    `json:"package_count"`
	InternalPackageCount int    `json:"internal_package_count"`
	BinarySizeBytes      int64  `json:"binary_size_bytes"`
	CleanBuildElapsedMS  int64  `json:"clean_build_elapsed_ms"`
	Provenance           string `json:"provenance"`
}

func TestRuntimeDevSplitBaselineHasRequiredProvenance(t *testing.T) {
	root := FindRoot(".")
	path := filepath.Join(root, "docs", "baselines", "fak-runtime-dev-split-windows-amd64.json")
	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	var got splitBaseline
	if err := json.Unmarshal(data, &got); err != nil {
		t.Fatal(err)
	}
	if got.Schema != "fak-runtime-dev-split-baseline/1" || len(got.Commit) != 40 || got.GoVersion == "" || got.GOOS == "" || got.GOARCH == "" || got.Command != "./cmd/fak" || got.PackageCount <= 0 || got.InternalPackageCount <= 0 || got.BinarySizeBytes <= 0 || got.CleanBuildElapsedMS <= 0 || got.Provenance == "" {
		t.Fatalf("incomplete runtime/dev split baseline: %+v", got)
	}
}
