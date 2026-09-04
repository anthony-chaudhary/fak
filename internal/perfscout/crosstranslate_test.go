package perfscout

import (
	"strings"
	"testing"
)

func TestFilterInnovationsBySource(t *testing.T) {
	mlxInnovations := FilterInnovations("mlx", "", "")
	if len(mlxInnovations) < 2 {
		t.Fatalf("expected at least 2 MLX innovations, got %d", len(mlxInnovations))
	}
	for _, in := range mlxInnovations {
		if !strings.Contains(strings.ToLower(in.SourcePlatform), "mlx") {
			t.Errorf("expected MLX in source platform, got: %s", in.SourcePlatform)
		}
	}
}

func TestFilterInnovationsByTarget(t *testing.T) {
	cudaTargets := FilterInnovations("", "cuda", "")
	if len(cudaTargets) < 4 {
		t.Fatalf("expected at least 4 CUDA-targeted innovations, got %d", len(cudaTargets))
	}

	metalTargets := FilterInnovations("", "metal", "")
	if len(metalTargets) < 3 {
		t.Fatalf("expected at least 3 Metal-targeted innovations, got %d", len(metalTargets))
	}
}

func TestFilterInnovationsByKeyword(t *testing.T) {
	roceInnovations := FilterInnovations("", "", "roce")
	if len(roceInnovations) == 0 {
		t.Fatalf("expected ROCe keyword matches, got 0")
	}

	md := RenderCrossInnovationsMarkdown(roceInnovations)
	if !strings.Contains(md, "XINNOV-03") {
		t.Errorf("expected XINNOV-03 in markdown, got:\n%s", md)
	}
}

func TestRenderCrossInnovationsJSON(t *testing.T) {
	all := DefaultCrossInnovations
	data, err := RenderCrossInnovationsJSON(all)
	if err != nil {
		t.Fatalf("RenderCrossInnovationsJSON error: %v", err)
	}
	if len(data) == 0 {
		t.Fatalf("expected non-empty JSON data")
	}
}
