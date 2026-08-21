package modelloadplan

import "testing"

func TestBuildSelectsBalancedQ4WhenItFits(t *testing.T) {
	p, err := Build(Request{Setup: "personal", Goal: "balanced", LocalPolicy: "auto", Memory: "unified", DeviceBytes: 32 << 30, DiskBytes: 32 << 30})
	if err != nil {
		t.Fatal(err)
	}
	if p.Selected == nil || p.Selected.Quantization != "Q4_K_M" {
		t.Fatalf("selected = %#v", p.Selected)
	}
	if p.Selected.URI != "hf://unsloth/Qwen3.8-27B-GGUF@"+GGUFRevision+"/Qwen3.8-27B-UD-Q4_K_M.gguf" {
		t.Fatalf("uri = %q", p.Selected.URI)
	}
}

func TestBuildQualityChoosesHighestFittingQuant(t *testing.T) {
	p, err := Build(Request{Goal: "quality", LocalPolicy: "require", Memory: "unified", DeviceBytes: 27 << 30, DiskBytes: 27 << 30})
	if err != nil {
		t.Fatal(err)
	}
	if p.Selected == nil || p.Selected.Quantization != "Q6_K" {
		t.Fatalf("selected = %#v", p.Selected)
	}
}

func TestBuildPlansSplitMemoryOffload(t *testing.T) {
	p, err := Build(Request{Goal: "balanced", LocalPolicy: "require", Memory: "split", DeviceBytes: 12 << 30, HostBytes: 16 << 30, DiskBytes: 24 << 30})
	if err != nil {
		t.Fatal(err)
	}
	if p.Selected == nil || p.Selected.Quantization != "Q4_K_M" {
		t.Fatalf("selected = %#v", p.Selected)
	}
	if p.Selected.HostBytes == 0 {
		t.Fatal("expected host offload")
	}
}

func TestBuildFallsBackHostedWhenCapacityUnknown(t *testing.T) {
	p, err := Build(Request{Setup: "shared"})
	if err != nil {
		t.Fatal(err)
	}
	if p.Selected == nil || p.Selected.Kind != "hosted" {
		t.Fatalf("selected = %#v", p.Selected)
	}
}

func TestBuildLocalRequiredDoesNotHideNoFit(t *testing.T) {
	p, err := Build(Request{LocalPolicy: "require", DeviceBytes: 8 << 30, DiskBytes: 8 << 30})
	if err != nil {
		t.Fatal(err)
	}
	if p.Selected != nil {
		t.Fatalf("unexpected selection %#v", p.Selected)
	}
}
