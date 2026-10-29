package v41

import (
	"errors"
	"os"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/compute"
)

func TestV41Placement1017(t *testing.T) {
	config, err := os.ReadFile("../testdata/deepseek_v41_flash_config.json")
	if err != nil {
		t.Fatal(err)
	}
	target := V41PlacementTarget{CheckpointRevision: V41Placement1017Revision, IndexSHA256: V41Placement1017IndexSHA256, GPUs: 1, DeviceBytes: 24000000000, HostBytes: 32 << 30, NVMeBytes: 600000000000, ContextTokens: 256, WorkspaceBytes: 4 << 30}
	p, err := PlanDeepSeekV41Placement(config, target)
	if err != nil {
		t.Fatal(err)
	}
	if p.TextTensorBytes != 501382642128 || p.Device.Total() > target.DeviceBytes || p.Host.Total() > target.HostBytes {
		t.Fatalf("bad1017 ledger %+v", p)
	}
	var encoded, excluded, tiers int64
	for _, r := range p.Classes {
		encoded += r.Bytes
	}
	for _, r := range p.Excluded {
		excluded += r.Bytes
	}
	for _, r := range p.Tiers {
		tiers += r.Bytes
	}
	if encoded != p.TextTensorBytes || encoded+excluded != p.CheckpointTensorBytes || tiers != encoded {
		t.Fatalf("lost/duplicated encoded bytes: %d %d %d", encoded, excluded, tiers)
	}
	if p.StreamedBytes <= 0 || p.SpillLayers != 40 {
		t.Fatalf("24GB conservative whole-layer spill: %+v", p)
	}
	// Refuse a disk budget for only cache misses: all source shards remain.
	target.NVMeBytes = p.StreamedBytes
	if _, err = PlanDeepSeekV41Placement(config, target); !errors.Is(err, ErrV41PlacementUnfit) {
		t.Fatalf("partial disk accepted: %v", err)
	}
	target.NVMeBytes = 600000000000
	target.DeviceBytes = 8 << 30
	_, err = PlanDeepSeekV41Placement(config, target)
	var fit *compute.FitError
	if !errors.Is(err, ErrV41PlacementUnfit) || !errors.As(err, &fit) || fit.Scope != compute.MemoryScopeDevice {
		t.Fatalf("want device refusal: %v", err)
	}
	target.DeviceBytes = 24000000000
	target.GPUs = 2
	if _, err = PlanDeepSeekV41Placement(config, target); err == nil {
		t.Fatal("multiGPU accepted")
	}
	target.GPUs = 1
	target.CheckpointRevision = "dba1be0a40aa45a94ad051997016db3960a90277"
	if _, err = PlanDeepSeekV41Placement(config, target); err == nil {
		t.Fatal("old revision accepted")
	}
	target.CheckpointRevision = V41Placement1017Revision
	target.IndexSHA256 = "wrong"
	if _, err = PlanDeepSeekV41Placement(config, target); err == nil {
		t.Fatal("wrong index accepted")
	}
	target.IndexSHA256 = V41Placement1017IndexSHA256
	mutated := append([]byte(nil), config...)
	mutated[0] = ' '
	if _, err = PlanDeepSeekV41Placement(mutated, target); err == nil {
		t.Fatal("mutated config accepted")
	}
	target.ContextTokens = 1 << 30
	if _, err = PlanDeepSeekV41Placement(config, target); err == nil {
		t.Fatal("unbounded context accepted")
	}
}
