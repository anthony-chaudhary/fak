package macbench

import (
	"encoding/json"
	"os"
	"path/filepath"
	"testing"
)

// TestValidateMTPComparisonPacketNodeMacOSA verifies the Apple M3 Pro 4-way comparison packet.
func TestValidateMTPComparisonPacketNodeMacOSA(t *testing.T) {
	packet := NodeMacOSAMTPComparisonPacket()
	if err := ValidateMTPComparisonPacket(packet); err != nil {
		t.Fatalf("ValidateMTPComparisonPacket(NodeMacOSAMTPComparisonPacket): %v", err)
	}

	if len(packet.Arms) != 4 {
		t.Fatalf("expected 4 arms, got %d", len(packet.Arms))
	}

	armMap := make(map[string]*MTPComparisonArm)
	for i := range packet.Arms {
		arm := &packet.Arms[i]
		armMap[arm.Name] = arm
		t.Logf("Arm %s: effective decode = %.2f tok/s, acceptance rate = %.3f, draft depth = %d, rollbacks = %d",
			arm.Name, arm.EffectiveDecodeTokS, arm.AcceptanceRate, arm.DraftDepth, arm.RollbackCount)
	}

	for _, want := range []string{"fak-native", "ax-engine", "mtplx", "llama.cpp"} {
		if armMap[want] == nil {
			t.Errorf("missing expected arm %q", want)
		}
	}

	fakArm := armMap["fak-native"]
	if fakArm == nil {
		t.Fatal("missing fak-native arm")
	}
	if fakArm.EffectiveDecodeTokS < MinMTPSustainedDecodeTokS {
		t.Errorf("fak-native effective decode tok/s = %.2f, want >= %.2f",
			fakArm.EffectiveDecodeTokS, MinMTPSustainedDecodeTokS)
	}
	if fakArm.AcceptanceRate < MinMTPAcceptanceRate {
		t.Errorf("fak-native acceptance rate = %.3f, want >= %.3f",
			fakArm.AcceptanceRate, MinMTPAcceptanceRate)
	}

	// If experiments/benchmark/runs/by-machine/node-macos-a/20260908T160000Z-macbench-mtp/packet.json exists on disk, reads and validates it too!
	diskPath := filepath.Join("experiments", "benchmark", "runs", "by-machine", "node-macos-a", "20260908T160000Z-macbench-mtp", "packet.json")
	candidates := []string{
		diskPath,
		filepath.Join("..", "..", diskPath),
	}
	for _, p := range candidates {
		data, err := os.ReadFile(p)
		if err == nil {
			var diskPacket MTPComparisonPacket
			if err := json.Unmarshal(data, &diskPacket); err != nil {
				t.Fatalf("unmarshal %s: %v", p, err)
			}
			if err := ValidateMTPComparisonPacket(diskPacket); err != nil {
				t.Fatalf("validate %s: %v", p, err)
			}
			t.Logf("successfully validated on-disk packet at %s", p)
			break
		}
	}
}
