package macobs

import (
	"testing"
)

func TestComputeHeadroomStandard(t *testing.T) {
	hw := HardwareTelemetry{
		TotalSystemMemoryBytes: 32 * 1024 * 1024 * 1024, // 32GB
		WiredMemoryLimitBytes:  24 * 1024 * 1024 * 1024, // 24GB
		Available:              true,
	}

	cfg := HeadroomConfig{
		Layers:             28,
		KVHeads:            4,
		HeadDim:            128,
		KVBytesPerElement:  2,
		ModelWeightBytes:   5 * 1024 * 1024 * 1024, // 5GB
		ContextTokens:      8192,
		SharedPrefixTokens: 4096,
		PrivateTailTokens:  2048,
		OSReserveBytes:     3 * 1024 * 1024 * 1024, // 3GB
	}

	head := ComputeHeadroom(hw, cfg)
	if !head.Available {
		t.Fatalf("expected head.Available to be true")
	}

	// 2 * 28 * 4 * 128 * 2 = 57,344 bytes
	wantKVBytes := uint64(57344)
	if head.ModelKVBytesPerToken != wantKVBytes {
		t.Errorf("got KVBytesPerToken %d, want %d", head.ModelKVBytesPerToken, wantKVBytes)
	}

	// 24GB - 8GB = 16GB
	wantPool := uint64(16 * 1024 * 1024 * 1024)
	if head.AvailableKVPoolBytes != wantPool {
		t.Errorf("got AvailableKVPoolBytes %d, want %d", head.AvailableKVPoolBytes, wantPool)
	}

	// Isolated: 16GB / (8192 * 57344) = 17,179,869,184 / 469,762,048 = 36 agents
	if head.MaxIsolatedAgents != 36 {
		t.Errorf("got MaxIsolatedAgents %d, want 36", head.MaxIsolatedAgents)
	}

	// Shared: (16GB - 4096*57344) / (2048*57344) = (17,179,869,184 - 234,881,024) / 117,440,512 = 144 agents
	if head.MaxSharedAgents != 144 {
		t.Errorf("got MaxSharedAgents %d, want 144", head.MaxSharedAgents)
	}

	// Concurrency advantage: 144 / 36 = 4.0
	if head.ConcurrencyAdvantage != 4.0 {
		t.Errorf("got ConcurrencyAdvantage %f, want 4.0", head.ConcurrencyAdvantage)
	}
}

func TestComputeHeadroomExceededBase(t *testing.T) {
	// 8GB system with 7GB weights + 3GB OS reserve exceeds capacity
	hw := HardwareTelemetry{
		TotalSystemMemoryBytes: 8 * 1024 * 1024 * 1024,
		WiredMemoryLimitBytes:  6 * 1024 * 1024 * 1024,
		Available:              true,
	}

	cfg := HeadroomConfig{
		Layers:            28,
		KVHeads:           4,
		HeadDim:           128,
		KVBytesPerElement: 2,
		ModelWeightBytes:  7 * 1024 * 1024 * 1024, // 7GB > 6GB limit
		ContextTokens:     8192,
		PrivateTailTokens: 2048,
		OSReserveBytes:    3 * 1024 * 1024 * 1024,
	}

	head := ComputeHeadroom(hw, cfg)
	if head.Available {
		t.Errorf("expected head.Available to be false when base memory exceeds limits")
	}
	if head.AvailableKVPoolBytes != 0 {
		t.Errorf("got AvailableKVPoolBytes %d, want 0", head.AvailableKVPoolBytes)
	}
	if head.MaxSharedAgents != 0 || head.MaxIsolatedAgents != 0 {
		t.Errorf("got agents (%d, %d), want (0, 0)", head.MaxSharedAgents, head.MaxIsolatedAgents)
	}
}

func TestComputeHeadroomDefaults(t *testing.T) {
	hw := HardwareTelemetry{
		TotalSystemMemoryBytes: 36 * 1024 * 1024 * 1024,
		Available:              true,
	}
	head := ComputeHeadroom(hw, DefaultHeadroomConfig())
	if !head.Available {
		t.Fatalf("expected head.Available with default config")
	}
	if head.MaxSharedAgents <= 0 {
		t.Errorf("expected positive MaxSharedAgents, got %d", head.MaxSharedAgents)
	}
}

func TestComputeHeadroom_MemoryLessThanOSReserve(t *testing.T) {
	// Total system memory 2GB, WiredMemoryLimitBytes unset (falls back to 75% = 1.5GB)
	// OS reserve is 3GB, model weights 1GB -> required base = 4GB > 1.5GB
	hw := HardwareTelemetry{
		TotalSystemMemoryBytes: 2 * 1024 * 1024 * 1024, // 2GB
		Available:              true,
	}
	cfg := HeadroomConfig{
		ModelWeightBytes: 1 * 1024 * 1024 * 1024,
		OSReserveBytes:   3 * 1024 * 1024 * 1024,
	}

	head := ComputeHeadroom(hw, cfg)
	if head.Available {
		t.Errorf("expected head.Available = false when physical memory < OS reserve")
	}
	if head.AvailableKVPoolBytes != 0 {
		t.Errorf("got AvailableKVPoolBytes %d, want 0", head.AvailableKVPoolBytes)
	}
	if head.MaxSharedAgents != 0 || head.MaxIsolatedAgents != 0 {
		t.Errorf("got agents (%d, %d), want (0, 0)", head.MaxSharedAgents, head.MaxIsolatedAgents)
	}
	if head.ConcurrencyAdvantage != 1.0 {
		t.Errorf("got ConcurrencyAdvantage %f, want 1.0", head.ConcurrencyAdvantage)
	}
}

func TestComputeHeadroom_ZeroKVBytesPerElement(t *testing.T) {
	hw := HardwareTelemetry{
		TotalSystemMemoryBytes: 16 * 1024 * 1024 * 1024,
		WiredMemoryLimitBytes:  12 * 1024 * 1024 * 1024,
		Available:              true,
	}
	// Pass KVBytesPerElement = 0; should fall back to default of 2 bytes
	cfg := HeadroomConfig{
		Layers:            28,
		KVHeads:           4,
		HeadDim:           128,
		KVBytesPerElement: 0, // zero input -> fallback to 2
		ModelWeightBytes:  4 * 1024 * 1024 * 1024,
		OSReserveBytes:    2 * 1024 * 1024 * 1024,
	}

	head := ComputeHeadroom(hw, cfg)
	if !head.Available {
		t.Fatalf("expected head.Available = true")
	}
	wantKVBytes := uint64(2 * 28 * 4 * 128 * 2) // 57344
	if head.ModelKVBytesPerToken != wantKVBytes {
		t.Errorf("got ModelKVBytesPerToken %d, want %d", head.ModelKVBytesPerToken, wantKVBytes)
	}
	if head.MaxSharedAgents <= 0 {
		t.Errorf("expected positive MaxSharedAgents, got %d", head.MaxSharedAgents)
	}
}

func TestComputeHeadroom_ZeroConfigFieldsFallback(t *testing.T) {
	hw := HardwareTelemetry{
		TotalSystemMemoryBytes: 32 * 1024 * 1024 * 1024,
		WiredMemoryLimitBytes:  24 * 1024 * 1024 * 1024,
		Available:              true,
	}
	// Pass all zero fields; should fall back cleanly without dividing by zero or panicking
	cfg := HeadroomConfig{}

	head := ComputeHeadroom(hw, cfg)
	if !head.Available {
		t.Fatalf("expected head.Available = true with default fallbacks")
	}
	if head.ModelKVBytesPerToken == 0 {
		t.Errorf("expected non-zero ModelKVBytesPerToken")
	}
	if head.MaxSharedAgents <= 0 {
		t.Errorf("expected positive MaxSharedAgents, got %d", head.MaxSharedAgents)
	}
	if head.SharedPrefixTokens != 0 {
		t.Errorf("got SharedPrefixTokens %d, want 0", head.SharedPrefixTokens)
	}
	if head.PrivateTailTokens != 2048 {
		t.Errorf("got PrivateTailTokens %d, want 2048", head.PrivateTailTokens)
	}
}

func TestComputeHeadroom_SharedPrefixExceedsContext(t *testing.T) {
	hw := HardwareTelemetry{
		TotalSystemMemoryBytes: 32 * 1024 * 1024 * 1024,
		WiredMemoryLimitBytes:  24 * 1024 * 1024 * 1024,
		Available:              true,
	}
	// Shared prefix greater than or equal to context tokens should be clamped to context/2
	cfg := HeadroomConfig{
		ContextTokens:      4096,
		SharedPrefixTokens: 8192, // exceeds context tokens
		PrivateTailTokens:  1024,
	}

	head := ComputeHeadroom(hw, cfg)
	if head.SharedPrefixTokens != 2048 {
		t.Errorf("got SharedPrefixTokens %d, want 2048 (clamped to context/2)", head.SharedPrefixTokens)
	}
}

func TestComputeHeadroom_OverflowProtection(t *testing.T) {
	hw := HardwareTelemetry{
		TotalSystemMemoryBytes: 64 * 1024 * 1024 * 1024,
		WiredMemoryLimitBytes:  48 * 1024 * 1024 * 1024,
		Available:              true,
	}

	// 1. ModelWeightBytes + OSReserveBytes uint64 addition overflow
	cfgAddOverflow := HeadroomConfig{
		ModelWeightBytes: ^uint64(0) - 100,
		OSReserveBytes:   1000, // wraps around if not protected
	}
	headAdd := ComputeHeadroom(hw, cfgAddOverflow)
	if headAdd.Available {
		t.Errorf("expected headAdd.Available = false on uint64 addition overflow")
	}
	if headAdd.AvailableKVPoolBytes != 0 {
		t.Errorf("got AvailableKVPoolBytes %d, want 0 on addition overflow", headAdd.AvailableKVPoolBytes)
	}

	// 2. Giant model architecture parameters causing multiplication overflow
	cfgMulOverflow := HeadroomConfig{
		Layers:            100000000,
		KVHeads:           100000000,
		HeadDim:           100000000,
		KVBytesPerElement: 2,
		ModelWeightBytes:  1 * 1024 * 1024 * 1024,
		OSReserveBytes:    1 * 1024 * 1024 * 1024,
	}
	headMul := ComputeHeadroom(hw, cfgMulOverflow)
	// Must not panic and must safely yield 0 agents
	if headMul.MaxSharedAgents != 0 || headMul.MaxIsolatedAgents != 0 {
		t.Errorf("expected 0 agents on multiplication overflow, got (%d, %d)",
			headMul.MaxSharedAgents, headMul.MaxIsolatedAgents)
	}

	// 3. Huge context tokens causing multiplication overflow with kvBytesPerToken
	cfgCtxOverflow := HeadroomConfig{
		ContextTokens:    ^uint64(0) / 10,
		ModelWeightBytes: 1 * 1024 * 1024 * 1024,
		OSReserveBytes:   1 * 1024 * 1024 * 1024,
	}
	headCtx := ComputeHeadroom(hw, cfgCtxOverflow)
	if headCtx.MaxIsolatedAgents != 0 {
		t.Errorf("expected MaxIsolatedAgents = 0 on context tokens overflow, got %d", headCtx.MaxIsolatedAgents)
	}
}

func TestComputeHeadroom_KVPoolLessThanSharedPrefix(t *testing.T) {
	// Available pool is 100MB, but shared prefix requires ~224MB
	// Should not underflow and should yield MaxSharedAgents = 0
	hw := HardwareTelemetry{
		TotalSystemMemoryBytes: 8 * 1024 * 1024 * 1024,
		WiredMemoryLimitBytes:  5*1024*1024*1024 + 100*1024*1024, // 5.1 GB
		Available:              true,
	}
	cfg := HeadroomConfig{
		Layers:             28,
		KVHeads:            4,
		HeadDim:            128,
		KVBytesPerElement:  2,
		ModelWeightBytes:   3 * 1024 * 1024 * 1024,
		OSReserveBytes:     2 * 1024 * 1024 * 1024, // Base = 5GB -> pool = 100MB
		ContextTokens:      8192,
		SharedPrefixTokens: 4096, // 4096 * 57344 = ~224MB > 100MB pool
		PrivateTailTokens:  2048,
	}

	head := ComputeHeadroom(hw, cfg)
	if !head.Available {
		t.Fatalf("expected head.Available = true when pool > 0")
	}
	if head.AvailableKVPoolBytes != 100*1024*1024 {
		t.Errorf("got AvailableKVPoolBytes %d, want %d", head.AvailableKVPoolBytes, 100*1024*1024)
	}
	if head.MaxSharedAgents != 0 {
		t.Errorf("got MaxSharedAgents %d, want 0 when pool < shared prefix", head.MaxSharedAgents)
	}
	if head.MaxIsolatedAgents != 0 {
		t.Errorf("got MaxIsolatedAgents %d, want 0", head.MaxIsolatedAgents)
	}
}

func TestComputeHeadroom_SingleTokenContext(t *testing.T) {
	hw := HardwareTelemetry{
		TotalSystemMemoryBytes: 16 * 1024 * 1024 * 1024,
		WiredMemoryLimitBytes:  12 * 1024 * 1024 * 1024,
		Available:              true,
	}
	// Smallest valid context: ContextTokens 2, SharedPrefix 1, PrivateTail 1
	cfg := HeadroomConfig{
		Layers:             28,
		KVHeads:            4,
		HeadDim:            128,
		KVBytesPerElement:  2,
		ModelWeightBytes:   4 * 1024 * 1024 * 1024,
		OSReserveBytes:     2 * 1024 * 1024 * 1024, // Pool = 6GB
		ContextTokens:      2,
		SharedPrefixTokens: 1,
		PrivateTailTokens:  1,
	}

	head := ComputeHeadroom(hw, cfg)
	if !head.Available {
		t.Fatalf("expected head.Available = true")
	}
	if head.MaxSharedAgents <= 0 || head.MaxIsolatedAgents <= 0 {
		t.Errorf("expected positive agents for single-token context, got (%d, %d)",
			head.MaxSharedAgents, head.MaxIsolatedAgents)
	}
}
