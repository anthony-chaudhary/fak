package vllmcompile

import "testing"

func BenchmarkBlockClassify(b *testing.B) {
	block := Block{
		Engine:              "vllm",
		EngineCommit:        "28242824e",
		CompileCacheEnabled: Bool(true),
		CompileCacheKey:     "key-12345",
		CUDAGraphMode:       "full",
		CaptureSizes:        []int{1, 2, 4, 8},
		WarmupComplete:      Bool(true),
	}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = block.Classify()
	}
}

func BenchmarkBlockGate(b *testing.B) {
	block := Block{
		Engine:              "vllm",
		CompileCacheEnabled: Bool(true),
		WarmupComplete:      Bool(true),
	}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = block.Gate()
	}
}

func BenchmarkGateRows(b *testing.B) {
	rows := []Block{
		{Engine: "vllm", CompileCacheEnabled: Bool(true), WarmupComplete: Bool(true)},
		{Engine: "vllm+fak", CompileCacheEnabled: Bool(true), WarmupComplete: Bool(true)},
		{Engine: "sglang", CompileCacheEnabled: Bool(true), WarmupComplete: Bool(true)},
	}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = GateRows(rows...)
	}
}

func BenchmarkNormalizeArch(b *testing.B) {
	inputs := []string{"Hopper (sm_90)", "9.0", "sm_100", "8.0"}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = NormalizeArch(inputs[i%len(inputs)])
	}
}

func BenchmarkCacheTupleKey(b *testing.B) {
	tuple := CacheTuple{
		Model:     "PhalaCloud/GLM-5.2-W4AFP8",
		Quant:     "w4afp8",
		Arch:      "Hopper (sm_90)",
		TP:        8,
		Ctx:       65536,
		Engine:    "sglang",
		EngineVer: "0.5.14",
		TorchVer:  "2.11.0+cu130",
	}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = tuple.Key()
	}
}

func BenchmarkReadout(b *testing.B) {
	key := "phalacloud-glm-5-2-w4afp8/w4afp8/sm90/tp8/ctx65536/sglang-0-5-14-torch2-11-0-cu130"
	dir := "/mnt/compile-cache/" + key
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = Readout(key, dir, true)
	}
}

func BenchmarkWithCacheTuple(b *testing.B) {
	tuple := CacheTuple{
		Model:     "zai-org/GLM-5.2-FP8",
		Quant:     "fp8",
		Arch:      "sm_90",
		TP:        8,
		Ctx:       65536,
		Engine:    "sglang",
		EngineVer: "0.5.14",
		TorchVer:  "2.11.0",
	}
	base := Block{Engine: "sglang", WarmupComplete: Bool(true)}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = base.WithCacheTuple(tuple, true)
	}
}
