package vllmcompile

import "testing"

// TestNormalizeArch is the executable form of #3052 acceptance bullet 2: the
// cache key must distinguish GPU arch so a GPU-server sm_80 cache can never mis-hit
// an H100 Mega sm_90 boot. Every spelling the preflight/nvidia-smi/human may hand
// us collapses to the same bare digit token, and unknown input fails safe.
func TestNormalizeArch(t *testing.T) {
	cases := map[string]string{
		"9.0":                "90",
		"sm_90":              "90",
		"sm90":               "90",
		"Hopper (sm_90)":     "90",
		"10.0":               "100",
		"sm_100":             "100",
		"Blackwell (sm_100)": "100",
		"8.0":                "80",
		"90":                 "90",
		"":                   "unknown",
		"n/a":                "unknown",
	}
	for in, want := range cases {
		if got := NormalizeArch(in); got != want {
			t.Errorf("NormalizeArch(%q) = %q, want %q", in, got, want)
		}
	}
}

// TestCacheKeyDistinguishesTuple is the executable form of #3052 acceptance
// bullet 2 + proposed-fix bullet 3: any tuple field change must yield a
// different directory (structural invalidation), and the key must be
// filesystem-safe for real checkpoint/version strings.
func TestCacheKeyDistinguishesTuple(t *testing.T) {
	base := CacheTuple{
		Model:     "PhalaCloud/GLM-5.2-W4AFP8",
		Quant:     "w4afp8",
		Arch:      "Hopper (sm_90)",
		TP:        8,
		Ctx:       65536,
		Engine:    "sglang",
		EngineVer: "0.5.14",
		TorchVer:  "2.11.0+cu130",
	}
	want := "phalacloud-glm-5-2-w4afp8/w4afp8/sm90/tp8/ctx65536/sglang-0-5-14-torch2-11-0-cu130"
	if got := base.Key(); got != want {
		t.Fatalf("Key() = %q, want %q", got, want)
	}

	// Every field is load-bearing: changing any one must move the directory so a
	// stale cache can never silently mis-serve.
	mutate := map[string]func(c *CacheTuple){
		"model":     func(c *CacheTuple) { c.Model = "zai-org/GLM-5.2-FP8" },
		"quant":     func(c *CacheTuple) { c.Quant = "fp8" },
		"arch":      func(c *CacheTuple) { c.Arch = "sm_80" },
		"tp":        func(c *CacheTuple) { c.TP = 4 },
		"ctx":       func(c *CacheTuple) { c.Ctx = 131072 },
		"engine":    func(c *CacheTuple) { c.Engine = "vllm" },
		"engineVer": func(c *CacheTuple) { c.EngineVer = "0.6.0" },
		"torchVer":  func(c *CacheTuple) { c.TorchVer = "2.12.0+cu130" },
	}
	for name, mut := range mutate {
		c := base
		mut(&c)
		if c.Key() == base.Key() {
			t.Errorf("changing %s did not change the cache key (%q)", name, c.Key())
		}
	}
}

// TestReadout pins the exact operator preflight line so the bash wrapper and the
// Go seam stay byte-identical.
func TestReadout(t *testing.T) {
	key := "phalacloud-glm-5-2-w4afp8/w4afp8/sm90/tp8/ctx65536/sglang-0-5-14-torch2-11-0-cu130"
	dir := "/mnt/compile-cache/" + key
	if got, want := Readout(key, dir, true), "COMPILE_CACHE hit dir="+dir+" tuple="+key; got != want {
		t.Errorf("Readout(hit) = %q, want %q", got, want)
	}
	if got, want := Readout(key, dir, false), "COMPILE_CACHE rebuild dir="+dir+" tuple="+key; got != want {
		t.Errorf("Readout(rebuild) = %q, want %q", got, want)
	}
}

// TestWithCacheTuple checks the L3 seam: a populated keyed dir stamps an
// enabled cache + key, and an empty one records a disabled (cold) cache — so the
// bench-honesty gate reads the same state the operator readout showed.
func TestWithCacheTuple(t *testing.T) {
	tup := CacheTuple{Model: "zai-org/GLM-5.2-FP8", Quant: "fp8", Arch: "sm_90", TP: 8, Ctx: 65536, Engine: "sglang", EngineVer: "0.5.14", TorchVer: "2.11.0"}

	warm := Block{Engine: "sglang", WarmupComplete: Bool(true)}.WithCacheTuple(tup, true)
	if warm.CompileCacheKey != tup.Key() {
		t.Errorf("warm key = %q, want %q", warm.CompileCacheKey, tup.Key())
	}
	if warm.CompileCacheEnabled == nil || !*warm.CompileCacheEnabled {
		t.Error("populated dir should record CompileCacheEnabled=true")
	}
	if !warm.Tuned() {
		t.Errorf("warm+populated block should be tuned, got %s (%s)", warm.Classify(), warm.Reason())
	}

	cold := Block{Engine: "sglang", WarmupComplete: Bool(true)}.WithCacheTuple(tup, false)
	if cold.CompileCacheEnabled == nil || *cold.CompileCacheEnabled {
		t.Error("empty dir should record CompileCacheEnabled=false")
	}
	if cold.Tuned() {
		t.Error("cold rebuild boot must not certify a tuned baseline")
	}
}
