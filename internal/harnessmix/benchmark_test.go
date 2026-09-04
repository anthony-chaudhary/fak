package harnessmix

import (
	"fmt"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/harnesscompose"
	"github.com/anthony-chaudhary/fak/internal/harnessresolve"
)

// BenchmarkHarnessMix measures end-to-end performance of mixing two verified harness locks
// with shared components, distinct capability requirements, and merged policy assets.
func BenchmarkHarnessMix(b *testing.B) {
	shared := benchmarkComponent("kernel", []string{"runtime"}, nil, harnessresolve.Budget{ContextTokens: 200, MemoryMiB: 64, Workers: 1})
	compA := benchmarkComponent("agent-support", []string{"support"}, []harnessresolve.Requirement{{Capability: "runtime"}}, harnessresolve.Budget{ContextTokens: 400, MemoryMiB: 128, Workers: 2})
	compB := benchmarkComponent("agent-research", []string{"research"}, []harnessresolve.Requirement{{Capability: "runtime"}}, harnessresolve.Budget{ContextTokens: 500, MemoryMiB: 256, Workers: 2})

	assetsA := []harnesscompose.EffectiveAsset{
		{Kind: "instruction", ID: "support-prompt", Value: "assist user queries", Source: "lock-a"},
		{Kind: "policy", ID: "security-floor", Grants: []string{"read"}, Denies: []string{"exec"}, Locked: true, Mandatory: true, Source: "lock-a"},
	}
	assetsB := []harnesscompose.EffectiveAsset{
		{Kind: "instruction", ID: "research-prompt", Value: "conduct literature search", Source: "lock-b"},
		{Kind: "policy", ID: "security-floor", Grants: []string{"read"}, Denies: []string{"write"}, Locked: true, Mandatory: true, Source: "lock-b"},
	}

	lockA := benchmarkLock(b, "support", []harnessresolve.LockedComponent{shared, compA}, assetsA)
	lockB := benchmarkLock(b, "research", []harnessresolve.LockedComponent{shared, compB}, assetsB)
	imports := []harnessresolve.Lock{lockA, lockB}
	limits := Limits{ContextTokens: 2000, MemoryMiB: 1024, Workers: 10}

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		res, err := Mix(imports, limits)
		if err != nil {
			b.Fatalf("mix failed: %v", err)
		}
		if res.Lock.ID == "" {
			b.Fatal("empty lock id")
		}
	}
}

// BenchmarkHarnessMixScaling measures mix performance as the number of imported locks scales.
func BenchmarkHarnessMixScaling(b *testing.B) {
	sharedKernel := benchmarkComponent("kernel", []string{"runtime"}, nil, harnessresolve.Budget{ContextTokens: 100, MemoryMiB: 32, Workers: 1})
	sharedLogger := benchmarkComponent("logger", []string{"logging"}, nil, harnessresolve.Budget{ContextTokens: 50, MemoryMiB: 16, Workers: 1})

	const lockCount = 5
	imports := make([]harnessresolve.Lock, lockCount)
	for i := 0; i < lockCount; i++ {
		compID := fmt.Sprintf("worker-service-%d", i)
		capID := fmt.Sprintf("service-%d", i)
		comp := benchmarkComponent(compID, []string{capID}, []harnessresolve.Requirement{{Capability: "runtime"}}, harnessresolve.Budget{ContextTokens: 150, MemoryMiB: 64, Workers: 1})
		asset := harnesscompose.EffectiveAsset{
			Kind:   "instruction",
			ID:     fmt.Sprintf("instruction-%d", i),
			Value:  fmt.Sprintf("run worker %d", i),
			Source: fmt.Sprintf("lock-%d", i),
		}
		imports[i] = benchmarkLock(b, fmt.Sprintf("lock-%d", i), []harnessresolve.LockedComponent{sharedKernel, sharedLogger, comp}, []harnesscompose.EffectiveAsset{asset})
	}
	limits := Limits{ContextTokens: 5000, MemoryMiB: 4096, Workers: 20}

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		res, err := Mix(imports, limits)
		if err != nil {
			b.Fatalf("scaled mix failed: %v", err)
		}
		if len(res.Lock.Components) != lockCount+2 {
			b.Fatalf("expected %d components, got %d", lockCount+2, len(res.Lock.Components))
		}
	}
}

// BenchmarkHarnessMixDeduplication measures throughput when resolving large sets of overlapping components.
func BenchmarkHarnessMixDeduplication(b *testing.B) {
	const sharedCount = 20
	shared := make([]harnessresolve.LockedComponent, sharedCount)
	for i := 0; i < sharedCount; i++ {
		id := fmt.Sprintf("shared-lib-%02d", i)
		shared[i] = benchmarkComponent(id, []string{id}, nil, harnessresolve.Budget{ContextTokens: 10, MemoryMiB: 8, Workers: 1})
	}

	compA := benchmarkComponent("app-a", []string{"service-a"}, nil, harnessresolve.Budget{ContextTokens: 100})
	compB := benchmarkComponent("app-b", []string{"service-b"}, nil, harnessresolve.Budget{ContextTokens: 100})

	compsA := append(append([]harnessresolve.LockedComponent(nil), shared...), compA)
	compsB := append(append([]harnessresolve.LockedComponent(nil), shared...), compB)

	lockA := benchmarkLock(b, "lock-a", compsA, nil)
	lockB := benchmarkLock(b, "lock-b", compsB, nil)
	imports := []harnessresolve.Lock{lockA, lockB}
	limits := Limits{ContextTokens: 10000, MemoryMiB: 10000, Workers: 100}

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		res, err := Mix(imports, limits)
		if err != nil {
			b.Fatalf("dedup mix failed: %v", err)
		}
		if len(res.Receipt.Deduplicated) != sharedCount {
			b.Fatalf("expected %d deduplicated, got %d", sharedCount, len(res.Receipt.Deduplicated))
		}
	}
}

// TestBenchmarkHarnessMixSanity verifies that BenchmarkHarnessMix executes without error.
func TestBenchmarkHarnessMixSanity(t *testing.T) {
	res := testing.Benchmark(BenchmarkHarnessMix)
	if res.N <= 0 {
		t.Fatalf("expected benchmark iterations > 0, got %d", res.N)
	}
}

func benchmarkComponent(id string, provides []string, requires []harnessresolve.Requirement, cost harnessresolve.Budget) harnessresolve.LockedComponent {
	return harnessresolve.LockedComponent{
		ID:            id,
		Version:       "1.0.0",
		Digest:        "sha256:" + id,
		Source:        "registry/" + id,
		Reason:        "selected import",
		Provider:      id,
		Provides:      provides,
		Requires:      requires,
		Compatibility: harnessresolve.Compatibility{OS: []string{"linux"}, Arch: []string{"amd64"}, Contract: "v1"},
		Cost:          cost,
		Adapters:      []string{"instruction"},
	}
}

func benchmarkLock(b testing.TB, name string, cs []harnessresolve.LockedComponent, assets []harnesscompose.EffectiveAsset) harnessresolve.Lock {
	b.Helper()
	l := harnessresolve.Lock{
		Schema:      harnessresolve.LockSchema,
		Environment: harnessresolve.Environment{OS: "linux", Arch: "amd64", Contract: "v1"},
		Components:  cs,
		Assets:      assets,
	}
	if err := harnessresolve.ReidentifyLock(&l); err != nil {
		b.Fatalf("reidentify lock %s: %v", name, err)
	}
	return l
}
