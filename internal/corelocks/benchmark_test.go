// Package corelocks benchmarks measure parsing, classification throughput,
// pattern specificity scoring, and root authority census validation for
// declarative core locks.
//
// Classification is on the critical path of file admission and agent dispatch,
// requiring zero allocations and sub-microsecond latency on hot classification
// paths to prevent scheduler latency spikes.
package corelocks

import (
	"os"
	"path/filepath"
	"testing"
)

var (
	benchTaxonomySink    *Taxonomy
	benchClassSink       string
	benchReasonSink      string
	benchBoolSink        bool
	benchIntSink         int
	benchVerdictSink     RootVerdict
	benchCensusSink      Census
	benchStringSink      string
	benchStringsSink     []string
	benchErrSink         error
	benchDestructiveSink []DestructiveReader
)

// BenchmarkParse_Fixture measures parsing and structural validation of the canonical corelocks fixture.
// Operating envelope: standard embedded TOML declaration (~1 KB payload).
// Allocation budget: <= 40 allocs/op and <= 3 KB/op for AST tokenization and taxonomy allocation.
// Latency ceiling: P50 < 10µs, P99 < 50µs with linear scaling on declaration byte size.
func BenchmarkParse_Fixture(b *testing.B) {
	data := Fixture()
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		t, err := Parse(data)
		if err != nil {
			b.Fatal(err)
		}
		benchTaxonomySink = t
	}
}

// BenchmarkParse_MultiClass measures parsing of a 5-class table declaration with varied globs and reasons.
// Operating envelope: multi-class TOML payload (~600 bytes) with 5 distinct lock classes.
// Allocation budget: <= 35 allocs/op and <= 3 KB/op.
// Latency ceiling: P50 < 8µs, P99 < 40µs.
func BenchmarkParse_MultiClass(b *testing.B) {
	data := []byte(`
# Declarative core-lock declaration benchmark
[[class]]
name   = "hard-self"
reason = "CORE_SELF_MODIFY"
globs  = ["internal/adjudicator/**", "internal/abi/**", "internal/corelocks/**"]

[[class]]
name   = "serial-core"
reason = "CORE_SERIAL_REQUIRED"
globs  = ["dos.toml", "internal/resume/**", "cmd/fak/loop_*.go"]

[[class]]
name   = "soft-contract"
reason = "CORE_CONTRACT_WITNESS_MISSING"
globs  = ["internal/canon/**", "internal/covmatrix/**"]

[[class]]
name   = "shadow-learn"
reason = "CORE_LOCK_UNCLASSIFIED"
globs  = ["internal/rsiloop/**"]

[[class]]
name   = "open-leaf"
reason = ""
globs  = []
`)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		t, err := Parse(data)
		if err != nil {
			b.Fatal(err)
		}
		benchTaxonomySink = t
	}
}

// BenchmarkClassify_HitHardSelf evaluates path classification hitting the hard-self lock class.
// Operating envelope: single repo-relative path matching directory glob "internal/adjudicator/**".
// Allocation budget: 0 allocs/op on the classification fast path.
// Latency ceiling: P50 < 100ns, P99 < 500ns with constant-time glob matching.
func BenchmarkClassify_HitHardSelf(b *testing.B) {
	tax, err := LoadFixture()
	if err != nil {
		b.Fatal(err)
	}
	p := "internal/adjudicator/decide.go"
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		c, r := tax.Classify(p)
		benchClassSink = c
		benchReasonSink = r
	}
}

// BenchmarkClassify_HitSerialCoreExact evaluates classification on an exact file glob match.
// Operating envelope: single path "dos.toml" matching the exact glob in serial-core.
// Allocation budget: 0 allocs/op.
// Latency ceiling: P50 < 80ns, P99 < 400ns.
func BenchmarkClassify_HitSerialCoreExact(b *testing.B) {
	tax, err := LoadFixture()
	if err != nil {
		b.Fatal(err)
	}
	p := "dos.toml"
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		c, r := tax.Classify(p)
		benchClassSink = c
		benchReasonSink = r
	}
}

// BenchmarkClassify_HitSoftContract evaluates path classification hitting the soft-contract class.
// Operating envelope: single repo-relative path matching directory glob "internal/canon/**".
// Allocation budget: 0 allocs/op.
// Latency ceiling: P50 < 100ns, P99 < 500ns.
func BenchmarkClassify_HitSoftContract(b *testing.B) {
	tax, err := LoadFixture()
	if err != nil {
		b.Fatal(err)
	}
	p := "internal/canon/canon.go"
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		c, r := tax.Classify(p)
		benchClassSink = c
		benchReasonSink = r
	}
}

// BenchmarkClassify_OpenLeafFallthrough evaluates fallthrough behavior when no declared glob matches.
// Operating envelope: unmatched path "cmd/fak/main.go" falling through to open-leaf.
// Allocation budget: 0 allocs/op.
// Latency ceiling: P50 < 120ns, P99 < 600ns after exhausting declared class globs.
func BenchmarkClassify_OpenLeafFallthrough(b *testing.B) {
	tax, err := LoadFixture()
	if err != nil {
		b.Fatal(err)
	}
	p := "cmd/fak/main.go"
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		c, r := tax.Classify(p)
		benchClassSink = c
		benchReasonSink = r
	}
}

// BenchmarkClassify_BatchPaths measures throughput across a mixed batch of 10 repository paths.
// Operating envelope: cyclic evaluation over 10 diverse paths hitting multiple classes and fallthrough.
// Allocation budget: 0 allocs/op across all evaluated paths.
// Latency ceiling: P50 < 100ns, P99 < 500ns per path query.
func BenchmarkClassify_BatchPaths(b *testing.B) {
	tax, err := LoadFixture()
	if err != nil {
		b.Fatal(err)
	}
	paths := []string{
		"internal/adjudicator/decide.go",
		"internal/abi/registry.go",
		"dos.toml",
		"internal/resume/engine.go",
		"internal/canon/canon.go",
		"internal/covmatrix/matrix.go",
		"internal/rsiloop/loop.go",
		"cmd/fak/main.go",
		"docs/readme.md",
		"internal/corelocks/corelocks.go",
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		p := paths[i%len(paths)]
		c, r := tax.Classify(p)
		benchClassSink = c
		benchReasonSink = r
	}
}

// BenchmarkPathUnderGlob_Containment measures hierarchical path containment evaluation.
// Operating envelope: double-wildcard glob "internal/adjudicator/**" against target file.
// Allocation budget: 0 allocs/op.
// Latency ceiling: P50 < 30ns, P99 < 150ns.
func BenchmarkPathUnderGlob_Containment(b *testing.B) {
	glob := "internal/adjudicator/**"
	target := "internal/adjudicator/decide.go"
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		benchBoolSink = pathUnderGlob(glob, target)
	}
}

// BenchmarkPathUnderGlob_Exact measures exact string match evaluation in pathUnderGlob.
// Operating envelope: literal glob "dos.toml" against identical target string.
// Allocation budget: 0 allocs/op.
// Latency ceiling: P50 < 20ns, P99 < 100ns.
func BenchmarkPathUnderGlob_Exact(b *testing.B) {
	glob := "dos.toml"
	target := "dos.toml"
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		benchBoolSink = pathUnderGlob(glob, target)
	}
}

// BenchmarkGlobSpecificity measures specificity score calculation for pattern precedence.
// Operating envelope: cyclic scoring over 5 representative glob patterns.
// Allocation budget: 0 allocs/op.
// Latency ceiling: P50 < 15ns, P99 < 80ns.
func BenchmarkGlobSpecificity(b *testing.B) {
	globs := []string{
		"**",
		"internal/**",
		"internal/adjudicator/**",
		"dos.toml",
		"cmd/fak/*.go",
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		g := globs[i%len(globs)]
		benchIntSink = globSpecificity(g)
	}
}

// BenchmarkCheckRoot_Authoritative measures workspace root authority checks when matching git top.
// Operating envelope: authoritative root directory containing .dos state directory.
// Allocation budget: <= 5 allocs/op and <= 256 B/op for filesystem stat operations.
// Latency ceiling: P50 < 5µs, P99 < 25µs dominated by OS directory inspection.
func BenchmarkCheckRoot_Authoritative(b *testing.B) {
	top := b.TempDir()
	if err := os.MkdirAll(filepath.Join(top, StateDir), 0o755); err != nil {
		b.Fatal(err)
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		benchVerdictSink = CheckRoot(top, top)
	}
}

// BenchmarkCheckRoot_DeepSubdirectory measures root authority checks from a 4-level nested subdirectory.
// Operating envelope: deep subdirectory "a/b/c/d" traversing up to the authoritative root.
// Allocation budget: <= 8 allocs/op and <= 384 B/op.
// Latency ceiling: P50 < 8µs, P99 < 40µs.
func BenchmarkCheckRoot_DeepSubdirectory(b *testing.B) {
	top := b.TempDir()
	if err := os.MkdirAll(filepath.Join(top, StateDir), 0o755); err != nil {
		b.Fatal(err)
	}
	sub := filepath.Join(top, "a", "b", "c", "d")
	if err := os.MkdirAll(sub, 0o755); err != nil {
		b.Fatal(err)
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		benchVerdictSink = CheckRoot(sub, top)
	}
}

// BenchmarkResolveRoot measures filesystem upward traversal to resolve the authoritative workspace root.
// Operating envelope: nested path "cmd/fak/agent" resolving upwards to top-level .dos.
// Allocation budget: <= 12 allocs/op and <= 512 B/op.
// Latency ceiling: P50 < 10µs, P99 < 50µs.
func BenchmarkResolveRoot(b *testing.B) {
	top := b.TempDir()
	if err := os.MkdirAll(filepath.Join(top, StateDir), 0o755); err != nil {
		b.Fatal(err)
	}
	sub := filepath.Join(top, "cmd", "fak", "agent")
	if err := os.MkdirAll(sub, 0o755); err != nil {
		b.Fatal(err)
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		root, ok := ResolveRoot(sub)
		benchStringSink = root
		benchBoolSink = ok
	}
}

// BenchmarkReadCensus_Authoritative measures census acquisition with a live counter callback.
// Operating envelope: authoritative root directory with injected constant counter.
// Allocation budget: <= 6 allocs/op and <= 256 B/op.
// Latency ceiling: P50 < 6µs, P99 < 30µs.
func BenchmarkReadCensus_Authoritative(b *testing.B) {
	top := b.TempDir()
	if err := os.MkdirAll(filepath.Join(top, StateDir), 0o755); err != nil {
		b.Fatal(err)
	}
	counter := func(root string) (int, error) {
		return 5, nil
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		c, err := ReadCensus(top, top, counter)
		if err != nil {
			b.Fatal(err)
		}
		benchCensusSink = c
	}
}

// BenchmarkNewCensus_Authoritative measures constructing a Census record for an authoritative verdict.
// Operating envelope: authoritative RootVerdict with 7 held locks.
// Allocation budget: 0 allocs/op.
// Latency ceiling: P50 < 20ns, P99 < 100ns.
func BenchmarkNewCensus_Authoritative(b *testing.B) {
	v := RootVerdict{
		Start:         "/work/fak",
		Resolved:      "/work/fak",
		GitTop:        "/work/fak",
		Authoritative: true,
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		c, err := NewCensus(v, 7)
		if err != nil {
			b.Fatal(err)
		}
		benchCensusSink = c
	}
}

// BenchmarkNewCensus_Refused measures constructing a Census record for a shadowed refusal verdict.
// Operating envelope: shadowed RootVerdict returning a typed refusal error.
// Allocation budget: <= 2 allocs/op and <= 128 B/op for formatted error creation.
// Latency ceiling: P50 < 150ns, P99 < 800ns.
func BenchmarkNewCensus_Refused(b *testing.B) {
	v := RootVerdict{
		Start:         "/work/fak/docs",
		Resolved:      "/work/fak/docs",
		GitTop:        "/work/fak",
		Authoritative: false,
		Shadowed:      true,
		Cause:         "a SHADOW .dos root inside the repository shadows the real one",
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		c, err := NewCensus(v, 0)
		if err == nil {
			b.Fatal("expected refusal")
		}
		benchCensusSink = c
		benchErrSink = err
	}
}

// BenchmarkCensus_Line_Authoritative measures line rendering of an authoritative Census result.
// Operating envelope: authoritative Census record with held count 12.
// Allocation budget: <= 2 allocs/op and <= 64 B/op for string rendering.
// Latency ceiling: P50 < 100ns, P99 < 500ns.
func BenchmarkCensus_Line_Authoritative(b *testing.B) {
	c := Census{
		Root:          "/work/fak",
		Start:         "/work/fak",
		GitTop:        "/work/fak",
		Held:          12,
		Counted:       true,
		Authoritative: true,
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		benchStringSink = c.Line()
	}
}

// BenchmarkCensus_Line_RefusedShadow measures line rendering of a shadowed root refusal.
// Operating envelope: shadowed Census record with refusal reason.
// Allocation budget: <= 2 allocs/op and <= 128 B/op.
// Latency ceiling: P50 < 120ns, P99 < 600ns.
func BenchmarkCensus_Line_RefusedShadow(b *testing.B) {
	c := Census{
		Root:          "/work/fak/docs",
		Start:         "/work/fak/docs",
		GitTop:        "/work/fak",
		Held:          -1,
		Counted:       false,
		Authoritative: false,
		Shadowed:      true,
		Cause:         "a SHADOW .dos root inside the repository shadows the real one",
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		benchStringSink = c.Line()
	}
}

// BenchmarkCensus_Line_RefusedNoRoot measures line rendering when no workspace root exists.
// Operating envelope: missing root Census record with cause description.
// Allocation budget: <= 2 allocs/op and <= 128 B/op.
// Latency ceiling: P50 < 120ns, P99 < 600ns.
func BenchmarkCensus_Line_RefusedNoRoot(b *testing.B) {
	c := Census{
		Root:          "",
		Start:         "/tmp/sandbox",
		GitTop:        "/tmp/sandbox",
		Held:          -1,
		Counted:       false,
		Authoritative: false,
		Shadowed:      false,
		Cause:         "no .dos directory at or above /tmp/sandbox",
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		benchStringSink = c.Line()
	}
}

// BenchmarkUnanchoredStateIgnores measures scanning gitignore rules for unanchored state directory patterns.
// Operating envelope: representative gitignore file (~350 bytes) containing positive and negative patterns.
// Allocation budget: <= 8 allocs/op and <= 512 B/op for matched pattern slices.
// Latency ceiling: P50 < 1µs, P99 < 5µs.
func BenchmarkUnanchoredStateIgnores(b *testing.B) {
	gitignore := []byte(`
# Compiled binaries
*.exe
*.test
/bin/

# State roots
/.dos/
**/.dos/
.dos/
!.dos/
testdata/.dos/

# Editor and temporary files
*.swp
*~
.DS_Store
`)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		benchStringsSink = UnanchoredStateIgnores(gitignore, StateDir)
	}
}

// BenchmarkDestructiveReaders measures retrieval of registered destructive reader descriptors.
// Operating envelope: static slice of destructive tool reader configurations.
// Allocation budget: <= 1 allocs/op and <= 128 B/op.
// Latency ceiling: P50 < 30ns, P99 < 150ns.
func BenchmarkDestructiveReaders(b *testing.B) {
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		benchDestructiveSink = DestructiveReaders()
	}
}

// BenchmarkUnderDir measures filesystem path prefix containment checking.
// Operating envelope: path pair comparing target path against base workspace path.
// Allocation budget: 0 allocs/op.
// Latency ceiling: P50 < 50ns, P99 < 250ns.
func BenchmarkUnderDir(b *testing.B) {
	base := filepath.Join("work", "repo")
	target := filepath.Join("work", "repo", "internal", "adjudicator", "decide.go")
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		benchBoolSink = underDir(target, base)
	}
}
