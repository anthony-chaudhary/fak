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

func BenchmarkPathUnderGlob_Containment(b *testing.B) {
	glob := "internal/adjudicator/**"
	target := "internal/adjudicator/decide.go"
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		benchBoolSink = pathUnderGlob(glob, target)
	}
}

func BenchmarkPathUnderGlob_Exact(b *testing.B) {
	glob := "dos.toml"
	target := "dos.toml"
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		benchBoolSink = pathUnderGlob(glob, target)
	}
}

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

func BenchmarkDestructiveReaders(b *testing.B) {
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		benchDestructiveSink = DestructiveReaders()
	}
}

func BenchmarkUnderDir(b *testing.B) {
	base := filepath.Join("work", "repo")
	target := filepath.Join("work", "repo", "internal", "adjudicator", "decide.go")
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		benchBoolSink = underDir(target, base)
	}
}
