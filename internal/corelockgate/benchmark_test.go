package corelockgate

import (
	"bytes"
	"context"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/abi"
	"github.com/anthony-chaudhary/fak/internal/corelockaudit"
)

var (
	benchSinkDetail      string
	benchSinkFired       bool
	benchSinkFinding     corelockaudit.Finding
	benchSinkCorrelation WitnessCorrelation
	benchSinkGrammar     BinaryGrammar
	benchSinkOutcome     abi.WitnessOutcome
	benchSinkCause       string
)

// BenchmarkCheckCoreLockHardSelf measures the primary admission gate across open leaf,
// changed-witness fast-path, external resolver, missing-witness refusal, and mixed pathsets.
func BenchmarkCheckCoreLockHardSelf(b *testing.B) {
	ctx := context.Background()

	b.Run("OpenLeaf", func(b *testing.B) {
		req := CoreLockCheck{
			Changed: []string{ordinaryLeaf},
		}
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			benchSinkDetail, benchSinkFired = CheckCoreLockHardSelf(ctx, req)
		}
	})

	b.Run("ChangedWitnessConfirmed", func(b *testing.B) {
		req := CoreLockCheck{
			Changed: []string{lockedPath},
			Witness: ChangedWitnessKind + ":" + lockedPath,
		}
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			benchSinkDetail, benchSinkFired = CheckCoreLockHardSelf(ctx, req)
		}
	})

	b.Run("ResolverConfirmed", func(b *testing.B) {
		req := CoreLockCheck{
			Changed:  []string{lockedPath},
			Witness:  "commit:0f1e2d3c4b5a",
			Resolver: fixedResolver{outcome: abi.WitnessConfirmed},
			Observe:  func(c WitnessCorrelation) { benchSinkCorrelation = c },
		}
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			benchSinkDetail, benchSinkFired = CheckCoreLockHardSelf(ctx, req)
		}
	})

	b.Run("MissingWitnessRefusal", func(b *testing.B) {
		req := CoreLockCheck{
			Changed: []string{lockedPath},
			Witness: "",
		}
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			benchSinkDetail, benchSinkFired = CheckCoreLockHardSelf(ctx, req)
		}
	})

	b.Run("MixedPathset", func(b *testing.B) {
		paths := []string{
			"cmd/fak/main.go",
			"internal/corelockgate/corelockgate.go",
			"internal/tools/a.go",
			"internal/adjudicator/decide.go",
			"docs/README.md",
			"internal/corelocks/corelocks.go",
		}
		req := CoreLockCheck{
			Changed: paths,
			Witness: ChangedWitnessKind + ":internal/adjudicator/decide.go",
		}
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			benchSinkDetail, benchSinkFired = CheckCoreLockHardSelf(ctx, req)
		}
	})
}

// BenchmarkHardSelfFinding measures pathset taxonomy classification against embedded fixtures.
func BenchmarkHardSelfFinding(b *testing.B) {
	pathsSingle := []string{lockedPath}
	pathsMixed := []string{
		"cmd/fak/main.go",
		"internal/corelockgate/corelockgate.go",
		"internal/tools/a.go",
		"internal/adjudicator/decide.go",
		"docs/README.md",
	}

	b.Run("SingleLocked", func(b *testing.B) {
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			benchSinkFinding, benchSinkFired = HardSelfFinding(pathsSingle)
		}
	})

	b.Run("MixedPathset", func(b *testing.B) {
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			benchSinkFinding, benchSinkFired = HardSelfFinding(pathsMixed)
		}
	})
}

// BenchmarkCorrelateWitness measures witness-to-change path correlation performance.
func BenchmarkCorrelateWitness(b *testing.B) {
	changed := []string{
		"internal/adjudicator/oot_mention.go",
		"internal/adjudicator/oot_mention_test.go",
		"internal/adjudicator/outoftree.go",
	}

	b.Run("ExactMatch", func(b *testing.B) {
		claim := "committed:internal/adjudicator/outoftree.go"
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			benchSinkCorrelation = CorrelateWitness(claim, changed)
		}
	})

	b.Run("DirectoryMatch", func(b *testing.B) {
		claim := "committed:internal/adjudicator"
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			benchSinkCorrelation = CorrelateWitness(claim, changed)
		}
	})

	b.Run("Uncorrelated", func(b *testing.B) {
		claim := "committed:README.md"
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			benchSinkCorrelation = CorrelateWitness(claim, changed)
		}
	})

	b.Run("IndeterminateHistory", func(b *testing.B) {
		claim := "ancestor:HEAD"
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			benchSinkCorrelation = CorrelateWitness(claim, changed)
		}
	})
}

// BenchmarkScanBinaryGrammar measures streaming executable inspection for grammar markers.
func BenchmarkScanBinaryGrammar(b *testing.B) {
	currentData := synthBinary(CoreLockRemedyCommit)
	staleData := synthBinary("Use a privileged maintenance path, or rerun fak commit with --core-lock-maintenance-witness <claim>")
	unknownData := synthBinary("MZ\x90\x00 arbitrary binary payload without markers")

	b.Run("CurrentMarker", func(b *testing.B) {
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			benchSinkGrammar, _ = ScanBinaryGrammar(bytes.NewReader(currentData))
		}
	})

	b.Run("StaleMarker", func(b *testing.B) {
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			benchSinkGrammar, _ = ScanBinaryGrammar(bytes.NewReader(staleData))
		}
	})

	b.Run("UnknownMarker", func(b *testing.B) {
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			benchSinkGrammar, _ = ScanBinaryGrammar(bytes.NewReader(unknownData))
		}
	})
}

// BenchmarkResolveChangedWitness measures in-gate changed: witness resolution against pathsets.
func BenchmarkResolveChangedWitness(b *testing.B) {
	changed := []string{
		"internal/adjudicator/oot_mention.go",
		"internal/adjudicator/outoftree.go",
	}

	b.Run("ExactMatch", func(b *testing.B) {
		arg := "internal/adjudicator/outoftree.go"
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			benchSinkOutcome, benchSinkCause = resolveChangedWitness(arg, changed)
		}
	})

	b.Run("DirectoryRefuted", func(b *testing.B) {
		arg := "internal/adjudicator"
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			benchSinkOutcome, benchSinkCause = resolveChangedWitness(arg, changed)
		}
	})

	b.Run("NonMemberRefuted", func(b *testing.B) {
		arg := "README.md"
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			benchSinkOutcome, benchSinkCause = resolveChangedWitness(arg, changed)
		}
	})
}
