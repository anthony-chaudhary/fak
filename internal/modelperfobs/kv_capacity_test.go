package modelperfobs

import (
	"bytes"
	"os"
	"strings"
	"testing"
)

func TestKVCapacityDialectFixturesNormalizeEquivalentResidentTokens(t *testing.T) {
	block := readKVFixture(t, "testdata/kv-capacity-block.json", KVDialectBlock)
	direct := readKVFixture(t, "testdata/kv-capacity-direct.json", KVDialectDirect)

	blockSnapshot := NormalizeKVCapacity(block, nil)
	directSnapshot := NormalizeKVCapacity(direct, nil)
	assertDerivedUint64(t, blockSnapshot.Normalized.ResidentTokens, 16_384, UnitTokens, MethodBlocksTimesBlockTokens, ConfidenceExact)
	assertDerivedUint64(t, directSnapshot.Normalized.ResidentTokens, 16_384, UnitTokens, MethodDirectObservation, ConfidenceExact)
	assertDerivedUint64(t, blockSnapshot.Normalized.TotalTokens, 32_768, UnitTokens, MethodBlocksTimesBlockTokens, ConfidenceExact)
	assertDerivedUint64(t, directSnapshot.Normalized.TotalTokens, 32_768, UnitTokens, MethodDirectObservation, ConfidenceExact)

	if blockSnapshot.Native.UsedBlocks == nil || blockSnapshot.Native.UsedBlocks.Value != 1_024 {
		t.Fatalf("block native used blocks not retained: %+v", blockSnapshot.Native.UsedBlocks)
	}
	if directSnapshot.Native.ResidentTokens == nil || directSnapshot.Native.ResidentTokens.Value != 16_384 {
		t.Fatalf("direct native resident tokens not retained: %+v", directSnapshot.Native.ResidentTokens)
	}
	if got := string(blockSnapshot.RawMetrics["kv_used_blocks"]); got != "1024" {
		t.Fatalf("raw block metric=%q, want exact source value", got)
	}
	if got := string(blockSnapshot.RawGeometry["dtype_bytes"]); got != "2" {
		t.Fatalf("raw geometry=%q, want exact configured value", got)
	}
	if !blockSnapshot.Validation.Valid || !blockSnapshot.Validation.CrossUnitComparable {
		t.Fatalf("block validation=%+v, want valid and cross-unit comparable", blockSnapshot.Validation)
	}
	if !directSnapshot.Validation.Valid || !directSnapshot.Validation.CrossUnitComparable {
		t.Fatalf("direct validation=%+v, want valid and cross-unit comparable", directSnapshot.Validation)
	}

	var rendered bytes.Buffer
	if err := WriteKVCapacityMarkdown(&rendered, blockSnapshot); err != nil {
		t.Fatal(err)
	}
	for _, want := range []string{
		"Native values",
		"kv_used_blocks: 1024 blocks (observed",
		"Normalized values",
		"resident_tokens: 16384 tokens (blocks-times-block-tokens, exact)",
		EstimatedCapacityCaveat,
	} {
		if !strings.Contains(rendered.String(), want) {
			t.Errorf("render missing %q:\n%s", want, rendered.String())
		}
	}
}

func TestKVCapacityMissingBlockGeometryIsUnavailable(t *testing.T) {
	sample := readKVFixture(t, "testdata/kv-capacity-block.json", KVDialectBlock)
	sample.Native.BlockTokens = nil
	delete(sample.RawMetrics, "kv_block_size_tokens")

	snapshot := NormalizeKVCapacity(sample, nil)
	resident := snapshot.Normalized.ResidentTokens
	if resident.Value != nil {
		t.Fatalf("resident tokens=%d, want unavailable rather than a guessed zero or block count", *resident.Value)
	}
	if resident.Method != MethodUnavailable || resident.Confidence != ConfidenceUnavailable {
		t.Fatalf("unavailable provenance=%+v", resident)
	}
	if !strings.Contains(resident.UnavailableReason, "block size in tokens") {
		t.Fatalf("unavailable reason=%q", resident.UnavailableReason)
	}
	if snapshot.Validation.CrossUnitComparable {
		t.Fatalf("validation=%+v, want incomparable without block geometry", snapshot.Validation)
	}
}

func TestKVCapacityUsesObservedBytesBeforeGeometryEstimate(t *testing.T) {
	sample := readKVFixture(t, "testdata/kv-capacity-direct.json", KVDialectDirect)
	observed := NormalizeKVCapacity(sample, nil)
	assertDerivedUint64(t, observed.Normalized.ResidentBytes, 2_147_483_648, UnitBytes, MethodDirectObservation, ConfidenceExact)
	if observed.Normalized.ResidentBytes.Sources[0].Nature != NatureObserved {
		t.Fatalf("observed byte provenance=%+v", observed.Normalized.ResidentBytes)
	}

	sample.Native.ResidentBytes = nil
	delete(sample.RawMetrics, "kv_resident_bytes")
	estimated := NormalizeKVCapacity(sample, nil)
	assertDerivedUint64(t, estimated.Normalized.ResidentBytes, 2_147_483_648, UnitBytes, MethodModelGeometryEstimate, ConfidenceEstimated)
	if estimated.Normalized.ResidentBytes.Sources[0].Nature != NatureObserved {
		t.Fatalf("estimated byte sources should begin with resident-token observation: %+v", estimated.Normalized.ResidentBytes)
	}
}

func TestKVCapacityValidatesUnitsDenominatorsAndImpossibleOccupancy(t *testing.T) {
	tests := []struct {
		name string
		edit func(*KVMetricSample)
		code KVValidationCode
	}{
		{
			name: "zero block denominator",
			edit: func(sample *KVMetricSample) { sample.Native.TotalBlocks.Value = 0 },
			code: ValidationInvalidDenominator,
		},
		{
			name: "used exceeds total",
			edit: func(sample *KVMetricSample) { sample.Native.UsedBlocks.Value = 3_000 },
			code: ValidationImpossibleOccupancy,
		},
		{
			name: "reported ratio exceeds one",
			edit: func(sample *KVMetricSample) { sample.Native.Occupancy.Value = 1.1 },
			code: ValidationImpossibleOccupancy,
		},
		{
			name: "allocatable exceeds configured",
			edit: func(sample *KVMetricSample) {
				sample.Native.AllocatableBytes.Value = sample.Native.ConfiguredBytes.Value + 1
			},
			code: ValidationInvalidCapacity,
		},
		{
			name: "direct and block token units disagree",
			edit: func(sample *KVMetricSample) {
				sample.Native.ResidentTokens = uintMetric(9, UnitTokens, NatureObserved, "test", "resident_tokens")
			},
			code: ValidationUnitInvariant,
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			sample := readKVFixture(t, "testdata/kv-capacity-block.json", KVDialectBlock)
			test.edit(&sample)
			snapshot := NormalizeKVCapacity(sample, nil)
			if snapshot.Validation.Valid {
				t.Fatalf("validation=%+v, want invalid", snapshot.Validation)
			}
			if !hasKVValidationCode(snapshot.Validation.Issues, test.code) {
				t.Fatalf("issues=%+v, want %s", snapshot.Validation.Issues, test.code)
			}
		})
	}
}

func TestKVCapacityRejectsCounterResetsAndIdentityChangesBetweenScrapes(t *testing.T) {
	previous := readKVFixture(t, "testdata/kv-capacity-block.json", KVDialectBlock)
	current := readKVFixture(t, "testdata/kv-capacity-block.json", KVDialectBlock)
	previous.Native.Evictions.Value = current.Native.Evictions.Value + 1
	current.RuntimeID = "runtime-restarted"
	current.ScrapedAt = previous.ScrapedAt.Add(1)

	snapshot := NormalizeKVCapacity(current, &previous)
	if snapshot.Validation.TemporalComparable {
		t.Fatalf("validation=%+v, want temporal comparison refused", snapshot.Validation)
	}
	for _, code := range []KVValidationCode{ValidationCounterReset, ValidationIdentityChanged} {
		if !hasKVValidationCode(snapshot.Validation.Issues, code) {
			t.Errorf("issues=%+v, want %s", snapshot.Validation.Issues, code)
		}
	}
	if snapshot.CounterDeltas.Evictions.Value != nil || snapshot.CounterDeltas.Evictions.Method != MethodUnavailable {
		t.Fatalf("eviction delta=%+v, want unavailable on reset", snapshot.CounterDeltas.Evictions)
	}
}

func TestKVCapacityComputesCounterDeltasOnlyAcrossComparableScrapes(t *testing.T) {
	previous := readKVFixture(t, "testdata/kv-capacity-block.json", KVDialectBlock)
	current := readKVFixture(t, "testdata/kv-capacity-block.json", KVDialectBlock)
	previous.Native.Evictions.Value = 5
	previous.Native.Preemptions.Value = 1
	current.ScrapedAt = previous.ScrapedAt.Add(1)

	snapshot := NormalizeKVCapacity(current, &previous)
	if !snapshot.Validation.TemporalComparable {
		t.Fatalf("validation=%+v, want comparable stable identity and ordered scrapes", snapshot.Validation)
	}
	assertDerivedUint64(t, snapshot.CounterDeltas.Evictions, 2, UnitCount, MethodCounterDelta, ConfidenceExact)
	assertDerivedUint64(t, snapshot.CounterDeltas.Preemptions, 1, UnitCount, MethodCounterDelta, ConfidenceExact)

	current.ScrapedAt = previous.ScrapedAt
	snapshot = NormalizeKVCapacity(current, &previous)
	if snapshot.Validation.TemporalComparable || !hasKVValidationCode(snapshot.Validation.Issues, ValidationInvalidScrapeOrder) {
		t.Fatalf("same-time scrape validation=%+v", snapshot.Validation)
	}
}

func TestKVCapacityDerivesReusableAndHighWaterTokensWithoutErasingNativeUnits(t *testing.T) {
	sample := readKVFixture(t, "testdata/kv-capacity-block.json", KVDialectBlock)
	snapshot := NormalizeKVCapacity(sample, nil)
	assertDerivedUint64(t, snapshot.Normalized.ReusableTokens, 4_096, UnitTokens, MethodBlocksTimesBlockTokens, ConfidenceExact)
	assertDerivedUint64(t, snapshot.Normalized.HighWaterMarkTokens, 19_200, UnitTokens, MethodBlocksTimesBlockTokens, ConfidenceExact)
	if snapshot.Native.ReusableBlocks.Value != 256 || snapshot.Native.HighWaterMarkBlocks.Value != 1_200 {
		t.Fatalf("native units lost: reusable=%+v high-water=%+v", snapshot.Native.ReusableBlocks, snapshot.Native.HighWaterMarkBlocks)
	}
}

func readKVFixture(t *testing.T, path string, dialect KVDialect) KVMetricSample {
	t.Helper()
	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	sample, err := DecodeKVMetricSample(data, dialect)
	if err != nil {
		t.Fatal(err)
	}
	return sample
}

func assertDerivedUint64(t *testing.T, got KVDerivedUint64, value uint64, unit KVUnit, method KVDerivationMethod, confidence KVConfidence) {
	t.Helper()
	if got.Value == nil || *got.Value != value || got.Unit != unit || got.Method != method || got.Confidence != confidence {
		t.Fatalf("derived value=%+v, want value=%d unit=%s method=%s confidence=%s", got, value, unit, method, confidence)
	}
}

func hasKVValidationCode(issues []KVValidationIssue, code KVValidationCode) bool {
	for _, issue := range issues {
		if issue.Code == code {
			return true
		}
	}
	return false
}
