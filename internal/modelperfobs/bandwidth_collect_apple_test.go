package modelperfobs

import (
	"bytes"
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

func TestImportAppleMemoryDirectByteRates(t *testing.T) {
	options := AppleMemoryImportOptions{
		Provider: "fixture-apple-counter-export", ProviderVersion: "1.0",
		Scope: AppleMemoryScope{Kind: "package"}, Phase: PhaseDecode, Shape: ShapeSmall,
	}
	collection := importAppleMemoryFixture(t, "apple-memory-direct-rate.json", options)
	artifact := collection.AppleMemoryArtifact
	if artifact == nil || artifact.Schema != AppleMemoryArtifactSchema || artifact.ImportFormat != AppleMemoryFormatGenericJSON {
		t.Fatalf("Apple memory artifact=%+v", artifact)
	}
	if artifact.IntervalNS != int64(500*time.Millisecond) || collection.IntervalMS != 500 {
		t.Fatalf("interval artifact=%d collection_ms=%d", artifact.IntervalNS, collection.IntervalMS)
	}
	if artifact.Provider != options.Provider || artifact.ProviderVersion != options.ProviderVersion ||
		artifact.Scope != (AppleMemoryScope{Kind: "package"}) {
		t.Fatalf("provider/scope artifact=%+v", artifact)
	}
	if got, want := *artifact.ReadBytesPerSecond, uint64(85_000_000_000); got != want {
		t.Fatalf("read rate=%d want %d", got, want)
	}
	if got, want := *artifact.WriteBytesPerSecond, uint64(17_500_000_000); got != want {
		t.Fatalf("write rate=%d want %d", got, want)
	}
	if artifact.ReadBytes != nil || artifact.WriteBytes != nil {
		t.Fatalf("direct rates fabricated interval bytes: %+v", artifact)
	}
	if artifact.Counters[0].RawField != "memory_read_bytes_per_second" ||
		artifact.Counters[0].ByteProvenance != "direct-byte-rate" ||
		string(artifact.RawProviderFields["provider_scope"]) != `"package"` ||
		artifact.CaptureMetadata["capture_method"] != "synthetic-contract-fixture" {
		t.Fatalf("raw provenance not preserved: %+v", artifact)
	}
	sample := collection.Capture.Samples[0]
	if sample.Provenance.Source != "apple-memory-direct-byte-rates" ||
		sample.Provenance.Device != "apple-unified-memory" ||
		sample.Live.Scope == nil || sample.Live.Scope.Kind != "package" {
		t.Fatalf("sample provenance/scope=%+v live=%+v", sample.Provenance, sample.Live)
	}
	assertHostLive(t, sample.Live, 85, 17.5, 102.5)
	if sample.Host.ProcessReadBytes != nil || sample.Host.ProcessWriteBytes != nil ||
		sample.Software.PhysicalReadBytes != nil || sample.Software.PhysicalWriteBytes != nil {
		t.Fatalf("Apple memory counter conflated with process/software I/O: host=%+v software=%+v", sample.Host, sample.Software)
	}
	if !collection.Availability.DRAMCounters {
		t.Fatal("Apple DRAM counter availability not set")
	}
	if collection.Availability.DeviceCounters {
		t.Fatal("package-scoped Apple counters reported as device counters")
	}
}

func TestImportAppleMemoryRejectsNonHostScopesAndScopeIDs(t *testing.T) {
	data, err := os.ReadFile(filepath.Join("testdata", "apple-memory-direct-rate.json"))
	if err != nil {
		t.Fatal(err)
	}
	for _, tt := range []struct {
		name  string
		scope AppleMemoryScope
		want  string
	}{
		{name: "process", scope: AppleMemoryScope{Kind: "process", ID: "123"}, want: "must be system or package"},
		{name: "device", scope: AppleMemoryScope{Kind: "device", ID: "apple-gpu0"}, want: "must be system or package"},
		{name: "system-id", scope: AppleMemoryScope{Kind: "system", ID: "host0"}, want: "must not have an id"},
		{name: "package-id", scope: AppleMemoryScope{Kind: "package", ID: "package0"}, want: "must not have an id"},
	} {
		t.Run(tt.name, func(t *testing.T) {
			_, err := ImportAppleMemoryCounters(bytes.NewReader(data), appleFixtureOptions(tt.scope))
			if err == nil || !strings.Contains(err.Error(), tt.want) {
				t.Fatalf("scope=%+v error=%v want=%q", tt.scope, err, tt.want)
			}
		})
	}
}

func TestImportAppleMemoryMonotonicByteDeltas(t *testing.T) {
	collection := importAppleMemoryFixture(t, "apple-memory-byte-delta.json", AppleMemoryImportOptions{
		Provider: "fixture-apple-counter-export", ProviderVersion: "1.0",
		Scope: AppleMemoryScope{Kind: "system"}, Phase: PhasePrefill, Shape: ShapeLarge,
	})
	artifact := collection.AppleMemoryArtifact
	if got, want := *artifact.ReadBytes, uint64(42_500_000_000); got != want {
		t.Fatalf("read bytes=%d want %d", got, want)
	}
	if got, want := *artifact.WriteBytes, uint64(8_750_000_000); got != want {
		t.Fatalf("write bytes=%d want %d", got, want)
	}
	if artifact.ReadBytesPerSecond != nil || artifact.WriteBytesPerSecond != nil {
		t.Fatalf("delta import fabricated direct rates: %+v", artifact)
	}
	for _, counter := range artifact.Counters {
		if counter.ByteProvenance != "monotonic-byte-delta" || len(counter.RawValues) != 2 ||
			counter.ResetObserved == nil || *counter.ResetObserved {
			t.Fatalf("delta/reset provenance=%+v", counter)
		}
	}
	assertHostLive(t, collection.Capture.Samples[0].Live, 85, 17.5, 102.5)
}

func TestImportAppleMemoryRejectsUnsafeEvidence(t *testing.T) {
	delta := readGenericAppleFixture(t, "apple-memory-byte-delta.json")
	rate := readGenericAppleFixture(t, "apple-memory-direct-rate.json")
	tests := []struct {
		name  string
		input genericAppleMemoryImport
		scope AppleMemoryScope
		want  string
	}{
		{
			name: "reset", input: mutateGenericApple(delta, func(input *genericAppleMemoryImport) {
				yes := true
				input.Counters[0].Snapshots[1].ResetObserved = &yes
			}), scope: AppleMemoryScope{Kind: "system"}, want: "reset during capture",
		},
		{
			name: "ambiguous-reset", input: mutateGenericApple(delta, func(input *genericAppleMemoryImport) {
				input.Counters[0].Snapshots[1].ResetObserved = nil
			}), scope: AppleMemoryScope{Kind: "system"}, want: "reset state is ambiguous",
		},
		{
			name: "decrease", input: mutateGenericApple(delta, func(input *genericAppleMemoryImport) {
				input.Counters[0].Snapshots[1].Value = input.Counters[0].Snapshots[0].Value - 1
			}), scope: AppleMemoryScope{Kind: "system"}, want: "decreased without a supported wrap contract",
		},
		{
			name: "mixed-scope", input: mutateGenericApple(delta, func(input *genericAppleMemoryImport) {
				input.Counters[1].Scope = AppleMemoryScope{Kind: "device", ID: "gpu0"}
			}), scope: AppleMemoryScope{Kind: "system"}, want: "mixed Apple memory counter scopes",
		},
		{
			name: "unsupported-unit", input: mutateGenericApple(delta, func(input *genericAppleMemoryImport) {
				input.Counters[0].Unit = "pages"
			}), scope: AppleMemoryScope{Kind: "system"}, want: "unsupported Apple memory counter unit",
		},
		{
			name: "missing-pair", input: mutateGenericApple(rate, func(input *genericAppleMemoryImport) {
				input.Counters = input.Counters[:1]
			}), scope: AppleMemoryScope{Kind: "package"}, want: "requires one read/write counter pair",
		},
		{
			name: "mixed-rate-delta", input: mutateGenericApple(rate, func(input *genericAppleMemoryImport) {
				input.Counters[1] = delta.Counters[1]
				input.Counters[1].Scope = AppleMemoryScope{Kind: "package"}
				input.RawProviderFields["memory_write_bytes"] = delta.RawProviderFields["memory_write_bytes"]
			}), scope: AppleMemoryScope{Kind: "package"}, want: "mixed byte provenance",
		},
		{
			name: "process-io-field", input: mutateGenericApple(rate, func(input *genericAppleMemoryImport) {
				input.Counters[0].Field = "process_io_read_bytes"
			}), scope: AppleMemoryScope{Kind: "package"}, want: "unsupported process/storage I/O",
		},
		{
			name: "activity-utilization-field", input: mutateGenericApple(rate, func(input *genericAppleMemoryImport) {
				input.Counters[0].Field = "activity_monitor_memory_utilization"
			}), scope: AppleMemoryScope{Kind: "package"}, want: "power, utilization, capacity, or roofline",
		},
		{
			name: "capacity-field", input: mutateGenericApple(rate, func(input *genericAppleMemoryImport) {
				input.Counters[0].Field = "resident_bytes"
				input.RawProviderFields["resident_bytes"] = json.RawMessage(`123`)
			}), scope: AppleMemoryScope{Kind: "package"}, want: "unsupported process/storage I/O",
		},
		{
			name: "direction-field-conflict", input: mutateGenericApple(rate, func(input *genericAppleMemoryImport) {
				input.Counters[0].Field = "memory_write_bytes_per_second"
			}), scope: AppleMemoryScope{Kind: "package"}, want: "conflicts with declared read direction",
		},
		{
			name: "semantic-direction-conflict", input: mutateGenericApple(rate, func(input *genericAppleMemoryImport) {
				input.Counters[0].Semantic = "memory-write-bytes"
			}), scope: AppleMemoryScope{Kind: "package"}, want: "semantic must be",
		},
		{
			name: "missing-raw-provider-fields", input: mutateGenericApple(rate, func(input *genericAppleMemoryImport) {
				input.RawProviderFields = nil
			}), scope: AppleMemoryScope{Kind: "package"}, want: "requires raw_provider_fields",
		},
		{
			name: "raw-value-mismatch", input: mutateGenericApple(rate, func(input *genericAppleMemoryImport) {
				input.RawProviderFields["memory_read_bytes_per_second"] = json.RawMessage(`1`)
			}), scope: AppleMemoryScope{Kind: "package"}, want: "does not match normalized byte rate",
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			data, err := json.Marshal(tt.input)
			if err != nil {
				t.Fatal(err)
			}
			_, err = ImportAppleMemoryCounters(bytes.NewReader(data), AppleMemoryImportOptions{
				Provider: "fixture-apple-counter-export", ProviderVersion: "1.0",
				Scope: tt.scope, Phase: PhaseOther, Shape: ShapeSmall,
			})
			if err == nil || !strings.Contains(err.Error(), tt.want) {
				t.Fatalf("error=%v want %q", err, tt.want)
			}
		})
	}
}

func TestImportAppleMemoryRejectsPermissionUnsupportedMalformedAndRawPowermetrics(t *testing.T) {
	rate := readGenericAppleFixture(t, "apple-memory-direct-rate.json")
	for _, tt := range []struct {
		name   string
		status string
		want   string
	}{
		{name: "permission", status: "permission-denied", want: "permission-gated"},
		{name: "unsupported", status: "unsupported", want: "provider capture is unsupported"},
	} {
		t.Run(tt.name, func(t *testing.T) {
			input := mutateGenericApple(rate, func(input *genericAppleMemoryImport) {
				input.Capture.Status = tt.status
				input.Capture.Error = "provider refused capture"
			})
			data, err := json.Marshal(input)
			if err != nil {
				t.Fatal(err)
			}
			_, err = ImportAppleMemoryCounters(bytes.NewReader(data), appleFixtureOptions(AppleMemoryScope{Kind: "package"}))
			if err == nil || !strings.Contains(err.Error(), tt.want) {
				t.Fatalf("error=%v want %q", err, tt.want)
			}
		})
	}

	raw, err := os.ReadFile(filepath.Join("testdata", "apple-memory-direct-rate.json"))
	if err != nil {
		t.Fatal(err)
	}
	raw = bytes.Replace(raw, []byte(`"status": "ok"`), []byte(`"status": "ok", "unknown": true`), 1)
	if _, err := ImportAppleMemoryCounters(bytes.NewReader(raw), appleFixtureOptions(AppleMemoryScope{Kind: "package"})); err == nil || !strings.Contains(err.Error(), "unknown or non-exact field") {
		t.Fatalf("malformed unknown-field error=%v", err)
	}
	original, err := os.ReadFile(filepath.Join("testdata", "apple-memory-direct-rate.json"))
	if err != nil {
		t.Fatal(err)
	}
	duplicate := bytes.Replace(original, []byte(`"schema": "fak-apple-memory-counter-import/1"`), []byte(`"schema": "fak-apple-memory-counter-import/1", "schema": "duplicate"`), 1)
	if _, err := ImportAppleMemoryCounters(bytes.NewReader(duplicate), appleFixtureOptions(AppleMemoryScope{Kind: "package"})); err == nil || !strings.Contains(err.Error(), "duplicate JSON object name") {
		t.Fatalf("duplicate-key error=%v", err)
	}
	caseVariant := bytes.Replace(original, []byte(`"provider": "fixture-apple-counter-export"`), []byte(`"Provider": "fixture-apple-counter-export"`), 1)
	if _, err := ImportAppleMemoryCounters(bytes.NewReader(caseVariant), appleFixtureOptions(AppleMemoryScope{Kind: "package"})); err == nil || !strings.Contains(err.Error(), "unknown or non-exact field") {
		t.Fatalf("case-variant error=%v", err)
	}
	providerWhitespace := bytes.Replace(original, []byte(`"provider": "fixture-apple-counter-export"`), []byte(`"provider": " fixture-apple-counter-export"`), 1)
	if _, err := ImportAppleMemoryCounters(bytes.NewReader(providerWhitespace), appleFixtureOptions(AppleMemoryScope{Kind: "package"})); err == nil || !strings.Contains(err.Error(), "does not match declared provider") {
		t.Fatalf("provider exactness error=%v", err)
	}
	powermetrics := `<plist version="1.0"><dict><key>bandwidth_counters</key><array/></dict></plist>`
	if _, err := ImportAppleMemoryCounters(strings.NewReader(powermetrics), appleFixtureOptions(AppleMemoryScope{Kind: "package"})); err == nil || !strings.Contains(err.Error(), "not the supported generic JSON schema") {
		t.Fatalf("raw powermetrics must remain unsupported, error=%v", err)
	}
}

func TestAppleMemoryPressureFieldsCoexistWithoutBecomingBandwidth(t *testing.T) {
	collection := importAppleMemoryFixture(t, "apple-memory-direct-rate.json", appleFixtureOptions(AppleMemoryScope{Kind: "package"}))
	wired, compressed, pageIns := uint64(10<<30), uint64(3<<30), uint64(123)
	pageOutRate := 4.5
	collection.Capture.Samples[0].Host = HostSignals{
		MemoryWiredResidentBytes:      &wired,
		MemoryCompressedResidentBytes: &compressed,
		MemoryPageInPagesTotal:        &pageIns,
		MemoryPageOutPagesPerSecond:   &pageOutRate,
	}
	report, err := AnalyzeBandwidth(collection.Capture)
	if err != nil {
		t.Fatal(err)
	}
	observation := report.Observations[0]
	assertHostLive(t, observation.Live, 85, 17.5, 102.5)
	if observation.Host.MemoryWiredResidentBytes == nil || *observation.Host.MemoryWiredResidentBytes != wired ||
		observation.Host.MemoryCompressedResidentBytes == nil || *observation.Host.MemoryCompressedResidentBytes != compressed ||
		observation.Host.MemoryPageInPagesTotal == nil || *observation.Host.MemoryPageInPagesTotal != pageIns ||
		observation.Host.MemoryPageOutPagesPerSecond == nil || *observation.Host.MemoryPageOutPagesPerSecond != pageOutRate {
		t.Fatalf("Darwin pressure fields lost beside Apple rates: %+v", observation.Host)
	}
	if observation.Host.ProcessReadBytes != nil || observation.Host.ProcessWriteBytes != nil {
		t.Fatalf("Darwin pressure synthesized process traffic: %+v", observation.Host)
	}
}

func TestApplyHostRooflineMeasurementToApplePackageDoesNotContaminateDevices(t *testing.T) {
	collection := importAppleMemoryFixture(t, "apple-memory-direct-rate.json", appleFixtureOptions(AppleMemoryScope{Kind: "package"}))
	amdRead, amdWrite, amdRoofline := 90.0, 10.0, 500.0
	collection.Capture.Samples = append(collection.Capture.Samples, BandwidthSample{
		Phase: PhaseOther, Shape: ShapeSmall,
		Provenance: BandwidthProvenance{
			Source: "live-amd-rocm", Device: "AMD fixture GPU", Collector: "fixture-device", SampledAt: "2026-08-27T18:00:00.5Z",
		},
		Rooflines: Rooflines{MeasuredSustainableGBS: &amdRoofline},
		Live:      LiveBandwidth{ReadGBS: &amdRead, WriteGBS: &amdWrite},
	})
	nvidiaRead, nvidiaWrite, nvidiaRoofline := 200.0, 50.0, 750.0
	collection.Capture.Samples = append(collection.Capture.Samples, BandwidthSample{
		Phase: PhaseOther, Shape: ShapeSmall,
		Provenance: BandwidthProvenance{
			Source: "nvidia-profiled-kernel", Device: "NVIDIA fixture GPU", Collector: "nvidia-nsight-compute", SampledAt: "2026-08-27T18:00:00.5Z",
		},
		Rooflines: Rooflines{MeasuredSustainableGBS: &nvidiaRoofline},
		Live:      LiveBandwidth{ReadGBS: &nvidiaRead, WriteGBS: &nvidiaWrite},
	})
	measurement := RooflineMeasurement{Scope: "host-memory", MeasuredSustainableGBS: 120}
	if err := ApplyHostRooflineMeasurement(&collection, measurement); err != nil {
		t.Fatal(err)
	}
	if collection.RooflineMeasurement.AppliedHostObservationCount != 1 ||
		collection.Report.Observations[0].Rooflines.SelectedGBS == nil ||
		*collection.Report.Observations[0].Rooflines.SelectedGBS != 120 {
		t.Fatalf("Apple package roofline not applied: %+v", collection.Report.Observations[0].Rooflines)
	}
	device := collection.Report.Observations[1]
	if device.Rooflines.SelectedGBS == nil || *device.Rooflines.SelectedGBS != amdRoofline ||
		device.Live.TotalGBS == nil || *device.Live.TotalGBS != 100 {
		t.Fatalf("Apple host roofline contaminated AMD device: roofline=%+v live=%+v", device.Rooflines, device.Live)
	}
	nvidia := collection.Report.Observations[2]
	if nvidia.Rooflines.SelectedGBS == nil || *nvidia.Rooflines.SelectedGBS != nvidiaRoofline ||
		nvidia.Live.TotalGBS == nil || *nvidia.Live.TotalGBS != 250 {
		t.Fatalf("Apple host roofline contaminated NVIDIA device: roofline=%+v live=%+v", nvidia.Rooflines, nvidia.Live)
	}

	invalidArtifactScope := importAppleMemoryFixture(t, "apple-memory-direct-rate.json", appleFixtureOptions(AppleMemoryScope{Kind: "package"}))
	invalidArtifactScope.AppleMemoryArtifact.Scope = AppleMemoryScope{Kind: "device", ID: "apple-gpu0"}
	if err := ApplyHostRooflineMeasurement(&invalidArtifactScope, measurement); err == nil || !strings.Contains(err.Error(), "requires system/package") {
		t.Fatalf("invalid Apple artifact scope error=%v", err)
	}
	if invalidArtifactScope.RooflineMeasurement != nil || invalidArtifactScope.Report.Observations[0].Rooflines.SelectedGBS != nil {
		t.Fatalf("invalid-scope Apple observation mutated by host roofline: %+v", invalidArtifactScope)
	}

	mismatched := importAppleMemoryFixture(t, "apple-memory-direct-rate.json", appleFixtureOptions(AppleMemoryScope{Kind: "package"}))
	mismatched.Capture.Samples[0].Live.Scope = &BandwidthScope{Kind: "device", ID: "apple-gpu0"}
	if err := ApplyHostRooflineMeasurement(&mismatched, measurement); err == nil || !strings.Contains(err.Error(), "matches 0 imported samples") {
		t.Fatalf("artifact/sample scope mismatch error=%v", err)
	}
	if mismatched.RooflineMeasurement != nil || mismatched.Report.Observations[0].Rooflines.SelectedGBS != nil {
		t.Fatalf("scope-mismatched Apple observation mutated by host roofline: %+v", mismatched)
	}
}

func TestAppleMemoryArtifactJSONOmitsUnavailableValues(t *testing.T) {
	collection := importAppleMemoryFixture(t, "apple-memory-direct-rate.json", appleFixtureOptions(AppleMemoryScope{Kind: "package"}))
	data, err := json.Marshal(collection)
	if err != nil {
		t.Fatal(err)
	}
	text := string(data)
	for _, want := range []string{
		`"apple_memory_artifact":{"schema":"fak-apple-unified-memory-counters/1"`,
		`"provider_version":"1.0"`,
		`"scope":{"kind":"package"}`,
		`"capture_status":"ok"`,
		`"raw_provider_fields":{"memory_read_bytes_per_second":85000000000`,
		`"phase_attribution":"request phase is temporal attribution only; package/system traffic is not process or model traffic"`,
		`"read_gb_s":85`,
		`"write_gb_s":17.5`,
		`"total_gb_s":102.5`,
	} {
		if !strings.Contains(text, want) {
			t.Fatalf("Apple artifact JSON missing %s: %s", want, text)
		}
	}
	for _, forbidden := range []string{
		`"read_bytes":0`, `"write_bytes":0`, `"process_read_bytes":0`,
		`"process_write_bytes":0`, `"physical_read_bytes":0`, `"physical_write_bytes":0`,
	} {
		if strings.Contains(text, forbidden) {
			t.Fatalf("unavailable Apple value serialized as zero: %s", text)
		}
	}
}

func importAppleMemoryFixture(t *testing.T, name string, o AppleMemoryImportOptions) BandwidthCollection {
	t.Helper()
	data, err := os.ReadFile(filepath.Join("testdata", name))
	if err != nil {
		t.Fatal(err)
	}
	collection, err := ImportAppleMemoryCounters(bytes.NewReader(data), o)
	if err != nil {
		t.Fatal(err)
	}
	return collection
}

func readGenericAppleFixture(t *testing.T, name string) genericAppleMemoryImport {
	t.Helper()
	data, err := os.ReadFile(filepath.Join("testdata", name))
	if err != nil {
		t.Fatal(err)
	}
	var input genericAppleMemoryImport
	if err := json.Unmarshal(data, &input); err != nil {
		t.Fatal(err)
	}
	return input
}

func mutateGenericApple(input genericAppleMemoryImport, mutate func(*genericAppleMemoryImport)) genericAppleMemoryImport {
	data, err := json.Marshal(input)
	if err != nil {
		panic(err)
	}
	var cloned genericAppleMemoryImport
	if err := json.Unmarshal(data, &cloned); err != nil {
		panic(err)
	}
	mutate(&cloned)
	return cloned
}

func appleFixtureOptions(scope AppleMemoryScope) AppleMemoryImportOptions {
	return AppleMemoryImportOptions{
		Provider: "fixture-apple-counter-export", ProviderVersion: "1.0",
		Scope: scope, Phase: PhaseOther, Shape: ShapeSmall,
	}
}
