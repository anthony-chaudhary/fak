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

func TestImportHostControllerDirectBytes(t *testing.T) {
	collection := importHostControllerFixture(t, "host-controller-direct.json", HostControllerImportOptions{
		Format: HostCounterFormatGenericJSON, Provider: "fixture-imc", Scope: HostControllerScope{Kind: "system"},
		Phase: PhaseDecode, Shape: ShapeSmall,
	})
	if collection.HostControllerArtifact == nil || collection.HostControllerArtifact.Schema != HostControllerArtifactSchema {
		t.Fatalf("host artifact=%+v", collection.HostControllerArtifact)
	}
	artifact := collection.HostControllerArtifact
	if artifact.ImportFormat != HostCounterFormatGenericJSON || artifact.IntervalNS != int64(2*time.Second) || artifact.RunningRatio != 1 {
		t.Fatalf("artifact summary=%+v", artifact)
	}
	if artifact.CaptureStartedAt != "2026-08-27T10:00:00Z" || artifact.CaptureEndedAt != "2026-08-27T10:00:02Z" || collection.IntervalMS != 2000 {
		t.Fatalf("capture metadata artifact=%+v interval_ms=%d", artifact, collection.IntervalMS)
	}
	if got, want := *artifact.ReadBytes, uint64(3_000_000_000); got != want {
		t.Fatalf("read bytes=%d want %d", got, want)
	}
	if got, want := *artifact.WriteBytes, uint64(1_000_000_000); got != want {
		t.Fatalf("write bytes=%d want %d", got, want)
	}
	if artifact.Counters[0].RawEvent != "host_dram_read_bytes" || artifact.Counters[0].ByteProvenance != "direct-bytes" ||
		artifact.Counters[0].StartRawValue == nil || *artifact.Counters[0].StartRawValue != 1 ||
		artifact.Counters[0].EndRawValue == nil || *artifact.Counters[0].EndRawValue != 3_000_000_001 || artifact.Counters[0].DeltaRawValue != 3_000_000_000 ||
		artifact.Counters[0].TimeEnabledNS == nil || *artifact.Counters[0].TimeEnabledNS != uint64(2*time.Second) ||
		artifact.Counters[0].TimeRunningNS == nil || *artifact.Counters[0].TimeRunningNS != uint64(2*time.Second) {
		t.Fatalf("raw read artifact=%+v", artifact.Counters[0])
	}
	sample := collection.Capture.Samples[0]
	if sample.Provenance.Source != "host-controller-direct-bytes" || sample.Provenance.Device != "host-memory" {
		t.Fatalf("provenance=%+v", sample.Provenance)
	}
	assertHostLive(t, sample.Live, 1.5, 0.5, 2)
	if sample.Host.ProcessReadBytes != nil || sample.Host.ProcessWriteBytes != nil || sample.Software.PhysicalReadBytes != nil || sample.Software.PhysicalWriteBytes != nil {
		t.Fatalf("host controller counters conflated with process/software I/O: host=%+v software=%+v", sample.Host, sample.Software)
	}
	if !collection.Availability.DRAMCounters {
		t.Fatal("DRAM counter availability not set")
	}
}

func TestImportHostControllerCAS64ConversionAndScope(t *testing.T) {
	collection := importHostControllerFixture(t, "host-controller-cas64.json", HostControllerImportOptions{
		Provider: "linux-perf-intel-imc", Scope: HostControllerScope{Kind: "socket", ID: "0"},
		Phase: PhasePrefill, Shape: ShapeLarge,
	})
	artifact := collection.HostControllerArtifact
	if artifact == nil || artifact.Scope != (HostControllerScope{Kind: "socket", ID: "0"}) || artifact.RunningRatio != .95 {
		t.Fatalf("artifact=%+v", artifact)
	}
	for _, counter := range artifact.Counters {
		if counter.BytesPerEvent == nil || *counter.BytesPerEvent != 64 || counter.ByteProvenance != "converted-events" {
			t.Fatalf("conversion not preserved: %+v", counter)
		}
	}
	if artifact.Counters[0].RunningRatio != .99 || artifact.Counters[0].TimeEnabledNS != nil || artifact.Counters[0].TimeRunningNS != nil ||
		artifact.Counters[1].TimeEnabledNS == nil || *artifact.Counters[1].TimeEnabledNS != 100_000_000 ||
		artifact.Counters[1].TimeRunningNS == nil || *artifact.Counters[1].TimeRunningNS != 95_000_000 {
		t.Fatalf("counter scheduling metadata=%+v", artifact.Counters)
	}
	assertHostLive(t, collection.Capture.Samples[0].Live, .64, .32, .96)
	if collection.Capture.Samples[0].Rooflines.TheoreticalGBS != nil || collection.Capture.Samples[0].Rooflines.MeasuredSustainableGBS != nil || collection.Report.Observations[0].Rooflines.SelectedGBS != nil {
		t.Fatalf("system roofline applied to socket-scoped observation: %+v", collection.Report.Observations[0].Rooflines)
	}
}

func TestImportHostControllerWrap(t *testing.T) {
	collection := importHostControllerFixture(t, "host-controller-wrap.json", HostControllerImportOptions{
		Provider: "fixture-imc", Scope: HostControllerScope{Kind: "controller", ID: "imc0"}, Phase: PhaseOther, Shape: ShapeMedium,
	})
	read, write := collection.HostControllerArtifact.Counters[0], collection.HostControllerArtifact.Counters[1]
	if !read.Wrapped || read.DeltaRawValue != 16 || !write.Wrapped || write.DeltaRawValue != 20 {
		t.Fatalf("wrapped counters read=%+v write=%+v", read, write)
	}
}

func TestImportHostControllerRejectsUnsafeEvidence(t *testing.T) {
	tests := []struct {
		name    string
		fixture string
		scope   HostControllerScope
		want    string
	}{
		{name: "reset", fixture: "host-controller-reset.json", scope: HostControllerScope{Kind: "system"}, want: "reset during capture"},
		{name: "multiplexed", fixture: "host-controller-multiplexed.json", scope: HostControllerScope{Kind: "system"}, want: "below floor 0.90"},
		{name: "mixed-scope", fixture: "host-controller-mixed-scope.json", scope: HostControllerScope{Kind: "socket", ID: "0"}, want: "mixed host counter scopes"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			data, err := os.ReadFile(filepath.Join("testdata", tt.fixture))
			if err != nil {
				t.Fatal(err)
			}
			_, err = ImportHostControllerCounters(bytes.NewReader(data), HostControllerImportOptions{
				Provider: "fixture-imc", Scope: tt.scope, Phase: PhaseOther, Shape: ShapeSmall,
			})
			if err == nil || !strings.Contains(err.Error(), tt.want) {
				t.Fatalf("error=%v want %q", err, tt.want)
			}
		})
	}
}

func TestImportHostControllerRejectsAmbiguousResetAndNonpositiveInterval(t *testing.T) {
	raw, err := os.ReadFile(filepath.Join("testdata", "host-controller-wrap.json"))
	if err != nil {
		t.Fatal(err)
	}
	withoutResetState := bytes.ReplaceAll(raw, []byte(`, "reset_observed": false`), nil)
	_, err = ImportHostControllerCounters(bytes.NewReader(withoutResetState), HostControllerImportOptions{
		Provider: "fixture-imc", Scope: HostControllerScope{Kind: "controller", ID: "imc0"}, Phase: PhaseOther, Shape: ShapeSmall,
	})
	if err == nil || !strings.Contains(err.Error(), "reset state is ambiguous") {
		t.Fatalf("ambiguous reset error=%v", err)
	}

	direct, err := os.ReadFile(filepath.Join("testdata", "host-controller-direct.json"))
	if err != nil {
		t.Fatal(err)
	}
	nonpositive := bytes.ReplaceAll(direct, []byte("2026-08-27T10:00:02Z"), []byte("2026-08-27T10:00:00Z"))
	_, err = ImportHostControllerCounters(bytes.NewReader(nonpositive), HostControllerImportOptions{
		Provider: "fixture-imc", Scope: HostControllerScope{Kind: "system"}, Phase: PhaseOther, Shape: ShapeSmall,
	})
	if err == nil || !strings.Contains(err.Error(), "capture end must be after start") {
		t.Fatalf("nonpositive interval error=%v", err)
	}

	timingMismatch := bytes.ReplaceAll(direct, []byte("2000000000"), []byte("1000000000"))
	_, err = ImportHostControllerCounters(bytes.NewReader(timingMismatch), HostControllerImportOptions{
		Provider: "fixture-imc", Scope: HostControllerScope{Kind: "system"}, Phase: PhaseOther, Shape: ShapeSmall,
	})
	if err == nil || !strings.Contains(err.Error(), "conflicts with capture interval") {
		t.Fatalf("timing mismatch error=%v", err)
	}

	submillisecond := bytes.ReplaceAll(direct, []byte("2026-08-27T10:00:02Z"), []byte("2026-08-27T10:00:00.0005Z"))
	submillisecond = bytes.ReplaceAll(submillisecond, []byte("2000000000"), []byte("500000"))
	collection, err := ImportHostControllerCounters(bytes.NewReader(submillisecond), HostControllerImportOptions{
		Provider: "fixture-imc", Scope: HostControllerScope{Kind: "system"}, Phase: PhaseOther, Shape: ShapeSmall,
	})
	if err != nil {
		t.Fatal(err)
	}
	if collection.HostControllerArtifact.IntervalNS != 500_000 || collection.IntervalMS != 1 {
		t.Fatalf("submillisecond interval artifact_ns=%d collection_ms=%d", collection.HostControllerArtifact.IntervalNS, collection.IntervalMS)
	}

	mixedProvenance := bytes.Replace(direct,
		[]byte(`"event": "host_dram_write_bytes",
      "unit": "bytes",`),
		[]byte(`"event": "host_dram_write_bytes",
      "unit": "events",
      "bytes_per_event": 64,`), 1)
	_, err = ImportHostControllerCounters(bytes.NewReader(mixedProvenance), HostControllerImportOptions{
		Provider: "fixture-imc", Scope: HostControllerScope{Kind: "system"}, Phase: PhaseOther, Shape: ShapeSmall,
	})
	if err == nil || !strings.Contains(err.Error(), "mixed direct-byte and converted-event provenance") {
		t.Fatalf("mixed provenance error=%v", err)
	}
}

func TestImportHostControllerPerfStatJSONAndCSV(t *testing.T) {
	start := time.Date(2026, 8, 27, 10, 0, 0, 0, time.UTC)
	end := start.Add(2 * time.Second)
	jsonCollection := importHostControllerFixture(t, "host-controller-perf.json", HostControllerImportOptions{
		Provider: "linux-perf", Scope: HostControllerScope{Kind: "system"}, CaptureStartedAt: start, CaptureEndedAt: end,
		Phase: PhaseDecode, Shape: ShapeSmall,
	})
	jsonCounter := jsonCollection.HostControllerArtifact.Counters[0]
	if jsonCollection.HostControllerArtifact.ImportFormat != HostCounterFormatPerfJSON || jsonCounter.DeltaRawValue != 3_000_000_000 ||
		jsonCounter.TimeRunningNS == nil || *jsonCounter.TimeRunningNS != uint64(2*time.Second) || jsonCounter.TimeEnabledNS != nil || jsonCounter.RunningRatio != 1 {
		t.Fatalf("perf JSON artifact=%+v", jsonCollection.HostControllerArtifact)
	}
	assertHostLive(t, jsonCollection.Capture.Samples[0].Live, 1.5, .5, 2)

	conversion := uint64(64)
	csvCollection := importHostControllerFixture(t, "host-controller-perf.csv", HostControllerImportOptions{
		Format: HostCounterFormatPerfCSV, Provider: "linux-perf", Scope: HostControllerScope{Kind: "controller", ID: "imc0"},
		CaptureStartedAt: start, CaptureEndedAt: start.Add(100 * time.Millisecond), PerfBytesPerEvent: &conversion,
		Phase: PhaseDecode, Shape: ShapeSmall,
	})
	if csvCollection.HostControllerArtifact.ImportFormat != HostCounterFormatPerfCSV || csvCollection.HostControllerArtifact.RunningRatio != 1 {
		t.Fatalf("perf CSV artifact=%+v", csvCollection.HostControllerArtifact)
	}
	readCounter, writeCounter := csvCollection.HostControllerArtifact.Counters[0], csvCollection.HostControllerArtifact.Counters[1]
	if readCounter.TimeRunningNS == nil || *readCounter.TimeRunningNS != 100_000_000 || readCounter.TimeEnabledNS != nil ||
		writeCounter.TimeRunningNS == nil || *writeCounter.TimeRunningNS != 100_000_000 || writeCounter.TimeEnabledNS != nil ||
		readCounter.DeltaRawValue != 1_000_000 || readCounter.TrafficBytes != 64_000_000 {
		t.Fatalf("perf CSV scheduling/raw semantics read=%+v write=%+v", readCounter, writeCounter)
	}
	// The imported perf values are used as reported. Multiplexed perf rows are
	// rejected because perf's default counter scaling is not encoded in output.
	assertHostLive(t, csvCollection.Capture.Samples[0].Live, .64, .32, .96)
}

func TestImportHostControllerPerfScopeTimingAndScalingGuards(t *testing.T) {
	start := time.Date(2026, 8, 27, 10, 0, 0, 0, time.UTC)
	end := start.Add(100 * time.Millisecond)
	conversion := uint64(64)
	socketCSV := strings.Join([]string{
		"S0;4;1000000;;uncore_imc/cas_count_read/;400000000;100.00",
		"S0;4;500000;;uncore_imc/cas_count_write/;400000000;100.00",
	}, "\n")
	options := HostControllerImportOptions{
		Format: HostCounterFormatPerfCSV, Provider: "linux-perf", Scope: HostControllerScope{Kind: "socket", ID: "0"},
		CaptureStartedAt: start, CaptureEndedAt: end, PerfBytesPerEvent: &conversion, Phase: PhaseDecode, Shape: ShapeSmall,
	}
	collection, err := ImportHostControllerCounters(strings.NewReader(socketCSV), options)
	if err != nil {
		t.Fatal(err)
	}
	if collection.HostControllerArtifact.Scope != (HostControllerScope{Kind: "socket", ID: "0"}) {
		t.Fatalf("socket scope=%+v", collection.HostControllerArtifact.Scope)
	}
	if collection.HostControllerArtifact.Counters[0].PerfPMUCount == nil || *collection.HostControllerArtifact.Counters[0].PerfPMUCount != 4 {
		t.Fatalf("socket aggregate metadata=%+v", collection.HostControllerArtifact.Counters[0])
	}
	socketJSON := strings.Join([]string{
		`{"socket":"S0","counters":4,"counter-value":"64000000.000000","unit":"bytes","event":"dram_read","event-runtime":400000000,"pcnt-running":100.00}`,
		`{"socket":"S0","counters":4,"counter-value":"32000000.000000","unit":"bytes","event":"dram_write","event-runtime":400000000,"pcnt-running":100.00}`,
	}, "\n")
	jsonOptions := options
	jsonOptions.Format, jsonOptions.PerfBytesPerEvent = HostCounterFormatPerfJSON, nil
	jsonCollection, err := ImportHostControllerCounters(strings.NewReader(socketJSON), jsonOptions)
	if err != nil {
		t.Fatal(err)
	}
	assertHostLive(t, jsonCollection.Capture.Samples[0].Live, .64, .32, .96)

	tests := []struct {
		name   string
		format HostCounterImportFormat
		data   string
		scope  HostControllerScope
		want   string
	}{
		{
			name: "mixed-csv-sockets", format: HostCounterFormatPerfCSV,
			data: strings.Replace(socketCSV, "S0;4;500000", "S1;4;500000", 1), scope: HostControllerScope{Kind: "socket", ID: "0"},
			want: "mixed host counter scopes",
		},
		{
			name: "perf-provider-scaling-ambiguous", format: HostCounterFormatPerfCSV,
			data:  "1000000,,uncore_imc_0/cas_count_read/,95000000,95.00\n500000,,uncore_imc_0/cas_count_write/,100000000,100.00",
			scope: HostControllerScope{Kind: "controller", ID: "imc0"}, want: "counter-value scaling is ambiguous",
		},
		{
			name: "runtime-conflicts-with-capture", format: HostCounterFormatPerfCSV,
			data:  "1000000,,uncore_imc_0/cas_count_read/,50000000,100.00\n500000,,uncore_imc_0/cas_count_write/,100000000,100.00",
			scope: HostControllerScope{Kind: "controller", ID: "imc0"}, want: "conflicts with capture interval",
		},
		{
			name: "mixed-json-controllers", format: HostCounterFormatPerfJSON,
			data: `
{"counter-value":"100.000000","unit":"events","event":"uncore_imc_0/cas_count_read/","event-runtime":100000000,"pcnt-running":100.00}
{"counter-value":"100.000000","unit":"events","event":"uncore_imc_1/cas_count_write/","event-runtime":100000000,"pcnt-running":100.00}`,
			scope: HostControllerScope{Kind: "controller", ID: "imc0"}, want: "does not identify declared controller",
		},
		{
			name: "socket-json-missing-scope-metadata", format: HostCounterFormatPerfJSON,
			data: `
{"counter-value":"100.000000","unit":"events","event":"uncore_imc/cas_count_read/","event-runtime":100000000,"pcnt-running":100.00}
{"counter-value":"100.000000","unit":"events","event":"uncore_imc/cas_count_write/","event-runtime":100000000,"pcnt-running":100.00}`,
			scope: HostControllerScope{Kind: "socket", ID: "0"}, want: "requires per-socket metadata",
		},
		{
			name: "mixed-json-aggregate-counts", format: HostCounterFormatPerfJSON,
			data: `
{"socket":"S0","counters":4,"counter-value":"100.000000","unit":"events","event":"uncore_imc/cas_count_read/","event-runtime":400000000,"pcnt-running":100.00}
{"socket":"S0","counters":5,"counter-value":"100.000000","unit":"events","event":"uncore_imc/cas_count_write/","event-runtime":500000000,"pcnt-running":100.00}`,
			scope: HostControllerScope{Kind: "socket", ID: "0"}, want: "mixed perf aggregate counter counts",
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			o := options
			o.Format, o.Scope = tt.format, tt.scope
			_, err := ImportHostControllerCounters(strings.NewReader(tt.data), o)
			if err == nil || !strings.Contains(err.Error(), tt.want) {
				t.Fatalf("error=%v want %q", err, tt.want)
			}
		})
	}
}

func TestImportHostControllerRejectsProcessIOUnsupportedUnitsAndMissingPairs(t *testing.T) {
	start := time.Date(2026, 8, 27, 10, 0, 0, 0, time.UTC)
	baseOptions := HostControllerImportOptions{
		Format: HostCounterFormatPerfJSON, Provider: "linux-perf", Scope: HostControllerScope{Kind: "system"},
		CaptureStartedAt: start, CaptureEndedAt: start.Add(time.Second), Phase: PhaseOther, Shape: ShapeSmall,
	}
	tests := []struct {
		name string
		data string
		want string
	}{
		{
			name: "process-io",
			data: `[{
				"counter-value":"100","unit":"bytes","event":"process_read_bytes","event-runtime":"1000000000","pcnt-running":"100","direction":"read"
			}]`,
			want: "process/storage I/O",
		},
		{
			name: "storage-io",
			data: `[{"counter-value":"100.000000","unit":"bytes","event":"disk_dram_read_bytes","event-runtime":1000000000,"pcnt-running":100,"direction":"read"}]`,
			want: "process/storage I/O",
		},
		{
			name: "raw-proc-io-names",
			data: `[
				{"counter-value":"100.000000","unit":"bytes","event":"read_bytes","event-runtime":1000000000,"pcnt-running":100,"direction":"read"},
				{"counter-value":"100.000000","unit":"bytes","event":"write_bytes","event-runtime":1000000000,"pcnt-running":100,"direction":"write"}
			]`,
			want: "process/storage I/O",
		},
		{
			name: "unsupported-unit",
			data: `[
				{"counter-value":"100","unit":"MiB","event":"dram_read","event-runtime":"1000000000","pcnt-running":"100"},
				{"counter-value":"100","unit":"bytes","event":"dram_write","event-runtime":"1000000000","pcnt-running":"100"}
			]`,
			want: "unsupported host counter unit",
		},
		{
			name: "missing-pair",
			data: `[{"counter-value":"100","unit":"bytes","event":"dram_read","event-runtime":"1000000000","pcnt-running":"100"}]`,
			want: "read/write counter pair",
		},
		{
			name: "implicit-event-conversion",
			data: `[
				{"counter-value":"100","unit":"events","event":"cas_count_read","event-runtime":"1000000000","pcnt-running":"100"},
				{"counter-value":"100","unit":"events","event":"cas_count_write","event-runtime":"1000000000","pcnt-running":"100"}
			]`,
			want: "explicit positive bytes-per-event",
		},
		{
			name: "low-running-ratio",
			data: `[
				{"counter-value":"100.000000","unit":"bytes","event":"dram_read","event-runtime":890000000,"pcnt-running":89.00},
				{"counter-value":"100.000000","unit":"bytes","event":"dram_write","event-runtime":1000000000,"pcnt-running":100.00}
			]`,
			want: "below floor 0.90",
		},
		{
			name: "unavailable-counter",
			data: `[
				{"counter-value":"<not counted>","unit":"bytes","event":"dram_read","event-runtime":1000000000,"pcnt-running":100.00},
				{"counter-value":"100.000000","unit":"bytes","event":"dram_write","event-runtime":1000000000,"pcnt-running":100.00}
			]`,
			want: "counter-value is unavailable",
		},
		{
			name: "fractional-counter",
			data: `[
				{"counter-value":"100.5","unit":"bytes","event":"dram_read","event-runtime":1000000000,"pcnt-running":100.00},
				{"counter-value":"100.000000","unit":"bytes","event":"dram_write","event-runtime":1000000000,"pcnt-running":100.00}
			]`,
			want: "fractional hardware count",
		},
		{
			name: "missing-event-runtime",
			data: `[
				{"counter-value":"100.000000","unit":"bytes","event":"dram_read","pcnt-running":100.00},
				{"counter-value":"100.000000","unit":"bytes","event":"dram_write","event-runtime":1000000000,"pcnt-running":100.00}
			]`,
			want: "event-runtime is required",
		},
		{
			name: "mixed-perf-scope",
			data: `[
				{"socket":"S0","counter-value":"100.000000","unit":"bytes","event":"dram_read","event-runtime":1000000000,"pcnt-running":100.00},
				{"counter-value":"100.000000","unit":"bytes","event":"dram_write","event-runtime":1000000000,"pcnt-running":100.00}
			]`,
			want: "mixed host counter scopes",
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			_, err := ImportHostControllerCounters(strings.NewReader(tt.data), baseOptions)
			if err == nil || !strings.Contains(err.Error(), tt.want) {
				t.Fatalf("error=%v want %q", err, tt.want)
			}
		})
	}
}

func TestHostControllerArtifactJSONSchemaAndUnknownFieldRefusal(t *testing.T) {
	collection := importHostControllerFixture(t, "host-controller-direct.json", HostControllerImportOptions{
		Provider: "fixture-imc", Scope: HostControllerScope{Kind: "system"}, Phase: PhaseOther, Shape: ShapeSmall,
	})
	data, err := json.Marshal(collection)
	if err != nil {
		t.Fatal(err)
	}
	text := string(data)
	for _, want := range []string{
		`"schema":"fak-model-bandwidth-collection/1"`,
		`"host_controller_artifact":{"schema":"fak-host-controller-counters/1"`,
		`"raw_event":"host_dram_read_bytes"`,
		`"byte_provenance":"direct-bytes"`,
		`"running_ratio":1`,
		`"read_gb_s":1.5`,
		`"total_gb_s":2`,
	} {
		if !strings.Contains(text, want) {
			t.Fatalf("schema output missing %s: %s", want, text)
		}
	}
	for _, forbidden := range []string{`"process_read_bytes":0`, `"process_write_bytes":0`, `"physical_read_bytes":0`, `"physical_write_bytes":0`} {
		if strings.Contains(text, forbidden) {
			t.Fatalf("unavailable data serialized as zero: %s", text)
		}
	}

	raw, err := os.ReadFile(filepath.Join("testdata", "host-controller-direct.json"))
	if err != nil {
		t.Fatal(err)
	}
	raw = bytes.Replace(raw, []byte(`"provider": "fixture-imc"`), []byte(`"provider": "fixture-imc", "unknown": true`), 1)
	_, err = ImportHostControllerCounters(bytes.NewReader(raw), HostControllerImportOptions{
		Provider: "fixture-imc", Scope: HostControllerScope{Kind: "system"}, Phase: PhaseOther, Shape: ShapeSmall,
	})
	if err == nil || !strings.Contains(err.Error(), "unknown field") {
		t.Fatalf("unknown generic schema field error=%v", err)
	}
}

func TestApplyHostRooflineMeasurementMatchesImportedScope(t *testing.T) {
	system := importHostControllerFixture(t, "host-controller-direct.json", HostControllerImportOptions{
		Provider: "fixture-imc", Scope: HostControllerScope{Kind: "system"}, Phase: PhaseOther, Shape: ShapeSmall,
	})
	deviceRead, deviceWrite, deviceRoofline := 90.0, 10.0, 500.0
	system.Capture.Samples = append(system.Capture.Samples, BandwidthSample{
		Phase: PhaseOther, Shape: ShapeSmall,
		Provenance: BandwidthProvenance{Source: "live-amd-rocm", Device: "AMD fixture GPU", Collector: "fixture-device", SampledAt: "2026-08-27T10:00:02Z"},
		Rooflines:  Rooflines{MeasuredSustainableGBS: &deviceRoofline},
		Live:       LiveBandwidth{ReadGBS: &deviceRead, WriteGBS: &deviceWrite},
	})
	measurement := RooflineMeasurement{Scope: "host-memory", MeasuredSustainableGBS: 10}
	if err := ApplyHostRooflineMeasurement(&system, measurement); err != nil {
		t.Fatal(err)
	}
	if system.RooflineMeasurement.AppliedHostObservationCount != 1 || system.Report.Observations[0].Rooflines.SelectedGBS == nil || len(system.Report.Observations) != 2 {
		t.Fatalf("system scope not compared: measurement=%+v roofline=%+v", system.RooflineMeasurement, system.Report.Observations[0].Rooflines)
	}
	assertHostLive(t, system.Report.Observations[0].Live, 1.5, .5, 2)
	device := system.Report.Observations[1]
	if device.Rooflines.SelectedGBS == nil || *device.Rooflines.SelectedGBS != deviceRoofline || device.Live.TotalGBS == nil || *device.Live.TotalGBS != 100 {
		t.Fatalf("host import/roofline contaminated device sample: roofline=%+v live=%+v", device.Rooflines, device.Live)
	}

	controller := importHostControllerFixture(t, "host-controller-wrap.json", HostControllerImportOptions{
		Provider: "fixture-imc", Scope: HostControllerScope{Kind: "controller", ID: "imc0"}, Phase: PhaseOther, Shape: ShapeSmall,
	})
	if err := ApplyHostRooflineMeasurement(&controller, measurement); err == nil || !strings.Contains(err.Error(), "requires system-aggregate") {
		t.Fatalf("controller roofline application error=%v", err)
	}
	if controller.RooflineMeasurement != nil || controller.Report.Observations[0].Rooflines.SelectedGBS != nil {
		t.Fatalf("host roofline mutated controller scope: measurement=%+v roofline=%+v", controller.RooflineMeasurement, controller.Report.Observations[0].Rooflines)
	}

	raw, err := os.ReadFile(filepath.Join("testdata", "host-controller-wrap.json"))
	if err != nil {
		t.Fatal(err)
	}
	for name, options := range map[string]HostControllerImportOptions{
		"measured": {
			Provider: "fixture-imc", Scope: HostControllerScope{Kind: "controller", ID: "imc0"},
			Phase: PhaseOther, Shape: ShapeSmall, MeasuredHostGBS: &measurement.MeasuredSustainableGBS,
		},
		"theoretical": {
			Provider: "fixture-imc", Scope: HostControllerScope{Kind: "controller", ID: "imc0"},
			Phase: PhaseOther, Shape: ShapeSmall, TheoreticalGBS: &measurement.MeasuredSustainableGBS,
		},
	} {
		t.Run("reject-nonsystem-"+name+"-roofline", func(t *testing.T) {
			_, err := ImportHostControllerCounters(bytes.NewReader(raw), options)
			if err == nil || !strings.Contains(err.Error(), "require system-aggregate") {
				t.Fatalf("error=%v", err)
			}
		})
	}
}

func importHostControllerFixture(t *testing.T, name string, o HostControllerImportOptions) BandwidthCollection {
	t.Helper()
	data, err := os.ReadFile(filepath.Join("testdata", name))
	if err != nil {
		t.Fatal(err)
	}
	collection, err := ImportHostControllerCounters(bytes.NewReader(data), o)
	if err != nil {
		t.Fatal(err)
	}
	return collection
}

func assertHostLive(t *testing.T, live LiveBandwidth, read, write, total float64) {
	t.Helper()
	if live.ReadGBS == nil || live.WriteGBS == nil || live.TotalGBS == nil ||
		mathAbs(*live.ReadGBS-read) > 1e-12 || mathAbs(*live.WriteGBS-write) > 1e-12 || mathAbs(*live.TotalGBS-total) > 1e-12 {
		t.Fatalf("live=%+v want read=%g write=%g total=%g", live, read, write, total)
	}
}

func mathAbs(value float64) float64 {
	if value < 0 {
		return -value
	}
	return value
}
