package main

import (
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"net/http"
	"os"
	"runtime"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/modelperfobs"
)

var measureHostMemoryRooflineForObserve = modelperfobs.MeasureHostMemoryRoofline

func cmdModelObserve(args []string) {
	if len(args) < 1 {
		fmt.Fprintln(os.Stderr, modelObserveUsage())
		os.Exit(2)
	}
	switch args[0] {
	case "proxy":
		fs := flag.NewFlagSet("model-observe proxy", flag.ExitOnError)
		listen := fs.String("listen", "127.0.0.1:8091", "proxy listen address")
		backend := fs.String("backend", "", "OpenAI-compatible backend base URL")
		ledger := fs.String("ledger", "model-perf.jsonl", "append-only observation JSONL")
		_ = fs.Parse(args[1:])
		u, err := modelperfobs.ParseBackend(*backend)
		if err != nil {
			fmt.Fprintln(os.Stderr, "model-observe:", err)
			os.Exit(2)
		}
		server := &http.Server{Addr: *listen, Handler: &modelperfobs.Proxy{Backend: u, Ledger: *ledger}, ReadHeaderTimeout: 10 * time.Second}
		fmt.Fprintf(os.Stderr, "model-observe: proxy http://%s -> %s; ledger=%s\n", *listen, u, *ledger)
		if err := server.ListenAndServe(); err != nil {
			fmt.Fprintln(os.Stderr, "model-observe:", err)
			os.Exit(1)
		}
	case "report":
		fs := flag.NewFlagSet("model-observe report", flag.ExitOnError)
		input := fs.String("input", "model-perf.jsonl", "observation JSONL")
		format := fs.String("format", "md", "md or json")
		_ = fs.Parse(args[1:])
		f, err := os.Open(*input)
		if err != nil {
			fmt.Fprintln(os.Stderr, "model-observe:", err)
			os.Exit(1)
		}
		defer f.Close()
		rows, err := modelperfobs.ReadObservations(f)
		if err != nil {
			fmt.Fprintln(os.Stderr, "model-observe:", err)
			os.Exit(1)
		}
		s := modelperfobs.Summarize(rows)
		if *format == "json" {
			if err := json.NewEncoder(os.Stdout).Encode(s); err != nil {
				os.Exit(1)
			}
			return
		}
		if *format != "md" {
			fmt.Fprintln(os.Stderr, "model-observe: format must be md or json")
			os.Exit(2)
		}
		if err := modelperfobs.WriteMarkdown(os.Stdout, s); err != nil {
			os.Exit(1)
		}
	case "bandwidth":
		if err := runModelObserveBandwidth(args[1:]); err != nil {
			fmt.Fprintln(os.Stderr, "model-observe:", err)
			os.Exit(1)
		}
	case "cache-state-bench":
		if err := runModelObserveStateBench(args[1:]); err != nil {
			fmt.Fprintln(os.Stderr, "model-observe:", err)
			os.Exit(1)
		}
	default:
		fmt.Fprintf(os.Stderr, "model-observe: unknown subcommand %q\n", args[0])
		os.Exit(2)
	}
}

func runModelObserveBandwidth(args []string) error {
	if len(args) > 0 {
		switch args[0] {
		case "collect":
			return runModelObserveBandwidthCollect(args[1:])
		case "numa-import":
			return runModelObserveNUMAImport(args[1:])
		case "numa-topology":
			return runModelObserveNUMATopology(args[1:])
		}
	}
	fs := flag.NewFlagSet("model-observe bandwidth", flag.ContinueOnError)
	fs.SetOutput(os.Stderr)
	input := fs.String("input", "", "read a fak-bandwidth-observation/1 capture from this JSON file")
	output := fs.String("output", "", "write the analyzed JSON report to this path (default: stdout)")
	pretty := fs.Bool("pretty", true, "indent JSON output")
	if err := fs.Parse(args); err != nil {
		return err
	}
	if *input == "" {
		return fmt.Errorf("bandwidth requires --input FILE")
	}
	f, err := os.Open(*input)
	if err != nil {
		return err
	}
	defer f.Close()
	dec := json.NewDecoder(f)
	dec.DisallowUnknownFields()
	var capture modelperfobs.BandwidthCapture
	if err := dec.Decode(&capture); err != nil {
		return err
	}
	report, err := modelperfobs.AnalyzeBandwidth(capture)
	if err != nil {
		return err
	}
	w := io.Writer(os.Stdout)
	var out *os.File
	if *output != "" {
		out, err = os.Create(*output)
		if err != nil {
			return err
		}
		defer out.Close()
		w = out
	}
	enc := json.NewEncoder(w)
	if *pretty {
		enc.SetIndent("", "  ")
	}
	return enc.Encode(report)
}
func runModelObserveNUMAImport(args []string) error {
	fs := flag.NewFlagSet("model-observe bandwidth numa-import", flag.ContinueOnError)
	fs.SetOutput(os.Stderr)
	input := fs.String("input", "", "read a fak-numa-roofline-capture/1 artifact")
	output := fs.String("output", "", "write the normalized matrix JSON (default: stdout)")
	pretty := fs.Bool("pretty", true, "indent JSON output")
	if err := fs.Parse(args); err != nil {
		return err
	}
	if *input == "" {
		return fmt.Errorf("--input is required")
	}
	in, err := os.Open(*input)
	if err != nil {
		return err
	}
	defer in.Close()
	matrix, err := modelperfobs.ImportNUMARooflineCapture(in)
	if err != nil {
		return err
	}
	return writeModelObserveJSON(*output, *pretty, matrix)
}

func runModelObserveNUMATopology(args []string) error {
	fs := flag.NewFlagSet("model-observe bandwidth numa-topology", flag.ContinueOnError)
	fs.SetOutput(os.Stderr)
	output := fs.String("output", "", "write discovered Linux NUMA topology JSON (default: stdout)")
	pretty := fs.Bool("pretty", true, "indent JSON output")
	if err := fs.Parse(args); err != nil {
		return err
	}
	topology, err := modelperfobs.DiscoverNUMATopology()
	if err != nil {
		return err
	}
	return writeModelObserveJSON(*output, *pretty, topology)
}

func writeModelObserveJSON(output string, pretty bool, value any) error {
	var w io.Writer = os.Stdout
	var out *os.File
	if output != "" {
		var err error
		out, err = os.Create(output)
		if err != nil {
			return err
		}
		defer out.Close()
		w = out
	}
	enc := json.NewEncoder(w)
	if pretty {
		enc.SetIndent("", "  ")
	}
	return enc.Encode(value)
}
func runModelObserveBandwidthCollect(args []string) error {
	fs := flag.NewFlagSet("model-observe bandwidth collect", flag.ContinueOnError)
	fs.SetOutput(os.Stderr)
	count := fs.Int("count", 1, "number of host samples")
	interval := fs.Duration("interval", 100*time.Millisecond, "interval between host samples")
	phase := fs.String("phase", "other", "request phase")
	shape := fs.String("shape", "small", "request shape")
	theoretical := fs.Float64("theoretical-gb-s", 0, "theoretical memory roofline in GB/s")
	measured := fs.Float64("measured-gb-s", 0, "measured sustainable memory roofline in GB/s")
	deviceRoofline := fs.Float64("device-roofline-gb-s", 0, "measured sustainable roofline for the profiled NVIDIA device")
	latency := fs.Float64("latency-ms", 0, "request latency in milliseconds")
	promptTokens := fs.Int64("prompt-tokens", 0, "request prompt tokens")
	completionTokens := fs.Int64("completion-tokens", 0, "request completion tokens")
	logicalBytes := fs.Uint64("logical-bytes", 0, "logical software bytes")
	physicalReadBytes := fs.Uint64("physical-read-bytes", 0, "software physical read bytes; not a DRAM counter")
	physicalWriteBytes := fs.Uint64("physical-write-bytes", 0, "software physical write bytes; not a DRAM counter")
	nvidiaDevice := fs.String("nvidia-device", "", "NVIDIA device index or UUID; empty still probes device 0")
	amdDevice := fs.String("amd-device", "", "AMD device index, BDF, or UUID; empty probes AMD device 0 after NVIDIA")
	nvidiaNCUCSV := fs.String("nvidia-ncu-csv", "", "import Nsight Compute raw CSV instead of sampling the host")
	hostCounterImport := fs.String("host-counter-import", "", "import host DRAM controller counters instead of live sampling")
	hostCounterFormat := fs.String("host-counter-format", "auto", "host counter format: auto, generic-json, perf-json, or perf-csv")
	hostCounterProvider := fs.String("host-counter-provider", "", "provider that produced the host controller counters")
	hostCounterScope := fs.String("host-counter-scope", "", "host counter scope: system, socket, or controller")
	hostCounterScopeID := fs.String("host-counter-scope-id", "", "socket/controller scope identifier")
	hostCounterBytesPerEvent := fs.Uint64("host-counter-bytes-per-event", 0, "explicit byte conversion for perf event counters")
	appleMemoryImport := fs.String("apple-memory-import", "", "import normalized Apple unified-memory counters instead of live sampling")
	appleMemoryFormat := fs.String("apple-memory-format", "auto", "Apple memory counter format: auto or generic-json")
	appleMemoryProvider := fs.String("apple-memory-provider", "", "provider that produced the normalized Apple memory counters")
	appleMemoryProviderVersion := fs.String("apple-memory-provider-version", "", "exact Apple memory counter provider/tool version")
	appleMemoryScope := fs.String("apple-memory-scope", "", "Apple memory counter scope: system or package")
	appleMemoryInterval := fs.Duration("apple-memory-interval", 0, "explicit Apple counter capture interval; must match the artifact")
	profileDevice := fs.String("device", "", "profiled NVIDIA device name or UUID (required with --nvidia-ncu-csv)")
	captureStart := fs.String("capture-start", "", "import capture start time in RFC3339")
	captureEnd := fs.String("capture-end", "", "import capture end time in RFC3339")
	measureRoofline := fs.Bool("measure-host-roofline", false, "benchmark and record sustainable host-memory bandwidth")
	rooflineBytes := fs.Uint64("roofline-bytes", 64<<20, "host roofline benchmark working set")
	rooflineTrials := fs.Int("roofline-trials", 5, "host roofline benchmark trial count")
	rooflineDuration := fs.Duration("roofline-duration", 100*time.Millisecond, "target duration per host roofline trial")
	rooflineThreads := fs.Int("roofline-threads", 0, "parallel host roofline workers (default: GOMAXPROCS)")
	rooflineSweep := fs.Bool("roofline-sweep", false, "measure a geometric host roofline worker-count curve through --roofline-threads")
	rooflineKneeThreshold := fs.Float64("roofline-knee-threshold", modelperfobs.DefaultRooflineKneeThreshold, "saturation knee fraction of the peak point median")
	output := fs.String("output", "", "write collection JSON to this path")
	pretty := fs.Bool("pretty", true, "indent JSON output")
	if err := fs.Parse(args); err != nil {
		return err
	}
	o := modelperfobs.CollectionOptions{Count: *count, Interval: *interval, Phase: modelperfobs.RequestPhase(*phase), Shape: modelperfobs.RequestShape(*shape), NVIDIADevice: modelperfobs.NVIDIADeviceSelector(*nvidiaDevice), AMDDevice: modelperfobs.AMDDeviceSelector(*amdDevice)}
	visited := make(map[string]bool)
	fs.Visit(func(f *flag.Flag) {
		visited[f.Name] = true
		switch f.Name {
		case "theoretical-gb-s":
			o.TheoreticalGBS = theoretical
		case "measured-gb-s":
			o.MeasuredSustainableGBS = measured
		case "latency-ms":
			o.LatencyMS = latency
		case "prompt-tokens":
			o.PromptTokens = promptTokens
		case "completion-tokens":
			o.CompletionTokens = completionTokens
		case "logical-bytes":
			o.LogicalBytes = logicalBytes
		case "physical-read-bytes":
			o.PhysicalSoftwareRead = physicalReadBytes
		case "physical-write-bytes":
			o.PhysicalSoftwareWrite = physicalWriteBytes
		}
	})
	if *nvidiaNCUCSV != "" {
		for _, incompatible := range []string{"count", "interval", "measured-gb-s", "latency-ms", "prompt-tokens", "completion-tokens", "logical-bytes", "physical-read-bytes", "physical-write-bytes", "nvidia-device", "measure-host-roofline", "roofline-bytes", "roofline-trials", "roofline-duration", "roofline-threads", "roofline-sweep", "roofline-knee-threshold", "host-counter-import", "host-counter-format", "host-counter-provider", "host-counter-scope", "host-counter-scope-id", "host-counter-bytes-per-event", "apple-memory-import", "apple-memory-format", "apple-memory-provider", "apple-memory-provider-version", "apple-memory-scope", "apple-memory-interval"} {
			if visited[incompatible] {
				return fmt.Errorf("--%s cannot be combined with --nvidia-ncu-csv", incompatible)
			}
		}
	}
	if *hostCounterImport != "" {
		for _, incompatible := range []string{"count", "interval", "latency-ms", "prompt-tokens", "completion-tokens", "logical-bytes", "physical-read-bytes", "physical-write-bytes", "nvidia-device", "amd-device", "nvidia-ncu-csv", "device", "device-roofline-gb-s", "measure-host-roofline", "roofline-bytes", "roofline-trials", "roofline-duration", "roofline-threads", "roofline-sweep", "roofline-knee-threshold", "apple-memory-import", "apple-memory-format", "apple-memory-provider", "apple-memory-provider-version", "apple-memory-scope", "apple-memory-interval"} {
			if visited[incompatible] {
				return fmt.Errorf("--%s cannot be combined with --host-counter-import", incompatible)
			}
		}
		if strings.TrimSpace(*hostCounterProvider) == "" {
			return fmt.Errorf("--host-counter-provider is required with --host-counter-import")
		}
		if strings.TrimSpace(*hostCounterScope) == "" {
			return fmt.Errorf("--host-counter-scope is required with --host-counter-import")
		}
	}
	if *hostCounterImport == "" {
		for _, importOnly := range []string{"host-counter-format", "host-counter-provider", "host-counter-scope", "host-counter-scope-id", "host-counter-bytes-per-event"} {
			if visited[importOnly] {
				return fmt.Errorf("--%s requires --host-counter-import", importOnly)
			}
		}
	}
	if *appleMemoryImport != "" {
		for _, incompatible := range []string{"count", "interval", "latency-ms", "prompt-tokens", "completion-tokens", "logical-bytes", "physical-read-bytes", "physical-write-bytes", "nvidia-device", "amd-device", "nvidia-ncu-csv", "device", "device-roofline-gb-s", "host-counter-import", "host-counter-format", "host-counter-provider", "host-counter-scope", "host-counter-scope-id", "host-counter-bytes-per-event", "measure-host-roofline", "roofline-bytes", "roofline-trials", "roofline-duration", "roofline-threads", "roofline-sweep", "roofline-knee-threshold"} {
			if visited[incompatible] {
				return fmt.Errorf("--%s cannot be combined with --apple-memory-import", incompatible)
			}
		}
		if strings.TrimSpace(*appleMemoryProvider) == "" {
			return fmt.Errorf("--apple-memory-provider is required with --apple-memory-import")
		}
		if strings.TrimSpace(*appleMemoryProviderVersion) == "" {
			return fmt.Errorf("--apple-memory-provider-version is required with --apple-memory-import")
		}
		if strings.TrimSpace(*appleMemoryScope) == "" {
			return fmt.Errorf("--apple-memory-scope is required with --apple-memory-import")
		}
		if visited["apple-memory-interval"] && *appleMemoryInterval <= 0 {
			return fmt.Errorf("--apple-memory-interval must be positive")
		}
	}
	if *appleMemoryImport == "" {
		for _, importOnly := range []string{"apple-memory-format", "apple-memory-provider", "apple-memory-provider-version", "apple-memory-scope", "apple-memory-interval"} {
			if visited[importOnly] {
				return fmt.Errorf("--%s requires --apple-memory-import", importOnly)
			}
		}
	}
	rooflineFlags := []string{"roofline-bytes", "roofline-trials", "roofline-duration", "roofline-threads", "roofline-sweep", "roofline-knee-threshold"}
	if !*measureRoofline {
		for _, name := range rooflineFlags {
			if visited[name] {
				return fmt.Errorf("--%s requires --measure-host-roofline", name)
			}
		}
	}
	if visited["roofline-knee-threshold"] && !*rooflineSweep {
		return fmt.Errorf("--roofline-knee-threshold requires --roofline-sweep")
	}
	if *measureRoofline && visited["measured-gb-s"] {
		return fmt.Errorf("--measured-gb-s cannot be combined with --measure-host-roofline")
	}
	var rooflineOptions *modelperfobs.RooflineBenchmarkOptions
	if *measureRoofline {
		threads := *rooflineThreads
		if threads == 0 {
			threads = runtime.GOMAXPROCS(0)
		}
		rooflineOptions = &modelperfobs.RooflineBenchmarkOptions{
			WorkingSetBytes: *rooflineBytes,
			Trials:          *rooflineTrials,
			TargetDuration:  *rooflineDuration,
			Threads:         threads,
			Sweep:           *rooflineSweep,
			KneeThreshold:   *rooflineKneeThreshold,
		}
	}
	var collection modelperfobs.BandwidthCollection
	var err error
	if *appleMemoryImport != "" {
		var started, ended time.Time
		if *captureStart != "" {
			started, err = time.Parse(time.RFC3339, *captureStart)
			if err != nil {
				return fmt.Errorf("--capture-start must be RFC3339: %w", err)
			}
		}
		if *captureEnd != "" {
			ended, err = time.Parse(time.RFC3339, *captureEnd)
			if err != nil {
				return fmt.Errorf("--capture-end must be RFC3339: %w", err)
			}
		}
		if (*captureStart == "") != (*captureEnd == "") {
			return fmt.Errorf("--capture-start and --capture-end must be supplied together")
		}
		in, openErr := os.Open(*appleMemoryImport)
		if openErr != nil {
			return openErr
		}
		defer in.Close()
		collection, err = modelperfobs.ImportAppleMemoryCounters(in, modelperfobs.AppleMemoryImportOptions{
			Format: modelperfobs.AppleMemoryImportFormat(*appleMemoryFormat), Provider: strings.TrimSpace(*appleMemoryProvider),
			ProviderVersion:  strings.TrimSpace(*appleMemoryProviderVersion),
			Scope:            modelperfobs.AppleMemoryScope{Kind: strings.TrimSpace(*appleMemoryScope)},
			CaptureStartedAt: started, CaptureEndedAt: ended, Interval: *appleMemoryInterval,
			Phase: o.Phase, Shape: o.Shape, TheoreticalGBS: o.TheoreticalGBS, MeasuredHostGBS: o.MeasuredSustainableGBS,
		})
		if err != nil {
			return err
		}
	} else if *hostCounterImport != "" {
		var started, ended time.Time
		if *captureStart != "" {
			started, err = time.Parse(time.RFC3339, *captureStart)
			if err != nil {
				return fmt.Errorf("--capture-start must be RFC3339: %w", err)
			}
		}
		if *captureEnd != "" {
			ended, err = time.Parse(time.RFC3339, *captureEnd)
			if err != nil {
				return fmt.Errorf("--capture-end must be RFC3339: %w", err)
			}
		}
		if (*captureStart == "") != (*captureEnd == "") {
			return fmt.Errorf("--capture-start and --capture-end must be supplied together")
		}
		in, openErr := os.Open(*hostCounterImport)
		if openErr != nil {
			return openErr
		}
		defer in.Close()
		importOptions := modelperfobs.HostControllerImportOptions{
			Format: modelperfobs.HostCounterImportFormat(*hostCounterFormat), Provider: strings.TrimSpace(*hostCounterProvider),
			Scope:            modelperfobs.HostControllerScope{Kind: strings.TrimSpace(*hostCounterScope), ID: strings.TrimSpace(*hostCounterScopeID)},
			CaptureStartedAt: started, CaptureEndedAt: ended, Phase: o.Phase, Shape: o.Shape,
			TheoreticalGBS: o.TheoreticalGBS, MeasuredHostGBS: o.MeasuredSustainableGBS,
		}
		if visited["host-counter-bytes-per-event"] {
			importOptions.PerfBytesPerEvent = hostCounterBytesPerEvent
		}
		collection, err = modelperfobs.ImportHostControllerCounters(in, importOptions)
		if err != nil {
			return err
		}
	} else if *nvidiaNCUCSV != "" {
		started, err := time.Parse(time.RFC3339, *captureStart)
		if err != nil {
			return fmt.Errorf("--capture-start must be RFC3339: %w", err)
		}
		ended, err := time.Parse(time.RFC3339, *captureEnd)
		if err != nil {
			return fmt.Errorf("--capture-end must be RFC3339: %w", err)
		}
		in, err := os.Open(*nvidiaNCUCSV)
		if err != nil {
			return err
		}
		defer in.Close()
		profileOptions := modelperfobs.NVIDIAProfileOptions{
			Phase:            o.Phase,
			Shape:            o.Shape,
			Device:           *profileDevice,
			CaptureStartedAt: started,
			CaptureEndedAt:   ended,
			TheoreticalGBS:   o.TheoreticalGBS,
		}
		if visited["device-roofline-gb-s"] {
			profileOptions.MeasuredDeviceGBS = deviceRoofline
		}
		collection, err = modelperfobs.ImportNVIDIAProfile(in, profileOptions)
		if err != nil {
			return err
		}
	} else {
		for _, profileOnly := range []string{"device", "device-roofline-gb-s", "capture-start", "capture-end"} {
			if visited[profileOnly] {
				return fmt.Errorf("--%s requires --nvidia-ncu-csv", profileOnly)
			}
		}
		runControl := context.Background()
		cancel := context.CancelFunc(func() {})
		var hostRoofline *modelperfobs.RooflineMeasurement
		if rooflineOptions != nil {
			rooflineBudget, budgetErr := modelperfobs.RooflineRuntimeBudget(*rooflineOptions)
			if budgetErr != nil {
				return budgetErr
			}
			totalBudget, budgetErr := modelObserveRooflineCollectBudget(o, rooflineBudget)
			if budgetErr != nil {
				return budgetErr
			}
			runControl, cancel = context.WithTimeout(runControl, totalBudget)
			measuredRoofline, measureErr := measureHostMemoryRooflineForObserve(runControl, *rooflineOptions)
			if measureErr != nil {
				cancel()
				return measureErr
			}
			measuredRoofline.CommandBudgetMS = totalBudget.Milliseconds()
			hostRoofline = &measuredRoofline
		}
		defer cancel()
		collection, err = modelperfobs.CollectBandwidth(runControl, o)
		if err != nil {
			return err
		}
		if hostRoofline != nil {
			if err := modelperfobs.ApplyHostRooflineMeasurement(&collection, *hostRoofline); err != nil {
				return err
			}
		}
	}
	w := io.Writer(os.Stdout)
	var f *os.File
	if *output != "" {
		f, err = os.Create(*output)
		if err != nil {
			return err
		}
		defer f.Close()
		w = f
	}
	enc := json.NewEncoder(w)
	if *pretty {
		enc.SetIndent("", "  ")
	}
	return enc.Encode(collection)
}

const (
	modelObserveRooflineCollectionSampleBudget = 5 * time.Second
	maxModelObserveRooflineCLIRuntimeBudget    = 10 * time.Minute
)

func modelObserveRooflineCollectBudget(o modelperfobs.CollectionOptions, rooflineBudget time.Duration) (time.Duration, error) {
	if err := modelperfobs.ValidateCollectionOptions(o); err != nil {
		return 0, err
	}
	intervalBudget := time.Duration(o.Count-1) * o.Interval
	sampleBudget := time.Duration(o.Count) * modelObserveRooflineCollectionSampleBudget
	total := rooflineBudget
	for _, part := range []time.Duration{intervalBudget, sampleBudget, modelObserveRooflineCollectionSampleBudget} {
		if part < 0 || total > time.Duration(1<<63-1)-part {
			return 0, fmt.Errorf("model-observe roofline collection runtime budget overflow")
		}
		total += part
	}
	if total > maxModelObserveRooflineCLIRuntimeBudget {
		return 0, fmt.Errorf("model-observe roofline collection worst-case runtime budget %s exceeds maximum %s", total, maxModelObserveRooflineCLIRuntimeBudget)
	}
	return total, nil
}

func runModelObserveStateBench(args []string) error {
	fs := flag.NewFlagSet("model-observe cache-state-bench", flag.ContinueOnError)
	fs.SetOutput(os.Stderr)
	output := fs.String("output", "", "write the observed JSON witness to this path (default: stdout)")
	verify := fs.String("verify", "", "verify a captured cache-state report instead of running")
	pretty := fs.Bool("pretty", true, "indent JSON output")
	if err := fs.Parse(args); err != nil {
		return err
	}
	if *verify != "" {
		f, err := os.Open(*verify)
		if err != nil {
			return err
		}
		defer f.Close()
		report, err := modelperfobs.ReadStateReport(f)
		if err != nil {
			return err
		}
		fmt.Fprintf(os.Stdout, "STATE_TRANSITION_WITNESS_OK arms=%d provenance=%s scope=%s\n", len(report.Arms), report.Provenance.EvidenceKind, report.Provenance.Scope)
		return nil
	}
	report, err := modelperfobs.RunHermeticStateBenchmark(context.Background())
	if err != nil {
		return err
	}
	w := io.Writer(os.Stdout)
	var f *os.File
	if *output != "" {
		f, err = os.Create(*output)
		if err != nil {
			return err
		}
		defer f.Close()
		w = f
	}
	if err := modelperfobs.WriteStateReport(w, report, *pretty); err != nil {
		return err
	}
	if *output != "" {
		fmt.Fprintf(os.Stderr, "STATE_TRANSITION_WITNESS_WRITTEN path=%s arms=%d provenance=%s scope=%s\n", *output, len(report.Arms), report.Provenance.EvidenceKind, report.Provenance.Scope)
	}
	return nil
}

func modelObserveUsage() string {
	return "usage: fak model-observe proxy --backend URL [--listen ADDR --ledger FILE]\n" +
		"       fak model-observe report --input FILE [--format md|json]\n" +
		"       fak model-observe bandwidth --input FILE [--output FILE --pretty=true]\n" +
		"       fak model-observe bandwidth collect --measure-host-roofline [--roofline-sweep --roofline-threads N --roofline-knee-threshold 0.9]\n" +
		"       fak model-observe bandwidth numa-topology [--output FILE --pretty=true]\n" +
		"       fak model-observe bandwidth numa-import --input FILE [--output FILE --pretty=true]\n" +
		"       fak model-observe bandwidth collect --nvidia-ncu-csv FILE --device DEVICE --capture-start RFC3339 --capture-end RFC3339 --phase PHASE --shape SHAPE [--device-roofline-gb-s N]\n" +
		"       fak model-observe bandwidth collect --host-counter-import FILE --host-counter-provider PROVIDER --host-counter-scope system|socket|controller [--host-counter-scope-id ID --host-counter-format FORMAT --capture-start RFC3339 --capture-end RFC3339]\n" +
		"       fak model-observe bandwidth collect --apple-memory-import FILE --apple-memory-provider PROVIDER --apple-memory-provider-version VERSION --apple-memory-scope system|package [--apple-memory-format generic-json --apple-memory-interval DURATION --capture-start RFC3339 --capture-end RFC3339]\n" +
		"       fak model-observe cache-state-bench [--output FILE --pretty=true]\n" +
		"       fak model-observe cache-state-bench --verify FILE"
}
