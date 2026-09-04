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

type bandwidthCollectFlags struct {
	count                      *int
	interval                   *time.Duration
	phase                      *string
	shape                      *string
	theoretical                *float64
	measured                   *float64
	deviceRoofline             *float64
	latency                    *float64
	promptTokens               *int64
	completionTokens           *int64
	logicalBytes               *uint64
	physicalReadBytes          *uint64
	physicalWriteBytes         *uint64
	nvidiaDevice               *string
	amdDevice                  *string
	nvidiaNCUCSV               *string
	hostCounterImport          *string
	hostCounterFormat          *string
	hostCounterProvider        *string
	hostCounterScope           *string
	hostCounterScopeID         *string
	hostCounterBytesPerEvent   *uint64
	appleMemoryImport          *string
	appleMemoryFormat          *string
	appleMemoryProvider        *string
	appleMemoryProviderVersion *string
	appleMemoryScope           *string
	appleMemoryInterval        *time.Duration
	profileDevice              *string
	captureStart               *string
	captureEnd                 *string
	measureRoofline            *bool
	rooflineBytes              *uint64
	rooflineTrials             *int
	rooflineDuration           *time.Duration
	rooflineThreads            *int
	rooflineSweep              *bool
	rooflineKneeThreshold      *float64
	output                     *string
	pretty                     *bool
	visited                    map[string]bool
	o                          modelperfobs.CollectionOptions
}

func parseBandwidthCollectFlags(args []string) (*bandwidthCollectFlags, error) {
	fs := flag.NewFlagSet("model-observe bandwidth collect", flag.ContinueOnError)
	fs.SetOutput(os.Stderr)
	f := &bandwidthCollectFlags{
		count:                      fs.Int("count", 1, "number of host samples"),
		interval:                   fs.Duration("interval", 100*time.Millisecond, "interval between host samples"),
		phase:                      fs.String("phase", "other", "request phase"),
		shape:                      fs.String("shape", "small", "request shape"),
		theoretical:                fs.Float64("theoretical-gb-s", 0, "theoretical memory roofline in GB/s"),
		measured:                   fs.Float64("measured-gb-s", 0, "measured sustainable memory roofline in GB/s"),
		deviceRoofline:             fs.Float64("device-roofline-gb-s", 0, "measured sustainable roofline for the profiled NVIDIA device"),
		latency:                    fs.Float64("latency-ms", 0, "request latency in milliseconds"),
		promptTokens:               fs.Int64("prompt-tokens", 0, "request prompt tokens"),
		completionTokens:           fs.Int64("completion-tokens", 0, "request completion tokens"),
		logicalBytes:               fs.Uint64("logical-bytes", 0, "logical software bytes"),
		physicalReadBytes:          fs.Uint64("physical-read-bytes", 0, "software physical read bytes; not a DRAM counter"),
		physicalWriteBytes:         fs.Uint64("physical-write-bytes", 0, "software physical write bytes; not a DRAM counter"),
		nvidiaDevice:               fs.String("nvidia-device", "", "NVIDIA device index or UUID; empty still probes device 0"),
		amdDevice:                  fs.String("amd-device", "", "AMD device index, BDF, or UUID; empty probes AMD device 0 after NVIDIA"),
		nvidiaNCUCSV:               fs.String("nvidia-ncu-csv", "", "import Nsight Compute raw CSV instead of sampling the host"),
		hostCounterImport:          fs.String("host-counter-import", "", "import host DRAM controller counters instead of live sampling"),
		hostCounterFormat:          fs.String("host-counter-format", "auto", "host counter format: auto, generic-json, perf-json, or perf-csv"),
		hostCounterProvider:        fs.String("host-counter-provider", "", "provider that produced the host controller counters"),
		hostCounterScope:           fs.String("host-counter-scope", "", "host counter scope: system, socket, or controller"),
		hostCounterScopeID:         fs.String("host-counter-scope-id", "", "socket/controller scope identifier"),
		hostCounterBytesPerEvent:   fs.Uint64("host-counter-bytes-per-event", 0, "explicit byte conversion for perf event counters"),
		appleMemoryImport:          fs.String("apple-memory-import", "", "import normalized Apple unified-memory counters instead of live sampling"),
		appleMemoryFormat:          fs.String("apple-memory-format", "auto", "Apple memory counter format: auto or generic-json"),
		appleMemoryProvider:        fs.String("apple-memory-provider", "", "provider that produced the normalized Apple memory counters"),
		appleMemoryProviderVersion: fs.String("apple-memory-provider-version", "", "exact Apple memory counter provider/tool version"),
		appleMemoryScope:           fs.String("apple-memory-scope", "", "Apple memory counter scope: system or package"),
		appleMemoryInterval:        fs.Duration("apple-memory-interval", 0, "explicit Apple counter capture interval; must match the artifact"),
		profileDevice:              fs.String("device", "", "profiled NVIDIA device name or UUID (required with --nvidia-ncu-csv)"),
		captureStart:               fs.String("capture-start", "", "import capture start time in RFC3339"),
		captureEnd:                 fs.String("capture-end", "", "import capture end time in RFC3339"),
		measureRoofline:            fs.Bool("measure-host-roofline", false, "benchmark and record sustainable host-memory bandwidth"),
		rooflineBytes:              fs.Uint64("roofline-bytes", 64<<20, "host roofline benchmark working set"),
		rooflineTrials:             fs.Int("roofline-trials", 5, "host roofline benchmark trial count"),
		rooflineDuration:           fs.Duration("roofline-duration", 100*time.Millisecond, "target duration per host roofline trial"),
		rooflineThreads:            fs.Int("roofline-threads", 0, "parallel host roofline workers (default: GOMAXPROCS)"),
		rooflineSweep:              fs.Bool("roofline-sweep", false, "measure a geometric host roofline worker-count curve through --roofline-threads"),
		rooflineKneeThreshold:      fs.Float64("roofline-knee-threshold", modelperfobs.DefaultRooflineKneeThreshold, "saturation knee fraction of the peak point median"),
		output:                     fs.String("output", "", "write collection JSON to this path"),
		pretty:                     fs.Bool("pretty", true, "indent JSON output"),
	}
	if err := fs.Parse(args); err != nil {
		return nil, err
	}
	f.o = modelperfobs.CollectionOptions{
		Count:        *f.count,
		Interval:     *f.interval,
		Phase:        modelperfobs.RequestPhase(*f.phase),
		Shape:        modelperfobs.RequestShape(*f.shape),
		NVIDIADevice: modelperfobs.NVIDIADeviceSelector(*f.nvidiaDevice),
		AMDDevice:    modelperfobs.AMDDeviceSelector(*f.amdDevice),
	}
	f.visited = make(map[string]bool)
	fs.Visit(func(flag *flag.Flag) {
		f.visited[flag.Name] = true
		switch flag.Name {
		case "theoretical-gb-s":
			f.o.TheoreticalGBS = f.theoretical
		case "measured-gb-s":
			f.o.MeasuredSustainableGBS = f.measured
		case "latency-ms":
			f.o.LatencyMS = f.latency
		case "prompt-tokens":
			f.o.PromptTokens = f.promptTokens
		case "completion-tokens":
			f.o.CompletionTokens = f.completionTokens
		case "logical-bytes":
			f.o.LogicalBytes = f.logicalBytes
		case "physical-read-bytes":
			f.o.PhysicalSoftwareRead = f.physicalReadBytes
		case "physical-write-bytes":
			f.o.PhysicalSoftwareWrite = f.physicalWriteBytes
		}
	})
	return f, nil
}

func runModelObserveBandwidthCollect(args []string) error {
	bcf, parseErr := parseBandwidthCollectFlags(args)
	if parseErr != nil {
		return parseErr
	}
	deviceRoofline := bcf.deviceRoofline
	nvidiaNCUCSV := bcf.nvidiaNCUCSV
	hostCounterImport, hostCounterFormat := bcf.hostCounterImport, bcf.hostCounterFormat
	hostCounterProvider, hostCounterScope := bcf.hostCounterProvider, bcf.hostCounterScope
	hostCounterScopeID, hostCounterBytesPerEvent := bcf.hostCounterScopeID, bcf.hostCounterBytesPerEvent
	appleMemoryImport, appleMemoryFormat := bcf.appleMemoryImport, bcf.appleMemoryFormat
	appleMemoryProvider, appleMemoryProviderVersion := bcf.appleMemoryProvider, bcf.appleMemoryProviderVersion
	appleMemoryScope, appleMemoryInterval := bcf.appleMemoryScope, bcf.appleMemoryInterval
	profileDevice := bcf.profileDevice
	captureStart, captureEnd := bcf.captureStart, bcf.captureEnd
	measureRoofline := bcf.measureRoofline
	rooflineBytes, rooflineTrials := bcf.rooflineBytes, bcf.rooflineTrials
	rooflineDuration, rooflineThreads := bcf.rooflineDuration, bcf.rooflineThreads
	rooflineSweep, rooflineKneeThreshold := bcf.rooflineSweep, bcf.rooflineKneeThreshold
	output, pretty := bcf.output, bcf.pretty
	visited, o := bcf.visited, bcf.o
	if *nvidiaNCUCSV != "" {
		for _, incompatible := range []string{"count", "interval", "measured-gb-s", "latency-ms", "prompt-tokens", "completion-tokens", "logical-bytes", "physical-read-bytes", "physical-write-bytes", "nvidia-device", "measure-host-roofline", "roofline-bytes", "roofline-trials", "roofline-duration", "roofline-threads", "roofline-sweep", "roofline-knee-threshold", "host-counter-import", "host-counter-format", "host-counter-provider", "host-counter-scope", "host-counter-scope-id", "host-counter-bytes-per-event", "apple-memory-import", "apple-memory-format", "apple-memory-provider", "apple-memory-provider-version", "apple-memory-scope", "apple-memory-interval"} {
			if visited[incompatible] {
				return fmt.Errorf("--%s cannot be combined with --nvidia-ncu-csv", incompatible)
			}
		}
	}
	if err := validateImportFlags(visited, *hostCounterImport, *hostCounterProvider, *hostCounterScope, *appleMemoryImport, *appleMemoryProvider, *appleMemoryProviderVersion, *appleMemoryScope, *appleMemoryInterval); err != nil {
		return err
	}
	rooflineFlags := []string{"roofline-bytes", "roofline-trials", "roofline-duration", "roofline-threads", "roofline-sweep", "roofline-knee-threshold"}
	if err := requireDependentFlags(visited, *measureRoofline, "measure-host-roofline", rooflineFlags); err != nil {
		return err
	}
	if visited["roofline-knee-threshold"] && !*rooflineSweep {
		return fmt.Errorf("--roofline-knee-threshold requires --roofline-sweep")
	}
	if *measureRoofline && visited["measured-gb-s"] {
		return fmt.Errorf("--measured-gb-s cannot be combined with --measure-host-roofline")
	}
	rooflineOptions := buildRooflineOptions(*measureRoofline, *rooflineSweep, *rooflineBytes, *rooflineTrials, *rooflineDuration, *rooflineThreads, *rooflineKneeThreshold)
	var collection modelperfobs.BandwidthCollection
	var err error
	if *appleMemoryImport != "" {
		in, started, ended, openErr := openCaptureFile(*captureStart, *captureEnd, *appleMemoryImport)
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
		in, started, ended, openErr := openCaptureFile(*captureStart, *captureEnd, *hostCounterImport)
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

func validateImportFlags(visited map[string]bool, hostCounterImport, hostCounterProvider, hostCounterScope, appleMemoryImport, appleMemoryProvider, appleMemoryProviderVersion, appleMemoryScope string, appleMemoryInterval time.Duration) error {
	if hostCounterImport != "" {
		if err := checkIncompatibleFlags(visited, "host-counter-import", []string{"count", "interval", "latency-ms", "prompt-tokens", "completion-tokens", "logical-bytes", "physical-read-bytes", "physical-write-bytes", "nvidia-device", "amd-device", "nvidia-ncu-csv", "device", "device-roofline-gb-s", "measure-host-roofline", "roofline-bytes", "roofline-trials", "roofline-duration", "roofline-threads", "roofline-sweep", "roofline-knee-threshold", "apple-memory-import", "apple-memory-format", "apple-memory-provider", "apple-memory-provider-version", "apple-memory-scope", "apple-memory-interval"}); err != nil {
			return err
		}
		if strings.TrimSpace(hostCounterProvider) == "" {
			return fmt.Errorf("--host-counter-provider is required with --host-counter-import")
		}
		if strings.TrimSpace(hostCounterScope) == "" {
			return fmt.Errorf("--host-counter-scope is required with --host-counter-import")
		}
	}
	if err := requireDependentFlags(visited, hostCounterImport != "", "host-counter-import", []string{"host-counter-format", "host-counter-provider", "host-counter-scope", "host-counter-scope-id", "host-counter-bytes-per-event"}); err != nil {
		return err
	}
	if appleMemoryImport != "" {
		if err := checkIncompatibleFlags(visited, "apple-memory-import", []string{"count", "interval", "latency-ms", "prompt-tokens", "completion-tokens", "logical-bytes", "physical-read-bytes", "physical-write-bytes", "nvidia-device", "amd-device", "nvidia-ncu-csv", "device", "device-roofline-gb-s", "host-counter-import", "host-counter-format", "host-counter-provider", "host-counter-scope", "host-counter-scope-id", "host-counter-bytes-per-event", "measure-host-roofline", "roofline-bytes", "roofline-trials", "roofline-duration", "roofline-threads", "roofline-sweep", "roofline-knee-threshold"}); err != nil {
			return err
		}
		if strings.TrimSpace(appleMemoryProvider) == "" {
			return fmt.Errorf("--apple-memory-provider is required with --apple-memory-import")
		}
		if strings.TrimSpace(appleMemoryProviderVersion) == "" {
			return fmt.Errorf("--apple-memory-provider-version is required with --apple-memory-import")
		}
		if strings.TrimSpace(appleMemoryScope) == "" {
			return fmt.Errorf("--apple-memory-scope is required with --apple-memory-import")
		}
		if visited["apple-memory-interval"] && appleMemoryInterval <= 0 {
			return fmt.Errorf("--apple-memory-interval must be positive")
		}
	}
	return requireDependentFlags(visited, appleMemoryImport != "", "apple-memory-import", []string{"apple-memory-format", "apple-memory-provider", "apple-memory-provider-version", "apple-memory-scope", "apple-memory-interval"})
}

func buildRooflineOptions(measureRoofline, sweep bool, bytes uint64, trials int, duration time.Duration, threads int, knee float64) *modelperfobs.RooflineBenchmarkOptions {
	if !measureRoofline {
		return nil
	}
	if threads == 0 {
		threads = runtime.GOMAXPROCS(0)
	}
	return &modelperfobs.RooflineBenchmarkOptions{
		WorkingSetBytes: bytes,
		Trials:          trials,
		TargetDuration:  duration,
		Threads:         threads,
		Sweep:           sweep,
		KneeThreshold:   knee,
	}
}

func openCaptureFile(start, end, path string) (*os.File, time.Time, time.Time, error) {
	started, ended, err := parseCaptureWindow(start, end)
	if err != nil {
		return nil, time.Time{}, time.Time{}, err
	}
	in, openErr := os.Open(path)
	if openErr != nil {
		return nil, time.Time{}, time.Time{}, openErr
	}
	return in, started, ended, nil
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

func checkIncompatibleFlags(visited map[string]bool, primary string, incompatibles []string) error {
	for _, inc := range incompatibles {
		if visited[inc] {
			return fmt.Errorf("--%s cannot be combined with --%s", inc, primary)
		}
	}
	return nil
}

func requireDependentFlags(visited map[string]bool, condition bool, parentFlag string, dependentFlags []string) error {
	if !condition {
		for _, flag := range dependentFlags {
			if visited[flag] {
				return fmt.Errorf("--%s requires --%s", flag, parentFlag)
			}
		}
	}
	return nil
}

func parseCaptureWindow(start, end string) (time.Time, time.Time, error) {
	if (start == "") != (end == "") {
		return time.Time{}, time.Time{}, fmt.Errorf("--capture-start and --capture-end must be supplied together")
	}
	var started, ended time.Time
	var err error
	if start != "" {
		started, err = time.Parse(time.RFC3339, start)
		if err != nil {
			return time.Time{}, time.Time{}, fmt.Errorf("--capture-start must be RFC3339: %w", err)
		}
	}
	if end != "" {
		ended, err = time.Parse(time.RFC3339, end)
		if err != nil {
			return time.Time{}, time.Time{}, fmt.Errorf("--capture-end must be RFC3339: %w", err)
		}
	}
	return started, ended, nil
}
