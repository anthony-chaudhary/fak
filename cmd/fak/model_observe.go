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
	"time"

	"github.com/anthony-chaudhary/fak/internal/modelperfobs"
)

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
	if len(args) > 0 && args[0] == "collect" {
		return runModelObserveBandwidthCollect(args[1:])
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
func runModelObserveBandwidthCollect(args []string) error {
	fs := flag.NewFlagSet("model-observe bandwidth collect", flag.ContinueOnError)
	fs.SetOutput(os.Stderr)
	count := fs.Int("count", 1, "number of host samples")
	interval := fs.Duration("interval", 100*time.Millisecond, "interval between host samples")
	phase := fs.String("phase", "other", "request phase")
	shape := fs.String("shape", "small", "request shape")
	theoretical := fs.Float64("theoretical-gb-s", 0, "theoretical memory roofline in GB/s")
	measured := fs.Float64("measured-gb-s", 0, "measured sustainable memory roofline in GB/s")
	latency := fs.Float64("latency-ms", 0, "request latency in milliseconds")
	promptTokens := fs.Int64("prompt-tokens", 0, "request prompt tokens")
	completionTokens := fs.Int64("completion-tokens", 0, "request completion tokens")
	logicalBytes := fs.Uint64("logical-bytes", 0, "logical software bytes")
	physicalReadBytes := fs.Uint64("physical-read-bytes", 0, "software physical read bytes; not a DRAM counter")
	physicalWriteBytes := fs.Uint64("physical-write-bytes", 0, "software physical write bytes; not a DRAM counter")
	nvidiaDevice := fs.String("nvidia-device", "", "NVIDIA device index or UUID; empty still probes device 0")
	amdDevice := fs.String("amd-device", "", "AMD device index, BDF, or UUID; empty probes AMD device 0 after NVIDIA")
	measureRoofline := fs.Bool("measure-host-roofline", false, "benchmark and record sustainable host-memory bandwidth")
	rooflineBytes := fs.Uint64("roofline-bytes", 64<<20, "host roofline benchmark working set")
	rooflineTrials := fs.Int("roofline-trials", 5, "host roofline benchmark trial count")
	rooflineDuration := fs.Duration("roofline-duration", 100*time.Millisecond, "target duration per host roofline trial")
	rooflineThreads := fs.Int("roofline-threads", 0, "parallel host roofline workers (default: GOMAXPROCS)")
	output := fs.String("output", "", "write collection JSON to this path")
	pretty := fs.Bool("pretty", true, "indent JSON output")
	if err := fs.Parse(args); err != nil {
		return err
	}
	o := modelperfobs.CollectionOptions{Count: *count, Interval: *interval, Phase: modelperfobs.RequestPhase(*phase), Shape: modelperfobs.RequestShape(*shape), NVIDIADevice: modelperfobs.NVIDIADeviceSelector(*nvidiaDevice), AMDDevice: modelperfobs.AMDDeviceSelector(*amdDevice)}
	if *measureRoofline {
		threads := *rooflineThreads
		if threads == 0 {
			threads = runtime.GOMAXPROCS(0)
		}
		o.MeasureHostRoofline = &modelperfobs.RooflineBenchmarkOptions{WorkingSetBytes: *rooflineBytes, Trials: *rooflineTrials, TargetDuration: *rooflineDuration, Threads: threads}
	}
	fs.Visit(func(f *flag.Flag) {
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
	collection, err := modelperfobs.CollectBandwidth(context.Background(), o)
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
	enc := json.NewEncoder(w)
	if *pretty {
		enc.SetIndent("", "  ")
	}
	return enc.Encode(collection)
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
		"       fak model-observe cache-state-bench [--output FILE --pretty=true]\n" +
		"       fak model-observe cache-state-bench --verify FILE"
}
