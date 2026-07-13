package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"math"
	"os"

	"github.com/anthony-chaudhary/fak/internal/macfit"
)

func cmdMacFit(argv []string) { os.Exit(runMacFit(os.Stdout, os.Stderr, argv)) }

func runMacFit(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("macfit", flag.ContinueOnError)
	fs.SetOutput(stderr)
	memoryGiB := fs.Float64("memory-gib", 36, "unified-memory capacity in GiB")
	reserveGiB := fs.Float64("reserve-gib", 6, "OS/runtime/headroom reserve in GiB")
	weightsGiB := fs.Float64("weights-gib", 4.5, "resident model weight bytes in GiB")
	context := fs.Uint64("context", 32768, "full independent context tokens")
	layers := fs.Uint64("layers", 28, "transformer layer count")
	kvHeads := fs.Uint64("kv-heads", 4, "KV head count")
	headDim := fs.Uint64("head-dim", 128, "per-head dimension")
	kvBytes := fs.Uint64("kv-bytes", 2, "bytes per KV element (for example FP16=2)")
	prefix := fs.Uint64("shared-prefix", 8192, "system+tools prefix tokens stored once")
	tailCap := fs.Uint64("tail-cap", 8192, "O(1) private-tail cap tokens per agent")
	asJSON := fs.Bool("json", false, "emit the fak-macfit/1 JSON envelope")
	if rc, ok := parseFlagsOrHelp(fs, argv); !ok {
		return rc
	}
	if fs.NArg() != 0 {
		fmt.Fprintln(stderr, "fak macfit: unexpected positional arguments")
		return 2
	}
	toBytes := func(name string, v float64) (uint64, bool) {
		if math.IsNaN(v) || math.IsInf(v, 0) || v < 0 || v > float64(^uint64(0))/(1<<30) {
			fmt.Fprintf(stderr, "fak macfit: %s must be a non-negative finite GiB value\n", name)
			return 0, false
		}
		return uint64(v * (1 << 30)), true
	}
	memory, ok := toBytes("--memory-gib", *memoryGiB)
	if !ok {
		return 2
	}
	reserve, ok := toBytes("--reserve-gib", *reserveGiB)
	if !ok {
		return 2
	}
	weights, ok := toBytes("--weights-gib", *weightsGiB)
	if !ok {
		return 2
	}
	r, err := macfit.Calculate(macfit.Input{MemoryBytes: memory, ReserveBytes: reserve, WeightBytes: weights, ContextTokens: *context, Layers: *layers, KVHeads: *kvHeads, HeadDim: *headDim, KVBytesPerElement: *kvBytes, SharedPrefixTokens: *prefix, TailCapTokens: *tailCap})
	if err != nil {
		fmt.Fprintln(stderr, "fak macfit:", err)
		return 2
	}
	if *asJSON {
		enc := json.NewEncoder(stdout)
		enc.SetIndent("", "  ")
		if err := enc.Encode(r); err != nil {
			fmt.Fprintln(stderr, "fak macfit:", err)
			return 1
		}
		return 0
	}
	fmt.Fprintln(stdout, "fak macfit — modeled unified-memory capacity (not a hardware measurement)")
	fmt.Fprintf(stdout, "weights: %.2f GiB  KV pool after reserve+weights: %.2f GiB  KV/token: %d B\n", float64(r.WeightBytes)/(1<<30), float64(r.KVPoolBytes)/(1<<30), r.KVBytesPerToken)
	fmt.Fprintf(stdout, "caching off: %d agents (%d bytes KV/agent)\n", r.OffAgentsThatFit, r.OffKVBytesPerAgent)
	fmt.Fprintf(stdout, "caching on : %d agents (%d shared-prefix bytes once + %d tail bytes/agent)\n", r.OnAgentsThatFit, r.OnSharedKVBytes, r.OnTailKVBytesPerAgent)
	fmt.Fprintf(stdout, "delta: +%d agents", r.ExtraAgents)
	if r.CrossoverFound {
		fmt.Fprintf(stdout, "  crossover: %d context tokens", r.CrossoverContextTokens)
	}
	fmt.Fprintln(stdout)
	return 0
}
