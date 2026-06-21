// Command ggufprobe dumps a GGUF file's architecture using fak's own
// internal/ggufload parser — no llama.cpp. It prints the version, metadata KVs
// (large arrays summarized), the tensor-dtype histogram, a sample of tensor
// names, and whether fak's model.Config can be derived. It is the read-side
// witness for the GGUF loader work (#89) and for mapping a new architecture
// (e.g. qwen35 / Qwen3.6-27B, epic #88) against what the in-kernel engine
// currently supports.
package main

import (
	"flag"
	"fmt"
	"math"
	"os"
	"sort"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/ggufload"
)

func maxInt(a, b int) int {
	if a > b {
		return a
	}
	return b
}

func main() {
	full := flag.Bool("full", false, "print every tensor name (default: a representative sample)")
	dump := flag.String("dump", "", "comma-separated GGUF tensor names to dequant + summarize (mean/min/max/head)")
	flag.Parse()
	if flag.NArg() != 1 {
		fmt.Fprintln(os.Stderr, "usage: ggufprobe [-full] [-dump name,...] <model.gguf>")
		os.Exit(2)
	}
	path := flag.Arg(0)

	if *dump != "" {
		ws, err := ggufload.OpenWeights(path)
		if err != nil {
			fmt.Fprintln(os.Stderr, "openweights:", err)
			os.Exit(1)
		}
		defer ws.Close()
		for _, name := range strings.Split(*dump, ",") {
			name = strings.TrimSpace(name)
			data, info, err := ws.TensorF32(name)
			if err != nil {
				fmt.Printf("  %s: ERROR %v\n", name, err)
				continue
			}
			var sum, mn, mx float64
			mn, mx = math.MaxFloat64, -math.MaxFloat64
			for _, v := range data {
				sum += float64(v)
				if float64(v) < mn {
					mn = float64(v)
				}
				if float64(v) > mx {
					mx = float64(v)
				}
			}
			n := len(data)
			head := data
			if n > 6 {
				head = data[:6]
			}
			fmt.Printf("  %-40s type=%s n=%d mean=%.5f min=%.5f max=%.5f head=%v\n",
				name, info.Type, n, sum/float64(maxInt(n, 1)), mn, mx, head)
		}
		return
	}

	f, err := ggufload.Open(path)
	if err != nil {
		fmt.Fprintln(os.Stderr, "open:", err)
		os.Exit(1)
	}

	fmt.Printf("file:    %s\n", path)
	fmt.Printf("version: %d\n", f.Version)
	if arch, ok := f.String("general.architecture"); ok {
		fmt.Printf("arch:    %s\n", arch)
	}
	fmt.Printf("tensors: %d\n", len(f.Tensors))

	// Metadata, sorted, with big arrays summarized so a 248k-token vocab does not
	// flood the output.
	fmt.Println("\n--- metadata ---")
	keys := make([]string, 0, len(f.Metadata))
	for k := range f.Metadata {
		keys = append(keys, k)
	}
	sort.Strings(keys)
	for _, k := range keys {
		fmt.Printf("  %-44s %s\n", k, summarize(f.Metadata[k].Value))
	}

	// Tensor dtype histogram + name sample.
	fmt.Println("\n--- tensor dtypes ---")
	hist := map[string]int{}
	for _, t := range f.Tensors {
		hist[dtypeName(t.Type)]++
	}
	dts := make([]string, 0, len(hist))
	for d := range hist {
		dts = append(dts, d)
	}
	sort.Strings(dts)
	for _, d := range dts {
		fmt.Printf("  %-10s %d\n", d, hist[d])
	}

	fmt.Println("\n--- tensor names ---")
	names := make([]string, len(f.Tensors))
	for i, t := range f.Tensors {
		names[i] = fmt.Sprintf("%s  %v  %s", t.Name, t.Dims, dtypeName(t.Type))
	}
	if *full {
		for _, n := range names {
			fmt.Println("  " + n)
		}
	} else {
		// One block-0 sample + the non-blk (global) tensors — enough to read the arch.
		for _, n := range names {
			if strings.HasPrefix(n, "blk.0.") || !strings.HasPrefix(n, "blk.") {
				fmt.Println("  " + n)
			}
		}
		fmt.Println("  ... (use -full for all blocks)")
	}

	// Can fak's in-kernel engine derive a runnable Config from this file?
	fmt.Println("\n--- fak model.Config derivation ---")
	cfg, cerr := f.Config()
	if cerr != nil {
		fmt.Printf("  Config() error: %v\n", cerr)
	} else {
		fmt.Printf("  hidden=%d layers=%d heads=%d kv_heads=%d head_dim=%d inter=%d vocab=%d rope_theta=%.0f\n",
			cfg.HiddenSize, cfg.NumLayers, cfg.NumHeads, cfg.NumKVHeads, cfg.HeadDim,
			cfg.IntermediateSize, cfg.VocabSize, cfg.RopeTheta)
	}
}

// dtypeName maps the GGUF tensor type enum to a readable name.
func dtypeName(t ggufload.TensorType) string {
	switch t {
	case ggufload.TensorF32:
		return "F32"
	case ggufload.TensorF16:
		return "F16"
	case ggufload.TensorQ8_0:
		return "Q8_0"
	case ggufload.TensorQ2_K:
		return "Q2_K"
	case ggufload.TensorQ4_K:
		return "Q4_K"
	case ggufload.TensorQ6_K:
		return "Q6_K"
	case ggufload.TensorBF16:
		return "BF16"
	case ggufload.TensorMXFP4:
		return "MXFP4"
	default:
		return fmt.Sprintf("type%d", uint32(t))
	}
}

// summarize renders a metadata value, collapsing long slices to a count + head.
func summarize(v any) string {
	switch s := v.(type) {
	case []string:
		if len(s) > 6 {
			return fmt.Sprintf("[]string len=%d  e.g. %q", len(s), s[:4])
		}
		return fmt.Sprintf("%q", s)
	case []int32:
		if len(s) > 8 {
			return fmt.Sprintf("[]int32 len=%d  e.g. %v", len(s), s[:6])
		}
		return fmt.Sprintf("%v", s)
	case []float32:
		if len(s) > 8 {
			return fmt.Sprintf("[]float32 len=%d", len(s))
		}
		return fmt.Sprintf("%v", s)
	default:
		str := fmt.Sprintf("%v", v)
		if len(str) > 80 {
			return str[:77] + "..."
		}
		return str
	}
}
