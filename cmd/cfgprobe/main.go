// Command cfgprobe prints the MoE/dense FFN config axes a GGUF resolves to, so a
// dimension bug (e.g. expert_feed_forward_length not reaching cfg.MoEIntermediateSize)
// is caught from the metadata shard in seconds — no multi-minute model load. It is the
// cheap "what does the loader think the dims are?" diag the GLM-5.2 bring-up needs:
// the expert FFN width the forward requests (MoEIntermediateSize, falling back to the
// dense IntermediateSize when unset) must equal the stored expert tensor's out dim, or
// the resident-Q4_K read panics with a shape mismatch.
package main

import (
	"fmt"
	"os"

	"github.com/anthony-chaudhary/fak/internal/ggufload"
)

func main() {
	if len(os.Args) < 2 {
		fmt.Fprintln(os.Stderr, "usage: cfgprobe <gguf-path>")
		os.Exit(2)
	}
	f, err := ggufload.Open(os.Args[1])
	if err != nil {
		fmt.Fprintf(os.Stderr, "open: %v\n", err)
		os.Exit(1)
	}
	cfg, err := f.Config()
	if err != nil {
		fmt.Fprintf(os.Stderr, "config: %v\n", err)
		os.Exit(1)
	}
	// expertIntermediate() is unexported; replicate its rule (MoE width if set, else dense).
	expertFF := cfg.MoEIntermediateSize
	if expertFF <= 0 {
		expertFF = cfg.IntermediateSize
	}
	fmt.Printf("arch=%s\n", f.Metadata["general.architecture"].Value)
	fmt.Printf("HiddenSize=%d\n", cfg.HiddenSize)
	fmt.Printf("IntermediateSize(dense)=%d\n", cfg.IntermediateSize)
	fmt.Printf("MoEIntermediateSize=%d\n", cfg.MoEIntermediateSize)
	fmt.Printf("NumExperts=%d\n", cfg.NumExperts)
	fmt.Printf("NumExpertsPerTok=%d\n", cfg.NumExpertsPerTok)
	fmt.Printf("FirstKDenseReplace=%d\n", cfg.FirstKDenseReplace)
	fmt.Printf("SharedIntermediateSize=%d\n", cfg.SharedIntermediateSize)
	fmt.Printf("NSharedExperts=%d\n", cfg.NSharedExperts)
	fmt.Printf("=> expert gate_proj forward requests [out=%d, in=%d]\n", expertFF, cfg.HiddenSize)
}
