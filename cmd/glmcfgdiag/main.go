// Command glmcfgdiag is a cheap, no-reload GLM-5.2 config witness: it opens a GGUF
// shard's metadata, runs ggufload's real Config() derivation, and prints the MLA
// per-head dims (QKNopeHeadDim / VHeadDim) plus the latent ranks. It reads only the
// header, so it returns in seconds and never touches the ~466 GB of weights — the
// proven way to confirm the dim fix (qkNope=192, vHead=256) lands BEFORE paying for
// the ~1h41m full load. Exits non-zero if the dims are wrong, so it can gate a build.
//
//	go run ./cmd/glmcfgdiag /projects/glm52-q4/<shard1>.gguf
package main

import (
	"fmt"
	"os"

	"github.com/anthony-chaudhary/fak/internal/ggufload"
)

func main() {
	if len(os.Args) < 2 {
		fmt.Fprintln(os.Stderr, "usage: glmcfgdiag <gguf-shard>")
		os.Exit(2)
	}
	path := os.Args[1]
	f, err := ggufload.Open(path)
	if err != nil {
		fmt.Fprintln(os.Stderr, "open:", err)
		os.Exit(1)
	}
	cfg, err := f.Config()
	if err != nil {
		fmt.Fprintln(os.Stderr, "config:", err)
		os.Exit(1)
	}
	fmt.Printf("ModelType=%s\n", cfg.ModelType)
	fmt.Printf("NumLayers=%d NumHeads=%d HiddenSize=%d\n", cfg.NumLayers, cfg.NumHeads, cfg.HiddenSize)
	fmt.Printf("QKNopeHeadDim=%d VHeadDim=%d QKRopeHeadDim=%d\n", cfg.QKNopeHeadDim, cfg.VHeadDim, cfg.QKRopeHeadDim)
	fmt.Printf("KVLoraRank=%d QLoraRank=%d\n", cfg.KVLoraRank, cfg.QLoraRank)
	fmt.Printf("NumExperts=%d NumExpertsPerTok=%d\n", cfg.NumExperts, cfg.NumExpertsPerTok)

	// The dim fix (c8f3606): per-head MLA dims must resolve to 192/256, NOT 512/512.
	ok := cfg.QKNopeHeadDim == 192 && cfg.VHeadDim == 256
	if ok {
		fmt.Println("DIMS_OK qkNope=192 vHead=256")
	} else {
		fmt.Printf("DIMS_WRONG want qkNope=192 vHead=256 got %d/%d\n", cfg.QKNopeHeadDim, cfg.VHeadDim)
		os.Exit(3)
	}
}
