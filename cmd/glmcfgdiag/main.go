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
	"io"
	"os"

	"github.com/anthony-chaudhary/fak/internal/ggufload"
)

func run(args []string, stdout, stderr io.Writer) int {
	if len(args) < 1 {
		fmt.Fprintln(stderr, "usage: glmcfgdiag <gguf-shard>")
		return 2
	}
	f, err := ggufload.Open(args[0])
	if err != nil {
		fmt.Fprintln(stderr, "open:", err)
		return 1
	}
	cfg, err := f.Config()
	if err != nil {
		fmt.Fprintln(stderr, "config:", err)
		return 1
	}
	fmt.Fprintf(stdout, "ModelType=%s\n", cfg.ModelType)
	fmt.Fprintf(stdout, "NumLayers=%d NumHeads=%d HiddenSize=%d\n", cfg.NumLayers, cfg.NumHeads, cfg.HiddenSize)
	fmt.Fprintf(stdout, "QKNopeHeadDim=%d VHeadDim=%d QKRopeHeadDim=%d\n", cfg.QKNopeHeadDim, cfg.VHeadDim, cfg.QKRopeHeadDim)
	fmt.Fprintf(stdout, "KVLoraRank=%d QLoraRank=%d\n", cfg.KVLoraRank, cfg.QLoraRank)
	fmt.Fprintf(stdout, "NumExperts=%d NumExpertsPerTok=%d\n", cfg.NumExperts, cfg.NumExpertsPerTok)

	// The dim fix (c8f3606): per-head MLA dims must resolve to 192/256, NOT 512/512.
	if cfg.QKNopeHeadDim == 192 && cfg.VHeadDim == 256 {
		fmt.Fprintln(stdout, "DIMS_OK qkNope=192 vHead=256")
		return 0
	}
	fmt.Fprintf(stdout, "DIMS_WRONG want qkNope=192 vHead=256 got %d/%d\n", cfg.QKNopeHeadDim, cfg.VHeadDim)
	return 3
}

func main() { os.Exit(run(os.Args[1:], os.Stdout, os.Stderr)) }
