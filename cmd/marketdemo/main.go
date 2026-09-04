package main

import (
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"os"
	"path/filepath"

	market "github.com/anthony-chaudhary/fak/internal/marketplace"
)

func main() {
	selfcheck := flag.Bool("selfcheck", false, "run offline descriptor conformance witness")
	asJSON := flag.Bool("json", false, "print the inert descriptor catalog")
	flag.Parse()
	root, err := os.MkdirTemp("", "fak-marketdemo-")
	if err != nil {
		fatal(err)
	}
	defer os.RemoveAll(root)
	artifact := []byte("trusted compiled extension fixture\n")
	if err := os.WriteFile(filepath.Join(root, "fixture.bin"), artifact, 0o600); err != nil {
		fatal(err)
	}
	d := market.ComputeAdapter.Descriptor(market.Descriptor{ID: "example.org/fixture", Module: "internal/compute@r1+gabcdef0", Compatibility: market.Compatibility{Min: 1, Max: 1}, Artifact: "fixture.bin", ArtifactSHA256: market.SHA256(artifact), Trust: market.TrustCompiled, OnError: market.ErrorClosed, Capabilities: []string{"compute.execute"}})
	c := market.Catalog{Schema: market.Schema, Extensions: []market.Descriptor{d}}
	if *asJSON {
		b, _ := market.Marshal(c)
		fmt.Println(string(b))
		return
	}
	if *selfcheck {
		b, _ := json.Marshal(c)
		parsed, err := market.Parse(b, map[string]int{"fak-compute": 1})
		if err != nil {
			fatal(err)
		}
		r, err := market.Verify(context.Background(), parsed, market.VerifyOptions{Root: root})
		if err != nil {
			fatal(err)
		}
		fmt.Printf("PASS %s: %d descriptors locally reverified; metadata never executed\n", r.Schema, r.Descriptors)
		return
	}
	flag.Usage()
}
func fatal(err error) { fmt.Fprintln(os.Stderr, "marketdemo:", err); os.Exit(1) }
