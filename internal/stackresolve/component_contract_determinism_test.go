package stackresolve

import (
	"context"
	"reflect"
	"sync"
	"testing"
)

func TestPublishedComponentContractsDeterminism(t *testing.T) {
	contracts := []ComponentContract{
		{Schema: ComponentContractSchema, Component: Component{ID: "runtime", Kind: "runtime", Version: "1", Provides: []string{"runtime"}}},
		{Schema: ComponentContractSchema, Component: Component{ID: "engine", Kind: "engine", Version: "1", Provides: []string{"engine"}}},
		{Schema: ComponentContractSchema, Component: Component{ID: "gateway", Kind: "gateway", Version: "1", Provides: []string{"gateway"}}},
	}
	baselineManifest, err := ComposeContracts("serve", []string{"runtime", "engine", "gateway"}, contracts)
	if err != nil {
		t.Fatal(err)
	}
	baseline, err := Resolve(context.Background(), baselineManifest.Workload, baselineManifest.Roots, ManifestProvider{Manifest: baselineManifest})
	if err != nil {
		t.Fatal(err)
	}

	const runs = 100
	errCh := make(chan string, runs)
	var wg sync.WaitGroup
	for i := 0; i < runs; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			manifest, composeErr := ComposeContracts("serve", []string{"gateway", "runtime", "engine"}, []ComponentContract{contracts[2], contracts[0], contracts[1]})
			if composeErr != nil {
				errCh <- composeErr.Error()
				return
			}
			receipt, resolveErr := Resolve(context.Background(), manifest.Workload, manifest.Roots, ManifestProvider{Manifest: manifest})
			if resolveErr != nil {
				errCh <- resolveErr.Error()
				return
			}
			if !reflect.DeepEqual(receipt, baseline) {
				errCh <- "receipt depends on contract or root input order"
			}
		}()
	}
	wg.Wait()
	close(errCh)
	for failure := range errCh {
		t.Error(failure)
	}
}
