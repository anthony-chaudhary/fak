package stackresolve

import (
	"context"
	"strings"
	"testing"
)

func TestPublishedComponentContractsComposeAndResolve(t *testing.T) {
	contracts := []ComponentContract{
		parseContractForTest(t, `{"schema":"fak-component-contract/1","component":{"id":"cache.kv.radix@1","kind":"cache","version":"1.0.0","provides":["cache.kv.radix"],"relations":[{"kind":"requires","target":"kernel.paged-attention","evidence":{"authority":"cache-author","source":"cache/COMPATIBILITY.json"}},{"kind":"recommends","target":"runtime.cuda.graphs","evidence":{"authority":"cache-author","source":"benchmark-42"}}],"evidence":{"authority":"cache-author","source":"cache-release-1.0.0"}}}`),
		parseContractForTest(t, `{"schema":"fak-component-contract/1","component":{"id":"kernel.paged-attention@2","kind":"kernel","version":"2.1.0","provides":["kernel.paged-attention"],"relations":[{"kind":"requires","target":"runtime.cuda.12","evidence":{"authority":"kernel-author","source":"kernel/COMPATIBILITY.json"}}],"evidence":{"authority":"kernel-author","source":"kernel-release-2.1.0"}}}`),
		parseContractForTest(t, `{"schema":"fak-component-contract/1","component":{"id":"runtime.cuda@12.9","kind":"runtime","version":"12.9","provides":["runtime.cuda.12"],"evidence":{"authority":"runtime-author","source":"runtime-release-12.9"}}}`),
	}
	manifest, err := ComposeContracts("radix-cache", []string{"cache.kv.radix@1"}, contracts)
	if err != nil {
		t.Fatal(err)
	}
	receipt, err := Resolve(context.Background(), manifest.Workload, manifest.Roots, ManifestProvider{Manifest: manifest})
	if err != nil {
		t.Fatal(err)
	}
	if receipt.Status != "allow" || len(receipt.Selected) != 3 {
		t.Fatalf("receipt=%+v", receipt)
	}
	if len(receipt.Warnings) != 1 || receipt.Warnings[0].Wanted != "runtime.cuda.graphs" {
		t.Fatalf("warnings=%+v", receipt.Warnings)
	}
}

func TestPublishedContractMissingHardRequirementRefuses(t *testing.T) {
	cache := parseContractForTest(t, `{"schema":"fak-component-contract/1","component":{"id":"cache@1","kind":"cache","version":"1","relations":[{"kind":"requires","target":"kernel.fast","evidence":{"authority":"author","source":"contract"}}],"evidence":{"authority":"author","source":"release"}}}`)
	manifest, err := ComposeContracts("cache-check", []string{"cache@1"}, []ComponentContract{cache})
	if err != nil {
		t.Fatal(err)
	}
	receipt, err := Resolve(context.Background(), manifest.Workload, manifest.Roots, ManifestProvider{Manifest: manifest})
	if err != nil {
		t.Fatal(err)
	}
	if receipt.Status != "refuse" || receipt.Conflict == nil || receipt.Conflict.Wanted != "kernel.fast" {
		t.Fatalf("receipt=%+v", receipt)
	}
}

func TestComponentContractRejectsUnknownAndDuplicate(t *testing.T) {
	if _, err := ParseComponentContract([]byte(`{"schema":"fak-component-contract/1","component":{},"surprise":true}`)); err == nil || !strings.Contains(err.Error(), "unknown field") {
		t.Fatalf("err=%v", err)
	}
	c := parseContractForTest(t, `{"schema":"fak-component-contract/1","component":{"id":"same@1","kind":"cache","version":"1","evidence":{"authority":"a","source":"s"}}}`)
	if _, err := ComposeContracts("duplicate", []string{"same@1"}, []ComponentContract{c, c}); err == nil || !strings.Contains(err.Error(), "duplicate component") {
		t.Fatalf("err=%v", err)
	}
}

func parseContractForTest(t *testing.T, raw string) ComponentContract {
	t.Helper()
	contract, err := ParseComponentContract([]byte(raw))
	if err != nil {
		t.Fatal(err)
	}
	return contract
}
