package model

import (
	"os"
	"path/filepath"
	"regexp"
	"testing"
)

// These ops still compute on the host under device-only (gate/add, RMS scale,
// RMSNorm, gated pooling, shared add). Each stays pinned in the inventory until
// the change that moves it onto a device seam removes both the entry and its pin,
// so an empty inventory can never admit host compute.
var v41DeviceOnlyPinnedHostOps = []v41HostOp{
	{name: "engram-stream-injection", site: "Model.v41EngramInjectPrepared"},
	{name: "attention-input-norm", site: "Model.v41AttentionInputNormWithLeaf"},
	{name: "compressor-gated-pool", site: "V41CompressorPool.PushNormalized"},
}

// fak-test:runtime fast est=1s lane=default
func TestV41DeviceOnlyInventoryPinsUninventoriedHostOps(t *testing.T) {
	order := map[string]int{}
	for i, op := range v41DeviceOnlyHostOps {
		order[op.name] = i
	}
	sources, err := filepath.Glob("*.go")
	if err != nil {
		t.Fatal(err)
	}
	var corpus []byte
	for _, path := range sources {
		raw, err := os.ReadFile(path)
		if err != nil {
			t.Fatal(err)
		}
		corpus = append(corpus, raw...)
	}
	last := -1
	for _, pin := range v41DeviceOnlyPinnedHostOps {
		at, ok := order[pin.name]
		if !ok || v41DeviceOnlyHostOps[at].site != pin.site {
			t.Fatalf("host op %s (%s) is not inventoried; remove its pin only with its device seam", pin.name, pin.site)
		}
		if at < last {
			t.Fatalf("host op %s is out of forward order", pin.name)
		}
		last = at
		parts := regexp.MustCompile(`^(\w+)\.(\w+)$`).FindStringSubmatch(pin.site)
		decl := regexp.MustCompile(`func \(\w+ \*` + parts[1] + `\) ` + parts[2] + `\(`)
		if !decl.Match(corpus) {
			t.Fatalf("pinned host site %s no longer exists; re-point the inventory entry", pin.site)
		}
	}
	if err := v41DeviceOnlyRefusal(); err == nil {
		t.Fatal("device-only gate admits V4.1 while pinned host ops remain")
	}
}
