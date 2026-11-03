package main

import (
	"testing"

	"github.com/anthony-chaudhary/fak/internal/modelengine"
)

// fak-test:runtime fast est=1s lane=default
func TestServeNativeHostSwapBudget(t *testing.T) {
	fs, sf := newServeFlagSet()
	if err := fs.Parse([]string{"--native-kv-host-swap-bytes", "1048576"}); err != nil {
		t.Fatal(err)
	}
	if *sf.nativeHostSwapBytes != 1048576 {
		t.Fatal("flag did not bind its exact byte value")
	}
	if err := configureServeHostSwapBudget(sf, nil); err == nil {
		t.Fatal("positive budget silently accepted a non-native engine")
	}
	t.Setenv("FAK_MODEL_DIR", "")
	t.Setenv("FAK_NATIVE_KV_MAX_BLOCKS", "0")
	e := modelengine.New()
	if err := configureServeHostSwapBudget(sf, e); err == nil {
		t.Fatal("positive budget silently armed or ignored an absent block policy")
	}
	e.SetKVPreemptionPolicy(modelengine.NativePreemptionPolicy{Mode: modelengine.NativePreemptSwap, MaxBlocks: 2, BlockTokens: 16})
	if err := configureServeHostSwapBudget(sf, e); err != nil {
		t.Fatal(err)
	}
	if st := e.KVPreemptionStats(); st.HostSwapLimitBytes != 1048576 || st.MaxBlocks != 2 {
		t.Fatalf("flag did not reach the selected engine: %+v", st)
	}
	*sf.nativeHostSwapBytes = -1
	if err := configureServeHostSwapBudget(sf, e); err == nil || e.KVPreemptionStats().HostSwapLimitBytes != 1048576 {
		t.Fatal("negative flag changed the installed bound")
	}
	*sf.nativeHostSwapBytes = 0
	if err := configureServeHostSwapBudget(sf, nil); err != nil {
		t.Fatal("default zero changed the non-native route")
	}
	if err := validateServeHostSwapByteLimit(-1); err == nil {
		t.Fatal("negative startup validation passed")
	}
}
