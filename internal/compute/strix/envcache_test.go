package strix

import "testing"

// fak-test:runtime fast est=5ms lane=default
// The gfx1151 decode GEMV admission flags are process constants: after the first
// read the cached value is returned without touching the environment again, so a
// later mutation is not observed until the cache is explicitly reset.
func TestStrixDecodeGEMVEnvCachedOnce(t *testing.T) {
	t.Setenv(EnvStrixWave32GEMVDecode, "1")
	ResetStrixEnvCache()
	if !StrixDecodeGEMVDeviceRequested() {
		t.Fatal("explicit opt-in must be recognized")
	}
	t.Setenv(EnvStrixWave32GEMVDecode, "")
	if !StrixDecodeGEMVDeviceRequested() {
		t.Fatal("cached flag was re-read from the environment")
	}
	ResetStrixEnvCache()
	if StrixDecodeGEMVDeviceRequested() {
		t.Fatal("reset did not observe the current environment")
	}
}
