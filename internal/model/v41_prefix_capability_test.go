package model

import (
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/compute"
)

// TestV41PrefixCapabilityMatrix is the fak#13338 acceptance witness. It proves
// the THREE prefix-capability surfaces are distinct and fail closed, so a
// consumer must ask the right question rather than collapse them:
//
//  1. bare-KV support    -> Config.KVPrefixReuseSupported: is a *KVCache alone a
//     COMPLETE prefix? FALSE for V4.1 (bare cache omits continuation state) and
//     FALSE for the gemma4 recompute bridge.
//  2. host-complete-snapshot support -> Config.HostCompletePrefixSnapshotSupported:
//     does a device-less prefix snapshot carry every continuation byte? TRUE for
//     V4.1 (fak#13342's v41ForwardSnapshot) and the legacy cached architectures;
//     FALSE for gemma4.
//  3. backend-snapshot support -> Config.InKernelBackendPrefixReuseSupportedFor:
//     does a DEVICE session's snapshot own every byte on this exact backend?
//     TRUE for V4.1 only on the qualified backend name; every other backend
//     refused; nil refused.
//
// It also exercises the surfaces end to end: a real host V4.1 snapshot restores
// and continues identically, while the bare-KV constructors still refuse V4.1.
func TestV41PrefixCapabilityMatrix(t *testing.T) {
	t.Run("capability matrix distinguishes the three surfaces", func(t *testing.T) {
		v41 := Config{ModelType: "deepseek_v41"}
		cpuRef := compute.Pick("cpu-ref")

		// 1. bare-KV stays refused for V4.1; legacy cached archs keep it.
		if v41.KVPrefixReuseSupported() {
			t.Fatal("bare-KV KVPrefixReuseSupported now admits V4.1; the incomplete cache contract regressed")
		}
		// 2. host complete snapshot qualifies for V4.1.
		if !v41.HostCompletePrefixSnapshotSupported() {
			t.Fatal("host V4.1 complete snapshot did not qualify after the fak#13342 parity contract")
		}
		// 3. backend snapshot qualifies ONLY on the named qualified backend.
		if v41.InKernelBackendPrefixReuseSupportedFor(nil) {
			t.Fatal("backend-aware capability admitted a nil backend")
		}
		if cpuRef == nil {
			t.Skip("cpu-ref backend is not registered in this build")
		}
		if !v41.InKernelBackendPrefixReuseSupportedFor(cpuRef) {
			t.Fatal("backend-aware capability refused the qualified cpu-ref backend")
		}
		// An unqualified (non-V4.1) architecture is admitted through the legacy
		// branch, proving the V4.1 backend gate is the narrow one; the exhaustive
		// unqualified-IDENTITY negatives (wrong name/layout/epoch) are owned by the
		// fak_hitl build-tagged fak#13334 witness (v41_backend_prefix_admission_test.go).
		if (Config{ModelType: "llama"}).InKernelBackendPrefixReuseSupportedFor(cpuRef) {
			t.Fatal("a plain dense architecture was admitted to the V4.1 backend contract")
		}

		// Legacy surfaces are untouched: a cached architecture keeps every answer.
		for _, mt := range []string{"llama", "deepseek_v4"} {
			cfg := Config{ModelType: mt}
			if !cfg.KVPrefixReuseSupported() {
				t.Errorf("legacy cached architecture %q lost bare-KV support", mt)
			}
			if !cfg.HostCompletePrefixSnapshotSupported() {
				t.Errorf("legacy cached architecture %q lost host complete-snapshot support", mt)
			}
		}
	})

	t.Run("gemma4 recompute bridge stays ineligible on the host surface", func(t *testing.T) {
		gemma := Config{ModelType: "gemma4"}
		if gemma.KVPrefixReuseSupported() {
			t.Fatal("gemma4 bare-KV support regressed (recompute bridge state is token history)")
		}
		if gemma.HostCompletePrefixSnapshotSupported() {
			t.Fatal("gemma4 recompute bridge was admitted to the host complete-snapshot surface; no snapshot carries gemma4Hist")
		}
	})

	t.Run("host V4.1 complete snapshot restores and continues identically", func(t *testing.T) {
		s := v41PrefixSnapshotSession(t, 2, []int{1, 2}, []int{1, 3, 5, 7})
		if !s.M.Cfg.HostCompletePrefixSnapshotSupported() {
			t.Fatal("session config does not advertise the host complete-snapshot capability it demonstrably satisfies")
		}
		snap, err := s.PrefixSnapshot()
		if err != nil {
			t.Fatalf("PrefixSnapshot: %v", err)
		}
		if snap.v41 == nil {
			t.Fatal("host V4.1 snapshot carries no continuation state; capability overclaims")
		}
		branch := &Session{M: s.M, Cache: s.Cache.Clone()}
		if err := snap.Restore(branch); err != nil {
			t.Fatalf("Restore: %v", err)
		}
		want := s.Step(6)
		got := branch.Step(6)
		assertV41LogitsClose(t, got, want, "host-snapshot restored-vs-original")
	})

	t.Run("bare-KV constructors still refuse V4.1", func(t *testing.T) {
		m := &Model{Cfg: Config{ModelType: "deepseek_v41"}}
		assertV41PrefixPanic(t, "deepseek_v41", func() {
			m.SessionFromPrefix(NewKVCache(m.Cfg))
		})
	})
}

// DeepSeek V4.1 requires shared-KV, compressed-attention, and Engram state that
// KVCache and PrefixSnapshot do not yet represent. Until they carry that state,
// cloning KVCache alone is not a complete or safe prefix.
func TestV41KVPrefixReuseFailsClosed(t *testing.T) {
	tests := []struct {
		name string
		cfg  Config
	}{
		{"wrapper identity", Config{ModelType: "deepseek_v41"}},
		{"text identity", Config{ModelType: "deepseek_v41_text"}},
		{"retained metadata", Config{DeepSeekV41: &DeepSeekV41Config{
			WrapperModelType: "deepseek_v41",
			TextModelType:    "deepseek_v41_text",
		}}},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if tt.cfg.KVPrefixReuseSupported() {
				t.Fatal("V4.1 KV prefix reuse advertised before snapshots retain V4.1 session state")
			}
		})
	}
}

func TestV41PrefixCloneConstructorsRefuseByArchitecture(t *testing.T) {
	for _, modelType := range []string{"deepseek_v41", "deepseek_v41_text"} {
		t.Run(modelType+"/session", func(t *testing.T) {
			m := &Model{Cfg: Config{ModelType: modelType}}
			assertV41PrefixPanic(t, modelType, func() {
				m.SessionFromPrefix(NewKVCache(m.Cfg))
			})
		})
		t.Run(modelType+"/batch", func(t *testing.T) {
			m := &Model{Cfg: Config{ModelType: modelType}}
			assertV41PrefixPanic(t, modelType, func() {
				m.NewBatchFromPrefixReserve(NewKVCache(m.Cfg), 2, 4)
			})
		})
	}
}

func TestV41PrefixCapabilityGuardIsNarrow(t *testing.T) {
	for _, modelType := range []string{"llama", "deepseek_v4"} {
		cfg := Config{ModelType: modelType}
		if !cfg.KVPrefixReuseSupported() {
			t.Errorf("legacy cached architecture %q lost KV prefix reuse", modelType)
		}
	}
}

func assertV41PrefixPanic(t *testing.T, modelType string, call func()) {
	t.Helper()
	defer func() {
		r := recover()
		if r == nil {
			t.Fatal("prefix clone constructor accepted incomplete V4.1 session state")
		}
		msg, ok := r.(string)
		if !ok {
			t.Fatalf("prefix clone constructor panicked with %T, want named string refusal", r)
		}
		want := strings.NewReplacer("_", "", "-", "").Replace(modelType)
		if !strings.Contains(msg, want) {
			t.Fatalf("refusal %q does not name architecture %q", msg, want)
		}
	}()
	call()
}
