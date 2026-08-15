package compute

import (
	"errors"
	"fmt"
)

// KVHostSnapshot is a complete host-DRAM image of one KVStore. KRaw is retained
// alongside post-RoPE K and V so a restored store preserves the exact eviction /
// re-positioning semantics of the live cache rather than becoming attention-only data.
type KVHostSnapshot struct {
	Config KVConfig
	Pos    []int
	K      [][]float32
	KRaw   [][]float32
	V      [][]float32
}

// KVHostSnapshotter is the optional device-to-host transfer seam for a KVStore.
// Implementations copy every owned payload byte and leave the source store unchanged.
type KVHostSnapshotter interface {
	SnapshotToHost() (KVHostSnapshot, error)
}

// KVHostRestorer is the optional bulk host-to-device inverse. Backends that do not
// implement it use the portable AppendKV fallback in RestoreKVFromHost.
type KVHostRestorer interface {
	RestoreKVFromHost(KVHostSnapshot) (KVStore, error)
}

var ErrKVHostSnapshotUnsupported = errors.New("compute: KV store cannot snapshot complete state to host")

// ResidentBytes reports host payload bytes, excluding slice/header overhead.
func (s KVHostSnapshot) ResidentBytes() int64 {
	var floats int64
	for i := range s.K {
		floats += int64(len(s.K[i]) + len(s.KRaw[i]) + len(s.V[i]))
	}
	return floats*int64(F32.Bytes()) + int64(len(s.Pos))*8
}

// TransferBytes reports the bytes that cross the host/device boundary. Positions
// remain Go-owned metadata in the current KVStore implementations, so they are not
// included in the transfer count.
func (s KVHostSnapshot) TransferBytes() int64 {
	return s.ResidentBytes() - int64(len(s.Pos))*8
}

// Clone returns an independently owned host image.
func (s KVHostSnapshot) Clone() KVHostSnapshot {
	out := KVHostSnapshot{
		Config: cloneKVConfig(s.Config),
		Pos:    append([]int(nil), s.Pos...),
		K:      cloneFloatRows(s.K),
		KRaw:   cloneFloatRows(s.KRaw),
		V:      cloneFloatRows(s.V),
	}
	return out
}

// Validate checks the complete K/KRaw/V geometry before any restore allocates
// resident memory.
func (s KVHostSnapshot) Validate() error {
	if s.Config.NumLayers < 0 || s.Config.NumKVHeads < 0 || s.Config.HeadDim < 0 {
		return errors.New("compute: invalid negative KV host snapshot geometry")
	}
	if len(s.K) != s.Config.NumLayers || len(s.KRaw) != s.Config.NumLayers || len(s.V) != s.Config.NumLayers {
		return fmt.Errorf("compute: KV host snapshot layers K/KRaw/V=%d/%d/%d, want %d",
			len(s.K), len(s.KRaw), len(s.V), s.Config.NumLayers)
	}
	stride := s.Config.NumKVHeads * s.Config.HeadDim
	want := len(s.Pos) * stride
	for layer := 0; layer < s.Config.NumLayers; layer++ {
		k, raw, v := len(s.K[layer]), len(s.KRaw[layer]), len(s.V[layer])
		// Hybrid recurrent layers own no attention KV; all three empty rows are complete.
		if k == 0 && raw == 0 && v == 0 {
			continue
		}
		if k != want || raw != want || v != want {
			return fmt.Errorf("compute: KV host snapshot layer %d K/KRaw/V=%d/%d/%d, want %d or all empty",
				layer, k, raw, v, want)
		}
	}
	return nil
}

// SnapshotKVToHost copies the complete store into ordinary Go-owned host DRAM.
func SnapshotKVToHost(kv KVStore) (KVHostSnapshot, error) {
	if kv == nil {
		return KVHostSnapshot{}, errors.New("compute: cannot snapshot nil KV store")
	}
	snapshotter, ok := kv.(KVHostSnapshotter)
	if !ok {
		return KVHostSnapshot{}, ErrKVHostSnapshotUnsupported
	}
	state, err := snapshotter.SnapshotToHost()
	if err != nil {
		return KVHostSnapshot{}, err
	}
	if err := state.Validate(); err != nil {
		return KVHostSnapshot{}, err
	}
	return state, nil
}

// RestoreKVFromHost materializes an independently owned KVStore from a complete
// host image. A backend bulk restorer wins; the fallback uses the public AppendKV
// contract and therefore stays correct for any backend that can accept F32 uploads.
func RestoreKVFromHost(be Backend, state KVHostSnapshot) (KVStore, error) {
	if be == nil {
		return nil, errors.New("compute: cannot restore KV host snapshot without a backend")
	}
	if err := state.Validate(); err != nil {
		return nil, err
	}
	if restorer, ok := be.(KVHostRestorer); ok {
		return restorer.RestoreKVFromHost(state.Clone())
	}
	if len(state.Pos) > 0 && len(state.K) > 0 && len(state.K[0]) == 0 {
		return nil, fmt.Errorf(
			"compute: backend %q needs KVHostRestorer for a hybrid snapshot whose layer 0 has no attention KV",
			be.Name(),
		)
	}
	kv := be.NewKV(cloneKVConfig(state.Config))
	if kv == nil {
		return nil, fmt.Errorf("compute: backend %q returned nil KV store during host restore", be.Name())
	}
	fail := func(err error) (KVStore, error) {
		kv.Free()
		return nil, err
	}
	stride := state.Config.NumKVHeads * state.Config.HeadDim
	for posIndex, pos := range state.Pos {
		lo, hi := posIndex*stride, (posIndex+1)*stride
		for layer := 0; layer < state.Config.NumLayers; layer++ {
			if len(state.K[layer]) == 0 {
				continue
			}
			raw := be.Upload(NewF32(be, []int{stride}, state.KRaw[layer][lo:hi]), F32)
			rope := be.Upload(NewF32(be, []int{stride}, state.K[layer][lo:hi]), F32)
			value := be.Upload(NewF32(be, []int{stride}, state.V[layer][lo:hi]), F32)
			func() {
				defer be.Free(raw)
				defer be.Free(rope)
				defer be.Free(value)
				kv.AppendKV(layer, raw, rope, value, pos)
			}()
		}
	}
	if kv.Len() != len(state.Pos) {
		return fail(fmt.Errorf("compute: backend %q restored %d KV positions, want %d", be.Name(), kv.Len(), len(state.Pos)))
	}
	return kv, nil
}

func (k *cpuKV) SnapshotToHost() (KVHostSnapshot, error) {
	if k == nil {
		return KVHostSnapshot{}, errors.New("compute: cannot snapshot nil cpu KV")
	}
	return KVHostSnapshot{
		Config: cloneKVConfig(k.cfg),
		Pos:    append([]int(nil), k.pos...),
		K:      cloneFloatRows(k.K),
		KRaw:   cloneFloatRows(k.Kraw),
		V:      cloneFloatRows(k.V),
	}, nil
}

func (c *cpuBackend) RestoreKVFromHost(state KVHostSnapshot) (KVStore, error) {
	if err := state.Validate(); err != nil {
		return nil, err
	}
	return &cpuKV{
		be:   c,
		cfg:  cloneKVConfig(state.Config),
		K:    cloneFloatRows(state.K),
		Kraw: cloneFloatRows(state.KRaw),
		V:    cloneFloatRows(state.V),
		pos:  append([]int(nil), state.Pos...),
	}, nil
}

func cloneFloatRows(in [][]float32) [][]float32 {
	out := make([][]float32, len(in))
	for i := range in {
		out[i] = append([]float32(nil), in[i]...)
	}
	return out
}

func cloneKVConfig(in KVConfig) KVConfig {
	out := in
	out.WindowPerLayer = append([]int(nil), in.WindowPerLayer...)
	return out
}
