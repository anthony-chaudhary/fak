package compute

import "errors"

// BackendH2DAttribution partitions native-accounted H2D bytes by the upload
// call's MemoryClass, not by tensor role: weights need not be immutable, and
// activation-class staging may contain KV rows. UnattributedBytes covers native
// transfers outside the instrumented upload sites. DirectMappedBytes overlaps
// the partition: coherent UMA writes are already included in H2DBytes, not an
// additional device transfer or a second physical allocation.
//
// Like the existing H2D counter, recorded batch uploads need not have completed
// on the device yet. Completed-run evidence requires a successful enclosing
// execution window. Fixed fields preserve snapshot comparability and bound the
// vocabulary to the existing MemoryClass constants.
type BackendH2DAttribution struct {
	Unknown           uint64 `json:"unknown"`
	Weights           uint64 `json:"weights"`
	KVCache           uint64 `json:"kv_cache"`
	DDRCache          uint64 `json:"ddr_cache"`
	Offload           uint64 `json:"offload"`
	Scratchpad        uint64 `json:"scratchpad"`
	Activation        uint64 `json:"activation"`
	SignalBuffer      uint64 `json:"signal_buffer"`
	UnattributedBytes uint64 `json:"unattributed_bytes"`
	DirectMappedBytes uint64 `json:"direct_mapped_bytes"`
}

// record adds one serialized native counter interval, including any bytes
// moved before a partially completed upload failed. It never estimates bytes
// from tensor geometry and leaves the accumulator unchanged on failure.
// The string boundary uses the serialized MemoryClass labels; the native caller
// converts the actual MemoryClass, and the fixture binds these labels to capacity.go.
func (a *BackendH2DAttribution) record(class string, before, after uint64) error {
	if after < before {
		return errors.New("compute: H2D upload counter reset or wrapped")
	}
	field := &a.Unknown
	switch class {
	case "weights":
		field = &a.Weights
	case "kv_cache":
		field = &a.KVCache
	case "ddr_cache":
		field = &a.DDRCache
	case "offload":
		field = &a.Offload
	case "scratchpad":
		field = &a.Scratchpad
	case "activation":
		field = &a.Activation
	case "signal_buffer":
		field = &a.SignalBuffer
	}
	bytes := after - before
	if bytes > ^uint64(0)-*field {
		return errors.New("compute: H2D memory-class counter overflow")
	}
	*field += bytes
	return nil
}

func (a BackendH2DAttribution) withTotals(total, directMapped uint64) (BackendH2DAttribution, error) {
	var attributed uint64
	for _, bytes := range []uint64{a.Unknown, a.Weights, a.KVCache, a.DDRCache, a.Offload, a.Scratchpad, a.Activation, a.SignalBuffer} {
		if bytes > ^uint64(0)-attributed {
			return BackendH2DAttribution{}, errors.New("compute: H2D memory-class sum overflow")
		}
		attributed += bytes
	}
	if attributed > total || directMapped > total {
		return BackendH2DAttribution{}, errors.New("compute: H2D attribution exceeds the native total")
	}
	a.UnattributedBytes = total - attributed
	a.DirectMappedBytes = directMapped
	return a, nil
}

func (a BackendH2DAttribution) validate(total uint64, observed, transfersObserved bool) error {
	if !observed {
		if a != (BackendH2DAttribution{}) {
			return errors.New("compute: unavailable H2D attribution contains counters")
		}
		return nil
	}
	if !transfersObserved {
		return errors.New("compute: H2D attribution requires observed transfer counters")
	}
	complete, err := a.withTotals(total, a.DirectMappedBytes)
	if err != nil {
		return err
	}
	if complete != a {
		return errors.New("compute: H2D memory classes and unattributed bytes do not sum to the native total")
	}
	return nil
}

func subtractBackendH2DAttribution(before, after BackendH2DAttribution) (BackendH2DAttribution, error) {
	b := []uint64{before.Unknown, before.Weights, before.KVCache, before.DDRCache, before.Offload, before.Scratchpad, before.Activation, before.SignalBuffer, before.UnattributedBytes, before.DirectMappedBytes}
	a := []uint64{after.Unknown, after.Weights, after.KVCache, after.DDRCache, after.Offload, after.Scratchpad, after.Activation, after.SignalBuffer, after.UnattributedBytes, after.DirectMappedBytes}
	for i := range a {
		if a[i] < b[i] {
			return BackendH2DAttribution{}, errors.New("compute: H2D attribution counter reset or wrapped")
		}
		a[i] -= b[i]
	}
	return BackendH2DAttribution{Unknown: a[0], Weights: a[1], KVCache: a[2], DDRCache: a[3], Offload: a[4], Scratchpad: a[5], Activation: a[6], SignalBuffer: a[7], UnattributedBytes: a[8], DirectMappedBytes: a[9]}, nil
}
