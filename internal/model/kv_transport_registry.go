package model

import (
	"errors"
	"fmt"
	"net"
	"sort"
	"sync"

	"github.com/anthony-chaudhary/fak/internal/cachemeta"
)

// KVTransferBackend names the pluggable native KV transfer backend selected at
// the existing KVTransport seam. The shipped default remains same-host shm
// (implemented by the serializer round-trip); TCP is the real wire fallback; the
// remaining rows are declared fail-closed hooks for hardware/storage bindings.
type KVTransferBackend string

const (
	KVTransferBackendSHM     KVTransferBackend = "shm"
	KVTransferBackendTCP     KVTransferBackend = "tcp"
	KVTransferBackendUCXRDMA KVTransferBackend = "ucx-rdma"
	KVTransferBackendNVMeOF  KVTransferBackend = "nvme-of"
	KVTransferBackendObject  KVTransferBackend = "object"
)

// ErrKVTransferBackendUnavailable is returned when a matrix row is known but has
// no registered factory in this build.
var ErrKVTransferBackendUnavailable = errors.New("model: KV transfer backend unavailable")

// KVTransferBackendUnavailableError names the backend that failed closed.
type KVTransferBackendUnavailableError struct {
	Backend KVTransferBackend
	Reason  string
}

func (e KVTransferBackendUnavailableError) Error() string {
	if e.Reason == "" {
		return fmt.Sprintf("%s: %s", ErrKVTransferBackendUnavailable, e.Backend)
	}
	return fmt.Sprintf("%s: %s (%s)", ErrKVTransferBackendUnavailable, e.Backend, e.Reason)
}

func (e KVTransferBackendUnavailableError) Is(target error) bool {
	return target == ErrKVTransferBackendUnavailable
}

// KVTransferBackendSpec is one row in the NIXL-parity transfer matrix. Available
// means this build can open the backend; unavailable rows are still explicit so a
// selector never silently falls back from UCX/NVMe/object to a fake transfer.
type KVTransferBackendSpec struct {
	Backend   KVTransferBackend
	Label     string
	FromTier  cachemeta.ResidencyTier
	ToTier    cachemeta.ResidencyTier
	Share     cachemeta.ShareKind
	Async     bool
	Witness   string
	Available bool
}

// KVTransportOpenRequest carries the per-receiver resources needed to build a
// transport instance. TCP may use Conn; when Conn is nil the shipped factory uses
// a net.Pipe loopback peer as its no-network witness.
type KVTransportOpenRequest struct {
	Backend KVTransferBackend
	Pool    *PagedKVPool
	Conn    net.Conn
}

// KVTransportHandle is an opened backend plus any transport-owned cleanup.
type KVTransportHandle struct {
	Transport KVTransport
	Close     func() error
}

// KVTransportFactory opens one backend instance for a transfer receiver.
type KVTransportFactory func(KVTransportOpenRequest) (KVTransportHandle, error)

type kvTransportRegistration struct {
	spec    KVTransferBackendSpec
	factory KVTransportFactory
}

var (
	kvTransportMu    sync.RWMutex
	kvTransportOrder []KVTransferBackend
	kvTransports     = map[KVTransferBackend]kvTransportRegistration{}
)

func init() {
	registerKVTransportBackend(KVTransferBackendSpec{
		Backend:  KVTransferBackendSHM,
		Label:    "same-host shared-memory/in-process serializer",
		FromTier: cachemeta.TierDRAM,
		ToTier:   cachemeta.TierDRAM,
		Share:    cachemeta.ShareMmap,
		Witness:  "MarshalPagedKVTransfer -> UnmarshalPagedKVTransfer receipt",
	}, openSHMKVTransport)
	registerKVTransportBackend(KVTransferBackendSpec{
		Backend:  KVTransferBackendTCP,
		Label:    "TCP framed paged-KV transfer",
		FromTier: cachemeta.TierDRAM,
		ToTier:   cachemeta.TierRemote,
		Share:    cachemeta.ShareCopy,
		Witness:  "TCPKVTransport loopback byte-identical to shm receipt",
	}, openTCPKVTransport)
	registerKVTransportBackend(KVTransferBackendSpec{
		Backend:  KVTransferBackendUCXRDMA,
		Label:    "UCX/RDMA remote-memory transfer",
		FromTier: cachemeta.TierHBM,
		ToTier:   cachemeta.TierRemote,
		Share:    cachemeta.ShareRDMA,
		Async:    true,
		Witness:  "registered backend must emit the same checksum/lease receipt",
	}, nil)
	registerKVTransportBackend(KVTransferBackendSpec{
		Backend:  KVTransferBackendNVMeOF,
		Label:    "NVMe-oF/storage-tier transfer",
		FromTier: cachemeta.TierDRAM,
		ToTier:   cachemeta.TierDisk,
		Share:    cachemeta.ShareCopy,
		Async:    true,
		Witness:  "registered backend must emit the same checksum/lease receipt",
	}, nil)
	registerKVTransportBackend(KVTransferBackendSpec{
		Backend:  KVTransferBackendObject,
		Label:    "object-store cold-tier transfer",
		FromTier: cachemeta.TierDisk,
		ToTier:   cachemeta.TierRemote,
		Share:    cachemeta.ShareCopy,
		Async:    true,
		Witness:  "registered backend must emit the same checksum/lease receipt",
	}, nil)
}

// RegisterKVTransferBackend registers or replaces one matrix backend. It is the
// extension point for build-tagged UCX/RDMA, storedrv/NVMe-oF, and object-store
// implementations; callers that omit a factory create an explicit fail-closed row.
func RegisterKVTransferBackend(spec KVTransferBackendSpec, factory KVTransportFactory) {
	kvTransportMu.Lock()
	defer kvTransportMu.Unlock()
	registerKVTransportBackendLocked(spec, factory)
}

func registerKVTransportBackend(spec KVTransferBackendSpec, factory KVTransportFactory) {
	kvTransportMu.Lock()
	defer kvTransportMu.Unlock()
	registerKVTransportBackendLocked(spec, factory)
}

func registerKVTransportBackendLocked(spec KVTransferBackendSpec, factory KVTransportFactory) {
	if spec.Backend == "" {
		return
	}
	if _, exists := kvTransports[spec.Backend]; !exists {
		kvTransportOrder = append(kvTransportOrder, spec.Backend)
	}
	spec.Available = factory != nil
	kvTransports[spec.Backend] = kvTransportRegistration{spec: spec, factory: factory}
}

// KVTransferBackendMatrix returns a stable copy of the known backend matrix.
func KVTransferBackendMatrix() []KVTransferBackendSpec {
	kvTransportMu.RLock()
	defer kvTransportMu.RUnlock()
	out := make([]KVTransferBackendSpec, 0, len(kvTransports))
	seen := map[KVTransferBackend]bool{}
	for _, backend := range kvTransportOrder {
		reg, ok := kvTransports[backend]
		if !ok {
			continue
		}
		out = append(out, reg.spec)
		seen[backend] = true
	}
	if len(seen) != len(kvTransports) {
		var extra []string
		for backend := range kvTransports {
			if !seen[backend] {
				extra = append(extra, string(backend))
			}
		}
		sort.Strings(extra)
		for _, backend := range extra {
			out = append(out, kvTransports[KVTransferBackend(backend)].spec)
		}
	}
	return out
}

// OpenKVTransferBackend opens the named backend and wraps it so every receipt's
// cachemeta.KVTransfer carries the backend label. Unknown or unimplemented rows
// fail closed.
func OpenKVTransferBackend(req KVTransportOpenRequest) (KVTransportHandle, error) {
	if req.Backend == "" {
		req.Backend = KVTransferBackendSHM
	}
	kvTransportMu.RLock()
	reg, ok := kvTransports[req.Backend]
	kvTransportMu.RUnlock()
	if !ok {
		return KVTransportHandle{}, KVTransferBackendUnavailableError{Backend: req.Backend, Reason: "not registered"}
	}
	if reg.factory == nil {
		return KVTransportHandle{}, KVTransferBackendUnavailableError{Backend: req.Backend, Reason: "no factory in this build"}
	}
	h, err := reg.factory(req)
	if err != nil {
		return KVTransportHandle{}, err
	}
	if h.Transport == nil {
		return KVTransportHandle{}, KVTransferBackendUnavailableError{Backend: req.Backend, Reason: "factory returned nil transport"}
	}
	h.Transport = backendStampedKVTransport{backend: req.Backend, inner: h.Transport}
	return h, nil
}

// KVTransferPath is the pure path classifier for auto-picking a backend. It
// mirrors the NIXL/Dynamo shape: same host -> shm; cross-node -> UCX/RDMA unless
// the caller explicitly asks for TCP fallback; cold storage -> NVMe-oF/object.
type KVTransferPath struct {
	SameHost    bool
	CrossNode   bool
	ColdTier    bool
	FromTier    cachemeta.ResidencyTier
	ToTier      cachemeta.ResidencyTier
	TCPFallback bool
}

// SelectKVTransferBackend maps a transfer path to the intended matrix row. It
// does not check availability; OpenKVTransferBackend is the fail-closed gate.
func SelectKVTransferBackend(path KVTransferPath) KVTransferBackend {
	if path.ColdTier || path.ToTier == cachemeta.TierDisk {
		return KVTransferBackendNVMeOF
	}
	if path.ToTier == cachemeta.TierProvider {
		return KVTransferBackendObject
	}
	if path.CrossNode || path.ToTier == cachemeta.TierRemote {
		if path.TCPFallback {
			return KVTransferBackendTCP
		}
		return KVTransferBackendUCXRDMA
	}
	if path.SameHost {
		return KVTransferBackendSHM
	}
	return KVTransferBackendSHM
}

type backendStampedKVTransport struct {
	backend KVTransferBackend
	inner   KVTransport
}

func (t backendStampedKVTransport) Send(seq *PagedKV, transfer cachemeta.KVTransfer, from, n int) (PagedKVTransferReceipt, error) {
	if transfer.Backend == "" {
		transfer.Backend = string(t.backend)
	}
	return t.inner.Send(seq, transfer, from, n)
}

func openSHMKVTransport(req KVTransportOpenRequest) (KVTransportHandle, error) {
	if req.Pool == nil {
		return KVTransportHandle{}, errors.New("model: shm KV transfer needs a destination pool")
	}
	return KVTransportHandle{Transport: LocalKVTransport{Pool: req.Pool}}, nil
}

func openTCPKVTransport(req KVTransportOpenRequest) (KVTransportHandle, error) {
	if req.Pool == nil {
		return KVTransportHandle{}, errors.New("model: TCP KV transfer needs a destination pool")
	}
	if req.Conn != nil {
		return KVTransportHandle{
			Transport: NewTCPKVTransport(req.Conn, req.Pool),
			Close:     req.Conn.Close,
		}, nil
	}
	client, server := net.Pipe()
	done := make(chan error, 1)
	go func() {
		defer server.Close()
		done <- EchoKVTransferFrames(server)
	}()
	return KVTransportHandle{
		Transport: NewTCPKVTransport(client, req.Pool),
		Close: func() error {
			err := errors.Join(client.Close(), server.Close())
			select {
			case <-done:
			default:
			}
			return err
		},
	}, nil
}
