package fabricmap

import (
	"context"
	"errors"
	"fmt"
	"sort"
	"strings"
	"sync"
)

var ErrAdapterMissing = errors.New("fabric transport unsupported")

// Transfer names the byte-range effect requested from a transport adapter.
// Source and destination are explicit and are checked against the directed link.
type Transfer struct {
	Source      string    `json:"source"`
	Destination string    `json:"destination"`
	Offset      uint64    `json:"offset,omitempty"`
	Bytes       uint64    `json:"bytes"`
	Operation   Operation `json:"operation"`
}

// Adapter performs one technology-specific directed transfer. GPUDirect
// Storage, RDMA, NIXL, CXL, staged CPU copy, or future transports implement the
// same seam without becoming planner taxonomy.
type Adapter interface {
	Transfer(context.Context, Link, Access, Transfer) (HopReceipt, error)
}
type AdapterFunc func(context.Context, Link, Access, Transfer) (HopReceipt, error)

func (f AdapterFunc) Transfer(ctx context.Context, l Link, a Access, t Transfer) (HopReceipt, error) {
	return f(ctx, l, a, t)
}

// TransportRegistry dispatches exact transport names to adapters. Registration
// is additive and concurrency-safe; names are opaque to the generic planner.
type TransportRegistry struct {
	mu       sync.RWMutex
	adapters map[string]Adapter
}

func NewTransportRegistry() *TransportRegistry {
	return &TransportRegistry{adapters: make(map[string]Adapter)}
}
func (r *TransportRegistry) Register(name string, adapter Adapter) error {
	name = strings.TrimSpace(name)
	if name == "" {
		return errors.New("transport name is required")
	}
	if adapter == nil {
		return errors.New("transport adapter is required")
	}
	r.mu.Lock()
	defer r.mu.Unlock()
	if r.adapters == nil {
		r.adapters = make(map[string]Adapter)
	}
	if _, exists := r.adapters[name]; exists {
		return fmt.Errorf("transport %q already registered", name)
	}
	r.adapters[name] = adapter
	return nil
}
func (r *TransportRegistry) Transports() []string {
	if r == nil {
		return nil
	}
	r.mu.RLock()
	defer r.mu.RUnlock()
	out := make([]string, 0, len(r.adapters))
	for name := range r.adapters {
		out = append(out, name)
	}
	sort.Strings(out)
	return out
}

// ExecuteHop implements Executor. It refuses unknown transports and validates
// transfer direction before invoking any technology code.
func (r *TransportRegistry) ExecuteHop(ctx context.Context, request AuthorizationRequest, bytes uint64) (HopReceipt, error) {
	if r == nil {
		return HopReceipt{}, ErrAdapterMissing
	}
	link := request.Link
	if link.From == "" || link.To == "" || link.From == link.To {
		return HopReceipt{}, errors.New("directed transport link is invalid")
	}
	r.mu.RLock()
	adapter := r.adapters[link.Transport]
	r.mu.RUnlock()
	if adapter == nil {
		return HopReceipt{}, fmt.Errorf("%w: %q", ErrAdapterMissing, link.Transport)
	}
	transfer := Transfer{Source: link.From, Destination: link.To, Bytes: bytes, Operation: request.Access.Operation}
	receipt, err := adapter.Transfer(ctx, cloneLink(link), cloneAccess(request.Access), transfer)
	if err != nil {
		return HopReceipt{}, err
	}
	receipt.HopIndex = request.HopIndex
	// The route executor fills hop index validation; the registry owns the
	// technology boundary and refuses an adapter that reports another direction.
	if receipt.From != transfer.Source || receipt.To != transfer.Destination || receipt.LinkID != link.ID || receipt.Transport != link.Transport {
		return HopReceipt{}, errors.New("transport receipt does not match directed transfer")
	}
	return receipt, nil
}
