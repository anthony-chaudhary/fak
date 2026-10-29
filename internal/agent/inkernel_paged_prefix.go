package agent

import (
	"fmt"
	"os"
	"strings"
	"sync"

	"github.com/anthony-chaudhary/fak/internal/compute"
	"github.com/anthony-chaudhary/fak/internal/ctxmmu"
	"github.com/anthony-chaudhary/fak/internal/model"
	"github.com/anthony-chaudhary/fak/internal/radixkv"
)

// inKernelPagedPrefixPageTokens is the page granularity of the device planner's paged
// prefix ledger: the PagedAttention default block size the paged components already use.
const inKernelPagedPrefixPageTokens = 16

// inKernelPagedPrefixDefaultPages bounds the ledger when the radix tree is unbounded.
const inKernelPagedPrefixDefaultPages = 4096

// inKernelPagedPrefixLedger is the device-planner consumer of the paged prefix pool
// (radixkv.PagedRadixKVPool over ctxmmu.KVPool). Each device request takes a page
// lease over its prompt: full pages of a cached prefix are bound shared (refcount++),
// only the divergent tail gets fresh pages, and the prompt's page chain is published
// so the next request with the same prefix binds the same pages. Release drops the
// stream's references; the tree keeps the cached prefix pages until it evicts them.
//
// The ctxmmu pool is a metadata-only ledger (a 1-layer/1-head/1-dim FP8 geometry, so
// a page carries 2*page-tokens host bytes): the device KV bytes stay in the backend's
// own store. Page geometry for the real model is reported as KV elements per page.
type inKernelPagedPrefixLedger struct {
	mu         sync.Mutex
	pool       *radixkv.PagedRadixKVPool
	tpp        int
	pageElems  int64
	nextStream uint64

	leases         int64
	active         int64
	sharedPages    int64
	allocatedPages int64
	failOpen       int64
}

// InKernelPagedPrefixTelemetry is the paged prefix ledger's cumulative accounting.
type InKernelPagedPrefixTelemetry struct {
	Leases         int64 `json:"leases"`
	ActiveLeases   int64 `json:"active_leases"`
	SharedPages    int64 `json:"shared_pages"`
	AllocatedPages int64 `json:"allocated_pages"`
	FailOpen       int64 `json:"fail_open"`
	LivePages      int   `json:"live_pages"`
	TokensPerPage  int   `json:"tokens_per_page"`
	PageKVElems    int64 `json:"page_kv_elems"`
	SharedKVElems  int64 `json:"shared_kv_elems"`
}

func newInKernelPagedPrefixLedger(cfg model.Config, totalPages, tokensPerPage int) (*inKernelPagedPrefixLedger, error) {
	if tokensPerPage <= 0 {
		tokensPerPage = inKernelPagedPrefixPageTokens
	}
	if totalPages <= 0 {
		totalPages = inKernelPagedPrefixDefaultPages
	}
	kvPool, err := ctxmmu.NewKVPool(ctxmmu.KVPoolConfig{
		TotalPages:    totalPages,
		TokensPerPage: tokensPerPage,
		NumLayers:     1,
		NumKVHeads:    1,
		HeadDim:       1,
		DType:         ctxmmu.KVDTypeFP8,
		Topology:      ctxmmu.KVTopologyGQA,
	})
	if err != nil {
		return nil, err
	}
	return &inKernelPagedPrefixLedger{
		pool:      radixkv.NewPagedRadixKVPoolWithTree(radixkv.New(totalPages*tokensPerPage), kvPool),
		tpp:       tokensPerPage,
		pageElems: int64(cfg.KVCacheShape().KVElemsPerToken()) * int64(tokensPerPage),
	}, nil
}

// inKernelPagedPrefixLedgerFor wires the ledger for a device-backed planner whose radix
// prefix reuse is on. FAK_INKERNEL_PAGED_PREFIX=off disables it; a CPU planner never
// gets one.
func inKernelPagedPrefixLedgerFor(m *model.Model, backend compute.Backend, tree *radixkv.Tree) *inKernelPagedPrefixLedger {
	if m == nil || backend == nil || tree == nil {
		return nil
	}
	switch strings.ToLower(strings.TrimSpace(os.Getenv("FAK_INKERNEL_PAGED_PREFIX"))) {
	case "off", "0", "false", "no":
		return nil
	}
	pages := inKernelPagedPrefixDefaultPages
	if budget := tree.Stats().MaxTokens; budget > 0 {
		pages = (budget + inKernelPagedPrefixPageTokens - 1) / inKernelPagedPrefixPageTokens
	}
	l, err := newInKernelPagedPrefixLedger(m.Cfg, pages, inKernelPagedPrefixPageTokens)
	if err != nil {
		return nil
	}
	return l
}

// inKernelPagedPrefixLease is one request's hold on its prompt's page chain.
type inKernelPagedPrefixLease struct {
	ledger      *inKernelPagedPrefixLedger
	node        *radixkv.Node
	table       *radixkv.PagedBlockTable
	sharedPages int
	freshPages  int
	released    bool
}

// acquire leases the page chain for ids under namespace ns. Full pages of the longest
// cached prefix are bound shared; the divergent tail (including a partially matched
// page, which is copied rather than shared) gets fresh pages, and the whole chain is
// published on the request's leaf. It fails open (nil lease) on pool exhaustion or a
// refused admission: the ledger never fails a request.
func (l *inKernelPagedPrefixLedger) acquire(ns string, ids []int) *inKernelPagedPrefixLease {
	if l == nil || len(ids) == 0 {
		return nil
	}
	l.mu.Lock()
	defer l.mu.Unlock()

	boundary, matched := l.pool.LookupNS(ns, ids)
	node := boundary
	var fresh []int
	if matched < len(ids) {
		chain := l.pool.ResolveBlockIDs(boundary)
		shared := matched / l.tpp
		if shared > len(chain) {
			shared = len(chain)
		}
		need := (len(ids) - shared*l.tpp + l.tpp - 1) / l.tpp
		var ok bool
		if fresh, ok = l.allocateLocked(need); !ok {
			l.pool.Done(boundary)
			l.failOpen++
			return nil
		}
		alloc := l.pool.Pool().Allocator()
		leaf := l.pool.Insert(boundary, ids[matched:], nil)
		full := append(append([]int(nil), chain[:shared]...), fresh...)
		if leaf == nil || leaf == boundary || l.pool.SetNodeBlocks(leaf, full) != nil {
			for _, id := range fresh {
				alloc.Free(id)
			}
			fresh = nil
			if leaf != nil {
				node = leaf
			}
		} else {
			// The tree now owns the fresh pages; drop the allocation's initial reference.
			for _, id := range fresh {
				if blk, err := alloc.GetBlock(id); err == nil {
					blk.Release()
				}
			}
			node = leaf
		}
		l.pool.ReconcileEvictions()
	}
	l.nextStream++
	table, err := l.pool.BindNode(node, fmt.Sprintf("inkernel-paged-%d", l.nextStream))
	if err != nil {
		l.pool.Done(node)
		if len(fresh) == 0 {
			l.failOpen++
		}
		return nil
	}
	sharedPages := table.BlockCount() - len(fresh)
	l.leases++
	l.active++
	l.sharedPages += int64(sharedPages)
	l.allocatedPages += int64(len(fresh))
	return &inKernelPagedPrefixLease{ledger: l, node: node, table: table, sharedPages: sharedPages, freshPages: len(fresh)}
}

// release drops the request's stream references and its tree lease. Idempotent.
func (lease *inKernelPagedPrefixLease) release() {
	if lease == nil {
		return
	}
	l := lease.ledger
	l.mu.Lock()
	defer l.mu.Unlock()
	if lease.released {
		return
	}
	lease.released = true
	_ = l.pool.ReleaseBlockTable(lease.table)
	l.pool.Done(lease.node)
	l.active--
}

// telemetry returns the ledger's cumulative accounting.
func (l *inKernelPagedPrefixLedger) telemetry() InKernelPagedPrefixTelemetry {
	if l == nil {
		return InKernelPagedPrefixTelemetry{}
	}
	l.mu.Lock()
	defer l.mu.Unlock()
	return InKernelPagedPrefixTelemetry{
		Leases:         l.leases,
		ActiveLeases:   l.active,
		SharedPages:    l.sharedPages,
		AllocatedPages: l.allocatedPages,
		FailOpen:       l.failOpen,
		LivePages:      l.pool.Pool().PhysicalBlocksAllocated(),
		TokensPerPage:  l.tpp,
		PageKVElems:    l.pageElems,
		SharedKVElems:  l.sharedPages * l.pageElems,
	}
}

// PagedPrefixTelemetry reports the device planner's paged prefix ledger, and whether
// one is wired (a device-backed planner with radix reuse on).
func (p *InKernelPlanner) PagedPrefixTelemetry() (InKernelPagedPrefixTelemetry, bool) {
	if p == nil || p.pagedPrefix == nil {
		return InKernelPagedPrefixTelemetry{}, false
	}
	return p.pagedPrefix.telemetry(), true
}

// allocateLocked claims n fresh pages, reconciling tree evictions once on exhaustion.
func (l *inKernelPagedPrefixLedger) allocateLocked(n int) ([]int, bool) {
	alloc := l.pool.Pool().Allocator()
	out := make([]int, 0, n)
	reconciled := false
	for len(out) < n {
		blk, err := alloc.Allocate()
		if err != nil {
			if !reconciled {
				reconciled = true
				l.pool.ReconcileEvictions()
				continue
			}
			for _, id := range out {
				alloc.Free(id)
			}
			return nil, false
		}
		out = append(out, blk.ID)
	}
	return out, true
}
