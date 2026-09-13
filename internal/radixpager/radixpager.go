package radixpager

import (
	"sync"
	"unsafe"

	"github.com/anthony-chaudhary/fak/internal/ctxmmu"
	"github.com/anthony-chaudhary/fak/internal/model"
	"github.com/anthony-chaudhary/fak/internal/radixkv"
)

const (
	DefaultRadixPagerAlign  = 4096
	DefaultRadixPagerBlocks = 1024
)

// RadixKVPager pages pre-tokenized observation blocks directly into the
// kernel-owned RadixAttention prefix tree. It is the binding half of the
// context-MMU's zero-copy admission seam: an aligned observation is folded to
// one token per page, bound into the tree by Bind, and the resulting node handle
// is returned as an address-stable uintptr the caller can carry in a receipt.
//
// The pager lives in its own package because internal/radixkv already imports
// internal/ctxmmu (paged_binding.go), so internal/ctxmmu cannot import
// internal/radixkv without an import cycle. The pager imports both and satisfies
// ctxmmu.ObservationPager, which is the interface the admission gate holds.
type RadixKVPager struct {
	mu    sync.Mutex
	align int
	alloc *ctxmmu.PhysicalBlockAllocator
	tree  *radixkv.Tree
}

func NewRadixKVPager(existing *radixkv.Tree, pageAlign int, totalBlocks int) *RadixKVPager {
	if pageAlign <= 0 {
		pageAlign = DefaultRadixPagerAlign
	}
	if totalBlocks <= 0 {
		totalBlocks = DefaultRadixPagerBlocks
	}
	if existing == nil {
		existing = radixkv.New(totalBlocks)
	}
	return &RadixKVPager{
		align: pageAlign,
		alloc: ctxmmu.NewPhysicalBlockAllocator(totalBlocks, pageAlign),
		tree:  existing,
	}
}

// pageToken folds one page's bytes into a stable token id via 64-bit FNV-1a,
// reduced into the token-id space. Same bytes -> same token, address-independent.
func pageToken(page []byte) int {
	const (
		offset64 = uint64(14695981039346656037)
		prime64  = uint64(1099511628211)
	)
	h := offset64
	for _, b := range page {
		h ^= uint64(b)
		h *= prime64
	}
	return int(h & 0x7fffffff)
}

func (p *RadixKVPager) PageBytes(raw []byte) (tokens []int, pages int, nodeAligned bool, err error) {
	if p == nil {
		return nil, 0, false, nil
	}
	if len(raw) == 0 {
		return nil, 0, false, nil
	}
	align := p.align
	if align <= 0 {
		align = DefaultRadixPagerAlign
	}
	pages = (len(raw) + align - 1) / align
	tokens = make([]int, 0, pages)
	for off := 0; off < len(raw); off += align {
		end := off + align
		if end > len(raw) {
			end = len(raw)
		}
		tokens = append(tokens, pageToken(raw[off:end]))
	}
	nodeAligned = (uintptr(unsafe.Pointer(&raw[0])) % uintptr(align)) == 0
	return tokens, pages, nodeAligned, nil
}

// Bind attaches the token path to the Radix tree and returns a non-zero,
// address-stable uintptr handle for the bound node. Lookup leases the matched
// boundary, Insert hangs the suffix, Done releases the lease so the bound path
// stays LRU-evictable rather than pinned forever.
func (p *RadixKVPager) Bind(tokens []int) (nodeRef uintptr, err error) {
	if p == nil || len(tokens) == 0 {
		return 0, nil
	}
	p.mu.Lock()
	defer p.mu.Unlock()

	boundary, matched := p.tree.Lookup(tokens)
	leaf := p.tree.Insert(boundary, tokens[matched:], model.NewKVCache(model.Config{}))
	if leaf != nil {
		p.tree.Done(leaf)
		nodeRef = uintptr(unsafe.Pointer(leaf))
	} else if boundary != nil {
		p.tree.Done(boundary)
		nodeRef = uintptr(unsafe.Pointer(boundary))
	}
	return nodeRef, nil
}

func (p *RadixKVPager) AdmitRaw(raw []byte) (nodeRef uintptr, tokens []int, pages int, zeroCopy bool, err error) {
	if p == nil {
		return 0, nil, 0, false, nil
	}
	tokens, pages, aligned, perr := p.PageBytes(raw)
	if perr != nil {
		return 0, tokens, pages, false, perr
	}
	if !aligned || len(tokens) == 0 {
		return 0, tokens, pages, false, nil
	}
	nodeRef, err = p.Bind(tokens)
	if err != nil {
		return 0, tokens, pages, false, err
	}
	return nodeRef, tokens, pages, nodeRef != 0, nil
}

func (p *RadixKVPager) AllocatedBlocks() int {
	if p == nil {
		return 0
	}
	return p.alloc.AllocatedCount()
}

func (p *RadixKVPager) Alignment() int {
	if p == nil {
		return 0
	}
	return p.align
}

func (p *RadixKVPager) Tree() *radixkv.Tree {
	if p == nil {
		return nil
	}
	return p.tree
}
