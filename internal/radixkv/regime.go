package radixkv

import (
	"crypto/sha256"
	"encoding/hex"
	"errors"
	"fmt"
	"strconv"
	"strings"
	"sync"

	"github.com/anthony-chaudhary/fak/internal/model"
)

var (
	// ErrRegimeIncomplete indicates a decode regime is missing one or more required axes.
	ErrRegimeIncomplete = errors.New("radixkv: decode regime is incomplete")
	// ErrRegimeMismatch indicates a requested decode regime does not match the cached regime.
	ErrRegimeMismatch = errors.New("radixkv: decode regime mismatch")
)

// DType represents the numeric data type of stored KV tensors.
type DType string

const (
	DTypeFP32 DType = "FP32"
	DTypeFP16 DType = "FP16"
	DTypeBF16 DType = "BF16"
	DTypeFP8  DType = "FP8"
	DTypeINT4 DType = "INT4"
	DTypeINT8 DType = "INT8"
)

// QuantPolicy specifies the quantization scheme applied to KV cache blocks.
type QuantPolicy string

const (
	QuantPolicyNone QuantPolicy = "none"
	QuantPolicyFP8  QuantPolicy = "fp8"
	QuantPolicyINT4 QuantPolicy = "int4"
	QuantPolicyINT8 QuantPolicy = "int8"
)

// RoPEConfig captures the rotary position embedding parameters under which KV
// entries were rotated. Differences in base frequency, scaling factor, scaling
// algorithm, or rotary dimension cause immediate attention corruption if reused.
type RoPEConfig struct {
	Base        float64 // base theta frequency (e.g. 10000.0, 500000.0)
	Scale       float64 // frequency scaling factor (e.g. 1.0 for unscaled, 8.0 for extended)
	ScalingType string  // scaling algorithm (e.g. "none", "linear", "yarn", "llama3")
	Dim         int     // rotary head dimension (e.g. 64, 128)
}

// Complete reports whether all load-bearing RoPE fields are populated.
func (c RoPEConfig) Complete() bool {
	return c.Base > 0 && c.Dim > 0 && c.ScalingType != ""
}

// Key formats the RoPE parameters into a deterministic canonical segment.
func (c RoPEConfig) Key() string {
	baseStr := strconv.FormatFloat(c.Base, 'g', -1, 64)
	scaleStr := strconv.FormatFloat(c.Scale, 'g', -1, 64)
	return fmt.Sprintf("base=%s,scale=%s,type=%s,dim=%d", baseStr, scaleStr, c.ScalingType, c.Dim)
}

// Regime represents the composite decode regime descriptor that must match
// identically between cache insertion and lookup to prevent silent attention
// corruption.
type Regime struct {
	// ModelID identifies the model architecture and weight set.
	ModelID string
	// ModelSHA optionally pins the exact checkpoint revision or weights digest.
	ModelSHA string
	// DType is the numerical format of KV entries.
	DType DType
	// QuantPolicy is the quantization policy under which KV blocks were packed.
	QuantPolicy QuantPolicy
	// RoPE defines the rotary position embedding geometry and scaling.
	RoPE RoPEConfig
}

// Complete reports whether every axis required for safe prefix reuse is set.
// A regime missing any axis fails closed.
func (r Regime) Complete() bool {
	hasModel := r.ModelID != "" || r.ModelSHA != ""
	hasDType := r.DType != ""
	hasQuant := r.QuantPolicy != ""
	return hasModel && hasDType && hasQuant && r.RoPE.Complete()
}

// RegimeKey derives a stable, canonical, deterministic string key for the decode regime.
func (r Regime) RegimeKey() string {
	return strings.Join([]string{
		"model=" + r.ModelID,
		"sha=" + r.ModelSHA,
		"dtype=" + string(r.DType),
		"quant=" + string(r.QuantPolicy),
		"rope=" + r.RoPE.Key(),
	}, ";")
}

// Hash returns the 32-byte cryptographic SHA-256 digest of the canonical RegimeKey.
func (r Regime) Hash() [32]byte {
	return sha256.Sum256([]byte(r.RegimeKey()))
}

// HashHex returns the hex-encoded string of Hash().
func (r Regime) HashHex() string {
	h := r.Hash()
	return hex.EncodeToString(h[:])
}

// Match compares r against want, returning true if both are complete and identical,
// or false along with the first divergent axis as a typed reason.
func (r Regime) Match(want Regime) (bool, MismatchAxis) {
	if !r.Complete() || !want.Complete() {
		return false, AxisIncomplete
	}
	if r.ModelID != want.ModelID || r.ModelSHA != want.ModelSHA {
		return false, AxisModel
	}
	if r.DType != want.DType {
		return false, AxisDtype
	}
	if r.QuantPolicy != want.QuantPolicy {
		return false, AxisQuant
	}
	if r.RoPE != want.RoPE {
		return false, AxisRoPE
	}
	return true, AxisNone
}

// Reusable reports whether want is compatible with r on all axes.
func (r Regime) Reusable(want Regime) bool {
	ok, _ := r.Match(want)
	return ok
}

// extractRegimeKey parses the canonical regime key segment from a namespace string.
func extractRegimeKey(ns string) string {
	if strings.HasPrefix(ns, "regime:") {
		return strings.TrimPrefix(ns, "regime:")
	}
	if idx := strings.Index(ns, "@regime:"); idx != -1 {
		return ns[idx+len("@regime:"):]
	}
	return ""
}

// regimeNamespace builds the isolated namespace identifier combining baseNS and regime.
func regimeNamespace(baseNS string, regime Regime) string {
	key := regime.RegimeKey()
	if baseNS == "" {
		return "regime:" + key
	}
	return baseNS + "@regime:" + key
}

// RegimeTree is a Tree partition pinned to a single decode regime.
type RegimeTree struct {
	tree   *Tree
	regime Regime
	lock   sync.Locker
}

// NewRegimeTree builds a regime partition over tree using an internal mutex.
func NewRegimeTree(tree *Tree, regime Regime) *RegimeTree {
	return NewRegimeTreeWithLocker(tree, regime, &sync.Mutex{})
}

// NewRegimeTreeWithLocker builds a regime partition over tree using lock.
func NewRegimeTreeWithLocker(tree *Tree, regime Regime, lock sync.Locker) *RegimeTree {
	if tree == nil {
		tree = New(0)
	}
	if lock == nil {
		lock = &sync.Mutex{}
	}
	return &RegimeTree{
		tree:   tree,
		regime: regime,
		lock:   lock,
	}
}

// Regime returns the pinned decode regime.
func (rt *RegimeTree) Regime() Regime { return rt.regime }

// Tree returns the underlying Tree.
func (rt *RegimeTree) Tree() *Tree { return rt.tree }

// Lookup looks up the longest prefix under the pinned regime.
func (rt *RegimeTree) Lookup(tokens []int) (*node, int) {
	rt.lock.Lock()
	defer rt.lock.Unlock()
	return rt.tree.LookupRegime(rt.regime, tokens)
}

// LookupNS looks up the longest prefix under the pinned regime and namespace.
func (rt *RegimeTree) LookupNS(ns string, tokens []int) (*node, int) {
	rt.lock.Lock()
	defer rt.lock.Unlock()
	return rt.tree.LookupRegimeNS(rt.regime, ns, tokens)
}

// Insert attaches suffix to boundary under the pinned regime.
func (rt *RegimeTree) Insert(boundary *node, suffix []int, kv *model.KVCache) *node {
	rt.lock.Lock()
	defer rt.lock.Unlock()
	return rt.tree.InsertRegime(rt.regime, boundary, suffix, kv)
}

// InsertWithLogits attaches suffix with logits under the pinned regime.
func (rt *RegimeTree) InsertWithLogits(boundary *node, suffix []int, kv *model.KVCache, logits []float32) *node {
	rt.lock.Lock()
	defer rt.lock.Unlock()
	return rt.tree.InsertRegimeWithLogits(rt.regime, boundary, suffix, kv, logits)
}

// Done releases a lease on n.
func (rt *RegimeTree) Done(n *node) {
	rt.lock.Lock()
	defer rt.lock.Unlock()
	rt.tree.Done(n)
}

// MatchLen reports the matched prefix length under the pinned regime.
func (rt *RegimeTree) MatchLen(tokens []int) int {
	rt.lock.Lock()
	defer rt.lock.Unlock()
	return rt.tree.MatchLenRegime(rt.regime, tokens)
}

// EvictPrefix evicts matching prefix under the pinned regime.
func (rt *RegimeTree) EvictPrefix(tokens []int) int {
	rt.lock.Lock()
	defer rt.lock.Unlock()
	return rt.tree.EvictPrefixRegime(rt.regime, tokens)
}
