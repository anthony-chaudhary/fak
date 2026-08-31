package contextq

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"sort"
	"strconv"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/abi"
)

// TreeOperation is one bounded structural observation over immutable JSON.
type TreeOperation string

const (
	TreeKeys     TreeOperation = "keys"
	TreeChildren TreeOperation = "children"
	TreeGet      TreeOperation = "get"
)

type TreePlan struct {
	Operation TreeOperation `json:"operation"`
	Path      string        `json:"path,omitempty"`
	Paths     []string      `json:"paths,omitempty"`
	Offset    int           `json:"offset,omitempty"`
	Limit     int           `json:"limit,omitempty"`
}

type TreeLimits struct {
	MaxSourceBytes int `json:"max_source_bytes"`
	MaxOutputBytes int `json:"max_output_bytes"`
	MaxDepth       int `json:"max_depth"`
	MaxWidth       int `json:"max_width"`
	MaxPaths       int `json:"max_paths"`
	MaxNodes       int `json:"max_nodes"`
	MaxWorkUnits   int `json:"max_work_units"`
	MaxLeafBytes   int `json:"max_leaf_bytes"`
}

type TreeChild struct {
	Path       string `json:"path"`
	Type       string `json:"type"`
	ChildCount int    `json:"child_count"`
	ByteLength int    `json:"byte_length"`
	Digest     string `json:"digest"`
}

type TreeValue struct {
	Path  string          `json:"path"`
	Type  string          `json:"type"`
	Value json.RawMessage `json:"value"`
}

type TreeAccounting struct {
	SourceBytes  int `json:"source_bytes"`
	OutputBytes  int `json:"output_bytes"`
	NodesVisited int `json:"nodes_visited"`
	WorkDebit    int `json:"work_debit"`
}

type DerivedTreeView struct {
	Source       abi.Ref        `json:"source"`
	Plan         TreePlan       `json:"plan"`
	PlanDigest   string         `json:"plan_digest"`
	OutputDigest string         `json:"output_digest"`
	Keys         []string       `json:"keys,omitempty"`
	Children     []TreeChild    `json:"children,omitempty"`
	Values       []TreeValue    `json:"values,omitempty"`
	Total        int            `json:"total"`
	NextOffset   int            `json:"next_offset,omitempty"`
	Truncated    bool           `json:"truncated"`
	Accounting   TreeAccounting `json:"accounting"`
}

type TreeReason string

const (
	TreeReasonPlanInvalid   TreeReason = "plan_invalid"
	TreeReasonPathInvalid   TreeReason = "path_invalid"
	TreeReasonPathMissing   TreeReason = "path_missing"
	TreeReasonTypeMismatch  TreeReason = "type_mismatch"
	TreeReasonNonLeaf       TreeReason = "non_leaf"
	TreeReasonLeafLimit     TreeReason = "leaf_limit"
	TreeReasonDepthLimit    TreeReason = "depth_limit"
	TreeReasonWidthLimit    TreeReason = "width_limit"
	TreeReasonPathLimit     TreeReason = "path_limit"
	TreeReasonNodeLimit     TreeReason = "node_limit"
	TreeReasonWorkLimit     TreeReason = "work_limit"
	TreeReasonOutputLimit   TreeReason = "output_limit"
	TreeReasonSourceInvalid TreeReason = "source_invalid"
)

type TreeRefusal struct {
	Reason TreeReason
	Detail string
}

func (e *TreeRefusal) Error() string {
	return fmt.Sprintf("contextq tree refused: %s: %s", e.Reason, e.Detail)
}
func treeRefuse(r TreeReason, d string) error { return &TreeRefusal{Reason: r, Detail: d} }

// ExploreTree resolves source through the same ABI seam as record views and returns one bounded observation.
func ExploreTree(ctx context.Context, resolver abi.Resolver, source abi.Ref, plan TreePlan, limits TreeLimits) (DerivedTreeView, error) {
	if err := contextRefusal(ctx); err != nil {
		return DerivedTreeView{}, err
	}
	if limits.MaxSourceBytes <= 0 || limits.MaxOutputBytes <= 0 || limits.MaxDepth <= 0 || limits.MaxWidth <= 0 || limits.MaxPaths <= 0 || limits.MaxNodes <= 0 || limits.MaxWorkUnits <= 0 || limits.MaxLeafBytes <= 0 {
		return DerivedTreeView{}, treeRefuse(TreeReasonPlanInvalid, "all limits must be positive")
	}
	// Reuse #6518 validation/resolution so lineage, taint, scope, digest, and source limits stay identical.
	raw, err := resolveTreeSource(ctx, resolver, source, limits.MaxSourceBytes)
	if err != nil {
		return DerivedTreeView{}, err
	}
	dec := json.NewDecoder(bytes.NewReader(raw))
	dec.UseNumber()
	var root any
	if err := dec.Decode(&root); err != nil {
		return DerivedTreeView{}, treeRefuse(TreeReasonSourceInvalid, err.Error())
	}
	if dec.More() {
		return DerivedTreeView{}, treeRefuse(TreeReasonSourceInvalid, "multiple JSON values")
	}
	if _, ok := root.(map[string]any); !ok {
		if _, ok := root.([]any); !ok {
			return DerivedTreeView{}, treeRefuse(TreeReasonTypeMismatch, "source root must be object or array")
		}
	}

	plan.Path = canonicalTreePath(plan.Path)
	for i := range plan.Paths {
		plan.Paths[i] = canonicalTreePath(plan.Paths[i])
	}
	canonical, _ := json.Marshal(plan)
	out := DerivedTreeView{Source: source, Plan: plan, PlanDigest: digestBytes(canonical)}
	visited := 0
	observe := func(path string) (any, error) {
		parts, err := parseTreePath(path, limits.MaxDepth)
		if err != nil {
			return nil, err
		}
		v := root
		visited++
		for _, p := range parts {
			if err := contextRefusal(ctx); err != nil {
				return nil, err
			}
			visited++
			if visited > limits.MaxNodes {
				return nil, treeRefuse(TreeReasonNodeLimit, "visited nodes exceed max_nodes")
			}
			switch n := v.(type) {
			case map[string]any:
				x, ok := n[p]
				if !ok {
					return nil, treeRefuse(TreeReasonPathMissing, path)
				}
				v = x
			case []any:
				idx, e := strconv.Atoi(p)
				if e != nil || idx < 0 || idx >= len(n) {
					return nil, treeRefuse(TreeReasonPathMissing, path)
				}
				v = n[idx]
			default:
				return nil, treeRefuse(TreeReasonPathMissing, path)
			}
		}
		return v, nil
	}

	switch plan.Operation {
	case TreeKeys:
		v, err := observe(plan.Path)
		if err != nil {
			return DerivedTreeView{}, err
		}
		obj, ok := v.(map[string]any)
		if !ok {
			return DerivedTreeView{}, treeRefuse(TreeReasonTypeMismatch, "keys requires an object")
		}
		keys := make([]string, 0, len(obj))
		for k := range obj {
			keys = append(keys, k)
		}
		sort.Strings(keys)
		if len(keys) > limits.MaxWidth {
			return DerivedTreeView{}, treeRefuse(TreeReasonWidthLimit, "object width exceeds max_width")
		}
		page, next, err := treePage(keys, plan.Offset, plan.Limit)
		if err != nil {
			return DerivedTreeView{}, err
		}
		out.Keys, out.Total, out.NextOffset, out.Truncated = page, len(keys), next, next < len(keys)
	case TreeChildren:
		v, err := observe(plan.Path)
		if err != nil {
			return DerivedTreeView{}, err
		}
		children := make([]TreeChild, 0)
		switch n := v.(type) {
		case map[string]any:
			keys := make([]string, 0, len(n))
			for k := range n {
				keys = append(keys, k)
			}
			sort.Strings(keys)
			if len(keys) > limits.MaxWidth {
				return DerivedTreeView{}, treeRefuse(TreeReasonWidthLimit, "object width exceeds max_width")
			}
			for _, k := range keys {
				children = append(children, describeTreeChild(joinTreePath(plan.Path, k), n[k]))
			}
		case []any:
			if len(n) > limits.MaxWidth {
				return DerivedTreeView{}, treeRefuse(TreeReasonWidthLimit, "array width exceeds max_width")
			}
			for i, x := range n {
				children = append(children, describeTreeChild(joinTreePath(plan.Path, strconv.Itoa(i)), x))
			}
		default:
			return DerivedTreeView{}, treeRefuse(TreeReasonTypeMismatch, "children requires an object or array")
		}
		page, next, err := treePage(children, plan.Offset, plan.Limit)
		if err != nil {
			return DerivedTreeView{}, err
		}
		out.Children, out.Total, out.NextOffset, out.Truncated = page, len(children), next, next < len(children)
	case TreeGet:
		if len(plan.Paths) == 0 || len(plan.Paths) > limits.MaxPaths {
			return DerivedTreeView{}, treeRefuse(TreeReasonPathLimit, "paths count is outside max_paths")
		}
		seen := map[string]bool{}
		for _, p := range plan.Paths {
			if seen[p] {
				return DerivedTreeView{}, treeRefuse(TreeReasonPlanInvalid, "duplicate path")
			}
			seen[p] = true
			v, err := observe(p)
			if err != nil {
				return DerivedTreeView{}, err
			}
			if childCount(v) > 0 {
				return DerivedTreeView{}, treeRefuse(TreeReasonNonLeaf, p)
			}
			b, _ := json.Marshal(v)
			if len(b) > limits.MaxLeafBytes {
				return DerivedTreeView{}, treeRefuse(TreeReasonLeafLimit, p)
			}
			out.Values = append(out.Values, TreeValue{Path: p, Type: jsonType(v), Value: b})
		}
		out.Total = len(out.Values)
	default:
		return DerivedTreeView{}, treeRefuse(TreeReasonPlanInvalid, "unknown operation")
	}
	out.Accounting = TreeAccounting{SourceBytes: len(raw), NodesVisited: visited, WorkDebit: len(raw) + visited}
	if out.Accounting.WorkDebit > limits.MaxWorkUnits {
		return DerivedTreeView{}, treeRefuse(TreeReasonWorkLimit, "work debit exceeds max_work_units")
	}
	visible, _ := json.Marshal(struct {
		Keys     []string    `json:"keys,omitempty"`
		Children []TreeChild `json:"children,omitempty"`
		Values   []TreeValue `json:"values,omitempty"`
	}{out.Keys, out.Children, out.Values})
	out.Accounting.OutputBytes = len(visible)
	if len(visible) > limits.MaxOutputBytes {
		return DerivedTreeView{}, treeRefuse(TreeReasonOutputLimit, "output exceeds max_output_bytes")
	}
	out.OutputDigest = digestBytes(visible)
	return out, nil
}

func canonicalTreePath(p string) string {
	if p == "" {
		return "/"
	}
	return p
}
func parseTreePath(p string, max int) ([]string, error) {
	if p == "" || p == "/" {
		return nil, nil
	}
	if !strings.HasPrefix(p, "/") {
		return nil, treeRefuse(TreeReasonPathInvalid, "path must start with /")
	}
	raw := strings.Split(p[1:], "/")
	if len(raw) > max {
		return nil, treeRefuse(TreeReasonDepthLimit, "path exceeds max_depth")
	}
	out := make([]string, len(raw))
	for i, s := range raw {
		if s == "" {
			return nil, treeRefuse(TreeReasonPathInvalid, "empty path segment")
		}
		s = strings.ReplaceAll(strings.ReplaceAll(s, "~1", "/"), "~0", "~")
		if strings.Contains(s, "~") {
			return nil, treeRefuse(TreeReasonPathInvalid, "invalid escape")
		}
		out[i] = s
	}
	return out, nil
}
func joinTreePath(base, seg string) string {
	seg = strings.ReplaceAll(strings.ReplaceAll(seg, "~", "~0"), "/", "~1")
	if base == "" || base == "/" {
		return "/" + seg
	}
	return base + "/" + seg
}
func treePage[T any](all []T, off, lim int) ([]T, int, error) {
	if off < 0 || lim <= 0 || off > len(all) {
		return nil, 0, treeRefuse(TreeReasonPlanInvalid, "invalid offset or limit")
	}
	end := off + lim
	if end > len(all) {
		end = len(all)
	}
	return all[off:end], end, nil
}
func childCount(v any) int {
	switch n := v.(type) {
	case map[string]any:
		return len(n)
	case []any:
		return len(n)
	}
	return 0
}
func jsonType(v any) string {
	switch v.(type) {
	case nil:
		return "null"
	case bool:
		return "boolean"
	case json.Number:
		return "number"
	case string:
		return "string"
	case []any:
		return "array"
	case map[string]any:
		return "object"
	}
	return "unknown"
}
func describeTreeChild(path string, v any) TreeChild {
	b, _ := json.Marshal(v)
	return TreeChild{Path: path, Type: jsonType(v), ChildCount: childCount(v), ByteLength: len(b), Digest: digestBytes(b)}
}
func digestBytes(b []byte) string { sum := sha256.Sum256(b); return hex.EncodeToString(sum[:]) }

func resolveTreeSource(ctx context.Context, resolver abi.Resolver, source abi.Ref, max int) ([]byte, error) {
	if resolver == nil {
		return nil, refuse(DeriveReasonResolverMissing, "an abi.Resolver is required")
	}
	if !validTaintLabel(source.Taint) {
		return nil, refuse(DeriveReasonTaintInvalid, "source taint is outside the closed ABI vocabulary")
	}
	if !validShareScope(source.Scope) {
		return nil, refuse(DeriveReasonScopeInvalid, "source scope is outside the closed ABI vocabulary")
	}
	if source.Taint == abi.TaintQuarantined {
		return nil, refuse(DeriveReasonSourceQuarantined, "quarantined refs cannot be queried")
	}
	if !validDigest(source.Digest) {
		return nil, refuse(DeriveReasonDigestInvalid, "source digest must be 64 lowercase hexadecimal characters")
	}
	if source.Len < 0 {
		return nil, refuse(DeriveReasonLengthMismatch, "source length is negative")
	}
	if source.Len > int64(max) {
		return nil, refuse(DeriveReasonSourceLimit, "declared source length exceeds max_source_bytes")
	}
	body, err := resolver.Resolve(ctx, source)
	if err != nil {
		if cerr := contextRefusal(ctx); cerr != nil {
			return nil, cerr
		}
		return nil, refuse(DeriveReasonResolveFailed, err.Error())
	}
	if len(body) > max {
		return nil, refuse(DeriveReasonSourceLimit, "resolved source exceeds max_source_bytes")
	}
	if source.Len != int64(len(body)) {
		return nil, refuse(DeriveReasonLengthMismatch, "declared source length does not match resolved bytes")
	}
	if digest(body) != source.Digest {
		return nil, refuse(DeriveReasonDigestMismatch, "source digest does not match resolved bytes")
	}
	return body, nil
}
