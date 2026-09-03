package agentopt

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"errors"
	"fmt"
	"sort"
	"strings"
	"sync"
	"time"
)

// WorkerRequest describes an agent worker inference request containing
// prompts, repository context, or pre-tokenized prefix blocks.
type WorkerRequest struct {
	ID             string            `json:"id"`
	TaskKey        string            `json:"task_key,omitempty"`
	WorkerID       string            `json:"worker_id,omitempty"`
	SystemPrompt   string            `json:"system_prompt,omitempty"`
	WorkspaceScope string            `json:"workspace_scope,omitempty"`
	Prompt         string            `json:"prompt,omitempty"`
	Tokens         []string          `json:"tokens,omitempty"`
	Metadata       map[string]string `json:"metadata,omitempty"`
}

// AffinityRouteResult captures the chosen serving instance, prefix match metrics,
// and affinity score for a routed request.
type AffinityRouteResult struct {
	InstanceID    string    `json:"instance_id"`
	MatchedBlocks int       `json:"matched_blocks"`
	TotalBlocks   int       `json:"total_blocks"`
	MatchRatio    float64   `json:"match_ratio"`
	AffinityScore float64   `json:"affinity_score"`
	ActiveLoad    int       `json:"active_load"`
	Capacity      int       `json:"capacity"`
	WarmHit       bool      `json:"warm_hit"`
	RoutedAt      time.Time `json:"routed_at"`
	Reason        string    `json:"reason"`
}

// RouterConfig configures weights and thresholds for prefix-affinity routing.
type RouterConfig struct {
	PrefixWeight      float64 `json:"prefix_weight"`
	LoadWeight        float64 `json:"load_weight"`
	SaturationPenalty float64 `json:"saturation_penalty"`
	DefaultCapacity   int     `json:"default_capacity"`
	BlockSize         int     `json:"block_size"`
	MinMatchBlocks    int     `json:"min_match_blocks"`
}

// RouterStats records cumulative routing metrics.
type RouterStats struct {
	TotalRouted     int64   `json:"total_routed"`
	WarmHits        int64   `json:"warm_hits"`
	WarmHitRatio    float64 `json:"warm_hit_ratio"`
	ActiveInstances int     `json:"active_instances"`
}

// PrefixNode represents a node in the instance's prefix radix trie.
type PrefixNode struct {
	Hash     string
	Children map[string]*PrefixNode
	HitCount int64
	LastSeen time.Time
}

// PrefixTrie tracks resident KV prefix blocks in a serving engine instance.
type PrefixTrie struct {
	mu        sync.RWMutex
	root      *PrefixNode
	nodeCount int
}

// NewPrefixTrie initializes an empty prefix trie.
func NewPrefixTrie() *PrefixTrie {
	return &PrefixTrie{
		root: &PrefixNode{
			Children: make(map[string]*PrefixNode),
			LastSeen: time.Now(),
		},
	}
}

// Insert adds a sequence of prefix blocks into the trie.
func (t *PrefixTrie) Insert(blocks []string) int {
	if len(blocks) == 0 {
		return 0
	}
	t.mu.Lock()
	defer t.mu.Unlock()

	now := time.Now()
	curr := t.root
	curr.LastSeen = now
	inserted := 0

	for _, block := range blocks {
		child, exists := curr.Children[block]
		if !exists {
			child = &PrefixNode{
				Hash:     block,
				Children: make(map[string]*PrefixNode),
				LastSeen: now,
			}
			curr.Children[block] = child
			t.nodeCount++
			inserted++
		} else {
			child.LastSeen = now
			child.HitCount++
		}
		curr = child
	}
	return inserted
}

// MatchLength returns the longest common prefix match length against the trie.
func (t *PrefixTrie) MatchLength(blocks []string) int {
	if len(blocks) == 0 {
		return 0
	}
	t.mu.RLock()
	defer t.mu.RUnlock()

	curr := t.root
	matched := 0
	for _, block := range blocks {
		child, exists := curr.Children[block]
		if !exists {
			break
		}
		matched++
		curr = child
	}
	return matched
}

// NodeCount returns the total number of cached block nodes.
func (t *PrefixTrie) NodeCount() int {
	t.mu.RLock()
	defer t.mu.RUnlock()
	return t.nodeCount
}

// Clear flushes all cached prefix nodes.
func (t *PrefixTrie) Clear() {
	t.mu.Lock()
	defer t.mu.Unlock()
	t.root = &PrefixNode{
		Children: make(map[string]*PrefixNode),
		LastSeen: time.Now(),
	}
	t.nodeCount = 0
}

// ServingInstance represents a registered model serving engine.
type ServingInstance struct {
	mu             sync.RWMutex
	ID             string            `json:"id"`
	Address        string            `json:"address,omitempty"`
	Capacity       int               `json:"capacity"`
	ActiveRequests int               `json:"active_requests"`
	PrefixTree     *PrefixTrie       `json:"-"`
	TotalServed    int64             `json:"total_served"`
	WarmHits       int64             `json:"warm_hits"`
	Metadata       map[string]string `json:"metadata,omitempty"`
}

// NewServingInstance creates an engine instance with specified capacity.
func NewServingInstance(id string, capacity int) *ServingInstance {
	if capacity <= 0 {
		capacity = 8
	}
	return &ServingInstance{
		ID:         id,
		Capacity:   capacity,
		PrefixTree: NewPrefixTrie(),
		Metadata:   make(map[string]string),
	}
}

// GetActiveRequests returns the current in-flight request count.
func (e *ServingInstance) GetActiveRequests() int {
	e.mu.RLock()
	defer e.mu.RUnlock()
	return e.ActiveRequests
}

// SetActiveRequests explicitly updates the active request count.
func (e *ServingInstance) SetActiveRequests(n int) {
	e.mu.Lock()
	defer e.mu.Unlock()
	e.ActiveRequests = n
}

// IncrementActive increments active in-flight count.
func (e *ServingInstance) IncrementActive() int {
	e.mu.Lock()
	defer e.mu.Unlock()
	e.ActiveRequests++
	return e.ActiveRequests
}

// DecrementActive decrements active in-flight count safely.
func (e *ServingInstance) DecrementActive() int {
	e.mu.Lock()
	defer e.mu.Unlock()
	if e.ActiveRequests > 0 {
		e.ActiveRequests--
	}
	return e.ActiveRequests
}

// RecordServed updates instance serving and cache-hit counters.
func (e *ServingInstance) RecordServed(hit bool) {
	e.mu.Lock()
	defer e.mu.Unlock()
	e.TotalServed++
	if hit {
		e.WarmHits++
	}
}

// MatchPrefix computes the prefix match length against this instance's prefix trie.
func (e *ServingInstance) MatchPrefix(blocks []string) int {
	if e.PrefixTree == nil {
		return 0
	}
	return e.PrefixTree.MatchLength(blocks)
}

// InsertPrefix inserts prefix blocks into this instance's prefix trie.
func (e *ServingInstance) InsertPrefix(blocks []string) int {
	if e.PrefixTree == nil {
		return 0
	}
	return e.PrefixTree.Insert(blocks)
}

// PrefixAffinityRouter coordinates request placement across engine instances
// to maximize KV cache reuse while preventing instance saturation.
type PrefixAffinityRouter struct {
	mu        sync.RWMutex
	config    RouterConfig
	instances map[string]*ServingInstance
	stats     RouterStats
}

// DefaultRouterConfig returns standard tuning parameters.
func DefaultRouterConfig() RouterConfig {
	return RouterConfig{
		PrefixWeight:      1.0,
		LoadWeight:        0.5,
		SaturationPenalty: 2.0,
		DefaultCapacity:   8,
		BlockSize:         16,
		MinMatchBlocks:    1,
	}
}

// NewPrefixAffinityRouter constructs a router with optional configuration.
func NewPrefixAffinityRouter(cfg ...RouterConfig) *PrefixAffinityRouter {
	c := DefaultRouterConfig()
	if len(cfg) > 0 {
		user := cfg[0]
		if user.PrefixWeight > 0 {
			c.PrefixWeight = user.PrefixWeight
		}
		if user.LoadWeight >= 0 {
			c.LoadWeight = user.LoadWeight
		}
		if user.SaturationPenalty > 0 {
			c.SaturationPenalty = user.SaturationPenalty
		}
		if user.DefaultCapacity > 0 {
			c.DefaultCapacity = user.DefaultCapacity
		}
		if user.BlockSize > 0 {
			c.BlockSize = user.BlockSize
		}
		if user.MinMatchBlocks > 0 {
			c.MinMatchBlocks = user.MinMatchBlocks
		}
	}
	return &PrefixAffinityRouter{
		config:    c,
		instances: make(map[string]*ServingInstance),
	}
}

// RegisterInstance registers an engine instance with the router.
func (r *PrefixAffinityRouter) RegisterInstance(inst *ServingInstance) error {
	if inst == nil || strings.TrimSpace(inst.ID) == "" {
		return errors.New("cannot register nil or empty instance")
	}
	r.mu.Lock()
	defer r.mu.Unlock()
	r.instances[inst.ID] = inst
	r.stats.ActiveInstances = len(r.instances)
	return nil
}

// UnregisterInstance removes an engine instance.
func (r *PrefixAffinityRouter) UnregisterInstance(id string) error {
	r.mu.Lock()
	defer r.mu.Unlock()
	if _, exists := r.instances[id]; !exists {
		return fmt.Errorf("instance %q not found", id)
	}
	delete(r.instances, id)
	r.stats.ActiveInstances = len(r.instances)
	return nil
}

// GetInstance retrieves a registered instance by ID.
func (r *PrefixAffinityRouter) GetInstance(id string) (*ServingInstance, bool) {
	r.mu.RLock()
	defer r.mu.RUnlock()
	inst, exists := r.instances[id]
	return inst, exists
}

// ListInstances returns all registered instances in ID order.
func (r *PrefixAffinityRouter) ListInstances() []*ServingInstance {
	r.mu.RLock()
	defer r.mu.RUnlock()
	out := make([]*ServingInstance, 0, len(r.instances))
	for _, inst := range r.instances {
		out = append(out, inst)
	}
	sort.Slice(out, func(i, j int) bool {
		return out[i].ID < out[j].ID
	})
	return out
}

// WarmInstance populates an instance's prefix cache with the request's prefix.
func (r *PrefixAffinityRouter) WarmInstance(instanceID string, req WorkerRequest) error {
	blocks := r.ExtractBlocks(req)
	return r.WarmInstanceBlocks(instanceID, blocks)
}

// WarmInstanceBlocks inserts raw prefix blocks into an instance's cache.
func (r *PrefixAffinityRouter) WarmInstanceBlocks(instanceID string, blocks []string) error {
	r.mu.RLock()
	inst, exists := r.instances[instanceID]
	r.mu.RUnlock()
	if !exists {
		return fmt.Errorf("instance %q not found", instanceID)
	}
	inst.InsertPrefix(blocks)
	return nil
}

// ResetInstancePrefixes flushes the prefix cache of an instance.
func (r *PrefixAffinityRouter) ResetInstancePrefixes(instanceID string) error {
	r.mu.RLock()
	inst, exists := r.instances[instanceID]
	r.mu.RUnlock()
	if !exists {
		return fmt.Errorf("instance %q not found", instanceID)
	}
	if inst.PrefixTree != nil {
		inst.PrefixTree.Clear()
	}
	return nil
}

// SetInstanceLoad updates the active request count of an instance.
func (r *PrefixAffinityRouter) SetInstanceLoad(instanceID string, load int) error {
	r.mu.RLock()
	inst, exists := r.instances[instanceID]
	r.mu.RUnlock()
	if !exists {
		return fmt.Errorf("instance %q not found", instanceID)
	}
	inst.SetActiveRequests(load)
	return nil
}

// ExtractBlocks extracts ordered prefix block tokens from a request.
func (r *PrefixAffinityRouter) ExtractBlocks(req WorkerRequest) []string {
	if len(req.Tokens) > 0 {
		out := make([]string, len(req.Tokens))
		copy(out, req.Tokens)
		return out
	}

	blockSize := r.config.BlockSize
	if blockSize <= 0 {
		blockSize = 16
	}

	var blocks []string
	if req.SystemPrompt != "" {
		blocks = append(blocks, chunkAndHash("sys", req.SystemPrompt, blockSize)...)
	}
	if req.WorkspaceScope != "" {
		blocks = append(blocks, chunkAndHash("repo", req.WorkspaceScope, blockSize)...)
	}
	if req.Prompt != "" {
		blocks = append(blocks, chunkAndHash("prompt", req.Prompt, blockSize)...)
	}
	return blocks
}

func chunkAndHash(domain, text string, blockSize int) []string {
	words := strings.Fields(text)
	if len(words) == 0 {
		trimmed := strings.TrimSpace(text)
		if trimmed == "" {
			return nil
		}
		words = []string{trimmed}
	}

	var hashes []string
	for i := 0; i < len(words); i += blockSize {
		end := i + blockSize
		if end > len(words) {
			end = len(words)
		}
		chunk := strings.Join(words[i:end], " ")
		h := sha256.New()
		h.Write([]byte(domain))
		h.Write([]byte{0})
		h.Write([]byte(chunk))
		digest := hex.EncodeToString(h.Sum(nil))[:16]
		hashes = append(hashes, domain+":"+digest)
	}
	return hashes
}

// ComputeScore calculates the affinity score balancing prefix match against load.
func (r *PrefixAffinityRouter) ComputeScore(matchedBlocks, totalBlocks, activeRequests, capacity int) (float64, float64) {
	if totalBlocks <= 0 {
		totalBlocks = 1
	}
	if capacity <= 0 {
		capacity = r.config.DefaultCapacity
	}
	matchRatio := float64(matchedBlocks) / float64(totalBlocks)
	loadRatio := float64(activeRequests) / float64(capacity)

	score := (r.config.PrefixWeight * matchRatio) - (r.config.LoadWeight * loadRatio)
	if activeRequests >= capacity {
		score -= r.config.SaturationPenalty
	}
	return score, matchRatio
}

// Route evaluates target instances and selects the optimal serving replica.
func (r *PrefixAffinityRouter) Route(ctx context.Context, req WorkerRequest) (*AffinityRouteResult, error) {
	select {
	case <-ctx.Done():
		return nil, ctx.Err()
	default:
	}

	r.mu.Lock()
	defer r.mu.Unlock()

	if len(r.instances) == 0 {
		return nil, errors.New("no serving engine instances available")
	}

	blocks := r.ExtractBlocks(req)
	totalBlocks := len(blocks)
	if totalBlocks == 0 {
		totalBlocks = 1
	}

	type target struct {
		inst          *ServingInstance
		matchedBlocks int
		matchRatio    float64
		affinityScore float64
		activeLoad    int
		capacity      int
	}

	targets := make([]target, 0, len(r.instances))
	for _, inst := range r.instances {
		matched := inst.MatchPrefix(blocks)
		active := inst.GetActiveRequests()
		cap := inst.Capacity
		if cap <= 0 {
			cap = r.config.DefaultCapacity
		}
		score, ratio := r.ComputeScore(matched, totalBlocks, active, cap)
		targets = append(targets, target{
			inst:          inst,
			matchedBlocks: matched,
			matchRatio:    ratio,
			affinityScore: score,
			activeLoad:    active,
			capacity:      cap,
		})
	}

	sort.Slice(targets, func(i, j int) bool {
		if targets[i].affinityScore != targets[j].affinityScore {
			return targets[i].affinityScore > targets[j].affinityScore
		}
		if targets[i].matchedBlocks != targets[j].matchedBlocks {
			return targets[i].matchedBlocks > targets[j].matchedBlocks
		}
		if targets[i].activeLoad != targets[j].activeLoad {
			return targets[i].activeLoad < targets[j].activeLoad
		}
		return targets[i].inst.ID < targets[j].inst.ID
	})

	best := targets[0]
	isHit := best.matchedBlocks >= r.config.MinMatchBlocks

	r.stats.TotalRouted++
	if isHit {
		r.stats.WarmHits++
	}
	if r.stats.TotalRouted > 0 {
		r.stats.WarmHitRatio = float64(r.stats.WarmHits) / float64(r.stats.TotalRouted)
	}

	var reason string
	if isHit {
		reason = fmt.Sprintf("warm cache affinity match: %d/%d blocks (score: %.3f, load: %d/%d)",
			best.matchedBlocks, totalBlocks, best.affinityScore, best.activeLoad, best.capacity)
	} else {
		reason = fmt.Sprintf("cold/least-loaded routing (score: %.3f, load: %d/%d)",
			best.affinityScore, best.activeLoad, best.capacity)
	}

	return &AffinityRouteResult{
		InstanceID:    best.inst.ID,
		MatchedBlocks: best.matchedBlocks,
		TotalBlocks:   totalBlocks,
		MatchRatio:    best.matchRatio,
		AffinityScore: best.affinityScore,
		ActiveLoad:    best.activeLoad,
		Capacity:      best.capacity,
		WarmHit:       isHit,
		RoutedAt:      time.Now(),
		Reason:        reason,
	}, nil
}

// RouteRequest is a convenience method for routing without an explicit context.
func (r *PrefixAffinityRouter) RouteRequest(req WorkerRequest) (*AffinityRouteResult, error) {
	return r.Route(context.Background(), req)
}

// RouteAndAcquire routes the request, increments active load on the selected instance,
// warms the instance with the request prefix, and returns a release function.
func (r *PrefixAffinityRouter) RouteAndAcquire(ctx context.Context, req WorkerRequest) (*AffinityRouteResult, func(), error) {
	routeRes, err := r.Route(ctx, req)
	if err != nil {
		return nil, nil, err
	}

	r.mu.RLock()
	inst := r.instances[routeRes.InstanceID]
	r.mu.RUnlock()

	if inst != nil {
		inst.IncrementActive()
		blocks := r.ExtractBlocks(req)
		inst.InsertPrefix(blocks)
	}

	var once sync.Once
	release := func() {
		once.Do(func() {
			if inst != nil {
				inst.DecrementActive()
				inst.RecordServed(routeRes.WarmHit)
			}
		})
	}

	return routeRes, release, nil
}

// Stats returns a snapshot of router metrics.
func (r *PrefixAffinityRouter) Stats() RouterStats {
	r.mu.RLock()
	defer r.mu.RUnlock()
	return r.stats
}
