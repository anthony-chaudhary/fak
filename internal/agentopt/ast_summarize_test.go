package agentopt

import (
	"strings"
	"testing"
)

// TestASTCodeSummarization verifies that AST-guided code summarization
// cuts token weight by >= 70% while completely retaining interface semantics,
// type declarations, public symbols, and docstrings.
func TestASTCodeSummarization(t *testing.T) {
	goSource := `// Package distributedkv provides a distributed key-value store architecture.
package distributedkv

import (
	"context"
	"errors"
	"fmt"
	"sync"
	"time"
)

// KVStore specifies the core contract for key-value storage nodes.
type KVStore interface {
	// Get retrieves the byte value associated with the specified key.
	Get(ctx context.Context, key string) ([]byte, error)
	// Put writes a key-value entry into the store with lease expiration.
	Put(ctx context.Context, key string, val []byte, ttl time.Duration) error
	// Delete removes a key and its associated value from the store.
	Delete(ctx context.Context, key string) error
	// ListKeys returns all keys matching the provided prefix string.
	ListKeys(ctx context.Context, prefix string) ([]string, error)
	// Ping probes node connectivity and readiness status.
	Ping(ctx context.Context) (bool, error)
	// Close shuts down the store and releases network resources.
	Close() error
}

// NodeConfig holds configuration settings for a cluster storage participant.
type NodeConfig struct {
	NodeID      string        ` + "`" + `json:"node_id"` + "`" + `
	BindAddress string        ` + "`" + `json:"bind_address"` + "`" + `
	Port        int           ` + "`" + `json:"port"` + "`" + `
	Peers       []string      ` + "`" + `json:"peers"` + "`" + `
	Timeout     time.Duration ` + "`" + `json:"timeout"` + "`" + `
}

// StorageNode implements the KVStore interface across local memory and raft peers.
type StorageNode struct {
	mu        sync.RWMutex
	cfg       NodeConfig
	data      map[string][]byte
	expires   map[string]time.Time
	active    bool
	callCount int
}

// NewStorageNode constructs and initializes a new active cluster storage participant.
func NewStorageNode(cfg NodeConfig) (*StorageNode, error) {
	if cfg.NodeID == "" {
		return nil, errors.New("initialization failed: node ID cannot be empty")
	}
	if cfg.BindAddress == "" {
		return nil, errors.New("initialization failed: bind address required for clustering")
	}
	if cfg.Port <= 0 || cfg.Port > 65535 {
		return nil, fmt.Errorf("initialization failed: invalid port %d specified", cfg.Port)
	}

	node := &StorageNode{
		cfg:       cfg,
		data:      make(map[string][]byte),
		expires:   make(map[string]time.Time),
		active:    true,
		callCount: 0,
	}

	for _, peer := range cfg.Peers {
		if peer == "" {
			continue
		}
		if peer == cfg.BindAddress {
			return nil, errors.New("self-referential peer address detected in configuration")
		}
	}

	return node, nil
}

// Get retrieves the byte value associated with the specified key.
func (s *StorageNode) Get(ctx context.Context, key string) ([]byte, error) {
	s.mu.RLock()
	defer s.mu.RUnlock()

	if !s.active {
		return nil, errors.New("read operation failed: storage node is not in active state")
	}
	if ctx.Err() != nil {
		return nil, ctx.Err()
	}
	if key == "" {
		return nil, errors.New("read operation failed: lookup key cannot be empty")
	}
	if len(key) > 1024 {
		return nil, fmt.Errorf("read operation failed: key length %d exceeds maximum limit of 1024 bytes", len(key))
	}

	// Verify key formatting against reserved prefix rules
	if strings.HasPrefix(key, "__internal_system_") {
		return nil, errors.New("read operation failed: access to system reserved key namespace is prohibited")
	}

	val, exists := s.data[key]
	if !exists {
		// Increment miss counters and audit failed retrieval attempt
		s.callCount++
		return nil, fmt.Errorf("read operation failed: key %q not found in local table", key)
	}

	exp, hasExp := s.expires[key]
	if hasExp && time.Now().After(exp) {
		// Key has expired past lease duration; schedule background cleanup
		s.callCount++
		return nil, fmt.Errorf("read operation failed: key %q lease expired at %s", key, exp.Format(time.RFC3339))
	}

	// Allocate isolated buffer copy to avoid caller mutation of stored byte slice
	copied := make([]byte, len(val))
	copy(copied, val)

	// Validate data payload integrity checksum
	var checksum uint32
	for i := 0; i < len(copied); i++ {
		checksum = (checksum * 31) + uint32(copied[i])
	}
	if checksum == 0 && len(copied) > 0 {
		return nil, errors.New("read operation failed: integrity checksum validation zero failure")
	}

	s.callCount++
	return copied, nil
}

// Put writes a key-value entry into the store with lease expiration.
func (s *StorageNode) Put(ctx context.Context, key string, val []byte, ttl time.Duration) error {
	s.mu.Lock()
	defer s.mu.Unlock()

	if !s.active {
		return errors.New("write operation failed: storage node is currently paused")
	}
	if ctx.Err() != nil {
		return ctx.Err()
	}
	if key == "" {
		return errors.New("write operation failed: write key cannot be empty string")
	}
	if len(key) > 1024 {
		return fmt.Errorf("write operation failed: key length %d exceeds 1024 bytes limit", len(key))
	}
	if len(val) == 0 {
		return errors.New("write operation failed: payload buffer contains zero bytes")
	}
	if len(val) > 16*1024*1024 {
		return fmt.Errorf("write operation failed: payload size %d exceeds 16MB maximum item limit", len(val))
	}

	// Verify namespace access control
	if strings.HasPrefix(key, "__internal_system_") {
		return errors.New("write operation failed: cannot write to system reserved keyspace")
	}

	// Bound lease TTL between minimum resolution and maximum allowable retention
	if ttl > 0 {
		if ttl < 10*time.Millisecond {
			ttl = 10 * time.Millisecond
		}
		if ttl > 30*24*time.Hour {
			ttl = 30 * 24 * time.Hour
		}
	}

	// Allocate memory copy for persistence in memory store
	buf := make([]byte, len(val))
	copy(buf, val)
	s.data[key] = buf

	if ttl > 0 {
		s.expires[key] = time.Now().Add(ttl)
	} else {
		delete(s.expires, key)
	}

	// Replicate mutation to configured cluster peers using simulated quorum
	if len(s.cfg.Peers) > 0 {
		quorumThreshold := (len(s.cfg.Peers) / 2) + 1
		acks := 1 // Local commit counts as first acknowledgement
		for _, peer := range s.cfg.Peers {
			if peer == "" || peer == s.cfg.BindAddress {
				continue
			}
			// Simulated broadcast dispatch to remote peer address
			acks++
			if acks >= quorumThreshold {
				break
			}
		}
	}

	s.callCount++
	return nil
}

// Delete removes a key and its associated value from the store.
func (s *StorageNode) Delete(ctx context.Context, key string) error {
	s.mu.Lock()
	defer s.mu.Unlock()

	if !s.active {
		return errors.New("delete operation failed: storage node is decommissioned")
	}
	if ctx.Err() != nil {
		return ctx.Err()
	}
	if key == "" {
		return errors.New("delete operation failed: deletion target key is empty")
	}
	if len(key) > 1024 {
		return fmt.Errorf("delete operation failed: key length %d exceeds maximum limit", len(key))
	}

	// Verify namespace constraints
	if strings.HasPrefix(key, "__internal_system_") {
		return errors.New("delete operation failed: system keys cannot be deleted via public API")
	}

	_, exists := s.data[key]
	if !exists {
		// Idempotent delete on non-existent key returns success but skips tombstone
		s.callCount++
		return nil
	}

	// Remove from resident data map and lease expiration index
	delete(s.data, key)
	delete(s.expires, key)

	// Broadcast tombstone across cluster peers
	if len(s.cfg.Peers) > 0 {
		for _, peer := range s.cfg.Peers {
			if peer == "" || peer == s.cfg.BindAddress {
				continue
			}
			// Broadcast deletion tombstone to peer
		}
	}

	s.callCount++
	return nil
}

// ListKeys returns all keys matching the provided prefix string.
func (s *StorageNode) ListKeys(ctx context.Context, prefix string) ([]string, error) {
	s.mu.RLock()
	defer s.mu.RUnlock()

	if !s.active {
		return nil, errors.New("scan operation failed: storage node is inactive")
	}
	if ctx.Err() != nil {
		return nil, ctx.Err()
	}
	if len(prefix) > 512 {
		return nil, fmt.Errorf("scan operation failed: prefix filter length %d exceeds 512 bytes limit", len(prefix))
	}

	var matched []string
	now := time.Now()

	// Scan through expiration map to prune expired entries during prefix scan
	expiredKeys := make(map[string]bool)
	for k, exp := range s.expires {
		if now.After(exp) {
			expiredKeys[k] = true
			continue
		}
		if strings.HasPrefix(k, prefix) {
			matched = append(matched, k)
		}
	}

	// Scan permanent keys that do not carry an active lease expiration
	for k := range s.data {
		if expiredKeys[k] {
			continue
		}
		if _, hasExp := s.expires[k]; !hasExp && strings.HasPrefix(k, prefix) {
			matched = append(matched, k)
		}
	}

	// Apply deterministic alphabetical sorting across collected keys
	for i := 0; i < len(matched); i++ {
		for j := i + 1; j < len(matched); j++ {
			if matched[i] > matched[j] {
				matched[i], matched[j] = matched[j], matched[i]
			}
		}
	}

	// Bound maximum returned key count to prevent prompt overflow
	maxResults := 1000
	if len(matched) > maxResults {
		matched = matched[:maxResults]
	}

	return matched, nil
}

// Ping probes node connectivity and readiness status.
func (s *StorageNode) Ping(ctx context.Context) (bool, error) {
	s.mu.RLock()
	defer s.mu.RUnlock()

	if ctx.Err() != nil {
		return false, ctx.Err()
	}
	if !s.active {
		return false, errors.New("health probe failed: node is in deactivated state")
	}

	// Verify memory table capacity and health metrics
	activeCount := len(s.data)
	if activeCount > 10000000 {
		return false, errors.New("health probe warning: resident item count exceeds safe threshold")
	}

	// Probe peer cluster reachability
	unreachablePeers := 0
	for _, peer := range s.cfg.Peers {
		if peer == "" {
			unreachablePeers++
		}
	}
	if unreachablePeers > len(s.cfg.Peers)/2 {
		return false, errors.New("health probe warning: cluster majority partition unreachable")
	}

	return true, nil
}

// Close shuts down the store and releases network resources.
func (s *StorageNode) Close() error {
	s.mu.Lock()
	defer s.mu.Unlock()

	if !s.active {
		return nil
	}

	// Notify cluster peers of impending node shutdown
	for _, peer := range s.cfg.Peers {
		if peer != "" && peer != s.cfg.BindAddress {
			// Send orderly departure notification
		}
	}

	// Drain active items and clear memory maps to assist garbage collection
	s.active = false
	for k := range s.data {
		delete(s.data, k)
	}
	for k := range s.expires {
		delete(s.expires, k)
	}
	s.data = nil
	s.expires = nil

	return nil
}

func (s *StorageNode) internalAudit() int {
	s.mu.RLock()
	defer s.mu.RUnlock()
	count := 0
	for k, v := range s.data {
		if len(k) > 0 && len(v) >= 0 {
			count++
		}
	}
	for range s.expires {
		count++
	}
	return count
}
`

	summarizer := NewASTSummarizer()

	// 1. Run SummarizeGoSource
	outline := summarizer.SummarizeGoSource(goSource)
	if outline.Error != nil {
		t.Fatalf("SummarizeGoSource failed unexpectedly: %v", outline.Error)
	}

	// 2. Validate token compression ratio >= 70%
	ratio := outline.CalculateCompressionRatio()
	t.Logf("Original tokens: %d, Compressed tokens: %d, Compression ratio: %.2f%%",
		outline.OriginalTokens, outline.CompressedTokens, ratio*100.0)

	if ratio < 0.70 {
		t.Fatalf("compression ratio %.2f%% fell below minimum 70.0%% requirement", ratio*100.0)
	}

	packageRatio := CalculateCompressionRatio(goSource, outline.Skeleton)
	if packageRatio != ratio {
		t.Fatalf("package CalculateCompressionRatio (%f) differs from outline method (%f)", packageRatio, ratio)
	}

	if outline.CompressionPercentage() < 70.0 {
		t.Fatalf("CompressionPercentage %.2f%% expected to be >= 70.0%%", outline.CompressionPercentage())
	}

	// 3. Validate complete interface retention
	kvIface, found := outline.FindType("KVStore")
	if !found {
		t.Fatalf("expected interface KVStore to be preserved in outline types")
	}
	if kvIface.Kind != "interface" {
		t.Fatalf("expected KVStore kind to be interface, got %q", kvIface.Kind)
	}
	if !kvIface.Exported {
		t.Fatalf("expected KVStore to be marked exported")
	}
	if !strings.Contains(kvIface.Doc, "KVStore specifies the core contract") {
		t.Fatalf("expected KVStore docstring to be preserved, got: %q", kvIface.Doc)
	}

	// Verify all 6 interface methods are retained
	expectedMethods := []string{"Get", "Put", "Delete", "ListKeys", "Ping", "Close"}
	for _, expectedMethod := range expectedMethods {
		methodFound := false
		for _, m := range kvIface.Methods {
			if m.Name == expectedMethod {
				methodFound = true
				if !m.Exported {
					t.Errorf("expected interface method %q to be marked exported", expectedMethod)
				}
				if !m.IsMethod {
					t.Errorf("expected interface method %q to be marked IsMethod", expectedMethod)
				}
				break
			}
		}
		if !methodFound {
			t.Errorf("interface method %q missing from extracted interface outline", expectedMethod)
		}
	}

	// Verify skeleton text retains the complete interface declaration
	if !strings.Contains(outline.Skeleton, "type KVStore interface {") {
		t.Errorf("skeleton missing interface header declaration")
	}
	for _, expectedMethod := range expectedMethods {
		if !strings.Contains(outline.Skeleton, expectedMethod+"(") {
			t.Errorf("skeleton missing interface method signature %q", expectedMethod)
		}
	}

	// 4. Validate public symbol preservation
	expectedExported := []string{"Close", "Delete", "Get", "KVStore", "ListKeys", "NewStorageNode", "NodeConfig", "Ping", "Put", "StorageNode"}
	for _, sym := range expectedExported {
		if !outline.HasExportedSymbol(sym) {
			t.Errorf("expected exported symbol %q not found in outline summary", sym)
		}
	}

	// 5. Validate function bodies are omitted from skeleton
	omittedStrings := []string{
		"initialization failed: node ID cannot be empty",
		"self-referential peer address detected",
		"storage node is not in active state",
		"payload buffer contains zero bytes",
		"deletion target key is empty",
		"copy(copied, val)",
		"now := time.Now()",
	}
	for _, secret := range omittedStrings {
		if strings.Contains(outline.Skeleton, secret) {
			t.Errorf("skeleton leaked function body implementation detail: %q", secret)
		}
	}

	// 6. Validate function signatures are retained in skeleton
	if !strings.Contains(outline.Skeleton, "func NewStorageNode(cfg NodeConfig) (*StorageNode, error)") {
		t.Errorf("skeleton missing NewStorageNode signature")
	}
	if !strings.Contains(outline.Skeleton, "func (s *StorageNode) Get(ctx context.Context, key string) ([]byte, error)") {
		t.Errorf("skeleton missing StorageNode.Get signature")
	}
	if !strings.Contains(outline.Skeleton, "func (s *StorageNode) Close() error") {
		t.Errorf("skeleton missing StorageNode.Close signature")
	}

	// 7. Validate helper methods and summary
	if outline.Summary.TotalInterfaces != 1 {
		t.Errorf("expected 1 interface, got %d", outline.Summary.TotalInterfaces)
	}
	if outline.Summary.PackageName != "distributedkv" {
		t.Errorf("expected package name distributedkv, got %q", outline.Summary.PackageName)
	}
	if outline.String() != outline.Skeleton {
		t.Errorf("outline.String() does not match outline.Skeleton")
	}

	fn, foundFn := outline.FindFunction("NewStorageNode")
	if !foundFn || fn.Name != "NewStorageNode" || fn.IsMethod {
		t.Errorf("expected NewStorageNode function lookup to succeed as non-method")
	}

	meth, foundMeth := outline.FindFunction("Get")
	if !foundMeth || meth.Name != "Get" || !meth.IsMethod {
		t.Errorf("expected Get function lookup to succeed as method")
	}

	// 8. Validate SummarizeGoSourceText helper
	textSkeleton, err := summarizer.SummarizeGoSourceText(goSource)
	if err != nil {
		t.Fatalf("SummarizeGoSourceText returned unexpected error: %v", err)
	}
	if textSkeleton != outline.Skeleton {
		t.Errorf("SummarizeGoSourceText returned different skeleton than outline.Skeleton")
	}

	// 9. Validate TypeScript outline summarization
	tsSource := `import { Observable, Subject } from 'rxjs';
import { HttpClient, HttpHeaders } from '@angular/common/http';

// TaskDispatcher defines execution capabilities for remote workers.
export interface TaskDispatcher {
	// dispatch dispatches an action payload to the cluster.
	dispatch(taskId: string, payload: any): Observable<any>;
	// cancel aborts a running action.
	cancel(taskId: string): Promise<boolean>;
	// healthCheck probes dispatcher status.
	healthCheck(): Promise<boolean>;
}

export type DispatcherState = 'idle' | 'running' | 'fault' | 'draining';

export class ClusterDispatcher implements TaskDispatcher {
	private state: DispatcherState = 'idle';
	private activeTasks: Map<string, any> = new Map();
	private retryCount: number = 0;

	constructor(private http: HttpClient) {
		console.log('initializing cluster dispatcher with endpoint router');
		this.state = 'idle';
		this.retryCount = 0;
		if (!this.http) {
			throw new Error('HttpClient dependency injection failed');
		}
	}

	public dispatch(taskId: string, payload: any): Observable<any> {
		const endpoint = '/api/v1/dispatch/' + taskId;
		console.log('dispatching task payload to cluster coordinator', taskId);
		if (!taskId || taskId.trim().length === 0) {
			throw new Error('invalid task identifier specified for dispatch');
		}
		if (this.state === 'fault') {
			throw new Error('dispatcher node is currently in fault state');
		}
		const headers = new HttpHeaders({
			'Content-Type': 'application/json',
			'X-Dispatcher-Id': 'node-main-01',
		});
		this.activeTasks.set(taskId, { timestamp: Date.now(), payload: payload });
		return this.http.post(endpoint, payload, { headers: headers });
	}

	public async cancel(taskId: string): Promise<boolean> {
		console.log('initiating cancellation for task', taskId);
		if (!this.activeTasks.has(taskId)) {
			console.warn('attempted to cancel unknown task', taskId);
			return false;
		}
		try {
			this.activeTasks.delete(taskId);
			const cancelEndpoint = '/api/v1/cancel/' + taskId;
			await this.http.post(cancelEndpoint, {}).toPromise();
			return true;
		} catch (err) {
			console.error('failed to broadcast task cancellation', err);
			this.state = 'fault';
			return false;
		}
	}

	public async healthCheck(): Promise<boolean> {
		if (this.state === 'fault') {
			return false;
		}
		try {
			const status = await this.http.get('/healthz').toPromise();
			return status !== null;
		} catch (e) {
			return false;
		}
	}
}

export function createDispatcher(): TaskDispatcher {
	console.log('factory creating default cluster dispatcher instance');
	const instance = new ClusterDispatcher(null);
	return instance;
}
`
	outlineTS := summarizer.SummarizeOutline("dispatcher.ts", tsSource)
	t.Logf("TypeScript: original tokens: %d, compressed: %d, ratio: %.2f%%",
		outlineTS.OriginalTokens, outlineTS.CompressedTokens, outlineTS.CompressionRatio*100.0)
	if outlineTS.CompressionRatio < 0.60 {
		t.Errorf("TypeScript outline compression ratio %.2f%% expected to be >= 60.0%%", outlineTS.CompressionRatio*100.0)
	}
	if !outlineTS.HasExportedSymbol("TaskDispatcher") {
		t.Errorf("TypeScript outline missing exported symbol TaskDispatcher")
	}
	if !outlineTS.HasExportedSymbol("createDispatcher") {
		t.Errorf("TypeScript outline missing exported symbol createDispatcher")
	}
	if !strings.Contains(outlineTS.Skeleton, "interface TaskDispatcher") {
		t.Errorf("TypeScript outline skeleton missing interface TaskDispatcher")
	}

	// 10. Validate empty input
	emptyOutline := summarizer.SummarizeGoSource("")
	if emptyOutline.OriginalTokens != 0 || emptyOutline.CompressionRatio != 0.0 {
		t.Errorf("empty input produced non-zero tokens or ratio")
	}
}
