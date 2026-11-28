package grammar

import (
	"crypto/sha256"
	"encoding/binary"
	"encoding/hex"
	"errors"
	"fmt"
	"math"
	"reflect"
	"sync"
)

// ErrBatchFSMRefusal is returned when a batched constrained-decode operation
// receives an invalid row index, an illegal token transition, a step on a
// finished/dead row, or an incompatible rollback checkpoint.
var ErrBatchFSMRefusal = errors.New("grammar: batch bitmask FSM refusal")

const defaultSchemaCompileCacheCapacity = 64

type schemaCacheKey struct {
	schemaDigest string
	eos          int
	vocabKey     string
	hasVocabKey  bool
	dec          any
}

type schemaCacheEntry struct {
	key  schemaCacheKey
	mask *JSONSchemaMask
}

// SchemaCompileCache caches compiled *JSONSchemaMask instances across decode
// requests so mid-market 3B-32B serving loops reuse the compiled schema,
// sorted-vocab trie, and per-state bitmasks instead of recompiling per request.
type SchemaCompileCache struct {
	mu        sync.Mutex
	capacity  int
	entries   map[schemaCacheKey]*JSONSchemaMask
	order     []schemaCacheKey // FIFO eviction order
	hits      int64
	misses    int64
	evictions int64
}

// NewSchemaCompileCache creates a bounded thread-safe compile cache.
// A non-positive capacity defaults to 64 entries.
func NewSchemaCompileCache(capacity int) *SchemaCompileCache {
	if capacity <= 0 {
		capacity = defaultSchemaCompileCacheCapacity
	}
	return &SchemaCompileCache{
		capacity: capacity,
		entries:  make(map[schemaCacheKey]*JSONSchemaMask, capacity),
	}
}

// GetOrCompile returns a cached *JSONSchemaMask for (schema, opts.EOS, dec),
// compiling and caching it on a miss. The boolean reports whether the call hit.
func (c *SchemaCompileCache) GetOrCompile(schema []byte, dec TokenDecoder, opts CompileOptions) (*JSONSchemaMask, bool, error) {
	return c.getOrCompile(schema, dec, opts, "", false)
}

// GetOrCompileWithKey behaves like GetOrCompile using an explicit vocabKey
// (for example a model/tokenizer digest) so non-comparable TokenDecoder values
// share cache entries deterministically.
func (c *SchemaCompileCache) GetOrCompileWithKey(vocabKey string, schema []byte, dec TokenDecoder, opts CompileOptions) (*JSONSchemaMask, bool, error) {
	return c.getOrCompile(schema, dec, opts, vocabKey, true)
}

func (c *SchemaCompileCache) getOrCompile(schema []byte, dec TokenDecoder, opts CompileOptions, vocabKey string, hasVocabKey bool) (*JSONSchemaMask, bool, error) {
	if dec == nil {
		return nil, false, errors.New("grammar: CompileJSONSchema needs a TokenDecoder (inject tokenizer TokenBytes)")
	}
	key, cacheable := makeSchemaCacheKey(schema, opts.EOS, dec, vocabKey, hasVocabKey)
	if cacheable && c != nil {
		c.mu.Lock()
		if mask, ok := c.entries[key]; ok {
			c.hits++
			c.mu.Unlock()
			return mask, true, nil
		}
		c.misses++
		c.mu.Unlock()
	} else if c != nil {
		c.mu.Lock()
		c.misses++
		c.mu.Unlock()
	}

	mask, err := CompileJSONSchema(schema, dec, opts)
	if err != nil {
		return nil, false, err
	}

	if cacheable && c != nil {
		c.mu.Lock()
		if existing, ok := c.entries[key]; ok {
			c.mu.Unlock()
			return existing, false, nil
		}
		if len(c.order) >= c.capacity {
			oldest := c.order[0]
			c.order = c.order[1:]
			delete(c.entries, oldest)
			c.evictions++
		}
		c.entries[key] = mask
		c.order = append(c.order, key)
		c.mu.Unlock()
	}
	return mask, false, nil
}

// Hits returns the number of cache hits.
func (c *SchemaCompileCache) Hits() int64 {
	c.mu.Lock()
	defer c.mu.Unlock()
	return c.hits
}

// Misses returns the number of cache misses.
func (c *SchemaCompileCache) Misses() int64 {
	c.mu.Lock()
	defer c.mu.Unlock()
	return c.misses
}

// Evictions returns the number of evicted entries.
func (c *SchemaCompileCache) Evictions() int64 {
	c.mu.Lock()
	defer c.mu.Unlock()
	return c.evictions
}

// Len returns the number of currently cached compiled schemas.
func (c *SchemaCompileCache) Len() int {
	c.mu.Lock()
	defer c.mu.Unlock()
	return len(c.entries)
}

func makeSchemaCacheKey(schema []byte, eos int, dec TokenDecoder, vocabKey string, hasVocabKey bool) (schemaCacheKey, bool) {
	sum := sha256.Sum256(schema)
	digest := hex.EncodeToString(sum[:])
	if hasVocabKey {
		return schemaCacheKey{
			schemaDigest: digest,
			eos:          eos,
			vocabKey:     vocabKey,
			hasVocabKey:  true,
		}, true
	}
	t := reflect.TypeOf(dec)
	if t != nil && t.Comparable() {
		return schemaCacheKey{
			schemaDigest: digest,
			eos:          eos,
			dec:          dec,
		}, true
	}
	return schemaCacheKey{
		schemaDigest: digest,
		eos:          eos,
		vocabKey:     fingerprintDecoder(dec),
		hasVocabKey:  true,
	}, true
}

func fingerprintDecoder(dec TokenDecoder) string {
	h := sha256.New()
	var buf [8]byte
	// Probe a deterministic prefix of token IDs to distinguish vocabularies
	// when the decoder is a non-comparable slice/wrapper without an explicit key.
	const probeLimit = 256
	for id := 0; id < probeLimit; id++ {
		b := dec.TokenBytes(id)
		binary.LittleEndian.PutUint64(buf[:], uint64(len(b)))
		_, _ = h.Write(buf[:])
		if len(b) > 0 {
			_, _ = h.Write(b)
		}
	}
	return hex.EncodeToString(h.Sum(nil))[:16]
}

// RowCheckpoint is an immutable snapshot of a single batch row's FSM state
// used for speculative decode verification and rollback.
type RowCheckpoint struct {
	row      int
	mask     *JSONSchemaMask
	cursors  []jsonCursor
	dead     bool
	finished bool
	step     int
	valid    bool
}

// BatchSnapshot captures the full state of every row in a BatchBitmaskFSM.
type BatchSnapshot struct {
	rows []RowCheckpoint
}

type batchFSMRow struct {
	mask     *JSONSchemaMask
	state    jsonMaskState
	dead     bool
	finished bool
	step     int
}

// BatchBitmaskFSM maintains B independent decode-time constrained-decoding
// states over compiled *JSONSchemaMask grammars, emitting packed uint64 logit
// bitmasks and supporting speculative per-row or full-batch rollback.
type BatchBitmaskFSM struct {
	vocabSize    int
	words        int
	rows         []batchFSMRow
	rollbackHook func(row, fromStep, toStep int)
}

// NewBatchBitmaskFSM constructs a batch FSM of len(masks) rows over a
// vocabulary of vocabSize tokens. A nil entry in masks marks an unconstrained row.
func NewBatchBitmaskFSM(masks []*JSONSchemaMask, vocabSize int) (*BatchBitmaskFSM, error) {
	if len(masks) == 0 || vocabSize <= 0 {
		return nil, fmt.Errorf("%w: batch size (%d) and vocab size (%d) must be positive", ErrBatchFSMRefusal, len(masks), vocabSize)
	}
	rows := make([]batchFSMRow, len(masks))
	for i, m := range masks {
		rows[i] = newBatchFSMRow(m, vocabSize)
	}
	return &BatchBitmaskFSM{
		vocabSize: vocabSize,
		words:     (vocabSize + 63) / 64,
		rows:      rows,
	}, nil
}

// NewUniformBatchBitmaskFSM constructs a batch FSM where all batchSize rows
// share the same compiled *JSONSchemaMask.
func NewUniformBatchBitmaskFSM(mask *JSONSchemaMask, batchSize, vocabSize int) (*BatchBitmaskFSM, error) {
	if batchSize <= 0 {
		return nil, fmt.Errorf("%w: batch size (%d) must be positive", ErrBatchFSMRefusal, batchSize)
	}
	masks := make([]*JSONSchemaMask, batchSize)
	for i := range masks {
		masks[i] = mask
	}
	return NewBatchBitmaskFSM(masks, vocabSize)
}

func newBatchFSMRow(m *JSONSchemaMask, vocabSize int) batchFSMRow {
	if m == nil {
		return batchFSMRow{}
	}
	m.cacheDecoded(vocabSize)
	cursors := make([]jsonCursor, len(m.programs))
	for i := range m.programs {
		cursors[i] = m.programs[i].start()
	}
	return batchFSMRow{
		mask:  m,
		state: jsonMaskState{cursors: cursors},
	}
}

// SetRollbackHook registers an optional callback invoked whenever a row is
// rolled back from fromStep to toStep.
func (b *BatchBitmaskFSM) SetRollbackHook(fn func(row, fromStep, toStep int)) {
	if b != nil {
		b.rollbackHook = fn
	}
}

// BatchSize returns the number of sequences in the batch.
func (b *BatchBitmaskFSM) BatchSize() int {
	if b == nil {
		return 0
	}
	return len(b.rows)
}

// VocabSize returns the configured vocabulary size.
func (b *BatchBitmaskFSM) VocabSize() int {
	if b == nil {
		return 0
	}
	return b.vocabSize
}

// BitmaskWords returns the number of uint64 words per row bitmask.
func (b *BatchBitmaskFSM) BitmaskWords() int {
	if b == nil {
		return 0
	}
	return b.words
}

// RowStep returns the number of accepted tokens stepped on row.
func (b *BatchBitmaskFSM) RowStep(row int) (int, error) {
	if err := b.checkRow(row); err != nil {
		return 0, err
	}
	return b.rows[row].step, nil
}

// IsFinished reports whether row has consumed EOS after a complete schema match.
func (b *BatchBitmaskFSM) IsFinished(row int) (bool, error) {
	if err := b.checkRow(row); err != nil {
		return false, err
	}
	return b.rows[row].finished, nil
}

// IsComplete reports whether row's current prefix is a complete schema match.
func (b *BatchBitmaskFSM) IsComplete(row int) (bool, error) {
	if err := b.checkRow(row); err != nil {
		return false, err
	}
	return b.rows[row].isComplete(), nil
}

// IsDead reports whether row is in an unrecoverable dead state.
func (b *BatchBitmaskFSM) IsDead(row int) (bool, error) {
	if err := b.checkRow(row); err != nil {
		return false, err
	}
	return b.rows[row].dead, nil
}

func (r *batchFSMRow) isComplete() bool {
	if r.mask == nil || r.dead {
		return false
	}
	for i := range r.state.cursors {
		if r.mask.programs[i].complete(&r.state.cursors[i]) {
			return true
		}
	}
	return false
}

func (b *BatchBitmaskFSM) checkRow(row int) error {
	if b == nil || row < 0 || row >= len(b.rows) {
		return fmt.Errorf("%w: row %d out of range", ErrBatchFSMRefusal, row)
	}
	return nil
}

// FillRowBitmask writes the packed uint64 token bitmask for row into dst.
// dst must have length >= b.BitmaskWords().
func (b *BatchBitmaskFSM) FillRowBitmask(row int, dst []uint64) error {
	if err := b.checkRow(row); err != nil {
		return err
	}
	if len(dst) < b.words {
		return fmt.Errorf("%w: bitmask dst len %d < %d", ErrBatchFSMRefusal, len(dst), b.words)
	}
	clear(dst[:b.words])
	r := &b.rows[row]
	if r.mask == nil {
		for id := 0; id < b.vocabSize; id++ {
			dst[id>>6] |= 1 << (uint(id) & 63)
		}
		return nil
	}
	if r.dead {
		return nil
	}
	eos := r.mask.eos
	if r.finished {
		if eos >= 0 && eos < b.vocabSize {
			dst[eos>>6] |= 1 << (uint(eos) & 63)
		}
		return nil
	}
	admitted := r.mask.admittedTokens(r.state, b.vocabSize)
	copy(dst[:b.words], admitted)
	if eos >= 0 && eos < b.vocabSize {
		if r.isComplete() {
			dst[eos>>6] |= 1 << (uint(eos) & 63)
		} else {
			dst[eos>>6] &^= 1 << (uint(eos) & 63)
		}
	}
	return nil
}

// FillBatchBitmask writes all B row bitmasks into dst, laid out as
// [batchSize * BitmaskWords()]uint64.
func (b *BatchBitmaskFSM) FillBatchBitmask(dst []uint64) error {
	if b == nil {
		return ErrBatchFSMRefusal
	}
	need := len(b.rows) * b.words
	if len(dst) < need {
		return fmt.Errorf("%w: batch bitmask dst len %d < %d", ErrBatchFSMRefusal, len(dst), need)
	}
	for row := range b.rows {
		off := row * b.words
		if err := b.FillRowBitmask(row, dst[off:off+b.words]); err != nil {
			return err
		}
	}
	return nil
}

// ApplyBatchLogits masks disallowed tokens to -inf for each row in logits.
func (b *BatchBitmaskFSM) ApplyBatchLogits(logits [][]float32) error {
	if b == nil || len(logits) != len(b.rows) {
		return fmt.Errorf("%w: logits batch len mismatch", ErrBatchFSMRefusal)
	}
	scratch := make([]uint64, b.words)
	for row := range b.rows {
		if len(logits[row]) != b.vocabSize {
			return fmt.Errorf("%w: row %d logits len %d != %d", ErrBatchFSMRefusal, row, len(logits[row]), b.vocabSize)
		}
		if err := b.FillRowBitmask(row, scratch); err != nil {
			return err
		}
		applyBitmaskToLogits(scratch, logits[row])
	}
	return nil
}

// ApplyFlatLogits masks disallowed tokens to -inf in a flat [B * VocabSize]float32 tensor.
func (b *BatchBitmaskFSM) ApplyFlatLogits(logits []float32) error {
	if b == nil || len(logits) != len(b.rows)*b.vocabSize {
		return fmt.Errorf("%w: flat logits len mismatch", ErrBatchFSMRefusal)
	}
	scratch := make([]uint64, b.words)
	for row := range b.rows {
		if err := b.FillRowBitmask(row, scratch); err != nil {
			return err
		}
		off := row * b.vocabSize
		applyBitmaskToLogits(scratch, logits[off:off+b.vocabSize])
	}
	return nil
}

func applyBitmaskToLogits(bitmask []uint64, logits []float32) {
	neg := float32(math.Inf(-1))
	for id := range logits {
		if bitmask[id>>6]&(1<<(uint(id)&63)) == 0 {
			logits[id] = neg
		}
	}
}

// StepRow advances row by one tokenID. If tokenID is illegal for row's current
// state, StepRow returns ErrBatchFSMRefusal and leaves row unchanged.
func (b *BatchBitmaskFSM) StepRow(row, tokenID int) error {
	if err := b.checkRow(row); err != nil {
		return err
	}
	if tokenID < 0 || tokenID >= b.vocabSize {
		return fmt.Errorf("%w: token %d out of range [0,%d)", ErrBatchFSMRefusal, tokenID, b.vocabSize)
	}
	r := &b.rows[row]
	if r.finished || r.dead {
		return fmt.Errorf("%w: row %d cannot advance (finished=%v dead=%v)", ErrBatchFSMRefusal, row, r.finished, r.dead)
	}
	if r.mask == nil {
		r.step++
		return nil
	}
	if tokenID == r.mask.eos {
		if !r.isComplete() {
			return fmt.Errorf("%w: row %d EOS before complete schema", ErrBatchFSMRefusal, row)
		}
		r.finished = true
		r.step++
		return nil
	}
	piece := r.mask.tokenBytes(tokenID)
	if len(piece) == 0 {
		return fmt.Errorf("%w: row %d empty token %d", ErrBatchFSMRefusal, row, tokenID)
	}
	next := make([]jsonCursor, len(r.state.cursors))
	copy(next, r.state.cursors)
	for _, ch := range piece {
		alive := false
		for i := range next {
			if r.mask.programs[i].step(&next[i], ch) {
				alive = true
			}
		}
		if !alive {
			return fmt.Errorf("%w: row %d token %d violates grammar", ErrBatchFSMRefusal, row, tokenID)
		}
	}
	r.state.cursors = next
	r.step++
	return nil
}

// StepBatch atomically advances every row in the batch by tokens[row].
// If any row fails, the entire batch is left unchanged.
func (b *BatchBitmaskFSM) StepBatch(tokens []int) error {
	if b == nil || len(tokens) != len(b.rows) {
		return fmt.Errorf("%w: StepBatch len mismatch", ErrBatchFSMRefusal)
	}
	snap := b.Snapshot()
	for row, tok := range tokens {
		if err := b.StepRow(row, tok); err != nil {
			_ = b.rollbackSilent(snap)
			return err
		}
	}
	return nil
}

// CheckpointRow captures an isolated rollback checkpoint for row.
func (b *BatchBitmaskFSM) CheckpointRow(row int) (RowCheckpoint, error) {
	if err := b.checkRow(row); err != nil {
		return RowCheckpoint{}, err
	}
	return b.rows[row].checkpoint(row), nil
}

// RollbackRow restores row to cp without affecting any other row in the batch,
// and invokes the registered rollback hook if step decreased.
func (b *BatchBitmaskFSM) RollbackRow(row int, cp RowCheckpoint) error {
	if err := b.checkRow(row); err != nil {
		return err
	}
	if !cp.valid || cp.row != row || cp.mask != b.rows[row].mask {
		return fmt.Errorf("%w: invalid checkpoint for row %d", ErrBatchFSMRefusal, row)
	}
	fromStep := b.rows[row].step
	b.rows[row].restore(cp)
	if b.rollbackHook != nil && fromStep != cp.step {
		b.rollbackHook(row, fromStep, cp.step)
	}
	return nil
}

// Snapshot captures a rollback snapshot of the entire batch.
func (b *BatchBitmaskFSM) Snapshot() BatchSnapshot {
	if b == nil {
		return BatchSnapshot{}
	}
	rows := make([]RowCheckpoint, len(b.rows))
	for i := range b.rows {
		rows[i] = b.rows[i].checkpoint(i)
	}
	return BatchSnapshot{rows: rows}
}

// Rollback restores every row in the batch to snap, invoking the rollback hook
// for each row whose step changed.
func (b *BatchBitmaskFSM) Rollback(snap BatchSnapshot) error {
	if b == nil || len(snap.rows) != len(b.rows) {
		return fmt.Errorf("%w: invalid batch snapshot", ErrBatchFSMRefusal)
	}
	for i, cp := range snap.rows {
		if !cp.valid || cp.row != i || cp.mask != b.rows[i].mask {
			return fmt.Errorf("%w: invalid snapshot row %d", ErrBatchFSMRefusal, i)
		}
	}
	for i, cp := range snap.rows {
		fromStep := b.rows[i].step
		b.rows[i].restore(cp)
		if b.rollbackHook != nil && fromStep != cp.step {
			b.rollbackHook(i, fromStep, cp.step)
		}
	}
	return nil
}

func (b *BatchBitmaskFSM) rollbackSilent(snap BatchSnapshot) error {
	if b == nil || len(snap.rows) != len(b.rows) {
		return ErrBatchFSMRefusal
	}
	for i, cp := range snap.rows {
		b.rows[i].restore(cp)
	}
	return nil
}

func (r *batchFSMRow) checkpoint(row int) RowCheckpoint {
	var cursors []jsonCursor
	if len(r.state.cursors) > 0 {
		cursors = make([]jsonCursor, len(r.state.cursors))
		copy(cursors, r.state.cursors)
	}
	return RowCheckpoint{
		row:      row,
		mask:     r.mask,
		cursors:  cursors,
		dead:     r.dead,
		finished: r.finished,
		step:     r.step,
		valid:    true,
	}
}

func (r *batchFSMRow) restore(cp RowCheckpoint) {
	if len(cp.cursors) > 0 {
		r.state.cursors = make([]jsonCursor, len(cp.cursors))
		copy(r.state.cursors, cp.cursors)
	} else {
		r.state.cursors = nil
	}
	r.dead = cp.dead
	r.finished = cp.finished
	r.step = cp.step
}
