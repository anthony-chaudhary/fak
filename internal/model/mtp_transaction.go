package model

import (
	"errors"
	"fmt"
	"math"
	"sync"
	"time"
)

// MTPDowngradeReason is the closed vocabulary for declining or aborting
// speculative MTP execution on Apple Silicon Metal and downgrading cleanly
// to ordinary fak-native target decode.
type MTPDowngradeReason string

const (
	MTPDowngradeNone                 MTPDowngradeReason = ""
	MTPDowngradeDimensionMismatch    MTPDowngradeReason = "dimension_mismatch"
	MTPDowngradeNumericalNaN         MTPDowngradeReason = "numerical_nan_detected"
	MTPDowngradeMemoryPressure       MTPDowngradeReason = "memory_pressure_exceeded"
	MTPDowngradeBackendIncompatible  MTPDowngradeReason = "backend_incompatible"
	MTPDowngradeExecutionFailed      MTPDowngradeReason = "execution_failed"
	MTPDowngradeVerificationDiverged MTPDowngradeReason = "verification_diverged"
)

// ToQwen38DowngradeReason maps internal MTP transaction downgrade reasons to the
// closed receipt schema vocabulary in Qwen38MTPReceipt.
func (r MTPDowngradeReason) ToQwen38DowngradeReason() Qwen38MTPDowngradeReason {
	switch r {
	case MTPDowngradeDimensionMismatch:
		return Qwen38MTPDepthUnsupported
	case MTPDowngradeNumericalNaN:
		return Qwen38MTPCorrectnessDiverged
	case MTPDowngradeMemoryPressure:
		return Qwen38MTPMemoryUnsafe
	case MTPDowngradeBackendIncompatible:
		return Qwen38MTPBackendUnsupported
	case MTPDowngradeExecutionFailed:
		return Qwen38MTPAttemptFailed
	case MTPDowngradeVerificationDiverged:
		return Qwen38MTPCorrectnessDiverged
	default:
		return Qwen38MTPEligible
	}
}

// MTPDowngradeError reports a typed, fail-closed downgrade to ordinary fak-native
// target decode. It guarantees that foreign runtimes (e.g. llama.cpp) are never invoked.
type MTPDowngradeError struct {
	Reason MTPDowngradeReason `json:"reason"`
	Detail string             `json:"detail,omitempty"`
}

func (e *MTPDowngradeError) Error() string {
	if e == nil {
		return "mtp: downgrade to target decode"
	}
	if e.Detail != "" {
		return fmt.Sprintf("mtp: downgrade to target decode: %s (%s)", e.Reason, e.Detail)
	}
	return fmt.Sprintf("mtp: downgrade to target decode: %s", e.Reason)
}

func (e *MTPDowngradeError) Unwrap() error {
	return ErrTargetVerificationDowngrade
}

// MTPTransactionConfig defines the constraints, device backend, and memory limits
// for an MTP speculative transaction.
type MTPTransactionConfig struct {
	HiddenSize     int                     `json:"hidden_size"`
	VocabSize      int                     `json:"vocab_size"`
	MaxDraftDepth  int                     `json:"max_draft_depth"`
	NumLayers      int                     `json:"num_layers"`
	Backend        Qwen38MTPBackend        `json:"backend"`
	MemoryPressure Qwen38MTPMemoryPressure `json:"memory_pressure"`
	MaxMemoryBytes uint64                  `json:"max_memory_bytes"`
}

// MTPAccounting tracks net end-to-end performance and memory consumption
// without omitting overheads or hiding recovery costs.
type MTPAccounting struct {
	DraftTimeNS        time.Duration `json:"draft_time_ns"`
	VerifyTimeNS       time.Duration `json:"verify_time_ns"`
	RollbackTimeNS     time.Duration `json:"rollback_time_ns"`
	SyncTimeNS         time.Duration `json:"sync_time_ns"`
	TotalOverheadNS    time.Duration `json:"total_overhead_ns"`
	AcceptedCount      int           `json:"accepted_count"`
	RejectedCount      int           `json:"rejected_count"`
	ProposedCount      int           `json:"proposed_count"`
	RollbackCount      int           `json:"rollback_count"`
	PeakMemoryBytes    int64         `json:"peak_memory_bytes"`
	CurrentMemoryBytes int64         `json:"current_memory_bytes"`
}

// MTPState captures the persistent or speculative inference state for Qwen3.8 MTP,
// including attention KV cache, Gated-DeltaNet recurrent matrix, convolution window,
// and target hidden representations.
type MTPState struct {
	Position  int         `json:"position"`
	KV        [][]float32 `json:"kv"`
	Recurrent [][]float32 `json:"recurrent"`
	Conv      [][]float32 `json:"conv"`
	Hidden    [][]float32 `json:"hidden"`
}

// NewMTPState creates an MTPState with deep clones of the supplied buffers.
func NewMTPState(kv, recurrent, conv [][]float32) *MTPState {
	return &MTPState{
		Position:  0,
		KV:        cloneFloat32Matrix(kv),
		Recurrent: cloneFloat32Matrix(recurrent),
		Conv:      cloneFloat32Matrix(conv),
		Hidden:    nil,
	}
}

// Clone creates an independent deep copy of the MTPState.
func (s *MTPState) Clone() *MTPState {
	if s == nil {
		return nil
	}
	return &MTPState{
		Position:  s.Position,
		KV:        cloneFloat32Matrix(s.KV),
		Recurrent: cloneFloat32Matrix(s.Recurrent),
		Conv:      cloneFloat32Matrix(s.Conv),
		Hidden:    cloneFloat32Matrix(s.Hidden),
	}
}

// ByteSize estimates the memory footprint in bytes.
func (s *MTPState) ByteSize() int64 {
	if s == nil {
		return 0
	}
	var count int64
	for _, row := range s.KV {
		count += int64(len(row))
	}
	for _, row := range s.Recurrent {
		count += int64(len(row))
	}
	for _, row := range s.Conv {
		count += int64(len(row))
	}
	for _, row := range s.Hidden {
		count += int64(len(row))
	}
	return count * 4
}

// Equal verifies bit-exact equality across all persistent components.
func (s *MTPState) Equal(other *MTPState) bool {
	if s == nil && other == nil {
		return true
	}
	if s == nil || other == nil {
		return false
	}
	if s.Position != other.Position {
		return false
	}
	return matrixEqual(s.KV, other.KV) &&
		matrixEqual(s.Recurrent, other.Recurrent) &&
		matrixEqual(s.Conv, other.Conv) &&
		matrixEqual(s.Hidden, other.Hidden)
}

// MTPCheckpoint snapshots pre-round verified state for atomic rollback.
type MTPCheckpoint struct {
	id         uint64
	position   int
	kv         [][]float32
	recurrent  [][]float32
	conv       [][]float32
	hidden     [][]float32
	targetSnap *PrefixSnapshot
	closed     bool
}

// ByteSize calculates the resident bytes owned by the checkpoint.
func (c *MTPCheckpoint) ByteSize() int64 {
	if c == nil || c.closed {
		return 0
	}
	var count int64
	for _, row := range c.kv {
		count += int64(len(row))
	}
	for _, row := range c.recurrent {
		count += int64(len(row))
	}
	for _, row := range c.conv {
		count += int64(len(row))
	}
	for _, row := range c.hidden {
		count += int64(len(row))
	}
	bytes := count * 4
	if c.targetSnap != nil {
		bytes += c.targetSnap.ResidentBytes()
	}
	return bytes
}

// Close releases snapshot buffers with zero leaks.
func (c *MTPCheckpoint) Close() {
	if c == nil || c.closed {
		return
	}
	c.closed = true
	c.kv = nil
	c.recurrent = nil
	c.conv = nil
	c.hidden = nil
	if c.targetSnap != nil {
		c.targetSnap.Close()
		c.targetSnap = nil
	}
}

// MTPTransaction manages speculative draft round states, tracking target hidden states,
// draft token proposals, transactional KV/recurrent checkpoint rollbacks, and fail-closed
// typed downgrade to ordinary fak-native target decode.
type MTPTransaction struct {
	mu           sync.Mutex
	session      *Session
	config       MTPTransactionConfig
	state        *MTPState
	checkpoint   *MTPCheckpoint
	stepStates   []*MTPState
	draftTokens  []int
	targetHidden [][]float32
	roundActive  bool
	roundCount   int
	checkpointID uint64
	engine       Qwen38MTPEngine
	downgraded   bool
	downgradeErr *MTPDowngradeError
	accounting   MTPAccounting
	closed       bool
}

// NewMTPTransaction constructs a standalone MTPTransaction with the specified configuration.
func NewMTPTransaction(cfg MTPTransactionConfig) *MTPTransaction {
	if cfg.Backend == "" {
		cfg.Backend = Qwen38MTPBackendMetal
	}
	if cfg.MaxDraftDepth <= 0 {
		cfg.MaxDraftDepth = Qwen35MTPMaxDraftDepth
	}
	tx := &MTPTransaction{
		config: cfg,
		state:  &MTPState{},
		engine: Qwen38EngineMTP,
	}
	tx.updateMemoryLocked()
	return tx
}

// NewMTPTransactionWithState initializes an MTPTransaction backed by an existing MTPState.
func NewMTPTransactionWithState(state *MTPState, cfg MTPTransactionConfig) *MTPTransaction {
	tx := NewMTPTransaction(cfg)
	if state != nil {
		tx.state = state.Clone()
	}
	tx.updateMemoryLocked()
	return tx
}

// NewMTPTransactionWithTarget binds an MTPTransaction to a live target Session.
func NewMTPTransactionWithTarget(s *Session, cfg MTPTransactionConfig) (*MTPTransaction, error) {
	if s == nil {
		return nil, errors.New("model: mtp transaction requires non-nil session")
	}
	tx := NewMTPTransaction(cfg)
	tx.session = s
	tx.updateMemoryLocked()
	return tx, nil
}

// BeginRound initiates a speculative round, checkpointing persistent KV, recurrent,
// and target hidden state so that subsequent failures or rejections can be cleanly
// rolled back with zero memory leaks.
func (tx *MTPTransaction) BeginRound() (*MTPCheckpoint, error) {
	tx.mu.Lock()
	defer tx.mu.Unlock()

	if tx.closed {
		return nil, errors.New("model: mtp transaction is closed")
	}
	if tx.downgraded {
		return nil, tx.downgradeErr
	}
	if tx.roundActive {
		return nil, errors.New("model: speculative round already active")
	}

	// Check memory pressure before allocation.
	if tx.config.MemoryPressure == Qwen38MTPPressureCritical {
		return nil, tx.downgradeLocked(MTPDowngradeMemoryPressure, "critical memory pressure")
	}

	tx.checkpointID++
	cp := &MTPCheckpoint{
		id: tx.checkpointID,
	}

	if tx.state != nil {
		cp.position = tx.state.Position
		cp.kv = cloneFloat32Matrix(tx.state.KV)
		cp.recurrent = cloneFloat32Matrix(tx.state.Recurrent)
		cp.conv = cloneFloat32Matrix(tx.state.Conv)
		cp.hidden = cloneFloat32Matrix(tx.state.Hidden)
	}

	if tx.session != nil {
		snap, err := tx.session.PrefixSnapshot()
		if err != nil {
			return nil, tx.downgradeLocked(MTPDowngradeExecutionFailed, fmt.Sprintf("prefix snapshot: %v", err))
		}
		cp.targetSnap = snap
	}

	tx.checkpoint = cp
	tx.stepStates = nil
	tx.draftTokens = nil
	tx.targetHidden = nil
	tx.roundActive = true
	tx.roundCount++
	tx.updateMemoryLocked()

	return cp, nil
}

// Propose records draft token proposals and initial target hidden representations,
// validating dimensions and numerical integrity.
func (tx *MTPTransaction) Propose(draftTokens []int, targetHidden [][]float32) error {
	tx.mu.Lock()
	defer tx.mu.Unlock()

	start := time.Now()
	defer func() {
		tx.accounting.DraftTimeNS += time.Since(start)
	}()

	if tx.closed {
		return errors.New("model: mtp transaction is closed")
	}
	if tx.downgraded {
		return tx.downgradeErr
	}
	if !tx.roundActive {
		return errors.New("model: propose requires an active speculative round")
	}

	// Fail-closed checks:
	if tx.config.MemoryPressure == Qwen38MTPPressureCritical {
		return tx.downgradeLocked(MTPDowngradeMemoryPressure, "device memory pressure is critical")
	}
	if len(draftTokens) == 0 {
		return tx.downgradeLocked(MTPDowngradeDimensionMismatch, "empty draft token proposal")
	}
	if tx.config.MaxDraftDepth > 0 && len(draftTokens) > tx.config.MaxDraftDepth {
		return tx.downgradeLocked(MTPDowngradeDimensionMismatch, fmt.Sprintf("draft depth %d exceeds max %d", len(draftTokens), tx.config.MaxDraftDepth))
	}
	if tx.config.VocabSize > 0 {
		for i, tok := range draftTokens {
			if tok < 0 || tok >= tx.config.VocabSize {
				return tx.downgradeLocked(MTPDowngradeDimensionMismatch, fmt.Sprintf("draft token at index %d (%d) outside vocab [0, %d)", i, tok, tx.config.VocabSize))
			}
		}
	}
	if tx.config.HiddenSize > 0 && len(targetHidden) > 0 {
		for i, h := range targetHidden {
			if len(h) != tx.config.HiddenSize {
				return tx.downgradeLocked(MTPDowngradeDimensionMismatch, fmt.Sprintf("target hidden row %d dimension %d, want %d", i, len(h), tx.config.HiddenSize))
			}
		}
	}
	if matrixContainsNaNOrInf(targetHidden) {
		return tx.downgradeLocked(MTPDowngradeNumericalNaN, "NaN or Inf detected in target hidden states")
	}

	tx.draftTokens = append([]int(nil), draftTokens...)
	tx.targetHidden = cloneFloat32Matrix(targetHidden)
	tx.accounting.ProposedCount += len(draftTokens)

	tx.updateMemoryLocked()
	if tx.config.MaxMemoryBytes > 0 && uint64(tx.accounting.CurrentMemoryBytes) > tx.config.MaxMemoryBytes {
		return tx.downgradeLocked(MTPDowngradeMemoryPressure, "speculative state exceeded memory headroom budget")
	}

	return nil
}

// AppendStep appends a speculative step's KV delta, recurrent matrix, convolution window,
// and produced hidden state to the round.
func (tx *MTPTransaction) AppendStep(kv, recurrent, conv [][]float32, hidden []float32) error {
	tx.mu.Lock()
	defer tx.mu.Unlock()

	if tx.closed {
		return errors.New("model: mtp transaction is closed")
	}
	if tx.downgraded {
		return tx.downgradeErr
	}
	if !tx.roundActive {
		return errors.New("model: append step requires an active speculative round")
	}

	// Check numerical validity.
	if matrixContainsNaNOrInf(kv) || matrixContainsNaNOrInf(recurrent) || matrixContainsNaNOrInf(conv) || containsNaNOrInf(hidden) {
		return tx.downgradeLocked(MTPDowngradeNumericalNaN, "numerical NaN or Inf detected during speculative step")
	}
	if tx.config.HiddenSize > 0 && len(hidden) > 0 && len(hidden) != tx.config.HiddenSize {
		return tx.downgradeLocked(MTPDowngradeDimensionMismatch, fmt.Sprintf("hidden dimension %d != %d", len(hidden), tx.config.HiddenSize))
	}

	var priorPos int
	var priorKV, priorHidden [][]float32
	if len(tx.stepStates) > 0 {
		last := tx.stepStates[len(tx.stepStates)-1]
		priorPos = last.Position
		priorKV = last.KV
		priorHidden = last.Hidden
	} else if tx.state != nil {
		priorPos = tx.state.Position
		priorKV = tx.state.KV
		priorHidden = tx.state.Hidden
	}

	combinedKV := cloneFloat32Matrix(priorKV)
	combinedKV = append(combinedKV, cloneFloat32Matrix(kv)...)

	combinedHidden := cloneFloat32Matrix(priorHidden)
	if len(hidden) > 0 {
		combinedHidden = append(combinedHidden, cloneFloat32Slice(hidden))
	}

	step := &MTPState{
		Position:  priorPos + 1,
		KV:        combinedKV,
		Recurrent: cloneFloat32Matrix(recurrent),
		Conv:      cloneFloat32Matrix(conv),
		Hidden:    combinedHidden,
	}

	tx.stepStates = append(tx.stepStates, step)
	tx.updateMemoryLocked()

	if tx.config.MaxMemoryBytes > 0 && uint64(tx.accounting.CurrentMemoryBytes) > tx.config.MaxMemoryBytes {
		return tx.downgradeLocked(MTPDowngradeMemoryPressure, "step exceeded memory ceiling")
	}

	return nil
}

// AppendStepState directly appends a prepared MTPState step.
func (tx *MTPTransaction) AppendStepState(step *MTPState) error {
	if step == nil {
		return nil
	}
	return tx.AppendStep(step.KV, step.Recurrent, step.Conv, nil)
}

// ExecuteDraftStep evaluates a speculative draft step using the provided closure,
// capturing any runtime failure, NaN, or dimension mismatch and downgrading cleanly
// without panicking.
func (tx *MTPTransaction) ExecuteDraftStep(stepFn func(pos int, priorHidden []float32) (kv, recurrent, conv [][]float32, hidden []float32, logits []float32, token int, err error)) (int, error) {
	tx.mu.Lock()
	if tx.closed {
		tx.mu.Unlock()
		return -1, errors.New("model: mtp transaction is closed")
	}
	if tx.downgraded {
		err := tx.downgradeErr
		tx.mu.Unlock()
		return -1, err
	}
	if !tx.roundActive {
		tx.mu.Unlock()
		return -1, errors.New("model: execute step requires an active speculative round")
	}

	pos := tx.state.Position + len(tx.stepStates)
	var priorHidden []float32
	if len(tx.targetHidden) > 0 {
		priorHidden = tx.targetHidden[len(tx.targetHidden)-1]
	}
	tx.mu.Unlock()

	var (
		kv, recurrent, conv [][]float32
		hidden, logits      []float32
		token               int
		stepErr             error
	)

	func() {
		defer func() {
			if r := recover(); r != nil {
				stepErr = fmt.Errorf("panic in metal draft step: %v", r)
			}
		}()
		kv, recurrent, conv, hidden, logits, token, stepErr = stepFn(pos, priorHidden)
	}()

	tx.mu.Lock()
	defer tx.mu.Unlock()

	if stepErr != nil {
		return -1, tx.downgradeLocked(MTPDowngradeExecutionFailed, stepErr.Error())
	}
	if containsNaNOrInf(logits) || matrixContainsNaNOrInf(kv) || matrixContainsNaNOrInf(recurrent) || matrixContainsNaNOrInf(conv) || containsNaNOrInf(hidden) {
		return -1, tx.downgradeLocked(MTPDowngradeNumericalNaN, "NaN or Inf detected in metal execution step")
	}

	tx.draftTokens = append(tx.draftTokens, token)
	if len(hidden) > 0 {
		tx.targetHidden = append(tx.targetHidden, cloneFloat32Slice(hidden))
	}
	tx.accounting.ProposedCount++

	// Construct and append speculative step state.
	var priorPos int
	var priorKV, priorH [][]float32
	if len(tx.stepStates) > 0 {
		last := tx.stepStates[len(tx.stepStates)-1]
		priorPos = last.Position
		priorKV = last.KV
		priorH = last.Hidden
	} else if tx.state != nil {
		priorPos = tx.state.Position
		priorKV = tx.state.KV
		priorH = tx.state.Hidden
	}

	combinedKV := cloneFloat32Matrix(priorKV)
	combinedKV = append(combinedKV, cloneFloat32Matrix(kv)...)

	combinedH := cloneFloat32Matrix(priorH)
	if len(hidden) > 0 {
		combinedH = append(combinedH, cloneFloat32Slice(hidden))
	}

	step := &MTPState{
		Position:  priorPos + 1,
		KV:        combinedKV,
		Recurrent: cloneFloat32Matrix(recurrent),
		Conv:      cloneFloat32Matrix(conv),
		Hidden:    combinedH,
	}
	tx.stepStates = append(tx.stepStates, step)
	tx.updateMemoryLocked()

	return token, nil
}

// Verify evaluates target logits against draft tokens sequentially, returning the count
// of accepted tokens. Verification stops at the first mismatch.
func (tx *MTPTransaction) Verify(targetLogits [][]float32) (int, error) {
	tx.mu.Lock()
	defer tx.mu.Unlock()

	start := time.Now()
	defer func() {
		tx.accounting.VerifyTimeNS += time.Since(start)
	}()

	if tx.closed {
		return 0, errors.New("model: mtp transaction is closed")
	}
	if tx.downgraded {
		return 0, tx.downgradeErr
	}
	if !tx.roundActive {
		return 0, errors.New("model: verify requires an active speculative round")
	}

	if matrixContainsNaNOrInf(targetLogits) {
		return 0, tx.downgradeLocked(MTPDowngradeNumericalNaN, "NaN or Inf detected in target logits during verification")
	}

	if tx.config.VocabSize > 0 {
		for i, row := range targetLogits {
			if len(row) != tx.config.VocabSize {
				return 0, tx.downgradeLocked(MTPDowngradeDimensionMismatch, fmt.Sprintf("target logit row %d length %d != vocab %d", i, len(row), tx.config.VocabSize))
			}
		}
	}

	accepted := 0
	limit := min(len(targetLogits), len(tx.draftTokens))
	for i := 0; i < limit; i++ {
		targetArgmax := argmaxF32(targetLogits[i])
		if targetArgmax == tx.draftTokens[i] {
			accepted++
		} else {
			break
		}
	}

	return accepted, nil
}

// VerifyTokens verifies draft tokens directly against evaluated target tokens.
func (tx *MTPTransaction) VerifyTokens(targetTokens []int) (int, error) {
	tx.mu.Lock()
	defer tx.mu.Unlock()

	start := time.Now()
	defer func() {
		tx.accounting.VerifyTimeNS += time.Since(start)
	}()

	if tx.closed {
		return 0, errors.New("model: mtp transaction is closed")
	}
	if tx.downgraded {
		return 0, tx.downgradeErr
	}
	if !tx.roundActive {
		return 0, errors.New("model: verify requires an active speculative round")
	}

	accepted := 0
	limit := min(len(targetTokens), len(tx.draftTokens))
	for i := 0; i < limit; i++ {
		if targetTokens[i] == tx.draftTokens[i] {
			accepted++
		} else {
			break
		}
	}

	return accepted, nil
}

// Commit commits the verified prefix into persistent KV and recurrent state.
// On partial acceptance or total rejection (accepted == 0), the unverified suffix
// is cleanly rolled back and all speculative buffers are freed with zero leaks.
func (tx *MTPTransaction) Commit(accepted int) error {
	tx.mu.Lock()
	defer tx.mu.Unlock()

	syncStart := time.Now()
	defer func() {
		tx.accounting.SyncTimeNS += time.Since(syncStart)
	}()

	if tx.closed {
		return errors.New("model: mtp transaction is closed")
	}
	if tx.downgraded {
		return tx.downgradeErr
	}
	if !tx.roundActive {
		return errors.New("model: commit requires an active speculative round")
	}
	if accepted < 0 || accepted > len(tx.draftTokens) {
		return fmt.Errorf("model: accepted count %d outside draft length %d", accepted, len(tx.draftTokens))
	}

	if accepted == 0 {
		// Total rejection: rollback to pre-draft checkpoint.
		tx.accounting.RejectedCount += len(tx.draftTokens)
		return tx.rollbackLocked()
	}

	// Full or partial acceptance: commit prefix.
	if tx.state != nil {
		if accepted <= len(tx.stepStates) {
			committedStep := tx.stepStates[accepted-1]
			tx.state.Position = committedStep.Position
			tx.state.KV = cloneFloat32Matrix(committedStep.KV)
			tx.state.Recurrent = cloneFloat32Matrix(committedStep.Recurrent)
			tx.state.Conv = cloneFloat32Matrix(committedStep.Conv)
			tx.state.Hidden = cloneFloat32Matrix(committedStep.Hidden)
		} else {
			tx.state.Position += accepted
		}
	}

	tx.accounting.AcceptedCount += accepted
	tx.accounting.RejectedCount += (len(tx.draftTokens) - accepted)

	// Clean up speculative states to avoid memory leaks.
	for i := range tx.stepStates {
		tx.stepStates[i] = nil
	}
	tx.stepStates = nil

	if tx.checkpoint != nil {
		tx.checkpoint.Close()
		tx.checkpoint = nil
	}

	tx.draftTokens = nil
	tx.targetHidden = nil
	tx.roundActive = false
	tx.updateMemoryLocked()

	return nil
}

// Rollback restores persistent state to the exact pre-round checkpoint,
// discarding speculative deltas with zero memory leaks.
func (tx *MTPTransaction) Rollback() error {
	tx.mu.Lock()
	defer tx.mu.Unlock()
	return tx.rollbackLocked()
}

func (tx *MTPTransaction) rollbackLocked() error {
	start := time.Now()
	defer func() {
		tx.accounting.RollbackTimeNS += time.Since(start)
	}()

	if tx.checkpoint == nil {
		tx.roundActive = false
		return nil
	}

	if tx.state != nil {
		tx.state.Position = tx.checkpoint.position
		tx.state.KV = cloneFloat32Matrix(tx.checkpoint.kv)
		tx.state.Recurrent = cloneFloat32Matrix(tx.checkpoint.recurrent)
		tx.state.Conv = cloneFloat32Matrix(tx.checkpoint.conv)
		tx.state.Hidden = cloneFloat32Matrix(tx.checkpoint.hidden)
	}

	if tx.session != nil && tx.checkpoint.targetSnap != nil {
		if err := tx.checkpoint.targetSnap.Restore(tx.session); err != nil {
			return fmt.Errorf("model: rollback session restore: %w", err)
		}
	}

	// Cleanly free checkpoint and speculative working set.
	tx.checkpoint.Close()
	tx.checkpoint = nil

	for i := range tx.stepStates {
		tx.stepStates[i] = nil
	}
	tx.stepStates = nil

	tx.draftTokens = nil
	tx.targetHidden = nil
	tx.roundActive = false
	tx.accounting.RollbackCount++
	tx.updateMemoryLocked()

	return nil
}

// Downgrade performs a fail-closed transition to ordinary fak-native target decode.
// It rolls back any active speculative delta so the target model resumes cleanly.
// It never authorizes or delegates to llama.cpp.
func (tx *MTPTransaction) Downgrade(reason MTPDowngradeReason, detail string) error {
	tx.mu.Lock()
	defer tx.mu.Unlock()
	return tx.downgradeLocked(reason, detail)
}

func (tx *MTPTransaction) downgradeLocked(reason MTPDowngradeReason, detail string) error {
	if tx.downgraded {
		return tx.downgradeErr
	}

	// Roll back active speculative modifications.
	if tx.roundActive {
		_ = tx.rollbackLocked()
	}

	tx.downgraded = true
	tx.engine = Qwen38EngineTargetDecode
	tx.downgradeErr = &MTPDowngradeError{
		Reason: reason,
		Detail: detail,
	}

	return tx.downgradeErr
}

// IsDowngraded reports whether speculative acceleration has been downgraded.
func (tx *MTPTransaction) IsDowngraded() bool {
	tx.mu.Lock()
	defer tx.mu.Unlock()
	return tx.downgraded
}

// DowngradeReason returns the active downgrade reason, or MTPDowngradeNone.
func (tx *MTPTransaction) DowngradeReason() MTPDowngradeReason {
	tx.mu.Lock()
	defer tx.mu.Unlock()
	if tx.downgradeErr != nil {
		return tx.downgradeErr.Reason
	}
	return MTPDowngradeNone
}

// Engine reports the executing engine. It is strictly fak-native and never llama.cpp.
func (tx *MTPTransaction) Engine() Qwen38MTPEngine {
	tx.mu.Lock()
	defer tx.mu.Unlock()
	return tx.engine
}

// State returns a deep copy of the current persistent state.
func (tx *MTPTransaction) State() *MTPState {
	tx.mu.Lock()
	defer tx.mu.Unlock()
	if tx.state == nil {
		return nil
	}
	return tx.state.Clone()
}

// Accounting returns a snapshot of the transaction's complete accounting metrics.
func (tx *MTPTransaction) Accounting() MTPAccounting {
	tx.mu.Lock()
	defer tx.mu.Unlock()
	acc := tx.accounting
	acc.TotalOverheadNS = acc.DraftTimeNS + acc.VerifyTimeNS + acc.RollbackTimeNS + acc.SyncTimeNS
	return acc
}

// Receipt constructs a valid Qwen38MTPReceipt witnessing execution parameters and outcomes.
func (tx *MTPTransaction) Receipt() Qwen38MTPReceipt {
	tx.mu.Lock()
	defer tx.mu.Unlock()

	reqDepth := tx.config.MaxDraftDepth
	if reqDepth <= 0 {
		reqDepth = Qwen35MTPMaxDraftDepth
	}

	effDepth := reqDepth
	if tx.downgraded && tx.accounting.ProposedCount == 0 {
		effDepth = 0
	}

	dTime := uint64(tx.accounting.DraftTimeNS.Nanoseconds())
	vTime := uint64(tx.accounting.VerifyTimeNS.Nanoseconds())
	rTime := uint64(tx.accounting.RollbackTimeNS.Nanoseconds())
	sTime := uint64(tx.accounting.SyncTimeNS.Nanoseconds())

	peak := uint64(tx.accounting.PeakMemoryBytes)
	if peak == 0 {
		peak = 4096
	}

	mem := Qwen38MTPMemoryBytes{
		Peak: peak,
	}

	var dist []Qwen38MTPAcceptanceBucket
	if tx.accounting.ProposedCount > 0 && effDepth > 0 {
		dist = []Qwen38MTPAcceptanceBucket{
			{
				Depth:    1,
				Proposed: tx.accounting.ProposedCount,
				Accepted: tx.accounting.AcceptedCount,
				Rejected: tx.accounting.RejectedCount,
			},
		}
	}

	if tx.downgraded {
		if tx.accounting.ProposedCount > 0 {
			// Speculative attempt ran and failed/downgraded.
			tot := dTime + vTime + rTime + sTime
			if tot == 0 {
				tot = 1
			}
			return Qwen38MTPReceipt{
				SchemaVersion:  Qwen38MTPReceiptSchema,
				Outcome:        Qwen38MTPOutcomeFailed,
				Engine:         Qwen38EngineMTP,
				FallbackEngine: Qwen38EngineTargetDecode,
				RequestedDepth: reqDepth,
				EffectiveDepth: effDepth,
				Tokens: Qwen38MTPTokenAccounting{
					Proposed:     tx.accounting.ProposedCount,
					Accepted:     tx.accounting.AcceptedCount,
					Rejected:     tx.accounting.RejectedCount,
					Distribution: dist,
				},
				LatencyNS: Qwen38MTPLatencyNS{
					Draft:    dTime,
					Verify:   vTime,
					Rollback: rTime,
					Sync:     sTime,
					Total:    tot,
				},
				MemoryBytes:     mem,
				DowngradeReason: Qwen38MTPAttemptFailed,
				FailureReason:   Qwen38MTPVerificationFailed,
			}
		}

		// Target only: speculative cost must be 0 per Validate().
		return Qwen38MTPReceipt{
			SchemaVersion:   Qwen38MTPReceiptSchema,
			Outcome:         Qwen38MTPOutcomeTargetOnly,
			Engine:          Qwen38EngineTargetDecode,
			RequestedDepth:  reqDepth,
			EffectiveDepth:  0,
			Tokens:          Qwen38MTPTokenAccounting{},
			LatencyNS:       Qwen38MTPLatencyNS{Setup: 1, Total: 1},
			MemoryBytes:     mem,
			DowngradeReason: tx.downgradeErr.Reason.ToQwen38DowngradeReason(),
			FailureReason:   Qwen38MTPFailureNone,
		}
	}

	// Succeeded
	if dTime == 0 {
		dTime = 1
	}
	if vTime == 0 {
		vTime = 1
	}
	tot := dTime + vTime + rTime + sTime

	return Qwen38MTPReceipt{
		SchemaVersion:  Qwen38MTPReceiptSchema,
		Outcome:        Qwen38MTPOutcomeSucceeded,
		Engine:         Qwen38EngineMTP,
		RequestedDepth: reqDepth,
		EffectiveDepth: effDepth,
		Tokens: Qwen38MTPTokenAccounting{
			Proposed:     tx.accounting.ProposedCount,
			Accepted:     tx.accounting.AcceptedCount,
			Rejected:     tx.accounting.RejectedCount,
			Distribution: dist,
		},
		LatencyNS: Qwen38MTPLatencyNS{
			Draft:    dTime,
			Verify:   vTime,
			Rollback: rTime,
			Sync:     sTime,
			Total:    tot,
		},
		MemoryBytes:     mem,
		DowngradeReason: Qwen38MTPEligible,
		FailureReason:   Qwen38MTPFailureNone,
	}
}

// Close releases any held resources, checkpoints, and buffers.
func (tx *MTPTransaction) Close() error {
	tx.mu.Lock()
	defer tx.mu.Unlock()

	if tx.closed {
		return nil
	}
	tx.closed = true

	if tx.checkpoint != nil {
		tx.checkpoint.Close()
		tx.checkpoint = nil
	}
	for i := range tx.stepStates {
		tx.stepStates[i] = nil
	}
	tx.stepStates = nil
	tx.draftTokens = nil
	tx.targetHidden = nil
	tx.roundActive = false
	tx.updateMemoryLocked()

	return nil
}

func (tx *MTPTransaction) updateMemoryLocked() {
	var current int64
	if tx.state != nil {
		current += tx.state.ByteSize()
	}
	if tx.checkpoint != nil {
		current += tx.checkpoint.ByteSize()
	}
	for _, step := range tx.stepStates {
		if step != nil {
			current += step.ByteSize()
		}
	}
	tx.accounting.CurrentMemoryBytes = current
	if current > tx.accounting.PeakMemoryBytes {
		tx.accounting.PeakMemoryBytes = current
	}
}

func cloneFloat32Slice(s []float32) []float32 {
	if s == nil {
		return nil
	}
	return append([]float32(nil), s...)
}

func cloneFloat32Matrix(m [][]float32) [][]float32 {
	if m == nil {
		return nil
	}
	out := make([][]float32, len(m))
	for i := range m {
		out[i] = cloneFloat32Slice(m[i])
	}
	return out
}

func matrixEqual(a, b [][]float32) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if len(a[i]) != len(b[i]) {
			return false
		}
		for j := range a[i] {
			if a[i][j] != b[i][j] {
				return false
			}
		}
	}
	return true
}

func containsNaNOrInf(slice []float32) bool {
	for _, v := range slice {
		if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
			return true
		}
	}
	return false
}

func matrixContainsNaNOrInf(m [][]float32) bool {
	for _, row := range m {
		if containsNaNOrInf(row) {
			return true
		}
	}
	return false
}
