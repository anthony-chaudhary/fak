// Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Direct port of the memory-mapped dataset store from ai-dynamo/aiperf,
// src/aiperf/dataset/memory_map_utils.py (MemoryMapDatasetBackingStore /
// MemoryMapDatasetClient, PAYLOAD_BYTES path), pinned at
// https://github.com/ai-dynamo/aiperf/blob/ab610add4e8876fca223197596c728e17365f381/src/aiperf/dataset/memory_map_utils.py
// (commit ab610add4e8876fca223197596c728e17365f381, Apache-2.0).
//
// Mechanism: a writer streams pre-encoded request payloads to dataset.dat as
// they arrive (constant memory regardless of dataset size, insertion order
// preserved) and finalizes a byte-offset index (index.dat) for O(1) lookups.
// Readers open both files read-only and map the data file through the trunk
// MappedRegion seam (prefault.go, a direct port of the same upstream source);
// payload bytes are copied out of the shared read-only region verbatim, with
// no deserialization. loadgen uses this to replay stored requests against an
// endpoint instead of synthesizing prompts.
package main

import (
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"math"
	"os"
	"path/filepath"
	"sync"
)

// Storage formats, mirroring the upstream MemoryMapFormat enum.
const (
	// FormatConversation stores each entry as a serialized conversation object.
	FormatConversation = "conversation"
	// FormatPayloadBytes stores pre-encoded per-turn payload bytes for verbatim
	// API replay; readers send the bytes back to the transport untouched.
	FormatPayloadBytes = "payload_bytes"
)

// Sentinel errors for the mmap dataset store and client. Callers match these
// with errors.Is; wrapped messages carry context, never the identity.
var (
	// ErrNotFound marks a lookup for a session or turn the dataset does not hold
	// (including any payload lookup on a non-payload_bytes dataset).
	ErrNotFound = errors.New("loadgen/mmap: entry not found")
	// ErrAlreadyFinalized marks writes to a store whose index is already on disk.
	ErrAlreadyFinalized = errors.New("loadgen/mmap: store already finalized")
	// ErrDuplicateID marks a second add of the same session id.
	ErrDuplicateID = errors.New("loadgen/mmap: duplicate session id")
	// ErrIndexMismatch marks an index whose total_size disagrees with the data
	// file on disk.
	ErrIndexMismatch = errors.New("loadgen/mmap: index/data size mismatch")
	// ErrCorruptIndex marks an index that fails structural validation.
	ErrCorruptIndex = errors.New("loadgen/mmap: corrupt index")
)

// ConversationOffset is the byte position of one serialized conversation.
type ConversationOffset struct {
	Offset int `json:"offset"`
	Size   int `json:"size"`
}

// PayloadOffset is the byte position of one turn's payload plus the scalars
// persisted beside it so metric enrichment survives payload stripping. Null
// fields mirror the upstream schema (absent = unknown).
type PayloadOffset struct {
	Offset               int      `json:"offset"`
	Size                 int      `json:"size"`
	MaxTokens            *int     `json:"max_tokens"`
	TimestampMS          *float64 `json:"timestamp"`
	SourceKind           *string  `json:"source_kind"`
	AudioDurationSeconds *float64 `json:"audio_duration_seconds"`
	NumChoices           int      `json:"num_choices"`
}

// DatasetIndex is the on-disk index (index.dat), shaped like the upstream
// MemoryMapDatasetIndex; field order matches its serialization.
type DatasetIndex struct {
	ConversationIDs []string                      `json:"conversation_ids"`
	Format          string                        `json:"format"`
	Offsets         map[string]ConversationOffset `json:"offsets"`
	PayloadOffsets  map[string][]PayloadOffset    `json:"payload_offsets"`
	TotalSize       int                           `json:"total_size"`
}

// PayloadMeta carries the per-turn scalars a writer persists alongside payload
// offsets. Nil pointers mean "not recorded"; NumChoices 0 means "unset" and
// defaults to 1, matching the upstream default.
type PayloadMeta struct {
	MaxTokens            *int
	TimestampMS          *float64
	SourceKind           *string
	AudioDurationSeconds *float64
	NumChoices           int
}

// TurnRecord is one turn: its pre-encoded payload plus sidecar scalars.
type TurnRecord struct {
	Payload []byte
	Meta    PayloadMeta
}

// PayloadTurnData is a payload slice plus its restored scalars, the unit
// loadgen replays and enriches metrics from.
type PayloadTurnData struct {
	PayloadBytes         []byte
	MaxTokens            *int
	TimestampMS          *float64
	SourceKind           *string
	AudioDurationSeconds *float64
	NumChoices           int
}

// DatasetStore streams payload records to disk (writer side). One store owns
// one run directory: <baseDir>/loadgen_mmap_<id>/{dataset.dat,index.dat}.
//
// Writes are immediate and append-only, so memory stays constant regardless of
// dataset size and insertion order is preserved. Methods are serialized by an
// internal mutex; a store is safe for concurrent use.
type DatasetStore struct {
	dir       string
	format    string
	dataPath  string
	indexPath string

	mu             sync.Mutex
	dataFile       *os.File
	currentOffset  int
	sessionIDs     []string
	payloadOffsets map[string][]PayloadOffset
	finalized      bool
}

// NewDatasetStore creates the run directory and opens dataset.dat for
// streaming writes. Only FormatPayloadBytes is supported by this leaf; the
// conversation format belongs to its sibling leaf.
func NewDatasetStore(baseDir, id, format string) (*DatasetStore, error) {
	if id == "" {
		return nil, errors.New("loadgen/mmap: store id must be non-empty")
	}
	if format != FormatPayloadBytes {
		return nil, fmt.Errorf("loadgen/mmap: unsupported format %q (only %q)", format, FormatPayloadBytes)
	}
	dir := filepath.Join(baseDir, "loadgen_mmap_"+id)
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return nil, fmt.Errorf("loadgen/mmap: create dir: %w", err)
	}
	dataPath := filepath.Join(dir, "dataset.dat")
	indexPath := filepath.Join(dir, "index.dat")
	f, err := os.OpenFile(dataPath, os.O_CREATE|os.O_TRUNC|os.O_WRONLY, 0o644)
	if err != nil {
		return nil, fmt.Errorf("loadgen/mmap: open data file: %w", err)
	}
	return &DatasetStore{
		dir:            dir,
		format:         format,
		dataPath:       dataPath,
		indexPath:      indexPath,
		dataFile:       f,
		payloadOffsets: map[string][]PayloadOffset{},
	}, nil
}

// DataPath returns the on-disk location of dataset.dat.
func (s *DatasetStore) DataPath() string { return s.dataPath }

// IndexPath returns the on-disk location of index.dat.
func (s *DatasetStore) IndexPath() string { return s.indexPath }

// AddSession appends one session's turns to dataset.dat immediately and
// records their cumulative byte offsets. Session ids are unique per store;
// adding the same id twice fails with ErrDuplicateID.
func (s *DatasetStore) AddSession(sessionID string, turns []TurnRecord) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.finalized {
		return fmt.Errorf("add session %q: %w", sessionID, ErrAlreadyFinalized)
	}
	if sessionID == "" {
		return errors.New("loadgen/mmap: session id must be non-empty")
	}
	if _, exists := s.payloadOffsets[sessionID]; exists {
		return fmt.Errorf("add session %q: %w", sessionID, ErrDuplicateID)
	}
	offsets := make([]PayloadOffset, 0, len(turns))
	for i, t := range turns {
		maxTokens, err := resolveTurnMaxTokens(t.Meta, t.Payload)
		if err != nil {
			return fmt.Errorf("add session %q turn %d: %v", sessionID, i, err)
		}
		numChoices := t.Meta.NumChoices
		if numChoices == 0 {
			numChoices = 1
		}
		if numChoices < 0 {
			return fmt.Errorf("add session %q turn %d: num_choices must be >= 0, got %d", sessionID, i, numChoices)
		}
		if t.Meta.AudioDurationSeconds != nil && *t.Meta.AudioDurationSeconds <= 0 {
			return fmt.Errorf("add session %q turn %d: audio_duration_seconds must be > 0", sessionID, i)
		}
		if err := writeFull(s.dataFile, t.Payload); err != nil {
			return fmt.Errorf("add session %q turn %d: %w", sessionID, i, err)
		}
		offsets = append(offsets, PayloadOffset{
			Offset:               s.currentOffset,
			Size:                 len(t.Payload),
			MaxTokens:            maxTokens,
			TimestampMS:          t.Meta.TimestampMS,
			SourceKind:           t.Meta.SourceKind,
			AudioDurationSeconds: t.Meta.AudioDurationSeconds,
			NumChoices:           numChoices,
		})
		s.currentOffset += len(t.Payload)
	}
	s.payloadOffsets[sessionID] = offsets
	s.sessionIDs = append(s.sessionIDs, sessionID)
	return nil
}

// Finalize closes dataset.dat and writes index.dat. After Finalize the store
// is read-only history: further adds fail with ErrAlreadyFinalized.
func (s *DatasetStore) Finalize() error {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.finalized {
		return fmt.Errorf("finalize: %w", ErrAlreadyFinalized)
	}
	if err := s.closeDataFile(); err != nil {
		return fmt.Errorf("close data file: %w", err)
	}
	idx := DatasetIndex{
		ConversationIDs: s.sessionIDs,
		Format:          s.format,
		Offsets:         map[string]ConversationOffset{},
		PayloadOffsets:  s.payloadOffsets,
		TotalSize:       s.currentOffset,
	}
	if idx.ConversationIDs == nil {
		idx.ConversationIDs = []string{}
	}
	b, err := json.Marshal(&idx)
	if err != nil {
		return fmt.Errorf("serialize index: %w", err)
	}
	f, err := os.OpenFile(s.indexPath, os.O_CREATE|os.O_TRUNC|os.O_WRONLY, 0o644)
	if err != nil {
		return fmt.Errorf("open index file: %w", err)
	}
	if err := writeFull(f, b); err != nil {
		_ = f.Close()
		return fmt.Errorf("write index: %w", err)
	}
	if err := f.Close(); err != nil {
		return fmt.Errorf("close index file: %w", err)
	}
	s.finalized = true
	return nil
}

// Close releases the data file handle. After Finalize the file is already
// closed, so this is a no-op there; safe to call multiple times.
func (s *DatasetStore) Close() error {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.closeDataFile()
}

// closeDataFile closes the open data file exactly once. Callers hold s.mu.
func (s *DatasetStore) closeDataFile() error {
	if s.dataFile == nil {
		return nil
	}
	err := s.dataFile.Close()
	s.dataFile = nil
	return err
}

// DatasetClient reads a finalized dataset through the trunk MappedRegion
// read-mapping seam (reader side). Lookups are O(1) index reads plus a copy
// out of the shared read-only region; payload bytes come back verbatim with
// no deserialization.
//
// All methods are safe for concurrent use: the region is read-only and no
// client state mutates after open.
type DatasetClient struct {
	dataPath  string
	indexPath string
	index     DatasetIndex
	region    *MappedRegion // nil when total_size == 0
}

// OpenDatasetClient opens the data and index files read-only, validates the
// index, and maps the data file through the trunk MappedRegion seam. An empty
// dataset (total_size == 0) opens without a region. The caller must Close.
func OpenDatasetClient(dataPath, indexPath string) (*DatasetClient, error) {
	dfi, err := os.Stat(dataPath)
	if err != nil {
		return nil, fmt.Errorf("loadgen/mmap: stat data file: %w", err)
	}
	if _, err := os.Stat(indexPath); err != nil {
		return nil, fmt.Errorf("loadgen/mmap: stat index file: %w", err)
	}
	raw, err := os.ReadFile(indexPath)
	if err != nil {
		return nil, fmt.Errorf("loadgen/mmap: read index: %w", err)
	}
	var idx DatasetIndex
	if err := json.Unmarshal(raw, &idx); err != nil {
		return nil, fmt.Errorf("%w: %v", ErrCorruptIndex, err)
	}
	// Check the data-file size before the span check: a total_size that
	// disagrees with the file on disk is a mismatch, even though it also
	// breaks the payloads-span-total_size invariant.
	if dfi.Size() != int64(idx.TotalSize) {
		return nil, fmt.Errorf("%w: data file is %d bytes, index says %d", ErrIndexMismatch, dfi.Size(), idx.TotalSize)
	}
	if err := validateIndex(&idx); err != nil {
		return nil, err
	}
	var region *MappedRegion
	if idx.TotalSize > 0 {
		r, err := OpenMappedRegion(dataPath, false)
		if err != nil {
			return nil, fmt.Errorf("loadgen/mmap: open data region: %w", err)
		}
		region = r
	}
	return &DatasetClient{dataPath: dataPath, indexPath: indexPath, index: idx, region: region}, nil
}

// Index returns the parsed index.
func (c *DatasetClient) Index() DatasetIndex { return c.index }

// Sessions returns session ids in dataset (insertion) order.
func (c *DatasetClient) Sessions() []string { return c.index.ConversationIDs }

// DataPath returns the on-disk location of dataset.dat.
func (c *DatasetClient) DataPath() string { return c.dataPath }

// IndexPath returns the on-disk location of index.dat.
func (c *DatasetClient) IndexPath() string { return c.indexPath }

// PayloadBytes returns the pre-encoded payload for (session, turn), copied out
// of the shared read-only region verbatim — no deserialization. The returned
// slice is caller-owned and stays valid after Close. Datasets that are not in
// payload_bytes format, unknown sessions, and out-of-range turns all report
// ErrNotFound.
func (c *DatasetClient) PayloadBytes(sessionID string, turn int) ([]byte, error) {
	pos, ok := c.payloadPos(sessionID, turn)
	if !ok {
		return nil, fmt.Errorf("payload %s turn %d: %w", sessionID, turn, ErrNotFound)
	}
	b, err := c.region.Slice(int64(pos.Offset), int64(pos.Size))
	if err != nil {
		return nil, fmt.Errorf("payload %s turn %d: %w", sessionID, turn, err)
	}
	return b, nil
}

// PayloadTurn returns the payload plus its restored sidecar scalars. When the
// index recorded no max_tokens, it is recovered from the payload's own wire
// keys (max_tokens / max_completion_tokens / max_output_tokens).
func (c *DatasetClient) PayloadTurn(sessionID string, turn int) (PayloadTurnData, error) {
	pos, ok := c.payloadPos(sessionID, turn)
	if !ok {
		return PayloadTurnData{}, fmt.Errorf("payload %s turn %d: %w", sessionID, turn, ErrNotFound)
	}
	b, err := c.region.Slice(int64(pos.Offset), int64(pos.Size))
	if err != nil {
		return PayloadTurnData{}, fmt.Errorf("payload %s turn %d: %w", sessionID, turn, err)
	}
	md := PayloadTurnData{
		PayloadBytes:         b,
		MaxTokens:            pos.MaxTokens,
		TimestampMS:          pos.TimestampMS,
		SourceKind:           pos.SourceKind,
		AudioDurationSeconds: pos.AudioDurationSeconds,
		NumChoices:           pos.NumChoices,
	}
	if md.MaxTokens == nil {
		if v, ok := MaxTokensFromWirePayload(md.PayloadBytes); ok {
			md.MaxTokens = &v
		}
	}
	return md, nil
}

// payloadPos resolves (session, turn) to its index entry.
func (c *DatasetClient) payloadPos(sessionID string, turn int) (PayloadOffset, bool) {
	if c.index.Format != FormatPayloadBytes {
		return PayloadOffset{}, false
	}
	turns, ok := c.index.PayloadOffsets[sessionID]
	if !ok || turn < 0 || turn >= len(turns) {
		return PayloadOffset{}, false
	}
	return turns[turn], true
}

// Close releases the underlying mapped region. Safe to call multiple times.
func (c *DatasetClient) Close() error {
	if c.region == nil {
		return nil
	}
	return c.region.Close()
}

// validateIndex enforces the structural contracts of a well-formed index:
// unique session ids, a known format, non-negative sizes, and — for
// payload_bytes — strictly contiguous, in-bounds turn offsets spanning exactly
// total_size bytes.
func validateIndex(idx *DatasetIndex) error {
	if idx.Format != FormatConversation && idx.Format != FormatPayloadBytes {
		return fmt.Errorf("%w: unknown format %q", ErrCorruptIndex, idx.Format)
	}
	if idx.TotalSize < 0 {
		return fmt.Errorf("%w: negative total_size %d", ErrCorruptIndex, idx.TotalSize)
	}
	seen := make(map[string]bool, len(idx.ConversationIDs))
	for _, id := range idx.ConversationIDs {
		if seen[id] {
			return fmt.Errorf("%w: duplicate session id %q", ErrDuplicateID, id)
		}
		seen[id] = true
	}
	if idx.Format == FormatConversation {
		for id, o := range idx.Offsets {
			if o.Offset < 0 || o.Size < 0 || o.Offset+o.Size > idx.TotalSize {
				return fmt.Errorf("%w: out-of-bounds offset for %q", ErrCorruptIndex, id)
			}
		}
		return nil
	}
	cursor := 0
	for _, id := range idx.ConversationIDs {
		for _, po := range idx.PayloadOffsets[id] {
			if po.Offset != cursor {
				return fmt.Errorf("%w: payload for %q at offset %d, expected %d", ErrCorruptIndex, id, po.Offset, cursor)
			}
			if po.Size < 0 {
				return fmt.Errorf("%w: negative payload size for %q", ErrCorruptIndex, id)
			}
			if po.NumChoices < 1 {
				return fmt.Errorf("%w: num_choices %d for %q below 1", ErrCorruptIndex, po.NumChoices, id)
			}
			if po.MaxTokens != nil && *po.MaxTokens < 1 {
				return fmt.Errorf("%w: max_tokens %d for %q below 1", ErrCorruptIndex, *po.MaxTokens, id)
			}
			cursor += po.Size
		}
	}
	if cursor != idx.TotalSize {
		return fmt.Errorf("%w: payloads span %d bytes, total_size is %d", ErrCorruptIndex, cursor, idx.TotalSize)
	}
	return nil
}

// wireMaxTokenKeys lists the wire-body keys that encode the same output-length
// cap across endpoint dialects, in priority order (first present wins).
var wireMaxTokenKeys = []string{"max_tokens", "max_completion_tokens", "max_output_tokens"}

// MaxTokensFromWirePayload extracts a positive integer max-tokens value from a
// wire JSON body, if present. Booleans, strings, nulls, and non-integer
// numbers are ignored; the first key in wireMaxTokenKeys order carrying an
// integer >= 1 wins.
func MaxTokensFromWirePayload(payload []byte) (int, bool) {
	var obj map[string]any
	if err := json.Unmarshal(payload, &obj); err != nil {
		return 0, false
	}
	for _, key := range wireMaxTokenKeys {
		v, ok := obj[key]
		if !ok {
			continue
		}
		if _, isBool := v.(bool); isBool {
			continue
		}
		f, isNum := v.(float64)
		if !isNum || f < 1 || f != math.Trunc(f) || f > math.MaxInt32 {
			continue
		}
		return int(f), true
	}
	return 0, false
}

// ApplyMaxTokensToWirePayload returns a copy of a raw wire body with its
// output-length cap set to value. It overwrites every max-token key already
// present so the cap is honored across endpoint dialects; when none is
// present, it sets the canonical key for the body's dialect: responses bodies
// (keyed by "input") expect max_output_tokens, chat/completions bodies expect
// max_tokens.
func ApplyMaxTokensToWirePayload(payload []byte, value int) ([]byte, error) {
	var obj map[string]any
	if err := json.Unmarshal(payload, &obj); err != nil {
		return nil, fmt.Errorf("loadgen/mmap: apply max_tokens: %w", err)
	}
	present := make([]string, 0, len(wireMaxTokenKeys))
	for _, key := range wireMaxTokenKeys {
		if _, ok := obj[key]; ok {
			present = append(present, key)
		}
	}
	switch {
	case len(present) > 0:
		for _, key := range present {
			obj[key] = value
		}
	default:
		canonical := "max_tokens"
		if _, ok := obj["input"]; ok {
			canonical = "max_output_tokens"
		}
		obj[canonical] = value
	}
	out, err := json.Marshal(obj)
	if err != nil {
		return nil, fmt.Errorf("loadgen/mmap: apply max_tokens: %w", err)
	}
	return out, nil
}

// resolveTurnMaxTokens prefers the explicit scalar; when absent it falls back
// to the payload's wire keys, mirroring the upstream _resolve_turn_max_tokens.
func resolveTurnMaxTokens(meta PayloadMeta, payload []byte) (*int, error) {
	if meta.MaxTokens != nil {
		if *meta.MaxTokens < 1 {
			return nil, fmt.Errorf("max_tokens must be >= 1, got %d", *meta.MaxTokens)
		}
		return meta.MaxTokens, nil
	}
	if v, ok := MaxTokensFromWirePayload(payload); ok {
		return &v, nil
	}
	return nil, nil
}

// writeFull writes all of b to w, retrying short writes.
func writeFull(w io.Writer, b []byte) error {
	for len(b) > 0 {
		n, err := w.Write(b)
		if n > 0 {
			b = b[n:]
		}
		if err != nil {
			return err
		}
	}
	return nil
}
