package ggufload

import (
	"testing"
)

// The P/D q4k load path (ticket pd-q4k-engram-file-closed-20261108) hands a model to a
// serving role whose attached Engram stage re-reads its packed table rows on every served
// forward. LoadModelQ4KRetained must keep the shard readers open for that whole life; the
// lifetime-closing LoadModelQ4K entry documents the failure mode this cures.

var retainedWarmPrefix = []int{0, 1, 2, 0, 2, 1, 0, 1, 3, 2}

// fak-test:runtime fast est=200ms lane=default
func TestLoadModelQ4KRetainedKeepsEngramShardHandlesOpen(t *testing.T) {
	path := writeV41UnboundEngramFixture(t, true)
	m, err := LoadModelQ4KRetained(path)
	if m != nil {
		defer m.CloseWeights()
	}
	if err != nil {
		t.Fatalf("retained load refused: %v", err)
	}
	if !m.V41EngramAttached() {
		t.Fatal("loaded model does not report an attached Engram stage")
	}
	// A post-load bounded row read rides the retained shard reader: with the closing
	// entry the same read hits closed files (the witnessed P/D front-door panic); with
	// the retained entry it serves from the still-open shards.
	delta, err := m.WarmV41EngramPrefix(retainedWarmPrefix, nil)
	if err != nil {
		t.Fatalf("post-load Engram row read through the retained shards: %v", err)
	}
	if delta.Requested <= 0 || delta.Misses == 0 || delta.BytesRead <= 0 {
		t.Fatalf("warm delta=%+v, want a real backing read through the retained handles", delta)
	}
}

// fak-test:runtime fast est=200ms lane=default
func TestLoadModelQ4KClosingEntryClosesEngramShards(t *testing.T) {
	path := writeV41UnboundEngramFixture(t, true)
	m, err := LoadModelQ4K(path)
	if m != nil {
		defer m.CloseWeights()
	}
	if err != nil {
		t.Fatalf("closing load refused: %v", err)
	}
	if !m.V41EngramAttached() {
		t.Fatal("loaded model does not report an attached Engram stage")
	}
	// The closing entry retired the shard readers on return: the same post-load read
	// must fail closed rather than serve stale bytes.
	if _, err := m.WarmV41EngramPrefix(retainedWarmPrefix, nil); err == nil {
		t.Fatal("post-load Engram row read succeeded through closed shards")
	}
}
