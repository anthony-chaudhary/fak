package model

import (
	"errors"
	"reflect"
	"testing"
)

// fak-test:runtime fast est=1ms
func TestSpPpBoundaryRoundTrip(t *testing.T) {
	input := []float32{1, 2, 3, 4, 5, 6, 7, 8, 9, 10}
	shards, err := ShardSpPpBoundary(input, 5, 2, 3)
	want := [][]float32{{1, 2, 3, 4}, {5, 6, 7, 8}, {9, 10, 0, 0}}
	if err != nil || !reflect.DeepEqual(shards, want) {
		t.Fatalf("shards %v,%v", shards, err)
	}
	gathered, err := GatherSpPpBoundary(shards, 5, 2)
	if err != nil || !reflect.DeepEqual(gathered, input) {
		t.Fatalf("gather %v,%v", gathered, err)
	}
	reshards, err := ShardSpPpBoundary(gathered, 5, 2, 3)
	if err != nil || !reflect.DeepEqual(reshards, want) {
		t.Fatalf("reshard %v,%v", reshards, err)
	}
	shards[0][0] = 99
	if input[0] != 1 || gathered[0] != 1 || reshards[0][0] != 1 {
		t.Fatal("storage aliases across boundary")
	}
}

// fak-test:runtime fast est=1ms
func TestSpPpBoundaryEmptyAndRefusal(t *testing.T) {
	shards, err := ShardSpPpBoundary(nil, 0, 2, 3)
	if err != nil || len(shards) != 3 {
		t.Fatalf("empty shard: %v", err)
	}
	got, err := GatherSpPpBoundary(shards, 0, 2)
	if err != nil || len(got) != 0 {
		t.Fatalf("empty gather: %v", err)
	}
	for _, shape := range [][3]int{{-1, 1, 1}, {1, 0, 1}, {1, 1, 0}, {int(^uint(0) >> 1), 2, 1}, {0, 1, int(^uint(0) >> 1)}, {2, 1, 1}} {
		if out, err := ShardSpPpBoundary(nil, shape[0], shape[1], shape[2]); out != nil || !errors.Is(err, ErrSpPpBoundaryShape) {
			t.Fatalf("shape %v accepted", shape)
		}
	}
	if out, err := GatherSpPpBoundary([][]float32{{1}, {2, 3}}, 2, 1); out != nil || !errors.Is(err, ErrSpPpBoundaryShape) {
		t.Fatal("ragged shards admitted")
	}
}
