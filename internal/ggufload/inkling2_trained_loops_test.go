package ggufload

import (
	"bytes"
	"testing"
)

// fak-test:runtime fast est=2ms lane=default
func TestInkling2TrainedLoopsReadFromLoopCount(t *testing.T) {
	const arch = "inkling2_loop"
	data := buildInk2sGGUF(ink2sOpts{arch: arch, floor: 8192, loopKeys: true}, ink2sTensors())
	f, err := ReadSize(bytes.NewReader(data), int64(len(data)))
	if err != nil {
		t.Fatalf("read loop GGUF: %v", err)
	}
	b := &inkling2Binder{f: f, arch: arch, prefix: arch + ".", tensors: map[string]TensorInfo{}, shardOf: map[string]int{}}
	hp, err := b.readHParams(arch)
	if err != nil {
		t.Fatalf("readHParams with loop.count=3: %v", err)
	}
	if hp.Passes != 3 {
		t.Fatalf("loop.count=3 read as Passes = %d, want 3", hp.Passes)
	}
}
