package hostload

import (
	"encoding/binary"
	"math"
	"runtime"
	"testing"
)

// The 2026-10-17 census: Read reported no load on Windows, so fak-flow's
// load-scaled scoped-validate budget never stretched there and starved lands
// refused as LAND_VALIDATE_TIMEOUT.
//
// fak-test:justify why=regression when=changed:platform/hostload/**
// fak-test:runtime fast est=5ms lane=default
func TestWindowsLoad(t *testing.T) {
	t.Parallel()
	cases := []struct {
		name                         string
		idle0, total0, idle1, total1 uint64
		cores                        int
		queue                        float64
		want                         float64
		ok                           bool
	}{
		{"idle", 0, 0, 1000, 1000, 8, 0, 0, true},
		{"half_busy", 0, 0, 500, 1000, 8, 0, 4, true},
		{"saturated_with_queue", 100, 1000, 100, 2000, 16, 24, 40, true},
		{"no_elapsed_time", 0, 1000, 0, 1000, 8, 0, 0, false},
		{"idle_exceeds_total", 0, 0, 2000, 1000, 8, 0, 0, false},
		{"idle_went_backwards", 10, 0, 5, 1000, 8, 0, 0, false},
		{"no_cores", 0, 0, 500, 1000, 0, 0, 0, false},
		{"negative_queue", 0, 0, 500, 1000, 8, -1, 0, false},
		{"nan_queue", 0, 0, 500, 1000, 8, math.NaN(), 0, false},
	}
	for _, c := range cases {
		got, ok := windowsLoad(c.idle0, c.total0, c.idle1, c.total1, c.cores, c.queue)
		if ok != c.ok || got != c.want {
			t.Errorf("%s: windowsLoad = %v, %t; want %v, %t", c.name, got, ok, c.want, c.ok)
		}
	}
}

// fak-test:justify why=regression when=changed:platform/hostload/**
// fak-test:runtime fast est=1s lane=default
func TestReadReportsWindowsLoad(t *testing.T) {
	if runtime.GOOS != "windows" {
		t.Skip("Windows load sample")
	}
	l1, l5, l15, ok := Read()
	checkTriple(t, "windows", l1, l5, l15, ok)
	if !ok {
		t.Fatal("Read on Windows: ok=false, want a load sample")
	}
	if l1 != l5 || l1 != l15 {
		t.Fatalf("Read on Windows = %v %v %v, want one sample in all three", l1, l5, l15)
	}
}

type perfFixtureCounter struct{ name, size, offset uint32 }

// perfFixture lays out a winperf.h PERF_DATA_BLOCK holding an unrelated object
// then the given object, whose counter block carries value at offset 8.
func perfFixture(index uint32, instances int32, counters []perfFixtureCounter, value uint32) []byte {
	le := binary.LittleEndian
	object := func(index uint32, instances int32, counters []perfFixtureCounter, block []byte) []byte {
		const hdr = 64
		def := make([]byte, 0, 40*len(counters))
		for _, c := range counters {
			d := make([]byte, 40)
			le.PutUint32(d[0:], 40)
			le.PutUint32(d[4:], c.name)
			le.PutUint32(d[32:], c.size)
			le.PutUint32(d[36:], c.offset)
			def = append(def, d...)
		}
		o := make([]byte, hdr)
		le.PutUint32(o[0:], uint32(hdr+len(def)+len(block)))
		le.PutUint32(o[4:], uint32(hdr+len(def)))
		le.PutUint32(o[8:], hdr)
		le.PutUint32(o[12:], index)
		le.PutUint32(o[32:], uint32(len(counters)))
		le.PutUint32(o[40:], uint32(instances))
		return append(append(o, def...), block...)
	}
	block := make([]byte, 16)
	le.PutUint32(block[0:], 16)
	le.PutUint32(block[8:], value)
	head := make([]byte, 88)
	copy(head, []byte{'P', 0, 'E', 0, 'R', 0, 'F', 0})
	le.PutUint32(head[24:], 88)
	le.PutUint32(head[28:], 2)
	b := append(head, object(238, -1, []perfFixtureCounter{{name: perfCounterProcessorQueueLength, size: 4, offset: 4}}, make([]byte, 16))...)
	return append(b, object(index, instances, counters, block)...)
}

// fak-test:justify why=regression when=changed:platform/hostload/**
// fak-test:runtime fast est=5ms lane=default
func TestPerfDataCounter(t *testing.T) {
	t.Parallel()
	queue := []perfFixtureCounter{{name: 10, size: 8, offset: 4}, {name: perfCounterProcessorQueueLength, size: 4, offset: 8}}
	good := perfFixture(perfObjectSystem, -1, queue, 37)
	badSig := append([]byte(nil), good...)
	badSig[0] = 'X'
	cases := []struct {
		name string
		buf  []byte
		want uint32
		ok   bool
	}{
		{"system_queue_length", good, 37, true},
		{"truncated", good[:len(good)-6], 0, false},
		{"bad_signature", badSig, 0, false},
		{"object_absent", perfFixture(99, -1, queue, 37), 0, false},
		{"multi_instance_object", perfFixture(perfObjectSystem, 2, queue, 37), 0, false},
		{"counter_absent", perfFixture(perfObjectSystem, -1, queue[:1], 37), 0, false},
		{"wide_counter", perfFixture(perfObjectSystem, -1, []perfFixtureCounter{{name: perfCounterProcessorQueueLength, size: 8, offset: 8}}, 37), 0, false},
		{"empty", nil, 0, false},
	}
	for _, c := range cases {
		got, ok := perfDataCounter(c.buf, perfObjectSystem, perfCounterProcessorQueueLength)
		if ok != c.ok || got != c.want {
			t.Errorf("%s: perfDataCounter = %d, %t; want %d, %t", c.name, got, ok, c.want, c.ok)
		}
	}
}
