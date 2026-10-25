package compute

import (
	"errors"
	"math"
	"testing"
)

// fak-test:runtime fast est=1ms lane=default
func TestSelectHandoffRoute(t *testing.T) {
	yes := func(int, int) (bool, error) { return true, nil }
	no := func(int, int) (bool, error) { return false, nil }
	boom := func(int, int) (bool, error) { return false, errors.New("probe") }
	cases := []struct {
		name     string
		src, dst int
		probe    PeerAccessProbe
		force    bool
		want     HandoffRoute
		wantErr  error
	}{
		{"same device", 1, 1, boom, false, HandoffSameDevice, nil},
		{"peer", 0, 1, yes, false, HandoffPeer, nil},
		{"no peer", 0, 2, no, false, HandoffHostBounce, nil},
		{"forced bounce", 0, 1, yes, true, HandoffHostBounce, nil},
		{"no probe", 0, 1, nil, false, HandoffHostBounce, nil},
		{"probe error", 0, 1, boom, false, HandoffRouteUnknown, ErrHandoffFailed},
		{"negative ordinal", -1, 1, yes, false, HandoffRouteUnknown, ErrHandoffInvalid},
	}
	for _, tc := range cases {
		got, err := SelectHandoffRoute(tc.src, tc.dst, tc.probe, tc.force)
		if got != tc.want || !errors.Is(err, tc.wantErr) || (tc.wantErr == nil) != (err == nil) {
			t.Errorf("%s: route %s err %v, want %s err %v", tc.name, got, err, tc.want, tc.wantErr)
		}
	}
}

// fak-test:runtime fast est=1ms lane=default
func TestHostBounceChunksCoverExactly(t *testing.T) {
	for _, tc := range []struct{ n, staging, chunks int }{{0, 8, 0}, {8, 8, 1}, {9, 8, 2}, {100, 7, 15}} {
		got, err := HostBounceChunks(tc.n, tc.staging)
		if err != nil || len(got) != tc.chunks {
			t.Fatalf("n=%d staging=%d: %d chunks err %v, want %d", tc.n, tc.staging, len(got), err, tc.chunks)
		}
		next := 0
		for _, c := range got {
			if c.Off != next || c.Len <= 0 || c.Len > tc.staging {
				t.Fatalf("n=%d staging=%d: bad chunk %+v", tc.n, tc.staging, c)
			}
			next += c.Len
		}
		if next != tc.n {
			t.Fatalf("n=%d staging=%d: chunks cover %d bytes", tc.n, tc.staging, next)
		}
	}
	if _, err := HostBounceChunks(4, 0); !errors.Is(err, ErrHandoffInvalid) {
		t.Fatalf("zero staging err = %v, want ErrHandoffInvalid", err)
	}
}

// fak-test:runtime fast est=1ms lane=default
func TestHostHopPreservesEveryBit(t *testing.T) {
	src := []float32{0, float32(math.Copysign(0, -1)), 1.5, -3.25e-30, float32(math.Inf(1)), math.Float32frombits(0x7fc01234), math.Float32frombits(0x00000001)}
	for i := 0; i < 37; i++ {
		src = append(src, float32(i)*0.37-5)
	}
	for _, hop := range []HostHop{{}, {ForceBounce: true, StagingBytes: 12}, {ForceBounce: true}} {
		got, rec, err := hop.HopF32(src, 0, 1)
		if err != nil {
			t.Fatal(err)
		}
		for i := range src {
			if math.Float32bits(got[i]) != math.Float32bits(src[i]) {
				t.Fatalf("route %s: word %d = %#x, want %#x", rec.Route, i, math.Float32bits(got[i]), math.Float32bits(src[i]))
			}
		}
		wantChunks := 1
		if hop.StagingBytes == 12 {
			wantChunks = (len(src)*4 + 11) / 12
		}
		if rec.Route != HandoffHostBounce || rec.Bytes != int64(len(src)*4) || rec.Chunks != wantChunks {
			t.Fatalf("receipt route %s bytes %d chunks %d, want host-bounce %d %d", rec.Route, rec.Bytes, rec.Chunks, len(src)*4, wantChunks)
		}
		got[0] = 42
		if src[0] != 0 {
			t.Fatal("hop aliased its input")
		}
	}
}
