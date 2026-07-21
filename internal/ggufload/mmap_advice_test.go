package ggufload

import "testing"

func TestChooseMmapAdvice(t *testing.T) {
	tests := []struct {
		name   string
		topo   NumaTopology
		access AccessPattern
		want   MmapAdvice
	}{
		{
			name:   "numa multinode weight stream -> random (first-touch locality)",
			topo:   NumaTopology{NodeCount: 4, IsNuma: true},
			access: AccessWeightStream,
			want:   AdviceRandom,
		},
		{
			name:   "single node weight stream -> sequential (fast bulk load)",
			topo:   NumaTopology{NodeCount: 1, IsNuma: false},
			access: AccessWeightStream,
			want:   AdviceSequential,
		},
		{
			name:   "non-numa multi-socket-labelled but IsNuma false weight stream -> sequential",
			topo:   NumaTopology{NodeCount: 2, IsNuma: false},
			access: AccessWeightStream,
			want:   AdviceSequential,
		},
		{
			name:   "explicit interleave intent forces random even with IsNuma false",
			topo:   NumaTopology{NodeCount: 2, IsNuma: false, Interleave: true},
			access: AccessWeightStream,
			want:   AdviceRandom,
		},
		{
			name:   "expert random access overrides single-node topology -> random",
			topo:   NumaTopology{NodeCount: 1, IsNuma: false},
			access: AccessExpertRandom,
			want:   AdviceRandom,
		},
		{
			name:   "expert random access on numa -> random",
			topo:   NumaTopology{NodeCount: 8, IsNuma: true},
			access: AccessExpertRandom,
			want:   AdviceRandom,
		},
		{
			name:   "unknown topology (zero nodes) fails closed to normal",
			topo:   NumaTopology{NodeCount: 0},
			access: AccessWeightStream,
			want:   AdviceNormal,
		},
		{
			name:   "unknown topology fails closed even for random access",
			topo:   NumaTopology{NodeCount: 0, Interleave: true},
			access: AccessExpertRandom,
			want:   AdviceNormal,
		},
		{
			name:   "negative node count fails closed to normal",
			topo:   NumaTopology{NodeCount: -1, IsNuma: true},
			access: AccessWeightStream,
			want:   AdviceNormal,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := ChooseMmapAdvice(tt.topo, tt.access); got != tt.want {
				t.Fatalf("ChooseMmapAdvice(%+v, %v) = %v, want %v", tt.topo, tt.access, got, tt.want)
			}
		})
	}
}

func TestMmapAdviceString(t *testing.T) {
	cases := map[MmapAdvice]string{
		AdviceNormal:     "normal",
		AdviceRandom:     "random",
		AdviceSequential: "sequential",
		MmapAdvice(200):  "normal", // out-of-range falls back to the safe default token
	}
	for adv, want := range cases {
		if got := adv.String(); got != want {
			t.Fatalf("MmapAdvice(%d).String() = %q, want %q", adv, got, want)
		}
	}
}

func TestMmapAdviceZeroValueIsNormal(t *testing.T) {
	var zero MmapAdvice
	if zero != AdviceNormal {
		t.Fatalf("zero MmapAdvice = %v, want AdviceNormal", zero)
	}
}
