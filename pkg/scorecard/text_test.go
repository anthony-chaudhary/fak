package scorecard

import "testing"

func TestValueTextPreservesScorecardWireForms(t *testing.T) {
	tests := []struct {
		name string
		in   any
		want string
	}{
		{name: "string stays unquoted", in: "ready", want: "ready"},
		{name: "positive int", in: 42, want: "42"},
		{name: "negative int", in: -7, want: "-7"},
		{name: "boolean uses JSON", in: true, want: "true"},
		{name: "nil uses JSON", in: nil, want: "null"},
		{name: "object uses JSON", in: map[string]int{"score": 9}, want: `{"score":9}`},
		{name: "marshal failure stays empty", in: make(chan int), want: ""},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := ValueText(tt.in); got != tt.want {
				t.Fatalf("ValueText(%#v) = %q, want %q", tt.in, got, tt.want)
			}
		})
	}
}

func TestMetricTextPreservesScorecardNumericWireForms(t *testing.T) {
	tests := []struct {
		name string
		in   any
		want string
	}{
		{name: "string", in: "A", want: "A"},
		{name: "int", in: -3, want: "-3"},
		{name: "float shortest", in: 0.125, want: "0.125"},
		{name: "int64 fallback", in: int64(8), want: "8"},
		{name: "unsupported fallback", in: true, want: "0"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := MetricText(tt.in); got != tt.want {
				t.Fatalf("MetricText(%#v) = %q, want %q", tt.in, got, tt.want)
			}
		})
	}
	if got := ScoreValueText(87); got != "0.87" {
		t.Fatalf("ScoreValueText(87) = %q, want 0.87", got)
	}
}

func TestCountNounPreservesCompactPluralWireForm(t *testing.T) {
	for _, tc := range []struct {
		n    int
		want string
	}{
		{0, "0 defect(s)"}, {1, "1 defect"}, {2, "2 defect(s)"}, {-1, "-1 defect(s)"},
	} {
		if got := CountNoun(tc.n, "defect"); got != tc.want {
			t.Fatalf("CountNoun(%d) = %q, want %q", tc.n, got, tc.want)
		}
	}
}

func TestTrueAcceptsOnlyBooleanTrue(t *testing.T) {
	for _, tc := range []struct {
		in   any
		want bool
	}{
		{true, true}, {false, false}, {1, false}, {"true", false}, {nil, false},
	} {
		if got := True(tc.in); got != tc.want {
			t.Fatalf("True(%#v) = %v, want %v", tc.in, got, tc.want)
		}
	}
}
