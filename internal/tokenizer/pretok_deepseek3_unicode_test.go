package tokenizer

import (
	"reflect"
	"runtime"
	"testing"
	"unicode"
)

// Source-derived expectations from vcruz305/llama.cpp at
// 5210c7c5ed61dddaee6ed476623abf4b63093d16, src/unicode-data.cpp.
// MIT; Copyright (c) 2023-2026 The ggml authors.
// These tests do not claim execution of the C++ reference binary.
func TestDeepSeek3UnicodeVersion(t *testing.T) {
	if unicode.Version != "15.0.0" {
		t.Fatalf("Go Unicode tables changed to %s; re-audit the pinned llama.cpp categories before changing this guard", unicode.Version)
	}
	t.Logf("runtime=%s unicode=%s", runtime.Version(), unicode.Version)
}

func TestDeepSeek3MinimalUnicodeWitness(t *testing.T) {
	for _, tc := range []struct {
		name string
		in   string
		want []string
	}{
		{"extension_i_letter", "a\U0002EBF0b", []string{"a\U0002EBF0b"}},
		{"ideographic_symbol", "\u2ffcA", []string{"\u2ffc", "A"}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			got := preTokenizeDeepSeek3(tc.in)
			t.Logf("input=%+q got=%+q want=%+q (source-derived boundaries)", tc.in, got, tc.want)
			if !reflect.DeepEqual(got, tc.want) {
				t.Errorf("split mismatch")
			}
		})
	}
}

func TestDeepSeek3SupplementBoundaries(t *testing.T) {
	for _, tc := range []struct {
		r          rune
		letter, ps bool
	}{
		{0x2EBEF, false, false}, {0x2EBF0, true, false},
		{0x2EE5D, true, false}, {0x2EE5E, false, false},
		{0x2FFB, false, true}, {0x2FFC, false, true},
		{0x2FFD, false, true}, {0x2FFE, false, true},
		{0x2FFF, false, true}, {0x3000, false, false},
		{0x31EE, false, false}, {0x31EF, false, true},
		{0x31F0, true, false}, {-1, false, false},
		{unicode.MaxRune + 1, false, false},
	} {
		if l, ps := ds3Letter(tc.r), ds3PunctOrSymbol(tc.r); l != tc.letter || ps != tc.ps {
			t.Errorf("U+%04X letter=%t punct/symbol=%t; want %t %t", tc.r, l, ps, tc.letter, tc.ps)
		}
	}
}

func TestDeepSeek3EverySupplementSplit(t *testing.T) {
	mismatches := 0
	check := func(in string, want []string) {
		if got := preTokenizeDeepSeek3(in); !reflect.DeepEqual(got, want) {
			if mismatches < 3 {
				t.Logf("input=%+q got=%+q want=%+q", in, got, want)
			}
			mismatches++
		}
	}
	for r := rune(0x2EBF0); r <= 0x2EE5D; r++ {
		in := "a" + string(r) + "b"
		check(in, []string{in})
	}
	for _, r := range []rune{0x2FFC, 0x2FFD, 0x2FFE, 0x2FFF, 0x31EF} {
		check(string(r)+"A", []string{string(r), "A"})
	}
	if mismatches != 0 {
		t.Errorf("source-derived split mismatches=%d of 627 supplemental codepoints", mismatches)
	} else {
		t.Log("all 627 supplemental codepoint split probes passed")
	}
}
