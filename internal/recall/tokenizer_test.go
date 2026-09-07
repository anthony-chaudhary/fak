package recall

import (
	"reflect"
	"strings"
	"sync"
	"testing"
)

func TestTokenize_MixedIdentifiers(t *testing.T) {
	tests := []struct {
		input string
		want  []string
	}{
		{
			input: "L0录入",
			want:  []string{"l0", "录入"},
		},
		{
			input: "v2版本",
			want:  []string{"v2", "版本"},
		},
		{
			input: "录入L0",
			want:  []string{"录入", "l0"},
		},
		{
			input: "API接口v3",
			want:  []string{"api", "接口", "v3"},
		},
		{
			input: "build2026年",
			want:  []string{"build2026", "年"},
		},
	}

	for _, tt := range tests {
		got := Tokenize(tt.input)
		if !reflect.DeepEqual(got, tt.want) {
			t.Errorf("Tokenize(%q) = %v; want %v", tt.input, got, tt.want)
		}
	}
}

func TestTokenize_CJKBigrams(t *testing.T) {
	tests := []struct {
		name  string
		input string
		want  []string
	}{
		{
			name:  "chinese phrase",
			input: "分布式缓存",
			want:  []string{"分布式缓存", "分布", "布式", "式缓", "缓存"},
		},
		{
			name:  "japanese phrase with kanji and kana",
			input: "日本語テスト",
			want:  []string{"日本語テスト", "日本", "本語", "語テ", "テス", "スト"},
		},
		{
			name:  "korean hangul phrase",
			input: "데이터베이스",
			want:  []string{"데이터베이스", "데이", "이터", "터베", "베이", "이스"},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := Tokenize(tt.input)
			if !reflect.DeepEqual(got, tt.want) {
				t.Errorf("Tokenize(%q) = %v; want %v", tt.input, got, tt.want)
			}
		})
	}
}

func TestTokenize_Boundaries(t *testing.T) {
	t.Run("punctuation adjacency", func(t *testing.T) {
		tests := []struct {
			input string
			want  []string
		}{
			{
				input: "L0,录入",
				want:  []string{"l0", "录入"},
			},
			{
				input: "L0-录入",
				want:  []string{"l0", "录入"},
			},
			{
				input: "v1.0-版本",
				want:  []string{"v1", "0", "版本"},
			},
			{
				input: "分布式。缓存、系统！",
				want:  []string{"分布式", "分布", "布式", "缓存", "系统"},
			},
		}
		for _, tt := range tests {
			got := Tokenize(tt.input)
			if !reflect.DeepEqual(got, tt.want) {
				t.Errorf("Tokenize(%q) = %v; want %v", tt.input, got, tt.want)
			}
		}
	})

	t.Run("emoji adjacency", func(t *testing.T) {
		tests := []struct {
			input string
			want  []string
		}{
			{
				input: "🚀L0录入🎉",
				want:  []string{"l0", "录入"},
			},
			{
				input: "🔥分布式缓存💡",
				want:  []string{"分布式缓存", "分布", "布式", "式缓", "缓存"},
			},
			{
				input: "👋Hello, 🌏World!",
				want:  []string{"hello", "world"},
			},
		}
		for _, tt := range tests {
			got := Tokenize(tt.input)
			if !reflect.DeepEqual(got, tt.want) {
				t.Errorf("Tokenize(%q) = %v; want %v", tt.input, got, tt.want)
			}
		}
	})

	t.Run("short 1-character CJK runes", func(t *testing.T) {
		tests := []struct {
			input string
			want  []string
		}{
			{
				input: "我 and 你",
				want:  []string{"我", "and", "你"},
			},
			{
				input: "A文B",
				want:  []string{"a", "文", "b"},
			},
			{
				input: "文",
				want:  []string{"文"},
			},
			{
				input: "a b c",
				want:  []string{"a", "b", "c"},
			},
		}
		for _, tt := range tests {
			got := Tokenize(tt.input)
			if !reflect.DeepEqual(got, tt.want) {
				t.Errorf("Tokenize(%q) = %v; want %v", tt.input, got, tt.want)
			}
		}
	})

	t.Run("english text unchanged", func(t *testing.T) {
		tests := []struct {
			input string
			want  []string
		}{
			{
				input: "The quick brown FOX jumps over the lazy dog",
				want:  []string{"the", "quick", "brown", "fox", "jumps", "over", "lazy", "dog"},
			},
			{
				input: "gateway_retry_budget",
				want:  []string{"gateway", "retry", "budget"},
			},
			{
				input: "refund-fee 25 EUR",
				want:  []string{"refund", "fee", "25", "eur"},
			},
		}
		for _, tt := range tests {
			got := Tokenize(tt.input)
			if !reflect.DeepEqual(got, tt.want) {
				t.Errorf("Tokenize(%q) = %v; want %v", tt.input, got, tt.want)
			}
		}
	})
}

func TestTokenize_DeduplicationAndCapSafety(t *testing.T) {
	t.Run("deduplication", func(t *testing.T) {
		input := "test test TEST 录入 录入 人人人人"
		got := Tokenize(input)
		want := []string{"test", "录入", "人人人人", "人人"}
		if !reflect.DeepEqual(got, want) {
			t.Errorf("Tokenize(%q) = %v; want %v", input, got, want)
		}
	})

	t.Run("token cap safety on long CJK texts", func(t *testing.T) {
		// 40 runes > maxTokenLen (32): full run omitted, bigrams emitted
		runes40 := []rune("一二三四五六七八九十一二三四五六七八九十一二三四五六七八九十一二三四五六七八九十")
		if len(runes40) != 40 {
			t.Fatalf("expected 40 runes, got %d", len(runes40))
		}
		got := Tokenize(string(runes40))

		// Full run must not be present
		for _, token := range got {
			if token == string(runes40) {
				t.Errorf("full run of length 40 should not be emitted as token")
			}
		}

		// Bigrams should be bounded
		if len(got) > maxBigramsPerRun {
			t.Errorf("expected <= %d bigrams, got %d", maxBigramsPerRun, len(got))
		}

		// Very long 100-character diverse CJK text
		var sb strings.Builder
		for i := 0; i < 100; i++ {
			sb.WriteRune(rune(0x4e00 + i))
		}
		got100 := Tokenize(sb.String())
		if len(got100) > maxBigramsPerRun {
			t.Errorf("expected <= %d bigrams for 100-char run, got %d", maxBigramsPerRun, len(got100))
		}

		// Degenerate repeating 10,000 runes: must terminate and not blow up
		repeating := strings.Repeat("人", 10000)
		gotRepeat := Tokenize(repeating)
		if len(gotRepeat) > maxBigramsPerRun+1 {
			t.Errorf("expected <= %d tokens for degenerate input, got %d", maxBigramsPerRun+1, len(gotRepeat))
		}
	})
}

func TestTokenize_Concurrency(t *testing.T) {
	inputs := []string{
		"L0录入",
		"分布式缓存",
		"v2版本发布",
		"The quick brown fox",
		"🚀L0录入🎉",
		"日本語テスト",
		"데이터베이스",
	}

	var wg sync.WaitGroup
	const workers = 50
	const iterations = 50

	for i := 0; i < workers; i++ {
		wg.Add(1)
		go func(workerID int) {
			defer wg.Done()
			for j := 0; j < iterations; j++ {
				idx := (workerID + j) % len(inputs)
				toks := Tokenize(inputs[idx])
				if len(toks) == 0 {
					t.Errorf("unexpected empty tokens for input: %s", inputs[idx])
				}
			}
		}(i)
	}
	wg.Wait()
}

func TestIsCJK(t *testing.T) {
	tests := []struct {
		r    rune
		want bool
	}{
		{'录', true},
		{'分', true},
		{'文', true},
		{'\u3400', true}, // CJK Ext A start
		{'\u4dbf', true}, // CJK Ext A end
		{'あ', true},      // Hiragana
		{'ん', true},      // Hiragana
		{'ア', true},      // Katakana
		{'ン', true},      // Katakana
		{'ー', true},      // Katakana prolonged sound mark
		{'한', true},      // Hangul
		{'글', true},      // Hangul
		{'ㄅ', true},      // Bopomofo
		{'A', false},
		{'z', false},
		{'0', false},
		{'9', false},
		{' ', false},
		{',', false},
		{'.', false},
		{'。', false}, // CJK punctuation full stop
		{'、', false}, // CJK punctuation comma
		{'！', false}, // full-width exclamation
		{'🚀', false},
		{'🎉', false},
	}

	for _, tt := range tests {
		got := isCJK(tt.r)
		if got != tt.want {
			t.Errorf("isCJK(%q [%U]) = %v; want %v", tt.r, tt.r, got, tt.want)
		}
	}
}
