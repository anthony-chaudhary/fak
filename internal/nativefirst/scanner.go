package nativefirst

import "strings"

const Guidance = "llama.cpp may be selected only explicitly for benchmark, parity/reference diagnosis, interop/migration, study, or borrowing; native/performance paths must remain fak-native"

type Finding struct {
	Phrase string
	Reason string
}

func ScanLine(raw string) *Finding {
	phrase := strings.TrimSpace(raw)
	text := strings.ToLower(phrase)
	if !mentionsExternalLlama(text) || isWhitelistedReferenceUse(text) {
		return nil
	}
	native := containsAny(text, "native", "performance", "qwen3.8", "qwen38")
	substitute := containsAny(text, "default", "fallback", "fall back", "falls back", "auto", "delegate", "backend")
	if !native || !substitute {
		return nil
	}
	return &Finding{Phrase: phrase, Reason: Guidance}
}

func mentionsExternalLlama(s string) bool {
	return containsAny(s, "llama.cpp", "llama cpp", "llamacpp", "llama-server")
}

func isWhitelistedReferenceUse(s string) bool {
	return containsAny(s, "benchmark", "comparison", "compare", "reference", "parity", "diagnos", "interop", "migration", "borrow", "study", "explicit")
}

func containsAny(s string, needles ...string) bool {
	for _, needle := range needles {
		if strings.Contains(s, needle) {
			return true
		}
	}
	return false
}
