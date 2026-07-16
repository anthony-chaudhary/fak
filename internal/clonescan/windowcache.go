package clonescan

// windowcache.go — the OPTIONAL, injected-from-above tokenization cache seam (#4330).
//
// The one expensive step BuildTreeIndex repeats is pure: a file's exact bytes run
// through qualifyingWindows(goTokens(src, false)) always yield the identical
// (keys, spans). On a shared trunk the SAME ~5.7k tracked files are re-lexed on every
// commit gate and every `dup guard`, even though almost none changed. WindowCache lets
// a caller memoize that pure step keyed on the exact bytes, so an unchanged file is a
// lookup instead of a re-lex.
//
// The interface is defined HERE (the consumer) but implemented ELSEWHERE (a disk-backed
// cache under <git-common-dir>/fak/token-cache, internal/tokencache): clonescan itself
// does no I/O, so it stays pure and testable, and the concrete cache is injected. The
// contract is ACCELERATE-NEVER-GATE: any miss, error, or corrupt entry returns ok=false
// and BuildTreeIndex recomputes — a cache failure degrades to exactly the uncached path.

import (
	"crypto/sha256"
	"encoding/hex"
	"sort"
	"strconv"
)

// WindowCache memoizes the pure tokenization of a file's exact bytes into its qualifying
// clone-window (keys, spans). BuildTreeIndex consults it per file when one is supplied.
//
// The keys/spans pair is the SAME (keys, spans) qualifyingWindows returns: equal length,
// keys[i] the normalized token window and spans[i] its [startLine, endLine]. An
// implementation MUST content-address `src` and tag entries with a tokenizer version
// (see TokenizerVersion) so a source change or a tokenizer change misses. Get returning
// ok=true with empty slices is a valid hit for a file that carries no qualifying window.
type WindowCache interface {
	// Get returns the memoized (keys, spans) for the exact bytes `src`, or ok=false on a
	// miss. Implementations must fail closed: an unreadable/corrupt/stale entry is a miss,
	// never a wrong hit.
	Get(src string) (keys []string, spans []span, ok bool)
	// Put memoizes the (keys, spans) computed from `src`. Best-effort; callers ignore the
	// outcome — the cache accelerates, it never gates.
	Put(src string, keys []string, spans []span)
}

// tokenizerContractTag is the MANUAL half of the tokenizer version. Bump it whenever the
// tokenizer LOGIC changes in a way the geometry/table hash below cannot see — e.g. how
// goTokens consumes a literal, comment, or stray byte — so a persisted WindowCache misses
// entries an older lexer wrote instead of serving windows that lexer would not produce.
const tokenizerContractTag = "clonescan-window/v1"

// TokenizerVersion is the cache-invalidation tag for the (keys, spans) contract. It folds
// the manual tag with the engine geometry (WindowTokens, MinLogicTokens) and the exact
// token tables — the ORDER-sensitive goOps and the four logic/keyword sets — so editing
// any of them changes the tag automatically. A content-addressed cache keyed on it can
// never serve windows a different tokenizer produced (acceptance: bumping the tag
// invalidates prior entries). Pure computation, no I/O: clonescan stays disk-free.
func TokenizerVersion() string {
	h := sha256.New()
	h.Write([]byte(tokenizerContractTag))
	h.Write([]byte("|W=" + strconv.Itoa(WindowTokens) + "|L=" + strconv.Itoa(MinLogicTokens)))
	// goOps is order-sensitive (greedy longest-first match), so hash it in declared order.
	h.Write([]byte("|ops="))
	for _, op := range goOps {
		h.Write([]byte(op))
		h.Write([]byte{0x1f})
	}
	// The remaining tables are sets; hash their sorted keys so Go map iteration order can
	// never move the tag between runs.
	for _, tbl := range []map[string]bool{goKeywords, logicKeywords, logicOps, assignOps} {
		h.Write([]byte("|set="))
		keys := make([]string, 0, len(tbl))
		for k := range tbl {
			keys = append(keys, k)
		}
		sort.Strings(keys)
		for _, k := range keys {
			h.Write([]byte(k))
			h.Write([]byte{0x1f})
		}
	}
	return hex.EncodeToString(h.Sum(nil))
}
