package tokenizer

import (
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"unicode"
	"unicode/utf8"
)

// ErrEncodeUnsupported marks tokenizer.json shapes outside this leaf's current gate.
var ErrEncodeUnsupported = errors.New("tokenizer: encode unsupported for this tokenizer")

// Encoder is the tokenizer leaf contract. The model proof path remains token-id-in;
// this interface is for offline text/id conversion and kvmmu span precision.
type Encoder interface {
	Encode(text string) ([]int, error)
	Decode(ids []int) (string, error)
	Vocab() int
	SpecialTokens() map[int]string
}

// Tokenizer encodes and decodes a HuggingFace fast tokenizer.json with a BPE model,
// tokenizer.json merge ranks, added special tokens, and a ByteLevel decoder. The
// pre-tokenizer is chosen from tokenizer.json's pre_tokenizer field: an explicit
// Split(Regex) (Qwen2.5/Qwen3.6 family) vs the ByteLevel/Digits default
// (GPT-2/SmolLM2) — they differ on whether a non-space, non-alnum char attaches to
// a following letter run ("(n", "_case"), so the wrong one mis-tokenizes one family.
type Tokenizer struct {
	idToToken        []string
	tokenToID        map[string]int
	special          map[int]string
	specialByContent []specialToken
	mergeRank        map[tokenPair]int
	split            func(string) []string
	// metaspace selects SentencePiece-style tokenization (Gemma 4): the vocab/merges
	// use ▁ (U+2581) for a space and store literal UTF-8 rather than the GPT-2 byte-level
	// alphabet, so encode replaces spaces with ▁ (no byte-level remap) and decode maps ▁
	// back to a space. false (default) keeps the GPT-2 ByteLevel path.
	metaspace bool
}

// metaspaceRune is SentencePiece's visible space marker U+2581 (▁).
const metaspaceRune = '▁'

// metaspaceEncode replaces ASCII spaces with the ▁ marker, leaving all other bytes as
// literal UTF-8 (Gemma 4's BPE operates on this form, not the GPT-2 byte-level alphabet).
func metaspaceEncode(s string) string {
	return strings.ReplaceAll(s, " ", string(metaspaceRune))
}

// byteFallback decodes a SentencePiece byte-fallback token of the form "<0xNN>" into the
// raw byte it represents. ok is false for any other token.
func byteFallback(piece string) (byte, bool) {
	if len(piece) != 6 || piece[0] != '<' || piece[1] != '0' || piece[2] != 'x' || piece[5] != '>' {
		return 0, false
	}
	hexVal := func(c byte) (int, bool) {
		switch {
		case c >= '0' && c <= '9':
			return int(c - '0'), true
		case c >= 'a' && c <= 'f':
			return int(c-'a') + 10, true
		case c >= 'A' && c <= 'F':
			return int(c-'A') + 10, true
		}
		return 0, false
	}
	hi, ok1 := hexVal(piece[3])
	lo, ok2 := hexVal(piece[4])
	if !ok1 || !ok2 {
		return 0, false
	}
	return byte(hi*16 + lo), true
}

type tokenizerJSON struct {
	Model struct {
		Type   string         `json:"type"`
		Vocab  map[string]int `json:"vocab"`
		Merges []string       `json:"merges"`
	} `json:"model"`
	Decoder struct {
		Type string `json:"type"`
	} `json:"decoder"`
	PreTokenizer preTokConfig     `json:"pre_tokenizer"`
	AddedTokens  []addedTokenJSON `json:"added_tokens"`
}

// preTokConfig captures only the pre_tokenizer shape we dispatch on.
type preTokConfig struct {
	Type          string         `json:"type"`
	Pretokenizers []preTokConfig `json:"pretokenizers"`
}

// hasSplit reports whether the pre_tokenizer (possibly a Sequence) contains an
// explicit Split stage — the marker of the Qwen/Llama explicit-regex family.
func (c preTokConfig) hasSplit() bool {
	if c.Type == "Split" {
		return true
	}
	for _, p := range c.Pretokenizers {
		if p.hasSplit() {
			return true
		}
	}
	return false
}

type addedTokenJSON struct {
	ID      int    `json:"id"`
	Content string `json:"content"`
	Special bool   `json:"special"`
}

type specialToken struct {
	content string
	id      int
}

type tokenPair struct {
	left  string
	right string
}

// LoadJSON loads the subset of HF fast tokenizer.json needed for text/id
// conversion: BPE vocab/merge ranks, added tokens, and a ByteLevel decoder.
func LoadJSON(path string) (*Tokenizer, error) {
	clean := filepath.Clean(path)
	b, err := os.ReadFile(clean)
	if err != nil {
		return nil, err
	}
	return ParseJSON(b)
}

// ParseJSON parses a HuggingFace fast tokenizer.json payload.
func ParseJSON(b []byte) (*Tokenizer, error) {
	var doc tokenizerJSON
	if err := json.Unmarshal(b, &doc); err != nil {
		return nil, err
	}
	if doc.Model.Type != "BPE" {
		return nil, fmt.Errorf("tokenizer: unsupported model type %q", doc.Model.Type)
	}
	if doc.Decoder.Type != "ByteLevel" {
		return nil, fmt.Errorf("tokenizer: unsupported decoder type %q", doc.Decoder.Type)
	}
	maxID := -1
	for tok, id := range doc.Model.Vocab {
		if tok == "" {
			return nil, fmt.Errorf("tokenizer: empty token at id %d", id)
		}
		if id < 0 {
			return nil, fmt.Errorf("tokenizer: negative id %d for token %q", id, tok)
		}
		if id > maxID {
			maxID = id
		}
	}
	for _, tok := range doc.AddedTokens {
		if tok.ID < 0 {
			return nil, fmt.Errorf("tokenizer: negative added token id %d", tok.ID)
		}
		if tok.ID > maxID {
			maxID = tok.ID
		}
	}
	if maxID < 0 {
		return nil, errors.New("tokenizer: empty vocab")
	}

	idToToken := make([]string, maxID+1)
	tokenToID := make(map[string]int, len(doc.Model.Vocab)+len(doc.AddedTokens))
	for tok, id := range doc.Model.Vocab {
		if idToToken[id] != "" {
			return nil, fmt.Errorf("tokenizer: duplicate id %d", id)
		}
		idToToken[id] = tok
		tokenToID[tok] = id
	}
	special := make(map[int]string)
	specialByContent := make([]specialToken, 0, len(doc.AddedTokens))
	for _, tok := range doc.AddedTokens {
		if tok.Content == "" {
			return nil, fmt.Errorf("tokenizer: empty added token at id %d", tok.ID)
		}
		if existing := idToToken[tok.ID]; existing != "" && existing != tok.Content {
			return nil, fmt.Errorf("tokenizer: added token id %d has content %q, vocab has %q", tok.ID, tok.Content, existing)
		}
		idToToken[tok.ID] = tok.Content
		tokenToID[tok.Content] = tok.ID
		if tok.Special {
			special[tok.ID] = tok.Content
			specialByContent = append(specialByContent, specialToken{content: tok.Content, id: tok.ID})
		}
	}
	sort.Slice(specialByContent, func(i, j int) bool {
		return len(specialByContent[i].content) > len(specialByContent[j].content)
	})

	mergeRank := make(map[tokenPair]int, len(doc.Model.Merges))
	for rank, merge := range doc.Model.Merges {
		fields := strings.Split(merge, " ")
		if len(fields) != 2 || fields[0] == "" || fields[1] == "" {
			return nil, fmt.Errorf("tokenizer: unsupported merge %q", merge)
		}
		pair := tokenPair{left: fields[0], right: fields[1]}
		if _, exists := mergeRank[pair]; exists {
			return nil, fmt.Errorf("tokenizer: duplicate merge %q", merge)
		}
		mergeRank[pair] = rank
	}

	// Dispatch the pre-tokenizer: explicit Split(Regex) => Qwen scanner; otherwise
	// the GPT-2 ByteLevel/Digits default (SmolLM2 and the bare-BPE fixtures).
	split := preTokenizeByteLevel
	if doc.PreTokenizer.hasSplit() {
		split = preTokenizeQwen
	}

	return &Tokenizer{
		idToToken:        idToToken,
		tokenToID:        tokenToID,
		special:          special,
		specialByContent: specialByContent,
		mergeRank:        mergeRank,
		split:            split,
	}, nil
}

// Encode applies the HF fast ByteLevel-BPE path this leaf supports: added special
// tokens are matched literally, normal text is split by the model's pre-tokenizer,
// then merged by tokenizer.json ranks.
func (t *Tokenizer) Encode(text string) ([]int, error) {
	if t.mergeRank == nil {
		return nil, ErrEncodeUnsupported
	}
	split := t.split
	if split == nil {
		split = preTokenizeByteLevel
	}
	var ids []int
	for len(text) > 0 {
		if sp, ok := t.matchSpecial(text); ok {
			ids = append(ids, sp.id)
			text = text[len(sp.content):]
			continue
		}
		nextSpecial := len(text)
		for _, sp := range t.specialByContent {
			if i := strings.Index(text, sp.content); i >= 0 && i < nextSpecial {
				nextSpecial = i
			}
		}
		chunk := text[:nextSpecial]
		text = text[nextSpecial:]
		// Metaspace (Gemma 4) BPEs the whole chunk on the ▁-marked literal text; the
		// GPT-2 ByteLevel path splits into regex pieces and remaps bytes to its alphabet.
		pieces := []string{chunk}
		if !t.metaspace {
			pieces = split(chunk)
		}
		for _, piece := range pieces {
			encoded := byteLevelEncode(piece)
			if t.metaspace {
				encoded = metaspaceEncode(piece)
			}
			syms, err := t.bpe(encoded)
			if err != nil {
				return nil, err
			}
			for _, sym := range syms {
				id, ok := t.tokenToID[sym]
				if !ok {
					return nil, fmt.Errorf("tokenizer: token %q not in vocab", sym)
				}
				ids = append(ids, id)
			}
		}
	}
	return ids, nil
}

// Decode converts ids to text using the ByteLevel byte-to-unicode inverse mapping.
func (t *Tokenizer) Decode(ids []int) (string, error) {
	var out []byte
	var text string
	flush := func() {
		if len(out) == 0 {
			return
		}
		text += string(out)
		out = out[:0]
	}
	for _, id := range ids {
		decoded, err := t.decodeToken(id)
		if err != nil {
			return "", err
		}
		if decoded.isSpecial {
			flush()
			text += decoded.special
			continue
		}
		out = append(out, decoded.bytes...)
	}
	flush()
	return text, nil
}

// Vocab returns the number of id slots in the tokenizer.
func (t *Tokenizer) Vocab() int {
	return len(t.idToToken)
}

// SpecialTokens returns a copy of the special-token id->content table.
func (t *Tokenizer) SpecialTokens() map[int]string {
	cp := make(map[int]string, len(t.special))
	for id, tok := range t.special {
		cp[id] = tok
	}
	return cp
}

var byteLevelDecode = makeByteLevelDecode()

func makeByteLevelDecode() map[rune]byte {
	bs := make([]byte, 0, 256)
	for b := int('!'); b <= int('~'); b++ {
		bs = append(bs, byte(b))
	}
	for b := 0xA1; b <= 0xAC; b++ {
		bs = append(bs, byte(b))
	}
	for b := 0xAE; b <= 0xFF; b++ {
		bs = append(bs, byte(b))
	}
	seen := make(map[byte]bool, 256)
	for _, b := range bs {
		seen[b] = true
	}
	out := make(map[rune]byte, 256)
	for _, b := range bs {
		out[rune(b)] = b
	}
	n := 0
	for b := 0; b < 256; b++ {
		bb := byte(b)
		if seen[bb] {
			continue
		}
		out[rune(256+n)] = bb
		n++
	}
	return out
}

var byteLevelEncodeRune = makeByteLevelEncode()

func makeByteLevelEncode() map[byte]rune {
	out := make(map[byte]rune, len(byteLevelDecode))
	for r, b := range byteLevelDecode {
		out[b] = r
	}
	return out
}

func (t *Tokenizer) matchSpecial(text string) (specialToken, bool) {
	for _, sp := range t.specialByContent {
		if strings.HasPrefix(text, sp.content) {
			return sp, true
		}
	}
	return specialToken{}, false
}

func byteLevelEncode(s string) string {
	var b strings.Builder
	for _, bb := range []byte(s) {
		b.WriteRune(byteLevelEncodeRune[bb])
	}
	return b.String()
}

func (t *Tokenizer) bpe(encoded string) ([]string, error) {
	var syms []string
	for _, r := range encoded {
		syms = append(syms, string(r))
	}
	if len(syms) == 0 {
		return nil, nil
	}
	for {
		bestRank := len(t.mergeRank)
		best := tokenPair{}
		found := false
		for i := 0; i+1 < len(syms); i++ {
			pair := tokenPair{left: syms[i], right: syms[i+1]}
			rank, ok := t.mergeRank[pair]
			if ok && rank < bestRank {
				bestRank = rank
				best = pair
				found = true
			}
		}
		if !found {
			return syms, nil
		}
		next := syms[:0]
		for i := 0; i < len(syms); i++ {
			if i+1 < len(syms) && syms[i] == best.left && syms[i+1] == best.right {
				next = append(next, best.left+best.right)
				i++
				continue
			}
			next = append(next, syms[i])
		}
		syms = next
	}
}

// preTokenizeByteLevel implements the GPT-2 ByteLevel(+Digits) pre-tokenizer used
// by SmolLM2 and bare-BPE fixtures: a leading space attaches only to a following
// letter/symbol run (` ?\p{L}+`), digits are isolated.
func preTokenizeByteLevel(s string) []string {
	var out []string
	for len(s) > 0 {
		if spaces := leadingSpaces(s); spaces > 0 {
			if spaces == len(s) {
				out = append(out, s)
				break
			}
			_, nextSize := utf8.DecodeRuneInString(s[spaces:])
			next := s[spaces : spaces+nextSize]
			if spaces > 1 || isDigitString(next) {
				emit := spaces
				if !isDigitString(next) && s[spaces-1] == ' ' {
					emit--
				}
				if emit > 0 {
					out = append(out, s[:emit])
					s = s[emit:]
					continue
				}
			}
			if s[0] == ' ' && !isDigitString(next) {
				piece, n := consumeNonSpaceWithOptionalSpace(s)
				out = append(out, piece)
				s = s[n:]
				continue
			}
			out = append(out, s[:spaces])
			s = s[spaces:]
			continue
		}
		if piece, n, ok := consumeContraction(s); ok {
			out = append(out, piece)
			s = s[n:]
			continue
		}
		r, size := utf8.DecodeRuneInString(s)
		if unicode.IsDigit(r) {
			out = append(out, s[:size])
			s = s[size:]
			continue
		}
		if unicode.IsLetter(r) {
			n := consumeRun(s, unicode.IsLetter)
			out = append(out, s[:n])
			s = s[n:]
			continue
		}
		if !unicode.IsSpace(r) && !unicode.IsLetter(r) && !unicode.IsNumber(r) {
			n := consumeRun(s, func(rr rune) bool {
				return !unicode.IsSpace(rr) && !unicode.IsLetter(rr) && !unicode.IsNumber(rr)
			})
			out = append(out, s[:n])
			s = s[n:]
			continue
		}
		out = append(out, s[:size])
		s = s[size:]
	}
	return out
}

func leadingSpaces(s string) int {
	n := 0
	for n < len(s) {
		if s[n] != ' ' {
			break
		}
		n++
	}
	return n
}

func consumeNonSpaceWithOptionalSpace(s string) (string, int) {
	if s == "" || s[0] != ' ' {
		return "", 0
	}
	r, size := utf8.DecodeRuneInString(s[1:])
	if r == utf8.RuneError && size == 0 {
		return s[:1], 1
	}
	if unicode.IsLetter(r) {
		n := 1 + consumeRun(s[1:], unicode.IsLetter)
		return s[:n], n
	}
	if !unicode.IsSpace(r) && !unicode.IsLetter(r) && !unicode.IsNumber(r) {
		n := 1 + consumeRun(s[1:], func(rr rune) bool {
			return !unicode.IsSpace(rr) && !unicode.IsLetter(rr) && !unicode.IsNumber(rr)
		})
		return s[:n], n
	}
	return s[:1], 1
}

func consumeContraction(s string) (string, int, bool) {
	for _, suffix := range []string{"'s", "'t", "'re", "'ve", "'m", "'ll", "'d"} {
		if strings.HasPrefix(s, suffix) {
			return suffix, len(suffix), true
		}
	}
	return "", 0, false
}

func consumeRun(s string, pred func(rune) bool) int {
	n := 0
	for n < len(s) {
		r, size := utf8.DecodeRuneInString(s[n:])
		if !pred(r) {
			break
		}
		n += size
	}
	return n
}

func isDigitString(s string) bool {
	r, size := utf8.DecodeRuneInString(s)
	return size == len(s) && unicode.IsDigit(r)
}

// preTokenizeQwen implements the explicit Qwen2.5/Qwen3.6 Split regex:
//
//	(?i:'s|'t|'re|'ve|'m|'ll|'d)
//	  | [^\r\n\p{L}\p{N}]?\p{L}+
//	  | \p{N}
//	  |  ?[^\s\p{L}\p{N}]+[\r\n]*
//	  | \s*[\r\n]+
//	  | \s+(?!\S)
//	  | \s+
//
// applied left-to-right ("Isolated"). Go's RE2 cannot express the \s+(?!\S)
// negative lookahead, and the repo avoids non-vendored deps, so this is a
// hand-written scanner, gated byte-exact against llama.cpp in oracle_qwen_test.go.
// Unlike the ByteLevel default it attaches ANY non-newline/non-alnum char to a
// following letter run ("(n", "_case", "\tand").
func preTokenizeQwen(text string) []string {
	rs := []rune(text)
	n := len(rs)
	var out []string
	for i := 0; i < n; {
		if m := matchQwenAt(rs, i); m > 0 {
			out = append(out, string(rs[i:i+m]))
			i += m
		} else {
			out = append(out, string(rs[i:i+1])) // defensive; unreachable for valid input
			i++
		}
	}
	return out
}

func isL(r rune) bool  { return unicode.IsLetter(r) }
func isN(r rune) bool  { return unicode.IsNumber(r) }
func isWS(r rune) bool { return unicode.IsSpace(r) }

// matchQwenAt returns the rune length of the first Qwen alternative matching at i.
func matchQwenAt(rs []rune, i int) int {
	n := len(rs)
	r := rs[i]

	// 1. contractions: (?i:'s|'t|'re|'ve|'m|'ll|'d)
	if r == '\'' && i+1 < n {
		c1 := unicode.ToLower(rs[i+1])
		if i+2 < n {
			c2 := unicode.ToLower(rs[i+2])
			if (c1 == 'r' && c2 == 'e') || (c1 == 'v' && c2 == 'e') || (c1 == 'l' && c2 == 'l') {
				return 3
			}
		}
		if c1 == 's' || c1 == 't' || c1 == 'm' || c1 == 'd' {
			return 2
		}
	}

	// 2. [^\r\n\p{L}\p{N}]?\p{L}+
	{
		j := i
		if r != '\r' && r != '\n' && !isL(r) && !isN(r) {
			j++
		}
		k := j
		for k < n && isL(rs[k]) {
			k++
		}
		if k > j {
			return k - i
		}
	}

	// 3. \p{N}
	if isN(r) {
		return 1
	}

	// 4.  ?[^\s\p{L}\p{N}]+[\r\n]*
	{
		j := i
		if r == ' ' {
			j++
		}
		start := j
		for j < n && !isWS(rs[j]) && !isL(rs[j]) && !isN(rs[j]) {
			j++
		}
		if j > start {
			for j < n && (rs[j] == '\r' || rs[j] == '\n') {
				j++
			}
			return j - i
		}
	}

	// 5. \s*[\r\n]+
	if isWS(r) {
		j := i
		lastNL := -1
		for j < n && isWS(rs[j]) {
			if rs[j] == '\r' || rs[j] == '\n' {
				lastNL = j
			}
			j++
		}
		if lastNL >= 0 {
			return lastNL - i + 1
		}
	}

	// 6. \s+(?!\S)  and  7. \s+
	if isWS(r) {
		j := i
		for j < n && isWS(rs[j]) {
			j++
		}
		runLen := j - i
		if j == n {
			return runLen
		}
		if runLen >= 2 {
			return runLen - 1
		}
		return runLen
	}

	return 0
}
