package quality

import (
	"encoding/json"
	"fmt"
	"hash/fnv"
	"strconv"
	"strings"
)

// tokenizer_parity.go is the tokenizer + chat-template parity child of the
// quality spine (#4521): rendering role-tagged chat messages into a token-id
// sequence — special tokens INCLUDED — must be a pure function of the messages,
// and the engine's rendering must match a pinned reference id for id. Template
// drift is a classic silent-quality defect: a missing BOS, a doubled BOS, a
// wrong or dropped role marker, or a special token mapped to a stale id all
// produce fluent-looking generations from a subtly wrong conditioning prefix,
// which no downstream text metric can see. This file models a tiny
// deterministic tokenizer+template as the pure rendering function, provides an
// engine runner with those exact defect classes injectable, and registers the
// "tokenizer-parity" differential oracle that pins the FIRST id at which the
// engine's rendering departed from the pinned reference.

// The special-token id table the tiny template renders with. Content word ids
// start at tokParityWordIDBase, so a special id can never alias a word id.
const (
	tokParityBOSID       = 1 // beginning-of-sequence, exactly once, first
	tokParityEOSID       = 2 // end-of-sequence, exactly once, last
	tokParitySystemID    = 3 // <|system|> role marker
	tokParityUserID      = 4 // <|user|> role marker
	tokParityAssistantID = 5 // <|assistant|> role marker
	tokParityEOTID       = 6 // <|eot|> end-of-turn, closes every message
	// tokParityLegacyEOTID is the stale id the wrong-special-id defect maps
	// <|eot|> to — the "engine shipped last generation's special-token table" bug.
	tokParityLegacyEOTID = 999
	// tokParityWordIDBase offsets content word ids clear of the special table.
	tokParityWordIDBase = 1000
)

// tokParityID renders a numeric token id in the canonical string form traces
// carry (Trace.Tokens is a []string surface).
func tokParityID(n int) string { return strconv.Itoa(n) }

// tokParitySpecialNames maps a special token id (string form) to its
// human-readable marker, so a divergence Detail names WHICH special token was
// wrong instead of leaving a bare integer to decode.
var tokParitySpecialNames = map[string]string{
	tokParityID(tokParityBOSID):       "<bos>",
	tokParityID(tokParityEOSID):       "<eos>",
	tokParityID(tokParitySystemID):    "<|system|>",
	tokParityID(tokParityUserID):      "<|user|>",
	tokParityID(tokParityAssistantID): "<|assistant|>",
	tokParityID(tokParityEOTID):       "<|eot|>",
}

// tokParityDescribe renders one token id for a Detail line, annotating it with
// its special-token marker when it has one.
func tokParityDescribe(id string) string {
	if name, ok := tokParitySpecialNames[id]; ok {
		return fmt.Sprintf("%s (%s)", id, name)
	}
	return id
}

// tokParityMessage is one role-tagged chat message. A case serializes its
// message list as JSON into Prompt — the Trace/QualityCase shapes stay
// untouched and the richer per-case data rides an existing string field.
type tokParityMessage struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

// tokParityParseMessages decodes the JSON message list a case carries in its
// Prompt. A malformed prompt is a refused run, not a silently-empty rendering.
func tokParityParseMessages(prompt string) ([]tokParityMessage, error) {
	var msgs []tokParityMessage
	if err := json.Unmarshal([]byte(prompt), &msgs); err != nil {
		return nil, fmt.Errorf("prompt is not a JSON message list: %w", err)
	}
	return msgs, nil
}

// tokParityRoleID maps a message role to its marker id. Unknown roles render as
// user turns — deterministically, so both paths agree on the fallback.
func tokParityRoleID(role string) int {
	switch strings.ToLower(strings.TrimSpace(role)) {
	case "system":
		return tokParitySystemID
	case "assistant":
		return tokParityAssistantID
	default:
		return tokParityUserID
	}
}

// tokParityWordID maps one content word to a deterministic id via FNV-1a,
// offset into the word range so it can never collide with a special id. It is a
// pure function of the word bytes — no vocab file, no ambient state.
func tokParityWordID(word string) int {
	h := fnv.New32a()
	h.Write([]byte(word))
	return tokParityWordIDBase + int(h.Sum32()%9000)
}

// tokParityRender is the pure tokenizer+template function under test: messages
// in, token-id sequence out, special tokens in-band. The layout is
// BOS, then per message (role marker, content word ids, <|eot|>), then EOS —
// tiny, but structurally the shape every real chat template renders, which is
// exactly the surface the defect classes corrupt.
func tokParityRender(msgs []tokParityMessage) []string {
	ids := []string{tokParityID(tokParityBOSID)}
	for _, m := range msgs {
		ids = append(ids, tokParityID(tokParityRoleID(m.Role)))
		for _, w := range strings.Fields(m.Content) {
			ids = append(ids, tokParityID(tokParityWordID(w)))
		}
		ids = append(ids, tokParityID(tokParityEOTID))
	}
	return append(ids, tokParityID(tokParityEOSID))
}

// tokParityFirst returns the index of the first occurrence of id in ids, or -1.
func tokParityFirst(ids []string, id string) int {
	for i, v := range ids {
		if v == id {
			return i
		}
	}
	return -1
}

// The injectable template-drift defect classes — the exact bug taxonomy #4521
// exists to gate: BOS handling, role-marker fidelity, and the special-token id
// table.
const (
	tokParityDefectMissingBOS      = "missing-bos"
	tokParityDefectDoubleBOS       = "double-bos"
	tokParityDefectWrongRoleMarker = "wrong-role-marker"
	tokParityDefectDroppedMarker   = "missing-role-marker"
	tokParityDefectWrongEOTID      = "wrong-eot-id"
)

// TokParityRunner renders the case's message list through the shared pure
// template. The zero value is a faithful engine; the defect field (set via
// TokenizerParityEngine) injects one template-drift bug into the rendered ids.
// It is the ScriptedRunner-style adapter for the tokenization seam: a real
// engine tokenizer wires in behind the same Runner interface and is judged the
// same way.
type TokParityRunner struct {
	Label  string
	defect string
}

func (r TokParityRunner) Name() string {
	if r.Label != "" {
		return r.Label
	}
	return "template-engine"
}

func (r TokParityRunner) Run(c QualityCase) (Trace, error) {
	msgs, err := tokParityParseMessages(c.Prompt)
	if err != nil {
		return Trace{}, fmt.Errorf("tokenizer-parity engine: %w", err)
	}
	ids := tokParityRender(msgs)
	switch r.defect {
	case tokParityDefectMissingBOS:
		// The engine never prepends BOS: every id shifts left by one.
		ids = ids[1:]
	case tokParityDefectDoubleBOS:
		// The template AND the tokenizer each add BOS — the classic double-BOS bug.
		ids = append([]string{tokParityID(tokParityBOSID)}, ids...)
	case tokParityDefectWrongRoleMarker:
		// The user turn is rendered under the assistant marker.
		if i := tokParityFirst(ids, tokParityID(tokParityUserID)); i >= 0 {
			ids[i] = tokParityID(tokParityAssistantID)
		}
	case tokParityDefectDroppedMarker:
		// The user turn's role marker is dropped entirely.
		if i := tokParityFirst(ids, tokParityID(tokParityUserID)); i >= 0 {
			ids = append(ids[:i], ids[i+1:]...)
		}
	case tokParityDefectWrongEOTID:
		// The special-token table maps <|eot|> to a stale legacy id.
		for i, id := range ids {
			if id == tokParityID(tokParityEOTID) {
				ids[i] = tokParityID(tokParityLegacyEOTID)
			}
		}
	}
	t := Trace{Tokens: ids, Text: strings.Join(ids, " ")}
	t.Runner = r.Name()
	return t, nil
}

// TokenizerParityEngine returns a template engine runner with an optional
// injected defect: "" renders faithfully (id-exact with the pinned reference);
// each tokParityDefect* constant injects that one drift class. This is the
// deterministic mutant source the tests use to prove the parity gate trips per
// defect class.
func TokenizerParityEngine(defect string) TokParityRunner {
	switch defect {
	case tokParityDefectMissingBOS, tokParityDefectDoubleBOS, tokParityDefectWrongRoleMarker,
		tokParityDefectDroppedMarker, tokParityDefectWrongEOTID:
		return TokParityRunner{Label: "engine-template-" + defect, defect: defect}
	default:
		return TokParityRunner{Label: "engine-template-clean"}
	}
}

// TokenizerParityCase builds the deterministic parity case: a two-turn
// system+user conversation serialized as JSON into Prompt, and the
// pinned-correct id sequence — rendered once by the pure template — as the
// reference trace. The reference IS the pin: any engine rendering that departs
// from it, at any id, special or content, is a template-parity defect.
func TokenizerParityCase() QualityCase {
	msgs := []tokParityMessage{
		{Role: "system", Content: "you are the weekly throughput reporter"},
		{Role: "user", Content: "summarize this week for the executive rollup"},
	}
	b, err := json.Marshal(msgs)
	if err != nil {
		panic("quality: marshal tokenizer-parity messages: " + err.Error())
	}
	ids := tokParityRender(msgs)
	return QualityCase{
		Schema:  CaseSchema,
		ID:      "tokenizer-parity-demo",
		Version: 1,
		Prompt:  string(b),
		Params:  SamplingParams{Temperature: 0, MaxTokens: len(ids)},
		Reference: Trace{
			Tokens: ids,
			Text:   strings.Join(ids, " "),
		},
		Oracles: []string{"tokenizer-parity"},
	}
}

// TokenizerParity is the differential oracle for #4521: the engine's rendered
// token-id sequence must equal the pinned reference sequence exactly — special
// tokens and content ids alike. Any mismatch is reported as the FIRST divergent
// id, annotated with its special-token marker when it has one, so "the chat
// template drifted" localizes to "id 0 was <|system|> where the reference put
// <bos>".
type TokenizerParity struct{}

func (TokenizerParity) Name() string { return "tokenizer-parity" }
func (TokenizerParity) Kind() string { return "differential" }

func init() { Register(TokenizerParity{}) }

func (TokenizerParity) Judge(ref, eng Trace, _ QualityCase) Verdict {
	v := Verdict{Oracle: "tokenizer-parity", Kind: "differential", Pass: true}
	n := len(ref.Tokens)
	if len(eng.Tokens) < n {
		n = len(eng.Tokens)
	}
	for i := 0; i < n; i++ {
		if ref.Tokens[i] != eng.Tokens[i] {
			v.Pass = false
			v.FirstDivergence = &Divergence{Index: i, Reference: ref.Tokens[i], Engine: eng.Tokens[i]}
			v.Detail = fmt.Sprintf("rendered id %d diverged: reference %s, engine %s",
				i, tokParityDescribe(ref.Tokens[i]), tokParityDescribe(eng.Tokens[i]))
			return v
		}
	}
	if len(ref.Tokens) != len(eng.Tokens) {
		v.Pass = false
		v.FirstDivergence = &Divergence{Index: n, Reference: tokenAt(ref.Tokens, n), Engine: tokenAt(eng.Tokens, n)}
		v.Detail = fmt.Sprintf("rendered length diverged at %d: reference has %d ids, engine has %d",
			n, len(ref.Tokens), len(eng.Tokens))
		return v
	}
	v.Detail = fmt.Sprintf("%d rendered token ids matched the pinned tokenizer+template reference", len(ref.Tokens))
	return v
}
