package microagent

import (
	"encoding/json"
	"errors"
	"sort"
	"strings"
)

const managedContextWireVersion = 1

// ArtifactPointer is a durable, independently readable artifact reference.
// ManagedContext retains pointers, never model-written summaries of stale text.
type ArtifactPointer struct {
	Kind string `json:"kind"`
	URI  string `json:"uri"`
}

type managedTurn struct {
	Msg      Msg               `json:"msg"`
	Pointers []ArtifactPointer `json:"pointers,omitempty"`
}

// ManagedContext is the M25 active-context layer over the M4 hard cap. At the
// cap it elides stale turns and carries forward only caller-supplied durable
// artifact pointers in a deterministic synthetic message. It never invokes a
// model and therefore cannot smuggle an unwitnessed recap into history.
type ManagedContext struct {
	cap         int
	turns       []managedTurn
	stale       map[string]ArtifactPointer
	compactions int
	peakTokens  int
}

func NewManagedContext(tokenCap int) *ManagedContext {
	if tokenCap <= 0 {
		tokenCap = DefaultContextCap
	}
	return &ManagedContext{cap: tokenCap, stale: map[string]ArtifactPointer{}}
}

func (c *ManagedContext) Cap() int         { return c.cap }
func (c *ManagedContext) Compactions() int { return c.compactions }
func (c *ManagedContext) PeakTokens() int  { return c.peakTokens }
func (c *ManagedContext) Tokens() int      { return estContextTokens(c.messages()) }
func (c *ManagedContext) Len() int         { return len(c.messages()) }

func (c *ManagedContext) Messages() []Msg {
	msgs := c.messages()
	out := make([]Msg, len(msgs))
	copy(out, msgs)
	return out
}

// Append adds one turn and compacts oldest turns until the context is bounded.
// Pointers are admitted only when both Kind and URI are non-empty. The return
// value is the number of stale turns elided by this append.
func (c *ManagedContext) Append(role, content string, pointers ...ArtifactPointer) (elided int) {
	clean := normalizePointers(pointers)
	c.turns = append(c.turns, managedTurn{Msg: Msg{Role: role, Content: content}, Pointers: clean})
	for c.Tokens() > c.cap && len(c.turns) > 1 {
		c.elideOldest()
		elided++
	}
	// If the newest turn alone fits but accumulated pointer text does not, keep
	// the newest pointers first and deterministically shed oldest stale refs.
	for c.Tokens() > c.cap && len(c.stale) > 0 {
		keys := sortedPointerKeys(c.stale)
		delete(c.stale, keys[0])
	}
	if c.Tokens() > c.cap && len(c.turns) == 1 {
		c.turns[0].Msg.Content = fitContent(c.turns[0].Msg.Role, c.turns[0].Msg.Content, c.cap)
	}
	if elided > 0 {
		c.compactions++
	}
	if tokens := c.Tokens(); tokens > c.peakTokens {
		c.peakTokens = tokens
	}
	return elided
}

func (c *ManagedContext) elideOldest() {
	old := c.turns[0]
	c.turns = c.turns[1:]
	for _, pointer := range old.Pointers {
		c.stale[pointerKey(pointer)] = pointer
	}
}

func (c *ManagedContext) messages() []Msg {
	out := make([]Msg, 0, len(c.turns)+1)
	if recap := renderPointerRecap(c.stale); recap != "" {
		out = append(out, Msg{Role: "system", Content: recap})
	}
	for _, turn := range c.turns {
		out = append(out, turn.Msg)
	}
	return out
}

func fitContent(role, content string, cap int) string {
	runes := []rune(content)
	const marker = "[latest-turn-elided]\n"
	lo, hi := 0, len(runes)
	for lo < hi {
		mid := (lo + hi + 1) / 2
		candidate := marker + string(runes[len(runes)-mid:])
		if estContextTokens([]Msg{{Role: role, Content: candidate}}) <= cap {
			lo = mid
		} else {
			hi = mid - 1
		}
	}
	return marker + string(runes[len(runes)-lo:])
}

func normalizePointers(in []ArtifactPointer) []ArtifactPointer {
	set := map[string]ArtifactPointer{}
	for _, pointer := range in {
		pointer.Kind = strings.TrimSpace(pointer.Kind)
		pointer.URI = strings.TrimSpace(pointer.URI)
		if pointer.Kind != "" && pointer.URI != "" {
			set[pointerKey(pointer)] = pointer
		}
	}
	keys := sortedPointerKeys(set)
	out := make([]ArtifactPointer, 0, len(keys))
	for _, key := range keys {
		out = append(out, set[key])
	}
	return out
}

func renderPointerRecap(pointers map[string]ArtifactPointer) string {
	if len(pointers) == 0 {
		return ""
	}
	var b strings.Builder
	b.WriteString("[durable-artifact-pointers]\n")
	for _, key := range sortedPointerKeys(pointers) {
		pointer := pointers[key]
		b.WriteString("- ")
		b.WriteString(pointer.Kind)
		b.WriteString(": ")
		b.WriteString(pointer.URI)
		b.WriteByte('\n')
	}
	return strings.TrimSuffix(b.String(), "\n")
}

func pointerKey(pointer ArtifactPointer) string { return pointer.Kind + "\x00" + pointer.URI }
func sortedPointerKeys(pointers map[string]ArtifactPointer) []string {
	keys := make([]string, 0, len(pointers))
	for key := range pointers {
		keys = append(keys, key)
	}
	sort.Strings(keys)
	return keys
}

type managedContextWire struct {
	Version     int                        `json:"version"`
	Cap         int                        `json:"cap"`
	Turns       []managedTurn              `json:"turns"`
	Stale       map[string]ArtifactPointer `json:"stale,omitempty"`
	Compactions int                        `json:"compactions"`
	PeakTokens  int                        `json:"peak_tokens"`
}

func (c *ManagedContext) Encode() ([]byte, error) {
	return json.Marshal(managedContextWire{Version: managedContextWireVersion, Cap: c.cap, Turns: c.turns, Stale: c.stale, Compactions: c.compactions, PeakTokens: c.peakTokens})
}
func (c *ManagedContext) Decode(data []byte) error {
	var wire managedContextWire
	if err := json.Unmarshal(data, &wire); err != nil {
		return err
	}
	if wire.Version != managedContextWireVersion {
		return errors.New("microagent: unsupported managed-context version")
	}
	if wire.Cap <= 0 {
		return errors.New("microagent: managed-context cap must be positive")
	}
	*c = ManagedContext{cap: wire.Cap, turns: wire.Turns, stale: wire.Stale, compactions: wire.Compactions, peakTokens: wire.PeakTokens}
	if c.stale == nil {
		c.stale = map[string]ArtifactPointer{}
	}
	if c.Tokens() > c.cap {
		return errors.New("microagent: decoded managed context exceeds cap")
	}
	return nil
}
