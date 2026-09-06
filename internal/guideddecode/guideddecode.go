// Package guideddecode is slice-1 of the constrained tool-JSON decoding gap
// (issue #26): a SOUND, byte-level compiler that constrains a model's decode to
// a valid tool-call envelope.
//
// The canonical tool-call envelope fak expects is:
//
//	{"name":"<TOOLNAME>","arguments":<json>}
//
// where <TOOLNAME> is exactly one of a declared set of tool names and <json> is
// any JSON value. This slice constrains only the tool-NAME enum and the fixed
// envelope skeleton around it; full arguments-schema enforcement (validating the
// <json> value against a per-tool parameter schema) is a later slice.
//
// Byte-level only: this package deliberately does NOT import internal/model or
// internal/tokenizer, so it stays tier-flexible and testable in isolation. The
// model-facing model.LogitMask adapter — which maps this byte-level admission
// set onto a tokenizer's vocabulary — is the next slice.
//
// # Soundness contract
//
// The load-bearing property of AllowedNextBytes is soundness: it never forbids a
// byte that could begin or continue SOME valid completion — it only forbids
// structurally-impossible bytes. It can therefore only PREVENT an invalid tool
// name or a broken envelope skeleton; it can never reject a byte that lies on a
// valid path. A decoder masked with this set can never be steered off every
// valid envelope by the mask itself.
//
// # Return convention
//
// AllowedNextBytes distinguishes three outcomes by the shape of its return:
//
//   - nil map        => UNCONSTRAINED: the prefix has consumed the fixed skeleton
//     up through `{"name":"<validname>","arguments":` — any byte is now allowed
//     (the arguments JSON value and the trailing `}` are not constrained in
//     slice-1).
//   - empty non-nil  => DEAD END: the prefix already diverged from every valid
//     envelope and no byte can rescue it. A decoder that has been masked
//     correctly at every prior step never reaches this state.
//   - non-empty      => exactly the set of bytes that keep the prefix on a valid
//     path toward some declared envelope.
package guideddecode

import "math/bits"

// ToolSchema is the minimal input to the constrainer: the set of declared,
// callable tool names. Names are treated as raw identifier bytes and matched
// literally; slice-1 assumes they contain no characters that would need JSON
// string escaping (a name is an identifier). A name containing `"` or `\` is out
// of slice-1 scope, but such a name is handled literally and never panics.
type ToolSchema struct {
	Names []string
}

// pre is the fixed skeleton that precedes the tool-name enum: `{"name":"`.
const pre = `{"name":"`

// suf is the fixed skeleton that follows the tool name: the closing quote of the
// name plus `,"arguments":`. Concatenated, pre + name + suf is the full
// constrained skeleton `{"name":"<name>","arguments":`, after which the decode is
// UNCONSTRAINED.
const suf = `","arguments":`

func isJSONWhitespace(b byte) bool {
	return b == ' ' || b == '\t' || b == '\n' || b == '\r'
}

func hasPrefixString(s string, prefix []byte) bool {
	if len(s) < len(prefix) {
		return false
	}
	for i := 0; i < len(prefix); i++ {
		if s[i] != prefix[i] {
			return false
		}
	}
	return true
}

func equalString(s string, b []byte) bool {
	if len(s) != len(b) {
		return false
	}
	for i := 0; i < len(b); i++ {
		if s[i] != b[i] {
			return false
		}
	}
	return true
}

func hasPrefixBytes(b []byte, prefix string) bool {
	if len(b) < len(prefix) {
		return false
	}
	for i := 0; i < len(prefix); i++ {
		if b[i] != prefix[i] {
			return false
		}
	}
	return true
}

// ByteBitset is a zero-allocation 256-bit set representing allowed bytes (0-255).
// It fits in 32 bytes ([4]uint64), passes by value on the stack, and operates
// without heap allocations.
type ByteBitset [4]uint64

// Set marks byte v as present in the bitset.
func (b *ByteBitset) Set(v byte) {
	b[v>>6] |= uint64(1) << (v & 63)
}

// Add is an alias for Set, marking byte v as present.
func (b *ByteBitset) Add(v byte) {
	b.Set(v)
}

// Remove clears byte v from the bitset.
func (b *ByteBitset) Remove(v byte) {
	b[v>>6] &^= uint64(1) << (v & 63)
}

// Clear is an alias for Remove, clearing byte v from the bitset.
func (b *ByteBitset) Clear(v byte) {
	b.Remove(v)
}

// Has reports whether byte v is present in the bitset.
func (b ByteBitset) Has(v byte) bool {
	return (b[v>>6] & (uint64(1) << (v & 63))) != 0
}

// Contains is an alias for Has, reporting whether byte v is present.
func (b ByteBitset) Contains(v byte) bool {
	return b.Has(v)
}

// Empty reports whether the bitset contains no bytes.
func (b ByteBitset) Empty() bool {
	return (b[0] | b[1] | b[2] | b[3]) == 0
}

// Count returns the number of bytes set in the bitset.
func (b ByteBitset) Count() int {
	return bits.OnesCount64(b[0]) + bits.OnesCount64(b[1]) + bits.OnesCount64(b[2]) + bits.OnesCount64(b[3])
}

// SingleByte returns (byte, true) if exactly one byte is present in the bitset,
// or (0, false) if the bitset has 0 or 2+ bytes.
func (b ByteBitset) SingleByte() (byte, bool) {
	if b.Count() != 1 {
		return 0, false
	}
	for i := 0; i < 4; i++ {
		w := b[i]
		if w != 0 {
			tz := bits.TrailingZeros64(w)
			return byte(i*64 + tz), true
		}
	}
	return 0, false
}

// Bytes returns a newly allocated slice containing all bytes present in the bitset
// in ascending order, or nil if the bitset is empty.
func (b ByteBitset) Bytes() []byte {
	cnt := b.Count()
	if cnt == 0 {
		return nil
	}
	out := make([]byte, 0, cnt)
	for i := 0; i < 4; i++ {
		w := b[i]
		for w != 0 {
			tz := bits.TrailingZeros64(w)
			out = append(out, byte(i*64+tz))
			w &^= uint64(1) << tz
		}
	}
	return out
}

// ToMap converts the bitset to a map[byte]bool representation.
// An empty bitset produces an empty non-nil map.
func (b ByteBitset) ToMap() map[byte]bool {
	cnt := b.Count()
	m := make(map[byte]bool, cnt)
	if cnt == 0 {
		return m
	}
	for i := 0; i < 4; i++ {
		w := b[i]
		for w != 0 {
			tz := bits.TrailingZeros64(w)
			m[byte(i*64+tz)] = true
			w &^= uint64(1) << tz
		}
	}
	return m
}

// SingleByteBitset returns a ByteBitset containing exactly the single byte b.
func SingleByteBitset(b byte) ByteBitset {
	var bs ByteBitset
	bs.Set(b)
	return bs
}

func singleByteBitset(b byte) ByteBitset {
	return SingleByteBitset(b)
}

// AllowedNextBitset computes which bytes may legally extend prefix toward a
// valid tool-call envelope {"name":"<one of schema.Names>","arguments":<json>},
// returning a zero-allocation ByteBitset.
//
// Return convention:
//   - unconstrained == true  => UNCONSTRAINED: envelope skeleton complete, any byte is allowed.
//   - unconstrained == false, allowed.Empty() == true => DEAD END: invalid envelope prefix.
//   - unconstrained == false, allowed.Empty() == false => CONSTRAINED: exactly the set of admitted bytes.
func AllowedNextBitset(prefix []byte, schema ToolSchema) (allowed ByteBitset, unconstrained bool) {
	// Step 1: skip optional whitespace before '{'
	pos := 0
	for pos < len(prefix) && isJSONWhitespace(prefix[pos]) {
		pos++
	}
	if pos == len(prefix) {
		return SingleByteBitset('{'), false
	}
	if prefix[pos] != '{' {
		return ByteBitset{}, false
	}
	pos++

	// Step 2: skip optional whitespace after '{'
	for pos < len(prefix) && isJSONWhitespace(prefix[pos]) {
		pos++
	}
	if pos == len(prefix) {
		return SingleByteBitset('"'), false
	}

	// Step 3: match literal `"name"`
	const nameKey = `"name"`
	for i := 0; i < len(nameKey); i++ {
		if pos == len(prefix) {
			return SingleByteBitset(nameKey[i]), false
		}
		if prefix[pos] != nameKey[i] {
			return ByteBitset{}, false
		}
		pos++
	}

	// Step 4: skip optional whitespace before ':' (around ':')
	for pos < len(prefix) && isJSONWhitespace(prefix[pos]) {
		pos++
	}
	if pos == len(prefix) {
		return SingleByteBitset(':'), false
	}
	if prefix[pos] != ':' {
		return ByteBitset{}, false
	}
	pos++

	// Step 5: skip optional whitespace after ':' (around ':')
	for pos < len(prefix) && isJSONWhitespace(prefix[pos]) {
		pos++
	}
	if pos == len(prefix) {
		return SingleByteBitset('"'), false
	}
	if prefix[pos] != '"' {
		return ByteBitset{}, false
	}
	pos++

	// Beyond the opening quote of the tool name.
	// rest holds the bytes from the start of the tool name.
	rest := prefix[pos:]
	const argKey = `"arguments"`

	for _, name := range schema.Names {
		if len(rest) < len(name) {
			if hasPrefixString(name, rest) {
				allowed.Set(name[len(rest)])
			}
			continue
		}
		if len(rest) == len(name) {
			if equalString(name, rest) {
				allowed.Set('"')
			}
			continue
		}
		// len(rest) > len(name)
		if !hasPrefixBytes(rest, name) || rest[len(name)] != '"' {
			continue
		}

		afterName := rest[len(name)+1:]
		p := 0
		// Step 6: skip optional whitespace before ',' (around ',')
		for p < len(afterName) && isJSONWhitespace(afterName[p]) {
			p++
		}
		if p == len(afterName) {
			allowed.Set(',')
			continue
		}
		if afterName[p] != ',' {
			continue
		}
		p++

		// Step 7: skip optional whitespace after ',' (around ',')
		for p < len(afterName) && isJSONWhitespace(afterName[p]) {
			p++
		}
		if p == len(afterName) {
			allowed.Set('"')
			continue
		}

		// Step 8: match `"arguments"`
		argKeyMatched := true
		for i := 0; i < len(argKey); i++ {
			if p == len(afterName) {
				allowed.Set(argKey[i])
				argKeyMatched = false
				break
			}
			if afterName[p] != argKey[i] {
				argKeyMatched = false
				break
			}
			p++
		}
		if !argKeyMatched {
			continue
		}

		// Step 9: skip optional whitespace before ':' (around ':')
		for p < len(afterName) && isJSONWhitespace(afterName[p]) {
			p++
		}
		if p == len(afterName) {
			allowed.Set(':')
			continue
		}
		if afterName[p] != ':' {
			continue
		}
		p++

		// Step 10: ':' matched! Tool envelope skeleton is complete.
		return ByteBitset{}, true
	}

	return allowed, false
}

// AllowedNextByteBitset is an alias for AllowedNextBitset.
func AllowedNextByteBitset(prefix []byte, schema ToolSchema) (ByteBitset, bool) {
	return AllowedNextBitset(prefix, schema)
}

// AllowedNextBytesBitset is an alias for AllowedNextBitset.
func AllowedNextBytesBitset(prefix []byte, schema ToolSchema) (ByteBitset, bool) {
	return AllowedNextBitset(prefix, schema)
}

// AllowedNextBytes returns which bytes may legally extend prefix toward a valid
// tool-call envelope {"name":"<one of schema.Names>","arguments":<json>}.
//
// Optional JSON whitespace (spaces, tabs, newlines, carriage returns) is tolerated
// in structural positions (after '{', around ':', after ',', before '{').
//
// Preserves backward compatibility by wrapping AllowedNextBitset and converting
// the result to map[byte]bool.
//
// See the package doc for the soundness contract and the nil / empty-non-nil /
// non-empty return convention.
func AllowedNextBytes(prefix []byte, schema ToolSchema) map[byte]bool {
	allowed, unconstrained := AllowedNextBitset(prefix, schema)
	if unconstrained {
		return nil
	}
	return allowed.ToMap()
}
