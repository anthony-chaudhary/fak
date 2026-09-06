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

import "bytes"

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

// AllowedNextBytes returns which bytes may legally extend prefix toward a valid
// tool-call envelope {"name":"<one of schema.Names>","arguments":<json>}.
//
// Optional JSON whitespace (spaces, tabs, newlines, carriage returns) is tolerated
// in structural positions (after '{', around ':', after ',', before '{').
//
// See the package doc for the soundness contract and the nil / empty-non-nil /
// non-empty return convention.
func AllowedNextBytes(prefix []byte, schema ToolSchema) map[byte]bool {
	// Step 1: skip optional whitespace before '{'
	pos := 0
	for pos < len(prefix) && isJSONWhitespace(prefix[pos]) {
		pos++
	}
	if pos == len(prefix) {
		return map[byte]bool{'{': true}
	}
	if prefix[pos] != '{' {
		return map[byte]bool{}
	}
	pos++

	// Step 2: skip optional whitespace after '{'
	for pos < len(prefix) && isJSONWhitespace(prefix[pos]) {
		pos++
	}
	if pos == len(prefix) {
		return map[byte]bool{'"': true}
	}

	// Step 3: match literal `"name"`
	const nameKey = `"name"`
	for i := 0; i < len(nameKey); i++ {
		if pos == len(prefix) {
			return map[byte]bool{nameKey[i]: true}
		}
		if prefix[pos] != nameKey[i] {
			return map[byte]bool{}
		}
		pos++
	}

	// Step 4: skip optional whitespace before ':' (around ':')
	for pos < len(prefix) && isJSONWhitespace(prefix[pos]) {
		pos++
	}
	if pos == len(prefix) {
		return map[byte]bool{':': true}
	}
	if prefix[pos] != ':' {
		return map[byte]bool{}
	}
	pos++

	// Step 5: skip optional whitespace after ':' (around ':')
	for pos < len(prefix) && isJSONWhitespace(prefix[pos]) {
		pos++
	}
	if pos == len(prefix) {
		return map[byte]bool{'"': true}
	}
	if prefix[pos] != '"' {
		return map[byte]bool{}
	}
	pos++

	// Beyond the opening quote of the tool name.
	// rest holds the bytes from the start of the tool name.
	rest := prefix[pos:]

	unconstrained := false
	allowed := map[byte]bool{}
	const argKey = `"arguments"`

	for _, name := range schema.Names {
		nameBytes := []byte(name)
		if len(rest) < len(nameBytes) {
			if bytes.HasPrefix(nameBytes, rest) {
				allowed[nameBytes[len(rest)]] = true
			}
			continue
		}
		if len(rest) == len(nameBytes) {
			if bytes.Equal(rest, nameBytes) {
				allowed['"'] = true
			}
			continue
		}
		// len(rest) > len(nameBytes)
		if !bytes.HasPrefix(rest, nameBytes) || rest[len(nameBytes)] != '"' {
			continue
		}

		afterName := rest[len(nameBytes)+1:]
		p := 0
		// Step 6: skip optional whitespace before ',' (around ',')
		for p < len(afterName) && isJSONWhitespace(afterName[p]) {
			p++
		}
		if p == len(afterName) {
			allowed[','] = true
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
			allowed['"'] = true
			continue
		}

		// Step 8: match `"arguments"`
		argKeyMatched := true
		for i := 0; i < len(argKey); i++ {
			if p == len(afterName) {
				allowed[argKey[i]] = true
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
			allowed[':'] = true
			continue
		}
		if afterName[p] != ':' {
			continue
		}
		p++

		// Step 10: ':' matched! Tool envelope skeleton is complete.
		unconstrained = true
	}

	if unconstrained {
		return nil
	}
	return allowed
}
