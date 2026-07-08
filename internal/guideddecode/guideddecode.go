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

// AllowedNextBytes returns which bytes may legally extend prefix toward a valid
// tool-call envelope {"name":"<one of schema.Names>","arguments":<json>}.
//
// See the package doc for the soundness contract and the nil / empty-non-nil /
// non-empty return convention.
func AllowedNextBytes(prefix []byte, schema ToolSchema) map[byte]bool {
	// Region 1: still consuming the fixed PRE literal. PRE is enforced from the
	// literal itself, independent of the name set — so an empty schema.Names
	// still admits '{' at prefix=="" and only dead-ends at the enum branch.
	if len(prefix) < len(pre) {
		if !bytes.Equal(prefix, []byte(pre)[:len(prefix)]) {
			return map[byte]bool{} // diverged from PRE => dead end
		}
		return map[byte]bool{pre[len(prefix)]: true}
	}

	// Past PRE: the prefix must reproduce PRE exactly in its first bytes, or it
	// has already diverged from every envelope.
	if !bytes.HasPrefix(prefix, []byte(pre)) {
		return map[byte]bool{} // diverged from PRE => dead end
	}

	// Region 2/3: the NAME enum and the fixed SUF that follows it, evaluated
	// per still-viable name. A name stays viable while prefix matches its
	// skeleton byte-for-byte; the next skeleton byte of each viable name is
	// admitted. Overlapping names (e.g. "get" and "get_weather") therefore
	// naturally admit both the closing quote of the shorter name and the next
	// identifier byte of the longer one.
	unconstrained := false
	allowed := map[byte]bool{}
	for _, name := range schema.Names {
		s := []byte(pre + name + suf)
		if len(prefix) >= len(s) {
			// The prefix is at or beyond the end of this skeleton. If it
			// reproduces the whole skeleton, we are in the UNCONSTRAINED region;
			// otherwise this name is no longer viable.
			if bytes.HasPrefix(prefix, s) {
				unconstrained = true
			}
			continue
		}
		if bytes.Equal(prefix, s[:len(prefix)]) {
			allowed[s[len(prefix)]] = true
		}
	}
	if unconstrained {
		return nil
	}
	return allowed
}
