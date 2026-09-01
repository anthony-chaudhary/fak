// Package validatedjson encodes artifacts whose types refuse to serialize in
// an invalid state: the value's own Validate gate runs before json.Marshal, so
// a caller can never spool a malformed artifact a downstream reader would only
// reject later.
package validatedjson

import "encoding/json"

// Validated is the contract a value must satisfy to be encoded here: a
// value-receiver Validate that returns an error for every state the type
// considers invalid.
type Validated interface {
	Validate() error
}

// Marshal returns the compact JSON encoding of v after v.Validate() passes.
// The Validate error is returned unwrapped, exactly as the value's own encode
// method returned it, so existing error wrapping and sentinels keep working;
// a json.Marshal error is likewise returned unwrapped.
func Marshal(v Validated) ([]byte, error) {
	if err := v.Validate(); err != nil {
		return nil, err
	}
	return json.Marshal(v)
}
