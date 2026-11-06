package pricebook

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"os"
)

// Decode decodes exactly one serialized price book from data, refusing
// anything the v1 file schema does not declare: bad JSON, a field outside
// the schema, or trailing data. Decoding enforces the closed validation
// rules too, so the returned built book is ready to use and ready to stamp.
//
// Faults split into two classes: a payload-shape fault (unparseable bytes,
// a foreign field, trailing data) reports ErrMalformedPayload, while a
// field that parses but violates the closed contract (for example a rate
// that is neither a number nor null) reports the specific validation
// sentinel, matchable up the chain to ErrInvalidPriceBook.
func Decode(data []byte) (*PriceBook, error) {
	var b PriceBook
	dec := json.NewDecoder(bytes.NewReader(data))
	dec.DisallowUnknownFields()
	if err := dec.Decode(&b); err != nil {
		if errors.Is(err, ErrInvalidPriceBook) {
			return nil, err
		}
		return nil, fmt.Errorf("%w: %v", ErrMalformedPayload, err)
	}
	if dec.More() {
		return nil, fmt.Errorf("%w: trailing data after price book", ErrMalformedPayload)
	}
	if err := b.Validate(); err != nil {
		return nil, err
	}
	return &b, nil
}

// LoadFile reads path and Decodes it.
func LoadFile(path string) (*PriceBook, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	return Decode(data)
}

// MarshalJSON encodes the book in the operator-file format: the same strict
// v1 schema Decode accepts, so a decoded book marshals back to bytes that
// decode to an identical book. Encoding refuses a book that fails Validate.
func (b *PriceBook) MarshalJSON() ([]byte, error) {
	if err := b.Validate(); err != nil {
		return nil, err
	}
	type plain PriceBook
	return json.Marshal(plain(*b))
}
