package scorecard

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"math"
	"math/big"
	"strconv"
	"strings"
)

// Value is a decoded JSON value that keeps object key order, so catalog rows can
// be echoed back in the order Python's dict preserved them. The dynamic type is
// one of nil, bool, Number, string, []Value or *Object.
type Value any

// Number is a JSON number kept as its source literal; Python decides int versus
// float from the literal's spelling, and the echo must make the same call.
type Number string

// Object is an insertion-ordered JSON object. A repeated key keeps its first
// position and its last value, which is what json.loads into a dict does.
type Object struct {
	Keys []string
	Vals []Value
	idx  map[string]int
}

func newObject() *Object { return &Object{idx: map[string]int{}} }

// Get returns the value under key and whether the key is present.
func (o *Object) Get(key string) (Value, bool) {
	if o == nil {
		return nil, false
	}
	i, ok := o.idx[key]
	if !ok {
		return nil, false
	}
	return o.Vals[i], true
}

// Set inserts or replaces key, keeping the first position of an existing key.
func (o *Object) Set(key string, v Value) {
	if i, ok := o.idx[key]; ok {
		o.Vals[i] = v
		return
	}
	o.idx[key] = len(o.Keys)
	o.Keys = append(o.Keys, key)
	o.Vals = append(o.Vals, v)
}

// MarshalJSON writes the object in insertion order.
func (o *Object) MarshalJSON() ([]byte, error) {
	var b bytes.Buffer
	if err := writeValue(&b, o); err != nil {
		return nil, err
	}
	return b.Bytes(), nil
}

// MarshalJSON writes the number the way Python's json.dumps re-emits it.
func (n Number) MarshalJSON() ([]byte, error) { return []byte(pyNumberRepr(string(n))), nil }

func writeValue(b *bytes.Buffer, v Value) error {
	switch x := v.(type) {
	case nil:
		b.WriteString("null")
	case bool:
		if x {
			b.WriteString("true")
		} else {
			b.WriteString("false")
		}
	case Number:
		b.WriteString(pyNumberRepr(string(x)))
	case string:
		writeJSONString(b, x)
	case []Value:
		b.WriteByte('[')
		for i, e := range x {
			if i > 0 {
				b.WriteByte(',')
			}
			if err := writeValue(b, e); err != nil {
				return err
			}
		}
		b.WriteByte(']')
	case *Object:
		b.WriteByte('{')
		for i, k := range x.Keys {
			if i > 0 {
				b.WriteByte(',')
			}
			writeJSONString(b, k)
			b.WriteByte(':')
			if err := writeValue(b, x.Vals[i]); err != nil {
				return err
			}
		}
		b.WriteByte('}')
	default:
		enc, err := json.Marshal(x)
		if err != nil {
			return err
		}
		b.Write(enc)
	}
	return nil
}

// writeJSONString writes s as a JSON string without HTML escaping; asciiJSON
// later turns any non-ASCII byte into the escaped form json.dumps uses.
func writeJSONString(b *bytes.Buffer, s string) {
	const hex = "0123456789abcdef"
	b.WriteByte('"')
	for i := 0; i < len(s); i++ {
		c := s[i]
		switch c {
		case '"', '\\':
			b.WriteByte('\\')
			b.WriteByte(c)
		case '\b':
			b.WriteString(`\b`)
		case '\f':
			b.WriteString(`\f`)
		case '\n':
			b.WriteString(`\n`)
		case '\r':
			b.WriteString(`\r`)
		case '\t':
			b.WriteString(`\t`)
		default:
			if c < 0x20 {
				b.WriteString(`\u00`)
				b.WriteByte(hex[c>>4])
				b.WriteByte(hex[c&0xf])
			} else {
				b.WriteByte(c)
			}
		}
	}
	b.WriteByte('"')
}

// DecodeValue parses one JSON document into an order-preserving Value.
func DecodeValue(data []byte) (Value, error) {
	dec := json.NewDecoder(bytes.NewReader(data))
	dec.UseNumber()
	v, err := decodeNext(dec)
	if err != nil {
		return nil, err
	}
	if _, err := dec.Token(); !errors.Is(err, io.EOF) {
		return nil, errors.New("Extra data after the JSON document")
	}
	return v, nil
}

func decodeNext(dec *json.Decoder) (Value, error) {
	tok, err := dec.Token()
	if err != nil {
		return nil, err
	}
	switch t := tok.(type) {
	case json.Delim:
		switch t {
		case '{':
			o := newObject()
			for dec.More() {
				kt, err := dec.Token()
				if err != nil {
					return nil, err
				}
				key, ok := kt.(string)
				if !ok {
					return nil, fmt.Errorf("object key is %T", kt)
				}
				v, err := decodeNext(dec)
				if err != nil {
					return nil, err
				}
				o.Set(key, v)
			}
			if _, err := dec.Token(); err != nil {
				return nil, err
			}
			return o, nil
		case '[':
			out := []Value{}
			for dec.More() {
				v, err := decodeNext(dec)
				if err != nil {
					return nil, err
				}
				out = append(out, v)
			}
			if _, err := dec.Token(); err != nil {
				return nil, err
			}
			return out, nil
		}
		return nil, fmt.Errorf("unexpected delimiter %v", t)
	case json.Number:
		return Number(t.String()), nil
	default:
		return t, nil
	}
}

// isIntLiteral mirrors json.loads: a literal without '.', 'e' or 'E' is an int.
func isIntLiteral(lit string) bool { return !strings.ContainsAny(lit, ".eE") }

func pyNumberRepr(lit string) string {
	if isIntLiteral(lit) {
		n, ok := new(big.Int).SetString(lit, 10)
		if !ok {
			return lit
		}
		return n.String()
	}
	f, err := strconv.ParseFloat(lit, 64)
	if err != nil && !math.IsInf(f, 0) {
		return lit
	}
	return pyFloatRepr(f)
}

// numberInt returns the int value of a JSON number literal, as Python int() would.
func numberInt(lit string) (int, bool) {
	if isIntLiteral(lit) {
		n, err := strconv.Atoi(lit)
		return n, err == nil
	}
	f, err := strconv.ParseFloat(lit, 64)
	if err != nil || math.IsNaN(f) || math.IsInf(f, 0) {
		return 0, false
	}
	return int(math.Trunc(f)), true
}

// valueString returns v when it is a JSON string.
func valueString(v Value) (string, bool) {
	s, ok := v.(string)
	return s, ok
}

// truthy is Python's bool(v) for a decoded JSON value.
func truthy(v Value) bool {
	switch x := v.(type) {
	case nil:
		return false
	case bool:
		return x
	case Number:
		f, err := strconv.ParseFloat(string(x), 64)
		return err != nil || f != 0
	case string:
		return x != ""
	case []Value:
		return len(x) > 0
	case *Object:
		return len(x.Keys) > 0
	}
	return true
}

// iterValue is Python iteration over a value: list items, string characters,
// or object keys. Anything else iterates as empty (a falsy value under `or []`).
func iterValue(v Value) []Value {
	switch x := v.(type) {
	case []Value:
		return x
	case string:
		out := make([]Value, 0, len(x))
		for _, r := range x {
			out = append(out, string(r))
		}
		return out
	case *Object:
		out := make([]Value, 0, len(x.Keys))
		for _, k := range x.Keys {
			out = append(out, k)
		}
		return out
	}
	return nil
}

// valueEqual is Python == between two decoded JSON values.
func valueEqual(a, b Value) bool {
	switch x := a.(type) {
	case nil:
		return b == nil
	case bool:
		switch y := b.(type) {
		case bool:
			return x == y
		case Number:
			return numberEqualsBool(y, x)
		}
		return false
	case string:
		y, ok := b.(string)
		return ok && x == y
	case Number:
		switch y := b.(type) {
		case Number:
			fx, e1 := strconv.ParseFloat(string(x), 64)
			fy, e2 := strconv.ParseFloat(string(y), 64)
			return e1 == nil && e2 == nil && fx == fy
		case bool:
			return numberEqualsBool(x, y)
		}
		return false
	case []Value:
		y, ok := b.([]Value)
		if !ok || len(x) != len(y) {
			return false
		}
		for i := range x {
			if !valueEqual(x[i], y[i]) {
				return false
			}
		}
		return true
	case *Object:
		y, ok := b.(*Object)
		if !ok || len(x.Keys) != len(y.Keys) {
			return false
		}
		for i, k := range x.Keys {
			yv, ok := y.Get(k)
			if !ok || !valueEqual(x.Vals[i], yv) {
				return false
			}
		}
		return true
	}
	return false
}

func numberEqualsBool(n Number, b bool) bool {
	f, err := strconv.ParseFloat(string(n), 64)
	if err != nil {
		return false
	}
	if b {
		return f == 1
	}
	return f == 0
}
