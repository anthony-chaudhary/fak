// export_json.go ports aiperf's non-finite scrub before JSON export to Go.
//
// NOTICE: ported from ai-dynamo/aiperf, src/aiperf/exporters/metrics_json_exporter.py
// (scrub_non_finite before orjson.dumps), pinned at commit
// ab610add4e8876fca223197596c728e17365f381 (ai-dynamo/aiperf@ab610ad),
// Apache-2.0 licensed. The source rewrites every NaN, +Inf and -Inf in a
// nested export payload to JSON null so serialization never fails and the
// file stays valid JSON. This port walks maps, slices, arrays, pointers and
// interfaces the same way and marshals the scrubbed tree with encoding/json.
package perfledger

import (
	"encoding"
	"encoding/json"
	"math"
	"reflect"
)

// ScrubNonFiniteJSON returns a copy of v in which every non-finite float32 or
// float64 reachable through maps, slices, arrays, pointers and interfaces is
// replaced by nil, which encoding/json renders as null. Finite values and
// other leaves are returned unchanged; v itself is never mutated. Structs and
// values implementing json.Marshaler or encoding.TextMarshaler are treated as
// opaque leaves, so payloads should be flattened to maps first,
// mirroring the upstream dump-then-scrub order.
func ScrubNonFiniteJSON(v any) any {
	if v == nil {
		return nil
	}
	return jscrubValue(reflect.ValueOf(v))
}

// MarshalScrubbedJSON scrubs v with ScrubNonFiniteJSON and encodes it as
// two-space indented JSON, matching the upstream export layout.
func MarshalScrubbedJSON(v any) ([]byte, error) {
	return json.MarshalIndent(ScrubNonFiniteJSON(v), "", "  ")
}

var (
	jscrubAnyType           = reflect.TypeOf((*any)(nil)).Elem()
	jscrubMarshalerType     = reflect.TypeOf((*json.Marshaler)(nil)).Elem()
	jscrubTextMarshalerType = reflect.TypeOf((*encoding.TextMarshaler)(nil)).Elem()
)

func jscrubValue(rv reflect.Value) any {
	if rv.Kind() == reflect.Invalid {
		return nil
	}
	// Rebuilding a custom marshaler as plain maps or slices, or dereferencing a
	// pointer-receiver one, would silently change its encoding.
	if rv.Kind() != reflect.Interface && (rv.Type().Implements(jscrubMarshalerType) || rv.Type().Implements(jscrubTextMarshalerType)) {
		return rv.Interface()
	}
	switch rv.Kind() {
	case reflect.Float32, reflect.Float64:
		f := rv.Float()
		if math.IsNaN(f) || math.IsInf(f, 0) {
			return nil
		}
		return rv.Interface()
	case reflect.Interface, reflect.Pointer:
		if rv.IsNil() {
			return nil
		}
		return jscrubValue(rv.Elem())
	case reflect.Map:
		if rv.IsNil() {
			return nil
		}
		// Values widen to any so a scrubbed nil fits even in map[string]float64.
		out := reflect.MakeMapWithSize(reflect.MapOf(rv.Type().Key(), jscrubAnyType), rv.Len())
		iter := rv.MapRange()
		for iter.Next() {
			sv := jscrubValue(iter.Value())
			ev := reflect.Zero(jscrubAnyType)
			if sv != nil {
				ev = reflect.ValueOf(sv)
			}
			out.SetMapIndex(iter.Key(), ev)
		}
		return out.Interface()
	case reflect.Slice:
		if rv.IsNil() {
			return nil
		}
		// []byte encodes as base64 and holds no floats; keep it intact.
		if rv.Type().Elem().Kind() == reflect.Uint8 {
			return rv.Interface()
		}
		return jscrubSequence(rv)
	case reflect.Array:
		return jscrubSequence(rv)
	default:
		return rv.Interface()
	}
}

func jscrubSequence(rv reflect.Value) []any {
	out := make([]any, rv.Len())
	for i := range out {
		out[i] = jscrubValue(rv.Index(i))
	}
	return out
}
