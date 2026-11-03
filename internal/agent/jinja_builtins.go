package agent

import (
	"math"
	"sort"
	"strconv"
	"strings"
	"unicode/utf8"

	"github.com/anthony-chaudhary/fak/internal/tokenizer"
)

// Built-ins are a closed set: no filesystem, process, clock, imports, reflection,
// network or user-provided function is available. "safe" never changes provenance.
func (s *jinjaState) bind(args []*jinjaValue, names, allowed []string, min, max, pos int) []*jinjaValue {
	if len(args) != len(names) || len(args) < min || len(args) > max {
		s.fail("builtin-arity", pos)
	}
	out := make([]*jinjaValue, max)
	for i := range out {
		out[i] = &jinjaValue{kind: "undefined"}
	}
	used := make([]bool, max)
	for i, v := range args {
		j := i
		if names[i] != "" {
			j = -1
			for k, n := range allowed {
				if n == names[i] {
					j = k
					break
				}
			}
		}
		if j < 0 || j >= max || used[j] {
			s.fail("builtin-keyword", pos)
		}
		used[j] = true
		out[j] = v
	}
	for i := 0; i < min; i++ {
		if !used[i] {
			s.fail("builtin-argument", pos)
		}
	}
	return out
}
func (s *jinjaState) anyTaint(v *jinjaValue, pos int) bool {
	s.step(pos)
	s.depth++
	defer func() { s.depth-- }()
	if s.depth > jinjaMaxDepth {
		s.fail("value-depth", pos)
	}
	if v.tainted {
		return true
	}
	for _, p := range v.parts {
		if !p.Trusted {
			return true
		}
	}
	for _, x := range v.array {
		if s.anyTaint(x, pos) {
			return true
		}
	}
	if v.object != nil {
		for _, k := range v.object.keys {
			if s.anyTaint(v.object.values[k], pos) {
				return true
			}
		}
	}
	return false
}
func (s *jinjaState) test(name string, args []*jinjaValue, pos int) *jinjaValue {
	if len(args) < 1 || len(args) > 2 {
		s.fail("test-arity", pos)
	}
	v := args[0]
	r := false
	switch name {
	case "defined":
		r = v.kind != "undefined"
	case "undefined":
		r = v.kind == "undefined"
	case "none":
		r = v.kind == "none"
	case "boolean":
		r = v.kind == "bool"
	case "string":
		r = v.kind == "string"
	case "number":
		r = v.kind == "int" || v.kind == "float"
	case "integer":
		r = v.kind == "int"
	case "float":
		r = v.kind == "float"
	case "mapping":
		r = v.kind == "object"
	case "iterable", "sequence":
		r = v.kind == "array" || v.kind == "object" || v.kind == "string"
	case "true":
		r = v.kind == "bool" && v.b
	case "false":
		r = v.kind == "bool" && !v.b
	case "odd", "even":
		if v.kind != "int" {
			s.fail("test-type", pos)
		}
		r = v.i%2 != 0
		if name == "even" {
			r = !r
		}
	case "equalto", "eq":
		if len(args) != 2 {
			s.fail("test-arity", pos)
		}
		r = s.equal(v, args[1], pos)
	case "sameas":
		if len(args) != 2 {
			s.fail("test-arity", pos)
		}
		if v.kind == "none" || v.kind == "bool" {
			r = s.equal(v, args[1], pos)
		} else {
			s.fail("unsupported-identity-test", pos)
		}
	default:
		s.fail("unsupported-test", pos)
	}
	if len(args) != 1 && name != "equalto" && name != "eq" && name != "sameas" {
		s.fail("test-arity", pos)
	}
	return s.boolean(r, v.tainted)
}
func (s *jinjaState) builtin(name string, args []*jinjaValue, names []string, pos int) *jinjaValue {
	s.step(pos)
	switch name {
	case "raise_exception":
		a := s.bind(args, names, []string{"message"}, 1, 1, pos)
		if a[0].kind != "string" {
			s.fail("raise-type", pos)
		}
		s.fail("template-refused", pos)
	case "namespace", "dict":
		v := s.obj(name == "namespace")
		for i, arg := range args {
			if names[i] != "" {
				s.setObject(v.object, names[i], arg, pos)
				continue
			}
			if i != 0 || arg.kind != "object" {
				s.fail("object-constructor", pos)
			}
			for _, k := range arg.object.keys {
				s.setObject(v.object, k, arg.object.values[k], pos)
			}
			v.tainted = v.tainted || arg.tainted
		}
		return v
	case "range":
		if len(args) < 1 || len(args) > 3 {
			s.fail("range-arity", pos)
		}
		for i, a := range args {
			if names[i] != "" || a.kind != "int" {
				s.fail("range-type", pos)
			}
		}
		start, stop, step := int64(0), args[0].i, int64(1)
		if len(args) >= 2 {
			start, stop = args[0].i, args[1].i
		}
		if len(args) == 3 {
			step = args[2].i
		}
		if step == 0 {
			s.fail("range-step", pos)
		}
		v := s.value("array")
		for i := start; (step > 0 && i < stop) || (step < 0 && i > stop); {
			s.charge(0, 1, pos)
			v.array = append(v.array, s.integer(i, true))
			if (step > 0 && i > math.MaxInt64-step) || (step < 0 && i < math.MinInt64-step) {
				s.fail("range-overflow", pos)
			}
			i += step
		}
		return v
	case "default", "d":
		a := s.bind(args, names, []string{"value", "default_value", "boolean"}, 1, 3, pos)
		if a[2].kind != "undefined" && a[2].kind != "bool" {
			s.fail("default-type", pos)
		}
		if a[0].kind == "undefined" || (jinjaTruth(a[2]) && !jinjaTruth(a[0])) {
			if a[1].kind == "undefined" {
				return s.text("", true)
			}
			return a[1]
		}
		return a[0]
	case "length", "count":
		a := s.bind(args, names, []string{"value"}, 1, 1, pos)
		return s.integer(int64(s.length(a[0], pos)), a[0].tainted)
	case "string", "safe":
		a := s.bind(args, names, []string{"value"}, 1, 1, pos)
		return s.asText(a[0], pos)
	case "int", "float":
		a := s.bind(args, names, []string{"value"}, 1, 1, pos)
		v := a[0]
		if v.kind == "int" {
			if name == "int" {
				return v
			}
			return s.number(float64(v.i), v.tainted)
		}
		if v.kind == "float" {
			if name == "float" {
				return v
			}
			if v.f >= float64(math.MaxInt64) || v.f < float64(math.MinInt64) {
				s.fail("integer-overflow", pos)
			}
			return s.integer(int64(v.f), v.tainted)
		}
		if v.kind != "string" {
			s.fail("number-conversion", pos)
		}
		raw := strings.TrimSpace(s.flatten(v, pos))
		if name == "int" {
			n, e := strconv.ParseInt(raw, 10, 64)
			if e != nil {
				s.fail("number-conversion", pos)
			}
			return s.integer(n, v.tainted)
		}
		f, e := strconv.ParseFloat(raw, 64)
		if e != nil {
			s.fail("number-conversion", pos)
		}
		return s.number(f, v.tainted)
	case "lower", "upper":
		a := s.bind(args, names, []string{"value"}, 1, 1, pos)
		if a[0].kind != "string" {
			s.fail("string-type", pos)
		}
		v := s.value("string")
		for _, p := range a[0].parts {
			text := p.Text
			if name == "lower" {
				text = strings.ToLower(text)
			} else {
				text = strings.ToUpper(text)
			}
			v.parts = s.appendParts(v.parts, []tokenizer.Segment{{Text: text, Trusted: p.Trusted}}, pos)
		}
		v.tainted = a[0].tainted
		return v
	case "trim", "strip", "lstrip", "rstrip":
		a := s.bind(args, names, []string{"value", "chars"}, 1, 2, pos)
		if a[0].kind != "string" {
			s.fail("string-type", pos)
		}
		chars := " \t\r\n\v\f"
		if a[1].kind != "undefined" && a[1].kind != "none" {
			if a[1].kind != "string" {
				s.fail("strip-chars", pos)
			}
			chars = s.flatten(a[1], pos)
		}
		raw := s.flatten(a[0], pos)
		left := raw
		if name != "rstrip" {
			left = strings.TrimLeft(raw, chars)
		}
		right := left
		if name != "lstrip" {
			right = strings.TrimRight(left, chars)
		}
		start := utf8.RuneCountInString(raw[:len(raw)-len(left)])
		return s.sliceText(a[0], start, start+utf8.RuneCountInString(right), pos)
	case "replace":
		a := s.bind(args, names, []string{"value", "old", "new", "count"}, 3, 4, pos)
		for _, v := range a[:3] {
			if v.kind != "string" {
				s.fail("replace-type", pos)
			}
		}
		if a[3].kind != "undefined" && (a[3].kind != "int" || a[3].i != -1) {
			s.fail("unsupported-replace-count", pos)
		}
		raw, old, new := s.flatten(a[0], pos), s.flatten(a[1], pos), s.flatten(a[2], pos)
		n := strings.Count(raw, old)
		size := int64(len(raw)) + int64(n)*int64(len(new)-len(old))
		if size < 0 || size > int64(jinjaMaxMaterialized-s.bytes) {
			s.fail("string-budget", pos)
		}
		s.charge(int(size), 0, pos)
		return s.text(strings.ReplaceAll(raw, old, new), !(s.anyTaint(a[0], pos) || s.anyTaint(a[1], pos) || s.anyTaint(a[2], pos)))
	case "split":
		a := s.bind(args, names, []string{"value", "sep", "maxsplit"}, 2, 3, pos)
		if a[0].kind != "string" || a[1].kind != "string" {
			s.fail("split-type", pos)
		}
		raw, sep := s.flatten(a[0], pos), s.flatten(a[1], pos)
		if sep == "" {
			s.fail("split-separator", pos)
		}
		max := jinjaMaxItems
		limited := false
		if a[2].kind != "undefined" {
			if a[2].kind != "int" || a[2].i < -1 || a[2].i >= jinjaMaxItems {
				s.fail("split-count", pos)
			}
			if a[2].i >= 0 {
				max = int(a[2].i) + 1
				limited = true
			}
		}
		count := strings.Count(raw, sep) + 1
		if count > max {
			if !limited {
				s.fail("split-budget", pos)
			}
			count = max
		}
		s.charge(0, count, pos)
		v := s.value("array")
		v.tainted = s.anyTaint(a[0], pos) || s.anyTaint(a[1], pos)
		for _, part := range strings.SplitN(raw, sep, max) {
			v.array = append(v.array, s.text(part, !v.tainted))
		}
		return v
	case "join":
		a := s.bind(args, names, []string{"value", "d", "attribute"}, 1, 3, pos)
		if a[0].kind != "array" {
			s.fail("join-type", pos)
		}
		sep := a[1]
		if sep.kind == "undefined" {
			sep = s.text("", true)
		}
		if sep.kind != "string" {
			s.fail("join-separator", pos)
		}
		out := s.value("string")
		for i, v := range a[0].array {
			if a[2].kind != "undefined" {
				v = s.at(v, a[2], pos)
			}
			if i > 0 {
				out = s.concat(out, sep, pos)
			}
			out = s.concat(out, v, pos)
		}
		return out
	case "first", "last":
		a := s.bind(args, names, []string{"value"}, 1, 1, pos)
		n := s.length(a[0], pos)
		if n == 0 {
			return s.value("undefined")
		}
		i := int64(0)
		if name == "last" {
			i = int64(n - 1)
		}
		return s.at(a[0], s.integer(i, false), pos)
	case "list", "reverse":
		a := s.bind(args, names, []string{"value"}, 1, 1, pos)
		v := s.value("array")
		v.tainted = a[0].tainted
		switch a[0].kind {
		case "array":
			s.charge(0, len(a[0].array), pos)
			v.array = append(v.array, a[0].array...)
		case "object":
			for _, k := range a[0].object.keys {
				s.charge(0, 1, pos)
				v.array = append(v.array, s.text(k, !v.tainted))
			}
		case "string":
			n := s.length(a[0], pos)
			for i := 0; i < n; i++ {
				s.charge(0, 1, pos)
				v.array = append(v.array, s.sliceText(a[0], i, i+1, pos))
			}
		case "undefined":
		default:
			s.fail("list-type", pos)
		}
		if name == "reverse" {
			for i, j := 0, len(v.array)-1; i < j; i, j = i+1, j-1 {
				v.array[i], v.array[j] = v.array[j], v.array[i]
			}
		}
		return v
	case "items", "keys", "values", "dictsort":
		a := s.bind(args, names, []string{"value"}, 1, 1, pos)
		if a[0].kind != "object" || a[0].object.namespace {
			s.fail("object-method", pos)
		}
		keys := append([]string(nil), a[0].object.keys...)
		s.charge(0, len(keys), pos)
		if name == "dictsort" {
			sort.Strings(keys)
		}
		v := s.value("array")
		v.tainted = a[0].tainted
		for _, k := range keys {
			key := s.text(k, !v.tainted)
			value := a[0].object.values[k]
			item := value
			if name == "keys" {
				item = key
			} else if name == "items" || name == "dictsort" {
				item = s.value("array")
				s.charge(0, 2, pos)
				item.array = []*jinjaValue{key, value}
				item.tainted = v.tainted
			}
			s.charge(0, 1, pos)
			v.array = append(v.array, item)
		}
		return v
	case "get":
		a := s.bind(args, names, []string{"value", "key", "default"}, 2, 3, pos)
		if a[0].kind != "object" || a[1].kind != "string" {
			s.fail("get-type", pos)
		}
		if v, ok := a[0].object.values[s.flatten(a[1], pos)]; ok {
			return v
		}
		if a[2].kind == "undefined" {
			return s.value("none")
		}
		return a[2]
	case "startswith", "endswith":
		a := s.bind(args, names, []string{"value", "prefix"}, 2, 2, pos)
		if a[0].kind != "string" || a[1].kind != "string" {
			s.fail("string-type", pos)
		}
		x, y := s.flatten(a[0], pos), s.flatten(a[1], pos)
		r := strings.HasPrefix(x, y)
		if name == "endswith" {
			r = strings.HasSuffix(x, y)
		}
		return s.boolean(r, a[0].tainted || a[1].tainted)
	case "selectattr", "rejectattr", "select", "reject":
		var a []*jinjaValue
		var attr, test, arg *jinjaValue
		if name == "select" || name == "reject" {
			a = s.bind(args, names, []string{"value", "test", "argument"}, 1, 3, pos)
			attr = s.value("undefined")
			test = a[1]
			arg = a[2]
		} else {
			a = s.bind(args, names, []string{"value", "attribute", "test", "argument"}, 2, 4, pos)
			attr = a[1]
			test = a[2]
			arg = a[3]
		}
		if a[0].kind != "array" {
			s.fail("select-type", pos)
		}
		if attr.kind != "undefined" && attr.kind != "string" {
			s.fail("select-attribute", pos)
		}
		if test.kind != "undefined" && test.kind != "string" {
			s.fail("select-test", pos)
		}
		out := s.value("array")
		out.tainted = a[0].tainted
		for _, v := range a[0].array {
			x := v
			if attr.kind != "undefined" {
				x = s.at(v, attr, pos)
			}
			yes := jinjaTruth(x)
			if test.kind != "undefined" {
				args := []*jinjaValue{x}
				if arg.kind != "undefined" {
					args = append(args, arg)
				}
				yes = s.test(s.flatten(test, pos), args, pos).b
			}
			if name == "reject" || name == "rejectattr" {
				yes = !yes
			}
			if yes {
				s.charge(0, 1, pos)
				out.array = append(out.array, v)
			}
		}
		return out
	case "map":
		a := s.bind(args, names, []string{"value", "attribute", "default"}, 1, 3, pos)
		if a[0].kind != "array" || a[1].kind != "string" {
			s.fail("map-attribute", pos)
		}
		out := s.value("array")
		out.tainted = a[0].tainted
		for _, v := range a[0].array {
			x := s.at(v, a[1], pos)
			if x.kind == "undefined" && a[2].kind != "undefined" {
				x = a[2]
			}
			s.charge(0, 1, pos)
			out.array = append(out.array, x)
		}
		return out
	case "tojson":
		a := s.bind(args, names, []string{"value", "ensure_ascii", "indent", "separators", "sort_keys"}, 1, 5, pos)
		if a[1].kind != "undefined" && a[1].kind != "bool" {
			s.fail("tojson-ascii", pos)
		}
		if a[4].kind != "undefined" && (a[4].kind != "bool" || a[4].b) {
			s.fail("unsupported-json-sort", pos)
		}
		indent := -1
		if a[2].kind != "undefined" && a[2].kind != "none" {
			if a[2].kind != "int" || a[2].i < 0 || a[2].i > 8 {
				s.fail("tojson-indent", pos)
			}
			indent = int(a[2].i)
		}
		item, key := ", ", ": "
		if indent >= 0 {
			item = ","
		}
		if a[3].kind != "undefined" {
			if a[3].kind != "array" || len(a[3].array) != 2 || a[3].array[0].kind != "string" || a[3].array[1].kind != "string" {
				s.fail("tojson-separators", pos)
			}
			item = s.flatten(a[3].array[0], pos)
			key = s.flatten(a[3].array[1], pos)
			if len(item) > 16 || len(key) > 16 {
				s.fail("tojson-separators", pos)
			}
		}
		text := s.jsonText(a[0], indent, item, key, jinjaTruth(a[1]), pos)
		taint := s.anyTaint(a[0], pos)
		if a[3].kind != "undefined" {
			taint = taint || s.anyTaint(a[3], pos)
		}
		return s.text(text, !taint)
	}
	s.fail("unsupported-builtin", pos)
	return nil
}
func (s *jinjaState) jsonText(v *jinjaValue, indent int, item, key string, ascii bool, pos int) string {
	var b strings.Builder
	write := func(x string) {
		s.charge(len(x), 0, pos)
		if len(x) > jinjaMaxOutput-b.Len() {
			s.fail("json-output", pos)
		}
		b.WriteString(x)
	}
	quote := func(x string) string {
		// Reserve the complete worst-case JSON escape allocation before encoding.
		if len(x) > (jinjaMaxMaterialized-s.bytes-2)/6 {
			s.fail("json-string-budget", pos)
		}
		s.charge(len(x)*6+2, 0, pos)
		var out strings.Builder
		out.WriteByte('"')
		for _, r := range x {
			switch r {
			case '"':
				out.WriteString("\\\"")
			case '\\':
				out.WriteString("\\\\")
			case '\b':
				out.WriteString("\\b")
			case '\f':
				out.WriteString("\\f")
			case '\n':
				out.WriteString("\\n")
			case '\r':
				out.WriteString("\\r")
			case '\t':
				out.WriteString("\\t")
			default:
				if r < 32 {
					out.WriteString("\\u")
					out.WriteString(jinjaHex4(uint16(r)))
				} else if ascii && r > 127 {
					if r <= 0xffff {
						out.WriteString("\\u")
						out.WriteString(jinjaHex4(uint16(r)))
					} else {
						u := uint32(r) - 0x10000
						out.WriteString("\\u")
						out.WriteString(jinjaHex4(uint16(0xd800 + (u >> 10))))
						out.WriteString("\\u")
						out.WriteString(jinjaHex4(uint16(0xdc00 + (u & 1023))))
					}
				} else {
					out.WriteRune(r)
				}
			}
		}
		out.WriteByte('"')
		return out.String()
	}
	var emit func(*jinjaValue, int)
	emit = func(v *jinjaValue, depth int) {
		s.step(pos)
		if depth > jinjaMaxDepth {
			s.fail("json-depth", pos)
		}
		newline := func() {
			if indent >= 0 {
				write("\n" + strings.Repeat(" ", (depth+1)*indent))
			}
		}
		switch v.kind {
		case "none":
			write("null")
		case "bool":
			if v.b {
				write("true")
			} else {
				write("false")
			}
		case "int":
			write(strconv.FormatInt(v.i, 10))
		case "float":
			write(jinjaFloatText(v.f))
		case "string":
			write(quote(s.flatten(v, pos)))
		case "array":
			write("[")
			for i, x := range v.array {
				if i > 0 {
					write(item)
				}
				newline()
				emit(x, depth+1)
			}
			if len(v.array) > 0 && indent >= 0 {
				write("\n" + strings.Repeat(" ", depth*indent))
			}
			write("]")
		case "object":
			write("{")
			for i, k := range v.object.keys {
				if i > 0 {
					write(item)
				}
				newline()
				write(quote(k))
				write(key)
				emit(v.object.values[k], depth+1)
			}
			if len(v.object.keys) > 0 && indent >= 0 {
				write("\n" + strings.Repeat(" ", depth*indent))
			}
			write("}")
		default:
			s.fail("json-type", pos)
		}
	}
	emit(v, 0)
	return b.String()
}
func jinjaHex4(n uint16) string {
	const chars = "0123456789abcdef"
	return string([]byte{chars[(n>>12)&15], chars[(n>>8)&15], chars[(n>>4)&15], chars[n&15]})
}
