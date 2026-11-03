package agent

// Bounded AST execution and marked-string composition adapted from llama.cpp
// common/jinja/runtime.cpp and string.cpp at cd8427d018c07eefab6c110f7f115c31ec4c6157.
// All mutable scopes/namespace values belong to one render; the program is immutable.

import (
	"context"
	"encoding/json"
	"io"
	"math"
	"strconv"
	"strings"
	"unicode/utf8"

	"github.com/anthony-chaudhary/fak/internal/tokenizer"
)

const (
	jinjaMaxWork         = 2000000
	jinjaMaxInput        = 2 << 20
	jinjaMaxOutput       = 1 << 20
	jinjaMaxMaterialized = 16 << 20
	jinjaMaxItems        = 65536
)

type jinjaValue struct {
	kind    string
	parts   []tokenizer.Segment
	i       int64
	f       float64
	b       bool
	tainted bool
	array   []*jinjaValue
	object  *jinjaObject
	macro   *jinjaNode
	closure *jinjaScope
}
type jinjaObject struct {
	keys      []string
	values    map[string]*jinjaValue
	namespace bool
}
type jinjaScope struct {
	vars   map[string]*jinjaValue
	parent *jinjaScope
}

func (s *jinjaScope) get(name string) *jinjaValue {
	for ; s != nil; s = s.parent {
		if v, ok := s.vars[name]; ok {
			return v
		}
	}
	return &jinjaValue{kind: "undefined"}
}

type jinjaAbort struct{ err error }
type jinjaState struct {
	ctx                                   context.Context
	work, bytes, items, depth, macroDepth int
}

func (s *jinjaState) fail(reason string, pos int) {
	panic(jinjaAbort{jinjaFail("render", reason, pos)})
}
func (s *jinjaState) step(pos int) {
	s.work++
	if s.work > jinjaMaxWork {
		s.fail("work-budget", pos)
	}
	if s.ctx != nil {
		if err := s.ctx.Err(); err != nil {
			panic(jinjaAbort{err})
		}
	}
}
func (s *jinjaState) charge(bytes, items, pos int) {
	s.step(pos)
	if bytes < 0 || items < 0 || bytes > jinjaMaxMaterialized-s.bytes || items > jinjaMaxItems-s.items {
		s.fail("materialization-budget", pos)
	}
	s.bytes += bytes
	s.items += items
}
func (s *jinjaState) value(kind string) *jinjaValue {
	s.charge(0, 1, 0)
	return &jinjaValue{kind: kind}
}
func (s *jinjaState) text(text string, trusted bool) *jinjaValue {
	if !utf8.ValidString(text) {
		s.fail("invalid-utf8", 0)
	}
	s.charge(len(text), 1, 0)
	return &jinjaValue{kind: "string", parts: []tokenizer.Segment{{Text: text, Trusted: trusted}}, tainted: !trusted}
}
func (s *jinjaState) integer(n int64, taint bool) *jinjaValue {
	v := s.value("int")
	v.i = n
	v.tainted = taint
	return v
}
func (s *jinjaState) boolean(b bool, taint bool) *jinjaValue {
	v := s.value("bool")
	v.b = b
	v.tainted = taint
	return v
}
func (s *jinjaState) number(n float64, taint bool) *jinjaValue {
	if math.IsNaN(n) || math.IsInf(n, 0) {
		s.fail("nonfinite-number", 0)
	}
	v := s.value("float")
	v.f = n
	v.tainted = taint
	return v
}
func (s *jinjaState) obj(namespace bool) *jinjaValue {
	v := s.value("object")
	v.object = &jinjaObject{values: make(map[string]*jinjaValue), namespace: namespace}
	return v
}
func (s *jinjaState) setObject(o *jinjaObject, k string, v *jinjaValue, pos int) {
	if _, ok := o.values[k]; !ok {
		s.charge(len(k), 1, pos)
		o.keys = append(o.keys, k)
	}
	o.values[k] = v
}
func jinjaString(v *jinjaValue) string {
	var b strings.Builder
	for _, p := range v.parts {
		b.WriteString(p.Text)
	}
	return b.String()
}
func jinjaTruth(v *jinjaValue) bool {
	if v == nil {
		return false
	}
	switch v.kind {
	case "undefined", "none":
		return false
	case "bool":
		return v.b
	case "int":
		return v.i != 0
	case "float":
		return v.f != 0
	case "string":
		for _, p := range v.parts {
			if p.Text != "" {
				return true
			}
		}
		return false
	case "array":
		return len(v.array) > 0
	case "object":
		return len(v.object.keys) > 0
	default:
		return true
	}
}
func (s *jinjaState) asText(v *jinjaValue, pos int) *jinjaValue {
	s.step(pos)
	switch v.kind {
	case "string":
		return v
	case "undefined":
		return s.text("", true)
	case "none":
		return s.text("None", !v.tainted)
	case "bool":
		if v.b {
			return s.text("True", !v.tainted)
		}
		return s.text("False", !v.tainted)
	case "int":
		return s.text(strconv.FormatInt(v.i, 10), !v.tainted)
	case "float":
		return s.text(jinjaFloatText(v.f), !v.tainted)
	default:
		s.fail("string-type", pos)
		return nil
	}
}
func (s *jinjaState) appendParts(dst []tokenizer.Segment, src []tokenizer.Segment, pos int) []tokenizer.Segment {
	for _, p := range src {
		if p.Text == "" {
			continue
		}
		s.charge(len(p.Text), 1, pos)
		dst = append(dst, p)
	}
	return dst
}
func (s *jinjaState) concat(a, b *jinjaValue, pos int) *jinjaValue {
	a = s.asText(a, pos)
	b = s.asText(b, pos)
	v := s.value("string")
	v.parts = s.appendParts(v.parts, a.parts, pos)
	v.parts = s.appendParts(v.parts, b.parts, pos)
	v.tainted = a.tainted || b.tainted
	return v
}
func (s *jinjaState) flatten(v *jinjaValue, pos int) string {
	v = s.asText(v, pos)
	n := 0
	for _, p := range v.parts {
		if len(p.Text) > jinjaMaxMaterialized-n {
			s.fail("string-budget", pos)
		}
		n += len(p.Text)
	}
	s.charge(n, 0, pos)
	return jinjaString(v)
}
func (s *jinjaState) length(v *jinjaValue, pos int) int {
	switch v.kind {
	case "string":
		return utf8.RuneCountInString(s.flatten(v, pos))
	case "array":
		return len(v.array)
	case "object":
		return len(v.object.keys)
	case "undefined":
		return 0
	default:
		s.fail("length-type", pos)
		return 0
	}
}
func (s *jinjaState) at(v, k *jinjaValue, pos int) *jinjaValue {
	s.step(pos)
	if v.kind == "object" {
		if k.kind != "string" {
			s.fail("object-key", pos)
		}
		if x, ok := v.object.values[s.flatten(k, pos)]; ok {
			return x
		}
		return s.value("undefined")
	}
	if k.kind != "int" {
		s.fail("index-type", pos)
	}
	n := s.length(v, pos)
	i := k.i
	if i < 0 {
		i += int64(n)
	}
	if i < 0 || i >= int64(n) {
		return s.value("undefined")
	}
	if v.kind == "array" {
		return v.array[i]
	}
	if v.kind == "string" {
		return s.sliceText(v, int(i), int(i)+1, pos)
	}
	s.fail("index-type", pos)
	return nil
}
func (s *jinjaState) sliceText(v *jinjaValue, start, end, pos int) *jinjaValue {
	out := s.value("string")
	index := 0
	for _, part := range v.parts {
		s.charge(len(part.Text)*4, 0, pos)
		r := []rune(part.Text)
		lo, hi := start-index, end-index
		if lo < 0 {
			lo = 0
		}
		if hi > len(r) {
			hi = len(r)
		}
		if lo < hi && lo < len(r) && hi > 0 {
			q := tokenizer.Segment{Text: string(r[lo:hi]), Trusted: part.Trusted}
			out.parts = s.appendParts(out.parts, []tokenizer.Segment{q}, pos)
			out.tainted = out.tainted || !q.Trusted
		}
		index += len(r)
	}
	return out
}
func (s *jinjaState) equal(a, b *jinjaValue, pos int) bool {
	s.step(pos)
	s.depth++
	defer func() { s.depth-- }()
	if s.depth > jinjaMaxDepth {
		s.fail("depth", pos)
	}
	if (a.kind == "int" || a.kind == "float") && (b.kind == "int" || b.kind == "float") {
		if a.kind == "int" && b.kind == "int" {
			return a.i == b.i
		}
		if jinjaUnsafeMixedComparison(a, b) {
			s.fail("mixed-number-comparison", pos)
		}
		return jinjaFloat(a) == jinjaFloat(b)
	}
	if a.kind != b.kind {
		return false
	}
	switch a.kind {
	case "none", "undefined":
		return true
	case "string":
		return s.flatten(a, pos) == s.flatten(b, pos)
	case "bool":
		return a.b == b.b
	case "array":
		if len(a.array) != len(b.array) {
			return false
		}
		for i := range a.array {
			if !s.equal(a.array[i], b.array[i], pos) {
				return false
			}
		}
		return true
	case "object":
		if len(a.object.keys) != len(b.object.keys) {
			return false
		}
		for _, k := range a.object.keys {
			v, ok := b.object.values[k]
			if !ok || !s.equal(a.object.values[k], v, pos) {
				return false
			}
		}
		return true
	default:
		return a == b
	}
}
func jinjaUnsafeMixedComparison(a, b *jinjaValue) bool {
	const exact int64 = 1 << 53
	return a.kind == "int" && b.kind == "float" && (a.i > exact || a.i < -exact) || b.kind == "int" && a.kind == "float" && (b.i > exact || b.i < -exact)
}
func jinjaFloat(v *jinjaValue) float64 {
	if v.kind == "int" {
		return float64(v.i)
	}
	return v.f
}
func jinjaFloatText(v float64) string {
	raw := strconv.FormatFloat(v, 'g', -1, 64)
	if !strings.ContainsAny(raw, ".eE") {
		raw += ".0"
	}
	return raw
}
func (s *jinjaState) binary(op string, a, b *jinjaValue, pos int) *jinjaValue {
	taint := a.tainted || b.tainted
	if op == "~" {
		return s.concat(a, b, pos)
	}
	if op == "==" || op == "!=" {
		x := s.equal(a, b, pos)
		if op == "!=" {
			x = !x
		}
		return s.boolean(x, taint)
	}
	if op == "in" || op == "not in" {
		found := false
		switch b.kind {
		case "array":
			for _, x := range b.array {
				if s.equal(a, x, pos) {
					found = true
					break
				}
			}
		case "object":
			if a.kind != "string" {
				s.fail("membership-type", pos)
			}
			_, found = b.object.values[s.flatten(a, pos)]
		case "string":
			if a.kind != "string" {
				s.fail("membership-type", pos)
			}
			found = strings.Contains(s.flatten(b, pos), s.flatten(a, pos))
		default:
			s.fail("membership-type", pos)
		}
		if op == "not in" {
			found = !found
		}
		return s.boolean(found, taint)
	}
	if op == "+" && a.kind == "string" && b.kind == "string" {
		return s.concat(a, b, pos)
	}
	if op == "+" && a.kind == "array" && b.kind == "array" {
		s.charge(0, len(a.array)+len(b.array), pos)
		v := s.value("array")
		v.array = append(append([]*jinjaValue(nil), a.array...), b.array...)
		v.tainted = taint
		return v
	}
	if (op == "<" || op == ">" || op == "<=" || op == ">=") && a.kind == "string" && b.kind == "string" {
		x, y := s.flatten(a, pos), s.flatten(b, pos)
		r := false
		switch op {
		case "<":
			r = x < y
		case ">":
			r = x > y
		case "<=":
			r = x <= y
		case ">=":
			r = x >= y
		}
		return s.boolean(r, taint)
	}
	if op == "*" && (a.kind == "string" || a.kind == "array") && b.kind == "int" {
		if b.i < 0 {
			if a.kind == "array" {
				return s.value("array")
			}
			return s.text("", !taint)
		}
		if b.i > jinjaMaxItems {
			s.fail("repeat-budget", pos)
		}
		if a.kind == "string" {
			v := s.value("string")
			for n := int64(0); n < b.i; n++ {
				v.parts = s.appendParts(v.parts, a.parts, pos)
			}
			v.tainted = taint
			return v
		}
		if b.i > 0 && int64(len(a.array)) > int64(jinjaMaxItems)/b.i {
			s.fail("repeat-budget", pos)
		}
		s.charge(0, len(a.array)*int(b.i), pos)
		v := s.value("array")
		for n := int64(0); n < b.i; n++ {
			v.array = append(v.array, a.array...)
		}
		v.tainted = taint
		return v
	}
	if (a.kind != "int" && a.kind != "float") || (b.kind != "int" && b.kind != "float") {
		s.fail("binary-type", pos)
	}
	if a.kind == "int" && b.kind == "int" {
		switch op {
		case "<":
			return s.boolean(a.i < b.i, taint)
		case ">":
			return s.boolean(a.i > b.i, taint)
		case "<=":
			return s.boolean(a.i <= b.i, taint)
		case ">=":
			return s.boolean(a.i >= b.i, taint)
		}
	}
	if (op == "<" || op == ">" || op == "<=" || op == ">=") && jinjaUnsafeMixedComparison(a, b) {
		s.fail("mixed-number-comparison", pos)
	}
	x, y := jinjaFloat(a), jinjaFloat(b)
	switch op {
	case "<":
		return s.boolean(x < y, taint)
	case ">":
		return s.boolean(x > y, taint)
	case "<=":
		return s.boolean(x <= y, taint)
	case ">=":
		return s.boolean(x >= y, taint)
	}
	if (op == "/" || op == "//" || op == "%") && y == 0 {
		s.fail("division-zero", pos)
	}
	if a.kind == "int" && b.kind == "int" && op != "/" {
		var r int64
		switch op {
		case "+":
			r = a.i + b.i
			if (b.i > 0 && r < a.i) || (b.i < 0 && r > a.i) {
				s.fail("integer-overflow", pos)
			}
		case "-":
			if b.i == math.MinInt64 {
				s.fail("integer-overflow", pos)
			}
			r = a.i - b.i
			if (b.i > 0 && r > a.i) || (b.i < 0 && r < a.i) {
				s.fail("integer-overflow", pos)
			}
		case "*":
			if (a.i == math.MinInt64 && b.i == -1) || (b.i == math.MinInt64 && a.i == -1) {
				s.fail("integer-overflow", pos)
			}
			r = a.i * b.i
			if b.i != 0 && r/b.i != a.i {
				s.fail("integer-overflow", pos)
			}
		case "//", "%":
			if a.i == math.MinInt64 && b.i == -1 {
				s.fail("integer-overflow", pos)
			}
			q, rem := a.i/b.i, a.i%b.i
			if rem != 0 && (rem < 0) != (b.i < 0) {
				q--
				rem += b.i
			}
			r = q
			if op == "%" {
				r = rem
			}
		default:
			s.fail("operator", pos)
		}
		return s.integer(r, taint)
	}
	var r float64
	switch op {
	case "+":
		r = x + y
	case "-":
		r = x - y
	case "*":
		r = x * y
	case "/":
		r = x / y
	case "//":
		r = math.Floor(x / y)
	case "%":
		r = x - math.Floor(x/y)*y
	default:
		s.fail("operator", pos)
	}
	return s.number(r, taint)
}
func (s *jinjaState) eval(e *jinjaExpr, scope *jinjaScope) *jinjaValue {
	if e == nil {
		return s.value("undefined")
	}
	s.step(e.pos)
	s.depth++
	defer func() { s.depth-- }()
	if s.depth > jinjaMaxDepth {
		s.fail("depth", e.pos)
	}
	switch e.kind {
	case "string":
		return s.text(e.text, true)
	case "number":
		if strings.Contains(e.text, ".") {
			n, err := strconv.ParseFloat(e.text, 64)
			if err != nil {
				s.fail("number", e.pos)
			}
			return s.number(n, false)
		}
		n, err := strconv.ParseInt(e.text, 10, 64)
		if err != nil {
			s.fail("number", e.pos)
		}
		return s.integer(n, false)
	case "id":
		switch e.text {
		case "true", "True":
			return s.boolean(true, false)
		case "false", "False":
			return s.boolean(false, false)
		case "none", "None", "null":
			return s.value("none")
		}
		return scope.get(e.text)
	case "array":
		v := s.value("array")
		for _, x := range e.args {
			item := s.eval(x, scope)
			s.charge(0, 1, e.pos)
			v.array = append(v.array, item)
			v.tainted = v.tainted || item.tainted
		}
		return v
	case "object":
		v := s.obj(false)
		for i := 0; i < len(e.args); i += 2 {
			k := s.eval(e.args[i], scope)
			if k.kind != "string" {
				s.fail("object-key", e.pos)
			}
			item := s.eval(e.args[i+1], scope)
			key := s.flatten(k, e.pos)
			if _, ok := v.object.values[key]; ok {
				s.fail("duplicate-key", e.pos)
			}
			s.setObject(v.object, key, item, e.pos)
			v.tainted = v.tainted || k.tainted || item.tainted
		}
		return v
	case "attr":
		v := s.eval(e.args[0], scope)
		if v.kind != "object" {
			s.fail("attribute-type", e.pos)
		}
		if x, ok := v.object.values[e.text]; ok {
			return x
		}
		return s.value("undefined")
	case "index":
		return s.at(s.eval(e.args[0], scope), s.eval(e.args[1], scope), e.pos)
	case "slice":
		v := s.eval(e.args[0], scope)
		if v.kind != "string" && v.kind != "array" {
			s.fail("slice-type", e.pos)
		}
		n := s.length(v, e.pos)
		step := int64(1)
		if e.args[3] != nil {
			x := s.eval(e.args[3], scope)
			if x.kind != "int" {
				s.fail("slice-step", e.pos)
			}
			step = x.i
		}
		if step == 0 {
			s.fail("slice-step", e.pos)
		}
		bound := func(x *jinjaExpr, def int64) int64 {
			if x == nil {
				return def
			}
			v := s.eval(x, scope)
			if v.kind != "int" {
				s.fail("slice-bound", e.pos)
			}
			i := v.i
			if i < 0 {
				i += int64(n)
			}
			if step > 0 {
				if i < 0 {
					i = 0
				}
				if i > int64(n) {
					i = int64(n)
				}
			} else {
				if i < 0 {
					i = -1
				}
				if i >= int64(n) {
					i = int64(n) - 1
				}
			}
			return i
		}
		start, end := int64(0), int64(n)
		if step < 0 {
			start = int64(n) - 1
			end = -1
		}
		start = bound(e.args[1], start)
		end = bound(e.args[2], end)
		if v.kind == "string" && step == 1 {
			if end < start {
				end = start
			}
			return s.sliceText(v, int(start), int(end), e.pos)
		}
		out := s.value(v.kind)
		out.tainted = v.tainted
		for i := start; (step > 0 && i < end) || (step < 0 && i > end); {
			s.step(e.pos)
			if v.kind == "array" {
				s.charge(0, 1, e.pos)
				out.array = append(out.array, v.array[i])
			} else if v.kind == "string" {
				part := s.sliceText(v, int(i), int(i)+1, e.pos)
				out.parts = s.appendParts(out.parts, part.parts, e.pos)
			} else {
				s.fail("slice-type", e.pos)
			}
			if (step > 0 && i > math.MaxInt64-step) || (step < 0 && i < math.MinInt64-step) {
				break
			}
			i += step
		}
		return out
	case "unary":
		if e.text == "-" && e.args[0].kind == "number" && e.args[0].text == "9223372036854775808" {
			return s.integer(math.MinInt64, false)
		}
		v := s.eval(e.args[0], scope)
		if e.text == "not" {
			return s.boolean(!jinjaTruth(v), v.tainted)
		}
		if v.kind != "int" && v.kind != "float" {
			s.fail("unary-type", e.pos)
		}
		if e.text == "+" {
			return v
		}
		if v.kind == "int" {
			if v.i == math.MinInt64 {
				s.fail("integer-overflow", e.pos)
			}
			return s.integer(-v.i, v.tainted)
		}
		return s.number(-v.f, v.tainted)
	case "binary":
		a := s.eval(e.args[0], scope)
		if e.text == "and" && !jinjaTruth(a) || e.text == "or" && jinjaTruth(a) {
			return a
		}
		b := s.eval(e.args[1], scope)
		if e.text == "and" || e.text == "or" {
			return b
		}
		return s.binary(e.text, a, b, e.pos)
	case "ternary":
		if jinjaTruth(s.eval(e.args[1], scope)) {
			return s.eval(e.args[0], scope)
		}
		return s.eval(e.args[2], scope)
	case "select":
		s.fail("select-outside-loop", e.pos)
	case "test", "filter":
		args := make([]*jinjaValue, 0, len(e.args))
		for _, x := range e.args {
			args = append(args, s.eval(x, scope))
		}
		if e.kind == "test" {
			for _, name := range e.names {
				if name != "" {
					s.fail("test-keyword", e.pos)
				}
			}
			return s.test(e.text, args, e.pos)
		}
		return s.builtin(e.text, args, append([]string{""}, e.names...), e.pos)
	case "call":
		callee := e.args[0]
		var args []*jinjaValue
		names := append([]string(nil), e.names...)
		name := ""
		if callee.kind == "id" {
			name = callee.text
		} else if callee.kind == "attr" {
			name = callee.text
			args = append(args, s.eval(callee.args[0], scope))
			names = append([]string{""}, names...)
		} else {
			s.fail("call-target", e.pos)
		}
		for _, x := range e.args[1:] {
			args = append(args, s.eval(x, scope))
		}
		if callee.kind == "id" {
			v := scope.get(name)
			if v.kind == "macro" {
				return s.callMacro(v, args, names, e.pos)
			}
			if v.kind != "undefined" {
				s.fail("not-callable", e.pos)
			}
		}
		return s.builtin(name, args, names, e.pos)
	}
	s.fail("expression-kind", e.pos)
	return nil
}

func (s *jinjaState) callMacro(v *jinjaValue, args []*jinjaValue, names []string, pos int) *jinjaValue {
	s.macroDepth++
	defer func() { s.macroDepth-- }()
	if s.macroDepth > 32 {
		s.fail("macro-depth", pos)
	}
	n := v.macro
	scope := &jinjaScope{vars: map[string]*jinjaValue{}, parent: v.closure}
	assigned := map[string]bool{}
	for i, arg := range args {
		name := names[i]
		if name == "" {
			if i >= len(n.names) {
				s.fail("macro-arguments", pos)
			}
			name = n.names[i]
		}
		found := false
		for _, x := range n.names {
			if x == name {
				found = true
				break
			}
		}
		if !found || assigned[name] {
			s.fail("macro-arguments", pos)
		}
		assigned[name] = true
		scope.vars[name] = arg
	}
	for i, name := range n.names {
		if !assigned[name] {
			if n.exprs[i] == nil {
				s.fail("macro-missing-argument", pos)
			}
			scope.vars[name] = s.eval(n.exprs[i], v.closure)
		}
	}
	out, flow := s.execute(n.body, scope)
	if flow != "" {
		s.fail("macro-loop-control", pos)
	}
	return out
}
func (s *jinjaState) execute(nodes []*jinjaNode, scope *jinjaScope) (*jinjaValue, string) {
	s.depth++
	defer func() { s.depth-- }()
	if s.depth > jinjaMaxDepth {
		s.fail("depth", 0)
	}
	out := s.value("string")
	appendValue := func(v *jinjaValue, pos int) {
		v = s.asText(v, pos)
		out.parts = s.appendParts(out.parts, v.parts, pos)
		out.tainted = out.tainted || v.tainted
	}
	for _, n := range nodes {
		s.step(n.pos)
		switch n.kind {
		case "text":
			appendValue(s.text(n.text, true), n.pos)
		case "expr":
			appendValue(s.eval(n.exprs[0], scope), n.pos)
		case "if":
			body := n.other
			if jinjaTruth(s.eval(n.exprs[0], scope)) {
				body = n.body
			}
			v, flow := s.execute(body, scope)
			appendValue(v, n.pos)
			if flow != "" {
				return out, flow
			}
		case "generation":
			v, flow := s.execute(n.body, scope)
			appendValue(v, n.pos)
			if flow != "" {
				return out, flow
			}
		case "set":
			var v *jinjaValue
			if len(n.exprs) == 2 {
				v = s.eval(n.exprs[1], scope)
			} else {
				var flow string
				v, flow = s.execute(n.body, scope)
				if flow != "" {
					s.fail("set-loop-control", n.pos)
				}
			}
			target := n.exprs[0]
			if target.kind == "id" {
				scope.vars[target.text] = v
			} else {
				o := s.eval(target.args[0], scope)
				if o.kind != "object" || !o.object.namespace {
					s.fail("assignment-requires-namespace", n.pos)
				}
				s.setObject(o.object, target.text, v, n.pos)
			}
		case "macro":
			v := s.value("macro")
			v.macro = n
			v.closure = scope
			scope.vars[n.text] = v
		case "break", "continue":
			return out, n.kind
		case "for":
			expr := n.exprs[0]
			var filter *jinjaExpr
			if expr.kind == "select" {
				filter = expr.args[1]
				expr = expr.args[0]
			}
			iterable := s.eval(expr, scope)
			var items []*jinjaValue
			switch iterable.kind {
			case "array":
				items = iterable.array
			case "object":
				for _, k := range iterable.object.keys {
					s.charge(0, 1, n.pos)
					items = append(items, s.text(k, !iterable.tainted))
				}
			case "string":
				count := s.length(iterable, n.pos)
				for i := 0; i < count; i++ {
					s.charge(0, 1, n.pos)
					items = append(items, s.sliceText(iterable, i, i+1, n.pos))
				}
			case "undefined":
			default:
				s.fail("loop-type", n.pos)
			}
			bind := func(child *jinjaScope, v *jinjaValue) {
				if len(n.names) == 1 {
					child.vars[n.names[0]] = v
					return
				}
				if v.kind != "array" || len(v.array) != len(n.names) {
					s.fail("loop-unpack", n.pos)
				}
				for i, name := range n.names {
					child.vars[name] = v.array[i]
				}
			}
			var chosen []*jinjaValue
			for _, v := range items {
				s.step(n.pos)
				child := &jinjaScope{vars: map[string]*jinjaValue{}, parent: scope}
				bind(child, v)
				if filter == nil || jinjaTruth(s.eval(filter, child)) {
					s.charge(0, 1, n.pos)
					chosen = append(chosen, v)
				}
			}
			if len(chosen) == 0 {
				v, flow := s.execute(n.other, scope)
				appendValue(v, n.pos)
				if flow != "" {
					return out, flow
				}
				continue
			}
			for i, v := range chosen {
				child := &jinjaScope{vars: map[string]*jinjaValue{}, parent: scope}
				bind(child, v)
				loop := s.obj(false)
				for _, field := range []struct {
					name  string
					value int64
				}{{"index", int64(i + 1)}, {"index0", int64(i)}, {"revindex", int64(len(chosen) - i)}, {"revindex0", int64(len(chosen) - i - 1)}, {"length", int64(len(chosen))}} {
					s.setObject(loop.object, field.name, s.integer(field.value, true), n.pos)
				}
				s.setObject(loop.object, "first", s.boolean(i == 0, true), n.pos)
				s.setObject(loop.object, "last", s.boolean(i == len(chosen)-1, true), n.pos)
				if i > 0 {
					s.setObject(loop.object, "previtem", chosen[i-1], n.pos)
				}
				if i+1 < len(chosen) {
					s.setObject(loop.object, "nextitem", chosen[i+1], n.pos)
				}
				child.vars["loop"] = loop
				part, flow := s.execute(n.body, child)
				appendValue(part, n.pos)
				if flow == "break" {
					break
				}
			}
		default:
			s.fail("statement-kind", n.pos)
		}
	}
	return out, ""
}

func (p *nativeJinjaProgram) render(ctx context.Context, build func(*jinjaState) *jinjaScope) (parts []tokenizer.Segment, err error) {
	defer func() {
		if r := recover(); r != nil {
			if a, ok := r.(jinjaAbort); ok {
				parts = nil
				err = a.err
			} else {
				panic(r)
			}
		}
	}()
	if p == nil {
		return nil, jinjaFail("render", "missing-program", 0)
	}
	s := &jinjaState{ctx: ctx}
	scope := build(s)
	out, flow := s.execute(p.nodes, scope)
	if flow != "" {
		s.fail("loop-control", 0)
	}
	total := 0
	for _, x := range out.parts {
		if len(x.Text) > jinjaMaxOutput-total {
			s.fail("output-budget", 0)
		}
		total += len(x.Text)
	}
	s.step(0)
	return append([]tokenizer.Segment(nil), out.parts...), nil
}

// Ordered JSON conversion is bounded before recursively materializing values.
// Every input scalar and key is tainted; rendering never evaluates its text.
func (s *jinjaState) fromJSON(raw []byte, pos int) *jinjaValue {
	if len(raw) > jinjaMaxInput || !utf8.Valid(raw) {
		s.fail("input-json", pos)
	}
	s.charge(len(raw), 0, pos)
	dec := json.NewDecoder(strings.NewReader(string(raw)))
	dec.UseNumber()
	var read func(int) *jinjaValue
	read = func(depth int) *jinjaValue {
		if depth > jinjaMaxDepth {
			s.fail("input-depth", pos)
		}
		s.step(pos)
		token, e := dec.Token()
		if e != nil {
			s.fail("input-json", pos)
		}
		switch x := token.(type) {
		case nil:
			v := s.value("none")
			v.tainted = true
			return v
		case bool:
			return s.boolean(x, true)
		case string:
			return s.text(x, false)
		case json.Number:
			if !strings.ContainsAny(string(x), ".eE") {
				i, e := strconv.ParseInt(string(x), 10, 64)
				if e != nil {
					s.fail("input-integer", pos)
				}
				return s.integer(i, true)
			}
			f, e := strconv.ParseFloat(string(x), 64)
			if e != nil {
				s.fail("input-number", pos)
			}
			return s.number(f, true)
		case json.Delim:
			if x == '[' {
				v := s.value("array")
				v.tainted = true
				for dec.More() {
					s.charge(0, 1, pos)
					v.array = append(v.array, read(depth+1))
				}
				end, e := dec.Token()
				if e != nil || end != json.Delim(']') {
					s.fail("input-json", pos)
				}
				return v
			}
			if x == '{' {
				v := s.obj(false)
				v.tainted = true
				for dec.More() {
					key, e := dec.Token()
					if e != nil {
						s.fail("input-json", pos)
					}
					k, ok := key.(string)
					if !ok {
						s.fail("input-key", pos)
					}
					if _, ok := v.object.values[k]; ok {
						s.fail("input-duplicate-key", pos)
					}
					s.setObject(v.object, k, read(depth+1), pos)
				}
				end, e := dec.Token()
				if e != nil || end != json.Delim('}') {
					s.fail("input-json", pos)
				}
				return v
			}
		}
		s.fail("input-json", pos)
		return nil
	}
	v := read(0)
	if dec.More() {
		s.fail("input-json", pos)
	}
	var extra any
	if e := dec.Decode(&extra); e != io.EOF {
		s.fail("input-json-trailing", pos)
	}
	return v
}
