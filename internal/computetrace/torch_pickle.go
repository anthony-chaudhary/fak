package computetrace

import (
	"archive/zip"
	"encoding/binary"
	"errors"
	"fmt"
	"io"
	"math"
	"strings"
)

// ErrTorchPickle denotes an unsupported or malformed offline torch.save archive.
var ErrTorchPickle = errors.New("computetrace: invalid Torch pickle")

const torchMaxBytes = 64 << 20
const torchMaxItems = 1 << 20
const torchMaxElements = torchMaxBytes / 4

// TorchTensor is a CPU materialization in logical row-major order. Int64 storage
// retains integer values without the precision loss of an intermediate float.
type TorchTensor struct {
	Shape  []int
	Values []float32
	IDs    []int64
	kind   string
}

type torchGlobal string
type torchTuple []any
type torchList struct{ values []any }
type torchDict map[string]any
type torchMark struct{}
type torchStorage struct {
	kind, key string
	size      int64
}

func torchErr(format string, args ...any) error {
	return fmt.Errorf("%w: "+format, append([]any{ErrTorchPickle}, args...)...)
}

// ReadTorchPickle reads protocol-2 tensor/container pickles without executing
// Python. All globals, reductions and storage reads are explicitly allowlisted.
func ReadTorchPickle(path string) (any, error) {
	z, err := zip.OpenReader(path)
	if err != nil {
		return nil, torchErr("zip: %v", err)
	}
	defer z.Close()
	files := map[string]*zip.File{}
	var prefix string
	var total uint64
	for _, f := range z.File {
		if _, ok := files[f.Name]; ok {
			return nil, torchErr("duplicate record")
		}
		files[f.Name] = f
		if f.UncompressedSize64 > torchMaxBytes || total > torchMaxBytes-f.UncompressedSize64 {
			return nil, torchErr("archive exceeds limit")
		}
		total += f.UncompressedSize64
		if strings.HasSuffix(f.Name, "/data.pkl") {
			if prefix != "" {
				return nil, torchErr("multiple pickles")
			}
			prefix = strings.TrimSuffix(f.Name, "data.pkl")
		}
	}
	if prefix == "" {
		return nil, torchErr("missing data.pkl")
	}
	readBudget := torchMaxBytes
	read := func(name string) ([]byte, error) {
		f := files[prefix+name]
		if f == nil {
			return nil, torchErr("missing record %s", name)
		}
		if f.UncompressedSize64 > uint64(readBudget) {
			return nil, torchErr("record read budget")
		}
		readBudget -= int(f.UncompressedSize64)
		r, e := f.Open()
		if e != nil {
			return nil, torchErr("record: %v", e)
		}
		defer r.Close()
		b, e := io.ReadAll(io.LimitReader(r, torchMaxBytes+1))
		if e != nil || len(b) > torchMaxBytes {
			return nil, torchErr("record size/read")
		}
		return b, nil
	}
	order, err := read("byteorder")
	if err != nil {
		return nil, err
	}
	if string(order) != "little" {
		return nil, torchErr("unsupported byteorder")
	}
	data, err := read("data.pkl")
	if err != nil {
		return nil, err
	}
	p := torchParser{data: data, memo: map[uint32]any{}, read: read}
	value, err := p.parse()
	if err != nil {
		return nil, err
	}
	budget := torchMaxItems
	return torchPlain(value, 0, &budget)
}

type torchParser struct {
	data     []byte
	pos      int
	stack    []any
	memo     map[uint32]any
	read     func(string) ([]byte, error)
	elements int
}

func (p *torchParser) take(n int) ([]byte, error) {
	if n < 0 || n > len(p.data)-p.pos {
		return nil, torchErr("truncated operand")
	}
	b := p.data[p.pos : p.pos+n]
	p.pos += n
	return b, nil
}
func (p *torchParser) line() (string, error) {
	i := strings.IndexByte(string(p.data[p.pos:]), '\n')
	if i < 0 {
		return "", torchErr("unterminated global")
	}
	b, e := p.take(i + 1)
	return string(b[:i]), e
}
func (p *torchParser) pop() (any, error) {
	if len(p.stack) == 0 {
		return nil, torchErr("empty stack")
	}
	i := len(p.stack) - 1
	v := p.stack[i]
	p.stack = p.stack[:i]
	return v, nil
}
func (p *torchParser) marked() ([]any, error) {
	for i := len(p.stack) - 1; i >= 0; i-- {
		if _, ok := p.stack[i].(torchMark); ok {
			v := append([]any(nil), p.stack[i+1:]...)
			p.stack = p.stack[:i]
			return v, nil
		}
	}
	return nil, torchErr("missing MARK")
}
func (p *torchParser) parse() (any, error) {
	header, e := p.take(2)
	if e != nil || len(header) != 2 || header[0] != 0x80 || header[1] != 2 {
		return nil, torchErr("protocol must be 2")
	}
	for ops := 0; ops < torchMaxItems; ops++ {
		b, e := p.take(1)
		if e != nil {
			return nil, e
		}
		op := b[0]
		var v any
		push := true
		switch op {
		case '.':
			if len(p.stack) != 1 || p.pos != len(p.data) {
				return nil, torchErr("invalid STOP")
			}
			return p.stack[0], nil
		case '(':
			v = torchMark{}
		case 'N':
			v = nil
		case 0x88:
			v = true
		case 0x89:
			v = false
		case 'K':
			b, e = p.take(1)
			if e == nil {
				v = int64(b[0])
			}
		case 'M':
			b, e = p.take(2)
			if e == nil {
				v = int64(binary.LittleEndian.Uint16(b))
			}
		case 'J':
			b, e = p.take(4)
			if e == nil {
				v = int64(int32(binary.LittleEndian.Uint32(b)))
			}
		case 0x8a:
			b, e = p.take(1)
			if e != nil {
				return nil, e
			}
			n := int(b[0])
			if n > 8 {
				return nil, torchErr("LONG overflow")
			}
			b, e = p.take(n)
			if e == nil {
				var n64 uint64
				for i, c := range b {
					n64 |= uint64(c) << (8 * i)
				}
				if n > 0 && b[n-1]&0x80 != 0 && n < 8 {
					n64 |= ^uint64(0) << (8 * n)
				}
				v = int64(n64)
			}
		case 'G':
			b, e = p.take(8)
			if e == nil {
				v = math.Float64frombits(binary.BigEndian.Uint64(b))
			}
		case 'X':
			b, e = p.take(4)
			if e != nil {
				return nil, e
			}
			n := binary.LittleEndian.Uint32(b)
			if n > torchMaxBytes {
				return nil, torchErr("string limit")
			}
			b, e = p.take(int(n))
			if e == nil {
				v = string(b)
			}
		case 'c':
			var module, name string
			module, e = p.line()
			if e == nil {
				name, e = p.line()
			}
			g := module + "." + name
			switch g {
			case "torch._utils._rebuild_tensor_v2", "collections.OrderedDict", "torch.FloatStorage", "torch.BFloat16Storage", "torch.HalfStorage", "torch.LongStorage":
				v = torchGlobal(g)
			default:
				return nil, torchErr("global not allowed: %s", g)
			}
		case ')':
			v = torchTuple{}
		case 't':
			var vals []any
			vals, e = p.marked()
			v = torchTuple(vals)
		case 0x85, 0x86, 0x87:
			n := int(op - 0x84)
			if len(p.stack) < n {
				return nil, torchErr("tuple stack")
			}
			v = torchTuple(append([]any(nil), p.stack[len(p.stack)-n:]...))
			p.stack = p.stack[:len(p.stack)-n]
		case ']':
			v = &torchList{}
		case '}':
			v = torchDict{}
		case 'q', 'r', 'h', 'j':
			n := 1
			if op == 'r' || op == 'j' {
				n = 4
			}
			b, e = p.take(n)
			if e != nil {
				return nil, e
			}
			idx := uint32(b[0])
			if n == 4 {
				idx = binary.LittleEndian.Uint32(b)
			}
			if op == 'q' || op == 'r' {
				if len(p.stack) == 0 || len(p.memo) >= torchMaxItems {
					return nil, torchErr("memo limit/stack")
				}
				p.memo[idx] = p.stack[len(p.stack)-1]
				push = false
			} else {
				var ok bool
				v, ok = p.memo[idx]
				if !ok {
					return nil, torchErr("unknown memo")
				}
			}
		case 'a', 'e', 's', 'u':
			var vals []any
			if op == 'e' || op == 'u' {
				vals, e = p.marked()
			} else {
				n := 1
				if op == 's' {
					n = 2
				}
				if len(p.stack) < n {
					return nil, torchErr("container stack")
				}
				vals = append([]any(nil), p.stack[len(p.stack)-n:]...)
				p.stack = p.stack[:len(p.stack)-n]
			}
			if e != nil {
				return nil, e
			}
			if len(p.stack) == 0 {
				return nil, torchErr("missing container")
			}
			target := p.stack[len(p.stack)-1]
			if op == 'a' || op == 'e' {
				list, ok := target.(*torchList)
				if !ok || len(list.values)+len(vals) > torchMaxItems {
					return nil, torchErr("invalid list")
				}
				list.values = append(list.values, vals...)
			} else {
				dict, ok := target.(torchDict)
				if !ok || len(vals)%2 != 0 {
					return nil, torchErr("invalid dictionary")
				}
				for i := 0; i < len(vals); i += 2 {
					key, ok := vals[i].(string)
					if !ok {
						return nil, torchErr("dictionary key")
					}
					dict[key] = vals[i+1]
				}
			}
			push = false
		case 'Q':
			var id any
			id, e = p.pop()
			if e != nil {
				return nil, e
			}
			t, ok := id.(torchTuple)
			if !ok || len(t) != 5 {
				return nil, torchErr("storage id")
			}
			tag, ok := t[0].(string)
			g, gok := t[1].(torchGlobal)
			key, kok := t[2].(string)
			_, lok := t[3].(string)
			size, sok := t[4].(int64)
			if !ok || tag != "storage" || !gok || !kok || !lok || !sok || size < 0 || size > torchMaxBytes || strings.ContainsAny(key, "/\\.") {
				return nil, torchErr("invalid storage id")
			}
			v = torchStorage{string(g), key, size}
		case 'R':
			args, er := p.pop()
			if er != nil {
				return nil, er
			}
			fn, er := p.pop()
			if er != nil {
				return nil, er
			}
			g, ok := fn.(torchGlobal)
			t, tok := args.(torchTuple)
			if !ok || !tok {
				return nil, torchErr("invalid reduction")
			}
			switch g {
			case "collections.OrderedDict":
				if len(t) != 0 {
					return nil, torchErr("OrderedDict arguments")
				}
				v = torchDict{}
			case "torch._utils._rebuild_tensor_v2":
				v, e = p.tensor(t)
			default:
				return nil, torchErr("reduction not allowed")
			}
		default:
			return nil, torchErr("opcode not allowed: 0x%02x", op)
		}
		if e != nil {
			return nil, e
		}
		if push {
			if len(p.stack) >= torchMaxItems {
				return nil, torchErr("stack limit")
			}
			p.stack = append(p.stack, v)
		}
	}
	return nil, torchErr("opcode limit")
}

func torchDimensions(v any) ([]int, error) {
	t, ok := v.(torchTuple)
	if !ok || len(t) > 32 {
		return nil, torchErr("invalid shape")
	}
	out := make([]int, len(t))
	for i, x := range t {
		n, ok := x.(int64)
		if !ok || n < 0 || n > torchMaxBytes {
			return nil, torchErr("invalid dimension")
		}
		out[i] = int(n)
	}
	return out, nil
}

func (p *torchParser) tensor(args torchTuple) (any, error) {
	if len(args) != 6 {
		return nil, torchErr("tensor arguments")
	}
	storage, ok := args[0].(torchStorage)
	offset, ook := args[1].(int64)
	_, bok := args[4].(bool)
	hooks, hok := args[5].(torchDict)
	if !ok || !ook || offset < 0 || !bok || !hok || len(hooks) != 0 {
		return nil, torchErr("tensor metadata")
	}
	shape, e := torchDimensions(args[2])
	if e != nil {
		return nil, e
	}
	stride, e := torchDimensions(args[3])
	if e != nil || len(shape) != len(stride) {
		return nil, torchErr("invalid stride")
	}
	count := 1
	for _, n := range shape {
		if n != 0 && count > torchMaxElements/n {
			return nil, torchErr("tensor element limit")
		}
		count *= n
	}
	if count > torchMaxElements-p.elements {
		return nil, torchErr("materialized element limit")
	}
	p.elements += count
	width := 4
	switch storage.kind {
	case "torch.FloatStorage":
	case "torch.HalfStorage", "torch.BFloat16Storage":
		width = 2
	case "torch.LongStorage":
		width = 8
	default:
		return nil, torchErr("storage type")
	}
	raw, e := p.read("data/" + storage.key)
	if e != nil {
		return nil, e
	}
	if storage.size > torchMaxBytes/int64(width) || int64(len(raw)) != storage.size*int64(width) {
		return nil, torchErr("storage size")
	}
	result := TorchTensor{Shape: shape, kind: storage.kind}
	if width == 8 {
		result.IDs = make([]int64, count)
	} else {
		result.Values = make([]float32, count)
	}
	for i := 0; i < count; i++ {
		index := offset
		rem := i
		for d := len(shape) - 1; d >= 0; d-- {
			coordinate := rem % shape[d]
			rem /= shape[d]
			delta := int64(coordinate) * int64(stride[d])
			if delta > math.MaxInt64-index {
				return nil, torchErr("stride overflow")
			}
			index += delta
		}
		if index < 0 || index >= storage.size {
			return nil, torchErr("view exceeds storage")
		}
		b := raw[int(index)*width:]
		switch storage.kind {
		case "torch.FloatStorage":
			result.Values[i] = math.Float32frombits(binary.LittleEndian.Uint32(b))
		case "torch.BFloat16Storage":
			result.Values[i] = math.Float32frombits(uint32(binary.LittleEndian.Uint16(b)) << 16)
		case "torch.HalfStorage":
			result.Values[i] = torchHalf(binary.LittleEndian.Uint16(b))
		case "torch.LongStorage":
			result.IDs[i] = int64(binary.LittleEndian.Uint64(b))
		}
	}
	return result, nil
}

func torchHalf(h uint16) float32 {
	sign := uint32(h&0x8000) << 16
	exp := int((h >> 10) & 31)
	fraction := uint32(h & 1023)
	if exp == 31 {
		return math.Float32frombits(sign | 0x7f800000 | fraction<<13)
	}
	if exp == 0 {
		if fraction == 0 {
			return math.Float32frombits(sign)
		}
		exp = 1
		for fraction&1024 == 0 {
			fraction <<= 1
			exp--
		}
		fraction &= 1023
	}
	return math.Float32frombits(sign | uint32(exp+112)<<23 | fraction<<13)
}

func torchPlain(v any, depth int, budget *int) (any, error) {
	*budget--
	if depth > 64 || *budget < 0 {
		return nil, torchErr("container depth/cycle")
	}
	switch x := v.(type) {
	case torchDict:
		if len(x) > *budget {
			return nil, torchErr("container budget")
		}
		out := map[string]any{}
		for k, value := range x {
			p, e := torchPlain(value, depth+1, budget)
			if e != nil {
				return nil, e
			}
			out[k] = p
		}
		return out, nil
	case *torchList:
		return torchPlain(torchTuple(x.values), depth, budget)
	case torchTuple:
		if len(x) > *budget {
			return nil, torchErr("container budget")
		}
		out := make([]any, len(x))
		for i, value := range x {
			p, e := torchPlain(value, depth+1, budget)
			if e != nil {
				return nil, e
			}
			out[i] = p
		}
		return out, nil
	case nil, bool, string, int64, float64, TorchTensor:
		return v, nil
	default:
		return nil, torchErr("unfinished pickle object")
	}
}
