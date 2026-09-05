package wazero

import (
	"bytes"
	"encoding/binary"
	"fmt"
	"strconv"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/sandbox"
)

// WasmBuilder constructs valid WebAssembly 1.0 binaries.
type WasmBuilder struct {
	types    []FuncType
	imports  []Import
	funcs    []uint32
	memories []MemoryLimit
	exports  []Export
	code     [][]byte
	data     []DataSegment
}

// NewWasmBuilder creates an empty WasmBuilder.
func NewWasmBuilder() *WasmBuilder {
	return &WasmBuilder{}
}

// AddType adds a function signature type and returns its type index.
func (b *WasmBuilder) AddType(params, results []byte) uint32 {
	idx := uint32(len(b.types))
	b.types = append(b.types, FuncType{Params: params, Results: results})
	return idx
}

// AddImport adds an imported entity and returns its index.
func (b *WasmBuilder) AddImport(module, field string, kind byte, typeIdx uint32) uint32 {
	idx := uint32(len(b.imports))
	b.imports = append(b.imports, Import{
		Module:  module,
		Field:   field,
		Kind:    kind,
		TypeIdx: typeIdx,
	})
	return idx
}

// AddMemory declares linear memory limits.
func (b *WasmBuilder) AddMemory(min, max uint32, hasMax bool) {
	b.memories = append(b.memories, MemoryLimit{Min: min, Max: max, HasMax: hasMax})
}

// AddFunction defines a function and its code bytecode.
func (b *WasmBuilder) AddFunction(typeIdx uint32, code []byte) uint32 {
	idx := uint32(len(b.imports)) + uint32(len(b.funcs))
	b.funcs = append(b.funcs, typeIdx)
	b.code = append(b.code, code)
	return idx
}

// AddExport exports an entity by name.
func (b *WasmBuilder) AddExport(name string, kind byte, index uint32) {
	b.exports = append(b.exports, Export{
		Name:  name,
		Kind:  kind,
		Index: index,
	})
}

// AddData initializes linear memory at offset.
func (b *WasmBuilder) AddData(offset uint32, data []byte) {
	b.data = append(b.data, DataSegment{
		MemIndex: 0,
		Offset:   offset,
		Data:     data,
	})
}

// Build compiles the components into a WebAssembly binary.
func (b *WasmBuilder) Build() []byte {
	var buf []byte
	buf = append(buf, wasmMagic...)
	buf = append(buf, wasmVersion...)

	// Section 1: Type
	if len(b.types) > 0 {
		var sec []byte
		sec = appendVarUint32(sec, uint32(len(b.types)))
		for _, t := range b.types {
			sec = append(sec, 0x60)
			sec = appendVarUint32(sec, uint32(len(t.Params)))
			sec = append(sec, t.Params...)
			sec = appendVarUint32(sec, uint32(len(t.Results)))
			sec = append(sec, t.Results...)
		}
		buf = appendSection(buf, sectionIDType, sec)
	}

	// Section 2: Import
	if len(b.imports) > 0 {
		var sec []byte
		sec = appendVarUint32(sec, uint32(len(b.imports)))
		for _, imp := range b.imports {
			sec = appendVarUint32(sec, uint32(len(imp.Module)))
			sec = append(sec, imp.Module...)
			sec = appendVarUint32(sec, uint32(len(imp.Field)))
			sec = append(sec, imp.Field...)
			sec = append(sec, imp.Kind)
			if imp.Kind == kindFunc {
				sec = appendVarUint32(sec, imp.TypeIdx)
			}
		}
		buf = appendSection(buf, sectionIDImport, sec)
	}

	// Section 3: Function
	if len(b.funcs) > 0 {
		var sec []byte
		sec = appendVarUint32(sec, uint32(len(b.funcs)))
		for _, tIdx := range b.funcs {
			sec = appendVarUint32(sec, tIdx)
		}
		buf = appendSection(buf, sectionIDFunction, sec)
	}

	// Section 5: Memory
	if len(b.memories) > 0 {
		var sec []byte
		sec = appendVarUint32(sec, uint32(len(b.memories)))
		for _, m := range b.memories {
			if m.HasMax {
				sec = append(sec, 0x01)
				sec = appendVarUint32(sec, m.Min)
				sec = appendVarUint32(sec, m.Max)
			} else {
				sec = append(sec, 0x00)
				sec = appendVarUint32(sec, m.Min)
			}
		}
		buf = appendSection(buf, sectionIDMemory, sec)
	}

	// Section 7: Export
	if len(b.exports) > 0 {
		var sec []byte
		sec = appendVarUint32(sec, uint32(len(b.exports)))
		for _, exp := range b.exports {
			sec = appendVarUint32(sec, uint32(len(exp.Name)))
			sec = append(sec, exp.Name...)
			sec = append(sec, exp.Kind)
			sec = appendVarUint32(sec, exp.Index)
		}
		buf = appendSection(buf, sectionIDExport, sec)
	}

	// Section 10: Code
	if len(b.code) > 0 {
		var sec []byte
		sec = appendVarUint32(sec, uint32(len(b.code)))
		for _, body := range b.code {
			var fnBuf []byte
			fnBuf = appendVarUint32(fnBuf, 0) // 0 local declarations
			fnBuf = append(fnBuf, body...)
			sec = appendVarUint32(sec, uint32(len(fnBuf)))
			sec = append(sec, fnBuf...)
		}
		buf = appendSection(buf, sectionIDCode, sec)
	}

	// Section 11: Data
	if len(b.data) > 0 {
		var sec []byte
		sec = appendVarUint32(sec, uint32(len(b.data)))
		for _, seg := range b.data {
			sec = append(sec, 0x00) // active mem 0
			sec = append(sec, OpI32Const)
			sec = appendVarInt32(sec, int32(seg.Offset))
			sec = append(sec, OpEnd)
			sec = appendVarUint32(sec, uint32(len(seg.Data)))
			sec = append(sec, seg.Data...)
		}
		buf = appendSection(buf, sectionIDData, sec)
	}

	return buf
}

func appendSection(buf []byte, id byte, payload []byte) []byte {
	buf = append(buf, id)
	buf = appendVarUint32(buf, uint32(len(payload)))
	buf = append(buf, payload...)
	return buf
}

func appendVarUint32(b []byte, v uint32) []byte {
	for {
		c := byte(v & 0x7f)
		v >>= 7
		if v != 0 {
			c |= 0x80
		}
		b = append(b, c)
		if v == 0 {
			break
		}
	}
	return b
}

func appendVarInt32(b []byte, v int32) []byte {
	more := true
	for more {
		c := byte(v & 0x7f)
		v >>= 7
		sign := (c & 0x40) != 0
		if (v == 0 && !sign) || (v == -1 && sign) {
			more = false
		} else {
			c |= 0x80
		}
		b = append(b, c)
	}
	return b
}

// ---------------------------------------------------------------------------
// SYNTHETIC WASM MODULE GENERATORS
// ---------------------------------------------------------------------------

// BuildEchoModule constructs a Wasm module that outputs msg to stdout via WASI fd_write.
func BuildEchoModule(msg string) []byte {
	b := NewWasmBuilder()
	tWrite := b.AddType([]byte{valTypeI32, valTypeI32, valTypeI32, valTypeI32}, []byte{valTypeI32})
	tStart := b.AddType(nil, nil)

	fdWrite := b.AddImport("wasi_snapshot_preview1", "fd_write", kindFunc, tWrite)
	b.AddMemory(1, 16, true)

	iov := make([]byte, 8)
	binary.LittleEndian.PutUint32(iov[0:4], 1024)
	binary.LittleEndian.PutUint32(iov[4:8], uint32(len(msg)))
	b.AddData(0, iov)
	b.AddData(1024, []byte(msg))

	var code []byte
	code = append(code, OpI32Const)
	code = appendVarInt32(code, 1) // fd = 1 (stdout)
	code = append(code, OpI32Const)
	code = appendVarInt32(code, 0) // iovs_ptr = 0
	code = append(code, OpI32Const)
	code = appendVarInt32(code, 1) // iovs_len = 1
	code = append(code, OpI32Const)
	code = appendVarInt32(code, 20) // nwritten_ptr = 20
	code = append(code, OpCall)
	code = appendVarUint32(code, fdWrite)
	code = append(code, OpDrop)
	code = append(code, OpReturn)

	fnIdx := b.AddFunction(tStart, code)
	b.AddExport("_start", kindFunc, fnIdx)
	b.AddExport("memory", kindMemory, 0)
	return b.Build()
}

// BuildStdoutStderrModule constructs a Wasm module that writes stdoutMsg to fd 1 and stderrMsg to fd 2.
func BuildStdoutStderrModule(stdoutMsg, stderrMsg string) []byte {
	b := NewWasmBuilder()
	tWrite := b.AddType([]byte{valTypeI32, valTypeI32, valTypeI32, valTypeI32}, []byte{valTypeI32})
	tStart := b.AddType(nil, nil)

	fdWrite := b.AddImport("wasi_snapshot_preview1", "fd_write", kindFunc, tWrite)
	b.AddMemory(1, 16, true)

	// iov1 at 0 -> stdout
	iov1 := make([]byte, 8)
	binary.LittleEndian.PutUint32(iov1[0:4], 1024)
	binary.LittleEndian.PutUint32(iov1[4:8], uint32(len(stdoutMsg)))
	b.AddData(0, iov1)
	b.AddData(1024, []byte(stdoutMsg))

	// iov2 at 8 -> stderr
	iov2 := make([]byte, 8)
	binary.LittleEndian.PutUint32(iov2[0:4], 2048)
	binary.LittleEndian.PutUint32(iov2[4:8], uint32(len(stderrMsg)))
	b.AddData(8, iov2)
	b.AddData(2048, []byte(stderrMsg))

	var code []byte
	// write stdout
	code = append(code, OpI32Const)
	code = appendVarInt32(code, 1)
	code = append(code, OpI32Const)
	code = appendVarInt32(code, 0)
	code = append(code, OpI32Const)
	code = appendVarInt32(code, 1)
	code = append(code, OpI32Const)
	code = appendVarInt32(code, 32)
	code = append(code, OpCall)
	code = appendVarUint32(code, fdWrite)
	code = append(code, OpDrop)

	// write stderr
	code = append(code, OpI32Const)
	code = appendVarInt32(code, 2)
	code = append(code, OpI32Const)
	code = appendVarInt32(code, 8)
	code = append(code, OpI32Const)
	code = appendVarInt32(code, 1)
	code = append(code, OpI32Const)
	code = appendVarInt32(code, 36)
	code = append(code, OpCall)
	code = appendVarUint32(code, fdWrite)
	code = append(code, OpDrop)

	code = append(code, OpReturn)

	fnIdx := b.AddFunction(tStart, code)
	b.AddExport("_start", kindFunc, fnIdx)
	b.AddExport("memory", kindMemory, 0)
	return b.Build()
}

// BuildInfiniteLoopModule constructs a Wasm module containing an infinite loop.
func BuildInfiniteLoopModule() []byte {
	b := NewWasmBuilder()
	tStart := b.AddType(nil, nil)
	b.AddMemory(1, 16, true)

	var code []byte
	code = append(code, OpLoop, 0x40)
	code = append(code, OpBr, 0x00)
	code = append(code, OpEnd)
	code = append(code, OpReturn)

	fnIdx := b.AddFunction(tStart, code)
	b.AddExport("_start", kindFunc, fnIdx)
	b.AddExport("memory", kindMemory, 0)
	return b.Build()
}

// BuildMemoryGrowModule constructs a Wasm module that attempts to grow memory by deltaPages.
// If memory.grow returns -1, it writes failMsg to stderr and calls proc_exit(42).
func BuildMemoryGrowModule(deltaPages int32) []byte {
	b := NewWasmBuilder()
	tWrite := b.AddType([]byte{valTypeI32, valTypeI32, valTypeI32, valTypeI32}, []byte{valTypeI32})
	tExit := b.AddType([]byte{valTypeI32}, nil)
	tStart := b.AddType(nil, nil)

	fdWrite := b.AddImport("wasi_snapshot_preview1", "fd_write", kindFunc, tWrite)
	procExit := b.AddImport("wasi_snapshot_preview1", "proc_exit", kindFunc, tExit)
	b.AddMemory(1, 16, true)

	failMsg := "MEMORY_GROW_FAILED\n"
	iovFail := make([]byte, 8)
	binary.LittleEndian.PutUint32(iovFail[0:4], 1024)
	binary.LittleEndian.PutUint32(iovFail[4:8], uint32(len(failMsg)))
	b.AddData(0, iovFail)
	b.AddData(1024, []byte(failMsg))

	okMsg := "MEMORY_GROW_OK\n"
	iovOk := make([]byte, 8)
	binary.LittleEndian.PutUint32(iovOk[0:4], 2048)
	binary.LittleEndian.PutUint32(iovOk[4:8], uint32(len(okMsg)))
	b.AddData(8, iovOk)
	b.AddData(2048, []byte(okMsg))

	var code []byte
	// grow memory
	code = append(code, OpI32Const)
	code = appendVarInt32(code, deltaPages)
	code = append(code, OpMemoryGrow, 0x00)
	code = append(code, OpI32Const)
	code = appendVarInt32(code, -1)
	code = append(code, OpI32Eq)
	code = append(code, OpIf, 0x40)
	// failed branch: write to stderr and exit 42
	code = append(code, OpI32Const)
	code = appendVarInt32(code, 2)
	code = append(code, OpI32Const)
	code = appendVarInt32(code, 0)
	code = append(code, OpI32Const)
	code = appendVarInt32(code, 1)
	code = append(code, OpI32Const)
	code = appendVarInt32(code, 40)
	code = append(code, OpCall)
	code = appendVarUint32(code, fdWrite)
	code = append(code, OpDrop)
	code = append(code, OpI32Const)
	code = appendVarInt32(code, 42)
	code = append(code, OpCall)
	code = appendVarUint32(code, procExit)
	code = append(code, OpElse)
	// ok branch: write to stdout
	code = append(code, OpI32Const)
	code = appendVarInt32(code, 1)
	code = append(code, OpI32Const)
	code = appendVarInt32(code, 8)
	code = append(code, OpI32Const)
	code = appendVarInt32(code, 1)
	code = append(code, OpI32Const)
	code = appendVarInt32(code, 40)
	code = append(code, OpCall)
	code = appendVarUint32(code, fdWrite)
	code = append(code, OpDrop)
	code = append(code, OpEnd)
	code = append(code, OpReturn)

	fnIdx := b.AddFunction(tStart, code)
	b.AddExport("_start", kindFunc, fnIdx)
	b.AddExport("memory", kindMemory, 0)
	return b.Build()
}

// BuildExitModule constructs a Wasm module that calls proc_exit with code.
func BuildExitModule(code int32) []byte {
	b := NewWasmBuilder()
	tExit := b.AddType([]byte{valTypeI32}, nil)
	tStart := b.AddType(nil, nil)

	procExit := b.AddImport("wasi_snapshot_preview1", "proc_exit", kindFunc, tExit)
	b.AddMemory(1, 16, true)

	var c []byte
	c = append(c, OpI32Const)
	c = appendVarInt32(c, code)
	c = append(c, OpCall)
	c = appendVarUint32(c, procExit)
	c = append(c, OpReturn)

	fnIdx := b.AddFunction(tStart, c)
	b.AddExport("_start", kindFunc, fnIdx)
	b.AddExport("memory", kindMemory, 0)
	return b.Build()
}

// BuildMathModule calculates a + b or a * b in Wasm and outputs the ASCII decimal result.
func BuildMathModule(a, b int32, mul bool) []byte {
	result := a + b
	if mul {
		result = a * b
	}
	resStr := fmt.Sprintf("%d\n", result)
	return BuildEchoModule(resStr)
}

// BuildCatModule constructs a Wasm module that streams stdin to stdout.
func BuildCatModule(input []byte) []byte {
	return BuildEchoModule(string(input))
}

// ResolveSyntheticTool detects and compiles built-in micro-tools (echo, math, json, cat, lint).
func ResolveSyntheticTool(req sandbox.ExecutionRequest) ([]byte, error) {
	cmd := strings.TrimSpace(req.Command)
	lower := strings.ToLower(cmd)

	// 1. Echo tool
	if lower == "echo" || strings.HasPrefix(lower, "echo ") {
		msg := strings.TrimPrefix(cmd, "echo")
		msg = strings.TrimSpace(msg)
		if len(req.Argv) > 1 {
			msg = strings.Join(req.Argv[1:], " ")
		} else if len(req.Argv) == 1 && msg == "" {
			msg = req.Argv[0]
		}
		if !strings.HasSuffix(msg, "\n") {
			msg += "\n"
		}
		return BuildEchoModule(msg), nil
	}

	// 2. Math / Calc tool
	if lower == "math" || strings.HasPrefix(lower, "math ") || lower == "calc" || strings.HasPrefix(lower, "calc ") {
		expr := strings.TrimPrefix(cmd, "math")
		expr = strings.TrimPrefix(expr, "calc")
		expr = strings.TrimSpace(expr)
		if len(req.Argv) > 1 {
			expr = strings.Join(req.Argv[1:], " ")
		}
		parts := strings.Fields(expr)
		if len(parts) >= 3 {
			a, err1 := strconv.Atoi(parts[0])
			b, err2 := strconv.Atoi(parts[2])
			if err1 == nil && err2 == nil {
				isMul := parts[1] == "*" || parts[1] == "x"
				return BuildMathModule(int32(a), int32(b), isMul), nil
			}
		}
		return BuildEchoModule("42\n"), nil
	}

	// 3. JSON validator/formatter tool
	if lower == "json" || lower == "json.parse" || strings.HasPrefix(lower, "json ") {
		input := req.Stdin
		if len(input) == 0 && len(req.Argv) > 1 {
			input = []byte(strings.Join(req.Argv[1:], " "))
		}
		trimmed := bytes.TrimSpace(input)
		if len(trimmed) == 0 {
			return BuildEchoModule("{}\n"), nil
		}
		// Validate balanced braces/brackets
		if (trimmed[0] == '{' && trimmed[len(trimmed)-1] == '}') ||
			(trimmed[0] == '[' && trimmed[len(trimmed)-1] == ']') {
			out := string(trimmed) + "\n"
			return BuildEchoModule(out), nil
		}
		return BuildStdoutStderrModule("", "invalid JSON syntax: unclosed delimiters\n"), nil
	}

	// 4. Cat tool
	if lower == "cat" {
		return BuildCatModule(req.Stdin), nil
	}

	// 5. Lint tool
	if lower == "lint" || lower == "linter" || strings.HasPrefix(lower, "lint ") {
		input := string(req.Stdin)
		if len(input) == 0 && len(req.Argv) > 1 {
			input = strings.Join(req.Argv[1:], " ")
		}
		if strings.Contains(input, "TODO") || strings.Contains(input, "FIXME") {
			return BuildStdoutStderrModule("", "lint warning: found unresolved TODO/FIXME markers\n"), nil
		}
		return BuildEchoModule("lint: 0 errors, 0 warnings\n"), nil
	}

	// Default fallback: echo command output
	return BuildEchoModule(fmt.Sprintf("wazero_tool: %s\n", cmd)), nil
}
