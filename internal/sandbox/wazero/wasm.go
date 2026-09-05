package wazero

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/binary"
	"encoding/hex"
	"errors"
	"fmt"
	"io"
	"math"

	"github.com/anthony-chaudhary/fak/internal/sandbox"
)

// Standard WebAssembly binary magic and version.
var (
	wasmMagic   = []byte{0x00, 0x61, 0x73, 0x6d}
	wasmVersion = []byte{0x01, 0x00, 0x00, 0x00}
)

// WebAssembly section IDs.
const (
	sectionIDCustom   = 0
	sectionIDType     = 1
	sectionIDImport   = 2
	sectionIDFunction = 3
	sectionIDTable    = 4
	sectionIDMemory   = 5
	sectionIDGlobal   = 6
	sectionIDExport   = 7
	sectionIDStart    = 8
	sectionIDElement  = 9
	sectionIDCode     = 10
	sectionIDData     = 11
)

// WebAssembly value types.
const (
	valTypeI32    byte = 0x7f
	valTypeI64    byte = 0x7e
	valTypeF32    byte = 0x7d
	valTypeF64    byte = 0x7c
	blockTypeVoid byte = 0x40
)

// Import & Export kinds.
const (
	kindFunc   byte = 0
	kindTable  byte = 1
	kindMemory byte = 2
	kindGlobal byte = 3
)

// OpCode represents WebAssembly instruction opcodes.
type OpCode = byte

const (
	OpUnreachable OpCode = 0x00
	OpNop         OpCode = 0x01
	OpBlock       OpCode = 0x02
	OpLoop        OpCode = 0x03
	OpIf          OpCode = 0x04
	OpElse        OpCode = 0x05
	OpEnd         OpCode = 0x0b
	OpBr          OpCode = 0x0c
	OpBrIf        OpCode = 0x0d
	OpReturn      OpCode = 0x0f
	OpCall        OpCode = 0x10
	OpDrop        OpCode = 0x1a
	OpSelect      OpCode = 0x1b

	OpLocalGet  OpCode = 0x20
	OpLocalSet  OpCode = 0x21
	OpLocalTee  OpCode = 0x22
	OpGlobalGet OpCode = 0x23
	OpGlobalSet OpCode = 0x24

	OpI32Load    OpCode = 0x28
	OpI64Load    OpCode = 0x29
	OpI32Load8S  OpCode = 0x2c
	OpI32Load8U  OpCode = 0x2d
	OpI32Load16S OpCode = 0x2e
	OpI32Load16U OpCode = 0x2f

	OpI32Store   OpCode = 0x36
	OpI64Store   OpCode = 0x37
	OpI32Store8  OpCode = 0x3a
	OpI32Store16 OpCode = 0x3b

	OpMemorySize OpCode = 0x3f
	OpMemoryGrow OpCode = 0x40

	OpI32Const OpCode = 0x41
	OpI64Const OpCode = 0x42

	OpI32Eqz OpCode = 0x45
	OpI32Eq  OpCode = 0x46
	OpI32Ne  OpCode = 0x47
	OpI32LtS OpCode = 0x48
	OpI32LtU OpCode = 0x49
	OpI32GtS OpCode = 0x4a
	OpI32GtU OpCode = 0x4b
	OpI32LeS OpCode = 0x4c
	OpI32LeU OpCode = 0x4d
	OpI32GeS OpCode = 0x4e
	OpI32GeU OpCode = 0x4f

	OpI64Eqz OpCode = 0x50
	OpI64Eq  OpCode = 0x51
	OpI64Ne  OpCode = 0x52
	OpI64LtS OpCode = 0x53
	OpI64LtU OpCode = 0x54
	OpI64GtS OpCode = 0x55
	OpI64GtU OpCode = 0x56
	OpI64LeS OpCode = 0x57
	OpI64LeU OpCode = 0x58
	OpI64GeS OpCode = 0x59
	OpI64GeU OpCode = 0x5a

	OpI32Add  OpCode = 0x6a
	OpI32Sub  OpCode = 0x6b
	OpI32Mul  OpCode = 0x6c
	OpI32DivS OpCode = 0x6d
	OpI32DivU OpCode = 0x6e
	OpI32RemS OpCode = 0x6f
	OpI32RemU OpCode = 0x70
	OpI32And  OpCode = 0x71
	OpI32Or   OpCode = 0x72
	OpI32Xor  OpCode = 0x73
	OpI32Shl  OpCode = 0x74
	OpI32ShrS OpCode = 0x75
	OpI32ShrU OpCode = 0x76

	OpI64Add  OpCode = 0x7c
	OpI64Sub  OpCode = 0x7d
	OpI64Mul  OpCode = 0x7e
	OpI64DivS OpCode = 0x7f
	OpI64DivU OpCode = 0x80
	OpI64RemS OpCode = 0x81
	OpI64RemU OpCode = 0x82
	OpI64And  OpCode = 0x83
	OpI64Or   OpCode = 0x84
	OpI64Xor  OpCode = 0x85
	OpI64Shl  OpCode = 0x86
	OpI64ShrS OpCode = 0x87
	OpI64ShrU OpCode = 0x88

	OpI32WrapI64    OpCode = 0xa7
	OpI64ExtendI32S OpCode = 0xac
	OpI64ExtendI32U OpCode = 0xad
)

// FuncType describes a function signature.
type FuncType struct {
	Params  []byte
	Results []byte
}

// Import represents an imported entity.
type Import struct {
	Module  string
	Field   string
	Kind    byte
	TypeIdx uint32
}

// MemoryLimit describes linear memory limits.
type MemoryLimit struct {
	Min    uint32
	Max    uint32
	HasMax bool
}

// Global describes a global variable.
type Global struct {
	Type    byte
	Mutable bool
	InitVal uint64
}

// Export represents an exported entity.
type Export struct {
	Name  string
	Kind  byte
	Index uint32
}

// DataSegment represents an initialized linear memory data segment.
type DataSegment struct {
	MemIndex uint32
	Offset   uint32
	Data     []byte
}

// Inst is a pre-decoded instruction with pre-computed jump targets.
type Inst struct {
	Op       OpCode
	Imm      int64
	Imm2     uint32
	TargetPC int
	IsLoop   bool
}

// CompiledFunction contains pre-compiled instructions and local allocations.
type CompiledFunction struct {
	TypeIndex    uint32
	NumLocals    int
	Instructions []Inst
}

// CompiledModule is an immutable, parsed, and pre-compiled WebAssembly module.
type CompiledModule struct {
	Hash      string
	Types     []FuncType
	Imports   []Import
	Functions []uint32 // type index for each defined function
	Memories  []MemoryLimit
	Globals   []Global
	Exports   map[string]Export
	Code      []CompiledFunction
	Data      []DataSegment
	StartFunc int // -1 if none
}

// NumImportedFunctions counts imports with kindFunc.
func (m *CompiledModule) NumImportedFunctions() uint32 {
	var count uint32
	for _, imp := range m.Imports {
		if imp.Kind == kindFunc {
			count++
		}
	}
	return count
}

// Compile parses and pre-compiles Wasm bytecode.
func Compile(bytecode []byte) (*CompiledModule, error) {
	if len(bytecode) < 8 {
		return nil, errors.New("wasm binary too short")
	}
	if !bytes.Equal(bytecode[:4], wasmMagic) {
		return nil, errors.New("invalid wasm magic")
	}
	if !bytes.Equal(bytecode[4:8], wasmVersion) {
		return nil, errors.New("unsupported wasm version")
	}

	hashBytes := sha256.Sum256(bytecode)
	mod := &CompiledModule{
		Hash:      hex.EncodeToString(hashBytes[:]),
		Exports:   make(map[string]Export),
		StartFunc: -1,
	}

	offset := 8
	for offset < len(bytecode) {
		sectionID := bytecode[offset]
		offset++
		sectionLen, err := readVarUint32(bytecode, &offset)
		if err != nil {
			return nil, err
		}
		sectionEnd := offset + int(sectionLen)
		if sectionEnd > len(bytecode) {
			return nil, io.ErrUnexpectedEOF
		}
		payload := bytecode[offset:sectionEnd]
		offset = sectionEnd

		if err := parseSection(mod, sectionID, payload); err != nil {
			return nil, err
		}
	}

	return mod, nil
}

func parseSection(mod *CompiledModule, id byte, payload []byte) error {
	pOff := 0
	switch id {
	case sectionIDType:
		count, err := readVarUint32(payload, &pOff)
		if err != nil {
			return err
		}
		for i := uint32(0); i < count; i++ {
			if pOff >= len(payload) {
				return io.ErrUnexpectedEOF
			}
			form := payload[pOff]
			pOff++
			if form != 0x60 {
				return fmt.Errorf("unsupported func form: 0x%02x", form)
			}
			paramCount, err := readVarUint32(payload, &pOff)
			if err != nil {
				return err
			}
			params := make([]byte, paramCount)
			for p := uint32(0); p < paramCount; p++ {
				params[p] = payload[pOff]
				pOff++
			}
			resultCount, err := readVarUint32(payload, &pOff)
			if err != nil {
				return err
			}
			results := make([]byte, resultCount)
			for r := uint32(0); r < resultCount; r++ {
				results[r] = payload[pOff]
				pOff++
			}
			mod.Types = append(mod.Types, FuncType{Params: params, Results: results})
		}

	case sectionIDImport:
		count, err := readVarUint32(payload, &pOff)
		if err != nil {
			return err
		}
		for i := uint32(0); i < count; i++ {
			modLen, err := readVarUint32(payload, &pOff)
			if err != nil {
				return err
			}
			modName := string(payload[pOff : pOff+int(modLen)])
			pOff += int(modLen)

			fieldLen, err := readVarUint32(payload, &pOff)
			if err != nil {
				return err
			}
			fieldName := string(payload[pOff : pOff+int(fieldLen)])
			pOff += int(fieldLen)

			kind := payload[pOff]
			pOff++

			var typeIdx uint32
			if kind == kindFunc {
				typeIdx, err = readVarUint32(payload, &pOff)
				if err != nil {
					return err
				}
			} else if kind == kindMemory {
				flags, _ := readVarUint32(payload, &pOff)
				minPages, _ := readVarUint32(payload, &pOff)
				var maxPages uint32
				hasMax := (flags & 1) != 0
				if hasMax {
					maxPages, _ = readVarUint32(payload, &pOff)
				}
				mod.Memories = append(mod.Memories, MemoryLimit{Min: minPages, Max: maxPages, HasMax: hasMax})
			} else if kind == kindGlobal {
				vType := payload[pOff]
				pOff++
				mut := payload[pOff]
				pOff++
				mod.Globals = append(mod.Globals, Global{Type: vType, Mutable: mut == 1})
			} else if kind == kindTable {
				pOff++ // elemType
				flags, _ := readVarUint32(payload, &pOff)
				_, _ = readVarUint32(payload, &pOff)
				if flags&1 != 0 {
					_, _ = readVarUint32(payload, &pOff)
				}
			}

			mod.Imports = append(mod.Imports, Import{
				Module:  modName,
				Field:   fieldName,
				Kind:    kind,
				TypeIdx: typeIdx,
			})
		}

	case sectionIDFunction:
		count, err := readVarUint32(payload, &pOff)
		if err != nil {
			return err
		}
		for i := uint32(0); i < count; i++ {
			typeIdx, err := readVarUint32(payload, &pOff)
			if err != nil {
				return err
			}
			mod.Functions = append(mod.Functions, typeIdx)
		}

	case sectionIDMemory:
		count, err := readVarUint32(payload, &pOff)
		if err != nil {
			return err
		}
		for i := uint32(0); i < count; i++ {
			flags, err := readVarUint32(payload, &pOff)
			if err != nil {
				return err
			}
			minPages, err := readVarUint32(payload, &pOff)
			if err != nil {
				return err
			}
			var maxPages uint32
			hasMax := (flags & 1) != 0
			if hasMax {
				maxPages, err = readVarUint32(payload, &pOff)
				if err != nil {
					return err
				}
			}
			mod.Memories = append(mod.Memories, MemoryLimit{Min: minPages, Max: maxPages, HasMax: hasMax})
		}

	case sectionIDGlobal:
		count, err := readVarUint32(payload, &pOff)
		if err != nil {
			return err
		}
		for i := uint32(0); i < count; i++ {
			vType := payload[pOff]
			pOff++
			mut := payload[pOff]
			pOff++
			initVal, err := parseInitExpr(payload, &pOff)
			if err != nil {
				return err
			}
			mod.Globals = append(mod.Globals, Global{Type: vType, Mutable: mut == 1, InitVal: initVal})
		}

	case sectionIDExport:
		count, err := readVarUint32(payload, &pOff)
		if err != nil {
			return err
		}
		for i := uint32(0); i < count; i++ {
			nameLen, err := readVarUint32(payload, &pOff)
			if err != nil {
				return err
			}
			name := string(payload[pOff : pOff+int(nameLen)])
			pOff += int(nameLen)
			kind := payload[pOff]
			pOff++
			idx, err := readVarUint32(payload, &pOff)
			if err != nil {
				return err
			}
			mod.Exports[name] = Export{Name: name, Kind: kind, Index: idx}
		}

	case sectionIDStart:
		idx, err := readVarUint32(payload, &pOff)
		if err != nil {
			return err
		}
		mod.StartFunc = int(idx)

	case sectionIDCode:
		count, err := readVarUint32(payload, &pOff)
		if err != nil {
			return err
		}
		for i := uint32(0); i < count; i++ {
			bodySize, err := readVarUint32(payload, &pOff)
			if err != nil {
				return err
			}
			bodyEnd := pOff + int(bodySize)
			localGroupCount, err := readVarUint32(payload, &pOff)
			if err != nil {
				return err
			}
			numLocals := 0
			for g := uint32(0); g < localGroupCount; g++ {
				lCount, err := readVarUint32(payload, &pOff)
				if err != nil {
					return err
				}
				pOff++ // type
				numLocals += int(lCount)
			}
			fnCode := payload[pOff:bodyEnd]
			pOff = bodyEnd

			typeIdx := mod.Functions[i]
			compiledFn, err := compileFunction(typeIdx, numLocals, fnCode)
			if err != nil {
				return err
			}
			mod.Code = append(mod.Code, compiledFn)
		}

	case sectionIDData:
		count, err := readVarUint32(payload, &pOff)
		if err != nil {
			return err
		}
		for i := uint32(0); i < count; i++ {
			flags, err := readVarUint32(payload, &pOff)
			if err != nil {
				return err
			}
			var memIdx uint32
			var offset uint32
			if flags == 0 {
				memIdx = 0
				initVal, err := parseInitExpr(payload, &pOff)
				if err != nil {
					return err
				}
				offset = uint32(initVal)
			} else if flags == 2 {
				memIdx, err = readVarUint32(payload, &pOff)
				if err != nil {
					return err
				}
				initVal, err := parseInitExpr(payload, &pOff)
				if err != nil {
					return err
				}
				offset = uint32(initVal)
			}
			dataLen, err := readVarUint32(payload, &pOff)
			if err != nil {
				return err
			}
			dataBytes := make([]byte, dataLen)
			copy(dataBytes, payload[pOff:pOff+int(dataLen)])
			pOff += int(dataLen)

			mod.Data = append(mod.Data, DataSegment{
				MemIndex: memIdx,
				Offset:   offset,
				Data:     dataBytes,
			})
		}
	}
	return nil
}

type ctrlFrame struct {
	op      OpCode
	startPC int
	elsePC  int
	fixups  []int
}

func compileFunction(typeIdx uint32, numLocals int, code []byte) (CompiledFunction, error) {
	var insts []Inst
	var ctrlStack []ctrlFrame

	off := 0
	for off < len(code) {
		op := OpCode(code[off])
		off++

		switch op {
		case OpBlock, OpLoop, OpIf:
			bType := code[off]
			off++
			frame := ctrlFrame{op: op, startPC: len(insts)}
			ctrlStack = append(ctrlStack, frame)
			insts = append(insts, Inst{Op: op, Imm: int64(bType), TargetPC: -1})

		case OpElse:
			if len(ctrlStack) == 0 || ctrlStack[len(ctrlStack)-1].op != OpIf {
				return CompiledFunction{}, errors.New("else without matching if")
			}
			top := &ctrlStack[len(ctrlStack)-1]
			insts[top.startPC].TargetPC = len(insts) + 1
			top.elsePC = len(insts)
			insts = append(insts, Inst{Op: OpElse, TargetPC: -1})

		case OpEnd:
			if len(ctrlStack) == 0 {
				insts = append(insts, Inst{Op: OpReturn})
				break
			}
			top := ctrlStack[len(ctrlStack)-1]
			ctrlStack = ctrlStack[:len(ctrlStack)-1]
			endPC := len(insts)

			if top.op == OpBlock {
				insts[top.startPC].TargetPC = endPC
				for _, fix := range top.fixups {
					insts[fix].TargetPC = endPC
				}
			} else if top.op == OpLoop {
				insts[top.startPC].TargetPC = endPC
			} else if top.op == OpIf {
				if top.elsePC != 0 {
					insts[top.elsePC].TargetPC = endPC
				} else {
					insts[top.startPC].TargetPC = endPC
				}
				for _, fix := range top.fixups {
					insts[fix].TargetPC = endPC
				}
			}
			insts = append(insts, Inst{Op: OpEnd})

		case OpBr:
			depth, err := readVarUint32(code, &off)
			if err != nil {
				return CompiledFunction{}, err
			}
			tIdx := len(ctrlStack) - 1 - int(depth)
			if tIdx < 0 {
				return CompiledFunction{}, errors.New("invalid branch depth")
			}
			target := &ctrlStack[tIdx]
			if target.op == OpLoop {
				insts = append(insts, Inst{Op: OpBr, TargetPC: target.startPC, IsLoop: true})
			} else {
				idx := len(insts)
				insts = append(insts, Inst{Op: OpBr, TargetPC: -1, IsLoop: false})
				target.fixups = append(target.fixups, idx)
			}

		case OpBrIf:
			depth, err := readVarUint32(code, &off)
			if err != nil {
				return CompiledFunction{}, err
			}
			tIdx := len(ctrlStack) - 1 - int(depth)
			if tIdx < 0 {
				return CompiledFunction{}, errors.New("invalid br_if depth")
			}
			target := &ctrlStack[tIdx]
			if target.op == OpLoop {
				insts = append(insts, Inst{Op: OpBrIf, TargetPC: target.startPC, IsLoop: true})
			} else {
				idx := len(insts)
				insts = append(insts, Inst{Op: OpBrIf, TargetPC: -1, IsLoop: false})
				target.fixups = append(target.fixups, idx)
			}

		case OpReturn:
			insts = append(insts, Inst{Op: OpReturn})

		case OpCall:
			fnIdx, err := readVarUint32(code, &off)
			if err != nil {
				return CompiledFunction{}, err
			}
			insts = append(insts, Inst{Op: OpCall, Imm: int64(fnIdx)})

		case OpDrop, OpSelect, OpNop, OpUnreachable:
			insts = append(insts, Inst{Op: op})

		case OpLocalGet, OpLocalSet, OpLocalTee:
			lIdx, err := readVarUint32(code, &off)
			if err != nil {
				return CompiledFunction{}, err
			}
			insts = append(insts, Inst{Op: op, Imm: int64(lIdx)})

		case OpGlobalGet, OpGlobalSet:
			gIdx, err := readVarUint32(code, &off)
			if err != nil {
				return CompiledFunction{}, err
			}
			insts = append(insts, Inst{Op: op, Imm: int64(gIdx)})

		case OpI32Load, OpI64Load, OpI32Load8S, OpI32Load8U, OpI32Load16S, OpI32Load16U,
			OpI32Store, OpI64Store, OpI32Store8, OpI32Store16:
			align, err := readVarUint32(code, &off)
			if err != nil {
				return CompiledFunction{}, err
			}
			offset, err := readVarUint32(code, &off)
			if err != nil {
				return CompiledFunction{}, err
			}
			insts = append(insts, Inst{Op: op, Imm: int64(offset), Imm2: align})

		case OpMemorySize, OpMemoryGrow:
			off++ // reserved 0x00
			insts = append(insts, Inst{Op: op})

		case OpI32Const:
			val, err := readVarInt32(code, &off)
			if err != nil {
				return CompiledFunction{}, err
			}
			insts = append(insts, Inst{Op: OpI32Const, Imm: int64(val)})

		case OpI64Const:
			val, err := readVarInt64(code, &off)
			if err != nil {
				return CompiledFunction{}, err
			}
			insts = append(insts, Inst{Op: OpI64Const, Imm: val})

		default:
			// arithmetic, logical, and conversion opcodes
			insts = append(insts, Inst{Op: op})
		}
	}

	if len(insts) == 0 || insts[len(insts)-1].Op != OpReturn {
		insts = append(insts, Inst{Op: OpReturn})
	}

	return CompiledFunction{
		TypeIndex:    typeIdx,
		NumLocals:    numLocals,
		Instructions: insts,
	}, nil
}

func parseInitExpr(b []byte, offset *int) (uint64, error) {
	if *offset >= len(b) {
		return 0, io.ErrUnexpectedEOF
	}
	op := b[*offset]
	*offset++
	var val uint64
	switch op {
	case 0x41: // i32.const
		v, err := readVarInt32(b, offset)
		if err != nil {
			return 0, err
		}
		val = uint64(uint32(v))
	case 0x42: // i64.const
		v, err := readVarInt64(b, offset)
		if err != nil {
			return 0, err
		}
		val = uint64(v)
	default:
		for *offset < len(b) && b[*offset] != 0x0b {
			*offset++
		}
	}
	if *offset < len(b) && b[*offset] == 0x0b {
		*offset++
	}
	return val, nil
}

// LEB128 decoding helpers

func readVarUint32(b []byte, offset *int) (uint32, error) {
	var result uint32
	var shift uint
	for {
		if *offset >= len(b) {
			return 0, io.ErrUnexpectedEOF
		}
		byteVal := b[*offset]
		*offset++
		result |= uint32(byteVal&0x7f) << shift
		if (byteVal & 0x80) == 0 {
			break
		}
		shift += 7
		if shift > 35 {
			return 0, errors.New("varuint32 overflow")
		}
	}
	return result, nil
}

func readVarInt32(b []byte, offset *int) (int32, error) {
	var result int32
	var shift uint
	var byteVal byte
	for {
		if *offset >= len(b) {
			return 0, io.ErrUnexpectedEOF
		}
		byteVal = b[*offset]
		*offset++
		result |= int32(byteVal&0x7f) << shift
		shift += 7
		if (byteVal & 0x80) == 0 {
			break
		}
		if shift > 35 {
			return 0, errors.New("varint32 overflow")
		}
	}
	if shift < 32 && (byteVal&0x40) != 0 {
		result |= -int32(1 << shift)
	}
	return result, nil
}

func readVarInt64(b []byte, offset *int) (int64, error) {
	var result int64
	var shift uint
	var byteVal byte
	for {
		if *offset >= len(b) {
			return 0, io.ErrUnexpectedEOF
		}
		byteVal = b[*offset]
		*offset++
		result |= int64(byteVal&0x7f) << shift
		shift += 7
		if (byteVal & 0x80) == 0 {
			break
		}
		if shift > 70 {
			return 0, errors.New("varint64 overflow")
		}
	}
	if shift < 64 && (byteVal&0x40) != 0 {
		result |= -int64(1 << shift)
	}
	return result, nil
}

// ---------------------------------------------------------------------------
// VM EXECUTION ENGINE
// ---------------------------------------------------------------------------

// VM manages instance memory, stack, and execution state.
type VM struct {
	module    *CompiledModule
	memory    []byte
	maxPages  int
	fuelLimit int64
	fuelUsed  int64
	globals   []uint64
	wasi      *WASIContext
	exitCode  int
}

type vmFrame struct {
	fn       *CompiledFunction
	fnType   FuncType
	pc       int
	locals   []uint64
	stackPtr int
}

// NewVM instantiates a virtual machine for a compiled module.
func NewVM(mod *CompiledModule, maxMemoryBytes int64, fuelLimit int64, wasi *WASIContext) (*VM, error) {
	initialPages := 1
	for _, m := range mod.Memories {
		if int(m.Min) > initialPages {
			initialPages = int(m.Min)
		}
	}
	if maxMemoryBytes <= 0 {
		maxMemoryBytes = 16 * 1024 * 1024 // 16MB default
	}
	maxPages := int(maxMemoryBytes / 65536)
	if maxPages < 1 {
		maxPages = 1
	}
	if initialPages > maxPages {
		return nil, fmt.Errorf("initial memory (%d pages) exceeds cap (%d pages)", initialPages, maxPages)
	}

	mem := make([]byte, initialPages*65536)
	for _, seg := range mod.Data {
		end := int(seg.Offset) + len(seg.Data)
		if end > len(mem) {
			return nil, errors.New("data segment exceeds initial memory")
		}
		copy(mem[seg.Offset:end], seg.Data)
	}

	globals := make([]uint64, len(mod.Globals))
	for i, g := range mod.Globals {
		globals[i] = g.InitVal
	}

	if fuelLimit <= 0 {
		fuelLimit = 10_000_000
	}

	return &VM{
		module:    mod,
		memory:    mem,
		maxPages:  maxPages,
		fuelLimit: fuelLimit,
		globals:   globals,
		wasi:      wasi,
	}, nil
}

// MemoryBytes returns the current linear memory size in bytes.
func (vm *VM) MemoryBytes() int64 {
	return int64(len(vm.memory))
}

// FuelUsed returns the total instruction count executed.
func (vm *VM) FuelUsed() int64 {
	return vm.fuelUsed
}

// ExitCode returns the proc_exit code or 0.
func (vm *VM) ExitCode() int {
	return vm.exitCode
}

// Run executes the module entrypoint (_start or main).
func (vm *VM) Run(ctx context.Context) (int, error) {
	var entryFuncIdx uint32
	found := false

	if exp, ok := vm.module.Exports["_start"]; ok && exp.Kind == kindFunc {
		entryFuncIdx = exp.Index
		found = true
	} else if exp, ok := vm.module.Exports["main"]; ok && exp.Kind == kindFunc {
		entryFuncIdx = exp.Index
		found = true
	} else if vm.module.StartFunc >= 0 {
		entryFuncIdx = uint32(vm.module.StartFunc)
		found = true
	} else {
		// Fallback: search for first exported function
		for _, exp := range vm.module.Exports {
			if exp.Kind == kindFunc {
				entryFuncIdx = exp.Index
				found = true
				break
			}
		}
	}

	if !found {
		// Module without entrypoint (e.g. library or data-only) succeeds trivially
		return 0, nil
	}

	numImports := vm.module.NumImportedFunctions()
	if entryFuncIdx < numImports {
		return 0, errors.New("cannot run imported function as entrypoint")
	}

	defIdx := entryFuncIdx - numImports
	if int(defIdx) >= len(vm.module.Code) {
		return 0, errors.New("entrypoint function index out of range")
	}

	fn := &vm.module.Code[defIdx]
	fnType := vm.module.Types[vm.module.Functions[defIdx]]

	nParams := len(fnType.Params)
	locals := make([]uint64, nParams+fn.NumLocals)

	stack := make([]uint64, 0, 1024)
	frames := []vmFrame{{
		fn:       fn,
		fnType:   fnType,
		pc:       0,
		locals:   locals,
		stackPtr: 0,
	}}

	for len(frames) > 0 {
		frame := &frames[len(frames)-1]

		if vm.fuelLimit > 0 {
			if vm.fuelUsed >= vm.fuelLimit {
				return -1, sandbox.NewSandboxError("FUEL_EXHAUSTED", "instruction fuel budget exhausted")
			}
			vm.fuelUsed++
		}

		if (vm.fuelUsed&0x3ff) == 0 && ctx != nil {
			if err := ctx.Err(); err != nil {
				return -1, err
			}
		}

		if frame.pc >= len(frame.fn.Instructions) {
			nResults := len(frame.fnType.Results)
			results := make([]uint64, nResults)
			for r := nResults - 1; r >= 0; r-- {
				if len(stack) > 0 {
					results[r] = stack[len(stack)-1]
					stack = stack[:len(stack)-1]
				}
			}
			stack = stack[:frame.stackPtr]
			stack = append(stack, results...)
			frames = frames[:len(frames)-1]
			continue
		}

		inst := frame.fn.Instructions[frame.pc]
		frame.pc++

		switch inst.Op {
		case OpNop, OpBlock, OpLoop, OpEnd:
			// Handled via branch jumps

		case OpUnreachable:
			return -1, errors.New("wasm trap: unreachable instruction executed")

		case OpIf:
			if len(stack) == 0 {
				return -1, errors.New("wasm trap: stack underflow on if")
			}
			cond := stack[len(stack)-1]
			stack = stack[:len(stack)-1]
			if cond == 0 {
				frame.pc = inst.TargetPC
			}

		case OpElse:
			frame.pc = inst.TargetPC

		case OpBr:
			frame.pc = inst.TargetPC

		case OpBrIf:
			if len(stack) == 0 {
				return -1, errors.New("wasm trap: stack underflow on br_if")
			}
			cond := stack[len(stack)-1]
			stack = stack[:len(stack)-1]
			if cond != 0 {
				frame.pc = inst.TargetPC
			}

		case OpReturn:
			nResults := len(frame.fnType.Results)
			results := make([]uint64, nResults)
			for r := nResults - 1; r >= 0; r-- {
				if len(stack) > 0 {
					results[r] = stack[len(stack)-1]
					stack = stack[:len(stack)-1]
				}
			}
			stack = stack[:frame.stackPtr]
			stack = append(stack, results...)
			frames = frames[:len(frames)-1]

		case OpCall:
			callIdx := uint32(inst.Imm)
			if callIdx < numImports {
				imp := vm.module.Imports[callIdx]
				impType := vm.module.Types[imp.TypeIdx]
				nArgs := len(impType.Params)
				args := make([]uint64, nArgs)
				for a := nArgs - 1; a >= 0; a-- {
					if len(stack) == 0 {
						return -1, errors.New("wasm trap: stack underflow on host call")
					}
					args[a] = stack[len(stack)-1]
					stack = stack[:len(stack)-1]
				}
				retVal, err := vm.invokeHost(imp, args)
				if err != nil {
					var pExit *procExitError
					if errors.As(err, &pExit) {
						vm.exitCode = pExit.Code
						return pExit.Code, nil
					}
					return -1, err
				}
				if len(impType.Results) > 0 {
					stack = append(stack, retVal)
				}
			} else {
				targetDefIdx := callIdx - numImports
				if int(targetDefIdx) >= len(vm.module.Code) {
					return -1, errors.New("wasm trap: call function index out of range")
				}
				nextFn := &vm.module.Code[targetDefIdx]
				nextFnType := vm.module.Types[vm.module.Functions[targetDefIdx]]
				nextParams := len(nextFnType.Params)
				nextLocals := make([]uint64, nextParams+nextFn.NumLocals)
				for a := nextParams - 1; a >= 0; a-- {
					if len(stack) == 0 {
						return -1, errors.New("wasm trap: stack underflow on call")
					}
					nextLocals[a] = stack[len(stack)-1]
					stack = stack[:len(stack)-1]
				}
				newFrame := vmFrame{
					fn:       nextFn,
					fnType:   nextFnType,
					pc:       0,
					locals:   nextLocals,
					stackPtr: len(stack),
				}
				frames = append(frames, newFrame)
			}

		case OpDrop:
			if len(stack) > 0 {
				stack = stack[:len(stack)-1]
			}

		case OpSelect:
			if len(stack) < 3 {
				return -1, errors.New("wasm trap: stack underflow on select")
			}
			c := stack[len(stack)-1]
			v2 := stack[len(stack)-2]
			v1 := stack[len(stack)-3]
			stack = stack[:len(stack)-3]
			if c != 0 {
				stack = append(stack, v1)
			} else {
				stack = append(stack, v2)
			}

		case OpLocalGet:
			idx := int(inst.Imm)
			if idx >= len(frame.locals) {
				return -1, errors.New("wasm trap: local.get index out of bounds")
			}
			stack = append(stack, frame.locals[idx])

		case OpLocalSet:
			if len(stack) == 0 {
				return -1, errors.New("wasm trap: stack underflow on local.set")
			}
			idx := int(inst.Imm)
			val := stack[len(stack)-1]
			stack = stack[:len(stack)-1]
			if idx >= len(frame.locals) {
				return -1, errors.New("wasm trap: local.set index out of bounds")
			}
			frame.locals[idx] = val

		case OpLocalTee:
			if len(stack) == 0 {
				return -1, errors.New("wasm trap: stack underflow on local.tee")
			}
			idx := int(inst.Imm)
			val := stack[len(stack)-1]
			if idx >= len(frame.locals) {
				return -1, errors.New("wasm trap: local.tee index out of bounds")
			}
			frame.locals[idx] = val

		case OpGlobalGet:
			idx := int(inst.Imm)
			if idx >= len(vm.globals) {
				return -1, errors.New("wasm trap: global.get index out of bounds")
			}
			stack = append(stack, vm.globals[idx])

		case OpGlobalSet:
			if len(stack) == 0 {
				return -1, errors.New("wasm trap: stack underflow on global.set")
			}
			idx := int(inst.Imm)
			val := stack[len(stack)-1]
			stack = stack[:len(stack)-1]
			if idx >= len(vm.globals) {
				return -1, errors.New("wasm trap: global.set index out of bounds")
			}
			vm.globals[idx] = val

		case OpI32Const:
			stack = append(stack, uint64(uint32(inst.Imm)))

		case OpI64Const:
			stack = append(stack, uint64(inst.Imm))

		case OpI32Eqz:
			if len(stack) == 0 {
				return -1, errors.New("wasm trap: stack underflow")
			}
			v := uint32(stack[len(stack)-1])
			if v == 0 {
				stack[len(stack)-1] = 1
			} else {
				stack[len(stack)-1] = 0
			}

		case OpI32Eq:
			if len(stack) < 2 {
				return -1, errors.New("wasm trap: stack underflow")
			}
			b := uint32(stack[len(stack)-1])
			a := uint32(stack[len(stack)-2])
			stack = stack[:len(stack)-2]
			if a == b {
				stack = append(stack, 1)
			} else {
				stack = append(stack, 0)
			}

		case OpI32Ne:
			if len(stack) < 2 {
				return -1, errors.New("wasm trap: stack underflow")
			}
			b := uint32(stack[len(stack)-1])
			a := uint32(stack[len(stack)-2])
			stack = stack[:len(stack)-2]
			if a != b {
				stack = append(stack, 1)
			} else {
				stack = append(stack, 0)
			}

		case OpI32LtS:
			if len(stack) < 2 {
				return -1, errors.New("wasm trap: stack underflow")
			}
			b := int32(stack[len(stack)-1])
			a := int32(stack[len(stack)-2])
			stack = stack[:len(stack)-2]
			if a < b {
				stack = append(stack, 1)
			} else {
				stack = append(stack, 0)
			}

		case OpI32LtU:
			if len(stack) < 2 {
				return -1, errors.New("wasm trap: stack underflow")
			}
			b := uint32(stack[len(stack)-1])
			a := uint32(stack[len(stack)-2])
			stack = stack[:len(stack)-2]
			if a < b {
				stack = append(stack, 1)
			} else {
				stack = append(stack, 0)
			}

		case OpI32GtS:
			if len(stack) < 2 {
				return -1, errors.New("wasm trap: stack underflow")
			}
			b := int32(stack[len(stack)-1])
			a := int32(stack[len(stack)-2])
			stack = stack[:len(stack)-2]
			if a > b {
				stack = append(stack, 1)
			} else {
				stack = append(stack, 0)
			}

		case OpI32GtU:
			if len(stack) < 2 {
				return -1, errors.New("wasm trap: stack underflow")
			}
			b := uint32(stack[len(stack)-1])
			a := uint32(stack[len(stack)-2])
			stack = stack[:len(stack)-2]
			if a > b {
				stack = append(stack, 1)
			} else {
				stack = append(stack, 0)
			}

		case OpI32LeS:
			if len(stack) < 2 {
				return -1, errors.New("wasm trap: stack underflow")
			}
			b := int32(stack[len(stack)-1])
			a := int32(stack[len(stack)-2])
			stack = stack[:len(stack)-2]
			if a <= b {
				stack = append(stack, 1)
			} else {
				stack = append(stack, 0)
			}

		case OpI32LeU:
			if len(stack) < 2 {
				return -1, errors.New("wasm trap: stack underflow")
			}
			b := uint32(stack[len(stack)-1])
			a := uint32(stack[len(stack)-2])
			stack = stack[:len(stack)-2]
			if a <= b {
				stack = append(stack, 1)
			} else {
				stack = append(stack, 0)
			}

		case OpI32GeS:
			if len(stack) < 2 {
				return -1, errors.New("wasm trap: stack underflow")
			}
			b := int32(stack[len(stack)-1])
			a := int32(stack[len(stack)-2])
			stack = stack[:len(stack)-2]
			if a >= b {
				stack = append(stack, 1)
			} else {
				stack = append(stack, 0)
			}

		case OpI32GeU:
			if len(stack) < 2 {
				return -1, errors.New("wasm trap: stack underflow")
			}
			b := uint32(stack[len(stack)-1])
			a := uint32(stack[len(stack)-2])
			stack = stack[:len(stack)-2]
			if a >= b {
				stack = append(stack, 1)
			} else {
				stack = append(stack, 0)
			}

		case OpI32Add:
			if len(stack) < 2 {
				return -1, errors.New("wasm trap: stack underflow")
			}
			b := uint32(stack[len(stack)-1])
			a := uint32(stack[len(stack)-2])
			stack = stack[:len(stack)-2]
			stack = append(stack, uint64(a+b))

		case OpI32Sub:
			if len(stack) < 2 {
				return -1, errors.New("wasm trap: stack underflow")
			}
			b := uint32(stack[len(stack)-1])
			a := uint32(stack[len(stack)-2])
			stack = stack[:len(stack)-2]
			stack = append(stack, uint64(a-b))

		case OpI32Mul:
			if len(stack) < 2 {
				return -1, errors.New("wasm trap: stack underflow")
			}
			b := uint32(stack[len(stack)-1])
			a := uint32(stack[len(stack)-2])
			stack = stack[:len(stack)-2]
			stack = append(stack, uint64(a*b))

		case OpI32DivS:
			if len(stack) < 2 {
				return -1, errors.New("wasm trap: stack underflow")
			}
			b := int32(stack[len(stack)-1])
			a := int32(stack[len(stack)-2])
			stack = stack[:len(stack)-2]
			if b == 0 {
				return -1, errors.New("wasm trap: integer divide by zero")
			}
			if a == math.MinInt32 && b == -1 {
				return -1, errors.New("wasm trap: integer overflow")
			}
			stack = append(stack, uint64(uint32(a/b)))

		case OpI32DivU:
			if len(stack) < 2 {
				return -1, errors.New("wasm trap: stack underflow")
			}
			b := uint32(stack[len(stack)-1])
			a := uint32(stack[len(stack)-2])
			stack = stack[:len(stack)-2]
			if b == 0 {
				return -1, errors.New("wasm trap: integer divide by zero")
			}
			stack = append(stack, uint64(a/b))

		case OpI32RemS:
			if len(stack) < 2 {
				return -1, errors.New("wasm trap: stack underflow")
			}
			b := int32(stack[len(stack)-1])
			a := int32(stack[len(stack)-2])
			stack = stack[:len(stack)-2]
			if b == 0 {
				return -1, errors.New("wasm trap: integer divide by zero")
			}
			stack = append(stack, uint64(uint32(a%b)))

		case OpI32RemU:
			if len(stack) < 2 {
				return -1, errors.New("wasm trap: stack underflow")
			}
			b := uint32(stack[len(stack)-1])
			a := uint32(stack[len(stack)-2])
			stack = stack[:len(stack)-2]
			if b == 0 {
				return -1, errors.New("wasm trap: integer divide by zero")
			}
			stack = append(stack, uint64(a%b))

		case OpI32And:
			if len(stack) < 2 {
				return -1, errors.New("wasm trap: stack underflow")
			}
			b := uint32(stack[len(stack)-1])
			a := uint32(stack[len(stack)-2])
			stack = stack[:len(stack)-2]
			stack = append(stack, uint64(a&b))

		case OpI32Or:
			if len(stack) < 2 {
				return -1, errors.New("wasm trap: stack underflow")
			}
			b := uint32(stack[len(stack)-1])
			a := uint32(stack[len(stack)-2])
			stack = stack[:len(stack)-2]
			stack = append(stack, uint64(a|b))

		case OpI32Xor:
			if len(stack) < 2 {
				return -1, errors.New("wasm trap: stack underflow")
			}
			b := uint32(stack[len(stack)-1])
			a := uint32(stack[len(stack)-2])
			stack = stack[:len(stack)-2]
			stack = append(stack, uint64(a^b))

		case OpI32Shl:
			if len(stack) < 2 {
				return -1, errors.New("wasm trap: stack underflow")
			}
			b := uint32(stack[len(stack)-1])
			a := uint32(stack[len(stack)-2])
			stack = stack[:len(stack)-2]
			stack = append(stack, uint64(a<<(b&31)))

		case OpI32ShrS:
			if len(stack) < 2 {
				return -1, errors.New("wasm trap: stack underflow")
			}
			b := uint32(stack[len(stack)-1])
			a := int32(stack[len(stack)-2])
			stack = stack[:len(stack)-2]
			stack = append(stack, uint64(uint32(a>>(b&31))))

		case OpI32ShrU:
			if len(stack) < 2 {
				return -1, errors.New("wasm trap: stack underflow")
			}
			b := uint32(stack[len(stack)-1])
			a := uint32(stack[len(stack)-2])
			stack = stack[:len(stack)-2]
			stack = append(stack, uint64(a>>(b&31)))

		case OpI64Add:
			if len(stack) < 2 {
				return -1, errors.New("wasm trap: stack underflow")
			}
			b := stack[len(stack)-1]
			a := stack[len(stack)-2]
			stack = stack[:len(stack)-2]
			stack = append(stack, a+b)

		case OpI64Sub:
			if len(stack) < 2 {
				return -1, errors.New("wasm trap: stack underflow")
			}
			b := stack[len(stack)-1]
			a := stack[len(stack)-2]
			stack = stack[:len(stack)-2]
			stack = append(stack, a-b)

		case OpI64Mul:
			if len(stack) < 2 {
				return -1, errors.New("wasm trap: stack underflow")
			}
			b := stack[len(stack)-1]
			a := stack[len(stack)-2]
			stack = stack[:len(stack)-2]
			stack = append(stack, a*b)

		case OpI32Load:
			if len(stack) == 0 {
				return -1, errors.New("wasm trap: stack underflow on load")
			}
			addr := uint32(stack[len(stack)-1]) + uint32(inst.Imm)
			stack = stack[:len(stack)-1]
			if int(addr)+4 > len(vm.memory) {
				return -1, errors.New("wasm trap: out of bounds memory read")
			}
			val := binary.LittleEndian.Uint32(vm.memory[addr : addr+4])
			stack = append(stack, uint64(val))

		case OpI32Load8U:
			if len(stack) == 0 {
				return -1, errors.New("wasm trap: stack underflow on load")
			}
			addr := uint32(stack[len(stack)-1]) + uint32(inst.Imm)
			stack = stack[:len(stack)-1]
			if int(addr) >= len(vm.memory) {
				return -1, errors.New("wasm trap: out of bounds memory read")
			}
			stack = append(stack, uint64(vm.memory[addr]))

		case OpI32Load8S:
			if len(stack) == 0 {
				return -1, errors.New("wasm trap: stack underflow on load")
			}
			addr := uint32(stack[len(stack)-1]) + uint32(inst.Imm)
			stack = stack[:len(stack)-1]
			if int(addr) >= len(vm.memory) {
				return -1, errors.New("wasm trap: out of bounds memory read")
			}
			stack = append(stack, uint64(uint32(int32(int8(vm.memory[addr])))))

		case OpI32Store:
			if len(stack) < 2 {
				return -1, errors.New("wasm trap: stack underflow on store")
			}
			val := uint32(stack[len(stack)-1])
			addr := uint32(stack[len(stack)-2]) + uint32(inst.Imm)
			stack = stack[:len(stack)-2]
			if int(addr)+4 > len(vm.memory) {
				return -1, errors.New("wasm trap: out of bounds memory write")
			}
			binary.LittleEndian.PutUint32(vm.memory[addr:addr+4], val)

		case OpI32Store8:
			if len(stack) < 2 {
				return -1, errors.New("wasm trap: stack underflow on store8")
			}
			val := byte(stack[len(stack)-1])
			addr := uint32(stack[len(stack)-2]) + uint32(inst.Imm)
			stack = stack[:len(stack)-2]
			if int(addr) >= len(vm.memory) {
				return -1, errors.New("wasm trap: out of bounds memory write")
			}
			vm.memory[addr] = val

		case OpMemorySize:
			stack = append(stack, uint64(len(vm.memory)/65536))

		case OpMemoryGrow:
			if len(stack) == 0 {
				return -1, errors.New("wasm trap: stack underflow on memory.grow")
			}
			delta := int32(stack[len(stack)-1])
			stack = stack[:len(stack)-1]
			if delta < 0 {
				stack = append(stack, ^uint64(0)) // -1
			} else {
				currPages := len(vm.memory) / 65536
				newPages := currPages + int(delta)
				if newPages > vm.maxPages {
					stack = append(stack, ^uint64(0)) // -1 on exceeding cap
				} else {
					newMem := make([]byte, newPages*65536)
					copy(newMem, vm.memory)
					vm.memory = newMem
					stack = append(stack, uint64(currPages))
				}
			}

		case OpI32WrapI64:
			if len(stack) == 0 {
				return -1, errors.New("wasm trap: stack underflow")
			}
			v := stack[len(stack)-1]
			stack[len(stack)-1] = uint64(uint32(v))

		case OpI64ExtendI32S:
			if len(stack) == 0 {
				return -1, errors.New("wasm trap: stack underflow")
			}
			v := int32(stack[len(stack)-1])
			stack[len(stack)-1] = uint64(int64(v))

		case OpI64ExtendI32U:
			if len(stack) == 0 {
				return -1, errors.New("wasm trap: stack underflow")
			}
			v := uint32(stack[len(stack)-1])
			stack[len(stack)-1] = uint64(v)
		}
	}

	return vm.exitCode, nil
}

func (vm *VM) invokeHost(imp Import, args []uint64) (uint64, error) {
	if imp.Module == "wasi_snapshot_preview1" {
		return vm.wasi.Dispatch(vm, imp.Field, args)
	}
	return 0, fmt.Errorf("unsupported host module: %s", imp.Module)
}
