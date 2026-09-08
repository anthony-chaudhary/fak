package wazero

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
