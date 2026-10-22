package computebuild

import (
	"bytes"
	"crypto/sha256"
	"encoding/binary"
	"encoding/hex"
	"fmt"
)

// packedAttentionStructure is deliberately not a build receipt or admission bit.
// It records structural observations of the SAME byte snapshot that is hashed.
// It proves neither source correspondence nor arithmetic/control/data flow,
// descriptor bounds, barriers, physical allocation size or device support.
// No registry, native resolver or capability consumes this preparatory checker.
type packedAttentionStructure struct {
	ModuleSHA256           string
	LocalSize              [3]uint32
	PushBytes              uint32
	WorkgroupDeclaredBytes uint32
}

type packedSPIRVReflection struct {
	defs                                               map[uint32][]uint32 // complete instruction, including opcode word
	decorations                                        map[[2]uint32]uint32
	members                                            map[[3]uint32]uint32
	buffers, shared, pushes                            []uint32          // OpVariable IDs
	arithmetic                                         map[uint32]uint32 // result ID -> scalar type
	arithmeticOps                                      map[uint32]int
	expTypes                                           []uint32
	entry, localEntry, modeEntry                       uint32
	entries, locals, extensions, imports, memoryModels int
	local                                              [3]uint32
	modes, capabilities                                uint32
}

func packedContractError(why string) error {
	return fmt.Errorf("packed main attention SPIR-V structure: %s", why)
}

// inspectPackedAttentionSPIRV is a narrow structural inspector, not spirv-val.
// Caller must separately validate executable SPIR-V and witnessed source/compiler
// provenance before using its result. Unknown registry modules remain rejected:
// this function is intentionally not wired into any registry or receipt path.
func inspectPackedAttentionSPIRV(data []byte) (packedAttentionStructure, error) {
	var zero packedAttentionStructure
	if len(data) < 20 || len(data) > 1024*1024 || len(data)%4 != 0 {
		return zero, packedContractError("invalid module length")
	}
	w := make([]uint32, len(data)/4)
	for i := range w {
		w[i] = binary.LittleEndian.Uint32(data[4*i:])
	}
	if w[0] != 0x07230203 || w[1] < 0x10000 || w[1] > 0x10500 || w[1]&255 != 0 || w[3] == 0 || w[4] != 0 {
		return zero, packedContractError("invalid or unsupported header")
	}
	r, err := reflectPackedAttentionInstructions(w[5:], w[3])
	if err != nil {
		return zero, err
	}
	if err = validatePackedAttentionReflection(r); err != nil {
		return zero, err
	}
	digest := sha256.Sum256(data)
	return packedAttentionStructure{hex.EncodeToString(digest[:]), r.local, 32, 17936}, nil
}

func reflectPackedAttentionInstructions(words []uint32, bound uint32) (*packedSPIRVReflection, error) {
	r := &packedSPIRVReflection{defs: map[uint32][]uint32{}, decorations: map[[2]uint32]uint32{}, members: map[[3]uint32]uint32{}, arithmetic: map[uint32]uint32{}, arithmeticOps: map[uint32]int{}}
	validID := func(id uint32) bool { return id > 0 && id < bound }
	var importID uint32
	var expImports []uint32
	fail := func(why string) (*packedSPIRVReflection, error) { return nil, packedContractError(why) }
	for p := 0; p < len(words); {
		n, op := int(words[p]>>16), words[p]&0xffff
		if n == 0 || n > len(words)-p {
			return fail("malformed instruction framing")
		}
		a := words[p : p+n]
		switch op {
		case 20, 21, 22, 23, 28, 29, 30, 32, 43, 59:
			if (op == 20 && n != 2) || (op == 21 && n != 4) || (op == 22 && n != 3) || (op == 23 && n != 4) || (op == 28 && n != 4) || (op == 29 && n != 3) || (op == 30 && n < 2) || (op == 32 && n != 4) || (op == 43 && n != 4) || (op == 59 && n != 4 && n != 5) {
				return fail("unsupported type, constant or variable declaration")
			}
			id := a[1]
			if op == 43 || op == 59 {
				id = a[2]
			}
			if !validID(id) || r.defs[id] != nil {
				return fail("invalid or duplicate definition")
			}
			r.defs[id] = a
			if op == 21 && (a[2] != 32 || a[3] > 1) {
				return fail("only 32-bit integers supported")
			}
			if op == 22 && a[2] != 32 {
				return fail("only binary32 supported")
			}
			if op == 59 {
				if !validID(a[1]) {
					return fail("invalid variable type")
				}
				switch a[3] {
				case 9:
					r.pushes = append(r.pushes, id)
				case 12:
					r.buffers = append(r.buffers, id)
				case 4:
					r.shared = append(r.shared, id)
				case 1, 6, 7: // Input, Private, Function; not descriptors.
				default:
					return fail("unsupported variable storage class")
				}
			}
		case 71:
			if n < 3 || !validID(a[1]) {
				return fail("malformed decoration")
			}
			kind := a[2]
			if kind == 0 || kind == 1 || kind == 39 || kind == 40 {
				return fail("precision, specialization or fast-math override")
			}
			if kind == 2 || kind == 42 || kind == 6 || kind == 33 || kind == 34 {
				value := uint32(0)
				if kind == 2 || kind == 42 {
					if n != 3 {
						return fail("malformed marker decoration")
					}
				} else {
					if n != 4 {
						return fail("malformed value decoration")
					}
					value = a[3]
				}
				key := [2]uint32{a[1], kind}
				if _, ok := r.decorations[key]; ok {
					return fail("duplicate decoration")
				}
				r.decorations[key] = value
			} else if kind != 11 || n != 4 {
				return fail("unsupported decoration")
			} // BuiltIn
		case 72:
			if n < 4 || !validID(a[1]) {
				return fail("malformed member decoration")
			}
			value := uint32(0)
			if a[3] == 35 {
				if n != 5 {
					return fail("malformed member offset")
				}
				value = a[4]
			} else if a[3] == 24 || a[3] == 25 {
				if n != 4 || a[2] != 0 {
					return fail("unsupported access decoration")
				}
			} else {
				return fail("unsupported member decoration")
			}
			key := [3]uint32{a[1], a[2], a[3]}
			if _, ok := r.members[key]; ok {
				return fail("duplicate member decoration")
			}
			r.members[key] = value
		case 14:
			if n != 3 || a[1] != 0 || a[2] != 1 {
				return fail("expected Logical GLSL450 memory model")
			}
			r.memoryModels++
		case 15:
			if n < 5 || a[1] != 5 || !validID(a[2]) || a[3] != 0x6e69616d || a[4] != 0 {
				return fail("expected compute main entry")
			}
			r.entries++
			r.entry = a[2]
		case 16:
			if n < 3 || !validID(a[1]) {
				return fail("malformed execution mode")
			}
			if a[2] == 17 {
				if n != 6 {
					return fail("malformed local size")
				}
				r.locals++
				r.localEntry = a[1]
				r.local = [3]uint32{a[3], a[4], a[5]}
			} else {
				bit := uint32(0)
				switch a[2] {
				case 4459:
					bit = 1
				case 4461:
					bit = 2
				case 4462:
					bit = 4
				}
				if bit == 0 || n != 4 || a[3] != 32 || r.modes&bit != 0 || (r.modeEntry != 0 && r.modeEntry != a[1]) {
					return fail("unsupported or duplicate binary32 mode")
				}
				r.modes |= bit
				r.modeEntry = a[1]
			}
		case 17:
			if n != 2 {
				return fail("malformed capability")
			}
			bit := uint32(0)
			switch a[1] {
			case 1:
				bit = 1
			case 4464:
				bit = 2
			case 4466:
				bit = 4
			case 4467:
				bit = 8
			}
			if bit == 0 || r.capabilities&bit != 0 {
				return fail("unsupported or duplicate capability")
			}
			r.capabilities |= bit
		case 10, 11:
			start := 1
			want := "SPV_KHR_float_controls"
			if op == 11 {
				start = 2
				want = "GLSL.std.450"
			}
			if n <= start {
				return fail("malformed literal")
			}
			literal := make([]byte, 4*(n-start))
			for i, v := range a[start:] {
				binary.LittleEndian.PutUint32(literal[4*i:], v)
			}
			expected := append([]byte(want), 0)
			expected = append(expected, make([]byte, (4-len(expected)%4)%4)...)
			if !bytes.Equal(literal, expected) {
				return fail("unsupported extension or import")
			}
			if op == 10 {
				r.extensions++
			} else {
				if !validID(a[1]) {
					return fail("invalid import ID")
				}
				r.imports++
				importID = a[1]
			}
		case 129, 131, 133, 136:
			if n != 5 || !validID(a[1]) || !validID(a[2]) || !validID(a[3]) || !validID(a[4]) {
				return fail("malformed arithmetic")
			}
			if _, ok := r.arithmetic[a[2]]; ok {
				return fail("duplicate arithmetic result")
			}
			r.arithmetic[a[2]] = a[1]
			r.arithmeticOps[op]++
		case 12:
			if n != 6 || !validID(a[1]) || !validID(a[2]) || !validID(a[3]) || a[4] != 27 || !validID(a[5]) {
				return fail("only GLSL Exp is supported")
			}
			r.expTypes = append(r.expTypes, a[1])
			expImports = append(expImports, a[3])
		case 112: // UIntToF in the E4M3 subnormal decoder; require scalar binary32 later.
			if n != 4 || !validID(a[1]) || !validID(a[2]) || !validID(a[3]) {
				return fail("malformed unsigned conversion")
			}
			r.expTypes = append(r.expTypes, a[1])
		case 24, 48, 49, 50, 51, 52, 73, 74, 75, 109, 110, 111, 115, 116, 127, 140, 141, 142, 143, 144, 145, 146, 147, 148, 207, 208, 209, 210, 211, 212, 213, 214, 215, 265, 266, 269, 331, 332, 350, 352, 355, 358, 5632, 5633:
			return fail("unsupported alternate floating-point or indirect declaration operation")
		case 0, 2, 3, 4, 5, 6, 7, 8, 19, 33, 41, 42, 44, 46,
			54, 55, 56, 57, 61, 62, 65, 66, 79, 80, 81, 82, 83,
			124, 126, 128, 130, 132, 134, 135, 137, 138, 139,
			154, 155, 156, 157, 164, 165, 166, 167, 168, 169,
			170, 171, 172, 173, 174, 175, 176, 177, 178, 179,
			180, 181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 191,
			194, 195, 196, 197, 198, 199, 200, 224, 225,
			245, 246, 247, 248, 249, 250, 251, 253, 254, 255, 317, 330:
			// Ordinary debug, integer, load/store and control-flow instructions.
			// Their semantics and SSA validity MUST be checked by spirv-val and
			// the independent arithmetic/data-flow witness, not this inspector.
		default:
			return fail("unrecognized instruction")
		}
		p += n
	}
	if len(expImports) == 0 {
		return fail("missing Exp")
	}
	for _, id := range expImports {
		if id != importID {
			return fail("Exp import mismatch")
		}
	}
	return r, nil
}

func (r *packedSPIRVReflection) definition(id, op uint32, size int) []uint32 {
	a := r.defs[id]
	if len(a) != size || a[0]&0xffff != op {
		return nil
	}
	return a
}
func (r *packedSPIRVReflection) decorated(id, kind, value uint32) bool {
	got, ok := r.decorations[[2]uint32{id, kind}]
	return ok && got == value
}
func (r *packedSPIRVReflection) member(id, index, kind, value uint32) bool {
	got, ok := r.members[[3]uint32{id, index, kind}]
	return ok && got == value
}
func (r *packedSPIRVReflection) scalar(id uint32, unsigned bool) bool {
	if unsigned {
		a := r.definition(id, 21, 4)
		return a != nil && a[2] == 32 && a[3] == 0
	}
	a := r.definition(id, 22, 3)
	return a != nil && a[2] == 32
}

func validatePackedAttentionReflection(r *packedSPIRVReflection) error {
	fail := packedContractError
	if r.entries != 1 || r.locals != 1 || r.memoryModels != 1 || r.entry != r.localEntry || r.entry != r.modeEntry || r.local != [3]uint32{128, 1, 1} || r.modes != 7 || r.capabilities != 15 || r.extensions != 1 || r.imports != 1 {
		return fail("entry, local size, memory model or binary32 profile mismatch")
	}
	if len(r.pushes) != 1 || len(r.buffers) != 5 {
		return fail("expected one push block and five storage buffers")
	}
	for id, ty := range r.arithmetic {
		if !r.scalar(ty, false) || !r.decorated(id, 42, 0) {
			return fail("arithmetic is not scalar binary32 NoContraction")
		}
	}
	for _, op := range []uint32{129, 131, 133, 136} {
		if r.arithmeticOps[op] == 0 {
			return fail("missing attention arithmetic class")
		}
	}
	for _, ty := range r.expTypes {
		if !r.scalar(ty, false) {
			return fail("Exp or conversion is not binary32")
		}
	}
	for _, a := range r.defs {
		if a[0]&0xffff == 23 {
			ty := r.definition(a[2], 21, 4)
			if ty == nil {
				return fail("non-integer vector")
			}
		}
	}
	push := r.definition(r.pushes[0], 59, 4)
	if push == nil || push[3] != 9 {
		return fail("push variable mismatch")
	}
	ptr := r.definition(push[1], 32, 4)
	if ptr == nil || ptr[2] != 9 {
		return fail("push pointer mismatch")
	}
	st := r.definition(ptr[3], 30, 10)
	if st == nil || !r.decorated(ptr[3], 2, 0) {
		return fail("expected eight-field 32-byte push block")
	}
	for m := uint32(0); m < 8; m++ {
		if !r.member(ptr[3], m, 35, 4*m) {
			return fail("push offset mismatch")
		}
		if m < 5 {
			ty := r.definition(st[m+2], 21, 4)
			if ty == nil || ty[2] != 32 || ty[3] != 1 {
				return fail("push signed integer mismatch")
			}
		} else if !r.scalar(st[m+2], m > 5) {
			return fail("push scale or unsigned metadata mismatch")
		}
	}
	seen := uint32(0)
	for _, id := range r.buffers {
		v := r.definition(id, 59, 4)
		slot, ok := r.decorations[[2]uint32{id, 33}]
		if v == nil || v[3] != 12 || !ok || slot >= 5 || seen&(1<<slot) != 0 || !r.decorated(id, 34, 0) {
			return fail("descriptor set/binding mismatch")
		}
		seen |= 1 << slot
		p := r.definition(v[1], 32, 4)
		if p == nil || p[2] != 12 {
			return fail("storage pointer mismatch")
		}
		s := r.definition(p[3], 30, 3)
		if s == nil || !r.decorated(p[3], 2, 0) || !r.member(p[3], 0, 35, 0) {
			return fail("storage block mismatch")
		}
		a := r.definition(s[2], 29, 3)
		if a == nil || !r.decorated(s[2], 6, 4) || !r.scalar(a[2], slot == 1 || slot == 4) {
			return fail("storage array scalar or stride mismatch")
		}
		if r.member(p[3], 0, 24, 0) != (slot < 3) || r.member(p[3], 0, 25, 0) != (slot == 4) {
			return fail("storage access mismatch")
		}
	}
	// Supported Workgroup types have four-byte scalar alignment and no structs,
	// vectors or explicit-layout padding. Count declarations, not source text.
	// Driver allocation/granularity and device limits remain separate admission.
	var total uint64
	for _, id := range r.shared {
		v := r.definition(id, 59, 4)
		if v == nil || v[3] != 4 {
			return fail("workgroup variable mismatch")
		}
		p := r.definition(v[1], 32, 4)
		if p == nil || p[2] != 4 {
			return fail("workgroup pointer mismatch")
		}
		ty, n := p[3], uint32(1)
		if a := r.definition(ty, 28, 4); a != nil {
			if stride, ok := r.decorations[[2]uint32{ty, 6}]; ok && stride != 4 {
				return fail("unsupported workgroup array stride")
			}
			c := r.definition(a[3], 43, 4)
			if c == nil || !r.scalar(c[1], true) || c[3] == 0 {
				return fail("workgroup array bound mismatch")
			}
			ty, n = a[2], c[3]
		}
		f := r.definition(ty, 22, 3)
		i := r.definition(ty, 21, 4)
		if (f == nil || f[2] != 32) && (i == nil || i[2] != 32 || i[3] > 1) {
			return fail("unsupported workgroup layout")
		}
		total += uint64(n) * 4
		if total > 17936 {
			return fail("workgroup declarations exceed fixed budget")
		}
	}
	if total != 17936 {
		return fail("workgroup declaration byte count mismatch")
	}
	return nil
}
