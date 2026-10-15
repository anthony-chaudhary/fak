package compute

import (
	"context"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"runtime"
	"strings"
	"testing"
	"time"
)

type q2PreciseSPIRVInstruction struct {
	opcode     string
	resultType string
	args       []string
	position   int
}

type q2PreciseSPIRVModule struct {
	instructions  map[string]q2PreciseSPIRVInstruction
	noContraction map[string]bool
	stores        []q2PreciseSPIRVInstruction
}

// TestVulkanQ2PreciseOrder compiles the scalar mode-0 and token-panel Q2 paths
// and checks the optimized SPIR-V dataflow. Source-level `precise` spelling is
// insufficient: each dot, term, accumulation, and output reduction operation
// in the repaired path must retain NoContraction after optimization.
// fak-test:runtime medium est=5s lane=default
func TestVulkanQ2PreciseOrder(t *testing.T) {
	glslc := q2PreciseVulkanTool(t, "glslc")
	spirvDis := q2PreciseVulkanTool(t, "spirv-dis")
	for _, shader := range []string{"q2k_matvec.comp", "v41_q2k_token_panel.comp"} {
		t.Run(shader, func(t *testing.T) {
			dir := t.TempDir()
			source := filepath.Join("shaders", shader)
			binary := filepath.Join(dir, shader+".spv")
			assembly := filepath.Join(dir, shader+".spvasm")
			q2PreciseRunTool(t, glslc, "-O", "--target-env=vulkan1.2", "-fshader-stage=comp", source, "-o", binary)
			q2PreciseRunTool(t, spirvDis, binary, "-o", assembly)
			text, err := os.ReadFile(assembly)
			if err != nil {
				t.Fatal(err)
			}
			module := q2PreciseParseSPIRV(string(text))
			if err := module.checkQ2Order(); err != nil {
				t.Fatalf("optimized %s lost the precise Q2 order: %v", shader, err)
			}
		})
	}
}

func q2PreciseVulkanTool(t *testing.T, name string) string {
	t.Helper()
	if path, err := exec.LookPath(name); err == nil {
		return path
	}
	candidates := []string{}
	if sdk := os.Getenv("VULKAN_SDK"); sdk != "" {
		for _, executable := range []string{name, name + ".exe"} {
			candidates = append(candidates, filepath.Join(sdk, "Bin", executable), filepath.Join(sdk, "bin", executable))
		}
	}
	for _, candidate := range candidates {
		if info, err := os.Stat(candidate); err == nil && !info.IsDir() {
			return candidate
		}
	}
	t.Skipf("compiler witness unavailable: %s is required", name)
	return ""
}

func q2PreciseRunTool(t *testing.T, tool string, args ...string) {
	t.Helper()
	if runtime.GOOS != "windows" && strings.HasSuffix(strings.ToLower(tool), ".exe") {
		for i, arg := range args {
			if len(arg) > len("/mnt/c/") && strings.HasPrefix(arg, "/mnt/") && arg[6] == '/' {
				args[i] = strings.ToUpper(arg[5:6]) + `:\` + strings.ReplaceAll(arg[7:], "/", `\`)
			}
		}
	}
	ctx, cancel := context.WithTimeout(context.Background(), 20*time.Second)
	defer cancel()
	if output, err := exec.CommandContext(ctx, tool, args...).CombinedOutput(); err != nil {
		t.Fatalf("%s failed: %v\n%s", tool, err, output)
	}
}

func q2PreciseParseSPIRV(assembly string) q2PreciseSPIRVModule {
	module := q2PreciseSPIRVModule{
		instructions:  make(map[string]q2PreciseSPIRVInstruction),
		noContraction: make(map[string]bool),
	}
	position := 0
	for _, line := range strings.Split(assembly, "\n") {
		fields := strings.Fields(line)
		if len(fields) >= 3 && fields[0] == "OpDecorate" && fields[2] == "NoContraction" {
			module.noContraction[fields[1]] = true
			continue
		}
		if len(fields) == 3 && fields[0] == "OpStore" {
			module.stores = append(module.stores, q2PreciseSPIRVInstruction{opcode: fields[0], args: fields[1:], position: position})
			position++
			continue
		}
		if len(fields) < 5 || fields[1] != "=" || !strings.HasPrefix(fields[0], "%") || !strings.HasPrefix(fields[2], "Op") {
			continue
		}
		module.instructions[fields[0]] = q2PreciseSPIRVInstruction{
			opcode: fields[2], resultType: fields[3], args: fields[4:], position: position,
		}
		position++
	}
	return module
}

func (module q2PreciseSPIRVModule) checkQ2Order() error {
	termID, term, ok := module.q2Term()
	if !ok {
		return fmt.Errorf("missing NoContraction four-product left-associated dot and scaled-minus-min term")
	}
	perJ := module.firstPreciseAddAfter(term.position, func(_ string, instruction q2PreciseSPIRVInstruction) bool {
		return q2PreciseHasArg(instruction, termID)
	})
	if perJ.position == 0 {
		return fmt.Errorf("per-j sum accumulation lacks NoContraction")
	}
	superblock := module.firstPreciseAddAfter(perJ.position, func(_ string, instruction q2PreciseSPIRVInstruction) bool {
		return module.privateAccumulatorInputs(instruction)
	})
	if superblock.position == 0 {
		return fmt.Errorf("superblock accumulator lacks NoContraction")
	}
	reduction := module.firstPreciseAddAfter(superblock.position, func(_ string, instruction q2PreciseSPIRVInstruction) bool {
		return module.loadedFromStorage(instruction, "Workgroup")
	})
	if reduction.position == 0 {
		return fmt.Errorf("Q2 output workgroup reduction lacks NoContraction")
	}
	return nil
}

func (module q2PreciseSPIRVModule) q2Term() (string, q2PreciseSPIRVInstruction, bool) {
	for id, instruction := range module.instructions {
		if instruction.opcode != "OpFSub" || !module.noContraction[id] || len(instruction.args) != 2 {
			continue
		}
		left, leftOK := module.instructions[instruction.args[0]]
		right, rightOK := module.instructions[instruction.args[1]]
		if !leftOK || !rightOK || left.opcode != "OpFMul" || right.opcode != "OpFMul" ||
			!module.noContraction[instruction.args[0]] || !module.noContraction[instruction.args[1]] {
			continue
		}
		for _, candidate := range left.args {
			if module.preciseLeftDot(candidate) && module.otherInputIsPreciseMul(left, candidate) &&
				module.preciseMinXSum(right, instruction.position) && !module.reachesFma(id, nil) {
				return id, instruction, true
			}
		}
	}
	return "", q2PreciseSPIRVInstruction{}, false
}

func (module q2PreciseSPIRVModule) preciseMinXSum(minTerm q2PreciseSPIRVInstruction, before int) bool {
	for _, candidate := range minTerm.args {
		if module.preciseComponentSum(candidate) && module.otherInputIsPreciseMul(minTerm, candidate) {
			return true
		}
		load, ok := module.instructions[candidate]
		if !ok || load.opcode != "OpLoad" || len(load.args) != 1 {
			continue
		}
		loadPointer, ok := module.instructions[load.args[0]]
		if !ok || loadPointer.opcode != "OpAccessChain" || len(loadPointer.args) == 0 {
			continue
		}
		base := loadPointer.args[0]
		for _, store := range module.stores {
			if store.position >= before || len(store.args) != 2 {
				continue
			}
			storePointer, ok := module.instructions[store.args[0]]
			if ok && storePointer.opcode == "OpAccessChain" && len(storePointer.args) > 0 && storePointer.args[0] == base &&
				module.preciseComponentSum(store.args[1]) && module.otherInputIsPreciseMul(minTerm, candidate) {
				return true
			}
		}
	}
	return false
}

func (module q2PreciseSPIRVModule) preciseComponentSum(root string) bool {
	first, ok := module.preciseBinary(root, "OpFAdd")
	if !ok {
		return false
	}
	second, ok := module.preciseBinary(first.args[0], "OpFAdd")
	if !ok {
		return false
	}
	third, ok := module.preciseBinary(second.args[0], "OpFAdd")
	if !ok {
		return false
	}
	for _, leaf := range []string{third.args[0], third.args[1], second.args[1], first.args[1]} {
		producer, ok := module.instructions[leaf]
		if !ok || producer.opcode != "OpCompositeExtract" {
			return false
		}
	}
	return true
}

func (module q2PreciseSPIRVModule) otherInputIsPreciseMul(parent q2PreciseSPIRVInstruction, known string) bool {
	for _, candidate := range parent.args {
		if candidate != known {
			_, ok := module.preciseBinary(candidate, "OpFMul")
			return ok
		}
	}
	return false
}

func (module q2PreciseSPIRVModule) reachesFma(id string, seen map[string]bool) bool {
	if seen == nil {
		seen = make(map[string]bool)
	}
	if seen[id] {
		return false
	}
	seen[id] = true
	instruction, ok := module.instructions[id]
	if !ok {
		return false
	}
	if instruction.opcode == "OpFma" || (instruction.opcode == "OpExtInst" && q2PreciseHasArg(instruction, "Fma")) {
		return true
	}
	for _, arg := range instruction.args {
		if module.reachesFma(arg, seen) {
			return true
		}
	}
	return false
}

func (module q2PreciseSPIRVModule) preciseLeftDot(root string) bool {
	first, ok := module.preciseBinary(root, "OpFAdd")
	if !ok {
		return false
	}
	second, ok := module.preciseBinary(first.args[0], "OpFAdd")
	if !ok {
		return false
	}
	third, ok := module.preciseBinary(second.args[0], "OpFAdd")
	if !ok {
		return false
	}
	for _, leaf := range []string{third.args[0], third.args[1], second.args[1], first.args[1]} {
		if _, ok := module.preciseBinary(leaf, "OpFMul"); !ok {
			return false
		}
	}
	return true
}

func (module q2PreciseSPIRVModule) preciseBinary(id, opcode string) (q2PreciseSPIRVInstruction, bool) {
	instruction, ok := module.instructions[id]
	return instruction, ok && instruction.opcode == opcode && len(instruction.args) == 2 && module.noContraction[id]
}

func (module q2PreciseSPIRVModule) firstPreciseAddAfter(position int, accept func(string, q2PreciseSPIRVInstruction) bool) q2PreciseSPIRVInstruction {
	var found q2PreciseSPIRVInstruction
	for id, instruction := range module.instructions {
		if instruction.opcode == "OpFAdd" && instruction.position > position && module.noContraction[id] &&
			(found.position == 0 || instruction.position < found.position) && accept(id, instruction) {
			found = instruction
		}
	}
	return found
}

func (module q2PreciseSPIRVModule) privateAccumulatorInputs(instruction q2PreciseSPIRVInstruction) bool {
	if len(instruction.args) != 2 {
		return false
	}
	for _, id := range instruction.args {
		producer, ok := module.instructions[id]
		if !ok {
			return false
		}
		if producer.opcode == "OpPhi" {
			continue
		}
		if producer.opcode != "OpLoad" || len(producer.args) != 1 {
			return false
		}
		pointer, ok := module.instructions[producer.args[0]]
		if !ok || !strings.Contains(pointer.resultType, "Function") {
			return false
		}
	}
	return true
}

func (module q2PreciseSPIRVModule) loadedFromStorage(instruction q2PreciseSPIRVInstruction, storage string) bool {
	if len(instruction.args) != 2 {
		return false
	}
	for _, id := range instruction.args {
		load, ok := module.instructions[id]
		if !ok || load.opcode != "OpLoad" || len(load.args) != 1 {
			return false
		}
		pointer, ok := module.instructions[load.args[0]]
		if !ok || !strings.Contains(pointer.resultType, storage) {
			return false
		}
	}
	return true
}

func q2PreciseHasArg(instruction q2PreciseSPIRVInstruction, id string) bool {
	for _, arg := range instruction.args {
		if arg == id {
			return true
		}
	}
	return false
}

// This source contract pins both row roles, the guarded odd tail, the single
// activation prelude, and the complete ordered accumulation/reduction body.
// It is not compiled arithmetic or hardware parity evidence. The compiler test
// above finds one term and cannot by itself qualify both paired-row paths.
// fak-test:runtime fast est=1ms lane=default
func TestVulkanQ2TokenPanelActivationReuseSource(t *testing.T) {
	t.Parallel()
	data, err := os.ReadFile(filepath.Join("shaders", "v41_q2k_token_panel.comp"))
	if err != nil {
		t.Fatal(err)
	}
	source := string(data)
	if err := q2TokenPanelActivationReuseSourceContract(source); err != nil {
		t.Fatal(err)
	}
	// A complete precise chain can still be wrong if it uses the other row's
	// quant. The negative control must reject this otherwise plausible mutation.
	const row1Product = "qvRow1.y * xv.y"
	if strings.Count(source, row1Product) != 1 {
		t.Fatal("negative-control row-1 product is not unique")
	}
	mutated := strings.Replace(source, row1Product, "qvRow0.y * xv.y", 1)
	if err := q2TokenPanelActivationReuseSourceContract(mutated); err == nil {
		t.Fatal("source contract accepted a row-0 quant in the row-1 term")
	}
}

func q2TokenPanelActivationReuseSourceContract(source string) error {
	compact := func(text string) string { return strings.Join(strings.Fields(text), " ") }
	// Compare active text, so a comment cannot stand in for a declaration.
	// Exact preamble equality also rejects extra preprocessor directives.
	source = regexp.MustCompile(`(?s)/\*.*?\*/|//[^\n]*`).ReplaceAllString(source, " ")
	const main = "void main() {"
	start := strings.Index(source, main)
	if start < 0 || strings.Count(source, main) != 1 {
		return fmt.Errorf("paired-row shader must have one main body")
	}
	const preamble = `
#version 450
layout(local_size_x = 64) in;
const uint OUT_ROWS = 2u;
const uint LANES = 64u;
const uint MAX_TOKENS = 8u;
layout(std430, set = 0, binding = 0) readonly buffer W { uint raw[]; };
layout(std430, set = 0, binding = 1) readonly buffer X { vec4 x4[]; };
layout(std430, set = 0, binding = 2) writeonly buffer Y { float y[]; };
layout(push_constant) uniform PC { int outDim; int inDim; int tokens; } pc;
shared float red[MAX_TOKENS * OUT_ROWS * LANES];
vec4 bytesOf(uint w) {
    return vec4(float(w & 255u), float((w >> 8u) & 255u), float((w >> 16u) & 255u), float(w >> 24u));
}
`
	if compact(source[:start]) != compact(preamble) {
		return fmt.Errorf("paired-row shader version, geometry, bindings, or quant-byte decode changed")
	}
	// Only the row suffix changes this scalar-order chain. Sharing xv/xsum
	// must not share any quant, scale, term, or per-j sum between output rows.
	rowTerm := func(row string) string {
		return strings.ReplaceAll(`
precise float q0ROW = qvROW.x * xv.x;
precise float q1ROW = qvROW.y * xv.y;
precise float q2ROW = qvROW.z * xv.z;
precise float q3ROW = qvROW.w * xv.w;
precise float q01ROW = q0ROW + q1ROW;
precise float q012ROW = q01ROW + q2ROW;
precise float qdotROW = q012ROW + q3ROW;
precise float scaledDotROW = dlROW * qdotROW;
precise float minTermROW = mlROW * xsum;
precise float termROW = scaledDotROW - minTermROW;
precise float nextSumROW = sumROW[t] + termROW;
sumROW[t] = nextSumROW;
`, "ROW", row)
	}
	want := `
void main() {
    uint tid = gl_LocalInvocationID.x;
    uint rowBase = gl_WorkGroupID.x * OUT_ROWS;
    uint outDim = uint(pc.outDim), inDim = uint(pc.inDim);
    bool hasRow1 = rowBase + 1u < outDim;
    uint tokens = uint(pc.tokens), nsb = inDim / 256u;
    uint wordsPerRow = nsb * 21u;
    uint t16 = tid & 15u, h = t16 >> 3u, s = (t16 >> 2u) & 1u;
    uint lq = (t16 & 3u) * 4u;
    precise float acc[MAX_TOKENS * OUT_ROWS];
    for (uint t = 0u; t < tokens; ++t) {
        for (uint r = 0u; r < OUT_ROWS; ++r) acc[t * OUT_ROWS + r] = 0.0;
    }
    for (uint sb = tid >> 4u; sb < nsb; sb += LANES / 16u) {
        if (rowBase >= outDim) break;
        uint baseRow0 = rowBase * wordsPerRow + sb * 21u;
        uint qRow0 = raw[baseRow0 + 4u + t16];
        uint sc0Row0 = raw[baseRow0 + h * 2u];
        uint sc1Row0 = raw[baseRow0 + h * 2u + 1u];
        vec2 ddRow0 = unpackHalf2x16(raw[baseRow0 + 20u]);
        uint scjRow0[4];
        scjRow0[0] = (sc0Row0 >> (s * 8u)) & 255u;
        scjRow0[1] = (sc0Row0 >> ((2u + s) * 8u)) & 255u;
        scjRow0[2] = (sc1Row0 >> (s * 8u)) & 255u;
        scjRow0[3] = (sc1Row0 >> ((2u + s) * 8u)) & 255u;
        uint qRow1;
        vec2 ddRow1;
        uint scjRow1[4];
        if (hasRow1) {
            uint baseRow1 = (rowBase + 1u) * wordsPerRow + sb * 21u;
            qRow1 = raw[baseRow1 + 4u + t16];
            uint sc0Row1 = raw[baseRow1 + h * 2u];
            uint sc1Row1 = raw[baseRow1 + h * 2u + 1u];
            ddRow1 = unpackHalf2x16(raw[baseRow1 + 20u]);
            scjRow1[0] = (sc0Row1 >> (s * 8u)) & 255u;
            scjRow1[1] = (sc0Row1 >> ((2u + s) * 8u)) & 255u;
            scjRow1[2] = (sc1Row1 >> (s * 8u)) & 255u;
            scjRow1[3] = (sc1Row1 >> ((2u + s) * 8u)) & 255u;
        }
        precise float sumRow0[MAX_TOKENS];
        precise float sumRow1[MAX_TOKENS];
        for (uint t = 0u; t < tokens; ++t) {
            sumRow0[t] = 0.0;
            sumRow1[t] = 0.0;
        }
        for (uint j = 0u; j < 4u; ++j) {
            vec4 qvRow0 = bytesOf((qRow0 >> (2u * j)) & 0x03030303u);
            precise float dlRow0 = ddRow0.x * float(scjRow0[j] & 15u);
            precise float mlRow0 = ddRow0.y * float(scjRow0[j] >> 4u);
            vec4 qvRow1;
            precise float dlRow1;
            precise float mlRow1;
            if (hasRow1) {
                qvRow1 = bytesOf((qRow1 >> (2u * j)) & 0x03030303u);
                dlRow1 = ddRow1.x * float(scjRow1[j] & 15u);
                mlRow1 = ddRow1.y * float(scjRow1[j] >> 4u);
            }
            uint idx = (sb * 256u + h * 128u + j * 32u + s * 16u + lq) >> 2u;
            for (uint t = 0u; t < tokens; ++t) {
                vec4 xv = x4[t * (inDim >> 2u) + idx];
                precise float x01 = xv.x + xv.y;
                precise float x012 = x01 + xv.z;
                precise float xsum = x012 + xv.w;
` + rowTerm("Row0") + `
                if (hasRow1) {
` + rowTerm("Row1") + `
                }
            }
        }
        for (uint t = 0u; t < tokens; ++t) {
            uint accIdxRow0 = t * OUT_ROWS;
            precise float nextAccRow0 = acc[accIdxRow0] + sumRow0[t];
            acc[accIdxRow0] = nextAccRow0;
            if (hasRow1) {
                uint accIdxRow1 = t * OUT_ROWS + 1u;
                precise float nextAccRow1 = acc[accIdxRow1] + sumRow1[t];
                acc[accIdxRow1] = nextAccRow1;
            }
        }
    }
    for (uint t = 0u; t < tokens; ++t) {
        for (uint r = 0u; r < OUT_ROWS; ++r) red[(t * OUT_ROWS + r) * LANES + tid] = acc[t * OUT_ROWS + r];
    }
    barrier();
    for (uint stride = LANES / 2u; stride > 0u; stride >>= 1u) {
        if (tid < stride) {
            for (uint t = 0u; t < tokens; ++t) {
                for (uint r = 0u; r < OUT_ROWS; ++r) {
                    uint off = (t * OUT_ROWS + r) * LANES + tid;
                    precise float reduced = red[off] + red[off + stride];
                    red[off] = reduced;
                }
            }
        }
        barrier();
    }
    if (tid < OUT_ROWS && rowBase + tid < outDim) {
        for (uint t = 0u; t < tokens; ++t) y[t * outDim + rowBase + tid] = red[(t * OUT_ROWS + tid) * LANES];
    }
}
`
	if compact(source[start:]) != compact(want) {
		return fmt.Errorf("paired-row activation reuse lost its source arithmetic, guard, or reduction contract")
	}
	return nil
}
