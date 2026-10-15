package compute

import (
	"context"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
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
