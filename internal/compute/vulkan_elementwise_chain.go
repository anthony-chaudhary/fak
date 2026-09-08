package compute

import (
	"crypto/sha256"
	"encoding/binary"
	"encoding/hex"
	"fmt"
	"math"
)

// elementwiseOp is the deliberately small vocabulary admitted by the Vulkan
// straight-line lowering seam. Inputs are implicit values [0, Inputs); every
// instruction appends one value and may only reference earlier values.
type elementwiseOp uint8

const (
	elementwiseSigmoid elementwiseOp = iota + 1
	elementwiseMul
	elementwiseAdd
)

const (
	elementwiseUnusedOperand       = ^uint16(0)
	maxVulkanElementwiseInputs     = 4
	maxVulkanElementwiseOperations = 16
	vulkanElementwiseKernelSwiGLU  = "swiglu.spv"
)

type elementwiseInstruction struct {
	Op   elementwiseOp
	A, B uint16
}

type elementwiseChain struct {
	Inputs uint8
	Ops    []elementwiseInstruction
}

// vulkanElementwiseDispatch is the stable descriptor produced by lowering
// a typed chain. Digest is suitable for profiling joins; it is derived only from
// canonical op bytes, never pointer identity or map iteration order.
type vulkanElementwiseDispatch struct {
	Digest               string
	Kernel               string
	Dispatches           int
	IntermediateBarriers int
}

// recurrentSwiGLUElementwiseChain names the first admitted chain: the dense MLP
// activation used after both ordinary attention and recurrent Qwen GDN layers.
// Inputs are gate=0 and up=1; results occupy slots 2, 3, and 4.
func recurrentSwiGLUElementwiseChain() elementwiseChain {
	return elementwiseChain{
		Inputs: 2,
		Ops: []elementwiseInstruction{
			{Op: elementwiseSigmoid, A: 0, B: elementwiseUnusedOperand},
			{Op: elementwiseMul, A: 0, B: 2},
			{Op: elementwiseMul, A: 3, B: 1},
		},
	}
}

func validateElementwiseChain(chain elementwiseChain) error {
	if chain.Inputs == 0 || chain.Inputs > maxVulkanElementwiseInputs {
		return fmt.Errorf("compute: Vulkan elementwise chain has %d inputs, want 1..%d", chain.Inputs, maxVulkanElementwiseInputs)
	}
	if len(chain.Ops) == 0 || len(chain.Ops) > maxVulkanElementwiseOperations {
		return fmt.Errorf("compute: Vulkan elementwise chain has %d operations, want 1..%d", len(chain.Ops), maxVulkanElementwiseOperations)
	}
	for i, instruction := range chain.Ops {
		limit := uint16(chain.Inputs) + uint16(i)
		switch instruction.Op {
		case elementwiseSigmoid:
			if instruction.A >= limit || instruction.B != elementwiseUnusedOperand {
				return fmt.Errorf("compute: Vulkan elementwise sigmoid %d has invalid operands %d/%d", i, instruction.A, instruction.B)
			}
		case elementwiseMul, elementwiseAdd:
			if instruction.A >= limit || instruction.B >= limit {
				return fmt.Errorf("compute: Vulkan elementwise binary op %d has invalid operands %d/%d", i, instruction.A, instruction.B)
			}
		default:
			return fmt.Errorf("compute: Vulkan elementwise operation %d has unknown opcode %d", i, instruction.Op)
		}
	}
	return nil
}

func elementwiseChainDigest(chain elementwiseChain) (string, error) {
	if err := validateElementwiseChain(chain); err != nil {
		return "", err
	}
	canonical := make([]byte, 0, 35+len(chain.Ops)*5)
	canonical = append(canonical, []byte("fak/vulkan-elementwise-chain/v1\x00")...)
	canonical = append(canonical, chain.Inputs, byte(len(chain.Ops)))
	var operands [4]byte
	for _, instruction := range chain.Ops {
		canonical = append(canonical, byte(instruction.Op))
		binary.LittleEndian.PutUint16(operands[0:2], instruction.A)
		binary.LittleEndian.PutUint16(operands[2:4], instruction.B)
		canonical = append(canonical, operands[:]...)
	}
	digest := sha256.Sum256(canonical)
	return hex.EncodeToString(digest[:]), nil
}

func equalElementwiseChain(a, b elementwiseChain) bool {
	if a.Inputs != b.Inputs || len(a.Ops) != len(b.Ops) {
		return false
	}
	for i := range a.Ops {
		if a.Ops[i] != b.Ops[i] {
			return false
		}
	}
	return true
}

// lowerVulkanElementwiseChain fails closed for every expression except the one
// profiled candidate. Adding an opcode does not make it executable: a separate
// exact-chain arm and oracle are required before it can select a pipeline.
func lowerVulkanElementwiseChain(chain elementwiseChain) (vulkanElementwiseDispatch, error) {
	digest, err := elementwiseChainDigest(chain)
	if err != nil {
		return vulkanElementwiseDispatch{}, err
	}
	if !equalElementwiseChain(chain, recurrentSwiGLUElementwiseChain()) {
		return vulkanElementwiseDispatch{}, fmt.Errorf("compute: unsupported chain %s for Vulkan elementwise lowering", digest)
	}
	return vulkanElementwiseDispatch{
		Digest:               digest,
		Kernel:               vulkanElementwiseKernelSwiGLU,
		Dispatches:           1,
		IntermediateBarriers: 0,
	}, nil
}

func mustLowerVulkanElementwiseChain(chain elementwiseChain) vulkanElementwiseDispatch {
	dispatch, err := lowerVulkanElementwiseChain(chain)
	if err != nil {
		panic(err)
	}
	return dispatch
}

var recurrentSwiGLUVulkanDispatch = mustLowerVulkanElementwiseChain(recurrentSwiGLUElementwiseChain())

// evaluateElementwiseChain is the independent scalar oracle for admitted
// expressions. Device lowering never calls it.
func evaluateElementwiseChain(chain elementwiseChain, inputs ...[]float32) ([]float32, error) {
	if err := validateElementwiseChain(chain); err != nil {
		return nil, err
	}
	if len(inputs) != int(chain.Inputs) {
		return nil, fmt.Errorf("compute: Vulkan elementwise oracle got %d inputs, want %d", len(inputs), chain.Inputs)
	}
	n := len(inputs[0])
	for i := range inputs {
		if len(inputs[i]) != n {
			return nil, fmt.Errorf("compute: Vulkan elementwise oracle input %d has length %d, want %d", i, len(inputs[i]), n)
		}
	}
	out := make([]float32, n)
	values := make([]float32, int(chain.Inputs)+len(chain.Ops))
	for index := 0; index < n; index++ {
		for input := range inputs {
			values[input] = inputs[input][index]
		}
		for operation, instruction := range chain.Ops {
			var value float32
			switch instruction.Op {
			case elementwiseSigmoid:
				x := values[instruction.A]
				value = 1 / (1 + float32(math.Exp(float64(-x))))
			case elementwiseMul:
				value = values[instruction.A] * values[instruction.B]
			case elementwiseAdd:
				value = values[instruction.A] + values[instruction.B]
			}
			values[int(chain.Inputs)+operation] = value
		}
		out[index] = values[len(values)-1]
	}
	return out, nil
}
