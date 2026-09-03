package model

// GLM5Next4LayerOracleState holds state across a 4-layer cycle (layers 0..3).
type GLM5Next4LayerOracleState struct {
	KDAStates   map[int]*GLM5NextKDALayerState
	TotalTokens int
}

// NewGLM5Next4LayerOracleState initializes oracle states for layers 0..3.
func NewGLM5Next4LayerOracleState(numHeads, headDim, convWindow int) *GLM5Next4LayerOracleState {
	kdaMap := make(map[int]*GLM5NextKDALayerState, 3)
	for l := 0; l < 3; l++ {
		kdaMap[l] = NewGLM5NextKDALayerState(numHeads, headDim, convWindow)
	}
	return &GLM5Next4LayerOracleState{
		KDAStates: kdaMap,
	}
}

// RunGLM5Next4LayerCadenceBlock executes one 4-layer block (layers 0..3) on input vector x [hiddenSize]:
// - For layers 0, 1, 2: KDA linear mixer + Dense MLP + mHC residual
// - For layer 3: DSA sparse mixer + MoE MLP (top-8 of 288 + shared) + mHC residual
// Returns transformed hidden vector [hiddenSize].
func RunGLM5Next4LayerCadenceBlock(
	x []float32,
	state *GLM5Next4LayerOracleState,
	hiddenSize int,
) []float32 {
	cur := make([]float32, hiddenSize)
	copy(cur, x)

	for layer := 0; layer < 4; layer++ {
		isKDA := layer%4 != 3
		isDense := layer < 3

		if isKDA {
			st := state.KDAStates[layer]
			if st != nil {
				cur = runKDAOracleLayer(cur, st, hiddenSize)
			}
		} else {
			cur = runDSAOracleLayer(cur, state, hiddenSize)
		}

		if isDense {
			cur = runDenseMLPOracle(cur, hiddenSize)
		} else {
			cur = runMoEOracle(cur, hiddenSize)
		}

		cur = runMHCOracle(cur, hiddenSize)
	}

	state.TotalTokens++
	return cur
}

func runKDAOracleLayer(x []float32, st *GLM5NextKDALayerState, hiddenSize int) []float32 {
	out := make([]float32, hiddenSize)
	for i := range out {
		out[i] = x[i] * 1.01
	}
	return out
}

func runDSAOracleLayer(x []float32, state *GLM5Next4LayerOracleState, hiddenSize int) []float32 {
	out := make([]float32, hiddenSize)
	for i := range out {
		out[i] = x[i] * 1.02
	}
	return out
}

func runDenseMLPOracle(x []float32, hiddenSize int) []float32 {
	out := make([]float32, hiddenSize)
	for i := range out {
		out[i] = x[i] + silu(x[i])*0.05
	}
	return out
}

func runMoEOracle(x []float32, hiddenSize int) []float32 {
	out := make([]float32, hiddenSize)
	for i := range out {
		out[i] = x[i] + silu(x[i])*0.08
	}
	return out
}

func runMHCOracle(x []float32, hiddenSize int) []float32 {
	out := make([]float32, hiddenSize)
	for i := range out {
		out[i] = x[i] * 0.999
	}
	return out
}
