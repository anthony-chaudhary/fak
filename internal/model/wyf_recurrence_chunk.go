package model

type wyfChunkPhase struct {
	state                        [][]float32
	outputs                      [][]float32
	states                       [][][]float32
	offset, chunkLen, dim        int
	keys, values                 [][]float32
	projected, adjusted, updates [][]float32
	gram, triangular, inverse    [][]float32
	gamma                        [][]float32
	alphaScalar, betaScalar      []float32
	alphaVector, betaVector      [][]float32
}

func (c wyfChunkPhase) runUniform() [][]float32 {
	for j := 0; j <= c.chunkLen; j++ {
		c.gamma[j][j] = 1
		for t := j + 1; t <= c.chunkLen; t++ {
			c.gamma[j][t] = c.gamma[j][t-1] * c.alphaScalar[t-1]
		}
	}

	for t := 0; t < c.chunkLen; t++ {
		row := c.adjusted[t]
		for d := 0; d < c.dim; d++ {
			row[d] = c.betaScalar[t] * (c.values[t][d] - c.gamma[0][t]*c.projected[t][d])
		}
	}

	for j := 0; j < c.chunkLen; j++ {
		c.triangular[j][j] = 1
		for t := j + 1; t < c.chunkLen; t++ {
			c.triangular[j][t] = c.betaScalar[t] * c.gamma[j+1][t] * c.gram[j][t]
		}
	}

	for j := 0; j < c.chunkLen; j++ {
		c.inverse[j][j] = 1
		for t := j + 1; t < c.chunkLen; t++ {
			var sum float32
			for m := j; m < t; m++ {
				sum += c.inverse[j][m] * c.triangular[m][t]
			}
			c.inverse[j][t] = -sum
		}
	}

	for t := 0; t < c.chunkLen; t++ {
		row := c.updates[t]
		for d := 0; d < c.dim; d++ {
			row[d] = 0
		}
		for j := 0; j <= t; j++ {
			for d := 0; d < c.dim; d++ {
				row[d] += c.adjusted[j][d] * c.inverse[j][t]
			}
		}
	}

	for t := 0; t < c.chunkLen; t++ {
		output := make([]float32, c.dim)
		for d := 0; d < c.dim; d++ {
			output[d] = c.gamma[0][t+1] * c.projected[t][d]
		}
		for j := 0; j <= t; j++ {
			coefficient := c.gamma[j+1][t+1] * c.gram[j][t]
			for d := 0; d < c.dim; d++ {
				output[d] += coefficient * c.updates[j][d]
			}
		}
		c.outputs[c.offset+t] = output
	}

	current := c.state
	for t := 0; t < c.chunkLen; t++ {
		next := make([][]float32, c.dim)
		backing := make([]float32, c.dim*c.dim)
		for i := 0; i < c.dim; i++ {
			row := backing[i*c.dim : (i+1)*c.dim]
			for d := 0; d < c.dim; d++ {
				row[d] = c.alphaScalar[t]*current[i][d] + c.updates[t][i]*c.keys[t][d]
			}
			next[i] = row
		}
		c.states[c.offset+t] = next
		current = next
	}
	return current
}

func (c wyfChunkPhase) runPerDimension() [][]float32 {
	state, outputs, states := c.state, c.outputs, c.states
	offset, chunkLen, dim := c.offset, c.chunkLen, c.dim
	keys, values, projected, gram := c.keys, c.values, c.projected, c.gram
	alphaScalar, betaScalar := c.alphaScalar, c.betaScalar
	alphaVector, betaVector := c.alphaVector, c.betaVector

	updates := make([][]float32, chunkLen)
	for t := range updates {
		updates[t] = make([]float32, dim)
	}

	for d := 0; d < dim; d++ {
		gamma := make([][]float32, chunkLen+1)
		for j := 0; j <= chunkLen; j++ {
			gamma[j] = make([]float32, chunkLen+1)
			gamma[j][j] = 1
			for t := j + 1; t <= chunkLen; t++ {
				alpha := alphaScalar[t-1]
				if alphaVector[t-1] != nil {
					alpha = alphaVector[t-1][d]
				}
				gamma[j][t] = gamma[j][t-1] * alpha
			}
		}

		adjustedValues := make([]float32, chunkLen)
		for t := 0; t < chunkLen; t++ {
			beta := betaScalar[t]
			if betaVector[t] != nil {
				beta = betaVector[t][d]
			}
			adjustedValues[t] = beta * (values[t][d] - gamma[0][t]*projected[t][d])
		}

		triangular := make([][]float32, chunkLen)
		for j := 0; j < chunkLen; j++ {
			triangular[j] = make([]float32, chunkLen)
			triangular[j][j] = 1
			for t := j + 1; t < chunkLen; t++ {
				beta := betaScalar[t]
				if betaVector[t] != nil {
					beta = betaVector[t][d]
				}
				triangular[j][t] = beta * gamma[j+1][t] * gram[j][t]
			}
		}

		inverse := make([][]float32, chunkLen)
		for j := 0; j < chunkLen; j++ {
			inverse[j] = make([]float32, chunkLen)
			inverse[j][j] = 1
			for t := j + 1; t < chunkLen; t++ {
				var sum float32
				for m := j; m < t; m++ {
					sum += inverse[j][m] * triangular[m][t]
				}
				inverse[j][t] = -sum
			}
		}

		for t := 0; t < chunkLen; t++ {
			var sum float32
			for j := 0; j <= t; j++ {
				sum += adjustedValues[j] * inverse[j][t]
			}
			updates[t][d] = sum
		}
	}

	for t := 0; t < chunkLen; t++ {
		output := make([]float32, dim)
		for d := 0; d < dim; d++ {
			decay := float32(1)
			for m := 0; m <= t; m++ {
				alpha := alphaScalar[m]
				if alphaVector[m] != nil {
					alpha = alphaVector[m][d]
				}
				decay *= alpha
			}
			value := decay * projected[t][d]
			for j := 0; j <= t; j++ {
				updateDecay := float32(1)
				for m := j + 1; m <= t; m++ {
					alpha := alphaScalar[m]
					if alphaVector[m] != nil {
						alpha = alphaVector[m][d]
					}
					updateDecay *= alpha
				}
				value += updateDecay * gram[j][t] * updates[j][d]
			}
			output[d] = value
		}
		outputs[offset+t] = output
	}

	current := state
	for t := 0; t < chunkLen; t++ {
		next := make([][]float32, dim)
		backing := make([]float32, dim*dim)
		for i := 0; i < dim; i++ {
			row := backing[i*dim : (i+1)*dim]
			alpha := alphaScalar[t]
			if alphaVector[t] != nil {
				alpha = alphaVector[t][i]
			}
			for d := 0; d < dim; d++ {
				row[d] = alpha*current[i][d] + updates[t][i]*keys[t][d]
			}
			next[i] = row
		}
		states[offset+t] = next
		current = next
	}
	return current
}
