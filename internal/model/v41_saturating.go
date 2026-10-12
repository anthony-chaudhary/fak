package model

// v41SatMul returns the saturating product of its operands, clamping to bound
// on any overflow so an unrepresentable geometry can only refuse MORE.
func v41SatMul(bound int64, vals ...int64) int64 {
	acc := int64(1)
	for _, v := range vals {
		if v == 0 {
			return 0
		}
		if acc > 0 && v > 0 && acc > bound/v {
			return bound
		}
		acc *= v
	}
	return acc
}

// v41SatAdd returns the saturating sum of its operands, clamping to bound on any
// overflow. A negative operand is a geometry bug and clamps to bound as well.
func v41SatAdd(bound int64, vals ...int64) int64 {
	acc := int64(0)
	for _, v := range vals {
		if v < 0 || acc > bound-v {
			return bound
		}
		acc += v
	}
	return acc
}
