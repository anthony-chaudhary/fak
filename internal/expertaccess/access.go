// Package expertaccess defines the shared expert-touch descriptor used by
// model-side learners and residency policies. It imports no policy or model code.
package expertaccess

// Access identifies one observed expert touch. WeightBytes is zero when the
// producer records activation identity without sizing the backing weights.
type Access struct {
	Layer       int
	Expert      int
	WeightBytes int64
}
