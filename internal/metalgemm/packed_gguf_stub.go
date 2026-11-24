//go:build !darwin || !arm64 || !cgo

package metalgemm

type PackedGGUFStore struct{}
type PackedGGUFMatrix struct{}

func NewPackedGGUFStore(mapped []byte) (*PackedGGUFStore, error) {
	return nil, ErrMetalUnavailable
}
func (*PackedGGUFStore) Matrix(offset int, ggmlType uint32, rows, cols, experts int) (*PackedGGUFMatrix, error) {
	return nil, ErrMetalUnavailable
}
func (*PackedGGUFStore) Run(jobs []PackedGGUFJob) error { return ErrMetalUnavailable }
func (*PackedGGUFStore) Close()                         {}
func (*PackedGGUFStore) ExpertMLP(job PackedExpertMLP, x, dst []float32) error {
	return ErrMetalUnavailable
}

func (*PackedGGUFStore) RoutedFFN(job PackedRoutedFFN, x, dst []float32) (bool, error) {
	return false, ErrMetalUnavailable
}
