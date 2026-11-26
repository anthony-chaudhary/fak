//go:build !darwin || !arm64 || !cgo

package inklingmetal

func Available() bool { return false }

type Model struct{}
type Session struct{}

func openModel(path, draft string) (*Model, error)             { return nil, ErrUnavailable }
func (*Model) NewSession(int) (*Session, error)                { return nil, ErrUnavailable }
func (*Model) Close() error                                    { return nil }
func (*Model) PipelineStats() PipelineStats                    { return PipelineStats{} }
func (*Model) FlushPipelines() bool                            { return false }
func (*Session) Close() error                                  { return nil }
func (*Session) Prefill([]int32) ([]float32, error)            { return nil, ErrUnavailable }
func (*Session) BeginBlock([]int32) ([][]float32, error)       { return nil, ErrUnavailable }
func (*Session) BeginBlockIDs([]int32) ([]int32, error)        { return nil, ErrUnavailable }
func (*Session) Commit(int) error                              { return ErrUnavailable }
func (*Session) Draft(int32, int) ([]int32, error)             { return nil, ErrUnavailable }
func (*Session) DraftTree(int32, int, int) ([][3]int32, error) { return nil, ErrUnavailable }
func (*Session) BeginTreeIDs([][3]int32) ([]int32, error)      { return nil, ErrUnavailable }
func (*Session) CommitTree(int) error                          { return ErrUnavailable }
func (*Session) ProposeVerify(int32, int, int) ([][3]int32, []int32, error) {
	return nil, nil, ErrUnavailable
}

func (*Session) ProposeVerifyCommit(int32, int, int, int, int32) ([][3]int32, []int32, uint32, error) {
	return nil, nil, 0, ErrUnavailable
}

func (*Session) PrefillID([]int32) (int32, error)            { return 0, ErrUnavailable }
func estimateMemory(string, string, int) (MemoryPlan, error) { return MemoryPlan{}, ErrUnavailable }

func (*Model) StepBatch([]*Session, []int32) ([]int32, error) {
	return nil, ErrUnavailable
}

func (*Model) StepBatchWide([]*Session, []int32) ([]int32, error) {
	return nil, ErrUnavailable
}
