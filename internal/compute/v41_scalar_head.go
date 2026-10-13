package compute

// The small shared head composes ordinary scalar projections. These are the
// pinned Flash dimensions, not general matrix-multiplication admission.
func v41ScalarHeadShape(dtype Dtype, out, in, rows int) bool {
	return dtype == Q6_K && out == 129280 && in == 5120 && rows >= 1 && rows <= 4
}

// runV41ScalarHeadWindow closes an acquired recording window exactly once,
// including after a row panic. The first row error/panic wins over end failure.
// A failed begin owns no window and must never invoke end or replay a row.
func runV41ScalarHeadWindow(rows int, begin func() error, row func(int) error, end func() error) (err error) {
	if err = begin(); err != nil {
		return err
	}
	defer func() {
		primary := recover()
		var endErr error
		var endPanic any
		func() {
			defer func() { endPanic = recover() }()
			endErr = end()
		}()
		if primary != nil {
			panic(primary)
		}
		if err != nil {
			return
		}
		if endPanic != nil {
			panic(endPanic)
		}
		err = endErr
	}()
	for i := 0; i < rows; i++ {
		if err = row(i); err != nil {
			return err
		}
	}
	return nil
}
