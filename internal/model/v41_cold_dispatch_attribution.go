package model

// v41ColdDispatchState observes only the cold forward's scalar callbacks.
// Cohorts and incremental steps already own dispatch envelopes. A local copy
// leaves the session callbacks untouched, so repeated layers and later steps
// cannot accumulate wrappers. These are host-observed callback envelopes,
// including admission, down/readback and failure cleanup, not GPU-active time;
// the existing coverage labels describe overlap with activation timing.
func (m *Model) v41ColdDispatchState(st *v41ForwardState) *v41ForwardState {
	if st == nil || st.routedExpert == nil && st.expertGateUp == nil && st.expertDown == nil {
		return st
	}
	observed := *st
	if f := st.routedExpert; f != nil {
		observed.routedExpert = func(layer int, stem string, x []float32, weight float32) (values []float32, outcome v41RoutedExpertOutcome, cause error) {
			opened, returned := m.v41NowNanos(), false
			defer func() {
				if !returned || outcome == v41RoutedExpertHandled || outcome == v41RoutedExpertError {
					m.v41NoteIncrementalDeviceDispatch(true, opened)
					m.v41NoteIncrementalDeviceDispatch(false, 0)
				}
			}()
			values, outcome, cause = f(layer, stem, x, weight)
			returned = true
			return
		}
	}
	if f := st.expertGateUp; f != nil {
		observed.expertGateUp = func(layer int, stem string, x []float32) (values []float32, outcome v41ExpertGateUpOutcome, cause error) {
			opened, returned := m.v41NowNanos(), false
			defer func() {
				if !returned || outcome == v41GateUpHandled || outcome == v41GateUpError {
					m.v41NoteIncrementalDeviceDispatch(true, opened)
				}
			}()
			values, outcome, cause = f(layer, stem, x)
			returned = true
			return
		}
	}
	if f := st.expertDown; f != nil {
		observed.expertDown = func(layer int, stem string, x []float32) (values []float32, outcome v41ExpertDownOutcome, cause error) {
			opened, returned := m.v41NowNanos(), false
			defer func() {
				if !returned || outcome == v41DownHandled || outcome == v41DownError {
					m.v41NoteIncrementalDeviceDispatch(false, opened)
				}
			}()
			values, outcome, cause = f(layer, stem, x)
			returned = true
			return
		}
	}
	return &observed
}
