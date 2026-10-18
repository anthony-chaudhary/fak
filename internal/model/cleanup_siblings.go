package model

// runBackendSiblingCleanup registers every independent release before invoking
// a callback. Invocation order remains left-to-right. A panic does not skip a
// sibling and the first panic retains precedence, including legacy panic(nil).
// Goexit never returns from its callback boundary: the missing completion count
// keeps a later sibling panic from replacing that original goroutine exit.
func runBackendSiblingCleanup(actions ...func()) {
	var first backendCleanupPanic
	finished := 0
	defer func() {
		if finished == len(actions) && first.present {
			panic(first.value)
		}
	}()
	for i := len(actions) - 1; i >= 0; i-- {
		action := actions[i]
		defer func() {
			func() {
				complete := false
				defer func() {
					if !complete {
						value := recover()
						if !first.present {
							first.value, first.present = value, true
						}
					}
				}()
				action()
				complete = true
			}()
			finished++
		}()
	}
}

// Secondary cleanup panics are recovered without recovering an outer unwind.
// A cleanup callback that itself calls Goexit still exits the goroutine. The
// callback must independently register its siblings.
func suppressBackendCleanupPanic(action func()) {
	defer func() { _ = recover() }()
	action()
}
