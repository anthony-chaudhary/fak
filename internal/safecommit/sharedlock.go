package safecommit

// AcquireSharedLock acquires the same repository-scoped advisory writer lock used
// by Commit. Additive commit surfaces (for example the exact-patch transaction)
// use this rather than inventing a second lock domain.
func AcquireSharedLock(opts LockOptions) (func(), error) {
	return realLock(opts)
}
