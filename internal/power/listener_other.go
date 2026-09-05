//go:build !darwin && !linux && !windows

package power

func newPlatformSleepListener(b *PowerBroadcaster) SleepListener {
	return newNoOpSleepListener()
}
