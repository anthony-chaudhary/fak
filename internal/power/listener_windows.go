//go:build windows

package power

func newPlatformSleepListener(b *PowerBroadcaster) SleepListener {
	return newNoOpSleepListener()
}
