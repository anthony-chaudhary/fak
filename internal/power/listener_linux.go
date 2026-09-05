//go:build linux

package power

func newPlatformSleepListener(b *PowerBroadcaster) SleepListener {
	return newNoOpSleepListener()
}
