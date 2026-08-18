package mathx

// AgainstOracle returns achieved/oracle, with an absent oracle scoring one only
// when nothing was achieved and zero otherwise.
func AgainstOracle(achieved, oracle int) float64 {
	if oracle <= 0 {
		if achieved == 0 {
			return 1
		}
		return 0
	}
	return float64(achieved) / float64(oracle)
}
