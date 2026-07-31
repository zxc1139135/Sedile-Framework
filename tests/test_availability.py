from harmobridge.availability import replicated_hc_success, unreplicated_success


def test_replication_improves_availability():
    assert replicated_hc_success(0.2) > unreplicated_success(0.2)
