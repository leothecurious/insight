from insight.checkpoints import filename


def test_filename_identifies_series_and_epoch_without_repeating_length_sweep_epoch():
    assert filename("lewm_tworoom", 10, 100) == "lewm_tworoom_epoch_10.pt"
    assert filename("lewm_tworoom_ep10", 5, 10) == "lewm_tworoom_ep10_epoch_5.pt"
    assert filename("lewm_tworoom_ep10", 10, 10) == "lewm_tworoom_ep10.pt"
