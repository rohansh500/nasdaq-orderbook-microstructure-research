from orderbook_research.splits import purged_chronological_split


def test_purged_split_is_ordered_and_separated():
    split = purged_chronological_split(
        10_000,
        purge_events=100,
    )

    assert split.train_indices.max() < split.validation_indices.min()
    assert split.validation_indices.max() < split.test_indices.min()

    train_gap = split.validation_indices.min() - split.train_indices.max() - 1
    validation_gap = split.test_indices.min() - split.validation_indices.max() - 1

    assert train_gap >= 100
    assert validation_gap >= 100
