import pandas as pd

from portal_analysis.data.data_loader import TimeSeriesDataLoader


def test_load_labels_filters_invalid_ordinal_scores(tmp_path):
    docs = tmp_path / "hand_open_close" / "docs"
    docs.mkdir(parents=True)
    pd.DataFrame(
        {
            "ID": ["a", "b", "c", "d", "e", "a"],
            "snorkel_label": [0, 4, -1, 5, None, 2],
        }
    ).to_csv(docs / "weak_supervision_final.csv", index=False)

    loader = TimeSeriesDataLoader(task_name="hand_open_close", base_dir=tmp_path)
    labels = loader.load_labels(label_column="snorkel_label", label_merge_rules={4: 3})

    assert labels.to_dict() == {"a": 0, "b": 3}
