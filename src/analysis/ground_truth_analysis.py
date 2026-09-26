import pandas as pd


def analyze_ground_truth(gt_df):

    gt_df = gt_df.copy()

    gt_df["matched_entity_ids"] = (
        gt_df["matched_entity_ids"]
        .fillna("")
        .astype(str)
    )

    gt_df["match_count"] = gt_df[
        "matched_entity_ids"
    ].apply(
        lambda x: 0
        if x.strip() == ""
        else len(x.split(","))
    )

    summary = {
        "total_entities": len(gt_df),
        "singletons": (gt_df["match_count"] == 0).sum(),
        "one_match": (gt_df["match_count"] == 1).sum(),
        "two_matches": (gt_df["match_count"] == 2).sum(),
        "three_plus_matches": (gt_df["match_count"] >= 3).sum(),
        "average_matches": gt_df["match_count"].mean(),
        "maximum_matches": gt_df["match_count"].max()
    }

    return summary, gt_df