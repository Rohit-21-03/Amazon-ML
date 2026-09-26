def analyze_match_sources(gt_df):

    s2_count = 0
    s3_count = 0

    for ids in gt_df["matched_entity_ids"].fillna(""):

        if not ids:
            continue

        items = ids.split(",")

        for item in items:

            item = item.strip()

            if item.startswith("S2-"):
                s2_count += 1

            elif item.startswith("S3-"):
                s3_count += 1

    return {
        "s2_matches": s2_count,
        "s3_matches": s3_count
    }