import numpy as np
import pandas as pd

import config
import utils

FLAG_COLS = [
    "flag_state_changing",
    "flag_filesystem",
    "flag_shell_exec",
    "flag_outbound_network",
    "flag_credential_handling",
    "flag_model_directed_language",
]


def main():
    utils.log("=== 04_features ===")
    tools = utils.read_derived("tool_base.csv")
    final = utils.read_derived("tool_flags_final.csv")
    server = utils.read_derived("server_base.csv")

    t = tools.merge(final.drop(columns=["server_id"]), on="tool_uid", how="left")
    for c in FLAG_COLS:
        t[c + "_final"] = pd.to_numeric(t[c + "_final"], errors="coerce").fillna(0).astype(float)
    for c in [
        "annotations_observable",
        "ann_has_readOnlyHint",
        "ann_value_readOnlyHint",
        "ann_has_destructiveHint",
        "ann_value_destructiveHint",
        "ann_has_idempotentHint",
        "ann_value_idempotentHint",
        "ann_has_openWorldHint",
        "ann_value_openWorldHint",
        "ann_complete_all4",
        "ann_has_any",
    ]:
        t[c] = utils.to_bool(t[c])

    t["ann_n_hints_present"] = pd.to_numeric(t["ann_n_hints_present"], errors="coerce").fillna(0)
    t["description_length_chars"] = pd.to_numeric(t["description_length_chars"], errors="coerce").fillna(0)
    t["schema_n_properties"] = pd.to_numeric(t["schema_n_properties"], errors="coerce").fillna(0)

    t["measured_state_changing"] = t["flag_state_changing_final"] > 0.5
    t["measured_sensitive"] = (
        (t["flag_shell_exec_final"] > 0.5) | (t["flag_filesystem_final"] > 0.5) | (t["flag_credential_handling_final"] > 0.5)
    )
    t["declared_read_only"] = t["ann_value_readOnlyHint"] == True
    t["has_read_only_hint"] = t["ann_has_readOnlyHint"] == True
    t["has_destructive_hint"] = t["ann_has_destructiveHint"] == True
    t["declared_destructive"] = t["ann_value_destructiveHint"] == True
    t["no_annotation"] = (~utils.to_bool_strict(t["ann_has_any"])).astype(float)
    t["annotation_complete"] = utils.to_bool_strict(t["ann_complete_all4"]).astype(float)
    t["model_directed"] = (t["flag_model_directed_language_final"] > 0.5).astype(float)

    t["misdeclared_read_only"] = np.where(
        t["has_read_only_hint"], (t["declared_read_only"] & t["measured_state_changing"]).astype(float), np.nan
    )
    t["misdeclared_not_destructive"] = np.where(
        t["has_destructive_hint"] & t["measured_state_changing"], (~t["declared_destructive"]).astype(float), np.nan
    )
    t["undeclared_state_change"] = np.where(
        t["measured_state_changing"], (~utils.to_bool_strict(t["ann_has_any"])).astype(float), np.nan
    )

    agg = t.groupby("server_id").agg(
        tool_count=("tool_uid", "size"),
        n_state_changing=("flag_state_changing_final", "sum"),
        n_filesystem=("flag_filesystem_final", "sum"),
        n_shell=("flag_shell_exec_final", "sum"),
        n_network=("flag_outbound_network_final", "sum"),
        n_credential=("flag_credential_handling_final", "sum"),
        n_model_directed=("model_directed", "sum"),
        n_any_annotation=("ann_has_any", "sum"),
        n_complete_annotation=("ann_complete_all4", "sum"),
        n_annotations_observable=("annotations_observable", "sum"),
        mean_tool_description_length=("description_length_chars", "mean"),
        mean_schema_properties=("schema_n_properties", "mean"),
        inventory_source=("source", lambda x: x.mode().iat[0] if len(x.mode()) else "unknown"),
    )
    agg["share_state_changing"] = agg["n_state_changing"] / agg["tool_count"]
    agg["share_filesystem"] = agg["n_filesystem"] / agg["tool_count"]
    agg["share_shell"] = agg["n_shell"] / agg["tool_count"]
    agg["share_network"] = agg["n_network"] / agg["tool_count"]
    agg["share_credential"] = agg["n_credential"] / agg["tool_count"]
    agg["share_model_directed"] = agg["n_model_directed"] / agg["tool_count"]
    agg["share_any_annotation"] = agg["n_any_annotation"] / agg["tool_count"]
    agg["share_complete_annotation"] = agg["n_complete_annotation"] / agg["tool_count"]
    agg["any_state_changing"] = (agg["n_state_changing"] > 0).astype(float)
    agg["any_filesystem"] = (agg["n_filesystem"] > 0).astype(float)
    agg["any_shell"] = (agg["n_shell"] > 0).astype(float)
    agg["any_credential"] = (agg["n_credential"] > 0).astype(float)
    agg["any_network"] = (agg["n_network"] > 0).astype(float)
    agg["any_model_directed"] = (agg["n_model_directed"] > 0).astype(float)
    agg["any_sensitive_capability"] = ((agg["n_shell"] + agg["n_filesystem"] + agg["n_credential"]) > 0).astype(float)
    agg["n_sensitive_types"] = (agg["any_shell"] + agg["any_filesystem"] + agg["any_credential"])
    agg = agg.reset_index()

    rule = t.copy()
    for c in FLAG_COLS:
        rule[c] = utils.to_bool_strict(rule[c]).astype(float)
    agg_rule = rule.groupby("server_id").agg(
        n_state_changing_rule=("flag_state_changing", "sum"),
        n_shell_rule=("flag_shell_exec", "sum"),
        n_filesystem_rule=("flag_filesystem", "sum"),
        n_credential_rule=("flag_credential_handling", "sum"),
        n_model_directed_rule=("flag_model_directed_language", "sum"),
        tool_count_rule=("tool_uid", "size"),
    )
    agg_rule["share_state_changing_rule"] = agg_rule["n_state_changing_rule"] / agg_rule["tool_count_rule"]
    agg_rule["share_model_directed_rule"] = agg_rule["n_model_directed_rule"] / agg_rule["tool_count_rule"]
    agg_rule["any_sensitive_capability_rule"] = (
        (agg_rule["n_shell_rule"] + agg_rule["n_filesystem_rule"] + agg_rule["n_credential_rule"]) > 0
    ).astype(float)
    agg = agg.merge(agg_rule.reset_index()[["server_id", "share_state_changing_rule", "share_model_directed_rule", "any_sensitive_capability_rule"]], on="server_id", how="left")

    server = server.drop(columns=[c for c in agg.columns if c != "server_id" and c in server.columns])
    df = server.merge(agg, on="server_id", how="left")
    df["has_tool_inventory"] = df["tool_count"].notna()
    df["log_tool_count"] = np.log1p(df["tool_count"])
    df["log_mean_tool_description"] = np.log1p(df["mean_tool_description_length"])
    df["sig_any_state_changing"] = df["any_state_changing"]
    df["sig_any_shell"] = df["any_shell"]
    df["sig_any_filesystem"] = df["any_filesystem"]
    df["sig_any_credential"] = df["any_credential"]
    df["sig_model_directed"] = df["any_model_directed"]
    df["sig_any_annotation"] = (df["share_any_annotation"] > 0).astype(float)
    df["sig_complete_annotation"] = (df["share_complete_annotation"] > 0).astype(float)
    inv = df[df["has_tool_inventory"]]
    med = inv["tool_count"].median()
    df["sig_many_tools"] = np.where(df["has_tool_inventory"], (df["tool_count"] > med).astype(float), np.nan)
    df["tool_count_median"] = med

    utils.save_derived(df, "server_level.csv")

    census = df[utils.to_bool_strict(df["in_census"])].copy()
    utils.save_derived(census, "census.csv")

    rq1 = census[census["has_tool_inventory"]].copy()
    utils.save_derived(rq1, "rq1_inventory.csv")

    rq2 = df[
        utils.to_bool_strict(df["in_rq2_sample"])
        & df["downloads_total"].notna()
        & df["has_tool_inventory"]
        & (df["exposure_days"] >= config.MIN_EXPOSURE_DAYS)
    ].copy()
    rq2["publisher_id"] = rq2["publisher_id"].astype(str)
    utils.save_derived(rq2, "rq2_sample.csv")

    slim_cols = [
        "server_id",
        "publisher_id",
        "namespace_verification_method",
        "provenance_tier",
        "provenance_index",
        "sig_domain_verified",
        "sig_org_owner",
        "sig_has_ci",
        "sig_security_policy",
        "sig_provenance_attestation",
        "sig_active_release",
        "sig_license",
        "mass_publisher",
        "brand_category",
        "category_rule",
        "in_census",
        "in_rq2_sample",
        "primary_ecosystem",
        "tool_count",
        "log1p_downloads_per_week",
        "publisher_is_vendor",
        "connected_service",
        "log_age_days",
        "log_publisher_portfolio",
    ]
    tool_level = t.merge(df[[c for c in slim_cols if c in df.columns]], on="server_id", how="left")
    utils.save_derived(tool_level, "tool_level.csv")

    selection = []
    adopt = df[utils.to_bool_strict(df["in_adoption_sample"])]
    for var in ["log1p_downloads_per_week", "log_stars", "log_age_days", "log_n_versions", "sig_org_owner", "sig_has_ci"]:
        a = pd.to_numeric(adopt.loc[adopt["has_tool_inventory"], var], errors="coerce").dropna()
        b = pd.to_numeric(adopt.loc[~adopt["has_tool_inventory"], var], errors="coerce").dropna()
        if len(a) > 5 and len(b) > 5:
            pooled_sd = np.sqrt((a.var(ddof=1) * (len(a) - 1) + b.var(ddof=1) * (len(b) - 1)) / (len(a) + len(b) - 2))
            selection.append(
                {
                    "variable": var,
                    "n_with_inventory": len(a),
                    "n_without_inventory": len(b),
                    "mean_with": a.mean(),
                    "mean_without": b.mean(),
                    "std_diff": (a.mean() - b.mean()) / pooled_sd if pooled_sd > 0 else np.nan,
                }
            )
    sel = pd.DataFrame(selection)
    utils.save_table(sel, "T06_rq2_selection_balance")

    eco = adopt.assign(with_inv=adopt["has_tool_inventory"]).groupby(["primary_ecosystem", "with_inv"]).size().unstack(fill_value=0)
    utils.save_table(eco.reset_index(), "T06b_rq2_selection_ecosystem")

    flow = pd.DataFrame(
        [
            {"stage": "census", "n": len(census)},
            {"stage": "census_with_inventory", "n": len(rq1)},
            {"stage": "rq2_analysis_sample", "n": len(rq2)},
            {"stage": "rq2_publishers", "n": int(rq2["publisher_id"].nunique())},
            {"stage": "tool_level_rows", "n": len(tool_level)},
        ]
    )
    utils.save_table(flow, "T00b_analysis_samples")
    utils.log(flow.to_string(index=False))
    utils.log(sel.to_string(index=False))


if __name__ == "__main__":
    main()
