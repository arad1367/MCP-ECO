import numpy as np
import pandas as pd

import config
import utils

PROVENANCE = [
    "sig_org_owner",
    "sig_has_ci",
    "sig_security_policy",
    "sig_provenance_attestation",
    "sig_domain_verified",
    "sig_active_release",
]


def cluster_bootstrap_mean(df, col, cluster_col, n_boot=1000, seed=None):
    rng = np.random.default_rng(seed or config.SEED)
    v = pd.to_numeric(df[col], errors="coerce")
    g = df[cluster_col].astype(str)
    ok = v.notna()
    v, g = v[ok], g[ok]
    agg = pd.DataFrame({"s": v.groupby(g).sum(), "n": v.groupby(g).size()})
    s, n = agg["s"].values, agg["n"].values
    est = float(v.mean())
    k = len(s)
    if k < 2:
        return est, np.nan, np.nan, int(len(v))
    idx = rng.integers(0, k, size=(n_boot, k))
    boots = s[idx].sum(axis=1) / n[idx].sum(axis=1)
    return est, float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5)), int(len(v))


def prep(t):
    d = t.copy()
    for c in PROVENANCE:
        d[c] = pd.to_numeric(d[c], errors="coerce").fillna(0.0)
    d["publisher_id"] = d["publisher_id"].fillna("unknown").astype(str)
    d["log_tool_count"] = np.log1p(pd.to_numeric(d["tool_count"], errors="coerce"))
    d["log_age_days"] = pd.to_numeric(d["log_age_days"], errors="coerce").fillna(pd.to_numeric(d["log_age_days"], errors="coerce").median())
    d["log_publisher_portfolio"] = pd.to_numeric(d["log_publisher_portfolio"], errors="coerce").fillna(0.0)
    d["mass_publisher"] = pd.to_numeric(d["mass_publisher"], errors="coerce").fillna(0.0)
    d["description_length_chars"] = pd.to_numeric(d["description_length_chars"], errors="coerce").fillna(0.0)
    d["log_description_length_tool"] = np.log1p(d["description_length_chars"])
    return d


def main():
    utils.log("=== 07_rq3 ===")
    t = prep(utils.read_derived("tool_level.csv"))
    t = t[utils.to_bool_strict(t["in_census"])]

    rows = []
    for col, label in [
        ("ann_has_any", "any_annotation"),
        ("ann_complete_all4", "all_four_annotations"),
        ("ann_has_readOnlyHint", "readOnlyHint_present"),
        ("ann_has_destructiveHint", "destructiveHint_present"),
        ("ann_has_idempotentHint", "idempotentHint_present"),
        ("ann_has_openWorldHint", "openWorldHint_present"),
    ]:
        d = t.copy()
        d["_v"] = utils.to_bool_strict(d[col]).astype(float)
        est, lo, hi, n = cluster_bootstrap_mean(d, "_v", "publisher_id")
        rows.append({"annotation": label, "share": est, "ci_low": lo, "ci_high": hi, "n_tools": n})
    utils.save_table(pd.DataFrame(rows), "T27_annotation_presence")
    utils.log(pd.DataFrame(rows).round(3).to_string(index=False))

    by_tier = []
    for tier, g in t.groupby("provenance_tier"):
        g = g.copy()
        g["_v"] = utils.to_bool_strict(g["ann_has_any"]).astype(float)
        est, lo, hi, n = cluster_bootstrap_mean(g, "_v", "publisher_id", n_boot=500)
        by_tier.append({"provenance_tier": tier, "share_any_annotation": est, "ci_low": lo, "ci_high": hi, "n_tools": n, "n_servers": g["server_id"].nunique()})
    utils.save_table(pd.DataFrame(by_tier), "T28_annotation_by_provenance_tier")

    hint = t[t["has_read_only_hint"] == True].copy()
    tab = pd.crosstab(hint["declared_read_only"], hint["measured_state_changing"])
    utils.save_table(tab.reset_index(), "T29_readonly_vs_measured_crosstab")

    declared_write = (~hint["declared_read_only"]).astype(int).values
    measured_write = hint["measured_state_changing"].astype(int).values
    k, po, pe, n = utils.cohen_kappa(declared_write, measured_write)
    agreement = {
        "n_tools_with_readOnlyHint": int(n),
        "kappa_declared_vs_measured": float(k),
        "percent_agreement": float(po),
        "expected_agreement": float(pe),
        "n_servers": int(hint["server_id"].nunique()),
    }
    utils.json_dump(agreement, "T29b_readonly_agreement")
    utils.log(str(agreement))

    mis_rows = []
    for col, label, sub in [
        ("misdeclared_read_only", "declared_read_only_but_state_changing", t[t["has_read_only_hint"] == True]),
        ("misdeclared_not_destructive", "state_changing_declared_non_destructive", t[(t["has_destructive_hint"] == True) & (t["measured_state_changing"])]),
        ("undeclared_state_change", "state_changing_without_any_annotation", t[t["measured_state_changing"]]),
    ]:
        if len(sub) < 10:
            continue
        est, lo, hi, n = cluster_bootstrap_mean(sub, col, "publisher_id", n_boot=500)
        mis_rows.append({"measure": label, "rate": est, "ci_low": lo, "ci_high": hi, "n_tools": n, "n_servers": sub["server_id"].nunique()})
    utils.save_table(pd.DataFrame(mis_rows), "T30_misdeclaration_rates")
    utils.log(pd.DataFrame(mis_rows).round(3).to_string(index=False))

    hc = t[(t["flag_state_changing_score"].isna()) | (t["flag_state_changing_score"] >= 0.8) | (t["flag_state_changing_score"] <= 0.2)]
    hc_rows = []
    sub = hc[hc["has_read_only_hint"] == True]
    if len(sub) > 10:
        est, lo, hi, n = cluster_bootstrap_mean(sub, "misdeclared_read_only", "publisher_id", n_boot=500)
        hc_rows.append({"measure": "declared_read_only_but_state_changing_high_confidence", "rate": est, "ci_low": lo, "ci_high": hi, "n_tools": n})
    utils.save_table(pd.DataFrame(hc_rows), "T31_misdeclaration_high_confidence")

    model_rows = []
    m1 = t[t["has_read_only_hint"] == True].dropna(subset=["misdeclared_read_only"])
    if len(m1) > 100 and m1["misdeclared_read_only"].sum() > 20:
        res, X, _ = utils.fit_logit(
            m1,
            "misdeclared_read_only",
            PROVENANCE + ["log_tool_count", "log_age_days", "log_publisher_portfolio", "mass_publisher", "log_description_length_tool"],
            "publisher_id",
            [("primary_ecosystem", "eco"), ("category_rule", "cat")],
        )
        model_rows.append(utils.tidy(res, "logit_misdeclared_read_only", n_clusters=m1["publisher_id"].nunique(), extra={"estimator": "logit"}))

    t["no_annotation_any"] = (~utils.to_bool_strict(t["ann_has_any"])).astype(float)
    res2, _, _ = utils.fit_logit(
        t,
        "no_annotation_any",
        PROVENANCE + ["log_tool_count", "log_age_days", "log_publisher_portfolio", "mass_publisher", "log_description_length_tool"],
        "publisher_id",
        [("primary_ecosystem", "eco"), ("category_rule", "cat")],
    )
    model_rows.append(utils.tidy(res2, "logit_no_annotation", n_clusters=t["publisher_id"].nunique(), extra={"estimator": "logit"}))

    sc = t[t["measured_state_changing"]].dropna(subset=["undeclared_state_change"])
    if len(sc) > 100:
        res3, _, _ = utils.fit_logit(
            sc,
            "undeclared_state_change",
            PROVENANCE + ["log_tool_count", "log_age_days", "log_publisher_portfolio", "mass_publisher"],
            "publisher_id",
            [("primary_ecosystem", "eco"), ("category_rule", "cat")],
        )
        model_rows.append(utils.tidy(res3, "logit_undeclared_state_change", n_clusters=sc["publisher_id"].nunique(), extra={"estimator": "logit"}))

    mt = pd.concat(model_rows, ignore_index=True)
    mt["odds_ratio"] = np.exp(mt["coef"])
    utils.save_table(mt, "T32_disclosure_logistic_models")
    utils.log(mt[mt["term"].isin(PROVENANCE)][["model", "term", "coef", "se", "p", "odds_ratio"]].round(3).to_string(index=False))

    srv = utils.read_derived("rq1_inventory.csv")
    srv_rows = []
    for tier, g in srv.groupby("provenance_tier"):
        srv_rows.append(
            {
                "provenance_tier": tier,
                "n_servers": len(g),
                "share_servers_any_annotation": float((pd.to_numeric(g["share_any_annotation"], errors="coerce") > 0).mean()),
                "mean_share_tools_annotated": float(pd.to_numeric(g["share_any_annotation"], errors="coerce").mean()),
                "mean_share_complete": float(pd.to_numeric(g["share_complete_annotation"], errors="coerce").mean()),
                "mean_share_state_changing": float(pd.to_numeric(g["share_state_changing"], errors="coerce").mean()),
            }
        )
    utils.save_table(pd.DataFrame(srv_rows), "T33_server_disclosure_by_tier")

    src_rows = []
    for src, g in t.groupby("source"):
        g = g.copy()
        g["_v"] = utils.to_bool_strict(g["ann_has_any"]).astype(float)
        est, lo, hi, n = cluster_bootstrap_mean(g, "_v", "publisher_id", n_boot=300)
        src_rows.append({"inventory_source": src, "share_any_annotation": est, "ci_low": lo, "ci_high": hi, "n_tools": n})
    utils.save_table(pd.DataFrame(src_rows), "T34_annotation_by_inventory_source")


if __name__ == "__main__":
    main()
