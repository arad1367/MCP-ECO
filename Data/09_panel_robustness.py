import numpy as np
import pandas as pd
import statsmodels.api as sm

import config
import utils

CONTROLS = [
    "log_age_days",
    "log_n_versions",
    "log_description_length",
    "log_publisher_portfolio",
    "is_deprecated",
    "mass_publisher",
]

PERMISSIVENESS = ["log_tool_count", "share_state_changing", "any_sensitive_capability"]

PROVENANCE = [
    "sig_org_owner",
    "sig_has_ci",
    "sig_security_policy",
    "sig_provenance_attestation",
    "sig_domain_verified",
    "sig_active_release",
]

FE = [("primary_ecosystem", "eco"), ("category_rule", "cat")]
KEY = ["log_tool_count", "share_state_changing", "any_sensitive_capability", "share_model_directed"]


def prep(df):
    d = df.copy()
    for c in PROVENANCE:
        d[c] = pd.to_numeric(d[c], errors="coerce").fillna(0.0)
    for c in ["share_state_changing", "any_sensitive_capability", "share_model_directed", "share_state_changing_rule", "share_model_directed_rule", "any_sensitive_capability_rule"]:
        if c in d.columns:
            d[c] = pd.to_numeric(d[c], errors="coerce").fillna(0.0)
    d["log_tool_count"] = pd.to_numeric(d["log_tool_count"], errors="coerce")
    d["downloads_total"] = pd.to_numeric(d["downloads_total"], errors="coerce")
    d["log_stars"] = pd.to_numeric(d["log_stars"], errors="coerce").fillna(0.0)
    d["has_stars"] = d["stars"].notna().astype(float)
    return d[d["downloads_total"].notna() & d["log_tool_count"].notna()]


def panel_models(panel):
    rows = []
    if len(panel) < 40:
        return pd.DataFrame(rows)
    p = panel.copy()
    p["package_id"] = p["package_id"].astype(str)
    p["share_flag_state_changing"] = pd.to_numeric(p["share_flag_state_changing"], errors="coerce").fillna(0.0)
    p["log_tool_count"] = pd.to_numeric(p["log_tool_count"], errors="coerce").fillna(0.0)
    d = pd.get_dummies(p["package_id"], prefix="pkg", drop_first=True).astype(float)
    X = pd.concat([p[["log_tool_count", "share_flag_state_changing"]].astype(float).reset_index(drop=True), d.reset_index(drop=True)], axis=1)
    X = sm.add_constant(X, has_constant="add")
    X = utils.prune_design(X)
    y = p["log_downloads_per_week_post"].astype(float).reset_index(drop=True)
    res = sm.OLS(y, X).fit(cov_type="cluster", cov_kwds={"groups": p["package_id"].values})
    t = utils.tidy(res, "within_package_fe", n_clusters=p["package_id"].nunique(), extra={"estimator": "OLS_FE"})
    rows.append(t[~t["term"].str.startswith("pkg_")])

    p = p.sort_values(["package_id", "version_published_dt"])
    p["d_log_downloads"] = p.groupby("package_id")["log_downloads_per_week_post"].diff()
    p["d_log_tool_count"] = p.groupby("package_id")["log_tool_count"].diff()
    p["d_share_state_changing"] = p.groupby("package_id")["share_flag_state_changing"].diff()
    fd = p.dropna(subset=["d_log_downloads", "d_log_tool_count"])
    if len(fd) >= 30:
        Xf = sm.add_constant(fd[["d_log_tool_count", "d_share_state_changing"]].astype(float), has_constant="add")
        resf = sm.OLS(fd["d_log_downloads"].astype(float), Xf).fit(cov_type="cluster", cov_kwds={"groups": fd["package_id"].values})
        rows.append(utils.tidy(resf, "first_difference", n_clusters=fd["package_id"].nunique(), extra={"estimator": "OLS_FD"}))

    p["lag_log_downloads"] = p.groupby("package_id")["log_downloads_per_week_post"].shift(1)
    lead = p.dropna(subset=["d_log_tool_count", "lag_log_downloads"])
    if len(lead) >= 30:
        Xl = sm.add_constant(lead[["lag_log_downloads"]].astype(float), has_constant="add")
        resl = sm.OLS(lead["d_log_tool_count"].astype(float), Xl).fit(cov_type="cluster", cov_kwds={"groups": lead["package_id"].values})
        rows.append(utils.tidy(resl, "reverse_causality_check", n_clusters=lead["package_id"].nunique(), extra={"estimator": "OLS_lead"}))
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def main():
    utils.log("=== 09_panel_robustness ===")
    panel = utils.read_derived("version_panel.csv")
    pm = panel_models(panel)
    if len(pm):
        utils.save_table(pm, "T42_version_panel_models")
        utils.log(pm[["model", "term", "coef", "se", "p", "n_obs", "n_clusters"]].round(4).to_string(index=False))
    desc = panel.groupby("package_id").agg(
        n_versions=("version", "size"),
        tool_count_change=("tool_count", lambda x: float(x.max() - x.min())),
        state_share_change=("share_flag_state_changing", lambda x: float(x.max() - x.min())),
        annotation_share_change=("share_tools_any_annotation", lambda x: float(np.nanmax(x) - np.nanmin(x))),
    )
    summary = {
        "n_packages": int(len(desc)),
        "n_observations": int(len(panel)),
        "packages_changing_tool_count": int((desc["tool_count_change"] > 0).sum()),
        "packages_changing_state_share": int((desc["state_share_change"] > 0).sum()),
        "packages_changing_annotation_share": int((desc["annotation_share_change"] > 0).sum()),
    }
    utils.json_dump(summary, "T43_version_panel_summary")
    utils.log(str(summary))

    rq2 = prep(utils.read_derived("rq2_sample.csv"))
    base = CONTROLS + PERMISSIVENESS + PROVENANCE + ["share_model_directed"]
    rows = []

    def add(name, d, regs, y="downloads_total", fe=FE, offset_col="log_exposure_days"):
        if len(d) < 60:
            return
        off = d[offset_col].values if offset_col else None
        r, _, _ = utils.fit_ppml(d, y, regs, "publisher_id", fe, offset=off)
        t = utils.tidy(r, name, n_clusters=d["publisher_id"].nunique(), extra={"estimator": "PPML", "n": len(d)})
        rows.append(t[t["term"].isin(KEY + ["share_model_directed_rule", "share_state_changing_rule", "any_sensitive_capability_rule", "log_stars"])])

    add("baseline", rq2, base)
    add("exclude_mass_publishers", rq2[rq2["mass_publisher"] == 0], [c for c in base if c != "mass_publisher"])
    add("npm_only", rq2[rq2["primary_ecosystem"] == "npm"], base, fe=[("category_rule", "cat")])
    add("pypi_only", rq2[rq2["primary_ecosystem"] == "pypi"], base, fe=[("category_rule", "cat")])
    add("local_probe_inventory_only", rq2[rq2["inventory_source"] == "local_probe"], base)
    add("with_stars_control", rq2[rq2["stars"].notna()], base + ["log_stars"])
    add("rule_based_measures", rq2, CONTROLS + ["log_tool_count", "share_state_changing_rule", "any_sensitive_capability_rule"] + PROVENANCE + ["share_model_directed_rule"])
    add("trim_top1pct_downloads", rq2[rq2["downloads_total"] <= rq2["downloads_total"].quantile(0.99)], base)
    add("exclude_zero_downloads", rq2[rq2["downloads_total"] > 0], base)

    nospike = rq2[rq2["downloads_total_nospike"].notna()].copy()
    if len(nospike) >= 60:
        r, _, _ = utils.fit_ppml(nospike, "downloads_total_nospike", base, "publisher_id", FE, offset=nospike["log_exposure_days"].values)
        t = utils.tidy(r, "spike_filtered_outcome", n_clusters=nospike["publisher_id"].nunique(), extra={"estimator": "PPML", "n": len(nospike)})
        rows.append(t[t["term"].isin(KEY)])

    last12 = rq2[rq2["downloads_last12w"].notna()].copy()
    if len(last12) >= 60:
        last12["downloads_last12w"] = pd.to_numeric(last12["downloads_last12w"], errors="coerce")
        r, _, _ = utils.fit_ppml(last12, "downloads_last12w", base, "publisher_id", FE, offset=None)
        t = utils.tidy(r, "recent_12_weeks_outcome", n_clusters=last12["publisher_id"].nunique(), extra={"estimator": "PPML", "n": len(last12)})
        rows.append(t[t["term"].isin(KEY)])

    rob = pd.concat(rows, ignore_index=True)
    utils.save_table(rob, "T44_robustness_specifications")
    utils.log(rob[["model", "term", "coef", "se", "p", "n"]].round(4).to_string(index=False))

    loo = []
    for cat in rq2["category_rule"].dropna().unique():
        d = rq2[rq2["category_rule"] != cat]
        if len(d) < 100:
            continue
        r, _, _ = utils.fit_ppml(d, "downloads_total", base, "publisher_id", FE, offset=d["log_exposure_days"].values)
        t = utils.tidy(r, f"drop_{cat}", n_clusters=d["publisher_id"].nunique(), extra={"dropped_category": cat, "n": len(d)})
        loo.append(t[t["term"].isin(KEY)])
    if loo:
        utils.save_table(pd.concat(loo, ignore_index=True), "T45_leave_one_category_out")

    zero = {
        "share_zero_download_servers": float((rq2["downloads_total"] == 0).mean()),
        "n_zero": int((rq2["downloads_total"] == 0).sum()),
        "n": int(len(rq2)),
        "mean_downloads_total": float(rq2["downloads_total"].mean()),
        "variance_downloads_total": float(rq2["downloads_total"].var()),
        "overdispersion_ratio": float(rq2["downloads_total"].var() / max(rq2["downloads_total"].mean(), 1e-9)),
    }
    utils.json_dump(zero, "T46_outcome_distribution")
    utils.log(str(zero))


if __name__ == "__main__":
    main()
