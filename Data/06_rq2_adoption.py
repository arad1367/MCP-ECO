import numpy as np
import pandas as pd

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

KEY_TERMS = ["log_tool_count", "share_state_changing", "any_sensitive_capability", "share_model_directed"]


def prepare(df):
    d = df.copy()
    for c in PROVENANCE + ["sig_license"]:
        d[c + "_obs"] = d[c].notna().astype(float)
        d[c] = pd.to_numeric(d[c], errors="coerce").fillna(0.0)
    d["any_sensitive_capability"] = pd.to_numeric(d["any_sensitive_capability"], errors="coerce").fillna(0.0)
    d["share_state_changing"] = pd.to_numeric(d["share_state_changing"], errors="coerce").fillna(0.0)
    d["share_model_directed"] = pd.to_numeric(d["share_model_directed"], errors="coerce").fillna(0.0)
    d["log_tool_count"] = pd.to_numeric(d["log_tool_count"], errors="coerce")
    d["downloads_total"] = pd.to_numeric(d["downloads_total"], errors="coerce")
    d["log_stars"] = pd.to_numeric(d["log_stars"], errors="coerce").fillna(0.0)
    d["has_stars"] = d["stars"].notna().astype(float)
    d["is_enterprise_brand"] = pd.to_numeric(d["is_enterprise_brand"], errors="coerce").fillna(0.0)
    d["provenance_index_filled"] = pd.to_numeric(d["provenance_index"], errors="coerce").fillna(
        pd.to_numeric(d["provenance_index"], errors="coerce").median()
    )
    d = d[d["downloads_total"].notna() & d["log_tool_count"].notna()]
    return d


def run_models(d, tag, fe=FE):
    offset = d["log_exposure_days"].values
    specs = {
        f"{tag}_M1_controls": CONTROLS,
        f"{tag}_M2_scope": CONTROLS + ["log_tool_count"],
        f"{tag}_M3_permissiveness": CONTROLS + PERMISSIVENESS,
        f"{tag}_M4_provenance": CONTROLS + PERMISSIVENESS + PROVENANCE,
        f"{tag}_M5_full": CONTROLS + PERMISSIVENESS + PROVENANCE + ["share_model_directed"],
    }
    out = []
    fits = {}
    for name, regs in specs.items():
        res, X, _ = utils.fit_ppml(d, "downloads_total", regs, "publisher_id", fe, offset=offset)
        out.append(
            utils.tidy(
                res,
                name,
                n_clusters=d["publisher_id"].nunique(),
                extra={"estimator": "PPML", "pseudo_r2": 1 - res.deviance / res.null_deviance},
            )
        )
        fits[name] = (res, X)
    return pd.concat(out, ignore_index=True), fits


def main():
    utils.log("=== 06_rq2 ===")
    rq2 = prepare(utils.read_derived("rq2_sample.csv"))
    utils.log(f"n={len(rq2)} publishers={rq2['publisher_id'].nunique()}")

    desc = utils.describe_numeric(
        rq2,
        [
            "downloads_total",
            "downloads_per_week",
            "exposure_days",
            "tool_count",
            "share_state_changing",
            "share_model_directed",
            "share_any_annotation",
            "any_sensitive_capability",
            "age_days",
            "n_versions_any",
            "stars",
            "publisher_portfolio_size",
        ],
    )
    utils.save_table(desc, "T16_rq2_descriptives")

    corr = rq2[["log1p_downloads_per_week", "log_tool_count", "share_state_changing", "any_sensitive_capability", "share_model_directed", "provenance_index_filled", "log_age_days", "log_n_versions"]].corr()
    utils.save_table(corr.reset_index().rename(columns={"index": "variable"}), "T17_rq2_correlations")

    main_tab, fits = run_models(rq2, "main")
    utils.save_table(main_tab, "T18_rq2_ppml")
    utils.log(main_tab[main_tab["term"].isin(KEY_TERMS)][["model", "term", "coef", "se", "p", "irr"]].round(4).to_string(index=False))

    nb_rows = []
    offset = rq2["log_exposure_days"].values
    res_nb, X_nb, _ = utils.fit_negbin(
        rq2, "downloads_total", CONTROLS + PERMISSIVENESS + PROVENANCE + ["share_model_directed"], "publisher_id", FE, offset=offset
    )
    nb_rows.append(
        utils.tidy(res_nb, "nb2_full", n_clusters=rq2["publisher_id"].nunique(), extra={"estimator": "NB2", "alpha": res_nb.alpha_estimate})
    )
    utils.save_table(pd.concat(nb_rows, ignore_index=True), "T19_rq2_negbin")

    res_ols, X_ols, _ = utils.fit_ols(
        rq2, "log1p_downloads_per_week", CONTROLS + PERMISSIVENESS + PROVENANCE + ["share_model_directed"], "publisher_id", FE
    )
    utils.save_table(utils.tidy(res_ols, "ols_log1p_per_week", n_clusters=rq2["publisher_id"].nunique(), extra={"estimator": "OLS"}), "T20_rq2_ols")

    wild = utils.wild_cluster_bootstrap(
        rq2,
        "log1p_downloads_per_week",
        CONTROLS + PERMISSIVENESS + PROVENANCE + ["share_model_directed"],
        "publisher_id",
        KEY_TERMS,
        FE,
    )
    utils.save_table(wild, "T21_rq2_wild_cluster_bootstrap")
    utils.log(wild.round(4).to_string(index=False))

    utils.save_table(utils.vif_table(X_ols), "T22_rq2_vif")

    svc = rq2[rq2["connected_service"].notna()].copy()
    counts = svc["connected_service"].value_counts()
    svc = svc[svc["connected_service"].isin(counts[counts >= 3].index)]
    if len(svc) >= 40:
        res_svc, _, _ = utils.fit_ppml(
            svc,
            "downloads_total",
            CONTROLS + PERMISSIVENESS,
            "publisher_id",
            [("connected_service", "svc")],
            offset=svc["log_exposure_days"].values,
        )
        utils.save_table(
            utils.tidy(res_svc, "ppml_connected_service_fe", n_clusters=svc["publisher_id"].nunique(), extra={"estimator": "PPML", "n_services": svc["connected_service"].nunique()}),
            "T23_rq2_same_service",
        )
        utils.log(f"same-service subsample n={len(svc)} services={svc['connected_service'].nunique()}")
    else:
        utils.json_dump({"n_available": int(len(svc)), "note": "same-service subsample too small for connected-service fixed effects"}, "T23_rq2_same_service_note")

    inter = rq2.copy()
    inter["tool_x_enterprise"] = inter["log_tool_count"] * inter["is_enterprise_brand"]
    inter["state_x_enterprise"] = inter["share_state_changing"] * inter["is_enterprise_brand"]
    inter["tool_x_provenance"] = inter["log_tool_count"] * inter["provenance_index_filled"]
    res_i, _, _ = utils.fit_ppml(
        inter,
        "downloads_total",
        CONTROLS + PERMISSIVENESS + PROVENANCE + ["is_enterprise_brand", "provenance_index_filled", "tool_x_enterprise", "state_x_enterprise", "tool_x_provenance"],
        "publisher_id",
        FE,
        offset=inter["log_exposure_days"].values,
    )
    utils.save_table(utils.tidy(res_i, "ppml_interactions", n_clusters=inter["publisher_id"].nunique(), extra={"estimator": "PPML"}), "T24_rq2_interactions")

    tiers = []
    for tier, g in rq2.groupby("provenance_tier"):
        if len(g) < 40:
            continue
        r, _, _ = utils.fit_ppml(g, "downloads_total", CONTROLS + PERMISSIVENESS, "publisher_id", FE, offset=g["log_exposure_days"].values)
        t = utils.tidy(r, f"ppml_tier_{tier}", n_clusters=g["publisher_id"].nunique(), extra={"tier": tier, "n": len(g)})
        tiers.append(t)
    if tiers:
        utils.save_table(pd.concat(tiers, ignore_index=True), "T25_rq2_by_provenance_tier")

    quart = rq2.copy()
    quart["tool_count_quartile"] = pd.qcut(quart["tool_count"], 4, labels=["Q1", "Q2", "Q3", "Q4"], duplicates="drop")
    qt = quart.groupby("tool_count_quartile", observed=True).agg(
        n=("server_id", "size"),
        median_downloads_per_week=("downloads_per_week", "median"),
        mean_log_downloads=("log1p_downloads_per_week", "mean"),
        median_tool_count=("tool_count", "median"),
        share_sensitive=("any_sensitive_capability", "mean"),
    )
    utils.save_table(qt.reset_index(), "T26_rq2_adoption_by_tool_quartile")
    utils.log(qt.round(3).to_string())


if __name__ == "__main__":
    main()
