import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score

import config
import utils

ITEMS = [
    "sig_domain_verified",
    "sig_org_owner",
    "sig_has_ci",
    "sig_security_policy",
    "sig_provenance_attestation",
    "sig_active_release",
    "sig_any_annotation",
    "sig_complete_annotation",
    "sig_any_state_changing",
    "sig_any_shell",
    "sig_any_filesystem",
    "sig_any_credential",
    "sig_many_tools",
    "sig_model_directed",
]


def main():
    utils.log("=== 05_rq1 ===")
    census = utils.read_derived("census.csv")
    inv = utils.read_derived("rq1_inventory.csv")

    desc_cols = [
        "sig_domain_verified",
        "sig_org_owner",
        "sig_has_ci",
        "sig_security_policy",
        "sig_provenance_attestation",
        "sig_active_release",
        "sig_license",
    ]
    rows = []
    for c in desc_cols:
        for name, d in [("census", census), ("inventory_sample", inv)]:
            s = pd.to_numeric(d[c], errors="coerce")
            est, lo, hi = utils.bootstrap_ci(s.dropna().values)
            rows.append(
                {
                    "signal": c,
                    "sample": name,
                    "n_observed": int(s.notna().sum()),
                    "n_total": len(d),
                    "share": est,
                    "ci_low": lo,
                    "ci_high": hi,
                }
            )
    utils.save_table(pd.DataFrame(rows), "T07_provenance_signal_prevalence")

    cap_cols = [
        "tool_count",
        "share_state_changing",
        "share_network",
        "share_credential",
        "share_filesystem",
        "share_shell",
        "share_any_annotation",
        "share_complete_annotation",
        "share_model_directed",
    ]
    utils.save_table(utils.describe_numeric(inv, cap_cols), "T08_capability_descriptives")

    bin_cols = [c for c in ITEMS if c in inv.columns]
    corr = inv[bin_cols].astype(float).corr()
    utils.save_table(corr.reset_index().rename(columns={"index": "signal"}), "T09_signal_cooccurrence")

    cross = pd.crosstab(inv["provenance_tier"], inv["any_sensitive_capability"], normalize="index")
    utils.save_table(cross.reset_index(), "T10_tier_by_sensitive_capability")

    data = inv[bin_cols].astype(float).values
    fit_rows = []
    models = {}
    for k in range(1, config.LCA_MAX_CLASSES + 1):
        m = utils.BernoulliLCA(k, n_starts=config.LCA_N_STARTS, seed=config.SEED).fit(data)
        models[k] = m
        fit_rows.append(
            {
                "n_classes": k,
                "loglik": m.loglik_,
                "n_parameters": m.n_params_,
                "aic": m.aic_,
                "bic": m.bic_,
                "abic": m.abic_,
                "entropy": m.entropy_,
                "smallest_class_share": float(m.class_sizes_.min() / m.n_),
                "n": m.n_,
            }
        )
    fit = pd.DataFrame(fit_rows)

    blrt_rows = []
    for k in range(2, config.LCA_MAX_CLASSES + 1):
        blrt_rows.append(utils.blrt(data, k, n_boot=config.LCA_BLRT_B))
    blrt = pd.DataFrame(blrt_rows)
    fit = fit.merge(blrt, left_on="n_classes", right_on="k", how="left").drop(columns=["k"])
    utils.save_table(fit, "T11_lca_fit_statistics")
    utils.log(fit.to_string(index=False))

    valid = fit[(fit["smallest_class_share"] >= 0.03) & (fit["n_classes"] > 1)]
    k_best = int(valid.sort_values("bic").iloc[0]["n_classes"]) if len(valid) else 2
    best = models[k_best]

    stability = []
    for s in range(5):
        m = utils.BernoulliLCA(k_best, n_starts=10, seed=config.SEED + 1000 * (s + 1)).fit(data)
        stability.append(
            {
                "seed_offset": 1000 * (s + 1),
                "loglik": m.loglik_,
                "ari_vs_main": float(adjusted_rand_score(best.labels_, m.labels_)),
            }
        )
    utils.save_table(pd.DataFrame(stability), "T12_lca_stability")

    profile = pd.DataFrame(best.probs_.T, index=bin_cols, columns=[f"class_{i+1}" for i in range(k_best)])
    profile.loc["class_share"] = best.pi_
    profile.loc["class_n"] = best.class_sizes_
    utils.save_table(profile.reset_index().rename(columns={"index": "item"}), "T13_lca_profiles")
    utils.log(profile.round(3).to_string())

    inv = inv.copy()
    inv["lca_class"] = best.labels_ + 1
    inv["lca_max_posterior"] = best.resp_.max(axis=1)
    utils.save_derived(inv[["server_id", "lca_class", "lca_max_posterior"]], "lca_classes.csv")

    prof_stats = inv.groupby("lca_class").agg(
        n=("server_id", "size"),
        mean_tool_count=("tool_count", "mean"),
        share_state_changing=("share_state_changing", "mean"),
        share_any_annotation=("share_any_annotation", "mean"),
        share_model_directed=("share_model_directed", "mean"),
        share_sensitive=("any_sensitive_capability", "mean"),
        provenance_index=("provenance_index", "mean"),
        mass_publisher=("mass_publisher", "mean"),
        median_downloads_per_week=("downloads_per_week", "median"),
        n_with_downloads=("downloads_per_week", lambda x: int(x.notna().sum())),
    )
    utils.save_table(prof_stats.reset_index(), "T14_lca_class_characteristics")
    utils.log(prof_stats.round(3).to_string())

    cat = pd.crosstab(inv["lca_class"], inv["category_rule"], normalize="index")
    utils.save_table(cat.reset_index(), "T15_lca_class_by_category")

    src = pd.crosstab(inv["lca_class"], inv["inventory_source"], normalize="index")
    utils.save_table(src.reset_index(), "T15b_lca_class_by_inventory_source")

    utils.json_dump({"k_selected": k_best, "n_lca": int(best.n_), "entropy": float(best.entropy_)}, "T11b_lca_selection")


if __name__ == "__main__":
    main()
