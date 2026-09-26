import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import config
import utils


def read_table(name):
    path = config.TABLES / f"{name}.csv"
    return pd.read_csv(path) if path.exists() else None


def read_json(name):
    path = config.TABLES / f"{name}.json"
    return json.loads(path.read_text()) if path.exists() else None


def fig_lca_profiles():
    prof = read_table("T13_lca_profiles")
    if prof is None:
        return
    items = prof[~prof["item"].isin(["class_share", "class_n"])].set_index("item")
    shares = prof[prof["item"] == "class_share"].drop(columns=["item"]).iloc[0].astype(float)
    fig, ax = plt.subplots(figsize=(1.9 * items.shape[1] + 3, 0.42 * len(items) + 2))
    data = items.astype(float).values
    im = ax.imshow(data, aspect="auto", cmap="Greys", vmin=0, vmax=1)
    ax.set_xticks(range(items.shape[1]))
    ax.set_xticklabels([f"{c}\n({shares[c]*100:.0f}%)" for c in items.columns])
    ax.set_yticks(range(len(items)))
    ax.set_yticklabels([i.replace("sig_", "") for i in items.index])
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            ax.text(j, i, f"{data[i, j]:.2f}", ha="center", va="center", color="black" if data[i, j] < 0.55 else "white", fontsize=8)
    ax.set_title("Latent class item-response probabilities")
    fig.colorbar(im, ax=ax, shrink=0.7)
    fig.tight_layout()
    fig.savefig(config.FIGURES / "F1_lca_profiles.png", dpi=300)
    plt.close(fig)


def fig_coefficients():
    tab = read_table("T18_rq2_ppml")
    if tab is None:
        return
    d = tab[(tab["model"] == "main_M5_full")]
    terms = ["log_tool_count", "share_state_changing", "any_sensitive_capability", "share_model_directed", "sig_org_owner", "sig_has_ci", "sig_security_policy", "sig_provenance_attestation", "sig_domain_verified", "sig_active_release"]
    d = d[d["term"].isin(terms)].set_index("term").reindex(terms).dropna(subset=["coef"])
    fig, ax = plt.subplots(figsize=(7, 0.45 * len(d) + 2))
    y = np.arange(len(d))
    ax.errorbar(d["coef"], y, xerr=[d["coef"] - d["ci_low"], d["ci_high"] - d["coef"]], fmt="o", color="black", ecolor="grey", capsize=3)
    ax.axvline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_yticks(y)
    ax.set_yticklabels([t.replace("sig_", "").replace("_", " ") for t in d.index])
    ax.invert_yaxis()
    ax.set_xlabel("PPML coefficient (log downloads), 95% cluster-robust CI")
    ax.set_title("Adoption associations, full model")
    fig.tight_layout()
    fig.savefig(config.FIGURES / "F2_adoption_coefficients.png", dpi=300)
    plt.close(fig)


def fig_tier_adoption():
    rq2 = utils.read_derived("rq2_sample.csv")
    rq2["downloads_per_week"] = pd.to_numeric(rq2["downloads_per_week"], errors="coerce")
    order = ["low", "medium", "high", "unobserved"]
    tiers = [t for t in order if t in set(rq2["provenance_tier"])]
    data = [np.log1p(rq2.loc[rq2["provenance_tier"] == t, "downloads_per_week"].dropna().values) for t in tiers]
    if not data:
        return
    fig, ax = plt.subplots(figsize=(6, 4))
    tick_labels = [f"{t}\n(n={len(d)})" for t, d in zip(tiers, data)]
    try:
        ax.boxplot(data, tick_labels=tick_labels, showfliers=False)
    except TypeError:
        ax.boxplot(data, labels=tick_labels, showfliers=False)
    ax.set_ylabel("log(1 + downloads per week)")
    ax.set_xlabel("Provenance tier")
    ax.set_title("Adoption by provenance tier")
    fig.tight_layout()
    fig.savefig(config.FIGURES / "F3_adoption_by_provenance_tier.png", dpi=300)
    plt.close(fig)


def fig_permissiveness_adoption():
    q = read_table("T26_rq2_adoption_by_tool_quartile")
    if q is None:
        return
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(q["tool_count_quartile"].astype(str), q["median_downloads_per_week"], color="0.4")
    ax.set_ylabel("Median downloads per week")
    ax.set_xlabel("Tool-count quartile")
    ax.set_title("Adoption by connector scope")
    fig.tight_layout()
    fig.savefig(config.FIGURES / "F4_adoption_by_tool_quartile.png", dpi=300)
    plt.close(fig)


def main():
    utils.log("=== 10_report ===")
    key = {}

    flow = read_table("T00b_analysis_samples")
    if flow is not None:
        key["samples"] = {r["stage"]: int(r["n"]) for _, r in flow.iterrows()}

    acc = read_table("T05_final_measure_accuracy")
    if acc is not None:
        key["measurement"] = acc.set_index("flag")[["measure_source", "precision", "recall", "f1", "kappa", "prevalence_final", "adjusted_prevalence"]].to_dict("index")

    wprev = read_table("T02b_design_weighted_prevalence")
    if wprev is not None:
        key["design_weighted_prevalence"] = wprev.set_index("measure")[["weighted_prevalence_human", "ci_low", "ci_high"]].to_dict("index")

    lca = read_json("T11b_lca_selection")
    if lca:
        key["lca"] = lca

    ppml = read_table("T18_rq2_ppml")
    if ppml is not None:
        m5 = ppml[ppml["model"] == "main_M5_full"]
        key["rq2_full_model"] = m5[m5["term"].isin(["log_tool_count", "share_state_changing", "any_sensitive_capability", "share_model_directed"])][
            ["term", "coef", "se", "p", "irr", "n_obs", "n_clusters"]
        ].to_dict("records")

    wild = read_table("T21_rq2_wild_cluster_bootstrap")
    if wild is not None:
        key["rq2_wild_cluster_p"] = wild.set_index("term")["p_wild_cluster"].to_dict()

    ann = read_table("T27_annotation_presence")
    if ann is not None:
        key["annotation_presence"] = ann.set_index("annotation")[["share", "ci_low", "ci_high"]].to_dict("index")

    agree = read_json("T29b_readonly_agreement")
    if agree:
        key["readonly_agreement"] = agree

    mis = read_table("T30_misdeclaration_rates")
    if mis is not None:
        key["misdeclaration"] = mis.set_index("measure")[["rate", "ci_low", "ci_high", "n_tools"]].to_dict("index")

    md = read_json("T35_model_directed_overall")
    if md:
        key["model_directed_overall"] = md

    mdadopt = read_table("T39_model_directed_adoption")
    if mdadopt is not None:
        key["model_directed_adoption"] = mdadopt[mdadopt["term"].isin(["share_model_directed", "any_model_directed", "share_model_directed_rule"])][
            ["model", "term", "coef", "se", "p", "irr"]
        ].to_dict("records")

    panel = read_json("T43_version_panel_summary")
    if panel:
        key["version_panel"] = panel

    dist = read_json("T46_outcome_distribution")
    if dist:
        key["outcome_distribution"] = dist

    utils.json_dump(key, "T99_key_results")

    lines = ["KEY RESULTS", ""]
    for k, v in key.items():
        lines.append(f"[{k}]")
        lines.append(json.dumps(v, indent=2, default=str))
        lines.append("")
    utils.write_text("\n".join(lines), "T99_key_results_readable")

    fig_lca_profiles()
    fig_coefficients()
    fig_tier_adoption()
    fig_permissiveness_adoption()
    utils.log("figures written to output/figures")


if __name__ == "__main__":
    main()
