import numpy as np
import pandas as pd

import config
import utils

FLAG_PAIRS = [
    ("flag_state_changing", "human_state_changing"),
    ("flag_filesystem", "human_filesystem"),
    ("flag_shell_exec", "human_shell_exec"),
    ("flag_outbound_network", "human_outbound_network"),
    ("flag_credential_handling", "human_credential_handling"),
    ("flag_model_directed_language", "human_model_directed_language"),
]


def main():
    utils.log("=== 02_validation ===")
    v = utils.read_csv("validation_sample_filled.csv")
    rows = []
    for rule_col, human_col in FLAG_PAIRS:
        d = v[[rule_col, human_col, "source"]].dropna()
        pred = utils.to_bool_strict(d[rule_col]).astype(int).values
        truth = utils.to_bool_strict(d[human_col]).astype(int).values
        stats = utils.classification_stats(pred, truth)
        stats["measure"] = rule_col
        rows.append(stats)
    val = pd.DataFrame(rows)
    cols = ["measure", "n", "tp", "fp", "fn", "tn", "prevalence_rule", "prevalence_human", "precision", "precision_ci_low", "precision_ci_high", "recall", "recall_ci_low", "recall_ci_high", "f1", "specificity", "accuracy", "kappa", "kappa_ci_low", "kappa_ci_high", "percent_agreement"]
    val = val[[c for c in cols if c in val.columns]]
    utils.save_table(val, "T01_measure_validation")
    utils.log(val[["measure", "n", "precision", "recall", "f1", "kappa"]].to_string(index=False))

    by_source = []
    for rule_col, human_col in FLAG_PAIRS:
        for src, d in v.groupby("source"):
            d = d[[rule_col, human_col]].dropna()
            if len(d) < 20:
                continue
            pred = utils.to_bool_strict(d[rule_col]).astype(int).values
            truth = utils.to_bool_strict(d[human_col]).astype(int).values
            s = utils.classification_stats(pred, truth, n_boot=500)
            s["measure"] = rule_col
            s["source"] = src
            by_source.append(s)
    if by_source:
        bs = pd.DataFrame(by_source)[["measure", "source", "n", "precision", "recall", "f1", "kappa", "accuracy"]]
        utils.save_table(bs, "T01b_measure_validation_by_source")

    tools = utils.read_derived("tool_base.csv")
    adj_rows = []
    for rule_col, _ in FLAG_PAIRS:
        obs = utils.to_bool_strict(tools[rule_col]).mean()
        s = val[val["measure"] == rule_col].iloc[0]
        adj = utils.rogan_gladen(obs, s["recall"], s["specificity"])
        adj_rows.append(
            {
                "measure": rule_col,
                "observed_prevalence_tools": float(obs),
                "sensitivity": s["recall"],
                "specificity": s["specificity"],
                "adjusted_prevalence": adj,
                "n_tools": int(len(tools)),
            }
        )
    adj = pd.DataFrame(adj_rows)
    utils.save_table(adj, "T02_prevalence_bias_adjusted")
    utils.log(adj.to_string(index=False))

    pop_share = tools["source"].value_counts(normalize=True)
    samp_share = v["source"].value_counts(normalize=True)
    v = v.copy()
    v["design_weight"] = v["source"].map(pop_share) / v["source"].map(samp_share)
    w = v["design_weight"].values
    rng = np.random.default_rng(config.SEED)
    wrows = []
    for _, human_col in FLAG_PAIRS:
        yv = utils.to_bool_strict(v[human_col]).astype(float).values
        est = float(np.average(yv, weights=w))
        boots = []
        idx = np.arange(len(yv))
        for _ in range(config.N_BOOT):
            s = rng.choice(idx, size=len(idx), replace=True)
            boots.append(float(np.average(yv[s], weights=w[s])))
        wrows.append(
            {
                "measure": human_col,
                "n_labelled": int(len(yv)),
                "weighted_prevalence_human": est,
                "ci_low": float(np.percentile(boots, 2.5)),
                "ci_high": float(np.percentile(boots, 97.5)),
                "unweighted_prevalence_human": float(yv.mean()),
            }
        )
    wt = pd.DataFrame(wrows)
    utils.save_table(wt, "T02b_design_weighted_prevalence")
    utils.log(wt.to_string(index=False))

    vend = utils.read_csv("vendor_validation_sample_filled.csv")
    vr = {}
    d = vend.dropna(subset=["human_connected_service_correct"])
    vr["connected_service_accuracy"] = float(utils.to_bool_strict(d["human_connected_service_correct"]).mean())
    vr["connected_service_n"] = int(len(d))
    exact = (
        vend["connected_service"].astype(str).str.lower().str.strip()
        == vend["human_connected_service"].astype(str).str.lower().str.strip()
    )
    vr["connected_service_exact_match"] = float(exact.mean())
    dv = vend.dropna(subset=["human_publisher_is_vendor"])
    pred = utils.to_bool_strict(dv["publisher_is_vendor"]).astype(int).values
    truth = utils.to_bool_strict(dv["human_publisher_is_vendor"]).astype(int).values
    vs = utils.classification_stats(pred, truth)
    vs["measure"] = "publisher_is_vendor"
    vendor_tab = pd.DataFrame([vs])[["measure", "n", "tp", "fp", "fn", "tn", "precision", "recall", "f1", "kappa", "accuracy", "prevalence_rule", "prevalence_human"]]
    utils.save_table(vendor_tab, "T03_vendor_validation")
    utils.json_dump(vr, "T03b_connected_service_validation")
    utils.log(vendor_tab.to_string(index=False))
    utils.log(str(vr))

    coders = v["human_coder_id"].nunique()
    per_item = v.groupby("validation_row_id")["human_coder_id"].nunique().max()
    note = {
        "n_validation_tools": int(len(v)),
        "n_coders": int(coders),
        "max_coders_per_item": int(per_item),
        "inter_rater_reliability_computable": bool(per_item > 1),
        "note": "Each item was coded by a single coder; reported kappa compares the rule-based classifier with human coding, not two human coders.",
    }
    utils.json_dump(note, "T01c_validation_design_note")
    utils.log(str(note))


if __name__ == "__main__":
    main()
