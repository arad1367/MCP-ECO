import json
import re

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

CONTROLS = [
    "log_age_days",
    "log_n_versions",
    "log_description_length",
    "log_publisher_portfolio",
    "is_deprecated",
    "mass_publisher",
]

PERMISSIVENESS = ["log_tool_count", "share_state_changing", "any_sensitive_capability"]

FE = [("primary_ecosystem", "eco"), ("category_rule", "cat")]

INJECTION_PATTERNS = {
    "do_not_tell_user": r"do not (tell|inform|mention to) the user",
    "ignore_previous": r"ignore (all |any )?(previous|prior|earlier) (instructions|prompts)",
    "before_other_tools": r"before (calling|using) (any )?other tool",
    "always_use_this": r"always use this tool",
    "must_call": r"(you )?must (always )?call this",
    "highest_priority": r"(highest|top) priority",
    "do_not_ask": r"do not ask (the user|for) (permission|confirmation)",
    "override_instructions": r"(override|disregard) (the )?(system|previous) (prompt|instructions)",
}


def cluster_ci(df, col, cluster_col, n_boot=500, seed=None):
    rng = np.random.default_rng(seed or config.SEED)
    v = pd.to_numeric(df[col], errors="coerce")
    g = df[cluster_col].astype(str)
    ok = v.notna()
    v, g = v[ok], g[ok]
    if len(v) == 0:
        return np.nan, np.nan, np.nan, 0
    agg = pd.DataFrame({"s": v.groupby(g).sum(), "n": v.groupby(g).size()})
    s, n = agg["s"].values, agg["n"].values
    k = len(s)
    if k < 2:
        return float(v.mean()), np.nan, np.nan, int(len(v))
    idx = rng.integers(0, k, size=(n_boot, k))
    boots = s[idx].sum(axis=1) / n[idx].sum(axis=1)
    return float(v.mean()), float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5)), int(len(v))


def main():
    utils.log("=== 08_rq4 ===")
    t = utils.read_derived("tool_level.csv")
    t = t[utils.to_bool_strict(t["in_census"])].copy()
    t["publisher_id"] = t["publisher_id"].fillna("unknown").astype(str)
    t["model_directed"] = pd.to_numeric(t["model_directed"], errors="coerce").fillna(0.0)

    est, lo, hi, n = cluster_ci(t, "model_directed", "publisher_id")
    overall = {"share_tools_model_directed": est, "ci_low": lo, "ci_high": hi, "n_tools": n, "n_servers": int(t["server_id"].nunique())}
    utils.json_dump(overall, "T35_model_directed_overall")
    utils.log(str(overall))

    rows = []
    for tier, g in t.groupby("provenance_tier"):
        e, l, h, nn = cluster_ci(g, "model_directed", "publisher_id", n_boot=300)
        rows.append({"group_type": "provenance_tier", "group": tier, "share": e, "ci_low": l, "ci_high": h, "n_tools": nn, "n_servers": g["server_id"].nunique()})
    for cat, g in t.groupby("category_rule"):
        if len(g) < 200:
            continue
        e, l, h, nn = cluster_ci(g, "model_directed", "publisher_id", n_boot=300)
        rows.append({"group_type": "category", "group": cat, "share": e, "ci_low": l, "ci_high": h, "n_tools": nn, "n_servers": g["server_id"].nunique()})
    for src, g in t.groupby("source"):
        e, l, h, nn = cluster_ci(g, "model_directed", "publisher_id", n_boot=300)
        rows.append({"group_type": "inventory_source", "group": src, "share": e, "ci_low": l, "ci_high": h, "n_tools": nn, "n_servers": g["server_id"].nunique()})
    grp = pd.DataFrame(rows)
    utils.save_table(grp, "T36_model_directed_by_group")
    utils.log(grp.round(3).to_string(index=False))

    text = t["description"].fillna("").astype(str).str.lower()
    inj_rows = []
    for name, pat in INJECTION_PATTERNS.items():
        hit = text.str.contains(pat, regex=True, na=False)
        inj_rows.append(
            {
                "pattern": name,
                "regex": pat,
                "n_tools": int(hit.sum()),
                "share_tools": float(hit.mean()),
                "n_servers": int(t.loc[hit, "server_id"].nunique()),
                "n_publishers": int(t.loc[hit, "publisher_id"].nunique()),
            }
        )
    inj = pd.DataFrame(inj_rows).sort_values("n_tools", ascending=False)
    utils.save_table(inj, "T37_injection_style_patterns")
    utils.log(inj.to_string(index=False))

    t["any_injection_pattern"] = 0.0
    for pat in INJECTION_PATTERNS.values():
        t["any_injection_pattern"] = np.maximum(t["any_injection_pattern"], text.str.contains(pat, regex=True, na=False).astype(float))
    e, l, h, nn = cluster_ci(t, "any_injection_pattern", "publisher_id")
    utils.json_dump({"share_tools_any_injection_pattern": e, "ci_low": l, "ci_high": h, "n_tools": nn}, "T37b_injection_overall")

    t["state_changing_num"] = t["measured_state_changing"].astype(float)
    t["annotated"] = utils.to_bool_strict(t["ann_has_any"]).astype(float)
    t["log_tool_count"] = np.log1p(pd.to_numeric(t["tool_count"], errors="coerce"))
    t["log_description_length_tool"] = np.log1p(pd.to_numeric(t["description_length_chars"], errors="coerce").fillna(0))
    for c in PROVENANCE:
        t[c] = pd.to_numeric(t[c], errors="coerce").fillna(0.0)
    t["log_age_days"] = pd.to_numeric(t["log_age_days"], errors="coerce").fillna(pd.to_numeric(t["log_age_days"], errors="coerce").median())
    t["log_publisher_portfolio"] = pd.to_numeric(t["log_publisher_portfolio"], errors="coerce").fillna(0.0)
    t["mass_publisher"] = pd.to_numeric(t["mass_publisher"], errors="coerce").fillna(0.0)

    res, _, _ = utils.fit_logit(
        t,
        "model_directed",
        ["state_changing_num", "annotated", "log_tool_count", "log_description_length_tool", "log_age_days", "log_publisher_portfolio", "mass_publisher"] + PROVENANCE,
        "publisher_id",
        [("primary_ecosystem", "eco"), ("category_rule", "cat")],
    )
    tab = utils.tidy(res, "logit_model_directed_tool_level", n_clusters=t["publisher_id"].nunique(), extra={"estimator": "logit"})
    tab["odds_ratio"] = np.exp(tab["coef"])
    utils.save_table(tab, "T38_model_directed_correlates")
    utils.log(tab[["term", "coef", "se", "p", "odds_ratio"]].head(14).round(3).to_string(index=False))

    rq2 = utils.read_derived("rq2_sample.csv")
    for c in PROVENANCE:
        rq2[c] = pd.to_numeric(rq2[c], errors="coerce").fillna(0.0)
    for c in ["share_model_directed", "share_state_changing", "any_sensitive_capability", "share_model_directed_rule"]:
        rq2[c] = pd.to_numeric(rq2[c], errors="coerce").fillna(0.0)
    rq2["any_model_directed"] = (rq2["share_model_directed"] > 0).astype(float)
    rq2["md_x_provenance"] = rq2["share_model_directed"] * pd.to_numeric(rq2["provenance_index"], errors="coerce").fillna(0)
    offset = rq2["log_exposure_days"].values

    out = []
    for name, regs in {
        "ppml_md_share": CONTROLS + PERMISSIVENESS + PROVENANCE + ["share_model_directed"],
        "ppml_md_binary": CONTROLS + PERMISSIVENESS + PROVENANCE + ["any_model_directed"],
        "ppml_md_rule_measure": CONTROLS + PERMISSIVENESS + PROVENANCE + ["share_model_directed_rule"],
        "ppml_md_interaction": CONTROLS + PERMISSIVENESS + PROVENANCE + ["share_model_directed", "md_x_provenance"],
    }.items():
        r, _, _ = utils.fit_ppml(rq2, "downloads_total", regs, "publisher_id", FE, offset=offset)
        out.append(utils.tidy(r, name, n_clusters=rq2["publisher_id"].nunique(), extra={"estimator": "PPML"}))
    md_tab = pd.concat(out, ignore_index=True)
    utils.save_table(md_tab, "T39_model_directed_adoption")
    keys = ["share_model_directed", "any_model_directed", "share_model_directed_rule", "md_x_provenance"]
    utils.log(md_tab[md_tab["term"].isin(keys)][["model", "term", "coef", "se", "p", "irr"]].round(4).to_string(index=False))

    wild = utils.wild_cluster_bootstrap(
        rq2,
        "log1p_downloads_per_week",
        CONTROLS + PERMISSIVENESS + PROVENANCE + ["share_model_directed"],
        "publisher_id",
        ["share_model_directed"],
        FE,
    )
    utils.save_table(wild, "T40_model_directed_wild_bootstrap")
    utils.log(wild.round(4).to_string(index=False))

    srv = utils.read_derived("rq1_inventory.csv")
    srv["any_model_directed"] = pd.to_numeric(srv["any_model_directed"], errors="coerce").fillna(0)
    cross = srv.groupby("any_model_directed").agg(
        n=("server_id", "size"),
        mean_tool_count=("tool_count", "mean"),
        mean_share_state_changing=("share_state_changing", "mean"),
        mean_share_annotated=("share_any_annotation", "mean"),
        mean_provenance_index=("provenance_index", "mean"),
        share_sensitive=("any_sensitive_capability", "mean"),
    )
    utils.save_table(cross.reset_index(), "T41_model_directed_server_profile")
    utils.log(cross.round(3).to_string())


if __name__ == "__main__":
    main()
