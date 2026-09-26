import json
import re

import numpy as np
import pandas as pd

import config
import utils

SNAPSHOT = pd.Timestamp("2026-09-19", tz="UTC")
URL_RE = re.compile(r"https?://\S+|www\.\S+")
NS_RE = re.compile(r"\b(io|com|org|net|dev|app|ai|co|fr|world)\.[a-z0-9\.\-_]+", re.I)

TOOL_COLS = [
    "server_id",
    "tool_name",
    "tool_title",
    "description",
    "input_schema_json",
    "source",
    "source_endpoint",
    "server_version",
    "package_version",
    "tool_inventory_pass",
    "flag_state_changing",
    "flag_filesystem",
    "flag_shell_exec",
    "flag_outbound_network",
    "flag_credential_handling",
    "flag_model_directed_language",
    "description_length_chars",
    "has_description",
    "schema_n_properties",
    "schema_n_required",
    "annotations_observable",
    "ann_has_readOnlyHint",
    "ann_value_readOnlyHint",
    "ann_has_destructiveHint",
    "ann_value_destructiveHint",
    "ann_has_idempotentHint",
    "ann_value_idempotentHint",
    "ann_has_openWorldHint",
    "ann_value_openWorldHint",
    "ann_n_hints_present",
    "ann_complete_all4",
    "ann_has_any",
]

BOOL_TOOL_COLS = [
    "flag_state_changing",
    "flag_filesystem",
    "flag_shell_exec",
    "flag_outbound_network",
    "flag_credential_handling",
    "flag_model_directed_language",
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
    "has_description",
]


def check_files():
    missing = [f for f in config.REQUIRED_FILES if not (config.DATA / f).exists()]
    if missing:
        raise SystemExit("Missing required files in data/: " + ", ".join(missing))


def clean_text(s):
    s = "" if pd.isna(s) else str(s)
    s = URL_RE.sub(" ", s)
    s = NS_RE.sub(" ", s)
    return s.lower()


def brand_category(text):
    hits = []
    for cat, terms in config.BRAND_CATEGORIES.items():
        for t in terms:
            if re.search(r"(?<![a-z0-9])" + re.escape(t) + r"(?![a-z0-9])", text):
                hits.append(cat)
                break
    if not hits:
        return "none", 0
    return hits[0], len(hits)


def schema_text(js):
    if pd.isna(js):
        return ""
    try:
        obj = json.loads(js)
    except Exception:
        return ""
    props = obj.get("properties", {}) if isinstance(obj, dict) else {}
    if not isinstance(props, dict):
        return ""
    parts = []
    for k, v in props.items():
        parts.append(str(k))
        if isinstance(v, dict):
            d = v.get("description")
            if isinstance(d, str):
                parts.append(d[:120])
    return " ".join(parts)[:600]


def to_dt(s):
    return pd.to_datetime(s, errors="coerce", utc=True, format="mixed")


def load_servers():
    s = utils.read_csv("servers.csv")
    for c in [
        "in_census",
        "in_adoption_frame",
        "in_adoption_sample",
        "in_rq2_sample",
        "in_remote_probe_set",
        "has_package",
        "has_remote",
        "is_duplicate",
        "mass_publisher",
        "smithery_only",
        "smithery_verified",
        "publisher_is_vendor",
    ]:
        if c in s.columns:
            s[c] = utils.to_bool(s[c])
    s["published_at_dt"] = to_dt(s["published_at"])
    return s


def load_repos():
    r = utils.read_csv("repos.csv")
    for c in ["has_ci_workflows", "has_security_policy", "has_vulnerability_alerts", "is_archived", "is_fork"]:
        if c in r.columns:
            r[c] = utils.to_bool(r[c])
    r["repo_created_dt"] = to_dt(r["created_at"])
    r["repo_pushed_dt"] = to_dt(r["pushed_at"])
    keep = [
        "repo_url_canonical",
        "repo_status",
        "owner_type",
        "owner_login",
        "stars",
        "forks",
        "is_archived",
        "is_fork",
        "primary_language",
        "license_spdx",
        "open_issues",
        "release_count",
        "has_security_policy",
        "has_ci_workflows",
        "n_ci_workflow_files",
        "repo_created_dt",
        "repo_pushed_dt",
    ]
    return r[[c for c in keep if c in r.columns]].drop_duplicates("repo_url_canonical")


def load_packages_enriched():
    p = utils.read_csv("packages_enriched.csv")
    p["has_provenance_attestation"] = utils.to_bool(p["has_provenance_attestation"])
    p["is_deprecated"] = utils.to_bool(p["is_deprecated"])
    p["pkg_first_dt"] = to_dt(p["first_version_published_at"]).fillna(to_dt(p["created_at"])).fillna(to_dt(p["first_release_at"]))
    p["pkg_latest_dt"] = to_dt(p["latest_version_published_at"]).fillna(to_dt(p["latest_release_at"]))
    p["n_versions_any"] = pd.to_numeric(p["n_versions"], errors="coerce").fillna(pd.to_numeric(p["n_releases"], errors="coerce"))
    p["pkg_license"] = p["license"].fillna(p["license_expression"])
    keep = [
        "canonical_package_id",
        "server_id",
        "ecosystem",
        "package_name",
        "meta_status",
        "downloads_status",
        "downloads_window_start",
        "downloads_window_end",
        "has_provenance_attestation",
        "is_deprecated",
        "n_maintainers",
        "pkg_first_dt",
        "pkg_latest_dt",
        "n_versions_any",
        "pkg_license",
        "n_vulnerabilities_osv",
    ]
    return p[keep]


def aggregate_downloads():
    d = utils.read_csv("downloads_weekly.csv")
    d["spike_flag"] = utils.to_bool_strict(d["spike_flag"])
    agg = d.groupby("canonical_package_id").agg(
        downloads_total=("downloads", "sum"),
        weeks_observed=("downloads", "size"),
        downloads_median_week=("downloads", "median"),
        downloads_max_week=("downloads", "max"),
        n_spike_weeks=("spike_flag", "sum"),
    )
    nospike = d[~d["spike_flag"]].groupby("canonical_package_id").agg(downloads_total_nospike=("downloads", "sum"))
    last12 = d.sort_values("iso_week").groupby("canonical_package_id").tail(12).groupby("canonical_package_id").agg(
        downloads_last12w=("downloads", "sum")
    )
    return agg.join(nospike, how="left").join(last12, how="left").reset_index()


def build_signals(df):
    df["sig_domain_verified"] = (df["namespace_verification_method"] == "domain_based").astype(float)
    df["sig_org_owner"] = np.where(df["owner_type"].isna(), np.nan, (df["owner_type"] == "Organization").astype(float))
    df["sig_has_ci"] = df["has_ci_workflows"].map({True: 1.0, False: 0.0})
    df["sig_security_policy"] = df["has_security_policy"].map({True: 1.0, False: 0.0})
    df["sig_provenance_attestation"] = df["has_provenance_attestation"].map({True: 1.0, False: 0.0})
    has_lic = df["pkg_license"].notna() | df["license_spdx"].notna()
    unobserved = df["pkg_license"].isna() & df["license_spdx"].isna() & df["repo_status"].isna() & df["meta_status"].isna()
    df["sig_license"] = np.where(unobserved, np.nan, has_lic.astype(float))
    recent = df["pkg_latest_dt"].fillna(df["repo_pushed_dt"])
    days = (SNAPSHOT - recent).dt.days
    df["days_since_release"] = days
    df["sig_active_release"] = np.where(days.isna(), np.nan, (days <= 90).astype(float))
    df["n_provenance_signals"] = df[config.PROVENANCE_ITEMS].sum(axis=1, min_count=1)
    df["n_provenance_observed"] = df[config.PROVENANCE_ITEMS].notna().sum(axis=1)
    df["provenance_index"] = df["n_provenance_signals"] / df["n_provenance_observed"].replace(0, np.nan)
    df["provenance_tier"] = pd.cut(df["provenance_index"], bins=[-0.01, 0.2, 0.5, 1.01], labels=["low", "medium", "high"]).astype(str)
    df.loc[df["provenance_index"].isna(), "provenance_tier"] = "unobserved"
    return df


def build_controls(df):
    first = df["pkg_first_dt"].fillna(df["repo_created_dt"]).fillna(df["published_at_dt"])
    df["age_days"] = (SNAPSHOT - first).dt.days.clip(lower=1)
    df["log_age_days"] = np.log(df["age_days"])
    df["n_versions_any"] = pd.to_numeric(df["n_versions_any"], errors="coerce").fillna(1).clip(lower=1)
    df["log_n_versions"] = np.log(df["n_versions_any"])
    df["description_length"] = df["description"].fillna("").astype(str).str.len().clip(lower=1)
    df["log_description_length"] = np.log(df["description_length"])
    df["publisher_portfolio_size"] = pd.to_numeric(df["publisher_portfolio_size"], errors="coerce").fillna(1).clip(lower=1)
    df["log_publisher_portfolio"] = np.log(df["publisher_portfolio_size"])
    df["is_deprecated"] = utils.to_bool(df["is_deprecated"]).fillna(False).astype(float)
    df["stars"] = pd.to_numeric(df["stars"], errors="coerce")
    df["log_stars"] = np.log1p(df["stars"])
    df["mass_publisher"] = utils.to_bool(df["mass_publisher"]).fillna(False).astype(float)
    return df


def add_categories(df, tool_names):
    base_text = df["short_name"].fillna("").astype(str) + " " + df["title"].fillna("").astype(str) + " " + df["description"].fillna("").astype(str)
    txt = base_text.map(clean_text)
    tn = df["server_id"].map(tool_names).fillna("")
    txt = (txt + " " + tn.map(clean_text)).str.replace(r"\s+", " ", regex=True)
    cats = txt.map(brand_category)
    df["brand_category"] = [c[0] for c in cats]
    df["brand_category_n_matched"] = [c[1] for c in cats]
    df["category_rule"] = df["enterprise_category_primary"].fillna("none")
    df["is_enterprise_brand"] = (df["brand_category"] != "none").astype(float)
    return df


def build_version_panel():
    tv = utils.read_csv("tool_versions.csv")
    tv = tv[tv["probe_outcome"].astype(str).str.startswith("ok")].copy()
    tv["version_published_dt"] = to_dt(tv["version_published_at"])
    for c in ["tool_count", "share_flag_state_changing", "share_tools_any_annotation", "share_tools_complete_annotations"]:
        if c in tv.columns:
            tv[c] = pd.to_numeric(tv[c], errors="coerce")
    dw = utils.read_csv("downloads_weekly.csv")
    dw["week_start_dt"] = to_dt(dw["week_start_observed"])
    rows = []
    for pid, grp in tv.groupby("package_id"):
        wk = dw[dw["canonical_package_id"] == pid]
        if wk.empty:
            continue
        for _, r in grp.sort_values("version_published_dt").iterrows():
            if pd.isna(r["version_published_dt"]):
                continue
            start = r["version_published_dt"]
            end = start + pd.Timedelta(weeks=config.PANEL_POST_WEEKS)
            w = wk[(wk["week_start_dt"] >= start) & (wk["week_start_dt"] < end)]
            if len(w) < 2:
                continue
            rows.append(
                {
                    "package_id": pid,
                    "server_id": r["server_id"],
                    "version": r["version"],
                    "version_published_dt": start,
                    "tool_count": r["tool_count"],
                    "share_flag_state_changing": r["share_flag_state_changing"],
                    "share_tools_any_annotation": r.get("share_tools_any_annotation", np.nan),
                    "downloads_post": float(w["downloads"].sum()),
                    "weeks_post": int(len(w)),
                    "downloads_per_week_post": float(w["downloads"].mean()),
                    "ecosystem": r["ecosystem"],
                }
            )
    panel = pd.DataFrame(rows)
    if len(panel):
        panel = panel.sort_values(["package_id", "version_published_dt"])
        panel["log_downloads_per_week_post"] = np.log1p(panel["downloads_per_week_post"])
        panel["log_tool_count"] = np.log1p(panel["tool_count"])
        n = panel.groupby("package_id")["version"].transform("size")
        panel["n_versions_in_panel"] = n
        panel = panel[n >= 2]
    return panel


def main():
    check_files()
    utils.log("=== 01_prepare ===")

    servers = load_servers()
    features = utils.read_csv("features_server.csv")
    repos = load_repos()
    pkg = load_packages_enriched()
    downloads = aggregate_downloads()

    tools = utils.read_csv("tools.csv", usecols=TOOL_COLS)
    tools = tools[~tools["tool_inventory_pass"].astype(str).str.contains("version", case=False, na=False)].copy()
    tools = tools.reset_index(drop=True)
    tools["tool_uid"] = np.arange(len(tools))
    for c in BOOL_TOOL_COLS:
        tools[c] = utils.to_bool(tools[c])
    tools["schema_text"] = tools["input_schema_json"].map(schema_text)
    tools["text_all"] = (
        tools["tool_name"].fillna("").astype(str)
        + " || "
        + tools["tool_title"].fillna("").astype(str)
        + " || "
        + tools["description"].fillna("").astype(str)
        + " || "
        + tools["schema_text"].fillna("")
    )
    tools = tools.drop(columns=["input_schema_json"])
    utils.save_derived(tools, "tool_base.csv")

    tool_names = tools.groupby("server_id")["tool_name"].apply(lambda x: " ".join(map(str, x.head(60))))

    pkg_primary = pkg.sort_values("canonical_package_id").drop_duplicates("server_id").merge(downloads, on="canonical_package_id", how="left")

    feat_keep = [c for c in features.columns if c == "server_id" or c not in servers.columns]
    server = (
        servers.merge(features[feat_keep], on="server_id", how="left")
        .merge(repos, on="repo_url_canonical", how="left")
        .merge(pkg_primary, on="server_id", how="left")
    )
    server = build_signals(server)
    server = build_controls(server)
    server = add_categories(server, tool_names)

    server["downloads_total"] = pd.to_numeric(server["downloads_total"], errors="coerce")
    win_start = to_dt(server["downloads_window_start"])
    win_end = to_dt(server["downloads_window_end"])
    eff_start = pd.concat([win_start, server["pkg_first_dt"]], axis=1).max(axis=1)
    server["exposure_days"] = ((win_end - eff_start).dt.days + 1).clip(lower=1)
    server["exposure_weeks"] = server["exposure_days"] / 7.0
    server["log_exposure_days"] = np.log(server["exposure_days"])
    server["downloads_per_week"] = server["downloads_total"] / server["exposure_weeks"]
    server["log1p_downloads_per_week"] = np.log1p(server["downloads_per_week"])
    server["log1p_downloads_total"] = np.log1p(server["downloads_total"])
    server["downloads_total_nospike"] = pd.to_numeric(server["downloads_total_nospike"], errors="coerce")
    server["publisher_id"] = server["publisher_id"].fillna(server["namespace"]).astype(str)

    utils.save_derived(server, "server_base.csv")

    panel = build_version_panel()
    utils.save_derived(panel, "version_panel.csv")

    flow = pd.DataFrame(
        [
            {"stage": "servers_rows", "n": len(servers)},
            {"stage": "census", "n": int(utils.to_bool_strict(server["in_census"]).sum())},
            {"stage": "tool_records_primary", "n": len(tools)},
            {"stage": "servers_with_tool_inventory", "n": int(tools["server_id"].nunique())},
            {"stage": "version_panel_observations", "n": len(panel)},
            {"stage": "version_panel_packages", "n": int(panel["package_id"].nunique()) if len(panel) else 0},
        ]
    )
    utils.save_table(flow, "T00a_prepare_flow")
    utils.log(flow.to_string(index=False))


if __name__ == "__main__":
    main()
