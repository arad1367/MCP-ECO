from pathlib import Path

BASE = Path(__file__).resolve().parent
DATA = BASE / "data"
DERIVED = BASE / "derived"
OUT = BASE / "output"
TABLES = OUT / "tables"
FIGURES = OUT / "figures"
LOGS = OUT / "logs"

for _p in (DERIVED, OUT, TABLES, FIGURES, LOGS):
    _p.mkdir(parents=True, exist_ok=True)

SEED = 20260917
N_BOOT = 2000
N_WILD_BOOT = 999
LCA_MAX_CLASSES = 6
LCA_N_STARTS = 20
LCA_BLRT_B = 25
SPIKE_RULE_EXCLUDE = True
MIN_EXPOSURE_DAYS = 28
PANEL_POST_WEEKS = 8

REQUIRED_FILES = [
    "servers.csv",
    "features_server.csv",
    "tools.csv",
    "downloads_weekly.csv",
    "packages.csv",
    "packages_enriched.csv",
    "repos.csv",
    "probe_log.csv",
    "tool_versions.csv",
    "sample_flow.csv",
    "validation_sample_filled.csv",
    "vendor_validation_sample_filled.csv",
]

BRAND_CATEGORIES = {
    "crm": ["salesforce", "hubspot", "pipedrive", "zoho", "dynamics 365", "freshsales", "copper crm", "close.com"],
    "erp": ["sap", "netsuite", "odoo", "oracle erp", "workday financial"],
    "hr": ["workday", "bamboohr", "greenhouse", "lever", "personio", "rippling", "gusto", "hris"],
    "finance_payments": ["stripe", "paypal", "quickbooks", "xero", "plaid", "adyen", "braintree", "square", "wise", "revolut", "coinbase"],
    "databases": ["postgres", "postgresql", "mysql", "mariadb", "mongodb", "sqlite", "redis", "snowflake", "bigquery", "clickhouse", "elasticsearch", "supabase", "duckdb", "neo4j", "cassandra", "dynamodb"],
    "cloud": ["aws", "amazon web services", "azure", "google cloud", "gcp", "cloudflare", "digitalocean", "kubernetes", "terraform", "heroku", "vercel", "netlify", "docker"],
    "collaboration": ["slack", "notion", "confluence", "jira", "asana", "trello", "monday.com", "linear", "airtable", "microsoft teams", "sharepoint", "google workspace", "google drive", "gmail", "outlook", "zoom", "discord", "clickup"],
    "code_hosting": ["github", "gitlab", "bitbucket", "gitea", "azure devops", "sourcehut"],
}

PROVENANCE_ITEMS = [
    "sig_domain_verified",
    "sig_org_owner",
    "sig_has_ci",
    "sig_security_policy",
    "sig_provenance_attestation",
    "sig_license",
    "sig_active_release",
]

DISCLOSURE_ITEMS = [
    "sig_any_annotation",
    "sig_complete_annotation",
]

CAPABILITY_ITEMS = [
    "sig_any_state_changing",
    "sig_any_shell",
    "sig_any_filesystem",
    "sig_any_credential",
    "sig_many_tools",
]

LANGUAGE_ITEMS = ["sig_model_directed"]

LCA_ITEMS = PROVENANCE_ITEMS + DISCLOSURE_ITEMS + CAPABILITY_ITEMS + LANGUAGE_ITEMS

RQ2_CONTROLS = [
    "log_age_days",
    "log_n_versions",
    "log_description_length",
    "log_publisher_portfolio",
    "is_deprecated",
    "sig_license",
]

PROVENANCE_REGRESSORS = [
    "sig_domain_verified",
    "sig_org_owner",
    "sig_has_ci",
    "sig_security_policy",
    "sig_provenance_attestation",
    "sig_active_release",
]
