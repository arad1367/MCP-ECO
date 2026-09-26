import json
import warnings

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import linalg, stats

import config

warnings.filterwarnings("ignore")

TRUE_SET = {"true", "True", "TRUE", "1", "yes", "Y", True, 1}


def to_bool(series):
    if series.dtype == bool:
        return series
    return series.map(lambda v: True if v in TRUE_SET else (False if pd.notna(v) else np.nan))


def to_bool_strict(series):
    s = to_bool(series)
    return s.fillna(False).astype(bool)


def read_csv(name, **kwargs):
    return pd.read_csv(config.DATA / name, low_memory=False, **kwargs)


def read_derived(name, **kwargs):
    return pd.read_csv(config.DERIVED / name, low_memory=False, **kwargs)


def save_derived(df, name):
    df.to_csv(config.DERIVED / name, index=False)
    return df


def save_table(df, name, index=False):
    path = config.TABLES / f"{name}.csv"
    df.to_csv(path, index=index)
    return path


def write_text(text, name):
    path = config.TABLES / f"{name}.txt"
    path.write_text(text, encoding="utf-8")
    return path


def log(msg, logfile="analysis_log.txt"):
    print(msg, flush=True)
    with open(config.LOGS / logfile, "a", encoding="utf-8") as fh:
        fh.write(str(msg) + "\n")


def dummies(df, col, prefix, min_count=1, base=None):
    s = df[col].fillna("missing").astype(str)
    counts = s.value_counts()
    keep = counts[counts >= min_count].index.tolist()
    s = s.where(s.isin(keep), "other")
    if base is None:
        base = s.value_counts().index[0]
    d = pd.get_dummies(s, prefix=prefix, drop_first=False)
    drop_col = f"{prefix}_{base}"
    if drop_col in d.columns:
        d = d.drop(columns=[drop_col])
    return d.astype(float), base


def prune_design(X):
    keep = []
    for c in X.columns:
        v = X[c].astype(float)
        if c == "const" or v.std(ddof=0) > 1e-10:
            keep.append(c)
    X = X[keep]
    M = np.nan_to_num(X.astype(float).values, nan=0.0, posinf=0.0, neginf=0.0)
    if M.shape[1] == 0:
        return X
    scale = np.maximum(np.abs(M).max(axis=0), 1e-12)
    Ms = M / scale
    q, r, piv = linalg.qr(Ms, mode="economic", pivoting=True)
    diag = np.abs(np.diag(r))
    tol = max(Ms.shape) * np.finfo(float).eps * (diag[0] if len(diag) else 1.0)
    rank = int((diag > max(tol, 1e-10)).sum())
    cols = list(X.columns)
    selected = sorted(piv[:rank])
    return X[[cols[i] for i in selected]]


def build_design(df, regressors, fe_cols=(), add_const=True, fe_min_count=5):
    X = df[list(regressors)].astype(float).copy()
    fe_info = {}
    for col, prefix in fe_cols:
        d, base = dummies(df, col, prefix, min_count=fe_min_count)
        X = pd.concat([X, d], axis=1)
        fe_info[col] = base
    if add_const:
        X = sm.add_constant(X, has_constant="add")
        X = X[["const"] + [c for c in X.columns if c != "const"]]
    X = prune_design(X)
    return X, fe_info


def fit_ppml(df, y, regressors, clusters, fe_cols=(), offset=None):
    X, fe_info = build_design(df, regressors, fe_cols)
    model = sm.GLM(df[y].astype(float), X, family=sm.families.Poisson(), offset=offset)
    res = model.fit(cov_type="cluster", cov_kwds={"groups": df[clusters].astype(str).values}, maxiter=200)
    return res, X, fe_info


def fit_negbin(df, y, regressors, clusters, fe_cols=(), offset=None):
    X, fe_info = build_design(df, regressors, fe_cols)
    groups = df[clusters].astype(str).values
    try:
        model = sm.NegativeBinomial(df[y].astype(float), X, loglike_method="nb2", offset=offset)
        res = model.fit(disp=0, maxiter=500, cov_type="cluster", cov_kwds={"groups": groups})
        res.alpha_estimate = float(res.params.get("alpha", np.nan))
        return res, X, fe_info
    except Exception:
        for alpha in (1.0, 2.0, 5.0):
            try:
                model = sm.GLM(df[y].astype(float), X, family=sm.families.NegativeBinomial(alpha=alpha), offset=offset)
                res = model.fit(cov_type="cluster", cov_kwds={"groups": groups}, maxiter=300)
                res.alpha_estimate = alpha
                return res, X, fe_info
            except Exception:
                continue
        raise


def fit_logit(df, y, regressors, clusters, fe_cols=()):
    X, fe_info = build_design(df, regressors, fe_cols)
    model = sm.GLM(df[y].astype(float), X, family=sm.families.Binomial())
    res = model.fit(cov_type="cluster", cov_kwds={"groups": df[clusters].astype(str).values}, maxiter=200)
    return res, X, fe_info


def fit_ols(df, y, regressors, clusters, fe_cols=()):
    X, fe_info = build_design(df, regressors, fe_cols)
    res = sm.OLS(df[y].astype(float), X).fit(
        cov_type="cluster", cov_kwds={"groups": df[clusters].astype(str).values}
    )
    return res, X, fe_info


def tidy(res, model_name, n_clusters=None, extra=None):
    out = pd.DataFrame(
        {
            "model": model_name,
            "term": res.params.index,
            "coef": res.params.values,
            "se": res.bse.values,
            "z": res.tvalues.values,
            "p": res.pvalues.values,
            "ci_low": res.conf_int()[0].values,
            "ci_high": res.conf_int()[1].values,
        }
    )
    out["irr"] = np.exp(out["coef"])
    out["irr_low"] = np.exp(out["ci_low"])
    out["irr_high"] = np.exp(out["ci_high"])
    out["n_obs"] = int(res.nobs)
    out["n_clusters"] = n_clusters
    if extra:
        for k, v in extra.items():
            out[k] = v
    return out


def wild_cluster_bootstrap(df, y, regressors, clusters, test_terms, fe_cols=(), n_boot=None, seed=None):
    n_boot = n_boot or config.N_WILD_BOOT
    rng = np.random.default_rng(seed or config.SEED)
    X, _ = build_design(df, regressors, fe_cols)
    yv = df[y].astype(float).values
    g = df[clusters].astype(str).values
    groups = pd.unique(g)
    base = sm.OLS(yv, X.values).fit(cov_type="cluster", cov_kwds={"groups": g})
    cols = list(X.columns)
    rows = []
    for term in test_terms:
        if term not in cols:
            continue
        j = cols.index(term)
        t_obs = base.params[j] / base.bse[j]
        Xr = np.delete(X.values, j, axis=1)
        res_r = sm.OLS(yv, Xr).fit()
        resid_r = res_r.resid
        fitted_r = res_r.fittedvalues
        t_boot = np.empty(n_boot)
        for b in range(n_boot):
            w = rng.choice([-1.0, 1.0], size=len(groups))
            wmap = dict(zip(groups, w))
            wv = np.array([wmap[x] for x in g])
            yb = fitted_r + resid_r * wv
            fit_b = sm.OLS(yb, X.values).fit(cov_type="cluster", cov_kwds={"groups": g})
            t_boot[b] = fit_b.params[j] / fit_b.bse[j]
        p = float((np.abs(t_boot) >= abs(t_obs)).mean())
        rows.append({"term": term, "t_observed": t_obs, "p_wild_cluster": p, "n_boot": n_boot})
    return pd.DataFrame(rows)


def cohen_kappa(a, b):
    a = np.asarray(a).astype(int)
    b = np.asarray(b).astype(int)
    n = len(a)
    po = float((a == b).mean())
    pa1, pb1 = a.mean(), b.mean()
    pe = pa1 * pb1 + (1 - pa1) * (1 - pb1)
    kappa = (po - pe) / (1 - pe) if (1 - pe) > 0 else np.nan
    return kappa, po, pe, n


def classification_stats(pred, truth, n_boot=None, seed=None):
    n_boot = n_boot or config.N_BOOT
    rng = np.random.default_rng(seed or config.SEED)
    pred = np.asarray(pred).astype(int)
    truth = np.asarray(truth).astype(int)
    tp = int(((pred == 1) & (truth == 1)).sum())
    fp = int(((pred == 1) & (truth == 0)).sum())
    fn = int(((pred == 0) & (truth == 1)).sum())
    tn = int(((pred == 0) & (truth == 0)).sum())
    precision = tp / (tp + fp) if (tp + fp) else np.nan
    recall = tp / (tp + fn) if (tp + fn) else np.nan
    specificity = tn / (tn + fp) if (tn + fp) else np.nan
    f1 = 2 * precision * recall / (precision + recall) if precision and recall else np.nan
    accuracy = (tp + tn) / len(pred)
    kappa, po, pe, n = cohen_kappa(pred, truth)
    idx = np.arange(len(pred))
    stats_boot = {"precision": [], "recall": [], "f1": [], "kappa": [], "accuracy": []}
    for _ in range(n_boot):
        s = rng.choice(idx, size=len(idx), replace=True)
        p_, t_ = pred[s], truth[s]
        tp_ = ((p_ == 1) & (t_ == 1)).sum()
        fp_ = ((p_ == 1) & (t_ == 0)).sum()
        fn_ = ((p_ == 0) & (t_ == 1)).sum()
        tn_ = ((p_ == 0) & (t_ == 0)).sum()
        pr = tp_ / (tp_ + fp_) if (tp_ + fp_) else np.nan
        rc = tp_ / (tp_ + fn_) if (tp_ + fn_) else np.nan
        stats_boot["precision"].append(pr)
        stats_boot["recall"].append(rc)
        stats_boot["f1"].append(2 * pr * rc / (pr + rc) if pr and rc and (pr + rc) > 0 else np.nan)
        stats_boot["kappa"].append(cohen_kappa(p_, t_)[0])
        stats_boot["accuracy"].append((tp_ + tn_) / len(p_))
    out = {
        "n": len(pred),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "prevalence_human": float(truth.mean()),
        "prevalence_rule": float(pred.mean()),
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "f1": f1,
        "accuracy": accuracy,
        "kappa": kappa,
        "percent_agreement": po,
    }
    for k, v in stats_boot.items():
        arr = np.array([x for x in v if not (x is None or (isinstance(x, float) and np.isnan(x)))])
        if len(arr):
            out[f"{k}_ci_low"] = float(np.percentile(arr, 2.5))
            out[f"{k}_ci_high"] = float(np.percentile(arr, 97.5))
    return out


def rogan_gladen(observed_prevalence, sensitivity, specificity):
    denom = sensitivity + specificity - 1
    if denom <= 0 or any(pd.isna([observed_prevalence, sensitivity, specificity])):
        return np.nan
    adj = (observed_prevalence + specificity - 1) / denom
    return float(np.clip(adj, 0, 1))


def bootstrap_ci(values, func=np.mean, n_boot=None, seed=None):
    n_boot = n_boot or config.N_BOOT
    rng = np.random.default_rng(seed or config.SEED)
    values = np.asarray(values, dtype=float)
    values = values[~np.isnan(values)]
    if len(values) == 0:
        return np.nan, np.nan, np.nan
    stat = func(values)
    boots = [func(rng.choice(values, size=len(values), replace=True)) for _ in range(n_boot)]
    return float(stat), float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


class BernoulliLCA:
    def __init__(self, n_classes, n_starts=None, max_iter=500, tol=1e-6, seed=None):
        self.n_classes = n_classes
        self.n_starts = n_starts or config.LCA_N_STARTS
        self.max_iter = max_iter
        self.tol = tol
        self.seed = seed if seed is not None else config.SEED

    @staticmethod
    def _loglik_matrix(Y, M, probs):
        p = np.clip(probs, 1e-6, 1 - 1e-6)
        ll = Y @ np.log(p).T + (M - Y) @ np.log(1 - p).T
        return ll

    def _em(self, Y, M, rng):
        n, d = Y.shape
        pi = rng.dirichlet(np.ones(self.n_classes))
        probs = rng.uniform(0.15, 0.85, size=(self.n_classes, d))
        prev_ll = -np.inf
        for _ in range(self.max_iter):
            ll_mat = self._loglik_matrix(Y, M, probs) + np.log(np.clip(pi, 1e-12, None))
            mx = ll_mat.max(axis=1, keepdims=True)
            w = np.exp(ll_mat - mx)
            denom = w.sum(axis=1, keepdims=True)
            resp = w / denom
            ll = float((np.log(denom) + mx).sum())
            pi = resp.mean(axis=0)
            num = resp.T @ Y
            den = resp.T @ M
            probs = np.clip(num / np.clip(den, 1e-9, None), 1e-6, 1 - 1e-6)
            if abs(ll - prev_ll) < self.tol:
                break
            prev_ll = ll
        return pi, probs, resp, ll

    def fit(self, data):
        Y = np.nan_to_num(np.asarray(data, dtype=float), nan=0.0)
        M = (~np.isnan(np.asarray(data, dtype=float))).astype(float)
        best = None
        for s in range(self.n_starts):
            rng = np.random.default_rng(self.seed + s)
            pi, probs, resp, ll = self._em(Y, M, rng)
            if best is None or ll > best[3]:
                best = (pi, probs, resp, ll)
        self.pi_, self.probs_, self.resp_, self.loglik_ = best
        n, d = Y.shape
        self.n_ = n
        self.n_params_ = (self.n_classes - 1) + self.n_classes * d
        self.aic_ = -2 * self.loglik_ + 2 * self.n_params_
        self.bic_ = -2 * self.loglik_ + self.n_params_ * np.log(n)
        n_star = (n + 2) / 24
        self.abic_ = -2 * self.loglik_ + self.n_params_ * np.log(n_star)
        r = np.clip(self.resp_, 1e-12, 1)
        self.entropy_ = 1 - (-(r * np.log(r)).sum()) / (n * np.log(self.n_classes)) if self.n_classes > 1 else 1.0
        self.labels_ = self.resp_.argmax(axis=1)
        self.class_sizes_ = np.bincount(self.labels_, minlength=self.n_classes)
        return self

    def simulate(self, n, rng):
        classes = rng.choice(self.n_classes, size=n, p=self.pi_)
        probs = self.probs_[classes]
        return (rng.uniform(size=probs.shape) < probs).astype(float)


def blrt(data, k, n_boot=None, seed=None, n_starts=5):
    n_boot = n_boot or config.LCA_BLRT_B
    rng = np.random.default_rng(seed or config.SEED)
    m0 = BernoulliLCA(k - 1, n_starts=n_starts, seed=config.SEED).fit(data)
    m1 = BernoulliLCA(k, n_starts=n_starts, seed=config.SEED).fit(data)
    obs = 2 * (m1.loglik_ - m0.loglik_)
    draws = []
    for b in range(n_boot):
        sim = m0.simulate(len(data), rng)
        a = BernoulliLCA(k - 1, n_starts=2, max_iter=150, tol=1e-5, seed=config.SEED + b).fit(sim)
        c = BernoulliLCA(k, n_starts=2, max_iter=150, tol=1e-5, seed=config.SEED + b).fit(sim)
        draws.append(2 * (c.loglik_ - a.loglik_))
    draws = np.array(draws)
    p = float((draws >= obs).mean())
    return {"k": k, "lrt_observed": float(obs), "p_blrt": p, "n_bootstrap": n_boot}


def vif_table(X):
    from statsmodels.stats.outliers_influence import variance_inflation_factor

    cols = [c for c in X.columns if c != "const"]
    vals = []
    Xv = X[cols].astype(float).values
    for i, c in enumerate(cols):
        try:
            vals.append({"term": c, "vif": float(variance_inflation_factor(Xv, i))})
        except Exception:
            vals.append({"term": c, "vif": np.nan})
    return pd.DataFrame(vals).sort_values("vif", ascending=False)


def describe_numeric(df, cols):
    rows = []
    for c in cols:
        s = pd.to_numeric(df[c], errors="coerce")
        rows.append(
            {
                "variable": c,
                "n": int(s.notna().sum()),
                "mean": s.mean(),
                "sd": s.std(),
                "min": s.min(),
                "p25": s.quantile(0.25),
                "median": s.median(),
                "p75": s.quantile(0.75),
                "max": s.max(),
                "n_missing": int(s.isna().sum()),
            }
        )
    return pd.DataFrame(rows)


def cramers_v(a, b):
    tab = pd.crosstab(a, b)
    chi2 = stats.chi2_contingency(tab, correction=False)[0]
    n = tab.values.sum()
    r, k = tab.shape
    denom = n * (min(r, k) - 1)
    return float(np.sqrt(chi2 / denom)) if denom > 0 else np.nan


def json_dump(obj, name):
    path = config.TABLES / f"{name}.json"
    path.write_text(json.dumps(obj, indent=2, default=str), encoding="utf-8")
    return path
