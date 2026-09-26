import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline

import config
import utils

FLAGS = [
    ("flag_state_changing", "human_state_changing"),
    ("flag_filesystem", "human_filesystem"),
    ("flag_shell_exec", "human_shell_exec"),
    ("flag_outbound_network", "human_outbound_network"),
    ("flag_credential_handling", "human_credential_handling"),
    ("flag_model_directed_language", "human_model_directed_language"),
]

MIN_POSITIVES = 25
MIN_F1_GAIN = 0.05
C_GRID = [0.25, 1.0, 4.0, 16.0]


def build_text(df):
    return (
        df["tool_name"].fillna("").astype(str)
        + " || "
        + df.get("tool_title", pd.Series("", index=df.index)).fillna("").astype(str)
        + " || "
        + df["description"].fillna("").astype(str)
        + " || "
        + df.get("input_schema_json", pd.Series("", index=df.index)).fillna("").astype(str)[:0].reindex(df.index).fillna("")
    )


def make_pipeline(C):
    return Pipeline(
        [
            ("tfidf", TfidfVectorizer(lowercase=True, ngram_range=(1, 2), min_df=1, sublinear_tf=True, token_pattern=r"[A-Za-z_][A-Za-z_0-9]+")),
            ("clf", LogisticRegression(C=C, class_weight="balanced", max_iter=2000, solver="liblinear")),
        ]
    )


def oof_scores(text, y, C, seed):
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    proba = np.zeros(len(y))
    for tr, te in skf.split(text, y):
        pipe = make_pipeline(C)
        pipe.fit(text[tr], y[tr])
        proba[te] = pipe.predict_proba(text[te])[:, 1]
    return proba


def best_threshold(proba, y):
    best = (0.5, -1)
    for t in np.linspace(0.05, 0.95, 91):
        pred = (proba >= t).astype(int)
        tp = ((pred == 1) & (y == 1)).sum()
        fp = ((pred == 1) & (y == 0)).sum()
        fn = ((pred == 0) & (y == 1)).sum()
        if tp == 0:
            continue
        pr = tp / (tp + fp)
        rc = tp / (tp + fn)
        f1 = 2 * pr * rc / (pr + rc)
        if f1 > best[1]:
            best = (float(t), float(f1))
    return best


def main():
    utils.log("=== 03_reclassify ===")
    val = utils.read_csv("validation_sample_filled.csv")
    tools = utils.read_derived("tool_base.csv")

    val_text = (
        val["tool_name"].fillna("").astype(str)
        + " || "
        + val["description"].fillna("").astype(str)
        + " || "
        + val["input_schema_json"].fillna("").astype(str)
    ).values

    report = []
    adopted = {}
    for rule_col, human_col in FLAGS:
        y = utils.to_bool_strict(val[human_col]).astype(int).values
        rule_pred = utils.to_bool_strict(val[rule_col]).astype(int).values
        rule_stats = utils.classification_stats(rule_pred, y, n_boot=500)
        n_pos = int(y.sum())
        row = {
            "flag": rule_col,
            "n_labelled": len(y),
            "n_positive_human": n_pos,
            "rule_precision": rule_stats["precision"],
            "rule_recall": rule_stats["recall"],
            "rule_f1": rule_stats["f1"],
            "rule_kappa": rule_stats["kappa"],
        }
        if n_pos < MIN_POSITIVES or n_pos == len(y):
            row.update(
                {
                    "model_trained": False,
                    "decision": "keep_rule_insufficient_labels",
                    "model_precision": np.nan,
                    "model_recall": np.nan,
                    "model_f1": np.nan,
                    "model_kappa": np.nan,
                    "chosen_C": np.nan,
                    "chosen_threshold": np.nan,
                }
            )
            report.append(row)
            adopted[rule_col] = None
            continue

        best = None
        for C in C_GRID:
            proba = oof_scores(val_text, y, C, config.SEED)
            thr, f1 = best_threshold(proba, y)
            if best is None or f1 > best[2]:
                best = (C, thr, f1, proba)
        C, thr, f1, proba = best
        model_pred = (proba >= thr).astype(int)
        model_stats = utils.classification_stats(model_pred, y, n_boot=500)
        use_model = (model_stats["f1"] - (rule_stats["f1"] if not pd.isna(rule_stats["f1"]) else 0)) >= MIN_F1_GAIN
        row.update(
            {
                "model_trained": True,
                "model_precision": model_stats["precision"],
                "model_recall": model_stats["recall"],
                "model_f1": model_stats["f1"],
                "model_kappa": model_stats["kappa"],
                "chosen_C": C,
                "chosen_threshold": thr,
                "decision": "use_model" if use_model else "keep_rule",
            }
        )
        report.append(row)
        adopted[rule_col] = (C, thr) if use_model else None

    rep = pd.DataFrame(report)
    utils.save_table(rep, "T04_classifier_selection")
    utils.log(rep[["flag", "n_positive_human", "rule_f1", "model_f1", "decision"]].to_string(index=False))

    text_all = tools["text_all"].fillna("").astype(str).values
    final = tools[["tool_uid", "server_id"]].copy()
    accuracy_rows = []
    for rule_col, human_col in FLAGS:
        y = utils.to_bool_strict(val[human_col]).astype(int).values
        if adopted.get(rule_col):
            C, thr = adopted[rule_col]
            pipe = make_pipeline(C)
            pipe.fit(val_text, y)
            proba_full = pipe.predict_proba(text_all)[:, 1]
            final[rule_col + "_final"] = (proba_full >= thr).astype(int)
            final[rule_col + "_score"] = proba_full
            src = "supervised"
            stats_used = rep.loc[rep["flag"] == rule_col, ["model_precision", "model_recall", "model_f1", "model_kappa"]].iloc[0]
        else:
            final[rule_col + "_final"] = utils.to_bool_strict(tools[rule_col]).astype(int).values
            final[rule_col + "_score"] = np.nan
            src = "rule"
            stats_used = rep.loc[rep["flag"] == rule_col, ["rule_precision", "rule_recall", "rule_f1", "rule_kappa"]].iloc[0]
        accuracy_rows.append(
            {
                "flag": rule_col,
                "measure_source": src,
                "precision": float(stats_used.iloc[0]) if pd.notna(stats_used.iloc[0]) else np.nan,
                "recall": float(stats_used.iloc[1]) if pd.notna(stats_used.iloc[1]) else np.nan,
                "f1": float(stats_used.iloc[2]) if pd.notna(stats_used.iloc[2]) else np.nan,
                "kappa": float(stats_used.iloc[3]) if pd.notna(stats_used.iloc[3]) else np.nan,
                "prevalence_final": float(final[rule_col + "_final"].mean()),
                "prevalence_rule": float(utils.to_bool_strict(tools[rule_col]).mean()),
            }
        )

    acc = pd.DataFrame(accuracy_rows)
    spec_map = {}
    for rule_col, human_col in FLAGS:
        y = utils.to_bool_strict(val[human_col]).astype(int).values
        if adopted.get(rule_col):
            C, thr = adopted[rule_col]
            proba = oof_scores(val_text, y, C, config.SEED)
            pred = (proba >= thr).astype(int)
        else:
            pred = utils.to_bool_strict(val[rule_col]).astype(int).values
        s = utils.classification_stats(pred, y, n_boot=500)
        spec_map[rule_col] = (s["recall"], s["specificity"])
    acc["specificity"] = acc["flag"].map(lambda f: spec_map[f][1])
    acc["adjusted_prevalence"] = [
        utils.rogan_gladen(r["prevalence_final"], spec_map[r["flag"]][0], spec_map[r["flag"]][1]) for _, r in acc.iterrows()
    ]
    utils.save_table(acc, "T05_final_measure_accuracy")
    utils.log(acc.to_string(index=False))

    utils.save_derived(final, "tool_flags_final.csv")


if __name__ == "__main__":
    main()
