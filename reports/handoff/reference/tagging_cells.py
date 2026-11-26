# Reference for task T3. Cells for notebooks/edas/eda_tagging.ipynb, after the section 6 code cell.
# They use variables already defined in that notebook (np, pd, plt, re, ROOT, fx, labels, g).

#%% md
### 6b. Cross-validation on wording the model has not seen

Section 6 splits the effects at random, and its TF-IDF weights are fitted on all the data before splitting. Here the vectoriser is fitted on the training folds only, and the split is made harder in two steps:

1. **Random folds**: the optimistic reference.
2. **Unseen exact wording**: effects with identical masked text always land in the same fold.
3. **Unseen wording family**: effects linked by cosine similarity of 0.7 or more land in the same fold.

A 1-nearest-neighbour model (copy the labels of the most similar training effect) is the baseline.

The targets are still the regex labels, so this measures how well the wording-to-verb mapping generalises, not whether the labels are true.
#%% code
# --- Cross-validation that does not let the same wording sit in both train and test
from scipy.sparse.csgraph import connected_components
from scipy.sparse import csr_matrix
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import KFold, GroupKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import f1_score

usable = [k for k in labels if fx[k].sum() >= 15]
text = fx["masked"].to_numpy()
Y = fx[usable].to_numpy(dtype=int)

def logreg():      # TF-IDF is inside the pipeline, so it is fitted on the training folds only
    return make_pipeline(TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True),
                         OneVsRestClassifier(LogisticRegression(max_iter=2000, C=10, class_weight="balanced")))
def nearest():     # baseline: copy the labels of the most similar training effect
    return make_pipeline(TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True),
                         KNeighborsClassifier(n_neighbors=1, metric="cosine"))

def cross_validate(make, splitter, groups=None):
    pred = np.zeros_like(Y); fold_f1 = []
    for tr, te in splitter.split(text, Y, groups):
        pred[te] = make().fit(text[tr], Y[tr]).predict(text[te])
        fold_f1.append(f1_score(Y[te], pred[te], average="macro", zero_division=0))
    return pred, fold_f1

def summarise(pred, fold_f1):
    return {"macro F1": f1_score(Y, pred, average="macro", zero_division=0),
            "micro F1": f1_score(Y, pred, average="micro", zero_division=0),
            "all labels right": (pred == Y).all(axis=1).mean(),
            "fold spread": f"{min(fold_f1):.2f} - {max(fold_f1):.2f}"}

# group 1: identical masked wording.  group 2: "wording families" (chains of effects with cosine >= 0.7)
same_wording = pd.factorize(fx["masked"])[0]
S = cosine_similarity(TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True).fit_transform(text))
family = connected_components(csr_matrix(S >= 0.7), directed=False)[1]
print(f"{len(fx)} effects | {same_wording.max() + 1} distinct wordings | {family.max() + 1} wording families "
      f"(largest holds {np.bincount(family).max()} effects)")

settings = {
    "1. random folds (optimistic)":          (KFold(5, shuffle=True, random_state=0), None),
    "2. unseen exact wording":               (GroupKFold(5), same_wording),
    "3. unseen wording family (strictest)":  (GroupKFold(5), family),
}
rows, preds = {}, {}
for name, (splitter, groups) in settings.items():
    for model_name, make in (("logistic regression", logreg), ("1-nearest neighbour", nearest)):
        pred, folds = cross_validate(make, splitter, groups)
        rows[(name, model_name)] = summarise(pred, folds); preds[(name, model_name)] = pred
cv_table = pd.DataFrame(rows).T
print(cv_table.to_string(float_format=lambda v: f"{v:.2f}"))

per_label = pd.DataFrame({name.split(". ")[1]: f1_score(Y, preds[(name, "logistic regression")], average=None, zero_division=0)
                          for name in settings}, index=usable)
per_label["examples"] = Y.sum(axis=0)
per_label = per_label.sort_values("unseen wording family (strictest)")
ax = per_label.drop(columns="examples").plot.barh(figsize=(9, 7), width=0.8)
ax.set_xlim(0, 1.05); ax.set_xlabel("F1"); ax.set_title("Per verb: how much of the score survives on unseen wording")
ax.legend(loc="upper center", bbox_to_anchor=(0.4, -0.09), ncol=3, fontsize=8); plt.tight_layout(); plt.show()
print(per_label.to_string(float_format=lambda v: f"{v:.2f}"))

#%% md
### 6c. Against the hand-written labels

The classifier is trained on pattern labels with every gold wording removed, then scored on the hand-written effects. The regex score is shown for reference only: the patterns were tuned on these effects, so it is not an independent test.
#%% code
# --- Against the HAND-WRITTEN labels: train on pattern labels, test on the gold effects
gold_ids = g.index                                                   # gold effects that have text
is_gold = fx["move_id"].isin(gold_ids).to_numpy()
gold_wordings = set(fx.loc[is_gold, "masked"])
train = ~fx["masked"].isin(gold_wordings).to_numpy()                 # no gold wording in training
gold_text = fx.set_index("move_id").loc[gold_ids, "masked"].to_numpy()
Y_gold = g[usable].to_numpy(dtype=int)

model = logreg().fit(text[train], Y[train])
P_gold = model.predict(gold_text)
R_gold = fx.set_index("move_id").loc[gold_ids, usable].to_numpy(dtype=int)   # what the patterns say

def line(name, pred):
    print(f"{name:34s} macro F1 {f1_score(Y_gold, pred, average='macro', zero_division=0):.2f} | "
          f"micro F1 {f1_score(Y_gold, pred, average='micro', zero_division=0):.2f} | "
          f"all labels right {(pred == Y_gold).all(axis=1).sum()} of {len(Y_gold)}")
print(f"trained on {train.sum()} effects, tested on {len(gold_ids)} hand-written ones")
line("classifier (never saw the wording)", P_gold)
line("regex patterns (tuned on these)", R_gold)

wrong = [(i, [k for j, k in enumerate(usable) if Y_gold[n, j] and not P_gold[n, j]],
             [k for j, k in enumerate(usable) if P_gold[n, j] and not Y_gold[n, j]])
         for n, i in enumerate(gold_ids) if (P_gold[n] != Y_gold[n]).any()]
for i, missed, extra in wrong:
    print(f"  {i}: missed {missed} extra {extra} | {fx.set_index('move_id').loc[i, 'effect_text'][:80]}")
not_scored = [k for k in labels if k not in usable]
print("verbs not scored (fewer than 15 training examples):", not_scored,
      "| gold effects using them:", int(g[not_scored].any(axis=1).sum()))

#%% md
### 6d. Hand-labelled holdout

100 effects sampled with a fixed seed from wordings outside the gold set: 60 that the patterns label and 40 that they leave empty, one effect per masked wording. They were labelled by hand without seeing the pattern output.

The two groups are reported separately because unlabelled effects are over-represented in the sample (40% against about 18% of all effects). Once scored, the patterns must not be tuned on these rows, or they stop being a holdout.
#%% code
# --- Holdout: 100 effects labelled by hand, blind to the patterns' output
path = ROOT / "configs" / "tag_holdout.csv"
sep = ";" if ";" in path.open(encoding="utf-8-sig").readline() else ","      # some spreadsheet locales save with ;
h = pd.read_csv(path, sep=sep, dtype=str, encoding="utf-8-sig").fillna("")
h = h[h["verbs"].str.strip() != ""].set_index("move_id")                      # blank = not labelled yet
print(f"{len(h)} of 100 rows labelled")
assert len(h), "nothing labelled yet: fill the verbs column first"

typed = h["verbs"].str.lower().map(lambda s: set(re.split(r"[\s,;+]+", s.strip())) - {""})
unknown = sorted(set().union(*typed) - set(labels) - {"none"})
assert not unknown, f"not verb names (typo?): {unknown}"
truth = pd.DataFrame([{k: k in s for k in labels} for s in typed], index=h.index)

rows_fx = fx.set_index("move_id").loc[h.index]
train = ~fx["masked"].isin(set(rows_fx["masked"])).to_numpy()                 # no holdout wording in training
model = logreg().fit(fx.loc[train, "masked"].to_numpy(), fx.loc[train, usable].to_numpy(dtype=int))
by_regex = rows_fx[labels].astype(bool)
by_clf = pd.DataFrame(model.predict(rows_fx["masked"].to_numpy()).astype(bool), index=h.index, columns=usable)
by_clf = by_clf.reindex(columns=labels, fill_value=False)

def scores(pred, rows):
    t, p = truth.loc[rows].to_numpy(), pred.loc[rows].to_numpy()
    tp, fp, fn = (t & p).sum(), (~t & p).sum(), (t & ~p).sum()
    return {"effects": len(rows), "precision": tp / (tp + fp) if tp + fp else np.nan,
            "recall": tp / (tp + fn) if tp + fn else np.nan, "all verbs right": (t == p).all(axis=1).mean(),
            "tp": int(tp), "fp": int(fp), "fn": int(fn)}

strata = {"patterns found a verb": h.index[by_regex.any(axis=1)], "patterns found nothing": h.index[~by_regex.any(axis=1)]}
report = pd.DataFrame({(s, m): scores(pred, rows) for s, rows in strata.items() if len(rows)
                       for m, pred in (("regex", by_regex), ("classifier", by_clf))}).T
print(report.to_string(float_format=lambda v: f"{v:.2f}"))

missed = (truth & ~by_regex).sum(); extra = (~truth & by_regex).sum()
print("\nwhere the patterns go wrong, per verb:")
print(pd.DataFrame({"missed": missed, "extra": extra}).query("missed + extra > 0").sort_values("missed", ascending=False).to_string())
