"""Fit and persist per-bucket temperatures for Laya logits.

Ports the notebook LBFGS fitter (NLL on softmax(z/T)) and groups records by
`temp_bucket(qtype, k)` so the runtime's `temperature_by_options` lookup has a writer.
Fitting is CPU-side: records are `(qtype, logits, target, k)` and do not load Hub weights.

Two sample floors, on purpose:

- `MIN_BUCKET_N` is the per-bucket floor. Buckets with fewer examples are omitted from
  `temperature_by_options`; the type-level scalar covers them at lookup time.
- `MIN_TYPE_N` is the lower floor used only for that type-level scalar. It stays at the
  previous floor of 10 so a realistic dataset that fills no single bucket still gets a
  temperature instead of remaining at 1.0. Do not raise it to `MIN_BUCKET_N`.

Fitted temperatures go through `clamp_temperature` and therefore `[TEMP_MIN, TEMP_MAX]`
from `laya.common`, the same guard checkpoint load uses. There is no second pair of
bounds in this module.
"""
from numbers import Integral
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch

from .calibration import (  # noqa: F401 -- compatibility reexports
    CALIBRATION_VERSION,
    N_QTYPES,
    _CONFIG_IDENTITY_SKIP,
    apply_binning_map,
    _config_identity,
    calibration_payload,
    _warn_if_identity_mismatch,
    _rejected_temperatures,
    _install_temperatures,
    apply_calibration_payload,
)

from .common import (
    TEMP_MAX,
    TEMP_MIN,
    clamp_temperature,
    collate_items,
    ece_score,
    temp_bucket,
)

# Per-bucket floor. Below this, `temperature_by_options` omits the bucket.
MIN_BUCKET_N = 2000
# Type-level floor only. Lower than MIN_BUCKET_N; see the module docstring.
MIN_TYPE_N = 10
# Fraction of each bucket held out when `compute_ece=True`. Unused otherwise.
ECE_HOLDOUT_FRAC = 0.2
# Log-space search margin, in e-folds, outside the `[TEMP_MIN, TEMP_MAX]` answer band.
# The optimiser runs on log T with no bound of its own; any scale this far below
# `TEMP_MIN` already clamps to `TEMP_MIN`, so the band costs no real minimum, while an
# unbounded walk reaches `exp()` underflow (~-88 in float32), divides the objective by
# zero, and hands `clamp_temperature` a NaN -- which it reports as the neutral 1.0.
LOG_T_MARGIN = 10.0
# (qtype:int, logits:1d, target:1d, k:int)
Record = Tuple[int, Any, Any, int]



def _vec(x) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    return np.asarray(x, dtype=np.float32).reshape(-1)


def _validated_pair(logits, target, k=None):
    vectors = []
    for name, value in (("logits", logits), ("target", target)):
        if isinstance(value, torch.Tensor):
            value = value.detach().cpu().numpy()
        try:
            vector = np.asarray(value)
            if vector.ndim != 1 or vector.dtype.kind not in "iuf":
                raise ValueError("%s must be a one-dimensional numeric vector" % name)
            with np.errstate(over="ignore", invalid="ignore"):
                vector = vector.astype(np.float32)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("%s must be a one-dimensional numeric vector" % name) from exc
        if not np.isfinite(vector).all():
            raise ValueError("%s must contain only finite values" % name)
        vectors.append(vector)
    logits, target = vectors
    if len(logits) != len(target):
        raise ValueError("logits and target must have the same length")
    if k is None:
        k = len(logits)
    if isinstance(k, bool) or not isinstance(k, Integral) or k < 1 or k > len(logits):
        raise ValueError("k must be an integer between 1 and the vector length")
    logits, target = logits[:k], target[:k]
    if (target < 0).any() or not np.isclose(target.sum(), 1.0, rtol=1e-5, atol=1e-6):
        raise ValueError("target must be a nonnegative probability vector summing to 1")
    return logits, target, int(k)


def _pairs_by_width(pairs: Sequence) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    """One (logits, target) matrix per distinct option count, narrowest first.

    A ragged pool -- a question type holding 2-option and 12-option decisions -- cannot be
    one matrix. Any value used to square off the short rows is a real column in the single
    softmax every row then goes through, so it competes for the probability mass the fit is
    trying to allocate; at `-1e4` it wins the row it pads once the row's own logits drop
    below it, and it overflows to `inf`/`NaN` in float32 when the scale gets small. Grouping
    keeps each row inside its own option count, which is what the objective means.
    """
    groups: Dict[int, List[Tuple[np.ndarray, np.ndarray]]] = {}
    for z, t in pairs:
        groups.setdefault(len(_vec(z)), []).append((_vec(z), _vec(t)))
    return [(torch.from_numpy(np.stack([z for z, _ in groups[w]])),
             torch.from_numpy(np.stack([t for _t, t in groups[w]]))) for w in sorted(groups)]


def fit_one_temperature(pairs: Sequence, min_n: Optional[int] = None) -> float:
    """Fit one scalar T by NLL + LBFGS on log T.

    The result is `clamp_temperature` of the optimised scale, so it lies in
    `[TEMP_MIN, TEMP_MAX]` (or is the neutral 1.0 when the value is not a number).
    The search is bounded to `LOG_T_MARGIN` e-folds outside that band, which covers any
    minimum the answer can use; past it the float32 scale drifts toward `exp()` underflow,
    where the objective divides by zero and every gradient becomes NaN, and
    `clamp_temperature` reads that NaN back as the neutral 1.0.
    Returns 1.0 when fewer than `min_n` pairs are given. `min_n` defaults to
    `MIN_BUCKET_N` (the per-bucket floor). Type-level fits pass `MIN_TYPE_N`, which
    is lower, so a dataset that fills no bucket still gets a scalar instead of
    staying at 1.0.
    """
    if min_n is None:
        min_n = MIN_BUCKET_N
    if isinstance(min_n, bool) or not isinstance(min_n, Integral) or min_n < 1:
        raise ValueError("min_n must be a positive integer")
    sel = []
    for pair in pairs:
        try:
            z, t = pair
        except (TypeError, ValueError) as exc:
            raise ValueError("pair must be (logits, target)") from exc
        z, t, _k = _validated_pair(z, t)
        sel.append((z, t))
    if len(sel) < min_n:
        return 1.0
    groups = _pairs_by_width(sel)
    n_rows = float(len(sel))
    log_lo = float(np.log(TEMP_MIN)) - LOG_T_MARGIN
    log_hi = float(np.log(TEMP_MAX)) + LOG_T_MARGIN
    with torch.enable_grad():
        log_t = torch.zeros(1, requires_grad=True)
        opt = torch.optim.LBFGS([log_t], lr=0.1, max_iter=100)

        def closure():
            opt.zero_grad()
            scale = log_t.clamp(min=log_lo, max=log_hi).exp()
            nll = sum(-(t * torch.log_softmax(z / scale, -1)).sum(-1).sum()
                      for z, t in groups)
            loss = nll / n_rows
            loss.backward()
            return loss

        opt.step(closure)
    fitted = float(log_t.detach().clamp(min=log_lo, max=log_hi).exp().item())
    return clamp_temperature(fitted, TEMP_MIN, TEMP_MAX)


def _iter_records(records: Iterable) -> List[Tuple[int, np.ndarray, np.ndarray, int]]:
    out = []
    for rec in records:
        if not isinstance(rec, (list, tuple)) or len(rec) not in (3, 4):
            raise ValueError("record must be (qtype, logits, target[, k])")
        if len(rec) == 4:
            qtype, logits, target, k = rec
            if k is None:
                raise ValueError("k must be an integer between 1 and the vector length")
        else:
            qtype, logits, target = rec
            k = None
        if isinstance(qtype, bool) or not isinstance(qtype, Integral) or not 0 <= qtype < N_QTYPES:
            raise ValueError("qtype must be an integer between 0 and %d" % (N_QTYPES - 1))
        logits, target, k = _validated_pair(logits, target, k)
        out.append((int(qtype), logits, target, k))
    return out


def _softmax(z, t_scale: float) -> np.ndarray:
    z = np.asarray(z, dtype=np.float64) / max(1e-3, float(t_scale))
    z = z - z.max()
    p = np.exp(z)
    return p / p.sum()


def _ece_report(recs, temperature, temperature_by_options) -> Dict[str, Any]:
    conf_b, cor_b, conf_a, cor_a = [], [], [], []
    for qt, z, t, k in recs:
        y = int(np.argmax(t[:k]))
        p0 = _softmax(z[:k], 1.0)
        conf_b.append(float(p0.max()))
        cor_b.append(float(int(p0.argmax()) == y))
        t_scale = temperature_by_options.get(temp_bucket(qt, k), temperature[qt])
        p1 = _softmax(z[:k], t_scale)
        conf_a.append(float(p1.max()))
        cor_a.append(float(int(p1.argmax()) == y))
    return {
        "ece_before": ece_score(np.asarray(conf_b), np.asarray(cor_b)),
        "ece_after": ece_score(np.asarray(conf_a), np.asarray(cor_a)),
        "n": float(len(recs)),
    }


def _group_records(recs):
    by_type = {qt: [] for qt in range(N_QTYPES)}
    by_bucket = {}
    for qt, z, t, k in recs:
        if qt in by_type:
            by_type[qt].append((z, t))
        key = temp_bucket(qt, k)
        by_bucket.setdefault(key, []).append((z, t))
    return by_type, by_bucket


def _ece_split(recs, seed):
    """Split records into fit and eval, stratified by `temp_bucket`.

    Returns `(fit_recs, eval_recs, excluded_buckets)`. The split is a deterministic
    partition (`numpy.random.RandomState(seed)`): every record is in exactly one side.
    A bucket whose fit side would fall below `MIN_BUCKET_N` is not split; all of its
    records stay in `fit_recs` and its key is listed in `excluded_buckets`.
    """
    by_bucket = {}
    for i, rec in enumerate(recs):
        qt, _z, _t, k = rec
        by_bucket.setdefault(temp_bucket(qt, k), []).append(i)

    rng = np.random.RandomState(seed)
    fit_idx = []
    eval_idx = []
    excluded = []
    for key in sorted(by_bucket):
        idxs = by_bucket[key]
        n = len(idxs)
        n_eval = int(round(n * ECE_HOLDOUT_FRAC))
        if n - n_eval < MIN_BUCKET_N:
            fit_idx.extend(idxs)
            excluded.append(key)
            continue
        perm = rng.permutation(n)
        for j in perm[:n_eval]:
            eval_idx.append(idxs[int(j)])
        for j in perm[n_eval:]:
            fit_idx.append(idxs[int(j)])

    fit_idx.sort()
    eval_idx.sort()
    fit_recs = [recs[i] for i in fit_idx]
    eval_recs = [recs[i] for i in eval_idx]
    return fit_recs, eval_recs, excluded


def _fit_groups(by_type, by_bucket):
    temperature = [1.0] * N_QTYPES
    for qt in range(N_QTYPES):
        pairs = by_type[qt]
        # MIN_TYPE_N, not MIN_BUCKET_N: the scalar has to exist when every bucket is small.
        if len(pairs) >= MIN_TYPE_N:
            temperature[qt] = fit_one_temperature(pairs, min_n=MIN_TYPE_N)

    temperature_by_options = {}
    for key, pairs in by_bucket.items():
        if len(pairs) < MIN_BUCKET_N:
            continue
        temperature_by_options[key] = fit_one_temperature(pairs)
    return temperature, temperature_by_options


def fit_temperature_map(records: Iterable, compute_ece: bool = False, seed: int = 0) -> Dict[str, Any]:
    """Fit type-level scalars and per-bucket temperatures.

    `MIN_BUCKET_N` is the per-bucket floor: smaller buckets are omitted from
    `temperature_by_options` and the type-level scalar covers them. `MIN_TYPE_N` is
    the separate, lower floor for that scalar only.

    `compute_ece=False` (the default, and the path `Agent.fit_temperatures` stores)
    fits on every record and returns no `report` key. `seed` is ignored on this path.

    `compute_ece=True` holds out `ECE_HOLDOUT_FRAC` of each bucket, stratified by
    `temp_bucket`, using `seed` so the same records always split the same way. Temperatures
    are fit on the remainder only and ECE is scored only on the held-out records.
    `report["n"]` is the number of records passed in; `report["n_eval"]` is the held-out
    count the ECE rests on. A bucket that would drop below `MIN_BUCKET_N` after the
    holdout is fit on all of its records, left out of the eval set, and named in
    `report["buckets_excluded_from_eval"]` instead of being dropped. `n_by_bucket`
    always counts the full input, including when the fit itself used a subset.
    """
    recs = _iter_records(records)
    all_by_type, all_buckets = _group_records(recs)
    n_by_bucket = {key: len(pairs) for key, pairs in all_buckets.items()}

    if compute_ece:
        _fit_recs, eval_recs, excluded = _ece_split(recs, seed)
        by_type, by_bucket = _group_records(_fit_recs)
    else:
        eval_recs = None
        excluded = []
        by_type, by_bucket = all_by_type, all_buckets

    temperature, temperature_by_options = _fit_groups(by_type, by_bucket)
    out: Dict[str, Any] = {
        "temperature": temperature,
        "temperature_by_options": temperature_by_options,
        "n_by_bucket": n_by_bucket,
    }
    if compute_ece:
        report = _ece_report(eval_recs, temperature, temperature_by_options)
        # `n` stays the size of the input. `n_eval` is the held-out count ECE used.
        report["n"] = float(len(recs))
        report["n_eval"] = float(len(eval_recs))
        report["buckets_excluded_from_eval"] = excluded
        out["report"] = report
    return out


fit_temperatures = fit_temperature_map


# Per-bucket floor for abstention thresholds. Lower than `MIN_BUCKET_N` (which governs temperature
# fitting): a threshold is a single order statistic of the calibrated confidences, so it stabilises
# on far fewer examples than an LBFGS temperature does. Buckets below this are omitted; the caller's
# scalar `min_confidence` (or the map's "default") covers them.
MIN_ABSTAIN_BUCKET_N = 100


def _select_abstention_threshold(pairs: Sequence[Tuple[float, int]], target_error: float,
                                 conservative: bool) -> float:
    """Smallest confidence `tau` whose accepted set (`conf >= tau`) keeps error <= `target_error`.

    `pairs` are `(confidence, correct)`. Sweeping from the most confident down maximises coverage
    at the target risk (the selective-classification / split-conformal cut). `conservative` adds one
    pseudo-error so a bucket does not clear the gate on a lucky short run. Returns 1.0 when no cut
    holds the risk -- the bucket is too unreliable to accept anything short of a reported certainty.
    """
    ordered = sorted(pairs, key=lambda cc: cc[0], reverse=True)
    levels = sorted({c for c, _ in ordered}, reverse=True)
    n = len(ordered)
    n_acc = n_err = idx = 0
    best = None
    # Evaluate the error over the WHOLE accepted set {conf >= tau} at each distinct level, not
    # incrementally within a tie: a threshold accepts every answer at its own confidence, so a tie
    # group's errors must all be counted before the level is judged (else the cut sinks into a bad
    # cohort on its first few correct members).
    for tau in levels:
        while idx < n and ordered[idx][0] >= tau:
            n_acc += 1
            n_err += 0 if ordered[idx][1] else 1
            idx += 1
        rate = (n_err + 1.0) / (n_acc + 1.0) if conservative else (n_err / n_acc)
        if n_acc > 0 and rate <= target_error:
            best = tau
    if best is None:
        return 1.0
    return float(min(1.0, max(0.0, best)))


def fit_abstention_thresholds(records: Iterable, temperature: Sequence[float],
                              temperature_by_options: Dict[str, float], *,
                              binning_map: Optional[Dict[str, Dict[str, Any]]] = None,
                              target_error: float = 0.10,
                              min_bucket_n: int = MIN_ABSTAIN_BUCKET_N,
                              conservative: bool = True) -> Dict[str, float]:
    """Fit a per-`temp_bucket` abstention threshold so a gate keeps a target error in every bucket.

    A single `min_confidence` does not transfer across option counts (#394): the calibrated
    confidence of a 2-option and a 12-option answer live on different scales, so one cut over- or
    under-abstains depending on the question. This fits one cut per bucket instead, keyed exactly
    like `temperature_by_options` (`common.temp_bucket`, e.g. ``"choice:3-5"``), and the result is a
    `min_confidence` map that :func:`laya.confidence.check_min_confidence` /
    :func:`laya.confidence.apply_confidence_gate` accept directly.

    `records` are the same `(qtype, logits, target[, k])` tuples `fit_temperature_map` consumes
    (`records_from_labeled` builds them). Confidence is the **calibrated** `max(p)` -- the logits are
    scaled by the fitted `temperature` / `temperature_by_options` first, so thresholds and the
    numbers the runtime reports are on the same scale. `target_error` is the tolerated error among
    accepted answers; `min_bucket_n` omits buckets too small to fit, and `conservative` adds a
    one-sample margin. The thresholds are empirical cuts on the calibration set, not a formal
    coverage guarantee -- validate on held-out data (`fit_temperature_map(..., compute_ece=True)`
    gives a held-out split) for a production gate.

    Pass `binning_map` when the agent that will serve these thresholds has one installed -- by
    `Agent.fit_binning`, or by a calibration payload that carries `binning_map` -- because the
    runtime recalibrates `answer_confidence` through that map before anything reads it, so a cut
    fitted without it is a cut on a scale the gate never sees. The thresholds are then on the binned
    scale, and the order the two were fitted in stops mattering. Measured on 1,200 synthetic
    12-option records at `target_error=0.10`: the cut fitted without a map holds 9.8% error over 50%
    coverage on un-binned confidences, and admits 94.5% of answers at 25.6% error once the same
    number is compared against binned ones.
    """
    if not 0.0 <= target_error <= 1.0:
        raise ValueError("target_error must be in [0.0, 1.0], got %r" % (target_error,))
    recs = _iter_records(records)
    by_bucket: Dict[str, List[Tuple[float, int]]] = {}
    for qt, z, t, k in recs:
        y = int(np.argmax(t[:k]))
        bucket = temp_bucket(qt, k)
        t_scale = temperature_by_options.get(bucket, temperature[qt])
        p = _softmax(z[:k], t_scale)
        conf = float(p.max())
        if binning_map:
            # The runtime bins before anyone reads `answer_confidence` (`Agent._decode_answers`),
            # so the cut has to be chosen on the binned scale or it gates a different quantity.
            conf = apply_binning_map(conf, bucket, binning_map)
        # `_decode_answers` reports `answer_confidence` rounded to 4 decimals, after binning, and
        # the gate compares that. A cut picked at full precision can sit above the rounded value of
        # its own cohort (bin 5/6 is reported as 0.8333 < 0.83333...), which abstains all of it.
        conf = round(conf, 4)
        by_bucket.setdefault(bucket, []).append((conf, int(int(p.argmax()) == y)))
    out: Dict[str, float] = {}
    for key, pairs in by_bucket.items():
        if len(pairs) < min_bucket_n:
            continue
        out[key] = _select_abstention_threshold(pairs, target_error, conservative)
    return out


# Per-bucket floor for histogram-binning recalibration. Higher than the abstention floor because a
# binning map splits each bucket's examples across `bins`, so each bin needs its own sample; lower
# than the temperature floor because binning is a count per bin, not an optimisation.
MIN_BINNING_BUCKET_N = 200


def fit_binning_map(records: Iterable, temperature: Sequence[float],
                    temperature_by_options: Dict[str, float], *, bins: int = 15,
                    min_bucket_n: int = MIN_BINNING_BUCKET_N) -> Dict[str, Dict[str, Any]]:
    """Fit a per-`temp_bucket` histogram-binning recalibration map for `answer_confidence`.

    Temperature scaling applies one scalar per bucket; it cannot fix a bucket whose reliability
    curve is not a simple sharpening/softening (the pathological `choice:11+` the shipped English
    checkpoint carries is one). Histogram binning is the non-parametric alternative: split the
    calibrated confidences of a bucket into `bins` equal-width bins over [0, 1], and map every
    confidence that lands in a bin to that bin's empirical accuracy. It needs no monotonicity
    assumption and no extra dependency (NumPy only; isotonic regression would pull in scikit-learn).

    `records` are the same `(qtype, logits, target[, k])` tuples `fit_temperature_map` consumes;
    confidence is the calibrated `max(p)` (logits scaled by the fitted `temperature` /
    `temperature_by_options` first), so a binning map composes on top of a temperature map rather
    than replacing it. Returns `{bucket: {"bins": N, "values": [recalibrated confidence per bin]}}`;
    buckets below `min_bucket_n` are omitted. Apply it with :func:`apply_binning_map`. An empty bin
    (a confidence range the calibration set never produced) maps to its own midpoint, i.e. leaves
    that region unchanged, so an unseen value is never recalibrated to a fabricated 0.
    """
    if bins < 1:
        raise ValueError("bins must be >= 1, got %r" % (bins,))
    recs = _iter_records(records)
    by_bucket: Dict[str, List[Tuple[float, int]]] = {}
    for qt, z, t, k in recs:
        y = int(np.argmax(t[:k]))
        bucket = temp_bucket(qt, k)
        t_scale = temperature_by_options.get(bucket, temperature[qt])
        p = _softmax(z[:k], t_scale)
        by_bucket.setdefault(bucket, []).append((float(p.max()), int(int(p.argmax()) == y)))
    out: Dict[str, Dict[str, Any]] = {}
    for key, pairs in by_bucket.items():
        if len(pairs) < min_bucket_n:
            continue
        conf = np.asarray([c for c, _ in pairs], dtype=float)
        corr = np.asarray([c for _, c in pairs], dtype=float)
        idx = np.clip((conf * bins).astype(int), 0, bins - 1)
        values = []
        for b in range(bins):
            mask = idx == b
            values.append(float(corr[mask].mean()) if mask.any() else (b + 0.5) / bins)
        out[key] = {"bins": bins, "values": values}
    return out


@torch.no_grad()
def records_from_labeled(agent, pairs: Sequence) -> List[Record]:
    """Collect CPU records from `(state, questions, targets)`.

    Logits are the raw rows `_decode_answers` scales, not the probabilities it returns.
    `_encode_state` builds one state's items, `_forward` runs them, and `_option_logits`
    slices each question to the same marker width `_decode_answers` divides by temperature.
    No Hub download and no `model.safetensors` write. Tests stay weight-free by passing
    synthetic records to `fit_temperature_map` instead of calling this.
    """
    # Imported lazily: `agent` imports this module at load, and `_option_logits` is defined
    # on the way through that import.
    from .agent import _option_logits

    records: List[Record] = []
    for state, questions, targets in pairs:
        ids = list(questions.keys())
        if not ids:
            continue
        for qid in ids:
            agent._check_question(qid, questions[qid])
        internal = {qid: agent._to_internal(questions[qid]) for qid in ids}
        items = agent._encode_state(state, ids, internal)
        batch = collate_items([items], agent.tok.pad_token_id)
        if batch is None:
            continue
        logits, _act = agent._forward(batch)
        rows = _option_logits(logits, items, 0)
        for j, qid in enumerate(ids):
            k = len(items[j]["markers"])
            qt = int(items[j]["qtype"])
            records.append((qt, _vec(rows[j]), _vec(targets[qid])[:k], k))
    return records
