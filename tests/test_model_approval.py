"""FR-09: an approval decision is persisted with the threshold and metrics it approved.

`reviews approve` (src/cli/reviews.py) parses `--parameters-json` and
`--metrics-json` and hands them to `review_workflows.approve_model`; the row in
`lit_model_approvals` is the only durable record of which threshold was
accepted, on which labels and by whom. These read the row back through a fresh
query rather than trusting the returned instance.
"""

from __future__ import annotations

import hashlib
import json

import pytest
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError

from db.gold_models import ModelApproval
from transform.review_workflows import approve_model

VERSION = "c" * 64
MODEL_SHA = hashlib.sha256(b"screening-margin-v1").hexdigest()
LABELS_SHA = hashlib.sha256(b"labels-2026-11-06").hexdigest()
# The shape the CLI passes after `json.loads`, floats included, so the
# round-trip through the JSON column is exercised on real values.
PARAMETERS = json.loads('{"threshold": 0.0125, "signal": "relevance_margin"}')
METRICS = json.loads('{"recall": 0.97, "precision": 0.81, "holdout_n": 120, "kappa": 0.74}')


def _approve(session, **overrides):
    arguments = {
        "workflow": "screening",
        "dataset_version_id": VERSION,
        "model_sha256": MODEL_SHA,
        "label_set_sha256": LABELS_SHA,
        "parameters": PARAMETERS,
        "metrics": METRICS,
        "approved_by": "rene",
        "approved": True,
    } | overrides
    return approve_model(session, **arguments)


def test_the_approved_threshold_and_metrics_are_persisted(gold_session):
    _approve(gold_session)
    gold_session.expunge_all()

    row = gold_session.scalars(select(ModelApproval)).one()

    assert (row.workflow, row.dataset_version_id) == ("screening", VERSION)
    assert (row.model_sha256, row.label_set_sha256) == (MODEL_SHA, LABELS_SHA)
    assert row.parameters == PARAMETERS
    assert row.parameters["threshold"] == 0.0125
    assert row.metrics == METRICS
    assert (row.approved, row.approved_by) == (True, "rene")
    assert row.approved_at is not None


def test_a_rejection_is_recorded_too(gold_session):
    # `reviews approve --reject` writes approved=False: a refusal is evidence,
    # not the absence of a row.
    _approve(gold_session, approved=False)
    gold_session.expunge_all()

    assert gold_session.scalars(select(ModelApproval.approved)).all() == [False]


def test_a_hash_that_is_not_sha256_is_refused_before_anything_is_written(gold_session):
    with pytest.raises(ValueError, match="SHA-256"):
        _approve(gold_session, model_sha256="abc")

    assert gold_session.scalars(select(ModelApproval)).all() == []


def test_one_decision_per_model_and_version(gold_session):
    _approve(gold_session)

    with pytest.raises(IntegrityError):
        _approve(gold_session, approved_by="someone-else")
