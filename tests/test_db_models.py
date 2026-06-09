from stomserver.db.models import Account, ApiToken, Job, Study
from stomserver.db.session import create_all, make_engine, make_session_factory


def _session():
    engine = make_engine("sqlite://")  # in-memory
    create_all(engine)
    return make_session_factory(engine)()


def test_create_and_query_account_token():
    db = _session()
    acct = Account(name="Clinic A")
    db.add(acct)
    db.flush()
    db.add(ApiToken(token_hash="abc", account_id=acct.id))
    db.commit()

    found = db.query(ApiToken).filter_by(token_hash="abc").one()
    assert found.account_id == acct.id


def test_study_and_job_defaults():
    db = _session()
    acct = Account(name="A")
    db.add(acct)
    db.flush()
    study = Study(account_id=acct.id, original_filename="s.nii.gz",
                  storage_key="A/studies/1/volume.nii.gz",
                  shape="[8, 16, 16]", spacing="[0.3, 0.3, 0.3]")
    db.add(study)
    db.flush()
    job = Job(study_id=study.id, account_id=acct.id, model_name="dentalsegmentator")
    db.add(job)
    db.commit()

    assert job.status == "queued"
    assert job.error is None
    assert job.mask_storage_key is None
    assert job.created_at is not None
