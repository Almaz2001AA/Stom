from scripts.create_account import create_account
from stomserver.auth import hash_token
from stomserver.db.models import Account, ApiToken
from stomserver.db.session import create_all, make_engine, make_session_factory


def test_create_account_issues_token():
    engine = make_engine("sqlite://")
    create_all(engine)
    factory = make_session_factory(engine)

    token = create_account(factory, "Clinic X")
    assert isinstance(token, str) and len(token) > 20

    db = factory()
    acct = db.query(Account).filter_by(name="Clinic X").one()
    row = db.query(ApiToken).filter_by(token_hash=hash_token(token)).one()
    assert row.account_id == acct.id
