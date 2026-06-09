from stomserver.auth import hash_token


def test_hash_is_deterministic_and_hex64():
    h1 = hash_token("secret-token")
    h2 = hash_token("secret-token")
    assert h1 == h2
    assert len(h1) == 64
    assert all(c in "0123456789abcdef" for c in h1)


def test_different_tokens_differ():
    assert hash_token("a") != hash_token("b")
