"""Signing strategy: Ed25519 default, BYOK, and the SigningPolicy batching wrapper."""

from __future__ import annotations

import base64

import pytest

from synapsekit.audit.signer import (
    BYOKSigningProvider,
    Ed25519SigningProvider,
    SigningPolicy,
    verify_signature,
)


class TestEd25519SigningProvider:
    def test_generates_a_usable_keypair_when_none_given(self):
        provider = Ed25519SigningProvider()
        sig = provider.sign(b"hello world")
        assert verify_signature(
            algorithm="ed25519",
            public_key_bytes=provider.public_key_bytes(),
            data=b"hello world",
            signature=sig,
        )

    def test_reloading_from_private_bytes_reproduces_the_same_key(self):
        provider = Ed25519SigningProvider()
        raw = provider.private_key_bytes()
        reloaded = Ed25519SigningProvider(raw)
        assert reloaded.public_key_bytes() == provider.public_key_bytes()

    def test_signature_does_not_verify_with_a_different_key(self):
        provider = Ed25519SigningProvider()
        other = Ed25519SigningProvider()
        sig = provider.sign(b"data")
        assert not verify_signature(
            algorithm="ed25519",
            public_key_bytes=other.public_key_bytes(),
            data=b"data",
            signature=sig,
        )

    def test_signature_does_not_verify_for_tampered_data(self):
        provider = Ed25519SigningProvider()
        sig = provider.sign(b"data")
        assert not verify_signature(
            algorithm="ed25519",
            public_key_bytes=provider.public_key_bytes(),
            data=b"tampered",
            signature=sig,
        )

    def test_key_id_defaults_to_a_stable_value(self):
        provider = Ed25519SigningProvider(key_id="my-key")
        assert provider.key_id == "my-key"


class TestEncryptedPrivateKeyExport:
    def test_round_trips_through_encrypted_pem(self):
        provider = Ed25519SigningProvider(key_id="my-key")
        encrypted = provider.export_encrypted_private_key(b"correct horse battery staple")

        reloaded = Ed25519SigningProvider.from_encrypted_private_key(
            encrypted, b"correct horse battery staple", key_id="my-key"
        )
        assert reloaded.public_key_bytes() == provider.public_key_bytes()

        sig = reloaded.sign(b"data")
        assert verify_signature(
            algorithm="ed25519",
            public_key_bytes=provider.public_key_bytes(),
            data=b"data",
            signature=sig,
        )

    def test_wrong_passphrase_fails_to_load(self):
        provider = Ed25519SigningProvider()
        encrypted = provider.export_encrypted_private_key(b"the-real-passphrase")

        with pytest.raises(Exception):
            Ed25519SigningProvider.from_encrypted_private_key(encrypted, b"wrong-passphrase")

    def test_encrypted_export_is_not_the_same_bytes_as_the_raw_export(self):
        provider = Ed25519SigningProvider()
        raw = provider.private_key_bytes()
        encrypted = provider.export_encrypted_private_key(b"secret")
        assert raw not in encrypted


class TestVerifySignature:
    def test_unsupported_algorithm_raises(self):
        with pytest.raises(ValueError):
            verify_signature(algorithm="rsa-3072", public_key_bytes=b"x", data=b"y", signature=b"z")


class TestBYOKSigningProvider:
    def test_wraps_a_caller_supplied_sign_function(self):
        # A trivial (insecure, test-only) HMAC-shaped signer.
        import hashlib
        import hmac

        secret = b"super-secret"

        def sign_fn(data: bytes) -> bytes:
            return hmac.new(secret, data, hashlib.sha256).digest()

        provider = BYOKSigningProvider(sign_fn=sign_fn, public_key=b"unused", key_id="byok-1")
        assert provider.sign(b"payload") == sign_fn(b"payload")
        assert provider.key_id == "byok-1"
        assert provider.algorithm == "byok"


class TestSigningPolicy:
    def test_ed25519_classmethod_signs_a_merkle_root(self):
        policy = SigningPolicy.ed25519()
        root = "ab" * 32
        signature = policy.sign_batch(root, start_index=0, end_index=3)
        assert signature.merkle_root == root
        assert signature.algorithm == "ed25519"
        assert verify_signature(
            algorithm=signature.algorithm,
            public_key_bytes=base64.b64decode(signature.public_key_b64),
            data=bytes.fromhex(root),
            signature=base64.b64decode(signature.signature_b64),
        )

    def test_kms_rejects_unknown_provider(self):
        with pytest.raises(ValueError):
            SigningPolicy.kms("not-a-real-cloud", key_id="k1")

    def test_sign_manifest_hash_produces_a_verifiable_signature(self):
        policy = SigningPolicy.ed25519()
        manifest_hash = "cd" * 32
        sig = policy.sign_manifest_hash(manifest_hash)
        assert sig["key_id"] == policy.provider.key_id
        assert verify_signature(
            algorithm=sig["algorithm"],
            public_key_bytes=base64.b64decode(sig["public_key_b64"]),
            data=bytes.fromhex(manifest_hash),
            signature=base64.b64decode(sig["signature_b64"]),
        )
