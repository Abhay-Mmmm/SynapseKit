"""Signing strategy — Ed25519 by default, pluggable BYOK/KMS providers.

Nothing outside this module hardcodes Ed25519. Everything signs through
the :class:`SigningProvider` interface so swapping in a KMS-backed
provider later doesn't touch :mod:`trace`, :mod:`bundle`, or callers.
"""

from __future__ import annotations

import base64
from abc import ABC, abstractmethod
from collections.abc import Callable
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Protocol

from .metrics import AuditMetrics, default_metrics
from .types import Signature

if TYPE_CHECKING:
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey


class SigningProvider(ABC):
    """A signing key/backend that can sign bytes and expose its public key."""

    #: e.g. "ed25519", "aws-kms-ecdsa-p256", "byok"
    algorithm: str
    #: Stable identifier for this key, recorded in the manifest so a
    #: verifier can match a signature to the right public key across
    #: key rotations.
    key_id: str

    @abstractmethod
    def sign(self, data: bytes) -> bytes: ...

    @abstractmethod
    def public_key_bytes(self) -> bytes: ...


def verify_signature(
    *, algorithm: str, public_key_bytes: bytes, data: bytes, signature: bytes
) -> bool:
    """Verify ``signature`` over ``data`` for a given algorithm and public key.

    This is the single verification entry point used by both the
    in-process verifier and the standalone bundle verifier, so "how do
    we check a signature" is defined in exactly one place.
    """
    if algorithm == "ed25519":
        from cryptography.exceptions import InvalidSignature
        from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey

        try:
            Ed25519PublicKey.from_public_bytes(public_key_bytes).verify(signature, data)
            return True
        except InvalidSignature:
            return False
    raise ValueError(f"unsupported signing algorithm: {algorithm!r}")


class Ed25519SigningProvider(SigningProvider):
    """Default signing provider — Ed25519 via the ``cryptography`` package."""

    algorithm = "ed25519"

    def __init__(self, private_key: Any = None, *, key_id: str | None = None) -> None:
        from cryptography.hazmat.primitives.asymmetric.ed25519 import (
            Ed25519PrivateKey,
        )

        if private_key is None:
            private_key = Ed25519PrivateKey.generate()
        elif isinstance(private_key, bytes | bytearray):
            private_key = Ed25519PrivateKey.from_private_bytes(bytes(private_key))
        self._private_key: Ed25519PrivateKey = private_key
        self.key_id = key_id or self.public_key_b64()[:16]

    def sign(self, data: bytes) -> bytes:
        return self._private_key.sign(data)

    def public_key_bytes(self) -> bytes:
        from cryptography.hazmat.primitives import serialization

        return self._private_key.public_key().public_bytes(
            encoding=serialization.Encoding.Raw,
            format=serialization.PublicFormat.Raw,
        )

    def public_key_b64(self) -> str:
        return base64.b64encode(self.public_key_bytes()).decode("ascii")

    def private_key_bytes(self) -> bytes:
        """Raw, UNENCRYPTED private key bytes.

        Convenient for reusing the same key across processes in tests
        or local development, but the caller is responsible for secure
        storage — prefer :meth:`export_encrypted_private_key` for
        anything persisted outside memory in a regulated deployment.
        """
        from cryptography.hazmat.primitives import serialization

        return self._private_key.private_bytes(
            encoding=serialization.Encoding.Raw,
            format=serialization.PrivateFormat.Raw,
            encryption_algorithm=serialization.NoEncryption(),
        )

    def export_encrypted_private_key(self, passphrase: bytes) -> bytes:
        """Export the private key as passphrase-encrypted PEM (PKCS#8).

        Use this instead of :meth:`private_key_bytes` for anything
        written to disk, a secrets manager, or backup storage — the
        result is useless without ``passphrase``, unlike the raw bytes.
        """
        from cryptography.hazmat.primitives import serialization

        return self._private_key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.BestAvailableEncryption(passphrase),
        )

    @classmethod
    def from_encrypted_private_key(
        cls, data: bytes, passphrase: bytes, *, key_id: str | None = None
    ) -> Ed25519SigningProvider:
        """Load a provider from PEM bytes produced by :meth:`export_encrypted_private_key`."""
        from cryptography.hazmat.primitives import serialization

        private_key = serialization.load_pem_private_key(data, password=passphrase)
        return cls(private_key, key_id=key_id)


class SignFn(Protocol):
    def __call__(self, data: bytes) -> bytes: ...


class BYOKSigningProvider(SigningProvider):
    """Bring-your-own-key: wraps caller-supplied sign/public-key callables.

    Lets an organization plug in a signing mechanism SynapseKit doesn't
    know about (an HSM client, a custom enclave, etc.) without needing a
    dedicated subclass, as long as it can produce raw signature bytes.
    """

    def __init__(
        self,
        *,
        sign_fn: SignFn,
        public_key: bytes,
        key_id: str,
        algorithm: str = "byok",
    ) -> None:
        self._sign_fn = sign_fn
        self._public_key = public_key
        self.key_id = key_id
        self.algorithm = algorithm

    def sign(self, data: bytes) -> bytes:
        return self._sign_fn(data)

    def public_key_bytes(self) -> bytes:
        return self._public_key


class KMSSigningProvider(SigningProvider):
    """Base class for cloud KMS-backed signing providers.

    Concrete clouds (AWS/Azure/GCP) subclass this; each is responsible
    for translating ``sign``/``public_key_bytes`` into the relevant SDK
    calls. The abstraction means :class:`SigningPolicy` and everything
    downstream never needs to know which cloud is in play.
    """

    def __init__(self, *, key_id: str, algorithm: str, client: Any = None) -> None:
        self.key_id = key_id
        self.algorithm = algorithm
        self._client = client


class AWSKMSSigningProvider(KMSSigningProvider):
    """Signs via AWS KMS asymmetric keys (requires ``boto3`` and network access)."""

    def __init__(
        self, *, key_id: str, algorithm: str = "aws-kms-ecdsa-sha256", client: Any = None
    ) -> None:
        super().__init__(key_id=key_id, algorithm=algorithm)
        if client is None:
            import boto3

            client = boto3.client("kms")
        self._client = client

    def sign(self, data: bytes) -> bytes:
        response = self._client.sign(
            KeyId=self.key_id,
            Message=data,
            MessageType="RAW",
            SigningAlgorithm="ECDSA_SHA_256",
        )
        return bytes(response["Signature"])

    def public_key_bytes(self) -> bytes:
        response = self._client.get_public_key(KeyId=self.key_id)
        return bytes(response["PublicKey"])


class AzureKeyVaultSigningProvider(KMSSigningProvider):
    """Placeholder for Azure Key Vault-backed signing (future-friendly interface)."""

    def __init__(
        self, *, key_id: str, algorithm: str = "azure-kv-es256", client: Any = None
    ) -> None:
        super().__init__(key_id=key_id, algorithm=algorithm)
        self._client = client

    def sign(self, data: bytes) -> bytes:
        raise NotImplementedError(
            "Azure Key Vault signing requires 'azure-keyvault-keys'; "
            "pass a configured CryptographyClient via `client=`."
        )

    def public_key_bytes(self) -> bytes:
        raise NotImplementedError("Azure Key Vault signing is not yet wired up.")


class GCPKMSSigningProvider(KMSSigningProvider):
    """Placeholder for GCP Cloud KMS-backed signing (future-friendly interface)."""

    def __init__(
        self, *, key_id: str, algorithm: str = "gcp-kms-ec-sign-p256-sha256", client: Any = None
    ) -> None:
        super().__init__(key_id=key_id, algorithm=algorithm)
        self._client = client

    def sign(self, data: bytes) -> bytes:
        raise NotImplementedError(
            "GCP Cloud KMS signing requires 'google-cloud-kms'; "
            "pass a configured KeyManagementServiceClient via `client=`."
        )

    def public_key_bytes(self) -> bytes:
        raise NotImplementedError("GCP Cloud KMS signing is not yet wired up.")


class SigningPolicy:
    """Batch-signing policy: wraps a :class:`SigningProvider` plus batching knobs.

    Records are never signed individually — call :meth:`sign_batch` once
    per Merkle root at flush time. ``flush_interval_seconds`` and
    ``max_batch_size`` are read by :mod:`synapsekit.audit.bundle` /
    higher-level flush loops to decide when a batch closes.
    """

    def __init__(
        self,
        provider: SigningProvider,
        *,
        max_batch_size: int = 500,
        flush_interval_seconds: float = 5.0,
        metrics: AuditMetrics | None = None,
    ) -> None:
        self.provider = provider
        self.max_batch_size = max_batch_size
        self.flush_interval_seconds = flush_interval_seconds
        self._metrics = metrics if metrics is not None else default_metrics

    @classmethod
    def ed25519(
        cls,
        private_key: bytes | None = None,
        *,
        key_id: str | None = None,
        max_batch_size: int = 500,
        flush_interval_seconds: float = 5.0,
        metrics: AuditMetrics | None = None,
    ) -> SigningPolicy:
        return cls(
            Ed25519SigningProvider(private_key, key_id=key_id),
            max_batch_size=max_batch_size,
            flush_interval_seconds=flush_interval_seconds,
            metrics=metrics,
        )

    @classmethod
    def byok(
        cls,
        *,
        sign_fn: SignFn,
        public_key: bytes,
        key_id: str,
        algorithm: str = "byok",
        max_batch_size: int = 500,
        flush_interval_seconds: float = 5.0,
        metrics: AuditMetrics | None = None,
    ) -> SigningPolicy:
        return cls(
            BYOKSigningProvider(
                sign_fn=sign_fn, public_key=public_key, key_id=key_id, algorithm=algorithm
            ),
            max_batch_size=max_batch_size,
            flush_interval_seconds=flush_interval_seconds,
            metrics=metrics,
        )

    @classmethod
    def kms(
        cls,
        provider: str,
        *,
        key_id: str,
        client: Any = None,
        max_batch_size: int = 500,
        flush_interval_seconds: float = 5.0,
        metrics: AuditMetrics | None = None,
    ) -> SigningPolicy:
        providers: dict[str, Callable[..., KMSSigningProvider]] = {
            "aws": AWSKMSSigningProvider,
            "azure": AzureKeyVaultSigningProvider,
            "gcp": GCPKMSSigningProvider,
        }
        cls_ = providers.get(provider)
        if cls_ is None:
            raise ValueError(
                f"unknown KMS provider {provider!r}; expected one of {sorted(providers)}"
            )
        return cls(
            cls_(key_id=key_id, client=client),
            max_batch_size=max_batch_size,
            flush_interval_seconds=flush_interval_seconds,
            metrics=metrics,
        )

    def sign_batch(self, merkle_root: str, *, start_index: int, end_index: int) -> Signature:
        """Sign a Merkle root, producing the :class:`Signature` for one batch."""
        signature_bytes = self.provider.sign(bytes.fromhex(merkle_root))
        self._metrics.record_signed_batch(
            algorithm=self.provider.algorithm, count=end_index - start_index + 1
        )
        return Signature(
            algorithm=self.provider.algorithm,
            key_id=self.provider.key_id,
            public_key_b64=base64.b64encode(self.provider.public_key_bytes()).decode("ascii"),
            signature_b64=base64.b64encode(signature_bytes).decode("ascii"),
            merkle_root=merkle_root,
            signed_at=datetime.now(timezone.utc),
            start_index=start_index,
            end_index=end_index,
        )

    def sign_manifest_hash(self, manifest_hash: str) -> dict[str, str]:
        """Sign a bundle manifest's content hash (see :mod:`synapsekit.audit.bundle`).

        Distinct from :meth:`sign_batch`: a manifest signature isn't
        tied to a record range, so it's returned as a plain dict rather
        than the batch-shaped :class:`Signature` dataclass.
        """
        signature_bytes = self.provider.sign(bytes.fromhex(manifest_hash))
        return {
            "key_id": self.provider.key_id,
            "algorithm": self.provider.algorithm,
            "public_key_b64": base64.b64encode(self.provider.public_key_bytes()).decode("ascii"),
            "signature_b64": base64.b64encode(signature_bytes).decode("ascii"),
        }
