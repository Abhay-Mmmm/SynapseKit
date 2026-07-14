"""Merkle tree over record hashes — batch signing signs a root, not every record.

Records are hashed individually (their ``hash`` field). At flush time we
build a Merkle tree over those leaf hashes and sign only the root,
turning O(n) signature operations into O(1) per batch while still making
every record independently provable via a Merkle proof.

Uses the RFC 6962 (Certificate Transparency) tree construction: leaves
are never duplicated to pad an odd count, and internal-node hashes are
domain-separated from leaf hashes with a leading 0x01 byte. A naive
"duplicate the last leaf" construction (as used by early Bitcoin) makes
a tree with N leaves indistinguishable from a differently-shaped tree
with N+1 leaves where the last is a copy of the previous one — the class
of bug behind CVE-2012-2459. The recursive split here never produces
that ambiguity.

:class:`MerkleHasher` itself builds a tree over already-computed hex
leaf hashes (it doesn't know or care what a "leaf" represents) — use
:func:`hash_leaf` to turn a record's own ``hash`` field into the actual
RFC 6962 leaf hash (with the 0x00 domain-separation prefix) before
handing it to :meth:`MerkleHasher.root`/:meth:`MerkleHasher.proof`. That
keeps a leaf hash from ever being replayable as an internal node hash
(0x01-prefixed) or vice versa.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

#: Domain-separates internal node hashes from leaf hashes so a node hash
#: can never be replayed as if it were itself a valid leaf.
_NODE_DOMAIN = b"\x01"

#: Domain-separates leaf hashes from internal node hashes (RFC 6962 §2.1
#: uses 0x00 for leaves and 0x01 for internal nodes) so a raw record hash
#: can never be replayed as if it were itself an internal node hash.
_LEAF_DOMAIN = b"\x00"


def hash_leaf(value: str) -> str:
    """RFC 6962 leaf hash: ``SHA256(0x00 || value)`` over a hex-encoded input.

    Call this on a record's ``hash`` field before passing it to
    :meth:`MerkleHasher.root`/:meth:`MerkleHasher.proof` — those methods
    treat their inputs as opaque, already-domain-separated leaf hashes
    and do not apply this prefix themselves.
    """
    return hashlib.sha256(_LEAF_DOMAIN + bytes.fromhex(value)).hexdigest()


def _hash_pair(left: str, right: str) -> str:
    return hashlib.sha256(_NODE_DOMAIN + bytes.fromhex(left) + bytes.fromhex(right)).hexdigest()


def _largest_power_of_two_less_than(n: int) -> int:
    k = 1
    while k * 2 < n:
        k *= 2
    return k


@dataclass(frozen=True)
class MerkleProof:
    """Sibling hashes (bottom-up) plus left/right side markers for a leaf."""

    leaf: str
    siblings: list[str]
    # True if the sibling at this level is on the right of our node.
    sibling_is_right: list[bool]


class MerkleHasher:
    """Builds Merkle roots and inclusion proofs over hex leaf hashes.

    Handles 0 (empty root) and 1 (root == leaf) leaf batches as special
    cases; everything else uses the RFC 6962 recursive split so no leaf
    is ever duplicated regardless of batch size.
    """

    EMPTY_ROOT = hashlib.sha256(b"").hexdigest()

    @classmethod
    def root(cls, leaves: list[str]) -> str:
        return cls._mth(leaves)

    @classmethod
    def _mth(cls, leaves: list[str]) -> str:
        n = len(leaves)
        if n == 0:
            return cls.EMPTY_ROOT
        if n == 1:
            return leaves[0]
        k = _largest_power_of_two_less_than(n)
        left = cls._mth(leaves[:k])
        right = cls._mth(leaves[k:])
        return _hash_pair(left, right)

    @classmethod
    def proof(cls, leaves: list[str], index: int) -> MerkleProof:
        if not leaves:
            raise ValueError("cannot build a proof for an empty tree")
        if not (0 <= index < len(leaves)):
            raise IndexError(f"index {index} out of range for {len(leaves)} leaves")
        siblings, sides = cls._path(leaves, index)
        return MerkleProof(leaf=leaves[index], siblings=siblings, sibling_is_right=sides)

    @classmethod
    def _path(cls, leaves: list[str], m: int) -> tuple[list[str], list[bool]]:
        n = len(leaves)
        if n == 1:
            return [], []
        k = _largest_power_of_two_less_than(n)
        if m < k:
            siblings, sides = cls._path(leaves[:k], m)
            return [*siblings, cls._mth(leaves[k:])], [*sides, True]
        siblings, sides = cls._path(leaves[k:], m - k)
        return [*siblings, cls._mth(leaves[:k])], [*sides, False]

    @classmethod
    def verify(cls, proof: MerkleProof, root: str) -> bool:
        node = proof.leaf
        for sibling, is_right in zip(proof.siblings, proof.sibling_is_right, strict=True):
            node = _hash_pair(node, sibling) if is_right else _hash_pair(sibling, node)
        return node == root
