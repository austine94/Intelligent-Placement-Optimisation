"""Dummy enterprise graph data for the standalone attack-path demo.

The data is intentionally synthetic and small so the module can be run without
external services, credentials, scanners, or threat-intelligence feeds.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class Asset:
    """A synthetic asset in the enterprise graph."""

    asset_id: str
    name: str
    tier: str
    exposure: str
    business_value: int


@dataclass(frozen=True)
class AttackStep:
    """A directed transition that a defender wants to reason about."""

    source: str
    target: str
    technique: str
    likelihood: float
    impact: float
    mitigation: str


@dataclass(frozen=True)
class DummyEnvironment:
    """Container for assets and potential attack-path transitions."""

    assets: dict[str, Asset]
    steps: tuple[AttackStep, ...]
    entry_assets: tuple[str, ...]
    crown_jewels: tuple[str, ...]


def build_dummy_environment() -> DummyEnvironment:
    """Return a deterministic toy environment for local demonstrations."""

    assets = {
        "internet": Asset("internet", "Public internet", "external", "public", 1),
        "vpn": Asset("vpn", "Remote-access VPN", "edge", "public", 5),
        "web": Asset("web", "Customer web portal", "dmz", "public", 6),
        "jumpbox": Asset("jumpbox", "Operations jump box", "internal", "restricted", 7),
        "ci": Asset("ci", "Build server", "internal", "restricted", 8),
        "db": Asset("db", "Customer database", "restricted", "private", 10),
        "payments": Asset("payments", "Payments service", "restricted", "private", 10),
    }
    steps = (
        AttackStep("internet", "vpn", "stolen credentials", 0.55, 0.65, "enforce phishing-resistant MFA"),
        AttackStep("internet", "web", "public web exploit", 0.35, 0.70, "patch internet-facing web tier"),
        AttackStep("vpn", "jumpbox", "over-permissive network ACL", 0.60, 0.60, "restrict VPN-to-admin subnet access"),
        AttackStep("web", "ci", "leaked deployment token", 0.45, 0.80, "rotate deployment tokens and scope CI secrets"),
        AttackStep("jumpbox", "db", "standing admin session", 0.50, 0.95, "require just-in-time database administration"),
        AttackStep("jumpbox", "payments", "shared local admin", 0.40, 0.90, "remove shared admin accounts"),
        AttackStep("ci", "payments", "artifact signing bypass", 0.30, 0.95, "enforce signed build provenance"),
        AttackStep("ci", "db", "backup credential reuse", 0.25, 0.90, "isolate backup credentials"),
    )
    return DummyEnvironment(
        assets=assets,
        steps=steps,
        entry_assets=("internet",),
        crown_jewels=("db", "payments"),
    )
