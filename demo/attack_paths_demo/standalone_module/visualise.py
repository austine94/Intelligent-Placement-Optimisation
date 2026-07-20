"""SVG visualisation helpers for the standalone attack-path demo."""

from pathlib import Path
from xml.sax.saxutils import escape

from .dummy_data import DummyEnvironment
from .planning_stub import PlannedPath

NODE_POSITIONS = {
    "internet": (80, 190),
    "vpn": (250, 100),
    "web": (250, 280),
    "jumpbox": (450, 100),
    "ci": (450, 280),
    "db": (650, 100),
    "payments": (650, 280),
}

TIER_COLOURS = {
    "external": "#475569",
    "edge": "#2563eb",
    "dmz": "#0891b2",
    "internal": "#7c3aed",
    "restricted": "#dc2626",
}


def _top_path_edges(paths: list[PlannedPath]) -> set[tuple[str, str]]:
    if not paths:
        return set()
    top_assets = paths[0].assets
    return set(zip(top_assets, top_assets[1:]))


def render_attack_path_svg(
    environment: DummyEnvironment,
    paths: list[PlannedPath],
    output_path: Path | str,
) -> Path:
    """Render a standalone SVG showing the dummy graph and highest-risk path."""

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    highlighted_edges = _top_path_edges(paths)
    top_score = "n/a" if not paths else f"{paths[0].risk_score:.4f}"

    lines = [
        '<svg xmlns="http://www.w3.org/2000/svg" width="760" height="430" viewBox="0 0 760 430">',
        "<defs>",
        '<marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">',
        '<path d="M 0 0 L 10 5 L 0 10 z" fill="#64748b"/>',
        "</marker>",
        '<marker id="arrowHot" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">',
        '<path d="M 0 0 L 10 5 L 0 10 z" fill="#f97316"/>',
        "</marker>",
        "</defs>",
        '<rect width="760" height="430" rx="18" fill="#f8fafc"/>',
        '<text x="30" y="38" font-family="Arial, sans-serif" font-size="22" font-weight="700" fill="#0f172a">Standalone attack-path visualisation</text>',
        f'<text x="30" y="64" font-family="Arial, sans-serif" font-size="13" fill="#475569">Highest-risk path highlighted in orange; top score: {top_score}</text>',
    ]

    for step in environment.steps:
        x1, y1 = NODE_POSITIONS[step.source]
        x2, y2 = NODE_POSITIONS[step.target]
        is_highlighted = (step.source, step.target) in highlighted_edges
        colour = "#f97316" if is_highlighted else "#94a3b8"
        width = "4" if is_highlighted else "2"
        marker = "arrowHot" if is_highlighted else "arrow"
        label_x = (x1 + x2) / 2
        label_y = (y1 + y2) / 2 - 8
        lines.append(
            f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{colour}" stroke-width="{width}" marker-end="url(#{marker})"/>'
        )
        lines.append(
            f'<text x="{label_x:.0f}" y="{label_y:.0f}" text-anchor="middle" font-family="Arial, sans-serif" font-size="10" fill="#334155">{escape(step.technique)}</text>'
        )

    for asset_id, asset in environment.assets.items():
        x, y = NODE_POSITIONS[asset_id]
        fill = TIER_COLOURS.get(asset.tier, "#64748b")
        stroke = "#f97316" if any(asset_id in path.assets for path in paths[:1]) else "#ffffff"
        lines.extend(
            [
                f'<circle cx="{x}" cy="{y}" r="36" fill="{fill}" stroke="{stroke}" stroke-width="4"/>',
                f'<text x="{x}" y="{y - 4}" text-anchor="middle" font-family="Arial, sans-serif" font-size="11" font-weight="700" fill="#ffffff">{escape(asset.name)}</text>',
                f'<text x="{x}" y="{y + 12}" text-anchor="middle" font-family="Arial, sans-serif" font-size="10" fill="#e2e8f0">value {asset.business_value}</text>',
            ]
        )

    if paths:
        mitigations = paths[0].mitigations
        lines.append('<rect x="30" y="335" width="700" height="70" rx="12" fill="#ffffff" stroke="#cbd5e1"/>')
        lines.append('<text x="48" y="360" font-family="Arial, sans-serif" font-size="14" font-weight="700" fill="#0f172a">Top defensive actions</text>')
        for idx, mitigation in enumerate(mitigations[:3], start=1):
            lines.append(
                f'<text x="48" y="{360 + idx * 16}" font-family="Arial, sans-serif" font-size="12" fill="#334155">{idx}. {escape(mitigation)}</text>'
            )

    lines.append("</svg>")
    output.write_text("\n".join(lines) + "\n")
    return output
