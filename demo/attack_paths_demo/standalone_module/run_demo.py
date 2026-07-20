"""Run the standalone attack-path planning demo."""

from pathlib import Path

from .dummy_data import build_dummy_environment
from .planning_stub import plan_attack_paths
from .visualise import render_attack_path_svg


def main() -> None:
    environment = build_dummy_environment()
    paths = plan_attack_paths(environment)

    print("Standalone attack-path planning demo")
    print("====================================")
    print(f"Assets: {len(environment.assets)}")
    print(f"Candidate transitions: {len(environment.steps)}")
    print()

    visualisation_path = render_attack_path_svg(
        environment,
        paths,
        Path(__file__).resolve().parent / "outputs" / "attack_paths.svg",
    )
    print(f"Visualisation: {visualisation_path}")
    print()

    for rank, path in enumerate(paths, start=1):
        asset_names = " -> ".join(environment.assets[asset].name for asset in path.assets)
        print(f"#{rank} risk={path.risk_score:.4f}")
        print(f"  Path: {asset_names}")
        print(f"  Techniques: {', '.join(path.techniques)}")
        print("  Defensive actions:")
        for mitigation in path.mitigations:
            print(f"   - {mitigation}")
        print()


if __name__ == "__main__":
    main()
