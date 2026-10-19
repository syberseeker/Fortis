"""Release-consistency tests.

The app version lives in core/main.py (APP_VERSION) and the changelog
documents released versions. These tests keep the two from drifting.
"""
import re

import pytest
from fastapi.testclient import TestClient


def _changelog_versions() -> list:
    text = open("CHANGELOG.md", encoding="utf-8").read()
    return re.findall(r"^## \[([0-9]+\.[0-9]+\.[0-9]+)\]", text, re.MULTILINE)


def test_app_version_matches_latest_changelog_entry():
    from core.main import APP_VERSION

    versions = _changelog_versions()
    assert versions, "CHANGELOG.md must contain at least one version section"
    assert APP_VERSION == versions[0], (
        f"core/main.py APP_VERSION ({APP_VERSION}) drifted from the newest "
        f"changelog entry ({versions[0]})"
    )


def test_health_reports_app_version():
    from server.app import app as server_app
    from core.main import APP_VERSION

    with TestClient(server_app) as client:
        resp = client.get("/health")
        assert resp.status_code == 200
        assert resp.json()["version"] == APP_VERSION


def test_latest_changelog_entry_has_tag():
    """Every released version section should correspond to a git tag.

    Skipped on non-tag CI events: a main-push run for a release commit
    necessarily starts before the maintainer pushes the tag, so demanding
    the tag there would fail every release. The check still enforces the
    changelog<->tag pairing on dev machines (and on git-less sandboxes it
    skips entirely)."""
    latest = _changelog_versions()[0]
    import os
    import shutil
    import subprocess

    if shutil.which("git") is None:
        pytest.skip("git is not available in this environment")
    if os.getenv("GITHUB_ACTIONS") == "true" and not (
        os.getenv("GITHUB_REF") or ""
    ).startswith("refs/tags/"):
        pytest.skip("tag assertion is a dev-machine guard; CI main-push runs precede tagging")

    out = subprocess.run(
        ["git", "tag", "-l", f"v{latest}"], capture_output=True, text=True, check=True
    )
    assert f"v{latest}" in out.stdout.split(), (
        f"v{latest} is documented in CHANGELOG.md but not tagged"
    )


def test_readme_states_current_version():
    readme = open("README.md", encoding="utf-8").read()
    from core.main import APP_VERSION

    assert f"Current version: {APP_VERSION}" in readme
