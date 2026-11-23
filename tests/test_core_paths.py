"""Names from a server: hostile ones are errors, ones this platform cannot hold are reported entries."""

from __future__ import annotations

import pytest
from mlflow.entities import FileInfo

from mlflow_pull import paths
from mlflow_pull.gateway import entry_from_info
from mlflow_pull.paths import MAX_COMPONENT_BYTES, RelPath, UnsafePathError, UnsupportedNameError, fold
from mlflow_pull.ports import ArtifactDir, ArtifactFile, KnownSize, UnsupportedEntry

LEGAL_ON_POSIX = ["k:3.json", "C:x", "dir/C:x", "a\\b", "a?b", "COM1", "a."]
HOSTILE = ["../x", "a/../b", "/abs", "a//b", "a/", "a\x00b", "a\nb", "a\ud800b", "", "./a"]
TOO_LONG = ["a" * (MAX_COMPONENT_BYTES + 1), "d/" + "\u00e9" * 128]


@pytest.mark.parametrize("text", LEGAL_ON_POSIX)
def test_names_legal_on_posix_are_paths_there_and_unsupported_on_windows(
    monkeypatch: pytest.MonkeyPatch, text: str
) -> None:
    monkeypatch.setattr(paths, "IS_WINDOWS", False)
    assert RelPath(text).value == text
    monkeypatch.setattr(paths, "IS_WINDOWS", True)
    with pytest.raises(UnsupportedNameError):
        RelPath(text)


@pytest.mark.parametrize("text", TOO_LONG)
@pytest.mark.parametrize("windows", [False, True])
def test_an_over_long_name_is_unsupported_everywhere(monkeypatch: pytest.MonkeyPatch, text: str, windows: bool) -> None:
    monkeypatch.setattr(paths, "IS_WINDOWS", windows)
    with pytest.raises(UnsupportedNameError, match="255 bytes"):
        RelPath(text)


@pytest.mark.parametrize("text", HOSTILE)
@pytest.mark.parametrize("windows", [False, True])
def test_hostile_names_are_unsafe_not_merely_unsupported(
    monkeypatch: pytest.MonkeyPatch, text: str, windows: bool
) -> None:
    monkeypatch.setattr(paths, "IS_WINDOWS", windows)
    with pytest.raises(UnsafePathError) as caught:
        RelPath(text)
    assert not isinstance(caught.value, UnsupportedNameError)


def test_a_name_that_is_both_hostile_and_unsupported_is_hostile(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(paths, "IS_WINDOWS", True)
    with pytest.raises(UnsafePathError) as caught:
        RelPath("C:/../x")
    assert not isinstance(caught.value, UnsupportedNameError)


# --- the gateway boundary ----------------------------------------------------------------------------------


def test_the_listing_boundary_turns_unsupported_names_into_entries_not_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(paths, "IS_WINDOWS", True)

    file = entry_from_info(FileInfo("sub/k:3.json", False, 5))
    directory = entry_from_info(FileInfo("a?dir", True, None))

    assert isinstance(file, UnsupportedEntry) and file.path.text == "sub/k:3.json"
    assert isinstance(directory, UnsupportedEntry) and directory.path.text == "a?dir"
    assert entry_from_info(FileInfo("ok.txt", False, 5)) == ArtifactFile(RelPath("ok.txt"), KnownSize(5))
    assert entry_from_info(FileInfo("ok", True, None)) == ArtifactDir(RelPath("ok"))


def test_the_listing_boundary_still_raises_on_hostile_names() -> None:
    with pytest.raises(UnsafePathError):
        entry_from_info(FileInfo("../x", False, 1))
    with pytest.raises(UnsafePathError):
        entry_from_info(FileInfo("a\x00b", True, None))


def test_an_unsupported_entry_knows_where_it_is_listed() -> None:
    entry = UnsupportedEntry.named("a/b:c")
    assert entry.path.is_child_of(RelPath("a")) and not entry.path.is_child_of(None)
    assert UnsupportedEntry.named("b:c").path.is_child_of(None)
    assert entry.path.parts == ("a", "b:c")


# --- the fold key ------------------------------------------------------------------------------------------


def test_fold_is_the_standard_caseless_key() -> None:
    assert fold("\u0390") == fold("\u0399\u0308\u0301")  # NFC + casefold alone keeps these apart
    assert fold("Model.PKL") == fold("model.pkl")
    assert fold("\u00e9") == fold("e\u0301")
    assert fold("a") != fold("b")
