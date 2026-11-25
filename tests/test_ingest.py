import sys
from pathlib import Path

import pytest

from market_research import ingest
from market_research.indexing import IndexChanges
from market_research.rag import KnowledgeIndexReport


def test_ingest_parser_supports_force_aliases():
    parser = ingest.build_parser()

    force_args = parser.parse_args(["--force"])
    rebuild_args = parser.parse_args(["--force-rebuild"])

    assert force_args.force_rebuild is True
    assert rebuild_args.force_rebuild is True


def test_ingest_main_passes_options_and_reports_result(monkeypatch, capsys, tmp_path: Path):
    captured = {}

    def fake_ingest_knowledge_base(**kwargs):
        captured.update(kwargs)
        return KnowledgeIndexReport(
            knowledge_directory=tmp_path / "knowledge",
            persist_directory=tmp_path / "index",
            collection_name="test_collection",
            document_count=2,
            chunk_count=4,
            rebuilt=True,
            manifest_path=tmp_path / "index" / "test_collection.manifest.json",
        )

    monkeypatch.setattr(ingest, "ingest_knowledge_base", fake_ingest_knowledge_base)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "market-research-index",
            "--knowledge-dir",
            str(tmp_path / "knowledge"),
            "--index-dir",
            str(tmp_path / "index"),
            "--collection-name",
            "test_collection",
            "--force",
        ],
    )

    ingest.main()

    assert captured == {
        "knowledge_dir": tmp_path / "knowledge",
        "persist_directory": tmp_path / "index",
        "collection_name": "test_collection",
        "force_rebuild": True,
    }
    output = capsys.readouterr().out
    assert "Rebuilt knowledge index." in output
    assert "- Documents: 2" in output
    assert "- Chunks: 4" in output


@pytest.mark.parametrize("changed,action", [(True, "Updated"), (False, "Reused")])
def test_ingest_reports_incremental_action(monkeypatch, capsys, tmp_path, changed, action):
    report = KnowledgeIndexReport(
        knowledge_directory=tmp_path, persist_directory=tmp_path,
        collection_name="test_collection", document_count=1, chunk_count=1,
        rebuilt=False, manifest_path=tmp_path / "manifest.json",
        changes=IndexChanges(changed=changed, metadata_updates=int(changed)),
    )
    monkeypatch.setattr(ingest, "ingest_knowledge_base", lambda **kwargs: report)
    monkeypatch.setattr(sys, "argv", ["market-research-index"])
    ingest.main()
    output = capsys.readouterr().out
    assert f"{action} knowledge index." in output
    assert "Unique texts embedded: 0" in output
    assert f"Metadata updates: {int(changed)}" in output
