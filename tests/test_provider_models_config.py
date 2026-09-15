"""provider.models 手动模型列表：解析校验 + Settings/写回 roundtrip。"""

import tomllib

import pytest

import config_writer
from config import Settings, _parse_providers


def _providers(models):
    item = {"id": "p", "openai": {"base_url": "https://x.example.com/v1", "api_key": "k"}}
    if models is not None:
        item["models"] = models
    return [item]


def test_parse_models_strips_and_dedupes():
    """模型名去空白、去重，保持首次出现顺序。"""
    providers = _parse_providers(_providers(["  a ", "b", "a"]))
    assert providers[0].models == ["a", "b"]


def test_parse_models_missing_is_empty():
    providers = _parse_providers(_providers(None))
    assert providers[0].models == []


@pytest.mark.parametrize("bad", ["not-a-list", [1], [""], [None]])
def test_parse_models_rejects_invalid(bad):
    with pytest.raises(SystemExit):
        _parse_providers(_providers(bad))


def test_models_roundtrip(tmp_path, monkeypatch):
    """TOML → Settings → to_dict → config_writer → TOML 全程保持 models。"""
    text = """
[[providers]]
id = "volc"
api_key = "ark-k"
models = ["deepseek-v4-pro", "MiniMax/M3"]

[providers.openai]
base_url = "https://ark.example.com/api/v3/"
"""
    cfg = tmp_path / "config.toml"
    cfg.write_text(text, encoding="utf-8")
    monkeypatch.setenv("CONFIG_FILE", str(cfg))

    s = Settings()
    assert s.PROVIDERS["volc"].models == ["deepseek-v4-pro", "MiniMax/M3"]

    data = s.to_dict()
    assert data["providers"][0]["models"] == ["deepseek-v4-pro", "MiniMax/M3"]

    reparsed = tomllib.loads(config_writer.dumps(data))
    assert reparsed["providers"][0]["models"] == ["deepseek-v4-pro", "MiniMax/M3"]


def test_models_empty_omitted_in_writer(tmp_path, monkeypatch):
    """models 为空时 to_dict / 写回都不产出该字段。"""
    text = """
[[providers]]
id = "plain"

[providers.openai]
base_url = "https://x.example.com/v1/"
api_key = "k"
"""
    cfg = tmp_path / "config.toml"
    cfg.write_text(text, encoding="utf-8")
    monkeypatch.setenv("CONFIG_FILE", str(cfg))

    s = Settings()
    assert s.PROVIDERS["plain"].models == []
    assert "models" not in s.to_dict()["providers"][0]
    assert "models" not in tomllib.loads(config_writer.dumps(s.to_dict()))["providers"][0]
