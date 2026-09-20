"""
Tests for model tiers catalog, hardware-based tier suggestions, and role plumbing.
All tests are offline: monkeypatched HF API calls, temp directories for models.
"""
import os
import tempfile
import pytest
from unittest.mock import patch, MagicMock


# ---- Catalog shape tests ----------------------------------------------------

def test_tiers_catalog_has_all_nine_ids():
    from engine.model_manager import TIERS
    expected_ids = {"0.8b", "2b", "4b", "3b-llama", "1.5b-coder", "3b-coder", "8b-llama", "7b-coder", "7b-r1"}
    assert set(TIERS.keys()) == expected_ids
    assert "9b" not in TIERS


def test_tiers_catalog_entry_keys():
    from engine.model_manager import TIERS
    required_keys = {"repo", "quant", "size_gb", "min_vram_gb", "n_ctx", "roles", "label", "match"}
    for tier_id, entry in TIERS.items():
        assert set(entry.keys()) == required_keys, f"Tier {tier_id} has wrong keys"


def test_tiers_catalog_valid_quants():
    from engine.model_manager import TIERS
    valid_quants = {"q4_k_m", "q3_k_m"}
    for tier_id, entry in TIERS.items():
        assert entry["quant"] in valid_quants, f"Tier {tier_id} has invalid quant {entry['quant']}"


def test_tiers_catalog_valid_roles():
    from engine.model_manager import TIERS
    valid_roles = {"chat", "grc", "code", "osint", "cti"}
    for tier_id, entry in TIERS.items():
        assert set(entry["roles"]) <= valid_roles, f"Tier {tier_id} has invalid roles"


def test_tiers_catalog_q3_quant_for_high_tiers():
    from engine.model_manager import TIERS
    assert TIERS["8b-llama"]["quant"] == "q3_k_m"
    assert TIERS["7b-coder"]["quant"] == "q3_k_m"
    assert TIERS["7b-r1"]["quant"] == "q3_k_m"


def test_tiers_catalog_n_ctx_values():
    from engine.model_manager import TIERS
    assert TIERS["8b-llama"]["n_ctx"] == 12288
    assert TIERS["7b-coder"]["n_ctx"] == 12288
    for tier_id in ["0.8b", "2b", "4b", "3b-llama", "1.5b-coder", "3b-coder", "7b-r1"]:
        assert TIERS[tier_id]["n_ctx"] == 8192, f"Tier {tier_id} should have n_ctx=8192"


# ---- suggest_tiers matrix tests ---------------------------------------------

@pytest.mark.parametrize("ram_gb,vram_gb,has_nvidia,expected", [
    (32.0, 8.0, True, {"chat": "8b-llama", "grc": "8b-llama", "code": "7b-coder", "osint": "7b-r1", "cti": "8b-llama"}),
    (16.0, 4.0, True, {"chat": "4b", "grc": "4b", "code": "3b-coder", "osint": "4b", "cti": "4b"}),
    (16.0, 2.0, True, {"chat": "2b", "grc": "3b-llama", "code": "3b-coder", "osint": "2b", "cti": "2b"}),
    (8.0, 0.0, False, {"chat": "2b", "grc": "2b", "code": "1.5b-coder", "osint": "2b", "cti": "2b"}),
    (4.0, 0.0, False, {"chat": "0.8b", "grc": "0.8b", "code": "1.5b-coder", "osint": "0.8b", "cti": "0.8b"}),
])
def test_suggest_tiers_matrix(ram_gb, vram_gb, has_nvidia, expected):
    from engine.hardware import Hardware, suggest_tiers
    hw = Hardware(ram_gb=ram_gb, vram_gb=vram_gb, has_nvidia=has_nvidia)
    assert suggest_tiers(hw) == expected


def test_suggest_tier_wrapper_equals_chat_role():
    from engine.hardware import Hardware, suggest_tier, suggest_tiers
    hw = Hardware(ram_gb=32.0, vram_gb=8.0, has_nvidia=True)
    assert suggest_tier(hw) == suggest_tiers(hw)["chat"]


# ---- current_tier / tier_status / list_local_models tests -------------------

@pytest.fixture
def temp_models_dir(monkeypatch):
    from core.config import settings
    tmpdir = tempfile.mkdtemp(prefix="fortis-test-models-")
    monkeypatch.setattr(settings, "models_dir", tmpdir)
    yield tmpdir


def test_list_local_models_recursive(temp_models_dir):
    from engine.model_manager import list_local_models
    subdir = os.path.join(temp_models_dir, "subfolder")
    os.makedirs(subdir)
    fake_file = os.path.join(subdir, "test-model.gguf")
    with open(fake_file, "wb") as f:
        f.write(b"fake gguf content")
    
    models = list_local_models()
    assert len(models) == 1
    assert models[0]["name"] == "test-model.gguf"
    assert subdir in models[0]["path"]


def test_tier_status_includes_extended_keys(temp_models_dir):
    from engine.model_manager import tier_status, TIERS
    status = tier_status()
    for tier_id in TIERS:
        assert tier_id in status
        entry = status[tier_id]
        assert "label" in entry
        assert "size_gb" in entry
        assert "downloaded" in entry
        assert "active" in entry
        assert "roles" in entry
        assert "min_vram_gb" in entry
        assert "n_ctx" in entry


def test_current_tier_and_tier_status_with_fake_files(temp_models_dir, monkeypatch):
    from engine.model_manager import current_tier, tier_status, set_selection, TIERS
    
    subdir = os.path.join(temp_models_dir, "Llama-3.1-8B-Q3_K_M-GGUF")
    os.makedirs(subdir)
    fake_8b = os.path.join(subdir, "Llama-3.1-8B-Q3_K_M-GGUF.gguf")
    with open(fake_8b, "wb") as f:
        f.write(b"fake 8b llama gguf")
    
    set_selection(fake_8b)
    
    assert current_tier() == "8b-llama"
    
    status = tier_status()
    assert status["8b-llama"]["downloaded"] is True
    assert status["8b-llama"]["active"] is True
    assert status["7b-coder"]["downloaded"] is False


def test_tier_status_matches_unsloth_filenames(temp_models_dir, monkeypatch):
    from engine.model_manager import current_tier, set_selection
    
    filenames_and_expected_tiers = [
        ("Llama-3.1-8B-Q3_K_M-GGUF.gguf", "8b-llama"),
        ("Qwen2.5-Coder-7B-Instruct-Q3_K_M.gguf", "7b-coder"),
        ("DeepSeek-R1-Distill-Qwen-7B-Q3_K_M.gguf", "7b-r1"),
        ("Qwen3.5-4B-Q4_K_M.gguf", "4b"),
        ("Qwen3.5-2B-Q4_K_M.gguf", "2b"),
        ("Qwen3.5-0.8B-Q4_K_M.gguf", "0.8b"),
        ("Llama-3.2-3B-Q4_K_M.gguf", "3b-llama"),
        ("Qwen2.5-Coder-3B-Instruct-Q4_K_M.gguf", "3b-coder"),
        ("Qwen2.5-Coder-1.5B-Instruct-Q4_K_M.gguf", "1.5b-coder"),
    ]
    
    for filename, expected_tier in filenames_and_expected_tiers:
        filepath = os.path.join(temp_models_dir, filename)
        with open(filepath, "wb") as f:
            f.write(b"fake gguf")
        set_selection(filepath)
        assert current_tier() == expected_tier, f"Expected {expected_tier} for {filename}"
        os.remove(filepath)


# ---- _pick_gguf_file quant selection tests ----------------------------------

def test_pick_gguf_file_selects_q3_when_requested():
    from engine.model_manager import _pick_gguf_file
    
    fake_files = ["model-Q4_K_M.gguf", "model-Q3_K_M.gguf", "model-IQ1_S.gguf"]
    
    with patch("huggingface_hub.HfApi") as mock_api:
        mock_instance = MagicMock()
        mock_instance.list_repo_files.return_value = fake_files
        mock_api.return_value = mock_instance
        
        result = _pick_gguf_file("test/repo", quant="q3_k_m")
        assert result == "model-Q3_K_M.gguf"


def test_pick_gguf_file_default_selects_q4():
    from engine.model_manager import _pick_gguf_file
    
    fake_files = ["model-Q4_K_M.gguf", "model-Q3_K_M.gguf", "model-IQ1_S.gguf"]
    
    with patch("huggingface_hub.HfApi") as mock_api:
        mock_instance = MagicMock()
        mock_instance.list_repo_files.return_value = fake_files
        mock_api.return_value = mock_instance
        
        result = _pick_gguf_file("test/repo")
        assert result == "model-Q4_K_M.gguf"


def test_pick_gguf_file_empty_list_raises_runtime_error():
    from engine.model_manager import _pick_gguf_file
    
    with patch("huggingface_hub.HfApi") as mock_api:
        mock_instance = MagicMock()
        mock_instance.list_repo_files.return_value = []
        mock_api.return_value = mock_instance
        
        with pytest.raises(RuntimeError, match="No GGUF files found"):
            _pick_gguf_file("test/repo")


# ---- ROLE_PRESETS tests -----------------------------------------------------

def test_role_presets_exact_keys():
    from core.rag import ROLE_PRESETS
    assert set(ROLE_PRESETS.keys()) == {"general", "grc", "code", "osint", "cti"}


def test_grc_preset_mentions_control_ids():
    from core.rag import ROLE_PRESETS
    preset = ROLE_PRESETS["grc"].lower()
    assert "control" in preset or "control_id" in preset


def test_code_preset_mentions_cwe():
    from core.rag import ROLE_PRESETS
    assert "cwe" in ROLE_PRESETS["code"].lower()


def test_cti_preset_mentions_attack():
    from core.rag import ROLE_PRESETS
    assert "att&ck" in ROLE_PRESETS["cti"].lower() or "attack" in ROLE_PRESETS["cti"].lower()


def test_build_chat_messages_with_grc_role_injects_preset():
    from core.rag import ROLE_PRESETS, build_chat_messages, SYSTEM_PERSONA
    
    messages = build_chat_messages("test-eng", "test message", role="grc")
    system_content = messages[0]["content"]
    
    assert SYSTEM_PERSONA in system_content
    assert ROLE_PRESETS["grc"] in system_content


def test_build_chat_messages_with_none_role_uses_system_persona_only():
    from core.rag import ROLE_PRESETS, build_chat_messages, SYSTEM_PERSONA
    
    messages = build_chat_messages("test-eng", "test message", role=None)
    system_content = messages[0]["content"]
    
    assert system_content == SYSTEM_PERSONA
    for role_preset in ROLE_PRESETS.values():
        assert role_preset not in system_content


def test_build_chat_messages_with_bogus_role_uses_system_persona_only():
    from core.rag import ROLE_PRESETS, build_chat_messages, SYSTEM_PERSONA
    
    messages = build_chat_messages("test-eng", "test message", role="bogus")
    system_content = messages[0]["content"]
    
    assert system_content == SYSTEM_PERSONA
    for role_preset in ROLE_PRESETS.values():
        assert role_preset not in system_content


# ---- Chat endpoint role plumbing tests --------------------------------------

def test_chat_endpoint_role_reaches_build_chat_messages(monkeypatch):
    from fastapi.testclient import TestClient
    from core.main import app as core_app
    from core.routers import chat as chat_router
    
    captured_kwargs = {}
    
    original_build = chat_router.rag.build_chat_messages
    
    def capture_build(engagement_id, message, history=None, role=None, retrieval_query=None):
        captured_kwargs["role"] = role
        return original_build(engagement_id, message, history, role=role)
    
    monkeypatch.setattr(chat_router.rag, "build_chat_messages", capture_build)
    
    async def fake_chat(messages):
        return "stub reply"
    
    monkeypatch.setattr("core.routers.chat.chat", fake_chat)
    
    with TestClient(core_app) as client:
        resp = client.post("/engagements", json={"client_name": "Test Co", "engagement_name": "Test Eng"})
        eng_id = resp.json()["id"]
        
        resp = client.post("/chat", json={"engagement_id": eng_id, "message": "hello", "role": "grc"})
        assert resp.status_code == 200
        assert captured_kwargs.get("role") == "grc"


def test_chat_endpoint_unknown_role_coerced(monkeypatch):
    from fastapi.testclient import TestClient
    from core.main import app as core_app
    from core.routers import chat as chat_router
    
    captured_kwargs = {}
    
    original_build = chat_router.rag.build_chat_messages
    
    def capture_build(engagement_id, message, history=None, role=None, retrieval_query=None):
        captured_kwargs["role"] = role
        return original_build(engagement_id, message, history, role=role)
    
    monkeypatch.setattr(chat_router.rag, "build_chat_messages", capture_build)
    
    async def fake_chat(messages):
        return "stub reply"
    
    monkeypatch.setattr("core.routers.chat.chat", fake_chat)
    
    with TestClient(core_app) as client:
        resp = client.post("/engagements", json={"client_name": "Test Co", "engagement_name": "Test Eng"})
        eng_id = resp.json()["id"]
        
        resp = client.post("/chat", json={"engagement_id": eng_id, "message": "hello", "role": "unknown_role"})
        assert resp.status_code == 200
        assert captured_kwargs.get("role") is None


def test_chat_stream_endpoint_role_reaches_build_chat_messages(monkeypatch):
    from fastapi.testclient import TestClient
    from core.main import app as core_app
    from core.routers import chat as chat_router
    
    captured_kwargs = {}
    
    original_build = chat_router.rag.build_chat_messages
    
    def capture_build(engagement_id, message, history=None, role=None, retrieval_query=None):
        captured_kwargs["role"] = role
        return original_build(engagement_id, message, history, role=role)
    
    monkeypatch.setattr(chat_router.rag, "build_chat_messages", capture_build)
    
    async def fake_chat_stream(messages):
        yield "stub"
    
    async def fake_chat(messages):
        return "stub reply"
    
    monkeypatch.setattr("core.routers.chat.chat_stream", fake_chat_stream)
    monkeypatch.setattr("core.routers.chat.chat", fake_chat)
    
    with TestClient(core_app) as client:
        resp = client.post("/engagements", json={"client_name": "Test Co", "engagement_name": "Test Eng"})
        eng_id = resp.json()["id"]
        
        resp = client.post("/chat/stream", json={"engagement_id": eng_id, "message": "hello", "role": "code"})
        assert resp.status_code == 200
        assert captured_kwargs.get("role") == "code"


# ---- Orchestrator.chat_turn role passthrough tests --------------------------

def test_orchestrator_chat_turn_role_passes_to_chat(monkeypatch):
    from server.orchestration import Orchestrator
    from core import rag
    from core import store
    
    captured_role = {}
    
    def mock_get_active_engagement(user_id):
        return {"id": "test-eng", "client_name": "Test", "name": "Test", "status": "active", "files": []}
    
    def mock_get_engagement(eng_id):
        return {"id": "test-eng", "client_name": "Test", "name": "Test", "status": "active"}
    
    monkeypatch.setattr(store, "get_active_engagement", mock_get_active_engagement)
    monkeypatch.setattr(store, "get_engagement", mock_get_engagement)
    
    original_build = rag.build_chat_messages
    
    def capture_build(engagement_id, user_message, history=None, role=None, retrieval_query=None):
        captured_role["role"] = role
        return original_build(engagement_id, user_message, history, role=role)
    
    monkeypatch.setattr(rag, "build_chat_messages", capture_build)
    
    async def fake_engine_chat(messages):
        return "stub reply"
    
    monkeypatch.setattr("engine.chat", fake_engine_chat)
    
    orch = Orchestrator()
    result = orch.chat_turn("what security vulnerabilities are in this code?", role="osint")
    
    assert captured_role.get("role") == "osint"


# ---- /api/model/select n_ctx test -------------------------------------------

def test_model_select_sets_n_ctx_from_tier(monkeypatch):
    from server.app import app
    from fastapi.testclient import TestClient
    from core.config import settings
    import engine.model_manager as model_manager
    import tempfile
    import os
    
    original_n_ctx = settings.n_ctx
    
    tmpdir = tempfile.mkdtemp(prefix="fortis-test-select-")
    fake_model = os.path.join(tmpdir, "Llama-3.1-8B-Q3_K_M-GGUF.gguf")
    with open(fake_model, "wb") as f:
        f.write(b"fake 8b llama")
    
    def mock_list_local_models():
        return [{"path": fake_model, "name": "Llama-3.1-8B-Q3_K_M-GGUF.gguf", "size_gb": 3.9}]
    
    monkeypatch.setattr(model_manager, "list_local_models", mock_list_local_models)
    
    captured_n_ctx = {}
    
    original_set_selection = model_manager.set_selection
    
    def capture_set_selection(path):
        original_set_selection(path)
        captured_n_ctx["n_ctx"] = model_manager.TIERS.get("8b-llama", {}).get("n_ctx", 8192)
    
    monkeypatch.setattr(model_manager, "set_selection", capture_set_selection)
    
    with TestClient(app) as client:
        resp = client.post("/api/model/select", json={"tier": "8b-llama"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "selected"
    
    assert captured_n_ctx.get("n_ctx") == 12288
    
    settings.n_ctx = original_n_ctx
