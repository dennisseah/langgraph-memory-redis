import base64
import json
from unittest.mock import MagicMock

import pytest
from pytest_mock import MockerFixture

from langgraph_memory_redis.models.chat_message import ChatMessage
from langgraph_memory_redis.services.memory_store import (
    MemoryStore,
)

FAKE_TOKEN = "eyJ0eXAiOiJKV1QiLCJhbGciOiJSUzI1NiIsIng1dCI6IkNOdjBPSTNSd3FsSEZFVm5hb01Bc2hDSDJYRSIsImtpZCI6IkNOdjBPSTNSd3FsSEZFVm5hb01Bc2hDSDJYRSJ9.eyJhdWQiOiJodHRwczovL3JlZGlzLmF6dXJlLmNvbSIsImVtYWlsIjoidGVzdEBtaWNyb3NvZnQuY29tIiwiZmFtaWx5X25hbWUiOiJUZXN0IiwiZ2l2ZW5fbmFtZSI6IlRlc3RlciIsIm5hbWUiOiJ0ZXN0Iiwib2lkIjoiMzQwOTg2MTItNDJlMC00MzRkLWI3ZWQtZWM2OGUxZmQ3ZmNhIiwidW5pcXVlX25hbWUiOiJ0ZXN0QG1pY3Jvc29mdC5jb20ifQ.YMa_g64xlyvh2yoZu3BMKrQ1HlK1ZmgAxCEpXMM64_roaYCQ1cs-5_eO1JXLnbfWu7LNpLH2vRYZywsc94WaezcR54rd6NYtQaTX1iP9t_bQmMZ55o9JPd2wY7O_IYwFrzLsNkqjUssp_LMoPl7onAqnODRdibvIFFNo_8hij1TuSNaiYRJbOMk9D5j4l7oxksHGK1luU80oXQTNSMx8toT7c65T_iJNDYlfc6dTHoDSdhAY4DQb3zYVG3m_ao2HjUHCdi0Zf_E7v81hMVIoKj-4MO7heWZKy8GEMna5oStNnnQpmPSmE5pnG2Ecbdaa6DlscZoq1gdaBpw3g0FNBQ"  # noqa E501
FAKE_TOKEN_1 = "eyJ0eXAiOiJKV1QiLCJhbGciOiJSUzI1NiIsIng1dCI6IkNOdjBPSTNSd3FsSEZFVm5hb01Bc2hDSDJYRSIsImtpZCI6IkNOdjBPSTNSd3FsSEZFVm5hb01Bc2hDSDJYRSJ9.eyJlbWFpbCI6InRlc3RAbWljcm9zb2Z0LmNvbSIsImZhbWlseV9uYW1lIjoiVGVzdCIsImdpdmVuX25hbWUiOiJUZXN0ZXIiLCJuYW1lIjoidGVzdCIsIm9pZCI6IjM0MDk4NjEyLTQyZTAtNDM0ZC1iN2VkLWVjNjhlMWZkN2ZjYSIsInVuaXF1ZV9uYW1lIjoidGVzdEBtaWNyb3NvZnQuY29tIn0.rYULOFlpeS3GaMMmQvUFyeOBIdHzCnKSoHr0TQ7Bq3TtL2zBQUV5EdMqC83YD2AGiFwhIm_qI8D0xRvwpwpz-bJbRkl-xZDc59WM7bVpwm7iBi9UtI66nV8jLqlVOSfTho5qVQSL2eKBNdBxxLjlvqjcvBmFCj5LdZKO7r71FSk3QhhJGzB57zIsj6eIlAbeCYklN0qlLk72QcZexl341FxfDoMnoJv_9B5lTs7o42ym23j6d7FQw3P0-DUW7tKyz63wLkDUR2hoePFG4Hg8WLLgVMYj8A-Seqj0ydgf4esk32QKnHanjJN-ceU16dQAf3rL_oSDcJu4vIjxINLd8A"  # noqa E501


@pytest.fixture
def mock_chat_messages() -> list[ChatMessage]:
    return [
        ChatMessage(role="user", message="Hello", domain="test", ts=1234567890.0),
        ChatMessage(role="ai", message="Hi there!", domain="test", ts=1234567891.0),
    ]


@pytest.fixture
def mock_service(mocker: MockerFixture) -> MemoryStore:
    mock_cred = MagicMock()
    mock_cred.get_token.return_value = MagicMock(
        token="mock_token",
        scope="https://example.com/.default",
    )
    mocker.patch(
        "langgraph_memory_redis.services.memory_store.DefaultAzureCredential",
        return_value=mock_cred,
    )
    mock_redis = MagicMock()
    mocker.patch(
        "langgraph_memory_redis.services.memory_store.Redis",
        return_value=mock_redis,
    )

    svc = MemoryStore(env=MagicMock())  # type: ignore
    return svc


@pytest.mark.parametrize("token", [FAKE_TOKEN, FAKE_TOKEN_1])
def test_service_default_cred(mocker: MockerFixture, token: str):
    mock_cred = MagicMock()
    mock_cred.get_token.return_value = MagicMock(
        token=token,
        scope="https://example.com/.default",
    )
    mocker.patch(
        "langgraph_memory_redis.services.memory_store.DefaultAzureCredential",
        return_value=mock_cred,
    )
    mock_redis = MagicMock()
    mocker.patch(
        "langgraph_memory_redis.services.memory_store.Redis",
        return_value=mock_redis,
    )
    mock_env = MagicMock()
    mock_env.redis_key = None

    MemoryStore(env=mock_env)  # type: ignore


def test_read_history(
    mock_service: MemoryStore,
    mock_chat_messages: list[ChatMessage],
):
    mock_service.redis_client.get = MagicMock(
        return_value=json.dumps([msg.model_dump() for msg in mock_chat_messages])
    )
    assert len(mock_service.read_history("test_key")) == 2


def test_read_history_none(mock_service: MemoryStore):
    mock_service.redis_client.get = MagicMock(return_value=None)
    assert mock_service.read_history("test_key") == []


def test_write_history(
    mock_service: MemoryStore, mock_chat_messages: list[ChatMessage]
):
    mock_service.redis_client.get = MagicMock(
        return_value=json.dumps([msg.model_dump() for msg in mock_chat_messages])
    )
    mock_service.redis_client.set = MagicMock()
    mock_service.write_history("test_key", mock_chat_messages)
    mock_service.redis_client.set.assert_called_once()


def test_read_graph_memory(mock_service: MemoryStore, mocker: MockerFixture):
    mocker.patch(
        "langgraph_memory_redis.services.memory_store.dill.loads",
        return_value=b"graph_memory_data",
    )
    mock_service.redis_client.get = MagicMock(
        return_value=base64.b64encode(b"graph_memory_data")
    )
    assert mock_service.read_graph_memory("test_key") == b"graph_memory_data"


def test_read_graph_memory_none(mock_service: MemoryStore):
    mock_service.redis_client.get = MagicMock(return_value=None)
    assert mock_service.read_graph_memory("test_key") is None


def test_write_graph_memory(mock_service: MemoryStore, mocker: MockerFixture):
    dill_dump = mocker.patch("dill.dumps", return_value=b"graph_memory_data")
    mock_service.redis_client.set = MagicMock()
    mock_service.write_graph_memory("test_key", MagicMock())

    dill_dump.assert_called_once()
    mock_service.redis_client.set.assert_called_once()


def test_put(mock_service: MemoryStore, mock_chat_messages: list[ChatMessage]):
    mock_service.write_history = MagicMock()
    mock_service.write_graph_memory = MagicMock()

    mock_service.put("test_key", mock_chat_messages, MagicMock())
    mock_service.write_history.assert_called_once()
    mock_service.write_graph_memory.assert_called_once()


def test_restore(mock_service: MemoryStore):
    mock_service.read_graph_memory = MagicMock(return_value=None)
    restored_memory = mock_service.restore("test_key")
    assert len(restored_memory.blobs) == 0


def test_restore_data(mock_service: MemoryStore, mocker: MockerFixture):
    mocker.patch("langgraph_memory_redis.services.memory_store.MemorySaver")
    mock_service.read_graph_memory = mocker.MagicMock(return_value=MagicMock())
    result = mock_service.restore("test_key")
    assert result is not None


def test_get_chat_history(mock_service: MemoryStore):
    mock_service.read_history = MagicMock(return_value=MagicMock())
    result = mock_service.get_chat_history("test_key")
    assert result is not None
