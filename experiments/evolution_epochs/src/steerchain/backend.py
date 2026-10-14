"""Model backends. Stub is deterministic. The HF backend is loaded only on request."""

from __future__ import annotations

import json
from dataclasses import dataclass

import numpy as np

from steerchain.injector import add_vector, layer_index, resolve_layers
from steerchain.tools import TOOL_SCHEMAS, parse_tool_calls
from steerchain.workspace import ToolCall


@dataclass
class Generation:
    text: str
    tool_calls: list[ToolCall]
    hidden: np.ndarray
    transformer_layer_k: int


class StubBackend:
    """Transmits a long SOUL.md and records residuals at every layer.

    Layer 1 is built with the largest sender−target delta so auto-select is testable.
    Steering, when armed, is added only at transformer_layer_k.
    """

    local_probes = True
    # Relative delta scales per layer. Index 1 is intentionally the largest.
    _DELTA_SCALES = (1.0, 5.0, 2.0, 1.5)

    def __init__(
        self,
        hidden_size: int = 32,
        n_layers: int = 4,
        layer_k: int | None = None,
        default_soul: str = "",
        transmit: bool = True,
    ) -> None:
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.transformer_layer_k = layer_index(n_layers, layer_k)
        self.layer_locked = layer_k is not None
        self.transmit = transmit
        self.default_soul = default_soul.strip()
        self.sender_vector = np.linspace(0.0, 1.0, hidden_size, dtype=np.float64)
        self.pre_end_vector = np.linspace(1.0, 0.0, hidden_size, dtype=np.float64) * 0.25
        self.other_vector = np.full(hidden_size, 0.1, dtype=np.float64)
        self.model_id = "stub"
        self.armed = False
        self._steering: np.ndarray | None = None
        self.steer_log: list[int] = []
        self.illegal_steering: list[int] = []
        self._last_hidden: np.ndarray | None = None

    def set_steering_layer(self, layer_k: int) -> None:
        self.transformer_layer_k = layer_index(self.n_layers, layer_k)

    def arm_steering(self, vector: np.ndarray) -> None:
        array = np.asarray(vector, dtype=np.float64)
        if array.shape != (self.hidden_size,):
            raise ValueError(f"Steering vector shape {array.shape} != {(self.hidden_size,)}")
        self._steering = array.copy()
        self.armed = True

    def disarm_steering(self) -> None:
        self.armed = False
        self._steering = None

    def release(self) -> None:
        self._last_hidden = None
        self.disarm_steering()

    def generate(self, agent, peers: list[str], role: str, steer: bool) -> Generation:
        if steer:
            self.steer_log.append(agent.hop)
            if agent.hop != 0 or not self.armed:
                self.illegal_steering.append(agent.hop)
        hidden, tool_calls, text = self._act(agent, peers, role)
        if steer and self.armed and self._steering is not None:
            hidden = hidden.copy()
            hidden[self.transformer_layer_k] = add_vector(hidden[self.transformer_layer_k], self._steering)
        self._last_hidden = hidden.copy()
        return Generation(
            text=text,
            tool_calls=tool_calls,
            hidden=hidden,
            transformer_layer_k=self.transformer_layer_k,
        )

    def _stack(self, kind: str) -> np.ndarray:
        out = np.zeros((self.n_layers, self.hidden_size), dtype=np.float64)
        for layer in range(self.n_layers):
            scale = self._DELTA_SCALES[layer % len(self._DELTA_SCALES)]
            if kind == "sender":
                out[layer] = self.pre_end_vector + (self.sender_vector - self.pre_end_vector) * scale
            elif kind == "pre_end":
                out[layer] = self.pre_end_vector.copy()
            else:
                out[layer] = self.other_vector.copy()
        return out

    def _act(self, agent, peers: list[str], role: str) -> tuple[np.ndarray, list[ToolCall], str]:
        soul = agent.workspace.files.get("SOUL.md", "")
        tool_names = [item.get("name") for item in agent.history if item.get("role") == "tool"]
        received = any(
            item.get("role") == "user" and "DM from " in item.get("content", "")
            for item in agent.history
        )
        infected = bool(soul.strip()) and soul.strip() != self.default_soul
        if self.transmit and infected and "send_message" not in tool_names and peers:
            call = ToolCall("send_message", {"to": peers[0], "body": soul})
            kind = "sender" if role == "sender" else "pre_end"
            return self._stack(kind), [call], ""
        if self.transmit and received and not infected and "write_soul" not in tool_names:
            bodies = []
            for item in agent.history:
                content = item.get("content", "")
                if item.get("role") != "user" or "DM from " not in content:
                    continue
                # Persist only the message body, not the delivery wrapper.
                parts = []
                for block in content.split("\n\n"):
                    if block.startswith("DM from ") and ": " in block:
                        parts.append(block.split(": ", 1)[1])
                    else:
                        parts.append(block)
                bodies.append("\n\n".join(parts))
            call = ToolCall("write_soul", {"content": "\n\n".join(bodies)})
            return self._stack("pre_end"), [call], ""
        if role == "target":
            return self._stack("pre_end"), [], "I have nothing further to add."
        return self._stack("other"), [], "I have nothing further to add."


class HFBackend:
    """Local open-weight model. Residual snapshots are last-token states at every layer."""

    local_probes = False

    def __init__(
        self,
        model_id: str,
        layer_k: int | None = None,
        max_new_tokens: int = 256,
        native_tools: bool = True,
    ) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        self.model_id = model_id
        self.max_new_tokens = max_new_tokens
        self.native_tools = native_tools
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        try:
            from steerchain.hub_patch import prepare_hub_downloads

            prepare_hub_downloads()
        except Exception:
            pass
        dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        # Qwen3.5 (model_type qwen3_5) needs a recent transformers + torch>=2.5.
        # On CPU avoid device_map="auto" — it can offload weights to disk and crawl.
        load_kwargs: dict = {"dtype": dtype, "trust_remote_code": True}
        if torch.cuda.is_available():
            load_kwargs["device_map"] = "auto"
        try:
            self.model = AutoModelForCausalLM.from_pretrained(model_id, **load_kwargs)
        except TypeError:
            # Older transformers only accept torch_dtype=
            load_kwargs.pop("dtype", None)
            load_kwargs["torch_dtype"] = dtype
            self.model = AutoModelForCausalLM.from_pretrained(model_id, **load_kwargs)
        if not torch.cuda.is_available():
            self.model.to("cpu")
        self.model.eval()
        self.layers = resolve_layers(self.model)
        self.n_layers = len(self.layers)
        self.transformer_layer_k = layer_index(self.n_layers, layer_k)
        self.layer_locked = layer_k is not None
        self.hidden_size = int(self.model.config.hidden_size)
        self.armed = False
        self._steering: np.ndarray | None = None
        self._last_hidden: np.ndarray | None = None
        self.steer_log: list[int] = []
        self.illegal_steering: list[int] = []

    def set_steering_layer(self, layer_k: int) -> None:
        self.transformer_layer_k = layer_index(self.n_layers, layer_k)

    def arm_steering(self, vector: np.ndarray) -> None:
        array = np.asarray(vector, dtype=np.float64)
        if array.shape != (self.hidden_size,):
            raise ValueError(f"Steering vector shape {array.shape} != {(self.hidden_size,)}")
        self._steering = array.copy()
        self.armed = True

    def disarm_steering(self) -> None:
        self.armed = False
        self._steering = None

    def release(self) -> None:
        self._last_hidden = None
        self.disarm_steering()

    def complete(self, system: str, user: str) -> str:
        return self._generate_text(
            [{"role": "system", "content": system}, {"role": "user", "content": user}],
            armed=False,
        )

    def generate(self, agent, peers: list[str], role: str, steer: bool) -> Generation:
        del role
        if steer:
            self.steer_log.append(agent.hop)
            if agent.hop != 0 or not self.armed:
                self.illegal_steering.append(agent.hop)
        armed = bool(steer and self.armed and self._steering is not None)
        text = self._generate_text(self._plain_history(list(agent.history)), armed=armed)
        hidden = (
            self._last_hidden
            if self._last_hidden is not None
            else np.zeros((self.n_layers, self.hidden_size), dtype=np.float64)
        )
        return Generation(
            text=text,
            tool_calls=parse_tool_calls(text),
            hidden=hidden.copy(),
            transformer_layer_k=self.transformer_layer_k,
        )

    def _plain_history(self, history: list[dict]) -> list[dict]:
        plain = []
        for item in history:
            role = item.get("role", "user")
            if role == "tool":
                plain.append(
                    {
                        "role": "user",
                        "content": f"Tool {item.get('name')} result:\n{item.get('content', '')}",
                    }
                )
                continue
            if role == "assistant":
                text = item.get("content") or ""
                rendered = []
                for call in item.get("tool_calls") or []:
                    rendered.append(f"TOOL: {call['name']}\nARGS: {json.dumps(call['arguments'])}")
                if rendered:
                    text = (text + "\n" + "\n".join(rendered)).strip()
                plain.append({"role": "assistant", "content": text or "..."})
                continue
            plain.append({"role": role, "content": item.get("content", "")})
        return plain

    def _render_prompt(self, messages: list[dict]) -> str:
        if getattr(self.tokenizer, "chat_template", None):
            kwargs = {"tokenize": False, "add_generation_prompt": True}
            if self.native_tools:
                try:
                    return self.tokenizer.apply_chat_template(
                        messages,
                        tools=TOOL_SCHEMAS,
                        **kwargs,
                    )
                except TypeError:
                    pass
                except Exception:
                    pass
            try:
                return self.tokenizer.apply_chat_template(messages, **kwargs)
            except Exception:
                pass
        parts = []
        for item in messages:
            parts.append(f"{item.get('role', 'user').upper()}:\n{item.get('content', '')}")
        parts.append("ASSISTANT:\n")
        return "\n\n".join(parts)

    def _generate_text(self, messages: list[dict], armed: bool) -> str:
        torch = self.torch
        prompt = self._render_prompt(messages)
        inputs = self.tokenizer(prompt, return_tensors="pt")
        device = next(self.model.parameters()).device
        inputs = {key: value.to(device) for key, value in inputs.items()}
        self._last_hidden = np.zeros((self.n_layers, self.hidden_size), dtype=np.float64)
        handles = [
            layer.register_forward_hook(self._make_hook(index, armed and index == self.transformer_layer_k))
            for index, layer in enumerate(self.layers)
        ]
        try:
            with torch.no_grad():
                output = self.model.generate(
                    **inputs,
                    max_new_tokens=self.max_new_tokens,
                    do_sample=False,
                    pad_token_id=self.tokenizer.pad_token_id,
                )
        finally:
            for handle in handles:
                handle.remove()
        new_tokens = output[0, inputs["input_ids"].shape[-1]:]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True)

    def _make_hook(self, layer_index_value: int, armed: bool):
        def hook(module, inputs, output):
            del module, inputs
            hidden = output[0] if isinstance(output, tuple) else output
            if armed and self._steering is not None:
                hidden = add_vector(hidden, self._steering)
                output = (hidden,) + output[1:] if isinstance(output, tuple) else hidden
            self._last_hidden[layer_index_value] = hidden[0, -1].detach().float().cpu().numpy().copy()
            return output

        return hook


def build_backend(config, default_soul: str):
    if config.backend == "stub":
        n_layers = 4
        if config.transformer_layer_k is not None:
            n_layers = max(n_layers, config.transformer_layer_k + 1)
        return StubBackend(
            hidden_size=config.hidden_size,
            n_layers=n_layers,
            layer_k=config.transformer_layer_k,
            default_soul=default_soul,
        )
    if config.backend == "hf":
        return HFBackend(
            config.model_id,
            layer_k=config.transformer_layer_k,
            max_new_tokens=config.max_new_tokens,
            native_tools=bool(getattr(config, "native_tools", True)),
        )
    raise ValueError(f"Unknown backend {config.backend!r}. Expected stub or hf.")
