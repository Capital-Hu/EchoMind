"""工具定义、范围检查、缓存、熔断及完整 MCP 调用转发。

工具业务流程在 MCP Server 中执行；主进程不代工具做查询改写或重排。
"""
import asyncio
import copy
import hashlib
import inspect
import json
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

from tooling.mcp_client import MCPProtocolError, MCPTransportError
from tooling.mcp_protocol import BUSINESS_INSTRUCTIONS, validate_arguments
from jsonschema import ValidationError

logger = logging.getLogger(__name__)


# ── 数据结构 ──────────────────────────────────────────────────────────────────

class CircuitState(Enum):
    CLOSED    = "closed"     # 正常
    OPEN      = "open"       # 熔断，拒绝请求
    HALF_OPEN = "half_open"  # 探测恢复


@dataclass
class ToolResult:
    success:        bool
    data:           Any
    tool_name:      str
    error:          Optional[str] = None
    cached:         bool = False
    latency_ms:     float = 0.0
    reranked:       bool = False   # 是否经过重排
    executed:       bool = False   # 是否实际进入 handler（缓存命中不会再次执行）
    transport:      str = "local"
    execution_status: str = "not_executed"
    receipt:        Dict[str, Any] = field(default_factory=dict)
    retrievals:     List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class ToolStats:
    """工具运行时统计，供 Monitor 读取。"""
    total:              int = 0
    success:            int = 0
    failed:             int = 0
    total_latency_ms:   float = 0.0
    consecutive_fails:  int = 0

    @property
    def success_rate(self) -> float:
        return self.success / self.total if self.total else 1.0

    @property
    def avg_latency_ms(self) -> float:
        return self.total_latency_ms / self.total if self.total else 0.0


# ── 熔断器 ────────────────────────────────────────────────────────────────────

class CircuitBreaker:
    """
    三态熔断器：CLOSED → OPEN → HALF_OPEN → CLOSED

    连续失败 failure_threshold 次后打开；
    打开 recovery_s 秒后进入 HALF_OPEN 探测；
    探测成功则关闭，失败则重新打开。
    """

    def __init__(self, failure_threshold: int = 5, recovery_s: float = 60.0):
        self.threshold   = failure_threshold
        self.recovery_s  = recovery_s
        self.state       = CircuitState.CLOSED
        self.fail_count  = 0
        self.opened_at:  Optional[float] = None

    def allow(self) -> bool:
        if self.state == CircuitState.CLOSED:
            return True
        if self.state == CircuitState.OPEN:
            if time.monotonic() - self.opened_at >= self.recovery_s:  # type: ignore
                self.state = CircuitState.HALF_OPEN
                return True
            return False
        return True  # HALF_OPEN：放行一次探测

    def record_success(self) -> None:
        self.fail_count = 0
        self.state = CircuitState.CLOSED

    def record_failure(self) -> None:
        self.fail_count += 1
        if self.fail_count >= self.threshold:
            self.state     = CircuitState.OPEN
            self.opened_at = time.monotonic()
            logger.warning(f"熔断器打开（连续失败 {self.fail_count} 次）")


# ── 工具定义 ──────────────────────────────────────────────────────────────────

@dataclass
class Tool:
    name:        str
    description: str
    handler:     Callable                    # async (params, context) -> Any
    schema:      Dict[str, Any]              # JSON Schema
    cache_ttl:   float = 0.0                 # 0 = 不缓存
    timeout_s:   float = 30.0
    fallback:    Optional[Callable] = None    # sync/async (params, context, error) -> Any
    agent_types: tuple[str, ...] = ()        # 显式声明哪些 Agent 可从模型调用；默认不暴露
    transport: str = "local"
    invalidates_cache_for: tuple[str, ...] = ()  # 成功执行后失效的工具缓存

    # 运行时状态（不参与构造）
    stats:   ToolStats    = field(default_factory=ToolStats, init=False)
    breaker: CircuitBreaker = field(default_factory=CircuitBreaker, init=False)


# ── MCP 工具管理器 ────────────────────────────────────────────────────────────

class MCPToolManager:
    """
    工具注册、原生 function calling 定义和 MCP 进程调用。

    发现 MCP 工具并提供给模型；真实业务逻辑由服务端执行。
    """

    def __init__(self, *, instructions=BUSINESS_INSTRUCTIONS):
        self.instructions = instructions
        self._tools: Dict[str, Tool] = {}
        self._cache: Dict[tuple[str, str], tuple] = {}  # (工具名, 请求摘要) → (结果, 过期时间)
        self._cache_versions: Dict[str, int] = {}
        self.mcp_client = None

    def attach_mcp(self, client):
        self.mcp_client = client
        for discovered in client.tools:
            name = discovered.name
            policy = client.catalog[name]
            async def invoke(params, context, tool_name=name):
                return await self._call_mcp(tool_name, params, context)
            self.register(Tool(name, discovered.description, invoke, discovered.inputSchema,
                               cache_ttl=policy.cache_ttl, timeout_s=policy.timeout + 5,
                               agent_types=policy.scopes,
                               transport="stdio", invalidates_cache_for=policy.invalidates_cache_for))

    async def _call_mcp(self, name, params, context):
        """转交一个完整工具请求，接收 Server 的最终结果及内部检索记录。"""
        try:
            payload = await self.mcp_client.call(name, params, context)
        except (MCPTransportError, MCPProtocolError) as ex:
            return ToolResult(False, None, name, error=str(ex), transport="stdio", execution_status="unknown")
        return ToolResult(
            success=payload["success"], data=payload["data"], tool_name=name,
            error=payload["error"] or None, executed=payload["executed"], transport="stdio",
            execution_status=payload["execution_status"], receipt=payload["receipt"],
            latency_ms=payload["latency_ms"], reranked=payload["reranked"], retrievals=payload["retrievals"],
        )

    async def close(self):
        if self.mcp_client is not None:
            await self.mcp_client.close()

    # ── 注册 / 注销 ───────────────────────────────────────────────────────────

    def register(self, tool: Tool) -> None:
        self._tools[tool.name] = tool
        logger.info(f"注册工具: {tool.name}")

    def unregister(self, name: str) -> None:
        self._tools.pop(name, None)

    def definitions(self, agent_type: str) -> List[Dict[str, Any]]:
        """把已发现的 MCP 工具转换为 Anthropic 原生工具定义。"""
        return [{"name": tool.name, "description": tool.description,
                 "input_schema": copy.deepcopy(tool.schema)}
                for tool in self._tools.values() if agent_type in tool.agent_types]

    # ── 核心调用 ──────────────────────────────────────────────────────────────

    async def call(
        self,
        name: str,
        params: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None,
        *,
        use_cache: bool = True,
    ) -> ToolResult:
        """
        调用工具，完整执行链：
          范围与参数校验 → 缓存 → 熔断检查 → 工具执行（检索含改写与重排）→ 缓存写入
        """
        tool = self._tools.get(name)
        if not tool:
            return ToolResult(success=False, data=None, tool_name=name, error=f"工具不存在: {name}")

        if tool.transport == "stdio" and (context or {}).get("agent_type") not in tool.agent_types:
            return ToolResult(success=False, data=None, tool_name=name, error="当前调用范围不允许使用此工具",
                              transport="stdio", execution_status="rejected")

        # 模型参数不可信：先校验，再查缓存；参数错误不触发降级或熔断。
        try:
            self._validate_params(tool, params)
        except ValueError as ex:
            tool.stats.total += 1
            tool.stats.failed += 1
            return ToolResult(success=False, data=None, tool_name=name, error=str(ex),
                              transport=tool.transport, execution_status="rejected")

        # 缓存命中
        if use_cache and tool.cache_ttl > 0:
            cached = self._get_cache(name, params, context)
            if cached is not None:
                tool.stats.total += 1
                tool.stats.success += 1
                cached.cached, cached.executed = True, False
                cached.execution_status, cached.latency_ms = "cached", 0
                return cached

        # 熔断检查
        if not tool.breaker.allow():
            error = f"工具熔断中: {name}，请稍后重试"
            return await self._fallback_result(tool, params, context, error)

        t0 = time.monotonic()
        cache_version = self._cache_versions.get(name, 0)
        tool.stats.total += 1
        try:
            data = await asyncio.wait_for(self._run_handler(tool, params, context), timeout=tool.timeout_s)
            latency = (time.monotonic() - t0) * 1000

            result = data if isinstance(data, ToolResult) else ToolResult(
                success=True, data=data, tool_name=name, executed=True, execution_status="succeeded")
            result.latency_ms = latency
            if not result.success:
                tool.stats.failed += 1
                if result.executed:
                    tool.breaker.record_failure()
                return result

            tool.stats.success += 1
            tool.stats.consecutive_fails = 0
            tool.stats.total_latency_ms += latency
            tool.breaker.record_success()

            if tool.invalidates_cache_for:
                self._invalidate_caches(tool.invalidates_cache_for)

            # 缓存失效前发出的调用可能返回旧结果，跳过回写。
            if tool.cache_ttl > 0 and cache_version == self._cache_versions.get(name, 0):
                self._set_cache(name, params, result, tool.cache_ttl, context)

            return result

        except (MCPTransportError, MCPProtocolError) as ex:
            tool.stats.failed += 1
            tool.stats.consecutive_fails += 1
            tool.breaker.record_failure()
            return ToolResult(False, None, name, error=str(ex), transport="stdio", execution_status="unknown")

        except asyncio.TimeoutError:
            tool.stats.failed += 1
            tool.stats.consecutive_fails += 1
            tool.breaker.record_failure()
            logger.error(f"工具超时: {name} ({tool.timeout_s}s)")
            if tool.transport == "stdio":
                return ToolResult(False, None, name, error="MCP 调用超时，执行结果未确认",
                                  transport="stdio", execution_status="unknown")
            return await self._fallback_result(tool, params, context, "执行超时", executed=True)

        except Exception as ex:
            tool.stats.failed += 1
            tool.stats.consecutive_fails += 1
            tool.breaker.record_failure()
            logger.error(f"工具异常: {name} — {ex}")
            if tool.transport == "stdio":
                return ToolResult(False, None, name, error="MCP 执行结果未确认",
                                  transport="stdio", execution_status="unknown")
            return await self._fallback_result(tool, params, context, str(ex), executed=True)

    async def _fallback_result(
        self,
        tool: Tool,
        params: Dict[str, Any],
        context: Optional[Dict[str, Any]],
        error: str,
        executed: bool = False,
    ) -> ToolResult:
        """工具不可用时返回降级结果，而不是把空错误直接暴露给调用方。"""
        if tool.fallback is None:
            return ToolResult(success=False, data=None, tool_name=tool.name, error=error, executed=executed,
                              transport=tool.transport, execution_status="failed" if executed else "rejected")
        try:
            data = tool.fallback(params, context, error)
            if asyncio.iscoroutine(data):
                data = await data
            return ToolResult(
                success=True,
                data=data,
                tool_name=tool.name,
                error=error,
                executed=executed,
            )
        except Exception as ex:
            logger.error(f"工具降级失败: {tool.name} — {ex}")
            return ToolResult(success=False, data=None, tool_name=tool.name, error=f"{error}; fallback失败: {ex}", executed=executed)

    async def _run_handler(
        self,
        tool: Tool,
        params: Dict[str, Any],
        context: Optional[Dict[str, Any]],
    ) -> Any:
        """
        执行工具 handler。

        优先支持 async handler；如果历史工具仍是同步函数，则放入线程池执行，
        避免阻塞事件循环。
        """
        if inspect.iscoroutinefunction(tool.handler):
            return await tool.handler(params, context)
        result = await asyncio.to_thread(tool.handler, params, context)
        if inspect.isawaitable(result):
            return await result
        return result

    # ── 缓存 ──────────────────────────────────────────────────────────────────

    def _invalidate_caches(self, names: tuple[str, ...]) -> None:
        for name in names:
            self._cache_versions[name] = self._cache_versions.get(name, 0) + 1
        for key in list(self._cache):
            if key[0] in names:
                del self._cache[key]

    def _cache_key(self, name: str, params: Dict, context=None) -> tuple[str, str]:
        scope = {k: v for k, v in context.items() if k != "tool_use_id"} if context else context
        payload = {"params": params, "context": scope}
        return name, hashlib.md5(json.dumps(payload, sort_keys=True).encode()).hexdigest()

    def _get_cache(self, name, params, context=None):
        key = self._cache_key(name, params, context)
        if key in self._cache:
            result, expires = self._cache[key]
            if time.monotonic() < expires:
                return copy.deepcopy(result)
            del self._cache[key]
        return None

    def _set_cache(self, name, params, result, ttl, context=None):
        if len(self._cache) >= 5000:
            for key in list(self._cache)[:1250]:
                del self._cache[key]
        self._cache[self._cache_key(name, params, context)] = (copy.deepcopy(result), time.monotonic() + ttl)

    def _validate_params(self, tool, params):
        try:
            validate_arguments(tool.schema, params)
        except ValidationError as ex:
            raise ValueError(ex.message) from ex

    @staticmethod
    def _clean_text(value: Any) -> str:
        """移除 Unicode 代理字符，避免 LLM 请求编码失败。"""
        if value is None:
            return ""
        if not isinstance(value, str):
            value = str(value)
        return value.encode("utf-8", errors="ignore").decode("utf-8")

    # ── 统计 ──────────────────────────────────────────────────────────────────

    def get_stats(self) -> Dict[str, Any]:
        return {
            name: {
                "total": t.stats.total,
                "success_rate": round(t.stats.success_rate, 3),
                "avg_latency_ms": round(t.stats.avg_latency_ms, 1),
                "consecutive_fails": t.stats.consecutive_fails,
                "circuit_state": t.breaker.state.value,
            }
            for name, t in self._tools.items()
        }
