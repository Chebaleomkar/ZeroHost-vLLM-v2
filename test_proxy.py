"""Local check of the load-balancing proxy against two fake vLLM upstreams: python test_proxy.py"""
import asyncio
import json
import tempfile

from aiohttp import ClientSession, web

import zerohost_v2 as zh

KEY = "test-key"
AUTH = {"Authorization": f"Bearer {KEY}"}
A, B, DEAD = 18001, 18002, 18009


def fake_vllm(port):
    async def handler(request):
        body = await request.json()
        await asyncio.sleep(body.get("delay", 0))
        leaked = "x-forwarded-for" in {k.lower() for k in request.headers}
        if body.get("stream"):
            resp = web.StreamResponse(headers={"Content-Type": "text/event-stream"})
            await resp.prepare(request)
            for piece in ["Hel", "lo"]:
                await resp.write(f'data: {json.dumps({"choices": [{"delta": {"content": piece}}]})}\n\n'.encode())
            await resp.write(f'data: {json.dumps({"choices": [{"delta": {}, "finish_reason": "stop"}]})}\n\ndata: [DONE]\n\n'.encode())
            return resp
        return web.json_response({"choices": [{"message": {"content": f"{port}|{leaked}"}, "finish_reason": "stop"}],
                                  "usage": {"prompt_tokens": 5, "completion_tokens": 2}})
    return handler


async def start_fake(port):
    app = web.Application()
    app.router.add_route("*", "/{tail:.*}", fake_vllm(port))
    runner = web.AppRunner(app)
    await runner.setup()
    await web.TCPSite(runner, "127.0.0.1", port).start()


def upstreams(*ports):
    return [{"port": p, "inflight": 0, "down_until": 0.0} for p in ports]


async def main():
    zh.DATA_DIR = tempfile.mkdtemp()
    zh.PROXY_PORT = 18000
    await start_fake(A)
    await start_fake(B)
    zh.STATE.update(mode="dp", upstreams=upstreams(A, B))
    asyncio.create_task(zh.serve_proxy(KEY))
    await asyncio.sleep(0.5)

    url = f"http://127.0.0.1:{zh.PROXY_PORT}/v1/chat/completions"
    req = {"model": "m", "messages": [{"role": "user", "content": "hi"}]}

    async def ask(s, extra=None, headers=None):
        async with s.post(url, json={**req, **(extra or {})}, headers={**AUTH, **(headers or {})}) as r:
            return r.status, (await r.json())["choices"][0]["message"]["content"] if r.status == 200 else None

    async with ClientSession() as s:
        async with s.post(url, json=req) as r:
            assert r.status == 401, r.status

        # least outstanding: while one engine holds a slow request, new requests go to the other one
        slow = asyncio.create_task(ask(s, {"delay": 1.0}))
        await asyncio.sleep(0.2)
        busy = zh.STATE["upstreams"][0]["port"] if zh.STATE["upstreams"][0]["inflight"] else zh.STATE["upstreams"][1]["port"]
        fast = [await ask(s) for _ in range(3)]
        assert all(int(out.split("|")[0]) != busy for _, out in fast), (busy, fast)
        await slow

        # client IP headers are stripped before reaching vLLM (so it can't log them)
        _, out = await ask(s, headers={"X-Forwarded-For": "203.0.113.7"})
        assert out.endswith("|False"), out

        # streaming through the balancer is traced with mode and upstream
        async with s.post(url, json={**req, "stream": True}, headers=AUTH) as r:
            assert "Hel" in await r.text()

        # failover: a dead engine is skipped and marked down, every request still succeeds
        zh.STATE["upstreams"] = upstreams(DEAD, B)
        results = [await ask(s) for _ in range(4)]
        assert all(status == 200 and out.startswith(str(B)) for status, out in results), results
        assert zh.STATE["upstreams"][0]["down_until"] > 0

        # no upstreams at all (between modes) -> 503, not a crash
        zh.STATE["upstreams"] = []
        status, _ = await ask(s)
        assert status == 503, status

    traces = [json.loads(l) for l in open(f"{zh.DATA_DIR}/requests.jsonl")]
    stream = next(t for t in traces if (t.get("request") or {}).get("stream"))
    assert stream["output"] == "Hello" and stream["mode"] == "dp" and stream["upstream"] in (A, B), stream
    assert sum(t.get("failovers", 0) for t in traces) >= 1
    print(f"all proxy checks passed ({len(traces)} traces)")


asyncio.run(main())
