"""主应用的评测页面与服务入口。"""
import os
from pathlib import Path

import httpx
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import FileResponse, Response

router = APIRouter()


@router.get('/eval', include_in_schema=False)
async def page():
    return FileResponse(Path(__file__).with_name('evaluation.html'))


@router.api_route('/eval/api/{path:path}', methods=['GET', 'POST'], tags=['τ 评测'])
async def evaluation_api(path: str, request: Request):
    try:
        async with httpx.AsyncClient(base_url=os.environ['TAU_SERVICE_URL'], timeout=30) as client:
            upstream = await client.request(request.method, os.environ['TAU_SERVICE_URL'].rstrip('/') + '/' + path,
                params=request.query_params,
                content=await request.body(), headers={'Content-Type': 'application/json'})
    except httpx.RequestError as error:
        raise HTTPException(503, '评测服务不可达，请检查 Docker 中的 tau-eval 服务') from error
    headers = {name: upstream.headers[name] for name in ('content-type', 'content-disposition')
               if name in upstream.headers}
    return Response(content=upstream.content, status_code=upstream.status_code, headers=headers)
