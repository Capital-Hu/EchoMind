"""Explicit remote Chroma connection; evaluation never falls back to local storage."""
import hashlib
import os

import chromadb


def database_name(directory):
    return 'eval_' + hashlib.sha256(str(directory).encode()).hexdigest()[:32]


def connect(database, *, create=False):
    host = os.environ['ECHOMIND_EVAL_CHROMA_HOST']
    port = int(os.environ.get('ECHOMIND_EVAL_CHROMA_PORT', '8000'))
    settings = chromadb.Settings(anonymized_telemetry=False)
    if create:
        admin = chromadb.AdminClient(settings=chromadb.Settings(
            chroma_api_impl='chromadb.api.fastapi.FastAPI',
            chroma_server_host=host, chroma_server_http_port=port,
            anonymized_telemetry=False,
        ))
        admin.create_database(database)
    client = chromadb.HttpClient(host=host, port=port, database=database, settings=settings)
    client.heartbeat()
    return client
