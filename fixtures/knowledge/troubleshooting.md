# EchoMind 接入排障演示

无法连接 EchoMind 时，先检查容器状态，再访问 `/health` 确认服务是否就绪。
如果知识库检索失败，检查 `/mcp/status` 和 `/knowledge/stats`，并查看应用日志。
多轮聊天需保持相同的 `user_id` 和 `conv_id`；重试同一条消息需沿用原 `client_request_id`。
