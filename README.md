# LLM 基础组件

这是一个基于 Go 语言的 LLM（大语言模型）基础组件库，提供了统一的接口来访问不同的 LLM 服务提供商。

## 主要功能

- 支持多个 LLM 提供商的统一接口
- 提供文本嵌入（Embedding）功能
- 支持聊天和文本补全功能
- 内置并发控制和错误处理
- 支持自定义模型和参数配置

## 当前支持的提供商

- Ollama
  - 默认嵌入模型: `mxbai-embed-large`
  - 支持自定义模型
  - 支持聊天和文本补全功能

## 安装

```bash
go get github.com/hewenyu/llm
```

## 使用示例

### 创建服务实例

```go
// 创建服务
service := llm.NewService()

// 创建 Ollama 提供商
provider, err := llm.NewOllamaProvider("http://localhost:11434")
if err != nil {
    log.Fatal(err)
}

// 注册提供商
err = service.RegisterProvider(provider)
if err != nil {
    log.Fatal(err)
}
```

### 文本嵌入

```go
// 创建嵌入器
embedder := llm.NewLLMEmbedder(service, "ollama", "mxbai-embed-large", 1536)

// 单个文本嵌入
embedding, err := embedder.Embed(context.Background(), "这是一个测试文本")
if err != nil {
    log.Fatal(err)
}

// 批量文本嵌入
texts := []interface{}{"文本1", "文本2", "文本3"}
embeddings, err := embedder.BatchEmbed(context.Background(), texts)
if err != nil {
    log.Fatal(err)
}
```

### 聊天功能

```go
request := llm.ChatRequest{
    Messages: []llm.Message{
        {
            Role:    "user",
            Content: "你好，请介绍一下自己",
        },
    },
    Temperature: 0.7,
}

response, err := service.Chat(context.Background(), "ollama", "qwen2.5", request)
if err != nil {
    log.Fatal(err)
}
```

### 文本补全

```go
request := llm.CompletionRequest{
    Prompt:      "从前有座山，山里有座庙，庙里有个",
    Temperature: 0.7,
}

response, err := service.Complete(context.Background(), "ollama", "qwen2.5", request)
if err != nil {
    log.Fatal(err)
}
```

## 测试结果

所有测试用例均已通过，包括：

- 基础功能测试
  - 服务创建和提供商注册
  - 嵌入器创建和配置
  - 并发池大小设置

- Ollama 提供商测试
  - 模型列表获取
  - 模型信息查询
  - 聊天功能
  - 文本补全
  - 文本嵌入（支持多种输入类型）

- 错误处理测试
  - 无效输入处理
  - 上下文超时处理
  - 并发控制

## 贡献

欢迎提交 Issue 和 Pull Request。

## 许可证

MIT License
