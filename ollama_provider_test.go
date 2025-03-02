package llm

import (
	"context"
	"encoding/json"
	"math"
	"net/http"
	"net/http/httptest"
	"net/url"
	"reflect"
	"strings"
	"testing"
	"time"

	"github.com/ollama/ollama/api"
)

func TestOllamaProvider(t *testing.T) {
	// 创建provider实例
	provider, err := NewOllamaProvider("http://localhost:11434")
	if err != nil {
		t.Fatalf("创建OllamaProvider失败: %v", err)
	}

	// 测试获取名称
	t.Run("测试获取提供者名称", func(t *testing.T) {
		name := provider.Name()
		if name != "ollama" {
			t.Errorf("期望名称为'ollama'，实际得到：%s", name)
		}
	})

	// 测试列出模型
	t.Run("测试列出可用模型", func(t *testing.T) {
		ctx := context.Background()
		models, err := provider.ListModels(ctx)
		if err != nil {
			t.Fatalf("列出模型失败: %v", err)
		}
		if len(models) == 0 {
			t.Error("没有找到可用模型")
		}
		// 打印找到的模型
		t.Logf("找到 %d 个模型", len(models))
		for _, model := range models {
			t.Logf("模型名称: %s", model.Name)
		}
	})

	// 获取第一个可用的模型名称
	ctx := context.Background()
	models, err := provider.ListModels(ctx)
	if err != nil || len(models) == 0 {
		t.Fatal("无法获取可用模型")
	}
	modelName := models[1].Name

	// 测试获取模型信息
	t.Run("测试获取模型信息", func(t *testing.T) {
		ctx := context.Background()
		modelInfo, err := provider.GetModel(ctx, modelName)
		if err != nil {
			t.Fatalf("获取模型信息失败: %v", err)
		}
		t.Logf("模型信息: %+v", modelInfo)
	})

	// 测试聊天功能
	t.Run("测试聊天功能", func(t *testing.T) {
		ctx := context.Background()
		request := ChatRequest{
			Messages: []Message{
				{
					Role:    "user",
					Content: "你好，请用一句话介绍自己",
				},
			},
			Temperature: 0.7,
		}

		response, err := provider.Chat(ctx, modelName, request)
		if err != nil {
			t.Fatalf("聊天请求失败: %v", err)
		}

		if response.Message.Content == "" {
			t.Error("聊天响应内容为空")
		} else {
			t.Logf("聊天响应: %s", response.Message.Content)
			t.Logf("Token使用情况: 提示词=%d, 补全=%d, 总计=%d",
				response.Usage.PromptTokens,
				response.Usage.CompletionTokens,
				response.Usage.TotalTokens)
		}
	})

	// 测试文本补全
	t.Run("测试文本补全", func(t *testing.T) {
		ctx := context.Background()
		request := CompletionRequest{
			Prompt:      "从前有座山，山里有座庙，庙里有个",
			Temperature: 0.7,
		}

		response, err := provider.Complete(ctx, modelName, request)
		if err != nil {
			t.Fatalf("文本补全请求失败: %v", err)
		}

		if response.Text == "" {
			t.Error("补全响应内容为空")
		} else {
			t.Logf("补全响应: %s", response.Text)
			t.Logf("Token使用情况: 提示词=%d, 补全=%d, 总计=%d",
				response.Usage.PromptTokens,
				response.Usage.CompletionTokens,
				response.Usage.TotalTokens)
		}
	})

	// 测试文本嵌入
	t.Run("测试文本嵌入", func(t *testing.T) {
		ctx := context.Background()

		testCases := []struct {
			name  string
			input string
		}{
			{
				name:  "短文本",
				input: "这是一个测试文本",
			},
			{
				name:  "长文本",
				input: "向量数据库是一种专门设计用于存储和高效检索向量（嵌入）的数据库系统。它可以用来存储文本、图像等数据的向量表示，并支持相似度搜索。",
			},
			{
				name:  "特殊字符",
				input: "测试!@#$%^&*()_+ 😊 换行\n制表符\t",
			},
			{
				name:  "多语言",
				input: "中文 English 日本語 한국어",
			},
		}

		for _, tc := range testCases {
			t.Run(tc.name, func(t *testing.T) {
				request := EmbeddingRequest{
					Input: tc.input,
				}

				response, err := provider.Embed(ctx, modelName, request)
				if err != nil {
					t.Fatalf("生成嵌入向量失败: %v", err)
				}

				// 验证向量维度
				if len(response.Embedding) == 0 {
					t.Error("嵌入向量为空")
				}

				// 验证向量值范围
				for i, v := range response.Embedding {
					if math.IsNaN(v) || math.IsInf(v, 0) {
						t.Errorf("向量第%d个元素是无效值: %v", i, v)
					}
				}

				t.Logf("输入文本长度: %d", len(tc.input))
				t.Logf("嵌入向量维度: %d", len(response.Embedding))
				t.Logf("Token使用情况: 提示词=%d, 总计=%d",
					response.Usage.PromptTokens,
					response.Usage.TotalTokens)

				// 验证相同输入得到相同向量
				response2, err := provider.Embed(ctx, modelName, request)
				if err != nil {
					t.Fatalf("重复生成嵌入向量失败: %v", err)
				}

				if !reflect.DeepEqual(response.Embedding, response2.Embedding) {
					t.Error("相同输入得到不同的嵌入向量")
				}
			})
		}
	})
}

// TestOllamaProviderError 测试错误情况
func TestOllamaProviderError(t *testing.T) {
	// 测试无效的模型名称
	t.Run("测试无效的模型名称", func(t *testing.T) {
		provider, err := NewOllamaProvider("http://localhost:11434")
		if err != nil {
			t.Fatalf("创建OllamaProvider失败: %v", err)
		}

		ctx := context.Background()
		_, err = provider.Chat(ctx, "不存在的模型", ChatRequest{
			Messages: []Message{
				{
					Role:    "user",
					Content: "你好",
				},
			},
		})
		if err == nil {
			t.Error("使用无效模型名称时期望得到错误，但没有")
		}
	})
}

func setupMockOllamaServer() (*httptest.Server, *OllamaProvider) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")

		// 检查上下文是否已取消
		if r.Context().Err() != nil {
			w.WriteHeader(http.StatusGatewayTimeout)
			json.NewEncoder(w).Encode(map[string]string{"error": "context deadline exceeded"})
			return
		}

		// 读取请求体
		var req struct {
			Input  string `json:"input"`
			Model  string `json:"model"`
			Prompt string `json:"prompt"`
		}
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			w.WriteHeader(http.StatusBadRequest)
			json.NewEncoder(w).Encode(map[string]string{"error": "invalid request body"})
			return
		}

		// 检查输入
		input := req.Input
		if input == "" {
			input = req.Prompt // 某些API使用prompt字段
		}
		if input == "" {
			w.WriteHeader(http.StatusBadRequest)
			json.NewEncoder(w).Encode(map[string]string{"error": "empty input is not allowed"})
			return
		}

		// 模拟处理延迟
		select {
		case <-r.Context().Done():
			w.WriteHeader(http.StatusGatewayTimeout)
			json.NewEncoder(w).Encode(map[string]string{"error": "context deadline exceeded"})
			return
		case <-time.After(100 * time.Millisecond):
			// 继续处理
		}

		switch r.URL.Path {
		case "/api/embeddings":
			// 检查模型
			if req.Model != "" && req.Model != "mxbai-embed-large" && !strings.Contains(req.Model, "custom") {
				w.WriteHeader(http.StatusBadRequest)
				json.NewEncoder(w).Encode(map[string]string{"error": "unsupported model"})
				return
			}
			json.NewEncoder(w).Encode(map[string]interface{}{
				"embedding": []float64{0.1, 0.2, 0.3},
				"usage": map[string]int{
					"prompt_tokens": len(input),
					"total_tokens":  len(input),
				},
			})
		case "/api/generate":
			json.NewEncoder(w).Encode(map[string]interface{}{
				"response": "test response",
				"usage": map[string]int{
					"prompt_tokens":     len(input),
					"completion_tokens": 10,
					"total_tokens":      len(input) + 10,
				},
			})
		default:
			w.WriteHeader(http.StatusNotFound)
			json.NewEncoder(w).Encode(map[string]string{"error": "unknown endpoint"})
		}
	}))

	serverURL, _ := url.Parse(server.URL)
	provider := &OllamaProvider{
		embedModel: "mxbai-embed-large",
		client:     api.NewClient(serverURL, server.Client()),
	}

	return server, provider
}

func TestOllamaProvider_GetEmbedModel(t *testing.T) {
	server, provider := setupMockOllamaServer()
	defer server.Close()

	got := provider.GetEmbedModel()
	want := "mxbai-embed-large"

	if got != want {
		t.Errorf("GetEmbedModel() = %v, want %v", got, want)
	}
}

func TestOllamaProvider_Embed(t *testing.T) {
	server, provider := setupMockOllamaServer()
	defer server.Close()

	ctx := context.Background()
	request := EmbeddingRequest{
		Input: "test text",
	}

	// 使用默认的嵌入模型
	embedModel := provider.GetEmbedModel()
	response, err := provider.Embed(ctx, embedModel, request)

	if err != nil {
		t.Errorf("Embed() error = %v", err)
		return
	}

	// 验证返回的嵌入向量
	expectedEmbedding := []float64{0.1, 0.2, 0.3}
	if len(response.Embedding) != len(expectedEmbedding) {
		t.Errorf("Embed() returned embedding of length %d, want %d", len(response.Embedding), len(expectedEmbedding))
	}
}

func TestOllamaProvider_EmbedWithCustomModel(t *testing.T) {
	server, provider := setupMockOllamaServer()
	defer server.Close()

	ctx := context.Background()
	request := EmbeddingRequest{
		Input: "test text",
		Model: "custom-embed-model", // 使用自定义模型
	}

	// 使用自定义模型进行嵌入
	_, err := provider.Embed(ctx, request.Model, request)

	// 即使是mock服务器，我们也应该能得到正确的响应
	if err != nil {
		t.Errorf("Embed() with custom model error = %v", err)
	}
}

func TestOllamaProvider_EmbedWithInvalidInput(t *testing.T) {
	server, provider := setupMockOllamaServer()
	defer server.Close()

	ctx := context.Background()
	request := EmbeddingRequest{
		Input: "", // 空输入
	}

	embedModel := provider.GetEmbedModel()
	_, err := provider.Embed(ctx, embedModel, request)

	if err == nil {
		t.Error("Expected error with empty input, got nil")
	}
}

func TestOllamaProvider_EmbedWithContext(t *testing.T) {
	server, provider := setupMockOllamaServer()
	defer server.Close()

	// 创建一个带有超时的上下文
	ctx, cancel := context.WithTimeout(context.Background(), 1*time.Second)
	defer cancel()

	request := EmbeddingRequest{
		Input: "test text",
		Metadata: map[string]interface{}{
			"test_key": "test_value",
		},
	}

	embedModel := provider.GetEmbedModel()
	response, err := provider.Embed(ctx, embedModel, request)

	// 检查上下文是否已经超时
	if ctx.Err() != nil {
		t.Errorf("Context error: %v", ctx.Err())
		return
	}

	// 验证响应
	if err != nil {
		t.Errorf("Embed() error = %v", err)
		return
	}

	expectedEmbedding := []float64{0.1, 0.2, 0.3}
	if len(response.Embedding) != len(expectedEmbedding) {
		t.Errorf("Embed() returned embedding of length %d, want %d", len(response.Embedding), len(expectedEmbedding))
	}
}
