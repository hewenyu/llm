package llm

import (
	"context"
	"reflect"
	"strings"
	"testing"
)

// mockService 是一个模拟的 Service 实现，用于测试
type mockService struct {
	embedFunc func(ctx context.Context, provider, model string, request EmbeddingRequest) (EmbeddingResponse, error)
}

func (m *mockService) RegisterProvider(provider Provider) error {
	return nil
}

func (m *mockService) GetProvider(name string) (Provider, error) {
	return nil, nil
}

func (m *mockService) ListProviders() []string {
	return nil
}

func (m *mockService) ListModels(ctx context.Context) (map[string][]ModelInfo, error) {
	return nil, nil
}

func (m *mockService) GetModel(ctx context.Context, providerName, modelID string) (ModelInfo, error) {
	return ModelInfo{}, nil
}

func (m *mockService) Complete(ctx context.Context, providerName, modelID string, request CompletionRequest) (CompletionResponse, error) {
	return CompletionResponse{}, nil
}

func (m *mockService) Chat(ctx context.Context, providerName, modelID string, request ChatRequest) (ChatResponse, error) {
	return ChatResponse{}, nil
}

func (m *mockService) Embed(ctx context.Context, provider, model string, request EmbeddingRequest) (EmbeddingResponse, error) {
	if m.embedFunc != nil {
		return m.embedFunc(ctx, provider, model, request)
	}
	return EmbeddingResponse{}, nil
}

func TestNewLLMEmbedder(t *testing.T) {
	service := &mockService{}
	provider := "test-provider"
	model := "test-model"
	dimensions := 128

	embedder := NewLLMEmbedder(service, provider, model, dimensions)

	if embedder == nil {
		t.Fatal("NewLLMEmbedder returned nil")
	}

	if embedder.service != service {
		t.Error("service not set correctly")
	}

	if embedder.provider != provider {
		t.Error("provider not set correctly")
	}

	if embedder.model != model {
		t.Error("model not set correctly")
	}

	if embedder.dimensions != dimensions {
		t.Error("dimensions not set correctly")
	}

	if embedder.maxPoolSize != 10 {
		t.Error("default maxPoolSize not set correctly")
	}
}

func TestSetMaxPoolSize(t *testing.T) {
	embedder := NewLLMEmbedder(nil, "", "", 0)

	tests := []struct {
		name     string
		size     int
		expected int
	}{
		{"positive size", 5, 5},
		{"zero size", 0, 10},      // should not change from default
		{"negative size", -1, 10}, // should not change from default
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			embedder.SetMaxPoolSize(tt.size)
			if embedder.maxPoolSize != tt.expected {
				t.Errorf("SetMaxPoolSize(%d) = %d; want %d", tt.size, embedder.maxPoolSize, tt.expected)
			}
		})
	}
}

func TestEmbed(t *testing.T) {
	expectedEmbedding := []float64{0.1, 0.2, 0.3}
	mockSvc := &mockService{
		embedFunc: func(ctx context.Context, provider, model string, request EmbeddingRequest) (EmbeddingResponse, error) {
			return EmbeddingResponse{
				Embedding: expectedEmbedding,
			}, nil
		},
	}

	embedder := NewLLMEmbedder(mockSvc, "test-provider", "test-model", len(expectedEmbedding))

	tests := []struct {
		name        string
		input       interface{}
		want        []float64
		wantErr     bool
		errContains string
	}{
		{
			name:  "string input",
			input: "test text",
			want:  expectedEmbedding,
		},
		{
			name:  "bytes input",
			input: []byte("test text"),
			want:  expectedEmbedding,
		},
		{
			name:  "stringer input",
			input: stringerType("test text"),
			want:  expectedEmbedding,
		},
		{
			name:        "invalid input type",
			input:       123,
			wantErr:     true,
			errContains: "unsupported content type",
		},
	}

	ctx := context.Background()
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := embedder.Embed(ctx, tt.input)
			if (err != nil) != tt.wantErr {
				t.Errorf("Embed() error = %v, wantErr %v", err, tt.wantErr)
				return
			}
			if tt.wantErr {
				if err == nil || !strings.Contains(err.Error(), tt.errContains) {
					t.Errorf("Embed() error = %v, want error containing %q", err, tt.errContains)
				}
				return
			}
			if !reflect.DeepEqual(got, tt.want) {
				t.Errorf("Embed() = %v, want %v", got, tt.want)
			}
		})
	}
}

func TestBatchEmbed(t *testing.T) {
	expectedEmbedding := []float64{0.1, 0.2, 0.3}
	mockSvc := &mockService{
		embedFunc: func(ctx context.Context, provider, model string, request EmbeddingRequest) (EmbeddingResponse, error) {
			return EmbeddingResponse{
				Embedding: expectedEmbedding,
			}, nil
		},
	}

	embedder := NewLLMEmbedder(mockSvc, "test-provider", "test-model", len(expectedEmbedding))
	embedder.SetMaxPoolSize(2) // 设置较小的池大小以测试并发

	tests := []struct {
		name        string
		inputs      []interface{}
		want        [][]float64
		wantErr     bool
		errContains string
	}{
		{
			name:   "valid inputs",
			inputs: []interface{}{"text1", "text2", "text3"},
			want:   [][]float64{expectedEmbedding, expectedEmbedding, expectedEmbedding},
		},
		{
			name:        "invalid input type",
			inputs:      []interface{}{"text1", 123, "text3"},
			wantErr:     true,
			errContains: "unsupported content type",
		},
	}

	ctx := context.Background()
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := embedder.BatchEmbed(ctx, tt.inputs)
			if (err != nil) != tt.wantErr {
				t.Errorf("BatchEmbed() error = %v, wantErr %v", err, tt.wantErr)
				return
			}
			if tt.wantErr {
				if err == nil || !strings.Contains(err.Error(), tt.errContains) {
					t.Errorf("BatchEmbed() error = %v, want error containing %q", err, tt.errContains)
				}
				return
			}
			if !reflect.DeepEqual(got, tt.want) {
				t.Errorf("BatchEmbed() = %v, want %v", got, tt.want)
			}
		})
	}
}

// 辅助类型和函数
type stringerType string

func (s stringerType) String() string {
	return string(s)
}

// func contains(s, substr string) bool {
// 	return s != "" && substr != "" && s != substr && len(s) > len(substr) && s[len(s)-len(substr):] == substr
// }
