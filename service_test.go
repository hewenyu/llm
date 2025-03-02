package llm

import (
	"context"
	"fmt"
	"reflect"
	"sort"
	"testing"
)

// mockProvider 是一个模拟的 Provider 实现，用于测试
type mockProvider struct {
	name      string
	models    []ModelInfo
	embedFunc func(ctx context.Context, modelID string, request EmbeddingRequest) (EmbeddingResponse, error)
	chatFunc  func(ctx context.Context, modelID string, request ChatRequest) (ChatResponse, error)
}

func (m *mockProvider) Name() string {
	return m.name
}

func (m *mockProvider) ListModels(ctx context.Context) ([]ModelInfo, error) {
	return m.models, nil
}

func (m *mockProvider) GetModel(ctx context.Context, modelID string) (ModelInfo, error) {
	for _, model := range m.models {
		if model.Name == modelID {
			return model, nil
		}
	}
	return ModelInfo{}, fmt.Errorf("model %s not found", modelID)
}

func (m *mockProvider) Complete(ctx context.Context, modelID string, request CompletionRequest) (CompletionResponse, error) {
	return CompletionResponse{}, nil
}

func (m *mockProvider) Chat(ctx context.Context, modelID string, request ChatRequest) (ChatResponse, error) {
	if m.chatFunc != nil {
		return m.chatFunc(ctx, modelID, request)
	}
	return ChatResponse{}, nil
}

func (m *mockProvider) Embed(ctx context.Context, modelID string, request EmbeddingRequest) (EmbeddingResponse, error) {
	if m.embedFunc != nil {
		return m.embedFunc(ctx, modelID, request)
	}
	return EmbeddingResponse{}, nil
}

func (m *mockProvider) GetEmbedModel() string {
	return "mock-embed-model"
}

func TestNewService(t *testing.T) {
	svc := NewService()
	if svc == nil {
		t.Fatal("NewService returned nil")
	}
}

func TestRegisterProvider(t *testing.T) {
	svc := NewService()

	tests := []struct {
		name      string
		provider  Provider
		wantErr   bool
		errString string
	}{
		{
			name:     "valid provider",
			provider: &mockProvider{name: "test-provider"},
			wantErr:  false,
		},
		{
			name:      "nil provider",
			provider:  nil,
			wantErr:   true,
			errString: "provider cannot be nil",
		},
		{
			name:      "empty provider name",
			provider:  &mockProvider{name: ""},
			wantErr:   true,
			errString: "provider name cannot be empty",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			err := svc.RegisterProvider(tt.provider)
			if (err != nil) != tt.wantErr {
				t.Errorf("RegisterProvider() error = %v, wantErr %v", err, tt.wantErr)
				return
			}
			if tt.wantErr && err.Error() != tt.errString {
				t.Errorf("RegisterProvider() error = %v, want %v", err, tt.errString)
			}
		})
	}

	// Test registering duplicate provider
	provider := &mockProvider{name: "duplicate"}
	err := svc.RegisterProvider(provider)
	if err != nil {
		t.Fatalf("Failed to register first provider: %v", err)
	}

	err = svc.RegisterProvider(provider)
	if err == nil {
		t.Error("Expected error when registering duplicate provider, got nil")
	}
}

func TestGetProvider(t *testing.T) {
	svc := NewService()
	provider := &mockProvider{name: "test-provider"}
	_ = svc.RegisterProvider(provider)

	tests := []struct {
		name         string
		providerName string
		want         Provider
		wantErr      bool
	}{
		{
			name:         "existing provider",
			providerName: "test-provider",
			want:         provider,
			wantErr:      false,
		},
		{
			name:         "non-existent provider",
			providerName: "non-existent",
			want:         nil,
			wantErr:      true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := svc.GetProvider(tt.providerName)
			if (err != nil) != tt.wantErr {
				t.Errorf("GetProvider() error = %v, wantErr %v", err, tt.wantErr)
				return
			}
			if !tt.wantErr && got != tt.want {
				t.Errorf("GetProvider() = %v, want %v", got, tt.want)
			}
		})
	}
}

func TestListProviders(t *testing.T) {
	svc := NewService()
	providers := []struct {
		name string
	}{
		{"provider1"},
		{"provider2"},
		{"provider3"},
	}

	for _, p := range providers {
		_ = svc.RegisterProvider(&mockProvider{name: p.name})
	}

	got := svc.ListProviders()
	want := []string{"provider1", "provider2", "provider3"}

	sort.Strings(got)
	sort.Strings(want)

	if !reflect.DeepEqual(got, want) {
		t.Errorf("ListProviders() = %v, want %v", got, want)
	}
}

func TestListModels(t *testing.T) {
	svc := NewService()
	provider1Models := []ModelInfo{
		{Name: "model1", ContextWindowSize: 1024},
		{Name: "model2", ContextWindowSize: 2048},
	}
	provider2Models := []ModelInfo{
		{Name: "model3", ContextWindowSize: 4096},
	}

	_ = svc.RegisterProvider(&mockProvider{
		name:   "provider1",
		models: provider1Models,
	})
	_ = svc.RegisterProvider(&mockProvider{
		name:   "provider2",
		models: provider2Models,
	})

	ctx := context.Background()
	got, err := svc.ListModels(ctx)
	if err != nil {
		t.Fatalf("ListModels() error = %v", err)
	}

	want := map[string][]ModelInfo{
		"provider1": provider1Models,
		"provider2": provider2Models,
	}

	if !reflect.DeepEqual(got, want) {
		t.Errorf("ListModels() = %v, want %v", got, want)
	}
}

func TestChat(t *testing.T) {
	svc := NewService()
	expectedResponse := ChatResponse{
		Message: Message{
			Role:    "assistant",
			Content: "Hello!",
		},
	}

	provider := &mockProvider{
		name: "test-provider",
		chatFunc: func(ctx context.Context, modelID string, request ChatRequest) (ChatResponse, error) {
			return expectedResponse, nil
		},
	}
	_ = svc.RegisterProvider(provider)

	ctx := context.Background()
	request := ChatRequest{
		Messages: []Message{
			{Role: "user", Content: "Hi"},
		},
	}

	got, err := svc.Chat(ctx, "test-provider", "test-model", request)
	if err != nil {
		t.Fatalf("Chat() error = %v", err)
	}

	if !reflect.DeepEqual(got, expectedResponse) {
		t.Errorf("Chat() = %v, want %v", got, expectedResponse)
	}
}

func TestServiceEmbed(t *testing.T) {
	svc := NewService()
	expectedResponse := EmbeddingResponse{
		Embedding: []float64{0.1, 0.2, 0.3},
	}

	provider := &mockProvider{
		name: "test-provider",
		embedFunc: func(ctx context.Context, modelID string, request EmbeddingRequest) (EmbeddingResponse, error) {
			return expectedResponse, nil
		},
	}
	_ = svc.RegisterProvider(provider)

	ctx := context.Background()
	request := EmbeddingRequest{
		Input: "test text",
	}

	got, err := svc.Embed(ctx, "test-provider", "test-model", request)
	if err != nil {
		t.Fatalf("Embed() error = %v", err)
	}

	if !reflect.DeepEqual(got, expectedResponse) {
		t.Errorf("Embed() = %v, want %v", got, expectedResponse)
	}
}
