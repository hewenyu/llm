package llm

import (
	"context"
)

// EmbeddingFuncFlot32 返回一个用于生成嵌入向量的函数
type EmbeddingFuncFlot32 func(ctx context.Context, text string) ([]float32, error)

// EmbeddingFuncFlot64 返回一个用于生成嵌入向量的函数
type EmbeddingFuncFlot64 func(ctx context.Context, text string) ([]float64, error)

// NewEmbeddingFunc 返回一个用于生成嵌入向量的函数
func NewEmbeddingFuncFlot32(provider Provider) EmbeddingFuncFlot32 {
	return func(ctx context.Context, text string) ([]float32, error) {
		response, err := provider.Embed(ctx, provider.GetEmbedModel(), EmbeddingRequest{Input: text})
		if err != nil {
			return nil, err
		}
		// 将[]float64转换为[]float32
		embedding := make([]float32, len(response.Embedding))
		for i, v := range response.Embedding {
			embedding[i] = float32(v)
		}
		return embedding, nil
	}
}

func NewEmbeddingFuncFlot64(provider Provider) EmbeddingFuncFlot64 {
	return func(ctx context.Context, text string) ([]float64, error) {
		response, err := provider.Embed(ctx, provider.GetEmbedModel(), EmbeddingRequest{Input: text})
		if err != nil {
			return nil, err
		}
		return response.Embedding, nil
	}
}
