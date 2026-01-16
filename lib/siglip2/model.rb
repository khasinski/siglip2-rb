# frozen_string_literal: true

require "onnxruntime"
require "tokenizers"

module Siglip2
  class Model
    attr_reader :model_name, :quantization, :model_path

    # Image sizes for each model variant
    IMAGE_SIZES = {
      "base-patch16-224" => 224,
      "base-patch16-256" => 256,
      "base-patch16-384" => 384,
      "base-patch16-512" => 512,
      "base-patch32-256" => 256,
      "base-patch16-naflex" => 224,
      "large-patch16-256" => 256,
      "large-patch16-384" => 384,
      "large-patch16-512" => 512,
      "giant-opt-patch16-256" => 256,
      "giant-opt-patch16-384" => 384,
      "so400m-patch14-224" => 224,
      "so400m-patch14-384" => 384,
      "so400m-patch16-256" => 256,
      "so400m-patch16-384" => 384,
      "so400m-patch16-512" => 512
    }.freeze

    def initialize(model_name: Siglip2::DEFAULT_MODEL, quantization: Siglip2::DEFAULT_QUANTIZATION)
      @model_name = model_name
      @quantization = quantization
      @model_path = Siglip2.model_path(model_name, quantization: quantization)

      unless Siglip2.models_exist?(model_name, quantization: quantization)
        Siglip2.download_models(model_name, quantization: quantization)
      end

      @image_size = IMAGE_SIZES[model_name] || 224
      @image_preprocessor = ImagePreprocessor.new(size: @image_size)
    end

    def encode_text(text)
      input_ids, attention_mask = tokenize(text)

      output = text_model.run(
        nil,
        {
          "input_ids" => input_ids,
          "attention_mask" => attention_mask
        }
      )

      # Extract embeddings - output format depends on model
      embeddings = extract_embeddings(output)
      normalize_embeddings(embeddings)
    end

    def encode_image(image_path)
      pixel_values = @image_preprocessor.preprocess(image_path)

      output = vision_model.run(
        nil,
        {
          "pixel_values" => pixel_values
        }
      )

      # Extract embeddings
      embeddings = extract_embeddings(output)
      normalize_embeddings(embeddings)
    end

    def similarity(text, image_path)
      text_embedding = encode_text(text)
      image_embedding = encode_image(image_path)

      dot_product(text_embedding, image_embedding)
    end

    def batch_similarity(texts, image_paths)
      text_embeddings = texts.map { |t| encode_text(t) }
      image_embeddings = image_paths.map { |p| encode_image(p) }

      text_embeddings.map do |te|
        image_embeddings.map { |ie| dot_product(te, ie) }
      end
    end

    private

    def tokenizer
      @tokenizer ||= Tokenizers.from_file(File.join(@model_path, "tokenizer.json"))
    end

    def text_model
      @text_model ||= OnnxRuntime::Model.new(File.join(@model_path, "text_model.onnx"))
    end

    def vision_model
      @vision_model ||= OnnxRuntime::Model.new(File.join(@model_path, "vision_model.onnx"))
    end

    def tokenize(text)
      # SigLIP2 uses Gemma tokenizer - lowercase and add EOS
      processed_text = text.downcase

      encoding = tokenizer.encode(processed_text)
      input_ids = encoding.ids

      # Truncate or pad to max_length (64 is typical for SigLIP2)
      max_length = 64
      if input_ids.length > max_length
        input_ids = input_ids[0...max_length]
      end

      # Create attention mask (1 for real tokens, 0 for padding)
      attention_mask = Array.new(input_ids.length, 1)

      # Pad if necessary
      if input_ids.length < max_length
        padding_length = max_length - input_ids.length
        input_ids += Array.new(padding_length, 0)  # 0 is typically pad token
        attention_mask += Array.new(padding_length, 0)
      end

      # Return as 2D arrays (batch size = 1)
      [[input_ids], [attention_mask]]
    end

    def extract_embeddings(output)
      # ONNX output is typically a hash with output names as keys
      # SigLIP2 models output pooler_output or last_hidden_state
      result = output.first
      if result.is_a?(Hash)
        # Try common output names
        embedding = result["pooler_output"] || result["last_hidden_state"] || result.values.first
      else
        embedding = result
      end

      # Flatten to 1D array if needed
      embedding = embedding.flatten if embedding.is_a?(Array) && embedding[0].is_a?(Array)
      embedding
    end

    def normalize_embeddings(embeddings)
      norm = Math.sqrt(embeddings.map { |x| x * x }.sum)
      return embeddings if norm == 0

      embeddings.map { |x| x / norm }
    end

    def dot_product(a, b)
      a.zip(b).map { |x, y| x * y }.sum
    end
  end
end
