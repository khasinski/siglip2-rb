# frozen_string_literal: true

require_relative "siglip2/version"
require_relative "siglip2/model"
require_relative "siglip2/image_preprocessor"

require "net/http"
require "uri"
require "fileutils"

module Siglip2
  class Error < StandardError; end

  # Available models from onnx-community on HuggingFace
  AVAILABLE_MODELS = {
    # Base models
    "base-patch16-224" => "onnx-community/siglip2-base-patch16-224-ONNX",
    "base-patch16-256" => "onnx-community/siglip2-base-patch16-256-ONNX",
    "base-patch16-384" => "onnx-community/siglip2-base-patch16-384-ONNX",
    "base-patch16-512" => "onnx-community/siglip2-base-patch16-512-ONNX",
    "base-patch32-256" => "onnx-community/siglip2-base-patch32-256-ONNX",
    "base-patch16-naflex" => "onnx-community/siglip2-base-patch16-naflex-ONNX",

    # Large models
    "large-patch16-256" => "onnx-community/siglip2-large-patch16-256-ONNX",
    "large-patch16-384" => "onnx-community/siglip2-large-patch16-384-ONNX",
    "large-patch16-512" => "onnx-community/siglip2-large-patch16-512-ONNX",

    # Giant optimized models
    "giant-opt-patch16-256" => "onnx-community/siglip2-giant-opt-patch16-256-ONNX",
    "giant-opt-patch16-384" => "onnx-community/siglip2-giant-opt-patch16-384-ONNX",

    # SO400M models
    "so400m-patch14-224" => "onnx-community/siglip2-so400m-patch14-224-ONNX",
    "so400m-patch14-384" => "onnx-community/siglip2-so400m-patch14-384-ONNX",
    "so400m-patch16-256" => "onnx-community/siglip2-so400m-patch16-256-ONNX",
    "so400m-patch16-384" => "onnx-community/siglip2-so400m-patch16-384-ONNX",
    "so400m-patch16-512" => "onnx-community/siglip2-so400m-patch16-512-ONNX"
  }.freeze

  DEFAULT_MODEL = "base-patch16-224"

  # Model quantization options
  QUANTIZATION_OPTIONS = %w[
    fp32
    fp16
    int8
    uint8
    q4
    q4f16
    bnb4
  ].freeze

  DEFAULT_QUANTIZATION = "fp32"

  class << self
    def models_dir
      @models_dir ||= File.join(Dir.home, ".siglip2_models")
    end

    def models_dir=(path)
      @models_dir = path
    end

    def model_path(model_name, quantization: DEFAULT_QUANTIZATION)
      raise Error, "Unknown model: #{model_name}" unless AVAILABLE_MODELS.key?(model_name)
      raise Error, "Unknown quantization: #{quantization}" unless QUANTIZATION_OPTIONS.include?(quantization)

      File.join(models_dir, model_name, quantization)
    end

    def models_exist?(model_name, quantization: DEFAULT_QUANTIZATION)
      path = model_path(model_name, quantization: quantization)
      File.exist?(File.join(path, "vision_model.onnx")) &&
        File.exist?(File.join(path, "text_model.onnx")) &&
        File.exist?(File.join(path, "tokenizer.json"))
    end

    def download_models(model_name, quantization: DEFAULT_QUANTIZATION)
      raise Error, "Unknown model: #{model_name}" unless AVAILABLE_MODELS.key?(model_name)

      repo = AVAILABLE_MODELS[model_name]
      path = model_path(model_name, quantization: quantization)
      FileUtils.mkdir_p(path)

      # Determine file suffix based on quantization
      suffix = quantization_suffix(quantization)

      files = {
        "vision_model.onnx" => "onnx/vision_model#{suffix}.onnx",
        "text_model.onnx" => "onnx/text_model#{suffix}.onnx",
        "tokenizer.json" => "tokenizer.json"
      }

      files.each do |local_name, remote_path|
        local_path = File.join(path, local_name)
        next if File.exist?(local_path)

        url = "https://huggingface.co/#{repo}/resolve/main/#{remote_path}"
        puts "Downloading #{local_name} from #{url}..."
        download_file(url, local_path)
      end
    end

    def list_models
      AVAILABLE_MODELS.keys
    end

    def list_quantizations
      QUANTIZATION_OPTIONS
    end

    private

    def quantization_suffix(quantization)
      case quantization
      when "fp32" then ""
      when "fp16" then "_fp16"
      when "int8" then "_int8"
      when "uint8" then "_uint8"
      when "q4" then "_q4"
      when "q4f16" then "_q4f16"
      when "bnb4" then "_bnb4"
      else ""
      end
    end

    def download_file(url, path, redirect_limit = 10)
      raise Error, "Too many HTTP redirects" if redirect_limit == 0

      uri = URI.parse(url)
      http = Net::HTTP.new(uri.host, uri.port)
      http.use_ssl = uri.scheme == "https"
      http.read_timeout = 300

      request = Net::HTTP::Get.new(uri)

      http.request(request) do |response|
        case response
        when Net::HTTPSuccess
          File.open(path, "wb") do |file|
            response.read_body do |chunk|
              file.write(chunk)
            end
          end
        when Net::HTTPRedirection
          download_file(response["location"], path, redirect_limit - 1)
        else
          raise Error, "Failed to download #{url}: #{response.code} #{response.message}"
        end
      end
    end
  end
end
