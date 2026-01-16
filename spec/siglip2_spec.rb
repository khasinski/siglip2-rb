# frozen_string_literal: true

require "spec_helper"

RSpec.describe Siglip2 do
  describe ".list_models" do
    it "returns available model names" do
      models = Siglip2.list_models
      expect(models).to include("base-patch16-224")
      expect(models).to include("large-patch16-256")
      expect(models).to include("so400m-patch14-384")
    end
  end

  describe ".list_quantizations" do
    it "returns available quantization options" do
      quants = Siglip2.list_quantizations
      expect(quants).to include("fp32")
      expect(quants).to include("fp16")
      expect(quants).to include("int8")
    end
  end

  describe ".model_path" do
    it "returns path for valid model" do
      path = Siglip2.model_path("base-patch16-224")
      expect(path).to include("base-patch16-224")
      expect(path).to include("fp32")
    end

    it "raises error for unknown model" do
      expect { Siglip2.model_path("unknown-model") }.to raise_error(Siglip2::Error)
    end

    it "raises error for unknown quantization" do
      expect { Siglip2.model_path("base-patch16-224", quantization: "invalid") }.to raise_error(Siglip2::Error)
    end
  end
end
