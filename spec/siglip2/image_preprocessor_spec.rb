# frozen_string_literal: true

require "spec_helper"

RSpec.describe Siglip2::ImagePreprocessor do
  let(:preprocessor) { described_class.new(size: 224) }
  let(:cat_image) { "spec/fixtures/cat.jpg" }

  describe "#initialize" do
    it "sets default size to 224" do
      default = described_class.new
      expect(default.size).to eq(224)
    end

    it "accepts custom size" do
      custom = described_class.new(size: 384)
      expect(custom.size).to eq(384)
    end
  end

  describe "#preprocess" do
    it "returns array with batch dimension" do
      result = preprocessor.preprocess(cat_image)
      expect(result).to be_an(Array)
      expect(result.length).to eq(1) # batch size = 1
    end

    it "returns tensor with correct shape [1, 3, 224, 224]" do
      result = preprocessor.preprocess(cat_image)
      expect(result.length).to eq(1)           # batch
      expect(result[0].length).to eq(3)        # channels
      expect(result[0][0].length).to eq(224)   # height
      expect(result[0][0][0].length).to eq(224) # width
    end

    it "normalizes pixel values to [-1, 1] range" do
      result = preprocessor.preprocess(cat_image)
      flat = result.flatten

      # After normalization with mean=0.5, std=0.5: (x - 0.5) / 0.5
      # Input [0, 1] maps to [-1, 1]
      expect(flat.min).to be >= -1.0
      expect(flat.max).to be <= 1.0
    end
  end
end
