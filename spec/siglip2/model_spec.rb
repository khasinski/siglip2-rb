# frozen_string_literal: true

require "spec_helper"
require "json"

RSpec.describe Siglip2::Model do
  let(:model) { described_class.new }
  let(:reference_data) { JSON.parse(File.read("spec/fixtures/reference_embeddings.json")) }
  let(:cat_image) { "spec/fixtures/cat.jpg" }
  let(:dog_image) { "spec/fixtures/dog.jpg" }

  describe "#initialize" do
    it "creates model with default settings" do
      expect(model.model_name).to eq("base-patch16-224")
      expect(model.quantization).to eq("fp32")
    end

    it "accepts custom model name" do
      # Skip if model not downloaded
      skip "Large model not downloaded" unless Siglip2.models_exist?("large-patch16-256")
      custom = described_class.new(model_name: "large-patch16-256")
      expect(custom.model_name).to eq("large-patch16-256")
    end
  end

  describe "#encode_text" do
    it "returns embedding of size 768" do
      embedding = model.encode_text("a photo of a cat")
      expect(embedding.length).to eq(768)
    end

    it "returns normalized embedding" do
      embedding = model.encode_text("a photo of a cat")
      norm = Math.sqrt(embedding.map { |x| x * x }.sum)
      expect(norm).to be_within(0.001).of(1.0)
    end

    it "produces consistent embeddings for same input" do
      emb1 = model.encode_text("a photo of a cat")
      emb2 = model.encode_text("a photo of a cat")
      expect(emb1).to eq(emb2)
    end

    it "matches reference embedding for 'a photo of a cat'" do
      embedding = model.encode_text("a photo of a cat")
      ref = reference_data["text_embeddings"]["a photo of a cat"]

      expect(embedding.length).to eq(ref["size"])
      ref["first_5"].each_with_index do |val, i|
        expect(embedding[i]).to be_within(0.0001).of(val)
      end
    end

    it "matches reference embedding for 'a photo of a dog'" do
      embedding = model.encode_text("a photo of a dog")
      ref = reference_data["text_embeddings"]["a photo of a dog"]

      expect(embedding.length).to eq(ref["size"])
      ref["first_5"].each_with_index do |val, i|
        expect(embedding[i]).to be_within(0.0001).of(val)
      end
    end
  end

  describe "#encode_image" do
    it "returns embedding of size 768" do
      embedding = model.encode_image(cat_image)
      expect(embedding.length).to eq(768)
    end

    it "returns normalized embedding" do
      embedding = model.encode_image(cat_image)
      norm = Math.sqrt(embedding.map { |x| x * x }.sum)
      expect(norm).to be_within(0.001).of(1.0)
    end

    it "produces similar embedding for cat image across platforms" do
      embedding = model.encode_image(cat_image)
      ref = reference_data["image_embeddings"]["cat"]

      expect(embedding.length).to eq(ref["size"])
      # Higher tolerance due to ImageMagick differences across platforms
      ref["first_5"].each_with_index do |val, i|
        expect(embedding[i]).to be_within(0.01).of(val)
      end
    end

    it "produces similar embedding for dog image across platforms" do
      embedding = model.encode_image(dog_image)
      ref = reference_data["image_embeddings"]["dog"]

      expect(embedding.length).to eq(ref["size"])
      # Higher tolerance due to ImageMagick differences across platforms
      ref["first_5"].each_with_index do |val, i|
        expect(embedding[i]).to be_within(0.01).of(val)
      end
    end
  end

  describe "#similarity" do
    it "returns higher score for matching text-image pair (cat)" do
      cat_score = model.similarity("a photo of a cat", cat_image)
      dog_score = model.similarity("a photo of a cat", dog_image)
      expect(cat_score).to be > dog_score
    end

    it "returns higher score for matching text-image pair (dog)" do
      cat_score = model.similarity("a photo of a dog", cat_image)
      dog_score = model.similarity("a photo of a dog", dog_image)
      expect(dog_score).to be > cat_score
    end

    it "produces similar scores across platforms" do
      ref = reference_data["similarities"]

      # Higher tolerance due to ImageMagick differences across platforms
      expect(model.similarity("a photo of a cat", cat_image))
        .to be_within(0.02).of(ref["cat_text_cat_image"])

      expect(model.similarity("a photo of a cat", dog_image))
        .to be_within(0.02).of(ref["cat_text_dog_image"])

      expect(model.similarity("a photo of a dog", cat_image))
        .to be_within(0.02).of(ref["dog_text_cat_image"])

      expect(model.similarity("a photo of a dog", dog_image))
        .to be_within(0.02).of(ref["dog_text_dog_image"])
    end
  end

  describe "#batch_similarity" do
    it "returns matrix of similarity scores" do
      texts = ["a photo of a cat", "a photo of a dog"]
      images = [cat_image, dog_image]

      scores = model.batch_similarity(texts, images)

      expect(scores.length).to eq(2)
      expect(scores[0].length).to eq(2)

      # Cat text should match cat image better
      expect(scores[0][0]).to be > scores[0][1]
      # Dog text should match dog image better
      expect(scores[1][1]).to be > scores[1][0]
    end
  end
end
