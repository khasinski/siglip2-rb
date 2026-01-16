# frozen_string_literal: true

require "mini_magick"
require "numo/narray"

module Siglip2
  class ImagePreprocessor
    # SigLIP2 normalization constants
    # From preprocessor_config.json: mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]
    MEAN = [0.5, 0.5, 0.5].freeze
    STD = [0.5, 0.5, 0.5].freeze

    attr_reader :size

    def initialize(size: 224)
      @size = size
    end

    def preprocess(image_path)
      image = load_and_resize(image_path)
      tensor = image_to_tensor(image)
      tensor = normalize(tensor)
      add_batch_dimension(tensor)
    end

    private

    def load_and_resize(image_path)
      image = MiniMagick::Image.open(image_path)
      image.format("png")
      image.resize("#{@size}x#{@size}!")
      image.colorspace("sRGB")
      image
    end

    def image_to_tensor(image)
      # Get raw pixel data as RGB
      pixels = image.get_pixels

      # Convert to Numo::SFloat array
      # Shape: [height, width, 3]
      height = pixels.length
      width = pixels[0].length

      flat_pixels = pixels.flatten.map(&:to_f)
      tensor = Numo::SFloat.cast(flat_pixels)
      tensor = tensor.reshape(height, width, 3)

      # Rescale from [0, 255] to [0, 1]
      tensor = tensor / 255.0

      # Transpose from [H, W, C] to [C, H, W]
      tensor = tensor.transpose(2, 0, 1)

      tensor
    end

    def normalize(tensor)
      # Apply normalization: (x - mean) / std
      # For SigLIP2: (x - 0.5) / 0.5 = 2x - 1, maps [0,1] to [-1,1]
      result = Numo::SFloat.zeros(tensor.shape)

      3.times do |c|
        result[c, true, true] = (tensor[c, true, true] - MEAN[c]) / STD[c]
      end

      result
    end

    def add_batch_dimension(tensor)
      # Convert to nested Ruby arrays with batch dimension
      # Shape: [1, 3, height, width]
      [tensor.to_a]
    end
  end
end
