# SigLIP2-rb

Ruby implementation of Google's SigLIP2 (Sigmoid Loss for Language Image Pre-Training 2) for creating text and image embeddings. Uses ONNX models from HuggingFace [onnx-community](https://huggingface.co/onnx-community).

## Installation

Add this line to your application's Gemfile:

```ruby
gem 'siglip2-rb'
```

Or install directly:

```bash
gem install siglip2-rb
```

### Requirements

- Ruby >= 3.0.0
- ImageMagick (for image processing)

## Usage

### Ruby API

```ruby
require 'siglip2'

# Create model with default settings (base-patch16-224)
model = Siglip2::Model.new

# Or specify a different model and quantization
model = Siglip2::Model.new(
  model_name: "large-patch16-256",
  quantization: "int8"
)

# Encode text
text_embedding = model.encode_text("a photo of a cat")

# Encode image
image_embedding = model.encode_image("cat.jpg")

# Calculate similarity
score = model.similarity("a photo of a cat", "cat.jpg")
puts "Similarity: #{score}"

# Batch similarity
texts = ["a cat", "a dog", "a car"]
images = ["image1.jpg", "image2.jpg"]
scores = model.batch_similarity(texts, images)
```

### CLI Tools

#### Embed text

```bash
siglip2-embed-text "a photo of a cat"
siglip2-embed-text -m large-patch16-256 "a photo of a cat"
siglip2-embed-text -q int8 "a photo of a cat"
siglip2-embed-text -f csv "a photo of a cat"
```

#### Embed image

```bash
siglip2-embed-image cat.jpg
siglip2-embed-image -m large-patch16-256 cat.jpg
siglip2-embed-image -q int8 cat.jpg
```

#### Calculate similarity

```bash
siglip2-similarity "a photo of a cat" cat.jpg
```

#### List available models

```bash
siglip2-embed-text -l
```

#### List quantization options

```bash
siglip2-embed-text -L
```

## Available Models

| Model | Image Size | Description |
|-------|------------|-------------|
| `base-patch16-224` | 224x224 | Default, smallest |
| `base-patch16-256` | 256x256 | |
| `base-patch16-384` | 384x384 | |
| `base-patch16-512` | 512x512 | |
| `base-patch32-256` | 256x256 | Larger patch size |
| `base-patch16-naflex` | 224x224 | Flexible resolution |
| `large-patch16-256` | 256x256 | Larger model |
| `large-patch16-384` | 384x384 | |
| `large-patch16-512` | 512x512 | |
| `giant-opt-patch16-256` | 256x256 | Optimized giant |
| `giant-opt-patch16-384` | 384x384 | |
| `so400m-patch14-224` | 224x224 | 400M parameters |
| `so400m-patch14-384` | 384x384 | |
| `so400m-patch16-256` | 256x256 | |
| `so400m-patch16-384` | 384x384 | |
| `so400m-patch16-512` | 512x512 | |

## Quantization Options

| Option | Description |
|--------|-------------|
| `fp32` | Full precision (default) |
| `fp16` | Half precision |
| `int8` | 8-bit integer |
| `uint8` | Unsigned 8-bit integer |
| `q4` | 4-bit quantization |
| `q4f16` | 4-bit with fp16 |
| `bnb4` | BitsAndBytes 4-bit |

## Model Storage

Models are automatically downloaded on first use and stored in `~/.siglip2_models/`. You can change this location:

```ruby
Siglip2.models_dir = "/path/to/models"
```

## License

MIT License
