# frozen_string_literal: true

require_relative "lib/siglip2/version"

Gem::Specification.new do |spec|
  spec.name = "siglip2-rb"
  spec.version = Siglip2::VERSION
  spec.authors = ["Krzysztof Hasiński"]
  spec.email = ["krzysztof.hasinski@gmail.com"]

  spec.summary = "Google SigLIP2 embeddings using ONNX models"
  spec.description = "Ruby implementation of Google's SigLIP2 model for creating text and image embeddings. Uses ONNX models from HuggingFace onnx-community."
  spec.homepage = "https://github.com/khasinski/siglip2-rb"
  spec.license = "MIT"
  spec.required_ruby_version = ">= 3.0.0"

  spec.metadata["homepage_uri"] = spec.homepage
  spec.metadata["source_code_uri"] = spec.homepage
  spec.metadata["changelog_uri"] = "#{spec.homepage}/blob/main/CHANGELOG.md"

  gemspec = File.basename(__FILE__)
  spec.files = IO.popen(%w[git ls-files -z], chdir: __dir__, err: IO::NULL) do |ls|
    ls.readlines("\x0", chomp: true).reject do |f|
      (f == gemspec) ||
        f.start_with?(*%w[bin/ test/ spec/ features/ .git .github appveyor Gemfile])
    end
  end
  spec.bindir = "exe"
  spec.executables = spec.files.grep(%r{\Aexe/}) { |f| File.basename(f) }
  spec.require_paths = ["lib"]

  spec.add_dependency "onnxruntime", "~> 0.9"
  spec.add_dependency "net-http", "~> 0.6"
  spec.add_dependency "numo-narray", "~> 0.9"
  spec.add_dependency "mini_magick", "~> 5.0"
  spec.add_dependency "tokenizers", "~> 0.5"
end
