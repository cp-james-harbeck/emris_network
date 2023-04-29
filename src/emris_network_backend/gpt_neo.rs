use rust_bert::gpt_neo::{GptNeoConfig, GptNeoForCausalLM, GptNeoGenerator};
use rust_bert::pipelines::generation::{GenerateConfig, LanguageGenerator};
use rust_bert::resources::{RemoteResource, Resource};
use rust_bert::Config;
use std::fs::File;
use std::io::Write;
use std::path::Path;
use tch::Device;
use tch::nn::VarStore;

pub struct GptNeoTextGenerator {
    pub generator: GptNeoGenerator,
}

impl GptNeoTextGenerator {
    pub fn new() -> anyhow::Result<Self> {
        let device = Device::cuda_if_available();
        let config_resource = Resource::Remote(RemoteResource::from_pretrained(
            GptNeoConfig::gpt_neo_125m(),
        ));
        let model_resource = Resource::Remote(RemoteResource::from_pretrained(
            GptNeoForCausalLM::gpt_neo_125m(),
        ));
        let config_path = config_resource.get_local_path()?;
        let weights_path = model_resource.get_local_path()?;
        let config = GptNeoConfig::from_file(config_path);
        let gpt_neo_model = GptNeoForCausalLM::from_file(VarStore::new(device), &weights_path, &config)?;
        let generator = GptNeoGenerator::new(gpt_neo_model);

        Ok(Self { generator })
    }

    pub fn generate_text(&self, prompt: &str) -> String {
        let generate_config = GenerateConfig {
            max_length: 50,
            do_sample: true,
            ..Default::default()
        };
        let output = self.generator.generate(Some(vec![prompt]), Some(generate_config));
        output.get(0).unwrap().to_string()
    }

    // Save the GPT-Neo model to ONNX format
    pub fn save_model_to_onnx(&self, path: impl AsRef<Path>) -> anyhow::Result<()> {
        let input_shape = &[1, 128];
        let onnx_bytes = self.generator.model.to_onnx(input_shape, true).unwrap();
        let mut file = File::create(path)?;
        file.write_all(&onnx_bytes)?;
        Ok(())
    }
}

