use std::fs::File;
use std::io::Write;
use std::path::Path;
use tch::nn::{self, ModuleT};
use tch::Device;
use transformers::{
    BertConfig, BertForSequenceClassification, Config, Model, OptimizerBuilder, Tokenizer, TrainerBuilder,
};
use clap::{Arg, App};

// Save the fine-tuned BERT model to ONNX format
fn save_model_to_onnx(model: &BertForSequenceClassification, path: impl AsRef<Path>) {
    let input_shape = &[1, 128];
    let onnx_bytes = model.to_onnx(input_shape, true).unwrap();
    let mut file = File::create(path).unwrap();
    file.write_all(&onnx_bytes).unwrap();
}

fn main() {
    // Parse command-line arguments
    let matches = App::new("Fine-tuning BERT")
        .arg(Arg::with_name("config_path")
            .long("config")
            .value_name("CONFIG")
            .help("Path to the BERT model configuration file")
            .required(true)
            .takes_value(true))
        .arg(Arg::with_name("tokenizer_path")
            .long("tokenizer")
            .value_name("TOKENIZER")
            .help("Path to the BERT tokenizer")
            .required(true)
            .takes_value(true))
        .arg(Arg::with_name("dataset_path")
            .long("dataset")
            .value_name("DATASET")
            .help("Path to the dataset for fine-tuning")
            .required(true)
            .takes_value(true))
        .arg(Arg::with_name("learning_rate")
            .long("lr")
            .value_name("LEARNING_RATE")
            .help("Learning rate for the optimizer")
            .default_value("2e-5")
            .takes_value(true))
        .arg(Arg::with_name("num_epochs")
            .long("epochs")
            .value_name("NUM_EPOCHS")
            .help("Number of epochs for fine-tuning")
            .default_value("3")
            .takes_value(true))
        .arg(Arg::with_name("model_save_path")
            .long("output")
            .value_name("OUTPUT")
            .help("Path to save the fine-tuned BERT model in ONNX format")
            .required(true)
            .takes_value(true))
        .get_matches();

    // Set device (use CUDA if available)
    let device = Device::cuda_if_available();

    // Load the BERT model configuration
    let config_path = matches.value_of("config_path").unwrap();
    let config = BertConfig::from_file(config_path);

    // Load the BERT tokenizer
    let tokenizer_path = matches.value_of("tokenizer_path").unwrap();
    let tokenizer = Tokenizer::from_pretrained(tokenizer_path, None);

    // Load the BERT model for sequence classification
    let vs = nn::VarStore::new(device);
    let bert_model = BertForSequenceClassification::new(&vs.root(), &config);

    // Prepare the dataset for fine-tuning
    let dataset_path = matches.value_of("dataset_path").unwrap();
    let dataset = load_and_preprocess_dataset(&dataset_path, &tokenizer);

    // Set up the optimizer
    let learning_rate: f64 = matches.value_of("learning_rate").unwrap().parse().unwrap();
    let optimizer = OptimizerBuilder::from(&bert_model)
        .with_learning_rate(learning_rate)
        .build();

    // Set up the trainer
    let num_epochs: u64 = matches.value_of("num_epochs").unwrap().parse().unwrap();
    let mut trainer = TrainerBuilder::from(&bert_model, &dataset, &optimizer)
        .with_epochs(num_epochs)
        .with_device(device)
        .build();

    // Fine-tune the model
    trainer.train();

    // Save the fine-tuned BERT model to ONNX format
    let model_save_path = matches.value_of("model_save_path").unwrap();
    save_model_to_onnx(&bert_model, model_save_path);

    println!("Fine-tuning complete! Model saved to {:?}", model_save_path);
}



