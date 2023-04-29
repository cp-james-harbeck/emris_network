use std::fs::File;
use std::io::BufReader;
use serde_json::Value;
use tch::Tensor;

fn load_and_preprocess_dataset(json_file_path: &str, tokenizer: &BertTokenizer) -> Dataset {
    let file = File::open(json_file_path).expect("Unable to open JSON file");
    let reader = BufReader::new(file);
    let json_data: Value = serde_json::from_reader(reader).expect("Unable to parse JSON data");

    let mut input_ids = Vec::new();
    let mut attention_masks = Vec::new();
    let mut labels = Vec::new();

    if let Some(data) = json_data.as_array() {
        for item in data {
            if let Some(text) = item["text"].as_str() {
                if let Some(label) = item["label"].as_i64() {
                    let encoding = tokenizer.encode(text, None, 128, false, false);
                    input_ids.push(encoding.ids);
                    attention_masks.push(encoding.attention_mask);
                    labels.push(label);
                }
            }
        }
    }

    let input_ids = Tensor::of_slice(&input_ids).view([-1, 128]);
    let attention_masks = Tensor::of_slice(&attention_masks).view([-1, 128]);
    let labels = Tensor::of_slice(&labels).view([-1]);

    Dataset {
        input_ids,
        attention_masks,
        labels,
    }
}
