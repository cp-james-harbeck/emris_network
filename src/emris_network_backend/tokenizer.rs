use std::collections::HashMap;
use std::fs::File;
use std::io::BufReader;
use tokenizers::models::bpe::BpeBuilder;
use tokenizers::pre_tokenizers::whitespace::Whitespace;
use tokenizers::tokenizer::{AddedToken, Tokenizer};

pub struct SimpleTokenizer {
    tokenizer: Tokenizer,
}

impl SimpleTokenizer {
    pub fn new() -> Self {
        let mut tokenizer = Tokenizer::new();
        let file = File::open("models/tokenization/simple_tokens.json").unwrap();
        let reader = BufReader::new(file);
        let encoder: HashMap<String, u32> = serde_json::from_reader(reader).unwrap();

        let mut decoder: HashMap<u32, String> = HashMap::new();
        for (key, value) in encoder {
            decoder.insert(value, key);
        }

        tokenizer.add_tokens(
            decoder
                .iter()
                .map(|(_, token)| AddedToken::from(String::from(token)))
                .collect::<Vec<AddedToken>>(),
        );

        tokenizer.with_pre_tokenizer(Whitespace::default());

        SimpleTokenizer { tokenizer }
    }

    pub fn encode(&self, input: &str) -> Vec<u32> {
        self.tokenizer.encode(input, true).unwrap().get_ids()
    }

    pub fn decode(&self, input: &[u32]) -> String {
        self.tokenizer.decode(input.to_vec(), true).unwrap()
    }
}

pub struct GPT2Tokenizer {
    tokenizer: Tokenizer,
}

impl GPT2Tokenizer {
    pub fn new() -> Self {
        let mut tokenizer = Tokenizer::new();

        let bpe = BpeBuilder::from_files("models/tokenization/vocab.bpe")
            .unwrap()
            .dropout(None)
            .build()
            .unwrap();

        tokenizer.with_model(bpe);
        tokenizer.with_pre_tokenizer(Whitespace::default());

        GPT2Tokenizer { tokenizer }
    }

    pub fn encode(&self, input: &str) -> Vec<u32> {
        self.tokenizer.encode(input, true).unwrap().get_ids()
    }

    pub fn decode(&self, input: &[u32]) -> String {
        self.tokenizer.decode(input.to_vec(), true).unwrap()
    }
}
