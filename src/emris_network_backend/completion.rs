use ic_cdk::export::candid::{CandidType};
use serde::{Deserialize, Serialize};


// Define a struct representing a text completion.
#[derive(Clone, Deserialize, Serialize, CandidType)]
pub struct Completion {
    pub prompt: String, // The input prompt that was provided to generate the completion
    pub generated_text: String, // The generated text based on the input prompt
                        // TODO: Consider adding additional fields, such as a timestamp or confidence score.
}