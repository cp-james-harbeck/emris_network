use ic_cdk::export::candid::{CandidType};
use serde::{Deserialize, Serialize};

// Define a struct representing a training task for a machine learning model.
#[derive(Clone, Deserialize, Serialize, CandidType)]
pub struct TrainingTask {
    pub id: String,             // Unique identifier for the training task
    pub model_id: String,       // Identifier of the model associated with the training task
    pub training_data: Vec<u8>, // Training data used for the task (binary format)
    pub model_weights: Option<Vec<u8>>, // Optional model weights (binary format)
                                // TODO: Consider adding fields for training hyperparameters (e.g., learning rate, batch size).
                                // TODO: Implement logic for training the model using the provided training data and model weights.
}
