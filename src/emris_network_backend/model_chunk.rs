use ic_cdk::export::candid::{CandidType, Deserialize, Serialize};

// Define a struct representing a chunk of a larger data model.
#[derive(Clone, Deserialize, Serialize, CandidType)]
pub struct ModelChunk {
    pub id: String,      // Unique identifier for this chunk
    pub user_id: String, // Identifier of the user associated with this chunk
    pub data: Vec<u8>,   // Binary data representing the content of this chunk
                         // TODO: Consider adding metadata (e.g., timestamp, chunk size) to the struct.
                         // TODO: Implement logic for combining chunks to reconstruct the complete model.
}
