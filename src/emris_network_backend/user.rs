use ic_cdk::export::candid::{CandidType};
use serde::{Deserialize, Serialize};

// Define a struct representing a user in the system.
#[derive(Clone, Deserialize, Serialize, CandidType)]

pub struct User {
    pub id: String,     // User's unique identifier
    pub resources: u64, // Number of resources owned by the user
    pub rewards: u64,   // Number of rewards earned by the user
    pub rate_limit_tokens: u64, // Number of rate limit tokens available to the user
                        // TODO: Consider adding additional fields, such as user's display name or email address.
                        // TODO: Implement rate limiting logic based on the `rate_limit_tokens` field.
}
