use crate::completion::Completion;
use crate::model_chunk::ModelChunk;
use crate::task_manager_impl::Model;
use crate::training_task::TrainingTask;
use crate::user::User;
// use ic_cdk::export::candid::CandidType;
// use std::collections::HashMap;

pub trait TaskManagerInterface {
    fn init(&mut self);
    fn register_user(&mut self, user: User) -> Result<String, String>;
    fn update_user_resources(&mut self, id: &str, resources: u64) -> Result<(), String>;
    fn distribute_model_chunks(&mut self) -> Result<(), String>;
    fn submit_computed_chunk(&mut self, chunk: ModelChunk) -> Result<(), String>;
    fn get_model_chunks(&self, user_id: &str) -> Result<Vec<ModelChunk>, String>;
    fn get_rewards(&self, user_id: &str) -> Result<u64, String>;
    fn register_model(&mut self, model: Model) -> Result<String, String>;
    fn activate_model(&mut self, model_id: &str) -> Result<(), String>;
    fn deactivate_model(&mut self, model_id: &str) -> Result<(), String>;
    fn get_active_models(&self, offset: usize, limit: usize) -> Vec<Model>;
    fn get_models_needing_resources(&self, offset: usize, limit: usize) -> Vec<Model>;
    fn generate_completion(&self, prompt: &str) -> Result<Completion, String>;
    fn create_training_task(&mut self, task: TrainingTask) -> Result<String, String>;
    fn submit_training_results(
        &mut self,
        task_id: &str,
        model_weights: Vec<u8>,
    ) -> Result<(), String>;
}