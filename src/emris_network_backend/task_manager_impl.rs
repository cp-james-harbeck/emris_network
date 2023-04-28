use crate::completion::Completion;
use crate::model_chunk::ModelChunk;
use crate::task_manager::TaskManagerInterface;
use crate::training_task::TrainingTask;
use crate::user::User;
use ic_cdk::export::candid::CandidType;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

pub struct TaskManagerImpl {
    users: HashMap<String, User>,
    model_chunks: HashMap<String, ModelChunk>,
    models: HashMap<String, Model>,
    training_tasks: HashMap<String, TrainingTask>,
}

#[derive(Clone, Deserialize, Serialize, CandidType)]
pub struct Model {
    pub id: String,
    pub min_resources: u64,
    pub active: bool,
}

impl Default for TaskManagerImpl {
    fn default() -> Self {
        Self {
            users: HashMap::new(),
            model_chunks: HashMap::new(),
            models: HashMap::new(),
            training_tasks: HashMap::new(),
        }
    }
}

impl TaskManagerInterface for TaskManagerImpl {
    fn init(&mut self) {}

    fn register_user(&mut self, user: User) -> Result<String, String> {
        if self.users.contains_key(&user.id) {
            return Err("User already exists.".to_string());
        }
        let user_id = user.id.clone();
        self.users.insert(user.id.clone(), user);
        Ok(user_id)
    }

    fn update_user_resources(&mut self, id: &str, resources: u64) -> Result<(), String> {
        if let Some(user) = self.users.get_mut(id) {
            user.resources = resources;
            Ok(())
        } else {
            Err("User not found.".to_string())
        }
    }

    fn distribute_model_chunks(&mut self) -> Result<(), String> {
        let chunk_id = "chunk1";
        let user_id = "user1";
        let data = vec![0u8; 1024];
        let chunk = ModelChunk {
            id: chunk_id.to_string(),
            user_id: user_id.to_string(),
            data: data,
        };
        self.model_chunks.insert(chunk_id.to_string(), chunk);
        Ok(())
    }

    fn submit_computed_chunk(
        &mut self,
        chunk: ModelChunk,
        computed_results: Vec<u32>, // Accept computed results as an argument
    ) -> Result<(), String> {
        if let Some(existing_chunk) = self.model_chunks.get_mut(&chunk.id) {
            existing_chunk.data = chunk.data;
        } else {
            return Err("Model chunk not found.".to_string());
        }

        // Use the computed results directly instead of running the GPU computation
        let reward = self.calculate_rewards(&chunk.user_id, 1);
        println!("User {} earned {} tokens.", chunk.user_id, reward);

        Ok(())
    }

    fn get_model_chunks(&self, user_id: &str) -> Result<Vec<ModelChunk>, String> {
        let mut chunks = Vec::new();
        for chunk in self.model_chunks.values() {
            if chunk.user_id == user_id {
                chunks.push(chunk.clone());
            }
        }
        Ok(chunks)
    }

    fn get_rewards(&self, user_id: &str) -> Result<u64, String> {
        if let Some(user) = self.users.get(user_id) {
            Ok(user.rewards)
        } else {
            Err("User not found.".to_string())
        }
    }

    fn register_model(&mut self, model: Model) -> Result<String, String> {
        if self.models.contains_key(&model.id) {
            return Err("Model already exists.".to_string());
        }
        let model_id = model.id.clone();
        self.models.insert(model_id.clone(), model);

        Ok(model_id)
    }

    fn activate_model(&mut self, model_id: &str) -> Result<(), String> {
        let total_resources: u64 = self.users.values().map(|user| user.resources).sum();
        let model = self.models.get_mut(model_id).ok_or("Model not found.")?;
        if total_resources >= model.min_resources {
            model.active = true;
            Ok(())
        } else {
            Err("Insufficient resources to activate the model.".to_string())
        }
    }

    fn deactivate_model(&mut self, model_id: &str) -> Result<(), String> {
        let model = self.models.get_mut(model_id).ok_or("Model not found.")?;
        model.active = false;
        Ok(())
    }

    fn get_active_models(&self, offset: usize, limit: usize) -> Vec<Model> {
        self.models
            .values()
            .filter(|model| model.active)
            .skip(offset)
            .take(limit)
            .cloned()
            .collect()
    }

    fn get_models_needing_resources(&self, offset: usize, limit: usize) -> Vec<Model> {
        let total_resources: u64 = self.users.values().map(|user| user.resources).sum();
        self.models
            .values()
            .filter(|model| !model.active && model.min_resources > total_resources)
            .skip(offset)
            .take(limit)
            .cloned()
            .collect()
    }

    fn generate_completion(&self, prompt: &str) -> Result<Completion, String> {
        let active_models = self.get_active_models(0, 1);
        if active_models.is_empty() {
            return Err("No active models available.".to_string());
        }
        let model = &active_models[0];
        let generated_text = format!("Generated text for '{}' using model '{}'", prompt, model.id);
        Ok(Completion {
            prompt: prompt.to_string(),
            generated_text,
        })
    }

    fn create_training_task(&mut self, task: TrainingTask) -> Result<String, String> {
        if self.training_tasks.contains_key(&task.id) {
            return Err("Training task already exists.".to_string());
        }
        let task_id = task.id.clone();
        self.training_tasks.insert(task_id.clone(), task);
        Ok(task_id)
    }

    fn submit_training_results(
        &mut self,
        task_id: &str,
        model_weights: Vec<u8>,
    ) -> Result<(), String> {
        if let Some(task) = self.training_tasks.get_mut(task_id) {
            task.model_weights = Some(model_weights);
            Ok(())
        } else {
            Err("Training task not found.".to_string())
        }
    }
}

impl TaskManagerImpl {
    fn calculate_rewards(&mut self, user_id: &str, completed_chunks: usize) -> u64 {
        let reward = completed_chunks as u64;
        if let Some(user) = self.users.get_mut(user_id) {
            user.rewards += reward;
        }
        reward
    }
}
