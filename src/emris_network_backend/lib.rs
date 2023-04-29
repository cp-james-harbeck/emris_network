use ic_cdk_macros::*;
use std::sync::{Arc, Mutex};
use once_cell::sync::Lazy;

mod completion;
mod model_chunk;
mod task_manager;
mod task_manager_impl;
mod training_task;
mod user;
mod fine_tuning;

use completion::*;
use model_chunk::*;
use task_manager::*;
use task_manager_impl::*;
use training_task::*;
use user::*;

static TASK_MANAGER: Lazy<Arc<Mutex<TaskManagerImpl>>> = Lazy::new(|| {
  Arc::new(Mutex::new(TaskManagerImpl::default()))
});

// Error handling for RwLock poisoning
fn handle_rwlock_poisoned<T>(_: std::sync::PoisonError<T>) -> String {
    "Internal error: RwLock is poisoned.".to_string()
}

#[update]
fn register_user(user: User) -> Result<String, String> {
    TASK_MANAGER
        .lock()
        .map_err(handle_rwlock_poisoned)?
        .register_user(user)
}

#[update]
fn update_user_resources(id: String, resources: u64) -> Result<(), String> {
    TASK_MANAGER
        .lock()
        .map_err(handle_rwlock_poisoned)?
        .update_user_resources(&id, resources)
}

#[update]
fn distribute_model_chunks() -> Result<(), String> {
    TASK_MANAGER
        .lock()
        .map_err(handle_rwlock_poisoned)?
        .distribute_model_chunks()
}

#[update]
fn submit_computed_chunk(chunk: ModelChunk, computed_results: Vec<u32>) -> Result<(), String> {
  TASK_MANAGER
      .lock() // Use lock() to acquire a MutexGuard
      .map_err(handle_rwlock_poisoned)?
      .submit_computed_chunk(chunk, computed_results)
}

#[query]
fn get_model_chunks(user_id: String) -> Result<Vec<ModelChunk>, String> {
    TASK_MANAGER
        .lock()
        .map_err(handle_rwlock_poisoned)?
        .get_model_chunks(&user_id)
}

#[query]
fn get_rewards(user_id: String) -> Result<u64, String> {
    TASK_MANAGER
        .lock()
        .map_err(handle_rwlock_poisoned)?
        .get_rewards(&user_id)
}

#[update]
fn register_model(admin_token: String, model: Model) -> Result<String, String> {
    let mut task_manager = TASK_MANAGER.lock().map_err(handle_rwlock_poisoned)?;
    // task_manager.check_admin_access(&admin_token)?;
    task_manager.register_model(model)
}

#[update]
fn activate_model(admin_token: String, model_id: String) -> Result<(), String> {
    let mut task_manager = TASK_MANAGER.lock().map_err(handle_rwlock_poisoned)?;
    // task_manager.check_admin_access(&admin_token)?;
    task_manager.activate_model(&model_id)
}

#[update]
fn deactivate_model(admin_token: String, model_id: String) -> Result<(), String> {
    let mut task_manager = TASK_MANAGER.lock().map_err(handle_rwlock_poisoned)?;
    // task_manager.check_admin_access(&admin_token)?;
    task_manager.deactivate_model(&model_id)
}

#[query]
fn get_active_models(offset: usize, limit: usize) -> Result<Vec<Model>, String> {
    Ok(TASK_MANAGER
        .lock()
        .map_err(handle_rwlock_poisoned)?
        .get_active_models(offset, limit))
}

#[query]
fn get_models_needing_resources(offset: usize, limit: usize) -> Result<Vec<Model>, String> {
    Ok(TASK_MANAGER
        .lock()
        .map_err(handle_rwlock_poisoned)?
        .get_models_needing_resources(offset, limit))
}

#[query]
fn generate_completion(prompt: String) -> Result<Completion, String> {
    // Input validation example
    if prompt.is_empty() {
        return Err("Prompt cannot be empty.".to_string());
    }

    TASK_MANAGER
        .lock()
        .map_err(handle_rwlock_poisoned)?
        .generate_completion(&prompt)
}

#[update]
fn create_training_task(admin_token: String, task: TrainingTask) -> Result<String, String> {
    let mut task_manager = TASK_MANAGER.lock().map_err(handle_rwlock_poisoned)?;
    // task_manager.check_admin_access(&admin_token)?;
    task_manager.create_training_task(task)
}

#[update]
fn submit_training_results(task_id: String, model_weights: Vec<u8>) -> Result<(), String> {
    TASK_MANAGER
        .lock()
        .map_err(handle_rwlock_poisoned)?
        .submit_training_results(&task_id, model_weights)
}
