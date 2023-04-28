use crate::model_chunk::ModelChunk;
use crate::task_manager::TaskManagerInterface;
use crate::task_manager_impl::{Model, TaskManagerImpl};
use crate::training_task::TrainingTask;
use crate::user::User;

#[test]
fn test_register_user() {
    let mut task_manager = TaskManagerImpl::default();
    let user = User {
        id: "user1".to_string(),
        resources: 100,
        rewards: 0,
        rate_limit_tokens: 10,
    };
    let result = task_manager.register_user(user.clone());
    assert_eq!(result, Ok(user.id.clone()));
    assert!(task_manager.users.contains_key(&user.id));
}

#[test]
fn test_update_user_resources() {
    let mut task_manager = TaskManagerImpl::default();
    let user = User {
        id: "user1".to_string(),
        resources: 100,
        rewards: 0,
        rate_limit_tokens: 10,
    };
    task_manager.users.insert(user.id.clone(), user.clone());
    let result = task_manager.update_user_resources(&user.id, 200);
    assert_eq!(result, Ok(()));
    assert_eq!(task_manager.users.get(&user.id).unwrap().resources, 200);
}

#[test]
fn test_register_model() {
    let mut task_manager = TaskManagerImpl::default();
    let model = Model {
        id: "model1".to_string(),
        min_resources: 500,
        active: false,
    };
    let result = task_manager.register_model(model.clone());
    assert_eq!(result, Ok(model.id.clone()));
    assert!(task_manager.models.contains_key(&model.id));
}

#[test]
fn test_activate_deactivate_model() {
    let mut task_manager = TaskManagerImpl::default();
    let model = Model {
        id: "model1".to_string(),
        min_resources: 500,
        active: false,
    };
    task_manager.models.insert(model.id.clone(), model.clone());
    let user = User {
        id: "user1".to_string(),
        resources: 1000,
        rewards: 0,
        rate_limit_tokens: 10,
    };
    task_manager.users.insert(user.id.clone(), user.clone());
    let result_activate = task_manager.activate_model(&model.id);
    assert_eq!(result_activate, Ok(()));
    assert!(task_manager.models.get(&model.id).unwrap().active);
    let result_deactivate = task_manager.deactivate_model(&model.id);
    assert_eq!(result_deactivate, Ok(()));
    assert!(!task_manager.models.get(&model.id).unwrap().active);
}

#[test]
fn test_create_training_task() {
    let mut task_manager = TaskManagerImpl::default();
    let training_task = TrainingTask {
        id: "task1".to_string(),
        model_id: "model1".to_string(),
        training_data: vec![0u8; 1024],
        model_weights: None,
    };
    let result = task_manager.create_training_task(training_task.clone());
    assert_eq!(result, Ok(training_task.id.clone()));
    assert!(task_manager.training_tasks.contains_key(&training_task.id));
}

#[test]
fn test_submit_training_results() {
    let mut task_manager = TaskManagerImpl::default();
    let training_task = TrainingTask {
        id: "task1".to_string(),
        model_id: "model1".to_string(),
        training_data: vec![0u8; 1024],
        model_weights: None,
    };
    task_manager
        .training_tasks
        .insert(training_task.id.clone(), training_task.clone());
    let model_chunk = ModelChunk {
        id: "chunk1".to_string(),
        model_id: "model1".to_string(),
        data: vec![0u8; 1024],
    };
    let result = task_manager.submit_training_results(&training_task.id, model_chunk.clone());
    assert_eq!(result, Ok(model_chunk.id.clone()));
    assert!(task_manager.model_chunks.contains_key(&model_chunk.id));
}

#[test]
fn test_get_model_chunk() {
    let mut task_manager = TaskManagerImpl::default();
    let model_chunk = ModelChunk {
        id: "chunk1".to_string(),
        model_id: "model1".to_string(),
        data: vec![0u8; 1024],
    };
    task_manager
        .model_chunks
        .insert(model_chunk.id.clone(), model_chunk.clone());
    let result = task_manager.get_model_chunk(&model_chunk.id);
    assert_eq!(result, Ok(model_chunk.clone()));
}

#[test]
fn test_get_training_task() {
    let mut task_manager = TaskManagerImpl::default();
    let training_task = TrainingTask {
        id: "task1".to_string(),
        model_id: "model1".to_string(),
        training_data: vec![0u8; 1024],
        model_weights: None,
    };
    task_manager
        .training_tasks
        .insert(training_task.id.clone(), training_task.clone());
    let result = task_manager.get_training_task(&training_task.id);
    assert_eq!(result, Ok(training_task.clone()));
}

#[test]
fn test_get_user() {
    let mut task_manager = TaskManagerImpl::default();
    let user = User {
        id: "user1".to_string(),
        resources: 100,
        rewards: 0,
        rate_limit_tokens: 10,
    };
    task_manager.users.insert(user.id.clone(), user.clone());
    let result = task_manager.get_user(&user.id);
    assert_eq!(result, Ok(user.clone()));
}

#[test]
fn test_get_model() {
    let mut task_manager = TaskManagerImpl::default();
    let model = Model {
        id: "model1".to_string(),
        min_resources: 500,
        active: false,
    };
    task_manager.models.insert(model.id.clone(), model.clone());
    let result = task_manager.get_model(&model.id);
    assert_eq!(result, Ok(model.clone()));
}
