#[derive(Debug)]
pub enum TaskManagerError {
    UserAlreadyExists,
    UserNotFound,
    ModelAlreadyExists,
    ModelNotFound,
    TrainingTaskAlreadyExists,
    TrainingTaskNotFound,
    InsufficientResources,
    UnauthorizedAccess,
    RwLockPoisoned,
    GpuComputationFailed(String),
    // Additional error variants can be added here.
}

impl std::fmt::Display for TaskManagerError {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match self {
            TaskManagerError::UserAlreadyExists => write!(f, "User already exists."),
            TaskManagerError::UserNotFound => write!(f, "User not found."),
            // Implement display for other error variants.
            _ => write!(f, "An error occurred."),
        }
    }
}

impl std::error::Error for TaskManagerError {}
