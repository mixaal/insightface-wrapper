use std::fmt;

#[derive(Debug)]
pub struct InsightFaceError {
    pub message: String,
}

impl InsightFaceError {
    pub fn new<E: std::error::Error>(err: E) -> Self {
        InsightFaceError {
            message: err.to_string(),
        }
    }

    pub fn from_message<S: Into<String>>(msg: S) -> Self {
        InsightFaceError {
            message: msg.into(),
        }
    }
}

impl fmt::Display for InsightFaceError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Error: {}", self.message)
    }
}

impl std::error::Error for InsightFaceError {}
