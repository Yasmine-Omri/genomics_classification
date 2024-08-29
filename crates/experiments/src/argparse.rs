use clap::{Parser, ValueEnum};

/// Build an LZ78 SPA
#[derive(Parser)]
pub struct TrainCli {
    /// Path to save the file
    #[arg(long, short)]
    pub save_path: String,

    /// Which dataset to use
    #[arg(long, short)]
    pub experiment: Experiments,

    /// Number of times to repeat the experiment
    #[arg(long, default_value_t = 1)]
    pub repeat: u32,

    /// Number of samples to take. Defaults to the whole dataset.
    #[arg(long)]
    pub samples: Option<u64>,

    /// Whether to start from the root of the LZ78 tree for each new sample
    #[arg(long, default_value_t = false)]
    pub start_at_root: bool,

    /// LZ78 SPA smoothing parameter
    #[arg(long, default_value_t = 0.5)]
    pub gamma: f64,
}

/// Generate from an LZ78 SPA built from train.rs
#[derive(Parser)]
pub struct GenerateCli {
    /// Path where the SPA was saved
    #[arg(long, short)]
    pub save_path: String,

    /// Training dataset
    #[arg(long, short)]
    pub experiment: Experiments,
}

#[derive(ValueEnum, Clone)]
pub enum Experiments {
    /// Doc comment
    #[value()]
    FashionMnist,

    #[value()]
    Wikitext,
}
