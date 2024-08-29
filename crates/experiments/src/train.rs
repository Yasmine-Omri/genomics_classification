use std::{
    fs::File,
    io::Write,
    path::{Path, PathBuf},
    time::Instant,
};

use clap::Parser;
use itertools::Itertools;
use lz78::{
    sequence::{CharacterMap, CharacterSequence, U8Sequence},
    spa::{LZ78SPA, SPA},
};
use lz78_experiments::{
    argparse::{Experiments, TrainCli},
    utils::{read_fashion_mnist, read_wikitext, DatasetPartition},
};

use anyhow::{anyhow, bail, Result};
use parquet::{
    file::reader::{FileReader, SerializedFileReader},
    record::Field,
};
use png::Decoder;
use serde_pickle::SerOptions;

fn wikitext_experiment(cli: TrainCli) -> anyhow::Result<LZ78SPA> {
    let character_map = CharacterMap::from_data(
        &"abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ1234567890\n\t .,\"'?:;-_"
            .to_string(),
    );
    let mut text = read_wikitext("data/wikitext")?;
    if let Some(samples) = cli.samples {
        text = text.into_iter().take(samples as usize).collect_vec();
    }

    let mut spa = LZ78SPA::new(character_map.alphabet_size, cli.gamma);

    let n_loops = cli.repeat;
    let tic = Instant::now();
    for _ in 0..n_loops {
        for sample in text.iter() {
            let seq = CharacterSequence::from_data_filtered(sample.clone(), character_map.clone());
            spa.train_on_block(&seq, !cli.start_at_root)?;
        }
    }

    let time = tic.elapsed().as_secs_f32();
    println!(
        "Trained SPA on a block {n_loops} times with log loss {:.2} in {time:.3} seconds",
        spa.get_scaled_log_loss()
    );

    Ok(spa)
}

fn fashion_mnist_experiment(cli: TrainCli) -> anyhow::Result<LZ78SPA> {
    let mut bytes = read_fashion_mnist("data/fashion_mnist", DatasetPartition::Train)?;
    bytes = bytes
        .into_iter()
        .map(|v| v.into_iter().map(|x| x / 32).collect_vec())
        .collect_vec();
    if let Some(samples) = cli.samples {
        bytes = bytes.into_iter().take(samples as usize).collect_vec();
    }

    let alpha_size = 256 / 32;
    let mut spa = LZ78SPA::new(alpha_size, cli.gamma);

    let n_loops = cli.repeat;
    let tic = Instant::now();
    for _ in 0..n_loops {
        for img in bytes.iter() {
            let seq = U8Sequence::from_data(img.clone(), alpha_size)?;
            spa.train_on_block(&seq, !cli.start_at_root)?;
        }
    }

    let time = tic.elapsed().as_secs_f32();
    println!(
        "Trained SPA on a block {n_loops} times with log loss {:.2} in {time:.3} seconds",
        spa.get_scaled_log_loss()
    );

    Ok(spa)
}

fn main() {
    let cli = TrainCli::parse();
    let save_path = cli.save_path.clone();

    let spa = match cli.experiment {
        Experiments::Wikitext => wikitext_experiment(cli).expect("wikitext experiment failed"),
        Experiments::FashionMnist => {
            fashion_mnist_experiment(cli).expect("fashion mnist experiment failed")
        }
    };

    let serialized = serde_pickle::to_vec(&spa, SerOptions::new()).expect("serialization failed");
    let mut file = File::create(PathBuf::from(save_path)).expect("bad save path");
    file.write(&serialized).expect("write failed");
}
