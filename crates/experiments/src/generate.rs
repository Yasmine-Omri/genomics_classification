use std::{
    fs::File,
    io::{Read, Write},
};

use anyhow::Result;
use clap::Parser;
use itertools::Itertools;
use lz78::{
    sequence::{CharacterMap, CharacterSequence, U8Sequence},
    spa::{LZ78SPA, SPA},
};
use lz78_experiments::{
    argparse::{Experiments, GenerateCli},
    utils::{read_fashion_mnist, DatasetPartition},
};
use serde_pickle::DeOptions;

fn wikitext_experiment(_cli: GenerateCli, mut spa: LZ78SPA) -> Result<()> {
    let character_map = CharacterMap::from_data(
        &"abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ1234567890\n\t .,\"'?:;-_"
            .to_string(),
    );

    let mut generate_output = CharacterSequence::new(character_map.clone());
    spa.generate_data(
        &mut generate_output,
        5000,
        50,
        0.1,
        5,
        Some(&CharacterSequence::from_data_filtered(
            "This".to_string(),
            character_map.clone(),
        )),
    )?;

    println!("This{}", generate_output.data);

    Ok(())
}

fn fashion_mnist_experiment(_cli: GenerateCli, mut spa: LZ78SPA) -> Result<()> {
    let mut generate_output = U8Sequence::new(256);

    let mut test_set = read_fashion_mnist("data/fashion_mnist", DatasetPartition::Test)?;
    test_set = test_set
        .into_iter()
        .map(|v| v.into_iter().map(|x| x / 32).collect_vec())
        .collect_vec();
    let test_img = U8Sequence::from_data(test_set[40][0..28 * 28 / 2].to_vec(), 256 / 32)?;
    spa.generate_data(
        &mut generate_output,
        28 * 28 / 2,
        1000,
        0.5,
        6,
        Some(&test_img),
    )?;

    println!("test_data = np.array({:?}).reshape(-1, 28)", test_img.data);
    println!(
        "gen_data = np.array({:?}).reshape(-1, 28)",
        generate_output.data
    );

    Ok(())
}

fn main() {
    let cli = GenerateCli::parse();
    let mut file = File::open(cli.save_path.clone()).expect("could not open spa file");
    let mut spa_bytes: Vec<u8> = Vec::new();
    file.read_to_end(&mut spa_bytes)
        .expect("could not read file");

    let spa: LZ78SPA =
        serde_pickle::from_slice(&spa_bytes, DeOptions::new()).expect("could not deserialize");

    match cli.experiment {
        Experiments::FashionMnist => {
            fashion_mnist_experiment(cli, spa).expect("fashion mnist experiment failed")
        }
        Experiments::Wikitext => wikitext_experiment(cli, spa).expect("wikitext experiment failed"),
    }
}
