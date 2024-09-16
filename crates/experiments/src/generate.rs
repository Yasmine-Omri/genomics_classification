use anyhow::Result;
use clap::Parser;
use itertools::Itertools;
use lz78::{
    sequence::{CharacterSequence, U8Sequence},
    spa::{LZ78SPA, SPA},
};
use lz78_experiments::{
    argparse::{Experiments, GenerateCli},
    utils::{default_character_map, read_fashion_mnist, DatasetPartition},
};

fn wikitext_experiment(_cli: GenerateCli, mut spa: LZ78SPA) -> Result<()> {
    let character_map = default_character_map();

    let mut generate_output = CharacterSequence::new(character_map.clone());
    spa.generate_data(
        &mut generate_output,
        5000,
        500,
        0.2,
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
    let spa = LZ78SPA::from_file(cli.save_path.clone()).expect("read spa failed");

    match cli.experiment {
        Experiments::FashionMnist => {
            fashion_mnist_experiment(cli, spa).expect("fashion mnist experiment failed")
        }
        Experiments::Wikitext => wikitext_experiment(cli, spa).expect("wikitext experiment failed"),
        Experiments::C4 => wikitext_experiment(cli, spa).expect("c4 experiment failed"),
    }
}
