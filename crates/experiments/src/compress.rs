use std::time::Instant;

use lz78::{
    encoder::{Encoder, LZ8Encoder},
    sequence::{Sequence, U8Sequence},
    spa::{LZ78SPA, SPA},
};
use lz78_experiments::utils::read_file_to_string;

fn main() {
    let text = read_file_to_string("data/Shakespeare/finaldata.txt")
        .expect("could not read input file")
        .as_bytes()[0..1000]
        .to_owned();
    let tic = Instant::now();
    // let charmap = CharacterMap::from_data(&text);

    let mut spa = LZ78SPA::new(256, 0.009);
    let input = U8Sequence::from_data(text, 256).expect("could not create sequence");
    let loss = spa
        .train_on_block(&input, false)
        .expect("could not train SPA");
    let time = tic.elapsed().as_secs_f32();

    println!(
        "Trained in {time} sec. Estimated compression ratio: {}",
        (loss * input.len() as f64 / 8.0).ceil() / input.len() as f64
    );

    // now do actual LZ78
    let compressed = LZ8Encoder::new()
        .encode(&input)
        .expect("compression failed");
    println!("LZ78 compression ratio: {}", compressed.compression_ratio());
}
