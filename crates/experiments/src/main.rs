use std::{fs::File, io::Write};

use lz78::source::{LZ78Source, DiscreteThetaBinarySourceNode};
use rand::thread_rng;

fn main() {
    let mut ber_src = LZ78Source::new(
        2,
        DiscreteThetaBinarySourceNode::new(vec![0.5, 0.5], vec![0., 1.], &mut thread_rng()),
        None,
    );
    let data = ber_src
        .generate_symbols(10_000_000_000, &mut thread_rng())
        .unwrap()
        .data;

    let mut output_bytes: Vec<u8> = Vec::with_capacity((data.len() + 7) / 8);
    for bits in data.chunks(8) {
        let mut byte = 0;
        for &bit in bits.iter().rev() {
            byte = (byte << 1) | (bit as u8);
        }
        output_bytes.push(byte);
    }

    let mut file = File::create("ber_lz78_src_10B.bin").unwrap();
    file.write_all(&output_bytes).unwrap();
}
