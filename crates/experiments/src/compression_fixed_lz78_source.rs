use std::collections::HashMap;

use clap::Parser;
use itertools::Itertools;
use lz78::{
    encoder::{Encoder, LZ8Encoder},
    sequence::U32Sequence,
    source::{DefaultLZ78SourceNode, LZ78Source, SimplifiedBinarySourceNode},
};
use lz78_experiments::{
    argparse::SourceCompressionCli,
    utils::{read_mnist, DatasetPartition},
};
use rand::{thread_rng, Rng};
use rayon::iter::{IntoParallelIterator, ParallelIterator};

/// Only binary experiments so far
const N_BLOCKS: u64 = 100_000;
const BLOCK_SIZE: u64 = 50;
const A: u32 = 2;

fn empirical_entropy(data: &[usize]) -> f64 {
    let n = data.len();
    let mut counts: HashMap<usize, u64> = HashMap::new();
    for &val in data.iter() {
        counts.insert(val, *counts.get(&val).unwrap_or(&0) + 1);
    }

    let mut h = 0.;
    for (_, count) in counts {
        let p = count as f64 / n as f64;
        h += p * (1. / p).log2();
    }

    h
}

fn get_entropy_rate(cli: &SourceCompressionCli) -> f64 {
    match cli.data_generator {
        lz78_experiments::argparse::DataGenerators::Bernoulli => {
            if cli.prob_one == 0. || cli.prob_one == 1. {
                0.
            } else {
                cli.prob_one * (1.0 / cli.prob_one).log2()
                    + (1.0 - cli.prob_one) * (1.0 / (1.0 - cli.prob_one)).log2()
            }
        }
        lz78_experiments::argparse::DataGenerators::BernoulliLZ78Source => 0.,
    }
}

fn get_sequence_to_compress(cli: &SourceCompressionCli) -> Vec<u32> {
    match cli.data_generator {
        lz78_experiments::argparse::DataGenerators::Bernoulli => (0..cli.k_max)
            .map(|_| {
                if thread_rng().gen::<f64>() < cli.prob_one {
                    1u32
                } else {
                    0
                }
            })
            .collect_vec(),
        lz78_experiments::argparse::DataGenerators::BernoulliLZ78Source => {
            let mut ber_src = LZ78Source::new(
                2,
                SimplifiedBinarySourceNode::new(vec![0.5, 0.5], vec![0., 1.], &mut thread_rng()),
                None,
            );
            ber_src
                .generate_symbols(cli.k_max, &mut thread_rng())
                .unwrap()
                .data
        }
    }
}

fn main() {
    let cli = SourceCompressionCli::parse();

    for _trial in 0..cli.trials {
        // let data = read_file_to_string("data/enwik8").expect("file read error");
        // let seq = CharacterSequence::from_data_inferred_character_map(data);
        // let a = seq.alphabet_size();
        // let data = seq.encoded[0..1000].to_vec();
        let data = read_mnist("data/mnist", DatasetPartition::Train)
            .unwrap()
            .0
            .into_iter()
            .flatten()
            .map(|x| x as u32 / 128)
            .take(cli.k_max as usize)
            .collect_vec();

        // Find the LZ78 compression ratio
        let encoded = LZ8Encoder::new()
            .encode(&U32Sequence::from_data(data.clone(), A).expect("could not make U32 sequence"))
            .expect("encoding error");
        eprintln!("LZ78 Compression Ratio: {}", encoded.compression_ratio());

        let mut lz78_seq = Vec::new();
        for _ in 0..N_BLOCKS {
            let mut source = LZ78Source::new(A, DefaultLZ78SourceNode {}, Some(cli.gamma));
            lz78_seq.extend(
                source
                    .generate_symbols(BLOCK_SIZE, &mut thread_rng())
                    .expect("could not generate sequence")
                    .data,
            );
        }

        let mut phrase_starts = Vec::new();
        let mut phrase_lengths = Vec::new();

        let mut start_idx = 0;
        while start_idx < data.len() {
            // find longest match
            // TODO: find solution that isn't O(n^2)
            let block_size =
                (lz78_seq.len() + cli.max_thread as usize - 1) / cli.max_thread as usize;

            let (best_start, best_len) = (0..cli.max_thread as usize)
                .into_par_iter()
                .map(|block| {
                    let mut best_start = 0;
                    let mut best_len = 0;
                    for i in (block * block_size)..(lz78_seq.len().min(block_size * (block + 1))) {
                        let mut length = 0;
                        while length < (data.len() - start_idx).min(lz78_seq.len() - i)
                            && data[start_idx + length] == lz78_seq[i + length]
                        {
                            length += 1;
                        }
                        if length > best_len {
                            best_len = length;
                            best_start = i;
                        }
                    }
                    (best_start, best_len)
                })
                .max_by_key(|(_, len)| *len)
                .unwrap();
            phrase_starts.push(best_start);
            phrase_lengths.push(best_len);

            start_idx += best_len;
        }

        // println!("{phrase_starts:?}");
        // println!("{phrase_lengths:?}");

        println!(
            "{}",
            (lz78_seq.len() as f64).log2().ceil() * phrase_starts.len() as f64
        );
        println!("{}", empirical_entropy(&phrase_lengths));

        let est_comp = ((lz78_seq.len() as f64).log2().ceil() * phrase_starts.len() as f64
            + empirical_entropy(&phrase_lengths))
            / (data.len() as f64 * (A as f64).log2());

        println!("Compression Ratio: {est_comp}");
    }
}
