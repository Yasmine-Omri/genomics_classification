use itertools::Itertools;
use lz78::{
    sequence::Sequence,
    source::{DefaultLZ78SourceNode, LZ78Source},
};
use rand::{thread_rng, Rng};

const A: u32 = 2;
const GAMMA: f64 = 0.5;
const N_TRIALS: u64 = 40;

/// Experiment:
/// 1.Generate from LZ78 source (very long sequence)
/// 2; Generate Ber(q), look at first k symbols, and measure n_k, which is the first time we
///     find that k-tuple in the sequence generated in 1.
/// 3. Increment k by 1, and repeat.
/// log(n_k) would then be the number of bits that you would need in order to represent this k-tuple.
/// Does log(n_k) / k approach the entropy h2(q)?
/// Plot log(n_k) / k as a function of k to confirm.
fn main() {
    let q: f32 = 0.25;
    let h2 = q * (1.0 / q).log2() + (1.0 - q) * (1.0 / (1.0 - q)).log2();
    println!("h2(q) = {h2}",);

    for k in 1..50 {
        let n = 2.0f32.powf(h2 * k as f32).ceil() as u64;
        let mut n_ks = Vec::with_capacity(N_TRIALS as usize);
        for _ in 0..N_TRIALS {
            let mut found_nk = false;
            let mut nk_base = 0;
            let rand_seq = (0..k)
                .map(|_| {
                    if thread_rng().gen::<f32>() < q {
                        1u32
                    } else {
                        0
                    }
                })
                .collect_vec();

            let mut source = LZ78Source::new(A, DefaultLZ78SourceNode {}, Some(GAMMA));
            while !found_nk {
                let lz78_seq = source
                    .generate_symbols(n, &mut thread_rng())
                    .expect("could not generate");

                let n_k = (0..(n - k))
                    .filter(|i| {
                        (0..k)
                            .map(|j| lz78_seq.get(i + j).expect("invalid index"))
                            .collect_vec()
                            == rand_seq
                    })
                    .next();

                if n_k.is_none() {
                    nk_base += n;
                    continue;
                }

                let n_k = n_k.unwrap() + nk_base;
                n_ks.push(n_k);
                found_nk = true;
            }
        }

        let mean_nk = n_ks.iter().sum::<u64>() as f32 / N_TRIALS as f32;
        println!("k={k}, log(n_k) / k = {}", mean_nk.log2() / k as f32);
    }
}
