use std::{
    fs::File,
    io::{Read, Write},
};

use crate::{
    encoder::{
        lz78_bits_to_encode_phrase, lz78_decode, BitStorage, EncodedSequence, StreamingEncoder,
    },
    sequence::Sequence,
    tree::LZ78Tree,
    util::sample_from_pdf,
};
use anyhow::Result;
use itertools::Itertools;
use rand::{distributions::Uniform, prelude::Distribution, thread_rng};
use rkyv::{Archive, Deserialize, Serialize};

pub struct StreamingLZ78Encoder {
    encoded: EncodedSequence,
    tree: LZ78Tree,
    state: u64,
    n: u64,
    n_output_leaves: u64,
    n_output_bits_finished_phrases: u64,
    total_log_loss: f64,
}

impl StreamingLZ78Encoder {
    pub fn new(alpha_size: u32) -> Self {
        let bits = BitStorage::new();
        let encoded = EncodedSequence::from_data(bits, 0, alpha_size);
        Self {
            encoded,
            tree: LZ78Tree::new(alpha_size),
            state: LZ78Tree::ROOT_IDX,
            n: 0,
            n_output_leaves: 0,
            n_output_bits_finished_phrases: 0,
            total_log_loss: 0.0,
        }
    }
}

impl<T: Sequence> StreamingEncoder<T> for StreamingLZ78Encoder {
    fn encode_block(&mut self, input: &T) -> Result<()> {
        let mut start_idx = 0;

        let mut ref_idxs: Vec<Option<u64>> = Vec::new();
        let mut output_leaves: Vec<u32> = Vec::new();
        let mut parsing_in_progress = false;

        self.n += input.len();

        while start_idx < input.len() {
            let traversal_output = self.tree.traverse_to_leaf_from(
                self.state,
                input,
                start_idx,
                input.len(),
                true,
                true,
            )?;

            start_idx = traversal_output.phrase_end_idx + 1;
            self.total_log_loss += traversal_output.log_loss;

            ref_idxs.push(self.tree.phrase_num(traversal_output.state_idx));
            output_leaves.push(traversal_output.added_leaf.unwrap_or(0));

            if traversal_output.added_leaf.is_some() {
                self.state = LZ78Tree::ROOT_IDX;
            } else {
                self.state = traversal_output.state_idx;
                parsing_in_progress = true;
                break;
            }
        }

        // encode
        // compute output
        let mut n_output_bits = 0;
        let mut n_output_bits_finished_phrases = 0;
        for i in 0..(output_leaves.len() as u64) {
            n_output_bits_finished_phrases = n_output_bits;
            n_output_bits +=
                lz78_bits_to_encode_phrase(i + self.n_output_leaves, input.alphabet_size()) as u64;
        }

        let mut total_output_leaves = self.n_output_leaves + (output_leaves.len() - 1) as u64;
        if !parsing_in_progress {
            total_output_leaves += 1;
            n_output_bits_finished_phrases = n_output_bits;
        }

        self.encoded.set_uncompressed_len(self.n);
        self.encoded.truncate(self.n_output_bits_finished_phrases);
        self.encoded.extend_capacity((n_output_bits + 7) / 8);

        for (i, (leaf, ref_idx)) in output_leaves.into_iter().zip(ref_idxs).enumerate() {
            let ref_idx = if let Some(x) = ref_idx { x + 1 } else { 0 };
            let bitwidth =
                lz78_bits_to_encode_phrase(i as u64 + self.n_output_leaves, input.alphabet_size());
            let val = if i == 0 && self.n_output_leaves == 0 {
                leaf as u64
            } else {
                ref_idx * (input.alphabet_size() as u64) + (leaf as u64)
            };

            self.encoded.push(val, bitwidth);
        }

        self.n_output_bits_finished_phrases += n_output_bits_finished_phrases;
        self.n_output_leaves = total_output_leaves;

        Ok(())
    }

    fn get_encoded_sequence(&self) -> &EncodedSequence {
        &self.encoded
    }

    fn decode(&self, output: &mut T) -> Result<()> {
        lz78_decode(output, &self.encoded)
    }
}

pub trait SPA {
    fn train_on_block<T: ?Sized>(&mut self, input: &T, include_prev_context: bool) -> Result<f64>
    where
        T: Sequence;

    fn compute_test_loss<T: ?Sized>(&mut self, input: &T) -> Result<f64>
    where
        T: Sequence;

    fn compute_test_loss_from_root<T: ?Sized>(&mut self, input: &T) -> Result<f64>
    where
        T: Sequence;

    fn compute_spa_at_current_state(&self) -> Vec<f64>;

    fn get_scaled_log_loss(&self) -> f64;

    fn generate_data<T>(
        &mut self,
        output_seq: &mut T,
        len: u64,
        min_context: u64,
        temperature: f64,
        top_k: u32,
        seed_data: Option<&T>,
    ) -> Result<f64>
    where
        T: Sequence;
}

#[derive(Debug, Archive, Serialize, Deserialize)]
pub struct LZ78SPA {
    tree: LZ78Tree,
    state: u64,
    n: u64,
    total_log_loss: f64,
    alphabet_size: u32,
}

impl LZ78SPA {
    pub fn new(alpha_size: u32, gamma: f64) -> Self {
        Self {
            tree: LZ78Tree::new_spa(alpha_size, gamma),
            state: LZ78Tree::ROOT_IDX,
            n: 0,
            total_log_loss: 0.0,
            alphabet_size: alpha_size,
        }
    }

    /// Traverses the tree and returns the new state and loss
    fn compute_test_loss_on_slice_from_state<T: Sequence + ?Sized>(
        &mut self,
        input: &T,
        state: u64,
        start_idx: u64,
        len: u64,
    ) -> Result<(u64, f64)> {
        let mut start_idx = start_idx;
        let mut log_loss = 0.0;
        let end_idx = start_idx + len;

        let mut state = state;
        while start_idx < end_idx {
            let traversal_output = self
                .tree
                .traverse_to_leaf_from(state, input, start_idx, end_idx, false, false)?;

            start_idx = traversal_output.phrase_end_idx + 1;
            log_loss += traversal_output.log_loss;
            state = if traversal_output.reached_leaf {
                LZ78Tree::ROOT_IDX
            } else {
                traversal_output.state_idx
            };
        }

        Ok((state, log_loss))
    }

    fn maybe_traverse_once_to_leaf<T: Sequence>(
        &mut self,
        input: &T,
        state: u64,
        start_idx: u64,
        len: u64,
    ) -> Result<(u64, f64)> {
        let mut log_loss = 0.0;
        let end_idx = start_idx + len;

        let traversal_output = self
            .tree
            .traverse_to_leaf_from(state, input, start_idx, end_idx, false, false)?;

        log_loss += traversal_output.log_loss;
        let state = if traversal_output.reached_leaf {
            LZ78Tree::ROOT_IDX
        } else {
            traversal_output.state_idx
        };

        Ok((state, log_loss))
    }

    pub fn save_to_file(&self, path: String) -> Result<()> {
        let mut file = File::create(path)?;
        let mut bytes = rkyv::to_bytes::<_, 1024>(self)?;
        println!("Saving SPA: {:.3} MB", bytes.len() as f64 / 1e6);
        file.write_all(&mut bytes)?;

        Ok(())
    }

    pub fn from_file(path: String) -> Result<Self> {
        let mut file = File::open(path)?;
        let mut bytes: Vec<u8> = Vec::new();
        file.read_to_end(&mut bytes)?;
        let archived = unsafe { rkyv::archived_root::<Self>(&bytes[..]) };

        Ok(archived.deserialize(&mut rkyv::Infallible).unwrap())
    }
}

impl SPA for LZ78SPA {
    fn train_on_block<T: ?Sized>(&mut self, input: &T, include_prev_context: bool) -> Result<f64>
    where
        T: Sequence,
    {
        let prev_log_loss = self.total_log_loss;
        if !include_prev_context {
            self.state = LZ78Tree::ROOT_IDX;
        }

        let mut start_idx = 0;
        self.n += input.len();

        while start_idx < input.len() {
            let traversal_output = self.tree.traverse_to_leaf_from(
                self.state,
                input,
                start_idx,
                input.len(),
                true,
                true,
            )?;

            start_idx = traversal_output.phrase_end_idx + 1;
            self.total_log_loss += traversal_output.log_loss;

            if traversal_output.added_leaf.is_some() {
                self.state = LZ78Tree::ROOT_IDX;
            } else {
                self.state = traversal_output.state_idx;
                break;
            }
        }

        Ok((self.total_log_loss - prev_log_loss) / (input.len() as f64))
    }

    fn compute_test_loss<T: ?Sized>(&mut self, input: &T) -> Result<f64>
    where
        T: Sequence,
    {
        Ok(self
            .compute_test_loss_on_slice_from_state(input, self.state, 0, input.len())?
            .1)
    }

    fn compute_test_loss_from_root<T: ?Sized>(&mut self, input: &T) -> Result<f64>
    where
        T: Sequence,
    {
        Ok(self
            .compute_test_loss_on_slice_from_state(input, LZ78Tree::ROOT_IDX, 0, input.len())?
            .1)
    }

    fn compute_spa_at_current_state(&self) -> Vec<f64> {
        self.tree.compute_spa(self.state)
    }

    fn get_scaled_log_loss(&self) -> f64 {
        if self.n == 0 {
            0.0
        } else {
            self.total_log_loss / (self.n as f64)
        }
    }

    fn generate_data<T>(
        &mut self,
        output_seq: &mut T,
        len: u64,
        min_context: u64,
        temperature: f64,
        top_k: u32,
        seed_data: Option<&T>,
    ) -> Result<f64>
    where
        T: Sequence,
    {
        let mut log_loss: f64 = 0.0;

        let top_k = top_k.clamp(1, self.alphabet_size);
        let mut state = self.state;

        let mut rng = thread_rng();
        let samples = Uniform::new(0.0, 1.0)
            .sample_iter(&mut rng)
            .take(len as usize)
            .collect_vec();

        if let Some(data) = seed_data {
            (state, log_loss) = self.compute_test_loss_on_slice_from_state(
                data,
                LZ78Tree::ROOT_IDX,
                0,
                data.len(),
            )?;
        }

        for sample_num in 0..len {
            // If we're at a place with no information (root or leaf), we need to
            // re-seed the SPA with some context

            if state == LZ78Tree::ROOT_IDX || self.tree.is_leaf(state) {
                for k in (0..=min_context.min(sample_num)).rev() {
                    state = if k == 0 {
                        LZ78Tree::ROOT_IDX
                    } else {
                        self.maybe_traverse_once_to_leaf(
                            output_seq,
                            LZ78Tree::ROOT_IDX,
                            sample_num - k,
                            k,
                        )?
                        .0
                    };

                    if !self.tree.is_leaf(state) && state != LZ78Tree::ROOT_IDX {
                        break;
                    }
                }
            }

            let mut spa = self.tree.compute_spa(state);
            let most_likely_next_sym = (0..self.alphabet_size)
                .max_by(|i, j| spa[*i as usize].total_cmp(&spa[*j as usize]))
                .unwrap();
            if temperature != 0.0 && temperature != 1.0 {
                spa = spa
                    .iter()
                    .map(|x| 2.0_f64.powf(x.log2() / temperature))
                    .collect_vec();
            }

            // topk sampling
            (0..self.alphabet_size)
                .sorted_by(|i, j| spa[*i as usize].total_cmp(&spa[*j as usize]))
                .take((self.alphabet_size - top_k) as usize)
                .map(|i| {
                    spa[i as usize] = 0.0;
                })
                .collect_vec();

            let sum: f64 = spa.iter().sum();
            spa = spa.iter().map(|x| *x / sum).collect_vec();

            let new_sym = if temperature == 0.0 {
                most_likely_next_sym
            } else {
                sample_from_pdf(&spa, samples[sample_num as usize]) as u32
            };
            output_seq.put_sym(new_sym);

            let new_log_loss;
            (state, new_log_loss) =
                self.compute_test_loss_on_slice_from_state(output_seq, state, sample_num, 1)?;
            log_loss += new_log_loss;
        }

        Ok(log_loss / (len as f64))
    }
}

#[cfg(test)]
mod tests {
    use crate::sequence::{BinarySequence, CharacterSequence, U16Sequence, U32Sequence};
    use itertools::Itertools;
    use rand::{thread_rng, Rng};

    use super::*;

    #[test]
    fn sanity_check_streaming() {
        let mut all_data: Vec<u16> = Vec::new();
        let mut encoder: StreamingLZ78Encoder = StreamingLZ78Encoder::new(10);
        for _ in 0..20 {
            let new_vec = vec![
                0, 1, 2, 5, 9, 3, 4, 0, 1, 2, 5, 9, 4, 4, 4, 5, 5, 6, 7, 8, 9, 1, 2, 3,
            ];
            all_data.extend(new_vec.clone());
            let new_input = U16Sequence::from_data(new_vec, 10).expect("failed to create sequence");
            encoder.encode_block(&new_input).expect("could not encode");
        }
        let mut output = U16Sequence::new(10);
        encoder.decode(&mut output).expect("decoding failed");
        assert_eq!(all_data, output.data);
    }

    #[test]
    fn test_streaming_long() {
        let mut all_data: Vec<u32> = Vec::new();
        let alphabet_size = 100;
        let mut encoder: StreamingLZ78Encoder = StreamingLZ78Encoder::new(alphabet_size);
        let max_n = 10_000;

        let mut rng = thread_rng();
        for _ in 0..200 {
            let n = rng.gen_range(1..max_n);

            let new_vec = Uniform::new(0, alphabet_size)
                .sample_iter(&mut rng)
                .take(n as usize)
                .collect_vec();
            all_data.extend(new_vec.clone());
            let new_input =
                U32Sequence::from_data(new_vec, alphabet_size).expect("failed to create sequence");
            encoder.encode_block(&new_input).expect("could not encode");
        }
        let mut output = U32Sequence::new(alphabet_size);
        encoder.decode(&mut output).expect("decoding failed");
        assert_eq!(all_data, output.data);
    }

    #[test]
    fn sanity_check_log_loss() {
        let input =
            BinarySequence::from_data(vec![0, 1].repeat(500)).expect("failed to create sequence");
        let mut spa: LZ78SPA = LZ78SPA::new(2, 0.5);
        spa.train_on_block(&input, false)
            .expect("failed to train spa");

        let loss1 = spa
            .compute_test_loss(
                &BinarySequence::from_data(vec![0, 1, 0, 1, 0, 1, 0, 1, 0, 1])
                    .expect("failed to create sequence"),
            )
            .expect("failed to compute test loss");
        let loss2 = spa
            .compute_test_loss(
                &BinarySequence::from_data(vec![0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
                    .expect("failed to create sequence"),
            )
            .expect("failed to compute test loss");

        print!("loss 1: {loss1}, loss 2: {loss2}");
        assert!(loss1 < loss2);
    }

    #[test]
    fn sanity_check_generation() {
        let input = CharacterSequence::from_data_inferred_character_map(
            "hello world! this is a test. i hope that text generation works well here. "
                .to_string()
                .repeat(200),
        );
        let mut spa: LZ78SPA = LZ78SPA::new(input.alphabet_size(), 0.5);
        spa.train_on_block(&input, false)
            .expect("failed to train spa");

        let mut generation_output = CharacterSequence::new(input.character_map.clone());
        spa.generate_data(
            &mut generation_output,
            100,
            10,
            0.0,
            10,
            Some(
                &CharacterSequence::from_data("hello ".to_string(), input.character_map.clone())
                    .expect("failed to create sequence"),
            ),
        )
        .expect("generating data failed");

        println!(
            "Temperature 0, seed \"hello\": {:?}",
            generation_output.data
        );

        let mut generation_output2 = CharacterSequence::new(input.character_map.clone());
        spa.generate_data(
            &mut generation_output2,
            100,
            10,
            1.0,
            1,
            Some(
                &CharacterSequence::from_data("hello ".to_string(), input.character_map.clone())
                    .expect("failed to create sequence"),
            ),
        )
        .expect("generating data failed");

        println!(
            "Temperature 1, topk 1, seed \"hello\": {:?}",
            generation_output2.data
        );

        assert_eq!(generation_output.data, generation_output2.data);

        let mut generation_output = CharacterSequence::new(input.character_map.clone());
        spa.generate_data(
            &mut generation_output,
            100,
            10,
            2.0,
            5,
            Some(
                &CharacterSequence::from_data("hello".to_string(), input.character_map.clone())
                    .expect("failed to create sequence"),
            ),
        )
        .expect("generating data failed");

        println!(
            "Temperature 2, topk 5, seed \"hello\": {:?}",
            generation_output.data
        );

        let mut generation_output = CharacterSequence::new(input.character_map.clone());
        spa.generate_data(
            &mut generation_output,
            100,
            10,
            0.5,
            10,
            Some(
                &CharacterSequence::from_data("hello".to_string(), input.character_map.clone())
                    .expect("failed to create sequence"),
            ),
        )
        .expect("generating data failed");

        println!(
            "Temperature 0.5, topk 10, seed \"hello\": {:?}",
            generation_output.data
        );
    }
}
