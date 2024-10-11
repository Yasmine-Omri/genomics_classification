use anyhow::Result;
use bitvec::field::BitField;
use bitvec::vec::BitVec;

use crate::sequence::Sequence;
use crate::tree::LZ78Tree;

/// Stores an encoded bitstream, as well as the alphabet size and length of the
/// original sequence
#[derive(Debug, Clone)]
pub struct EncodedSequence {
    data: BitVec,
    pub uncompressed_length: u64,
    pub alphabet_size: u32,
}

impl EncodedSequence {
    pub fn from_data(data: BitVec, uncompressed_length: u64, alphabet_size: u32) -> Self {
        Self {
            data,
            uncompressed_length,
            alphabet_size,
        }
    }

    /// This is only true if the uncompressed filesize is equal to the number
    /// of symbols times log2(alphabet size). In other cases (e.g., unicode
    /// text, where symbols can be from 1 to 4 bytes), you should divide the
    /// compressed length by the known uncompressed size.
    pub fn compression_ratio(&self) -> f32 {
        if self.uncompressed_length == 0 {
            0.0
        } else {
            (self.data.len() as f32)
                / (self.uncompressed_length as f32 * (self.alphabet_size as f32).log2())
        }
    }

    /// Length of the bitstream, rounded up to the nearest byte
    pub fn compressed_len_bytes(&self) -> u64 {
        (self.data.len() as u64 + 7) / 8
    }

    // In-place, truncates the underlying compressed bitstream
    pub fn truncate(&mut self, num_bits: u64) {
        self.data.drain(num_bits as usize..);
    }

    /// Pushes some bits to the bitstream
    pub fn push(&mut self, val: u64, bitwidth: u16) {
        let old_len = self.data.len();
        self.data.resize(self.data.len() + bitwidth as usize, false);
        self.data[old_len..].store(val);
    }

    pub fn set_uncompressed_len(&mut self, len: u64) {
        self.uncompressed_length = len;
    }

    pub fn extend_capacity(&mut self, n: u64) {
        self.data.reserve(n as usize);
    }

    /// Returns a reference to the underlying data array
    pub fn get_raw(&self) -> &BitVec {
        &self.data
    }
}

/// Generic interface for encoding and decoding. Anything that implements
/// this trait can be used to encode or decode sequences.
pub trait Encoder {
    fn encode<T>(&self, input: &T) -> Result<EncodedSequence>
    where
        T: Sequence;

    fn decode<T>(&self, output: &mut T, input: &EncodedSequence) -> Result<()>
    where
        T: Sequence;
}

/// Interface for encoding blocks of a dataset in a streaming fashion; i.e.,
/// the input is passed in as several blocks.
pub trait StreamingEncoder {
    fn encode_block<T>(&mut self, input: &T) -> Result<()>
    where
        T: Sequence;

    /// Returns the encoded sequence, which is the compressed version of the
    /// concatenation of all inputs to `encode_block`
    fn get_encoded_sequence(&self) -> &EncodedSequence;

    fn decode<T>(&self, output: &mut T) -> Result<()>
    where
        T: Sequence;
}

/// LZ78 encoder implementation
pub struct LZ8Encoder {}

impl Encoder for LZ8Encoder {
    fn encode<T>(&self, input: &T) -> Result<EncodedSequence>
    where
        T: Sequence,
    {
        lz78_encode(input)
    }

    fn decode<T>(&self, output: &mut T, input: &EncodedSequence) -> Result<()>
    where
        T: Sequence,
    {
        lz78_decode(output, input)
    }
}

impl LZ8Encoder {
    pub fn new() -> Self {
        Self {}
    }
}

/// returns the number of bits required to encode the specified phrase
pub fn lz78_bits_to_encode_phrase(phrase_idx: u64, alpha_size: u32) -> u16 {
    (((phrase_idx + 1) as f64).log2() + (alpha_size as f64).log2()).ceil() as u16
}

/// Compresses a sequence using LZ78
pub fn lz78_encode<T>(input: &T) -> Result<EncodedSequence>
where
    T: Sequence,
{
    // LZ78 prefix tree that is being built during the encoding process
    let mut tree: LZ78Tree = LZ78Tree::new(input.alphabet_size());

    // Every LZ78 phrase consists of a prefix that is a previously-seen phrase
    // (using the convention that the "empty phrase" is the first phrase),
    // plus one extra bit. The `ref_idxs` array indexes the phrase equal to the
    // prefix.
    let mut ref_idxs: Vec<u64> = Vec::new();
    // `output_leaves` is the final bit of every phrase
    let mut output_leaves: Vec<u32> = Vec::new();

    // The start of the current phrase
    let mut start_idx: u64 = 0;

    // Compute the LZ78 phrases and build the tree
    while start_idx < input.len() {
        // Process a single phrase
        let traversal_result =
            tree.traverse_root_to_leaf(input, start_idx, input.len(), true, true)?;

        // Finds the previous phrase in the LZ78 parsing that forms the prefix
        // of the current phrase
        ref_idxs.push(traversal_result.state_idx);
        // The last element of the current phrase
        output_leaves.push(traversal_result.added_leaf.unwrap_or(0));
        start_idx = traversal_result.phrase_end_idx + 1;
    }

    // compute number of bits we will need in total for the output
    let n_output_bits: u64 = (0..output_leaves.len())
        .map(|i| lz78_bits_to_encode_phrase(i as u64, input.alphabet_size()) as u64)
        .sum();

    // compute output
    let mut bits = BitVec::with_capacity(n_output_bits as usize);
    bits.resize(n_output_bits as usize, false);

    let mut prev_bit_idx = 0;

    // Encode each phrase
    for (i, (leaf, ref_idx)) in output_leaves.into_iter().zip(ref_idxs).enumerate() {
        let bitwidth = lz78_bits_to_encode_phrase(i as u64, input.alphabet_size());

        // value to encode, as per original LZ78 paper
        let val: u64 = ref_idx * (input.alphabet_size() as u64) + (leaf as u64);
        bits[prev_bit_idx..prev_bit_idx + bitwidth as usize].store(val);
        prev_bit_idx += bitwidth as usize;
    }

    Ok(EncodedSequence::from_data(
        bits,
        input.len(),
        input.alphabet_size(),
    ))
}

/// Decodes a sequence using LZ78
pub fn lz78_decode<T>(output: &mut T, input: &EncodedSequence) -> Result<()>
where
    T: Sequence,
{
    // the indices at which phrases start in the uncompressed sequence.
    // The first phrase, by convention, is the empty phrase.
    let mut phrase_starts: Vec<u64> = vec![0];
    // how long each phrase is (can be computed from phrase_starts, but this
    // makes decoding easier).
    let mut phrase_lengths: Vec<u64> = vec![0];

    // number of bits that have been decoded so far
    let mut bits_decoded: usize = 0;

    let alphabet_size = input.alphabet_size;

    while bits_decoded < input.data.len() {
        // number of bits that were used to store the current phrase
        let bitwidth: i32 =
            lz78_bits_to_encode_phrase(phrase_starts.len() as u64 - 1, alphabet_size) as i32;

        let decoded_val = input.data[bits_decoded..bits_decoded + bitwidth as usize].load::<u64>();
        bits_decoded += bitwidth as usize;

        // find the index of the previous phrase that forms the prefix of the
        // current phrase
        let ref_idx = decoded_val / (alphabet_size as u64);
        // the final symbol of the phrase
        let new_sym = (decoded_val % (alphabet_size as u64)) as u32;

        let phrase_start = output.len();
        let phrase_len = phrase_lengths[ref_idx as usize] + 1;
        let copy_start = phrase_starts[ref_idx as usize];

        for j in 0..phrase_len - 1 {
            output.put_sym(output.try_get(copy_start + j)?);
            if output.len() >= input.uncompressed_length {
                return Ok(());
            }
        }

        output.put_sym(new_sym);
        phrase_lengths.push(phrase_len);
        phrase_starts.push(phrase_start);
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use crate::sequence::{BinarySequence, CharacterMap, CharacterSequence, U8Sequence};
    use bitvec::prelude::*;
    use itertools::Itertools;
    use rand::{distributions::Uniform, prelude::Distribution, thread_rng};

    use super::*;

    #[test]
    fn test_encode_decode_binary() {
        let input = BinarySequence::from_data(bitvec![
            0, 0, 1, 1, 1, 1, 0, 1, 0, 0, 0, 1, 1, 1, 0, 0, 1, 0, 1, 0, 1, 0,
        ]);
        let encoder = LZ8Encoder::new();
        let encoded = encoder.encode(&input).expect("encoding failed");

        let mut output = BinarySequence::new();
        encoder
            .decode(&mut output, &encoded)
            .expect("decoding failed");
        assert_eq!(input.data, output.data);
    }

    #[test]
    fn test_encode_decode_string() {
        let charmap = CharacterMap::from_data(&"abcdefghijklmnopqrstuvwxyz ".to_string());
        let input = CharacterSequence::from_data(
            "hello hello hello hello hello world hello world hello world wxyz".to_string(),
            charmap.clone(),
        )
        .expect("input sequence invalid");
        let encoder = LZ8Encoder::new();
        let encoded = encoder.encode(&input).expect("encoding failed");

        let mut output = CharacterSequence::new(charmap);
        encoder
            .decode(&mut output, &encoded)
            .expect("decoding failed");
        assert_eq!(input.data, output.data);
    }

    #[test]
    fn test_encode_long_sequence() {
        let alphabet_size = 100;
        let n = 1_000_000;

        let mut rng = thread_rng();
        let input = U8Sequence::from_data(
            Uniform::new(0, alphabet_size)
                .sample_iter(&mut rng)
                .take(n as usize)
                .collect_vec(),
            alphabet_size as u32,
        )
        .expect("creating sequence failed");

        let encoder = LZ8Encoder::new();
        let encoded = encoder.encode(&input).expect("encoding failed");

        let mut output = U8Sequence::new(alphabet_size as u32);
        encoder
            .decode(&mut output, &encoded)
            .expect("decoding failed");
        assert_eq!(input.data, output.data);
    }
}
