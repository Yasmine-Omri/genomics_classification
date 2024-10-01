use anyhow::Result;

use crate::sequence::Sequence;
use crate::tree::LZ78Tree;

/// Stores an encoded bitstream, as well as the alphabet size and length of the
/// original sequence
#[derive(Debug, Clone)]
pub struct EncodedSequence {
    data: BitStorage,
    pub uncompressed_length: u64,
    pub alphabet_size: u32,
}

impl EncodedSequence {
    pub fn from_data(data: BitStorage, uncompressed_length: u64, alphabet_size: u32) -> Self {
        Self {
            data,
            uncompressed_length,
            alphabet_size,
        }
    }

    pub fn compression_ratio(&self) -> f32 {
        if self.uncompressed_length == 0 {
            0.0
        } else {
            (self.data.bit_len() as f32)
                / (self.uncompressed_length as f32 * (self.alphabet_size as f32).log2())
        }
    }

    /// Length of the bitstream, rounded up to the nearest byte
    pub fn compressed_len_bytes(&self) -> u64 {
        (self.data.bit_len() + 7) / 8
    }

    // In-place, truncates the underlying compressed bitstream
    pub fn truncate(&mut self, num_bits: u64) {
        self.data.truncate(num_bits);
    }

    /// Pushes some bits to the bitstream
    pub fn push(&mut self, val: u64, bitwidth: u16) {
        self.data.push(val, bitwidth)
    }

    pub fn set_uncompressed_len(&mut self, len: u64) {
        self.uncompressed_length = len;
    }

    pub fn extend_capacity(&mut self, n: u64) {
        self.data.extend_capacity(n);
    }

    /// Returns a slice of the bitstream, as a byte array
    pub fn get_raw(&self) -> &[u8] {
        &self.data.data
    }
}

/// Generic interface for encoding and decoding
pub trait Encoder {
    fn encode<T>(&self, input: &T) -> Result<EncodedSequence>
    where
        T: Sequence;

    fn decode<T>(&self, output: &mut T, input: &EncodedSequence) -> Result<()>
    where
        T: Sequence;
}

/// Interface for encoding blocks of a dataset in a streaming fashion
pub trait StreamingEncoder<T: Sequence> {
    fn encode_block(&mut self, input: &T) -> Result<()>;

    fn get_encoded_sequence(&self) -> &EncodedSequence;

    fn decode(&self, output: &mut T) -> Result<()>;
}

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

/// Container for storing a bitstream, to which integers of arbitrary bitwidth
/// can be pushed.
#[derive(Debug, Clone)]
pub struct BitStorage {
    data: Vec<u8>,
    /// number of bits, modulo 8, i.e., the number of bits that do not evenly
    /// fit into a byte array. These are stored in the last index of `data`
    overflow_len: u16,
}

impl BitStorage {
    pub fn new() -> Self {
        Self {
            data: Vec::new(),
            overflow_len: 0,
        }
    }

    /// Extends the capacity of the underlying byte array; used for performance
    /// reasons in streaming compression
    pub fn extend_capacity(&mut self, n: u64) {
        self.data.reserve(n as usize);
    }

    /// Push some bits to the bitstream
    pub fn push(&mut self, val: u64, bitwidth: u16) {
        let mut buffer = if self.overflow_len > 0 {
            self.data.pop().unwrap_or(0)
        } else {
            0
        } as u64;

        buffer += val << self.overflow_len;
        self.overflow_len += bitwidth;

        while self.overflow_len >= 8 {
            self.data.push((buffer & 0xFF) as u8);
            buffer >>= 8;
            self.overflow_len -= 8;
        }
        if self.overflow_len > 0 {
            self.data.push(buffer as u8);
        }
    }

    /// Trucates the bitstream to `num_bits` bits
    pub fn truncate(&mut self, num_bits: u64) {
        self.data.truncate(((num_bits + 7) / 8) as usize);
        self.overflow_len = (num_bits % 8) as u16;
        if self.overflow_len > 0 {
            let t = self.data.len() - 1;
            self.data[t] &= (1 << self.overflow_len) - 1;
        }
    }

    /// Returns the number of bits being stored
    pub fn bit_len(&self) -> u64 {
        if self.overflow_len == 0 {
            self.data.len() as u64 * 8
        } else {
            (self.data.len() as u64 - 1) * 8 + self.overflow_len as u64
        }
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
    let mut tree: LZ78Tree = LZ78Tree::new(input.alphabet_size());
    let mut ref_idxs: Vec<Option<u64>> = Vec::new();
    let mut output_leaves: Vec<u32> = Vec::new();

    let mut start_idx: u64 = 0;

    // Compute the LZ78 phrases and build the tree
    while start_idx < input.len() {
        // Process a single phrase
        let traversal_result =
            tree.traverse_root_to_leaf(input, start_idx, input.len(), true, true)?;

        // Finds the previous phrase in the LZ78 parsing that forms the prefix
        // of the current phrase
        ref_idxs.push(tree.phrase_num(traversal_result.state_idx));
        // The last element of the current phrase
        output_leaves.push(traversal_result.added_leaf.unwrap_or(0));
        start_idx = traversal_result.phrase_end_idx + 1;
    }

    // compute output
    let mut bits = BitStorage::new();

    // compute number of bits we will need in total for the output
    let n_output_bits: u64 = (0..output_leaves.len())
        .map(|i| lz78_bits_to_encode_phrase(i as u64, input.alphabet_size()) as u64)
        .sum();
    bits.extend_capacity((n_output_bits + 7) / 8);

    // Encode each phrase in the bitarray
    for (i, (leaf, ref_idx)) in output_leaves.into_iter().zip(ref_idxs).enumerate() {
        let ref_idx = if let Some(x) = ref_idx { x + 1 } else { 0 };
        let bitwidth = lz78_bits_to_encode_phrase(i as u64, input.alphabet_size());
        let val = if i == 0 {
            leaf as u64
        } else {
            ref_idx * (input.alphabet_size() as u64) + (leaf as u64)
        };

        bits.push(val, bitwidth);
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
    // the indices at which phrases start in the uncompressed sequence
    let mut phrase_starts: Vec<u64> = Vec::new();
    // how long each phrase is (can be computed from phrase_starts, but this
    // makes decoding easier)
    let mut phrase_lengths: Vec<u64> = Vec::new();

    let mut bits_decoded: u64 = 0;
    
    let mut current_bit_offset: u16 = 0;

    let mut input_idx = 0;
    let raw_input = input.get_raw();

    let alphabet_size = input.alphabet_size;

    while bits_decoded < input.data.bit_len() {
        let i = phrase_starts.len();
        let mut bitwidth: i32 = lz78_bits_to_encode_phrase(i as u64, alphabet_size) as i32;
        bits_decoded += bitwidth as u64;

        let mut decoded_val: u64 =
            ((raw_input[input_idx] >> current_bit_offset) as u64) & ((1 << bitwidth) - 1);
        input_idx += 1;

        // We read (8 - current_bit_offset) bits
        bitwidth -= 8 - current_bit_offset as i32;

        let mut write_offset = 8 - current_bit_offset;
        current_bit_offset = 0;

        while bitwidth > 0 {
            decoded_val += (raw_input[input_idx] as u64 & ((1 << bitwidth) - 1)) << write_offset;
            write_offset += 8;
            bitwidth -= 8;
            input_idx += 1;
        }

        if bitwidth < 0 {
            current_bit_offset = (bitwidth + 8) as u16;
            input_idx -= 1;
        }

        let mut ref_idx = decoded_val / (alphabet_size as u64);
        let new_sym = (decoded_val % (alphabet_size as u64)) as u32;
        let mut phrase_len = 1;

        let phrase_start = output.len();
        if ref_idx >= 1 {
            ref_idx -= 1;
            phrase_len = phrase_lengths[ref_idx as usize] + 1;
            let copy_start = phrase_starts[ref_idx as usize];

            for j in 0..phrase_len - 1 {
                output.put_sym(output.get(copy_start + j)?);
                if output.len() >= input.uncompressed_length {
                    return Ok(());
                }
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
    use itertools::Itertools;
    use rand::{distributions::Uniform, prelude::Distribution, thread_rng};

    use super::*;

    #[test]
    fn test_encode_decode_binary() {
        let input = BinarySequence::from_data(vec![
            0, 0, 1, 1, 1, 1, 0, 1, 0, 0, 0, 1, 1, 1, 0, 0, 1, 0, 1, 0, 1, 0,
        ])
        .expect("creating sequence failed");
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
