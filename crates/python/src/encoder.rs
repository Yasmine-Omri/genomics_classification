use lz78::{
    encoder::Encoder,
    sequence::{CharacterSequence, Sequence as Sequence_LZ78, U32Sequence, U8Sequence},
};
use pyo3::prelude::*;

use crate::{sequence::SequenceType, Sequence};

#[derive(Clone)]
#[pyclass]
pub struct EncodedSequence {
    encoded_sequence: lz78::encoder::EncodedSequence,
    /// Used for inferring the right type when decoding
    empty_seq_of_correct_datatype: SequenceType,
}

#[pymethods]
impl EncodedSequence {
    pub fn compression_ratio(&self) -> PyResult<f32> {
        Ok(self.encoded_sequence.compression_ratio())
    }

    pub fn get_raw(&self) -> PyResult<Vec<usize>> {
        let mut v = self.encoded_sequence.get_raw().to_owned();
        v.force_align();
        Ok(v.into_vec())
    }
}

#[pyclass]
pub struct LZ78Encoder {
    encoder: lz78::encoder::LZ8Encoder,
}

#[pymethods]
impl LZ78Encoder {
    #[new]
    fn new() -> PyResult<Self> {
        Ok(Self {
            encoder: lz78::encoder::LZ8Encoder::new(),
        })
    }

    fn encode(&self, input: Sequence) -> PyResult<EncodedSequence> {
        let (encoded_sequence, empty_seq_of_correct_datatype) = match input.sequence {
            SequenceType::U8(x) => (
                self.encoder.encode(&x)?,
                SequenceType::U8(U8Sequence::new(x.alphabet_size())),
            ),
            SequenceType::U32(x) => (
                self.encoder.encode(&x)?,
                SequenceType::U32(U32Sequence::new(x.alphabet_size())),
            ),
            SequenceType::Char(x) => (
                self.encoder.encode(&x)?,
                SequenceType::Char(CharacterSequence::new(x.character_map.clone())),
            ),
        };

        Ok(EncodedSequence {
            encoded_sequence,
            empty_seq_of_correct_datatype,
        })
    }

    fn decode(&self, input: EncodedSequence) -> PyResult<Sequence> {
        let output = match &input.empty_seq_of_correct_datatype {
            SequenceType::U8(x) => {
                let mut seq = x.clone();
                self.encoder.decode(&mut seq, &input.encoded_sequence)?;
                Sequence {
                    sequence: SequenceType::U8(seq),
                }
            }
            SequenceType::U32(x) => {
                let mut seq = x.clone();
                self.encoder.decode(&mut seq, &input.encoded_sequence)?;
                Sequence {
                    sequence: SequenceType::U32(seq),
                }
            }
            SequenceType::Char(x) => {
                let mut seq = x.clone();
                self.encoder.decode(&mut seq, &input.encoded_sequence)?;
                Sequence {
                    sequence: SequenceType::Char(seq),
                }
            }
        };
        Ok(output)
    }
}
