use bytes::Bytes;
use lz78::{
    encoder::{Encoder, StreamingEncoder},
    sequence::{CharacterSequence, Sequence as Sequence_LZ78, U32Sequence, U8Sequence},
};
use pyo3::{exceptions::PyAssertionError, prelude::*, types::PyBytes};

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

    pub fn to_bytes<'py>(&self, py: Python<'py>) -> Bound<'py, PyBytes> {
        let mut bytes = self.encoded_sequence.to_bytes();
        bytes.extend(self.empty_seq_of_correct_datatype.to_bytes());
        PyBytes::new_bound(py, &bytes)
    }
}

#[pyfunction]
pub fn encoded_sequence_from_bytes<'py>(
    bytes: Py<PyBytes>,
    py: Python<'py>,
) -> PyResult<EncodedSequence> {
    let mut bytes: Bytes = bytes.as_bytes(py).to_owned().into();
    let encoded_sequence = lz78::encoder::EncodedSequence::from_bytes(&mut bytes);
    let empty_seq_of_correct_datatype = SequenceType::from_bytes(&mut bytes)?;

    Ok(EncodedSequence {
        encoded_sequence,
        empty_seq_of_correct_datatype,
    })
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

#[derive(Clone)]
#[pyclass]
pub struct StreamingLZ78Encoder {
    encoder: lz78::spa::StreamingLZ78Encoder,
    empty_seq_of_correct_datatype: Option<SequenceType>,
}

#[pymethods]
impl StreamingLZ78Encoder {
    #[new]
    fn new(alpha_size: u32) -> PyResult<Self> {
        Ok(Self {
            encoder: lz78::spa::StreamingLZ78Encoder::new(alpha_size),
            empty_seq_of_correct_datatype: None,
        })
    }

    fn encode_block(&mut self, input: Sequence) -> PyResult<()> {
        match input.sequence {
            SequenceType::U8(x) => {
                self.encoder.encode_block(&x)?;
                self.empty_seq_of_correct_datatype =
                    Some(SequenceType::U8(U8Sequence::new(x.alphabet_size())));
            }
            SequenceType::U32(x) => {
                self.encoder.encode_block(&x)?;
                self.empty_seq_of_correct_datatype =
                    Some(SequenceType::U32(U32Sequence::new(x.alphabet_size())))
            }
            SequenceType::Char(x) => {
                self.encoder.encode_block(&x)?;
                self.empty_seq_of_correct_datatype = Some(SequenceType::Char(
                    CharacterSequence::new(x.character_map.clone()),
                ))
            }
        };

        Ok(())
    }

    fn get_encoded_sequence(&mut self) -> PyResult<EncodedSequence> {
        if self.empty_seq_of_correct_datatype.is_none() {
            return Err(PyAssertionError::new_err("No blocks have been encoded yet"));
        }
        Ok(EncodedSequence {
            encoded_sequence: self.encoder.get_encoded_sequence().to_owned(),
            empty_seq_of_correct_datatype: self.empty_seq_of_correct_datatype.clone().unwrap(),
        })
    }

    fn decode(&self) -> PyResult<Sequence> {
        if self.empty_seq_of_correct_datatype.is_none() {
            return Err(PyAssertionError::new_err("No blocks have been encoded yet"));
        }
        let output = match self.empty_seq_of_correct_datatype.clone().unwrap() {
            SequenceType::U8(mut seq) => {
                self.encoder.decode(&mut seq)?;
                Sequence {
                    sequence: SequenceType::U8(seq),
                }
            }
            SequenceType::U32(mut seq) => {
                self.encoder.decode(&mut seq)?;
                Sequence {
                    sequence: SequenceType::U32(seq),
                }
            }
            SequenceType::Char(mut seq) => {
                self.encoder.decode(&mut seq)?;
                Sequence {
                    sequence: SequenceType::Char(seq),
                }
            }
        };
        Ok(output)
    }
}
