use pyo3::{exceptions::PyAssertionError, prelude::*};

use crate::encoder::{BitStorage, Encoder};

use super::sequence::{ByteSequence, IntSequence, SequenceType};

#[derive(Clone)]
#[pyclass]
pub struct EncodedSequence {
    encoded_sequence: crate::encoder::EncodedSequence,
}

#[pymethods]
impl EncodedSequence {
    pub fn compression_ratio(&self) -> PyResult<f32> {
        Ok(self.encoded_sequence.compression_ratio())
    }

    pub fn get_raw(&self) -> PyResult<Vec<u8>> {
        Ok(self.encoded_sequence.get_raw().to_vec())
    }
}

#[pyclass]
pub struct LZ78Encoder {
    encoder: crate::encoder::LZ8Encoder,
}

#[pymethods]
impl LZ78Encoder {
    #[new]
    fn new() -> PyResult<Self> {
        Ok(Self {
            encoder: crate::encoder::LZ8Encoder::new(),
        })
    }

    fn encode(&self, input: SequenceType) -> PyResult<EncodedSequence> {
        let encoded_sequence = match input {
            SequenceType::Byte(x) => self.encoder.encode(&x.sequence)?,
            SequenceType::Int(x) => self.encoder.encode(&x.sequence)?,
            SequenceType::Character(x) => self.encoder.encode(&x.sequence)?,
        };
        Ok(EncodedSequence { encoded_sequence })
    }

    fn decode(&self, input: EncodedSequence, mut output: SequenceType) -> PyResult<SequenceType> {
        match &mut output {
            SequenceType::Byte(x) => self
                .encoder
                .decode(&mut x.sequence, &input.encoded_sequence)?,
            SequenceType::Int(x) => self
                .encoder
                .decode(&mut x.sequence, &input.encoded_sequence)?,
            SequenceType::Character(x) => self
                .encoder
                .decode(&mut x.sequence, &input.encoded_sequence)?,
        };
        Ok(output)
    }
}
