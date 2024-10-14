use bytes::{Buf, BufMut, Bytes};
use lz78::{
    sequence::{CharacterSequence, U32Sequence, U8Sequence},
    spa::SPA,
};
use pyo3::{exceptions::PyAssertionError, prelude::*, types::PyBytes};

use crate::{Sequence, SequenceType};

#[pyclass]
pub struct LZ78SPA {
    spa: lz78::spa::LZ78SPA,
    alphabet_size: u32,
    empty_seq_of_correct_datatype: Option<SequenceType>,
}

#[pymethods]
impl LZ78SPA {
    #[new]
    #[pyo3(signature = (alphabet_size, gamma=0.5))]
    pub fn new(alphabet_size: u32, gamma: f64) -> PyResult<Self> {
        Ok(Self {
            spa: lz78::spa::LZ78SPA::new(alphabet_size, gamma),
            empty_seq_of_correct_datatype: None,
            alphabet_size,
        })
    }

    #[pyo3(signature = (input, include_prev_context=false))]
    pub fn train_on_block(&mut self, input: Sequence, include_prev_context: bool) -> PyResult<f64> {
        if self.empty_seq_of_correct_datatype.is_some() {
            self.empty_seq_of_correct_datatype
                .as_ref()
                .unwrap()
                .assert_types_match(&input.sequence)?;
        } else if self.alphabet_size != input.alphabet_size()? {
            return Err(PyAssertionError::new_err(format!(
                "Expected alphabet size of {}, got {}",
                self.alphabet_size,
                input.alphabet_size()?
            )));
        }
        Ok(match &input.sequence {
            crate::SequenceType::U8(u8_sequence) => {
                if self.empty_seq_of_correct_datatype.is_none() {
                    self.empty_seq_of_correct_datatype =
                        Some(SequenceType::U8(U8Sequence::new(input.alphabet_size()?)));
                }
                self.spa.train_on_block(u8_sequence, include_prev_context)?
            }
            crate::SequenceType::Char(character_sequence) => {
                if self.empty_seq_of_correct_datatype.is_none() {
                    self.empty_seq_of_correct_datatype = Some(SequenceType::Char(
                        CharacterSequence::new(character_sequence.character_map.clone()),
                    ));
                }
                self.spa
                    .train_on_block(character_sequence, include_prev_context)?
            }
            crate::SequenceType::U32(u32_sequence) => {
                if self.empty_seq_of_correct_datatype.is_none() {
                    self.empty_seq_of_correct_datatype =
                        Some(SequenceType::U32(U32Sequence::new(input.alphabet_size()?)));
                }
                self.spa
                    .train_on_block(u32_sequence, include_prev_context)?
            }
        })
    }

    #[pyo3(signature = (input, include_prev_context=false))]
    pub fn compute_test_loss(
        &mut self,
        input: Sequence,
        include_prev_context: bool,
    ) -> PyResult<f64> {
        if self.empty_seq_of_correct_datatype.is_some() {
            self.empty_seq_of_correct_datatype
                .as_ref()
                .unwrap()
                .assert_types_match(&input.sequence)?;
        } else {
            return Err(PyAssertionError::new_err("SPA hasn't been trained yet"));
        }

        Ok(match &input.sequence {
            crate::SequenceType::U8(u8_sequence) => self
                .spa
                .compute_test_loss(u8_sequence, include_prev_context)?,
            crate::SequenceType::Char(character_sequence) => self
                .spa
                .compute_test_loss(character_sequence, include_prev_context)?,
            crate::SequenceType::U32(u32_sequence) => self
                .spa
                .compute_test_loss(u32_sequence, include_prev_context)?,
        })
    }

    pub fn compute_spa_at_current_state(&self) -> Vec<f64> {
        self.spa.compute_spa_at_current_state()
    }

    pub fn get_normalized_log_loss(&self) -> f64 {
        self.spa.get_normalized_log_loss()
    }

    #[pyo3(signature = (len, min_context=0, temperature=0.1, top_k=10, seed_data=None))]
    pub fn generate_data(
        &mut self,
        len: u64,
        min_context: u64,
        temperature: f64,
        top_k: u32,
        seed_data: Option<Sequence>,
    ) -> PyResult<(Sequence, f64)> {
        if self.empty_seq_of_correct_datatype.is_some() && seed_data.is_some() {
            self.empty_seq_of_correct_datatype
                .as_ref()
                .unwrap()
                .assert_types_match(&seed_data.as_ref().unwrap().sequence)?;
        } else {
            return Err(PyAssertionError::new_err("SPA hasn't been trained yet"));
        }

        let mut output = self.empty_seq_of_correct_datatype.clone().unwrap();

        let loss = match seed_data {
            None => match &mut output {
                SequenceType::U8(u8_sequence) => self.spa.generate_data(
                    u8_sequence,
                    len,
                    min_context,
                    temperature,
                    top_k,
                    None,
                )?,
                SequenceType::Char(character_sequence) => self.spa.generate_data(
                    character_sequence,
                    len,
                    min_context,
                    temperature,
                    top_k,
                    None,
                )?,
                SequenceType::U32(u32_sequence) => self.spa.generate_data(
                    u32_sequence,
                    len,
                    min_context,
                    temperature,
                    top_k,
                    None,
                )?,
            },
            Some(seed_data) => match (&mut output, &seed_data.sequence) {
                (SequenceType::U8(output_seq), SequenceType::U8(seed_seq)) => {
                    self.spa.generate_data(
                        output_seq,
                        len,
                        min_context,
                        temperature,
                        top_k,
                        Some(seed_seq),
                    )?
                }
                (SequenceType::Char(output_seq), SequenceType::Char(seed_seq)) => {
                    self.spa.generate_data(
                        output_seq,
                        len,
                        min_context,
                        temperature,
                        top_k,
                        Some(seed_seq),
                    )?
                }
                (SequenceType::U32(output_seq), SequenceType::U32(seed_seq)) => {
                    self.spa.generate_data(
                        output_seq,
                        len,
                        min_context,
                        temperature,
                        top_k,
                        Some(seed_seq),
                    )?
                }
                _ => return Err(PyAssertionError::new_err("Unexpected seed data type")),
            },
        };

        Ok((Sequence { sequence: output }, loss))
    }

    pub fn to_bytes<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyBytes>> {
        let mut bytes = self.spa.to_bytes()?;
        match &self.empty_seq_of_correct_datatype {
            Some(seq) => {
                bytes.put_u8(0);
                bytes.extend(seq.to_bytes());
            }
            None => bytes.put_u8(1),
        };
        bytes.put_u32_le(self.alphabet_size);
        Ok(PyBytes::new_bound(py, &bytes))
    }
}

#[pyfunction]
pub fn spa_from_bytes<'py>(bytes: Py<PyBytes>, py: Python<'py>) -> PyResult<LZ78SPA> {
    let mut bytes: Bytes = bytes.as_bytes(py).to_owned().into();
    let spa = lz78::spa::LZ78SPA::from_bytes(&mut bytes)?;
    let empty_seq_of_correct_datatype = match bytes.get_u8() {
        0 => Some(SequenceType::from_bytes(&mut bytes)?),
        1 => None,
        _ => {
            return Err(PyAssertionError::new_err(
                "Error reading encoded sequence from bytes",
            ))
        }
    };
    let alphabet_size = bytes.get_u32_le();

    Ok(LZ78SPA {
        spa,
        alphabet_size,
        empty_seq_of_correct_datatype,
    })
}
