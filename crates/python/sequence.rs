use pyo3::{exceptions::PyAssertionError, prelude::*};

use crate::sequence::Sequence;

#[pyclass]
#[derive(Clone)]
pub struct ByteSequence {
    pub sequence: crate::sequence::U8Sequence,
}

#[pymethods]
impl ByteSequence {
    #[new]
    #[pyo3(signature = (alphabet_size, data=None))]
    pub fn new(alphabet_size: u8, data: Option<Vec<u8>>) -> PyResult<Self> {
        Ok(if let Some(x) = data {
            Self {
                sequence: crate::sequence::U8Sequence::from_data(x, alphabet_size)?,
            }
        } else {
            Self {
                sequence: crate::sequence::U8Sequence::new(alphabet_size),
            }
        })
    }

    pub fn extend(&mut self, data: Vec<u8>) -> PyResult<()> {
        self.sequence.extend(&data)?;
        Ok(())
    }

    pub fn alphabet_size(&self) -> PyResult<u32> {
        Ok(self.sequence.alphabet_size())
    }

    fn len(&self) -> PyResult<u64> {
        Ok(self.sequence.len())
    }

    fn get(&self, i: u64) -> PyResult<u32> {
        Ok(self.sequence.get(i)?)
    }

    fn put_sym(&mut self, sym: u32) -> PyResult<()> {
        self.sequence.put_sym(sym);
        Ok(())
    }

    #[pyo3(signature = (start_idx=None, end_idx=None))]
    fn get_data(&self, start_idx: Option<u64>, end_idx: Option<u64>) -> PyResult<Vec<u8>> {
        let start_idx = start_idx.unwrap_or(0);
        let end_idx = end_idx.unwrap_or(self.len()? as u64);
        Ok(self.sequence.data[start_idx as usize..end_idx as usize].to_vec())
    }
}

#[pyclass]
#[derive(Clone)]
pub struct IntSequence {
    pub sequence: crate::sequence::U32Sequence,
}

#[pymethods]
impl IntSequence {
    #[new]
    #[pyo3(signature = (alphabet_size, data=None))]
    pub fn new(alphabet_size: u32, data: Option<Vec<u32>>) -> PyResult<Self> {
        Ok(if let Some(x) = data {
            Self {
                sequence: crate::sequence::U32Sequence::from_data(x, alphabet_size)?,
            }
        } else {
            Self {
                sequence: crate::sequence::U32Sequence::new(alphabet_size),
            }
        })
    }

    pub fn extend(&mut self, data: Vec<u32>) -> PyResult<()> {
        self.sequence.extend(&data)?;
        Ok(())
    }

    pub fn alphabet_size(&self) -> PyResult<u32> {
        Ok(self.sequence.alphabet_size())
    }

    fn len(&self) -> PyResult<u64> {
        Ok(self.sequence.len())
    }

    fn get(&self, i: u64) -> PyResult<u32> {
        Ok(self.sequence.get(i)?)
    }

    fn put_sym(&mut self, sym: u32) -> PyResult<()> {
        self.sequence.put_sym(sym);
        Ok(())
    }

    #[pyo3(signature = (start_idx=None, end_idx=None))]
    fn get_data(&self, start_idx: Option<u64>, end_idx: Option<u64>) -> PyResult<Vec<u32>> {
        let start_idx = start_idx.unwrap_or(0);
        let end_idx = end_idx.unwrap_or(self.len()? as u64);
        Ok(self.sequence.data[start_idx as usize..end_idx as usize].to_vec())
    }
}

#[pyclass]
#[derive(Clone)]
pub struct CharacterMap {
    map: crate::sequence::CharacterMap,
}

#[pymethods]
impl CharacterMap {
    #[new]
    pub fn new(data: String) -> PyResult<Self> {
        Ok(Self {
            map: crate::sequence::CharacterMap::from_data(&data),
        })
    }

    fn encode(&self, char: String) -> PyResult<u32> {
        Ok(self
            .map
            .encode(&char)
            .ok_or(PyAssertionError::new_err("could not encode character"))?)
    }

    fn decode(&self, sym: u32) -> PyResult<String> {
        Ok(self
            .map
            .decode(sym)
            .ok_or(PyAssertionError::new_err("could not decode character"))?)
    }

    fn filter_string(&self, data: String) -> PyResult<String> {
        Ok(self.map.filter_string(&data))
    }
}

#[pyclass]
#[derive(Clone)]
pub struct CharacterSequence {
    pub sequence: crate::sequence::CharacterSequence,
}

#[pymethods]
impl CharacterSequence {
    #[new]
    #[pyo3(signature = (character_map, data=None, filter_data=false))]
    pub fn new(
        character_map: CharacterMap,
        data: Option<String>,
        filter_data: bool,
    ) -> PyResult<Self> {
        let data = data.unwrap_or("".to_string());
        Ok(if filter_data {
            Self {
                sequence: crate::sequence::CharacterSequence::from_data_filtered(
                    data,
                    character_map.map,
                ),
            }
        } else {
            Self {
                sequence: crate::sequence::CharacterSequence::from_data(data, character_map.map)?,
            }
        })
    }

    pub fn alphabet_size(&self) -> PyResult<u32> {
        Ok(self.sequence.alphabet_size())
    }

    fn len(&self) -> PyResult<u64> {
        Ok(self.sequence.len())
    }

    fn get(&self, i: u64) -> PyResult<u32> {
        Ok(self.sequence.get(i)?)
    }

    fn put_sym(&mut self, sym: u32) -> PyResult<()> {
        self.sequence.put_sym(sym);
        Ok(())
    }

    #[pyo3(signature = (start_idx=None, end_idx=None))]
    fn get_data(&self, start_idx: Option<u64>, end_idx: Option<u64>) -> PyResult<String> {
        let start_idx = start_idx.unwrap_or(0);
        let end_idx = end_idx.unwrap_or(self.len()? as u64);
        Ok(self.sequence.data[start_idx as usize..end_idx as usize].to_string())
    }
}

#[pyclass]
#[derive(Clone)]
pub enum SequenceType {
    Byte(ByteSequence),
    Int(IntSequence),
    Character(CharacterSequence),
}
