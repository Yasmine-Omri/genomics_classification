use lz78::sequence::{CharacterSequence, Sequence as Sequence_LZ78, U32Sequence, U8Sequence};
use pyo3::{exceptions::PyAssertionError, prelude::*};

#[derive(Clone)]
pub enum SequenceType {
    U8(U8Sequence),
    Char(CharacterSequence),
    U32(U32Sequence),
}

impl SequenceType {
    fn alphabet_size(&self) -> u32 {
        match self {
            SequenceType::U8(u8_sequence) => u8_sequence.alphabet_size(),
            SequenceType::Char(character_sequence) => character_sequence.alphabet_size(),
            SequenceType::U32(u32_sequence) => u32_sequence.alphabet_size(),
        }
    }

    fn len(&self) -> u64 {
        match self {
            SequenceType::U8(u8_sequence) => u8_sequence.len(),
            SequenceType::Char(character_sequence) => character_sequence.len(),
            SequenceType::U32(u32_sequence) => u32_sequence.len(),
        }
    }

    fn get(&self, i: u64) -> PyResult<u32> {
        Ok(match self {
            SequenceType::U8(u8_sequence) => u8_sequence.try_get(i)?,
            SequenceType::Char(character_sequence) => character_sequence.try_get(i)?,
            SequenceType::U32(u32_sequence) => u32_sequence.try_get(i)?,
        })
    }
}

#[pyclass]
#[derive(Clone)]
pub struct Sequence {
    pub sequence: SequenceType,
}

#[pymethods]
impl Sequence {
    #[new]
    #[pyo3(signature = (alphabet_size, data=None))]
    pub fn new<'py>(alphabet_size: u32, data: Option<Bound<'py, PyAny>>) -> PyResult<Self> {
        if let Some(obj) = data {
            if alphabet_size <= 256 {
                return Ok(Self {
                    sequence: SequenceType::U8(U8Sequence::from_data(
                        obj.extract::<Vec<u8>>()?,
                        alphabet_size,
                    )?),
                });
            } else {
                return Ok(Self {
                    sequence: SequenceType::U32(U32Sequence::from_data(
                        obj.extract::<Vec<u32>>()?,
                        alphabet_size,
                    )?),
                });
            }
        }

        todo!()
    }

    pub fn extend<'py>(&mut self, data: Bound<'py, PyAny>) -> PyResult<()> {
        match &mut self.sequence {
            SequenceType::U8(u8_sequence) => {
                let new_seq = data.extract::<Vec<u8>>()?;
                u8_sequence.extend(&new_seq)?;
            }
            SequenceType::Char(character_sequence) => todo!(),
            SequenceType::U32(u32_sequence) => {
                let new_seq = data.extract::<Vec<u32>>()?;
                u32_sequence.extend(&new_seq)?;
            }
        }

        Ok(())
    }

    pub fn alphabet_size(&self) -> PyResult<u32> {
        Ok(self.sequence.alphabet_size())
    }

    fn __len__(&self) -> usize {
        self.sequence.len() as usize
    }

    fn __getitem__(&self, i: usize) -> PyResult<u32> {
        self.sequence.get(i as u64)
    }

    // #[pyo3(signature = (start_idx=None, end_idx=None))]
    // fn get_data(&self, start_idx: Option<u64>, end_idx: Option<u64>) -> PyResult<Vec<u8>> {
    //     let start_idx = start_idx.unwrap_or(0);
    //     let end_idx = end_idx.unwrap_or(self.len()? as u64);
    //     Ok(self.sequence.data[start_idx as usize..end_idx as usize].to_vec())
    // }
}

#[pyclass]
#[derive(Clone)]
pub struct CharacterMap {
    map: lz78::sequence::CharacterMap,
}

#[pymethods]
impl CharacterMap {
    #[new]
    pub fn new(data: String) -> PyResult<Self> {
        Ok(Self {
            map: lz78::sequence::CharacterMap::from_data(&data),
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
