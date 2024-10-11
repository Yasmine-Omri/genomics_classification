use lz78::sequence::{CharacterSequence, Sequence as Sequence_LZ78, U32Sequence, U8Sequence};
use pyo3::{
    exceptions::PyAssertionError,
    prelude::*,
    types::{PyList, PySlice},
};

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
    #[pyo3(signature = (data, alphabet_size=None, charmap=None))]
    pub fn new<'py>(
        data: Bound<'py, PyAny>,
        alphabet_size: Option<u32>,
        charmap: Option<CharacterMap>,
    ) -> PyResult<Self> {
        // check if this is a string
        let maybe_input_string = data.extract::<String>();
        if charmap.is_some() || maybe_input_string.is_ok() {
            if let Some(cmap) = charmap {
                return Ok(Self {
                    sequence: SequenceType::Char(CharacterSequence::from_data(
                        maybe_input_string.unwrap(),
                        cmap.map,
                    )?),
                });
            } else {
                return Ok(Self {
                    sequence: SequenceType::Char(
                        CharacterSequence::from_data_inferred_character_map(
                            maybe_input_string.unwrap(),
                        ),
                    ),
                });
            }
        }

        let alphabet_size = match alphabet_size {
            Some(x) => x,
            None => data.extract::<Vec<u32>>()?.into_iter().max().unwrap_or(0) + 1,
        };

        if alphabet_size <= 256 {
            return Ok(Self {
                sequence: SequenceType::U8(U8Sequence::from_data(
                    data.extract::<Vec<u8>>()?,
                    alphabet_size,
                )?),
            });
        } else {
            return Ok(Self {
                sequence: SequenceType::U32(U32Sequence::from_data(
                    data.extract::<Vec<u32>>()?,
                    alphabet_size,
                )?),
            });
        }
    }

    pub fn extend<'py>(&mut self, data: Bound<'py, PyAny>) -> PyResult<()> {
        match &mut self.sequence {
            SequenceType::U8(u8_sequence) => {
                let new_seq = data.extract::<Vec<u8>>()?;
                u8_sequence.extend(&new_seq)?;
            }
            SequenceType::Char(character_sequence) => {
                let new_seq = data.extract::<String>()?;
                character_sequence.extend(&new_seq)?;
            }
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

    fn __getitem__<'py>(&self, py: Python<'py>, i: Bound<'py, PyAny>) -> PyResult<PyObject> {
        if let Ok(i) = i.extract::<isize>() {
            let i = if i < 0 {
                (self.__len__() as isize + i) as u64
            } else {
                i as u64
            };
            return Ok(self.sequence.get(i)?.to_object(py));
        } else if i.is_instance_of::<PySlice>() {
            // handle indexing by a slice
            let res = PyList::empty_bound(py);

            // start defaults to 0
            let mut start = if i.getattr("start")?.is_none() {
                0
            } else {
                i.getattr("start")?.extract::<isize>()?
            };
            if start < 0 {
                start = self.__len__() as isize - start;
            }

            // stop defaults to the end of the sequence
            let mut stop = if i.getattr("stop")?.is_none() {
                self.__len__() as isize
            } else {
                i.getattr("stop")?.extract::<isize>()?
            };
            if stop < 0 {
                stop = self.__len__() as isize - stop;
            }

            // step defaults to 1
            let step = if i.getattr("step")?.is_none() {
                1
            } else {
                i.getattr("step")?.extract::<isize>()?
            };

            for idx in (start..stop).step_by(step.abs() as usize) {
                res.append(self.sequence.get(idx as u64)?)?;
            }
            if step < 0 {
                res.reverse()?;
            }

            return Ok(res.unbind().into_any());
        }
        Err(PyAssertionError::new_err(
            "Expected a slice or integer index",
        ))
    }
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

    fn encode_all(&self, s: String) -> PyResult<Vec<u32>> {
        Ok(self.map.try_encode_all(&s)?)
    }

    fn decode(&self, sym: u32) -> PyResult<String> {
        Ok(self
            .map
            .decode(sym)
            .ok_or(PyAssertionError::new_err("could not decode character"))?)
    }

    fn decode_all(&self, syms: Vec<u32>) -> PyResult<String> {
        Ok(self.map.try_decode_all(syms)?)
    }

    fn filter_string(&self, data: String) -> PyResult<String> {
        Ok(self.map.filter_string(&data))
    }

    pub fn alphabet_size(&self) -> PyResult<u32> {
        Ok(self.map.alphabet_size)
    }
}
