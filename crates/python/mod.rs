pub mod encoder;
pub mod sequence;
use encoder::*;
use pyo3::prelude::*;
use sequence::*;

#[pymodule]
fn lz78_rust(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<ByteSequence>()?;
    m.add_class::<IntSequence>()?;
    m.add_class::<CharacterMap>()?;
    m.add_class::<CharacterSequence>()?;
    m.add_class::<SequenceType>()?;
    m.add_class::<EncodedSequence>()?;
    m.add_class::<LZ78Encoder>()?;
    Ok(())
}
