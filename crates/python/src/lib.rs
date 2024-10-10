pub mod encoder;
pub mod sequence;
use encoder::*;
use pyo3::prelude::*;
use sequence::*;

#[pymodule]
fn lz78(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<CharacterMap>()?;
    m.add_class::<Sequence>()?;
    m.add_class::<EncodedSequence>()?;
    m.add_class::<LZ78Encoder>()?;
    Ok(())
}
