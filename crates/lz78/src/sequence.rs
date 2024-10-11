use std::{
    collections::{HashMap, HashSet},
    fs::File,
    io::{Read, Write},
};

use anyhow::{anyhow, bail, Result};
use bitvec::vec::BitVec;
use bytes::{Buf, BufMut, Bytes};

/// Interface for dealing with individual sequences, with the basic operations
/// that need to be performed on an individual sequence
pub trait Sequence: Sync {
    fn alphabet_size(&self) -> u32;

    fn len(&self) -> u64;

    /// Returns the symbol at the provided index, or returns an error if the
    /// index is past the end of the sequence
    fn try_get(&self, i: u64) -> Result<u32>;

    /// Puts a u32 symbol into the array. This does not check if the symbol
    /// is less than the alphabet size, but maybe it should.
    fn put_sym(&mut self, sym: u32);
}

/// Stored a binary sequence as a BitVec
pub struct BinarySequence {
    pub data: BitVec,
}

impl Sequence for BinarySequence {
    fn alphabet_size(&self) -> u32 {
        2
    }

    fn len(&self) -> u64 {
        self.data.len() as u64
    }

    fn try_get(&self, i: u64) -> Result<u32> {
        if i >= self.len() {
            bail!("invalid index {i} for sequence of length {}", self.len());
        }
        Ok(self.data[i as usize] as u32)
    }

    fn put_sym(&mut self, sym: u32) {
        self.data.push(sym != 0);
    }
}

impl BinarySequence {
    pub fn new() -> Self {
        Self {
            data: BitVec::new(),
        }
    }
    pub fn from_data(data: BitVec) -> Self {
        Self { data }
    }

    pub fn extend(&mut self, data: &BitVec) {
        self.data.extend(data);
    }
}

/// Maps characters in a string to u32 values in a contiguous range, so that a
/// string can be used as an individual sequence. The string must be valid
/// UTF-8 to construct a CharacterMap.
#[derive(Debug, Clone)]
pub struct CharacterMap {
    /// Maps characters to the corresponding integer value
    char_to_sym: HashMap<String, u32>,
    /// Maps integer values to the corresponding character
    pub sym_to_char: Vec<String>,
    pub alphabet_size: u32,
}

impl CharacterMap {
    /// Creates a CharacterMap consisting of all the unique characters found in
    /// `data`, in the order that they appear in the string.
    pub fn from_data(data: &String) -> Self {
        let mut char_to_sym: HashMap<String, u32> = HashMap::new();
        // hashset of the characters seen so far, so that we can figure out if
        // a character is already present in the charactermap
        let mut charset: HashSet<String> = HashSet::new();
        let mut sym_to_char: Vec<String> = Vec::new();

        let mut alphabet_size = 0;

        for char in data.chars() {
            let key = char.to_string();
            if !charset.contains(&key) {
                // character hasn't been seen before
                char_to_sym.insert(key.clone(), alphabet_size);
                charset.insert(key.clone());
                sym_to_char.push(key);
                alphabet_size += 1;
            }
        }

        Self {
            alphabet_size,
            char_to_sym,
            sym_to_char,
        }
    }

    /// Returns the integer value corresponding to the single input character
    pub fn encode(&self, char: &String) -> Option<u32> {
        if self.char_to_sym.contains_key(char) {
            Some(self.char_to_sym[char])
        } else {
            None
        }
    }

    /// Loops through all characters in the string, and encodes each one,
    /// returning an error if any character is not found in the mapping.
    pub fn try_encode_all(&self, data: &String) -> Result<Vec<u32>> {
        let mut res = Vec::new();
        for char in data.chars() {
            res.push(
                self.encode(&char.to_string())
                    .ok_or(anyhow!("Character \"{char}\" not in mapping"))?,
            );
        }
        Ok(res)
    }

    /// Given a string, returns a new string with the characters not present
    /// in the CharacterMap removed.
    pub fn filter_string(&self, data: &String) -> String {
        let mut filt = String::with_capacity(data.len());
        for char in data.chars() {
            let key = char.to_string();

            if self.char_to_sym.contains_key(&key) {
                filt.push_str(&key);
            }
        }
        filt
    }

    /// Given a single symbol, returns the corresponding character if it exists
    /// in the mapping
    pub fn decode(&self, sym: u32) -> Option<String> {
        if sym < self.alphabet_size {
            Some(self.sym_to_char[sym as usize].clone())
        } else {
            None
        }
    }

    pub fn try_decode_all(&self, syms: Vec<u32>) -> Result<String> {
        let mut res = String::new();
        for sym in syms {
            res.push_str(&self.decode(sym).ok_or(anyhow!(
                "Symbol larger than alphabet size of {}",
                self.alphabet_size
            ))?);
        }
        Ok(res)
    }

    pub fn save_to_file(&self, path: String) -> Result<()> {
        let mut file = File::create(path)?;
        let mut bytes: Vec<u8> = Vec::new();
        bytes.put_u32_le(self.alphabet_size);

        // write the string lengths
        for s in self.sym_to_char.iter() {
            bytes.put_u32_le(s.len() as u32);
        }

        // now write the strings
        for s in self.sym_to_char.iter() {
            bytes.extend(s.as_bytes());
        }
        file.write_all(&mut bytes)?;

        Ok(())
    }

    pub fn from_file(path: String) -> Result<Self> {
        let mut file = File::open(path)?;
        let mut bytes: Vec<u8> = Vec::new();
        file.read_to_end(&mut bytes)?;
        let mut bytes: Bytes = bytes.into();

        let alphabet_size = bytes.get_u32_le();

        // get the number of bytes forming each character
        let mut str_lengths: Vec<usize> = Vec::with_capacity(alphabet_size as usize);
        for _ in 0..alphabet_size {
            str_lengths.push(bytes.get_u32_le() as usize);
        }

        // Loop through the strings in the character map and form the main
        // data structures
        let mut sym_to_char: Vec<String> = Vec::new();
        let mut char_to_sym: HashMap<String, u32> = HashMap::new();

        for i in 0..alphabet_size {
            let s = String::from_utf8(bytes.split_to(str_lengths[i as usize]).to_vec())?;
            char_to_sym.insert(s.clone(), i);
            sym_to_char.push(s);
        }

        Ok(Self {
            char_to_sym,
            sym_to_char,
            alphabet_size,
        })
    }
}

/// Stores a string-based individual sequence as a string and a CharacterMap.
/// For easy indexing, also stores the integer representation of the string
/// as a Vec<u32>.
#[derive(Clone)]
pub struct CharacterSequence {
    pub data: String,
    pub encoded: Vec<u32>,
    pub character_map: CharacterMap,
}

impl Sequence for CharacterSequence {
    fn alphabet_size(&self) -> u32 {
        self.character_map.alphabet_size
    }

    fn len(&self) -> u64 {
        self.encoded.len() as u64
    }

    fn try_get(&self, i: u64) -> Result<u32> {
        if i >= self.len() {
            bail!("invalid index {i} for sequence of length {}", self.len());
        }
        Ok(self.encoded[i as usize])
    }

    fn put_sym(&mut self, sym: u32) {
        self.data
            .push_str(&self.character_map.decode(sym).unwrap_or("".to_string()));
        self.encoded.push(sym);
    }
}

impl CharacterSequence {
    pub fn new(character_map: CharacterMap) -> Self {
        Self {
            data: String::new(),
            character_map,
            encoded: Vec::new(),
        }
    }

    pub fn from_data(data: String, character_map: CharacterMap) -> Result<Self> {
        for char in data.chars() {
            if character_map.encode(&char.to_string()).is_none() {
                bail!("Invalid symbol in input: {}", &char);
            }
        }
        Ok(Self {
            encoded: character_map.try_encode_all(&data)?,
            data,
            character_map,
        })
    }

    /// Builds a character map from the data string, and then encodes the
    /// string using the character map
    pub fn from_data_inferred_character_map(data: String) -> Self {
        let character_map = CharacterMap::from_data(&data);
        Self {
            encoded: character_map.try_encode_all(&data).unwrap(),
            character_map,
            data,
        }
    }

    /// Forms a CharacterSequence, using the provided CharacterMap to filter
    /// the provided data string.
    pub fn from_data_filtered(data: String, character_map: CharacterMap) -> Self {
        let data = character_map.filter_string(&data);
        Self {
            encoded: character_map.try_encode_all(&data).unwrap(),
            data,
            character_map,
        }
    }

    pub fn extend(&mut self, data: &String) -> Result<()> {
        let encoded = self.character_map.try_encode_all(data)?;
        self.data.push_str(&data);
        self.encoded.extend(encoded);

        Ok(())
    }
}

/// U8 sequence, for alphabet sizes between 3 and 256
#[derive(Clone)]
pub struct U8Sequence {
    pub data: Vec<u8>,
    alphabet_size: u32,
}

impl Sequence for U8Sequence {
    fn alphabet_size(&self) -> u32 {
        self.alphabet_size as u32
    }

    fn len(&self) -> u64 {
        self.data.len() as u64
    }

    fn try_get(&self, i: u64) -> Result<u32> {
        if i >= self.len() {
            bail!("invalid index {i} for sequence of length {}", self.len());
        }
        Ok(self.data[i as usize] as u32)
    }

    fn put_sym(&mut self, sym: u32) {
        self.data.push(sym as u8);
    }
}

impl U8Sequence {
    pub fn new(alphabet_size: u32) -> Self {
        Self {
            data: Vec::new(),
            alphabet_size,
        }
    }

    pub fn from_data(data: Vec<u8>, alphabet_size: u32) -> Result<Self> {
        if data.iter().any(|x| *x as u32 > alphabet_size) {
            bail!(
                "invalid symbol found for alphabet size of {}",
                alphabet_size
            );
        }
        Ok(Self {
            data,
            alphabet_size,
        })
    }

    pub fn from_data_inferred_alphabet_size(data: Vec<u8>) -> Result<Self> {
        if data.len() == 0 {
            bail!("cannot infer alphabet size from an empty sequence");
        }
        let alphabet_size = *(data.iter().max().unwrap()) as u32 + 1;
        Ok(Self {
            data,
            alphabet_size,
        })
    }

    pub fn extend(&mut self, data: &[u8]) -> Result<()> {
        if data.iter().any(|x| *x as u32 > self.alphabet_size) {
            bail!(
                "invalid symbol found for alphabet size of {}",
                self.alphabet_size
            );
        }
        self.data.extend(data);
        Ok(())
    }
}

/// U8 sequence, for alphabet sizes between 257 and 2^16
pub struct U16Sequence {
    pub data: Vec<u16>,
    alphabet_size: u32,
}

impl Sequence for U16Sequence {
    fn alphabet_size(&self) -> u32 {
        self.alphabet_size
    }

    fn len(&self) -> u64 {
        self.data.len() as u64
    }

    fn try_get(&self, i: u64) -> Result<u32> {
        if i >= self.len() {
            bail!("invalid index {i} for sequence of length {}", self.len());
        }
        Ok(self.data[i as usize] as u32)
    }

    fn put_sym(&mut self, sym: u32) {
        self.data.push(sym as u16);
    }
}

impl U16Sequence {
    pub fn new(alphabet_size: u32) -> Self {
        Self {
            data: Vec::new(),
            alphabet_size,
        }
    }

    pub fn from_data(data: Vec<u16>, alphabet_size: u32) -> Result<Self> {
        if data.iter().any(|x| *x as u32 > alphabet_size) {
            bail!(
                "invalid symbol found for alphabet size of {}",
                alphabet_size
            );
        }
        Ok(Self {
            data,
            alphabet_size,
        })
    }

    pub fn from_data_inferred_alphabet_size(data: Vec<u16>) -> Result<Self> {
        if data.len() == 0 {
            bail!("cannot infer alphabet size from an empty sequence");
        }
        let alphabet_size = *(data.iter().max().unwrap()) as u32 + 1;
        Ok(Self {
            data,
            alphabet_size,
        })
    }

    pub fn extend(&mut self, data: &[u16]) -> Result<()> {
        if data.iter().any(|x| *x as u32 > self.alphabet_size) {
            bail!(
                "invalid symbol found for alphabet size of {}",
                self.alphabet_size
            );
        }
        self.data.extend(data);
        Ok(())
    }
}

/// U8 sequence, for alphabet sizes between 2^16 + 1 and 2^32. Alphabet sizes
/// this large are not recommended, due to the convergence properties of LZ78.
#[derive(Clone)]
pub struct U32Sequence {
    pub data: Vec<u32>,
    alphabet_size: u32,
}

impl Sequence for U32Sequence {
    fn alphabet_size(&self) -> u32 {
        self.alphabet_size as u32
    }

    fn len(&self) -> u64 {
        self.data.len() as u64
    }

    fn try_get(&self, i: u64) -> Result<u32> {
        if i >= self.len() {
            bail!("invalid index {i} for sequence of length {}", self.len());
        }
        Ok(self.data[i as usize] as u32)
    }

    fn put_sym(&mut self, sym: u32) {
        self.data.push(sym as u32);
    }
}

impl U32Sequence {
    pub fn new(alphabet_size: u32) -> Self {
        Self {
            data: Vec::new(),
            alphabet_size,
        }
    }

    pub fn from_data(data: Vec<u32>, alphabet_size: u32) -> Result<Self> {
        if data.iter().any(|x| *x > alphabet_size) {
            bail!(
                "invalid symbol found for alphabet size of {}",
                alphabet_size
            );
        }
        Ok(Self {
            data,
            alphabet_size,
        })
    }

    pub fn from_data_inferred_alphabet_size(data: Vec<u32>) -> Result<Self> {
        if data.len() == 0 {
            bail!("cannot infer alphabet size from an empty sequence");
        }
        let alphabet_size = *(data.iter().max().unwrap()) + 1;
        Ok(Self {
            data,
            alphabet_size,
        })
    }

    pub fn extend(&mut self, data: &[u32]) -> Result<()> {
        if data.iter().any(|x| *x > self.alphabet_size) {
            bail!(
                "invalid symbol found for alphabet size of {}",
                self.alphabet_size
            );
        }
        self.data.extend(data);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use crate::sequence::CharacterMap;

    #[test]
    fn test_charmap() {
        let charmap = CharacterMap::from_data(&"abcdefghijklmnopqrstuvwxyz ".to_string());
        assert_eq!(charmap.alphabet_size, 27);
        assert_eq!(charmap.encode(&"j".to_string()), Some(9));
        assert_eq!(charmap.encode(&"z".to_string()), Some(25));
        assert!(charmap.encode(&"1".to_string()).is_none());

        assert_eq!(
            charmap.filter_string(&"hello world 123".to_string()),
            "hello world ".to_string()
        );

        assert_eq!(charmap.decode(1), Some("b".to_string()));
        assert_eq!(charmap.decode(26), Some(" ".to_string()));
        assert_eq!(charmap.decode(24), Some("y".to_string()));
        assert!(charmap.decode(27).is_none());
    }
}
