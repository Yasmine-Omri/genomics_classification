use std::{
    fs::File,
    io::{Read, Write},
    path::Path,
};

use anyhow::{anyhow, bail, Result};
use flate2::bufread::GzDecoder;
use json::parse;
use lz78::sequence::CharacterMap;
use parquet::{
    file::reader::{FileReader, SerializedFileReader},
    record::Field,
};
use png::Decoder;

pub enum DatasetPartition {
    Train,
    Test,
    Validation,
}

pub fn default_character_map() -> CharacterMap {
    CharacterMap::from_data(
        &"abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ1234567890\n\t .,\"'?:;-_"
            .to_string(),
    )
}

pub fn read_c4_realnewslike(c4_dir: &str, part: u64) -> Result<Vec<String>> {
    let path = format!("{c4_dir}/realnewslike/c4-train.{part:05}-of-00512.json.gz");
    let mut file = File::open(path)?;
    let mut buf: Vec<u8> = Vec::new();
    file.read_to_end(&mut buf)?;
    drop(file);

    let mut gz = GzDecoder::new(&buf[..]);
    let mut s = String::new();
    gz.read_to_string(&mut s)?;

    let mut result: Vec<String> = Vec::new();
    for line in s.splitn(s.len(), '\n') {
        let json = parse(line);
        if json.is_err() {
            break;
        }
        let json = json?;
        if json["text"].is_string() {
            result.push(
                json["text"]
                    .as_str()
                    .ok_or(anyhow!("parsing failure"))?
                    .to_owned(),
            );
        }
    }

    Ok(result)
}

pub fn read_wikitext(wikitext_dir: &str) -> Result<Vec<String>> {
    let mut result: Vec<String> = Vec::new();
    let paths = vec![
        format!("{wikitext_dir}/wikitext-103-v1/train-00000-of-00002.parquet"),
        format!("{wikitext_dir}/wikitext-103-v1/train-00001-of-00002.parquet"),
    ];

    for path in paths {
        let path = Path::new(&path);
        let file = File::open(&path)?;
        let reader = SerializedFileReader::new(file).unwrap();
        let row_group_reader = reader.get_row_group(0).unwrap();

        for row in row_group_reader.get_row_iter(None)? {
            let row = row?;
            let field = &row.into_columns()[0].1;
            let row_str = if let Field::Str(s) = field {
                Some(s)
            } else {
                None
            }
            .ok_or(anyhow!("error parsing Wikitext parquet file"))?;
            result.push(row_str.to_string());
        }
    }

    Ok(result)
}

pub fn read_fashion_mnist(dir: &str, partition: DatasetPartition) -> Result<Vec<Vec<u8>>> {
    let path = match partition {
        DatasetPartition::Train => format!("{dir}/fashion_mnist/train-00000-of-00001.parquet"),
        DatasetPartition::Test => format!("{dir}/fashion_mnist/test-00000-of-00001.parquet"),
        DatasetPartition::Validation => bail!("FashionMnist has no validation set"),
    };

    let mut result: Vec<Vec<u8>> = Vec::new();
    let path = Path::new(&path);
    let file = File::open(&path)?;
    let reader = SerializedFileReader::new(file).unwrap();
    let row_group_reader = reader.get_row_group(0).unwrap();

    for outer_row in row_group_reader.get_row_iter(None)? {
        let outer_row = outer_row?;
        let outer_field = &outer_row.into_columns()[0].1;
        let row = if let Field::Group(r) = &outer_field {
            Some(r)
        } else {
            None
        }
        .ok_or(anyhow!("error parsing parquet file"))?
        .clone();
        let field = row.into_columns()[0].clone().1;

        let png_bytes = if let Field::Bytes(b) = field {
            Some(b)
        } else {
            None
        }
        .ok_or(anyhow!("error parsing parquet file"))?
        .data()
        .to_vec();

        let tmp_path = format!("{dir}/tmp.png");
        let tmp_path = Path::new(&tmp_path);
        let mut tmp = File::create(tmp_path)?;
        tmp.write_all(&png_bytes)?;
        drop(tmp);

        let decoder = Decoder::new(File::open(tmp_path)?);
        let mut reader = decoder.read_info().unwrap();
        // Allocate the output buffer.
        let mut buf = vec![0; reader.output_buffer_size()];
        let info = reader.next_frame(&mut buf)?;
        let bytes = &buf[..info.buffer_size()];

        if bytes.len() != 28 * 28 {
            bail!(
                "error parsing parquet file. Expected array with length {}, got {}",
                28 * 28,
                bytes.len()
            );
        }

        result.push(bytes.to_vec());
    }

    Ok(result)
}

pub fn read_test_file(path: &str) -> Result<String> {
    let path = Path::new(&path);
    let mut file = File::open(&path)?;
    let mut buf = String::new();
    file.read_to_string(&mut buf)?;

    Ok(buf)
}
