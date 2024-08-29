use std::collections::HashMap;

use anyhow::{bail, Result};
use rand::Rng;

use crate::{
    sequence::{Sequence, U32Sequence},
    tree::LZ78Tree,
    util::sample_from_pdf,
};

pub trait SourceNode {
    fn new_child(&self, rng: &mut impl Rng) -> Self;

    fn spa(&self, tree: &LZ78Tree, node_idx: u64) -> Vec<f64>;
}

pub struct SimplifiedBinarySourceNode {
    theta: f64,
    theta_pdf: Vec<f64>,
    theta_values: Vec<f64>,
}

impl SourceNode for SimplifiedBinarySourceNode {
    fn new_child(&self, rng: &mut impl Rng) -> Self {
        let sample = rng.gen_range(0.0..1.0);
        let new_theta = self.theta_values[sample_from_pdf(&self.theta_pdf, sample) as usize];
        Self {
            theta: new_theta,
            theta_pdf: self.theta_pdf.clone(),
            theta_values: self.theta_values.clone(),
        }
    }

    fn spa(&self, _tree: &LZ78Tree, _node_idx: u64) -> Vec<f64> {
        vec![1.0 - self.theta, self.theta]
    }
}

impl SimplifiedBinarySourceNode {
    pub fn new(theta_pdf: Vec<f64>, theta_values: Vec<f64>, rng: &mut impl Rng) -> Self {
        let sample = rng.gen_range(0.0..1.0);
        let new_theta = theta_values[sample_from_pdf(&theta_pdf, sample) as usize];
        Self {
            theta: new_theta,
            theta_pdf,
            theta_values,
        }
    }
}

pub struct DefaultLZ78SourceNode {}

impl SourceNode for DefaultLZ78SourceNode {
    fn new_child(&self, _rng: &mut impl Rng) -> Self {
        Self {}
    }

    fn spa(&self, tree: &LZ78Tree, node_idx: u64) -> Vec<f64> {
        tree.compute_spa(node_idx)
    }
}

pub struct LZ78Source<T: SourceNode> {
    tree: LZ78Tree,
    tree_node_to_source_node: HashMap<u64, T>,
    state: u64,
    alphabet_size: u32,
    log_loss: f64,
}

impl<T> LZ78Source<T>
where
    T: SourceNode,
{
    pub fn new(alphabet_size: u32, source_node: T) -> Self {
        let mut tree_node_to_source_node: HashMap<u64, T> = HashMap::new();
        tree_node_to_source_node.insert(LZ78Tree::ROOT_IDX, source_node);
        Self {
            tree: LZ78Tree::new(alphabet_size),
            tree_node_to_source_node,
            state: LZ78Tree::ROOT_IDX,
            alphabet_size,
            log_loss: 0.0,
        }
    }

    pub fn generate_symbols(&mut self, n: u64, rng: &mut impl Rng) -> Result<U32Sequence> {
        let mut syms = U32Sequence::new(self.alphabet_size);

        for i in 0..n {
            let node = &self.tree_node_to_source_node[&self.state];
            let spa = node.spa(&self.tree, self.state);
            if spa.len() as u32 != self.alphabet_size {
                bail!("alphabet size specified incompatible with SourceNode implementation");
            }
            let next_sym = sample_from_pdf(&spa, rng.gen_range(0.0..1.0)) as u32;
            syms.put_sym(next_sym);

            let traverse_result =
                self.tree
                    .traverse_to_leaf_from(self.state, &syms, i, i + 1, true, true)?;
            self.state = traverse_result.state_idx;
            self.log_loss += traverse_result.log_loss;

            if let Some(leaf) = traverse_result.added_leaf {
                self.tree_node_to_source_node.insert(
                    self.tree.get_node(self.state).branch_idxs[&leaf],
                    node.new_child(rng),
                );
                self.state = LZ78Tree::ROOT_IDX;
            }
        }

        Ok(syms)
    }
}

#[cfg(test)]
mod tests {

    use rand::thread_rng;

    use super::*;

    #[test]
    fn test_bernoulli_source() {
        let mut rng = thread_rng();
        let mut source = LZ78Source::new(
            2,
            SimplifiedBinarySourceNode::new(vec![0.5, 0.5], vec![0.0, 1.0], &mut rng),
        );

        let output = source
            .generate_symbols(100, &mut rng)
            .expect("generation failed");

        let mut i = 0;
        let mut phrase_num = 0;
        while i + 2 * phrase_num + 1 < output.len() {
            assert_eq!(
                output.data[i as usize..=(i + phrase_num) as usize],
                output.data[(i + phrase_num + 1) as usize..=(i + 2 * phrase_num + 1) as usize]
            );
            i += phrase_num + 1;
            phrase_num += 1;
        }
    }

    #[test]
    fn sanity_check_lz778_source() {
        let mut rng = thread_rng();
        let mut source = LZ78Source::new(4, DefaultLZ78SourceNode {});

        let output = source
            .generate_symbols(50, &mut rng)
            .expect("generation failed");

        println!("{:?}", output.data);
    }
}
