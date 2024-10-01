use anyhow::Result;
use rkyv::{Archive, Deserialize, Serialize};
use std::collections::HashMap;

use crate::sequence::Sequence;

#[derive(Debug, Serialize, Deserialize, Archive)]
pub struct LZ78TreeNode {
    pub seen_count: u64,
    pub phrase_num: Option<u64>,
    pub branch_idxs: HashMap<u32, u64>,
}

#[derive(Debug, Serialize, Deserialize, Archive)]
pub struct LZ78Tree {
    nodes: Vec<LZ78TreeNode>,
    spa_gamma: f64,
    alphabet_size: u32,
}

pub struct LZ78TraversalResult {
    pub phrase_end_idx: u64,
    pub added_leaf: Option<u32>,
    pub state_idx: u64,
    pub reached_leaf: bool,
    pub log_loss: f64,
}

impl LZ78Tree {
    pub const ROOT_IDX: u64 = 0;

    pub fn new(alphabet_size: u32) -> Self {
        let root = LZ78TreeNode {
            seen_count: 1,
            phrase_num: None,
            branch_idxs: HashMap::new(),
        };

        Self {
            nodes: vec![root],
            spa_gamma: 0.5,
            alphabet_size,
        }
    }

    pub fn new_spa(alphabet_size: u32, spa_gamma: f64) -> Self {
        let root = LZ78TreeNode {
            seen_count: 1,
            phrase_num: None,
            branch_idxs: HashMap::new(),
        };

        Self {
            nodes: vec![root],
            spa_gamma,
            alphabet_size,
        }
    }

    pub fn compute_spa(&self, current_state: u64) -> Vec<f64> {
        let mut spa: Vec<f64> =
            vec![self.compute_spa_val(current_state, None)].repeat(self.alphabet_size as usize);

        for (sym, child_idx) in self.get_node(current_state).branch_idxs.iter() {
            spa[*sym as usize] = self.compute_spa_val(current_state, Some(*child_idx));
        }
        spa
    }

    pub fn compute_spa_val(&self, current_state: u64, next_state: Option<u64>) -> f64 {
        let next_state_count = if let Some(state) = next_state {
            self.get_node(state).seen_count
        } else {
            0
        } as f64;
        (next_state_count + self.spa_gamma)
            / (self.get_node(current_state).seen_count as f64 - 1.0
                + self.spa_gamma * self.alphabet_size as f64)
    }

    pub fn compute_instantaneous_log_loss(
        &self,
        current_state: u64,
        next_state: Option<u64>,
    ) -> f64 {
        let spa = self.compute_spa_val(current_state, next_state);
        (1.0 / spa).log2()
    }

    pub fn num_phrases(&self) -> u64 {
        self.nodes.len() as u64 - 1
    }

    pub fn phrase_num(&self, idx: u64) -> Option<u64> {
        self.get_node(idx).phrase_num
    }

    pub fn is_leaf(&self, idx: u64) -> bool {
        self.get_node(idx).seen_count == 1
    }

    pub fn get_node(&self, idx: u64) -> &LZ78TreeNode {
        &self.nodes[idx as usize]
    }

    fn get_node_mut(&mut self, idx: u64) -> &mut LZ78TreeNode {
        &mut self.nodes[idx as usize]
    }

    pub fn traverse_root_to_leaf<T>(
        &mut self,
        x: &T,
        start_idx: u64,
        end_idx: u64,
        grow: bool,
        update_counts: bool,
    ) -> Result<LZ78TraversalResult>
    where
        T: Sequence,
    {
        self.traverse_to_leaf_from(Self::ROOT_IDX, x, start_idx, end_idx, grow, update_counts)
    }

    pub fn traverse_to_leaf_from<T: ?Sized>(
        &mut self,
        node_idx: u64,
        x: &T,
        start_idx: u64,
        end_idx: u64,
        grow: bool,
        update_counts: bool,
    ) -> Result<LZ78TraversalResult>
    where
        T: Sequence,
    {
        let num_phrases = self.num_phrases();

        let mut state_idx = node_idx;

        let mut new_leaf: Option<u32> = None;
        let mut end_idx = end_idx;

        let mut log_loss: f64 = 0.0;

        for i in start_idx..end_idx {
            if update_counts {
                self.get_node_mut(state_idx).seen_count += 1;
            }
            let val = x.get(i)?;
            let prev_state_idx = state_idx;

            if self.get_node(state_idx).branch_idxs.contains_key(&val) {
                state_idx = self.get_node(state_idx).branch_idxs[&val];

                log_loss += self.compute_instantaneous_log_loss(prev_state_idx, Some(state_idx));
            } else {
                new_leaf = Some(val);
                end_idx = i;
                log_loss += self.compute_instantaneous_log_loss(prev_state_idx, None);
                break;
            }
        }

        let reached_leaf = new_leaf.is_some();
        let added_leaf = if grow { new_leaf } else { None };
        if added_leaf.is_some() {
            self.get_node_mut(state_idx)
                .branch_idxs
                .insert(new_leaf.unwrap(), num_phrases + 1);

            self.nodes.push(LZ78TreeNode {
                seen_count: 1,
                phrase_num: Some(num_phrases),
                branch_idxs: HashMap::new(),
            });
        }

        Ok(LZ78TraversalResult {
            phrase_end_idx: end_idx,
            state_idx,
            added_leaf,
            log_loss,
            reached_leaf,
        })
    }
}
