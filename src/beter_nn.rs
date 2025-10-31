use rand::Rng;
use serde_derive::{Deserialize, Serialize};

#[derive(Debug, Default, Clone, Serialize, Deserialize)]
pub struct BetterNN {
    pub weights: Vec<f32>,
    layers: Vec<u32>,

    buf1: Vec<f32>,
    buf2: Vec<f32>,
}


impl BetterNN {
    pub fn new<R: Rng>(rng: &mut R, layers: &[u32]) -> Self {
        let mut weights = vec![];

        for i in 0..layers.len()-1 {
            let curr_size = layers[i];
            let next_size = layers[i+1];

            for _ in 0..next_size {
                for _ in 0..curr_size {
                    weights.push(rng.random_range(-1.0..1.0));
                }
            }
        }

        Self {
            weights,
            layers: layers.to_vec(),
            buf1: vec![],
            buf2: vec![],
        }
    }

    pub fn forward<'a>(&self, ForwardBuffers { buf1, buf2 }: &'a mut ForwardBuffers, inputs: &[f32]) -> &'a [f32] {
        debug_assert_eq!(inputs.len(), self.layers[0] as usize);
        buf1.clear();
        buf2.clear();
        buf1.extend_from_slice(inputs);
        let inp = buf1;
        let out = buf2;
        let mut weight_offset = 0;

        for i in 0..self.layers.len() - 1 {
            let curr_size = self.layers[i] as usize;
            let next_size = self.layers[i + 1] as usize;

            out.clear();
            out.reserve(next_size);

            for neuron in 0..next_size {
                let start = weight_offset + neuron * curr_size;
                let end = start + curr_size;

                let sum: f32 = self.weights[start..end]
                    .iter()
                    .zip(inp.iter())
                    .map(|(w, x)| w * x)
                    .sum();

                out.push(fast_tanh(sum));
            }

            core::mem::swap(inp, out);
            weight_offset += curr_size * next_size;
        }

        inp.as_slice()
    }


    pub fn mutate<R: Rng>(&mut self, rng: &mut R, rate: f32) {
        for w in &mut self.weights {
            if rng.random_bool(0.3) {
                *w += rng.random_range(-1.0..1.0) * rate
            }
        }
    }
}

fn fast_tanh(x: f32) -> f32 { x / (1.0 + x.abs()) }


pub struct ForwardBuffers {
    buf1: Vec<f32>,
    buf2: Vec<f32>,
}


impl ForwardBuffers {
    pub fn new() -> Self {
        Self {
            buf1: vec![],
            buf2: vec![],
        }
    }
}
