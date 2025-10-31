use std::sync::Mutex;

use glam::{Vec2, Vec4};
use rand::{rngs::SmallRng, Rng, SeedableRng};
use serde_derive::{Deserialize, Serialize};
use sti::hash::fxhash::{fxhash32, fxhash64};
use wgpu::wgt::bytemuck_wrapper;

use crate::renderer::Renderer;


#[derive(Debug, Default, Clone, Serialize, Deserialize)]
pub struct NeuralNetwork {
    weights: Vec<Vec<Vec<f32>>>,
}



impl NeuralNetwork {
    pub fn new(layers: &[u32]) -> Self {
        let mut rng = rand::rng();

        let weights = (0..layers.len()-1)
            .map(|i| {
                let curr_size = layers[i];
                let next_size = layers[i+1];

                (0..next_size)
                    .map(|_| (0..curr_size).map(|_| rng.random_range(-1.0..1.0)).collect())
                    .collect()
            })
            .collect();

        NeuralNetwork { weights }
    }


    
    pub fn forward(&self, inputs: &[f32]) -> Vec<f32> {
        let mut inp = inputs.to_vec();
        for layer in &self.weights {
            inp = layer.iter()
                .map(|weights| {
                    let sum: f32 = weights.iter().zip(inputs).map(|(w, i)| w*i).sum();
                    sum.tanh()
                })
                .collect()
        }

        inp
    }


    pub fn mutate(&mut self, rate: f32) {
        let mut rng = rand::rng();
        for layer in &mut self.weights {
            layer.iter_mut()
                .for_each(|weights| {
                    weights.iter_mut()
                        .for_each(|w| if rng.random_bool(0.3) { *w += rng.random_range(-1.0..1.0) * rate });
                });
        }
    }
}


fn ray_detect_food(
    start: Vec2,
    direction: Vec2, // normalized
    max_distance: f32,
    foods: impl Iterator<Item=(Vec2, f32)>,
) -> f32 {
    for (food, signal) in foods {
        let to_food = food - start;
        let projection = to_food.dot(direction); // how far along the ray the food is

        if projection < 0.0 || projection > max_distance {
            continue; // behind the start or beyond max distance
        }

        // distance from food center to ray
        let closest_point = start + direction * projection;
        let distance_to_food = (food - closest_point).length();

        if distance_to_food < max_distance { // food radius
            return signal;
        }
    }

    0.0
}

