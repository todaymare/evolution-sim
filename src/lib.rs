#![feature(portable_simd)]

pub mod renderer;
pub mod shader;
pub mod buffer;
pub mod uniform;
pub mod neural_network;
pub mod new_sim;
pub mod beter_nn;

use std::{collections::{HashSet, VecDeque}, f32::consts::PI, sync::Mutex};

use glam::{IVec2, Mat2, Mat4, Quat, UVec2, Vec2, Vec3, Vec3Swizzles, Vec4};
use libnoise::{Generator, Source};
use rand::{random_range, Rng};
use rapier2d::prelude::ColliderHandle;
use rayon::iter::{ParallelBridge, ParallelIterator};
use serde_derive::{Deserialize, Serialize};
use sti::{define_key, vec::KVec};

use crate::{neural_network::NeuralNetwork, renderer::{ParticleInstance, Renderer}};

define_key!(pub EntityId(u32));


#[derive(Serialize, Deserialize)]
pub struct Simulation {
    entities: Vec<Entity>,

    food: Vec<Vec2>,
    world: Vec<bool>,
    empty_cells: Vec<u32>,
    world_size: UVec2,
    pub current_tick: u32,

    current_save: u32,
    save_tick: u32,
}


#[derive(Debug, Clone, Serialize, Deserialize)]
struct Entity {
    position: Vec2,
    velocity: Vec2,
    angular_velocity: f32,
    rot: f32,
    speed: f32,
    health: f32,
    can_heal: u32,
    fullness: f32,

    stats: EntityStats,
    radius_squared: f32,
    cells: Vec<Cell>,
    active_cells: Vec<Cell>,
    brains: NeuralNetwork,
    generation: u32,
    born: u32,


    memory: [f32; 8],
}


#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
struct Cell {
    offset: IVec2,
    mat: Mat4,
    kind: CellKind,
}


#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
enum CellKind {
    BasicCell,
    SpeedCell,
    FatCell,
    Spike,
    HealthyCell,
}



#[derive(Debug, Clone, Serialize, Deserialize)]
struct EntityStats {
    acceleration: f32,
    weight: f32,
    max_fullness: f32,
    max_hp: f32,
    basal_metabolic_rate: f32,
    max_speed: f32,
}


impl Simulation {
    pub fn new() -> Self {
        let world_size = UVec2::new(250, 250);
        let mut world = vec![false; (world_size.x * world_size.y) as usize];
        let dist = world_size.length_squared();
        let noise = Source::improved_perlin(31204);
        for y in 0..world_size.y {
            for x in 0..world_size.x {
                let pos = y * world_size.x + x;
                let sample = noise.sample([x as f64 * 0.01, y as f64 * 0.01 ]);
                world[pos as usize] = sample > 0.6;
            }
        }


        let mut this = Self {
            entities: Vec::new(),
            food: vec![],
            current_tick: 0,
            empty_cells: world.iter().enumerate().filter(|x| !x.1).map(|x| x.0 as u32).collect(),
            world,
            world_size,
            current_save: 0,
            save_tick: 0,
        };


        for _ in 0..100 {
            this.entities.push(Entity {
                position: rand::rng().random::<Vec2>() * world_size.as_vec2(),
                angular_velocity: 0.0,
                born: 0,
                can_heal: 0,
                velocity: Vec2::ZERO,
                radius_squared: 0.0,
                rot: 0.0,
                speed: 0.0,
                health: 25.0,
                fullness: 10.0,
                cells: vec![
                    Cell { offset: IVec2::ZERO, kind: CellKind::BasicCell, mat: Mat4::IDENTITY },
                    Cell { offset: IVec2::new(1, 0), kind: CellKind::BasicCell, mat: Mat4::IDENTITY },
                    Cell { offset: IVec2::new(-1, 0), kind: CellKind::BasicCell, mat: Mat4::IDENTITY },
                    Cell { offset: IVec2::new(0, 1), kind: CellKind::BasicCell, mat: Mat4::IDENTITY },
                    Cell { offset: IVec2::new(0, -1), kind: CellKind::BasicCell, mat: Mat4::IDENTITY },
                ],
                active_cells: vec![],

                brains: NeuralNetwork::new(&[18+8, 64, 64, 64, 2+8]),
                memory: [0.0; 8],

                stats: EntityStats::new(),

                generation: 0,


            });

            for i in 0..10 {
                this.entities.last_mut().unwrap().mutate_ex(true);
            }

            this.entities.last_mut().unwrap().spawn();
        }


        for _ in 0..5000 {
            this.spawn_food();
        }

        this
    }



    pub fn tick(&mut self) {
        self.current_tick += 1;



        // update

        #[derive(Clone, Copy)]
        struct SendPtr<T>(*mut T);
        unsafe impl<T> Send for SendPtr<T> {}
        unsafe impl<T> Sync for SendPtr<T> {}

        let ptr = self.entities.as_mut_ptr();
        let ptr = SendPtr(ptr);
        let len = self.entities.len();

        (0..len)
        .par_bridge()
        .for_each(|offset| unsafe {
            let ptr = ptr.clone().0;
            let entity = ptr.add(offset);
            let dir = Vec2::from_angle((*entity).rot);
            let radius = (*entity).radius_squared.sqrt();

            let eye_offsets = [
                0.0f32,
                5.0, 10.0, 15.0, 20.0,
                -5.0, -10.0, -15.0, -20.0,
            ];

            let rays = core::array::from_fn::<f32, 9, _>(|i| {
                let rads = eye_offsets[i].to_radians();
                let dir = dir.rotate(Vec2::new(rads.cos(), rads.sin()));
                let iter = self.food.iter()
                    .map(|x| (*x, 1.0))
                    .chain((0..len)
                        .map(|i| {
                            let entity = &*ptr.add(i);
                            ((*entity).position, (*entity).health/(*entity).stats.max_hp)
                        }));

                raycast((*entity).position, dir, 50.0, iter, self.world_size, &self.world) as u8 as f32
            });

            // run the brains
            //let dir_to_food = (entity.position - self.food).normalize();
            let memory = (*entity).memory;
            let inputs = [
                //dir.x, dir.y,
                (*entity).speed / (*entity).stats.max_speed,
                (*entity).rot,
                (*entity).angular_velocity,
                (*entity).fullness / (*entity).stats.max_fullness,
                (*entity).health / (*entity).stats.max_hp,
                (*entity).active_cells.len() as f32 / (*entity).cells.len() as f32,
                (self.current_tick - (*entity).born) as f32,
                ((*entity).position.x.abs() / self.world_size.x as f32),
                ((*entity).position.y.abs() / self.world_size.y as f32),
                //(*entity).stats.acceleration,


                rays[0], rays[1], rays[2], rays[3],
                rays[4], rays[5], rays[6], rays[7],
                rays[8],


                memory[0], memory[1],
                memory[2], memory[3],
                memory[4], memory[5],
                memory[6], memory[7],

                //dir_to_food.x, dir_to_food.y,
            ];

            let outputs = (*entity).brains.forward(&inputs);

            let dir = outputs[0];
            (*entity).angular_velocity += dir.clamp(-0.1, 0.1);
            (*entity).angular_velocity *= 0.6;
            (*entity).speed += outputs[1] * (*entity).stats.acceleration * 0.1 / ((*entity).stats.weight * (radius * 0.8));
            (*entity).speed = (*entity).speed.clamp(0.0, (*entity).stats.max_speed);

            (*entity).velocity = Vec2::from_angle((*entity).rot) * (*entity).speed / (*entity).angular_velocity.abs().max(1.0);
            (*entity).rot += (*entity).angular_velocity * 0.05;
            (*entity).memory.copy_from_slice(&outputs[outputs.len()-8..]);

            let speed_exhaust = (*entity).speed / (*entity).stats.max_speed;
            (*entity).fullness -= ((speed_exhaust * 0.016
                                    + (*entity).stats.basal_metabolic_rate
                                    + ((*entity).angular_velocity.abs() + 1.0).powi(2) * 50.0
                                )) * (*entity).stats.weight.sqrt() * 0.00015 * 0.25;
        });


        let mut bye_food = Mutex::new(vec![]);
        let mut new_entities = Mutex::new(vec![]);
        let mut bye_goobers = Mutex::new(vec![]);

        // eat and reproduce
        self.entities
        .iter_mut()
        .enumerate()
        .par_bridge()
        .for_each(|(i, entity)| {
            entity.position += entity.velocity;
            if entity.can_heal == 0 {
                entity.health += 0.1;
            } else {
                entity.can_heal -= 1;
                return;
            }
            entity.health = entity.health.min(entity.stats.max_hp);

            
            // handle movement
            let mut total_push = Vec2::ZERO;
            let mut total_torque = 0.0;
            let rot = Mat2::from_angle(entity.rot);
            let grid_cell_size = 1.0;

            for cell in &entity.cells {
                let cell_pos = entity.position + rot * cell.offset.as_vec2();

                let g = (cell_pos / grid_cell_size).floor().as_ivec2();

                for dy in -1..=1 {
                    for dx in -1..=1 {
                        let nx = g.x + dx;
                        let ny = g.y + dy;

                        if nx < 0 || ny < 0 || nx >= self.world_size.x as i32 || ny >= self.world_size.y as i32 {
                            entity.fullness -= 0.01;
                            continue;
                        }

                        let idx = ny as usize * self.world_size.x as usize + nx as usize;
                        if !self.world[idx] { continue; }

                        let cell_center = Vec2::new(
                            (nx as f32 + 0.5) * grid_cell_size,
                            (ny as f32 + 0.5) * grid_cell_size,
                        );

                        let delta = cell_pos - cell_center;
                        let dist = delta.length();
                        let overlap = 1.0 + grid_cell_size * 0.5 - dist;

                        if overlap > 0.0 {
                            let push = delta.normalize() * overlap;
                            total_push += push;

                            // torque = r x F in 2D
                            let r = cell_pos - entity.position;
                            total_torque += r.perp_dot(push);
                        }
                    }
                }
            }

            entity.position += total_push;
            entity.fullness -= total_push.length();
            //entity.rot += total_torque * 0.01;


            let model = Mat4::from_rotation_translation(
                Quat::from_rotation_z(entity.rot),
                Vec3::new(entity.position.x, entity.position.y, 0.0)
            );


            'l: for cell in &entity.active_cells {
                let mat = cell.mat;

                let mut pos = None;

                for (i, &food) in self.food.iter().enumerate() {
                    if food.distance_squared(entity.position) > entity.radius_squared { continue }
                    let pos = match pos.as_ref() {
                        Some(v) => &v,
                        None => {
                            pos = Some((model * mat).to_scale_rotation_translation().2.xy());
                            pos.as_ref().unwrap()
                        },
                    };

                    if pos.distance_squared(food) < 2.0 {
                        let mut bye_food = bye_food.lock().unwrap();
                        if let Err(val) = bye_food.binary_search(&i) {
                            entity.fullness += 2.0 * (cell.offset.length_squared() as f32 / entity.radius_squared);
                            entity.fullness = entity.fullness.min(entity.stats.max_fullness);

                            bye_food.insert(val, i);

                            if entity.active_cells.len() != entity.cells.len() {
                                entity.grow();
                                entity.grow();
                                entity.grow();
                                entity.grow();
                                entity.grow();
                                entity.grow();
                                entity.grow();
                                entity.grow();
                                entity.grow();
                                break 'l;
                            }

                            let mut new_entities = new_entities.lock().unwrap();
                            let mut goober = entity.clone();
                            goober.velocity = Vec2::ZERO;
                            let size = entity.radius_squared * 5.0;
                            goober.position += rand::rng().random::<Vec2>() * size - size*0.5;
                            goober.rot = entity.rot + rand::rng().random_range(0f32..PI*2.0);
                            goober.generation += 1;
                            goober.born = self.current_tick;
                            goober.active_cells.clear();


                            let base_rate = 1.0;
                            let decay_factor = 0.01;   // how fast it decays per generation


                            let mut rate = (base_rate * (-decay_factor * entity.generation as f32).exp()).max(0.05);

                            if rand::rng().random_bool(0.5) {
                                rate *= 4.0;
                            }


                            if rand::rng().random_bool(0.5) {
                                rate *= 4.0;
                            }


                            if rand::rng().random_bool(0.3) {
                                rate *= 4.0;
                            }


                            if rand::rng().random_bool(0.1) {
                                rate *= 4.0;
                            }


                            if rand::rng().random_bool(0.02) {
                                rate *= 4.0;
                            }

                            goober.brains.mutate(rate);
                            goober.mutate();
                            goober.spawn();

                            goober.fullness = goober.stats.max_fullness;
                            goober.health = goober.stats.max_hp;

                            new_entities.push(goober);

                            break 'l;
                        }
                    }

                }

            }


            if entity.fullness <= 0.0 {
                let mut bye_goobers = bye_goobers.lock().unwrap();
                let index = bye_goobers.binary_search(&i).unwrap_err();
                bye_goobers.insert(index, i);
            }


        });


        for food in bye_goobers.get_mut().unwrap().iter().rev() {
            self.entities.remove(*food);
        }



        let mut bye_goobers : Mutex<Vec<(usize, usize)>> = Mutex::new(vec![]);
        let len = self.entities.len();
        (0..len)
        .par_bridge()
        .for_each(|offset| unsafe {
            let ptr = ptr.clone().0;
            let entity = ptr.add(offset);

            let model = Mat4::from_rotation_translation(
                Quat::from_rotation_z((*entity).rot),
                Vec3::new((*entity).position.x, (*entity).position.y, 0.0)
            );


            'l: for oth_offset in 0..len {
                if offset == oth_offset {
                    continue;
                }

                let oth_entity = ptr.add(oth_offset);
                if (*oth_entity).position.distance_squared((*entity).position) > (*entity).radius_squared {
                    continue;
                }

                let oth_model = Mat4::from_rotation_translation(
                    Quat::from_rotation_z((*oth_entity).rot),
                    Vec3::new((*oth_entity).position.x, (*oth_entity).position.y, 0.0)
                );



                for my_cell in &(*entity).active_cells {
                    let pos = (model * my_cell.mat).to_scale_rotation_translation().2.xy();
                    for oth_cell in &(*oth_entity).active_cells {
                        let other_pos = (oth_model * oth_cell.mat).to_scale_rotation_translation().2.xy();

                        if pos.distance_squared(other_pos) < 1.0 {
                            // Angle difference between entity facing and attack direction
                            let delta = ((*oth_entity).rot - (*entity).rot ).rem_euclid(2.0 * PI);
                            
                            // Alternatively, simpler: 1 when exactly perpendicular, 0 when front/back
                            let mut damage = (delta).sin().abs();
                            if matches!(oth_cell.kind, CellKind::Spike) {
                                damage *= 2.5;
                            }
                            if matches!(my_cell.kind, CellKind::HealthyCell) {
                                damage *= 2.5;
                            }


                            damage *= (*oth_entity).stats.weight * 0.5;

                            (*entity).health -= damage;
                            (*entity).speed *= 0.5;
                            (*entity).angular_velocity *= 0.5;
                            (*entity).can_heal = 25;
                            if (*entity).health < 0.0 {
                                let mut bye_goobers = bye_goobers.lock().unwrap();
                                if let Err(index) = bye_goobers.binary_search_by_key(&offset, |x| x.0) {
                                    bye_goobers.insert(index, (offset, oth_offset));
                                }

                            }
                            break 'l; // we died.

                        }


                    }
                }


            }
        });



        for goober in new_entities.into_inner().unwrap().into_iter() {
            self.entities.push(goober);
        }

        for food in bye_food.get_mut().unwrap().iter().rev() {
            self.food.remove(*food);
        }


        for (curr_goober, oth_goober) in bye_goobers.get_mut().unwrap().iter().rev() {
            let oth = &mut self.entities[*oth_goober];
            oth.health = oth.stats.max_hp;
            oth.fullness = oth.stats.max_fullness;
        }

        for (removed, _) in bye_goobers.get_mut().unwrap().iter().rev() {
            self.entities.remove(*removed);
        }


        if (self.current_tick % 1) == 0 {
            self.spawn_food();
            self.spawn_food();
        }



        if self.entities.is_empty() {
            println!("went extinct");
            self.save_tick = self.current_tick;
            if self.current_save <= 2 {
                *self = Self::new();
                return;
            }
            self.current_save -= 2;
            let save = std::fs::read(&format!("save_{}.json", self.current_save)).unwrap();
            let save : Self = serde_json::from_slice(&save).unwrap();
            self.entities = save.entities;
            self.world_size = save.world_size;
            self.world = save.world;
            self.food = save.food;
            self.current_tick = save.current_tick;
        }

        if (self.current_tick % 25_000) == 0 {
            let save = format!("save_{}.json", self.current_save);
            std::fs::write(&save, &*serde_json::to_vec(&*self).unwrap()).unwrap();
            self.current_save += 1;
            println!("generation: {:?} current tick {}", self.entities.iter().map(|x| x.generation).max(), self.current_tick);
        }
    }



    pub fn spawn_food(&mut self) {
        let index = self.empty_cells[rand::rng().random_range(0..self.empty_cells.len())];
        self.food.push(UVec2::new(index % self.world_size.x, index / self.world_size.x).as_vec2());
    }



    pub fn render(&self, renderer: &mut Renderer) {
        for entity in &self.entities {
            let model = Mat4::from_rotation_translation(
                Quat::from_rotation_z(entity.rot),
                Vec3::new(entity.position.x, entity.position.y, 0.0)
            );

            for cell in &entity.cells {
                let offset = cell.offset.as_vec2();

                let mat = Mat4::from_translation(
                    Vec3::new(offset.x, offset.y, 0.0),
                );

                let mat = model * mat;

                renderer.particle_at(ParticleInstance::new(mat, (cell.kind.colour()).with_z(1.0).with_w(0.1)));
            }

            for cell in &entity.active_cells {
                let offset = cell.offset.as_vec2();

                let mat = Mat4::from_scale_rotation_translation(
                    Vec3::new(0.8, 0.8, 1.0),
                    Quat::IDENTITY,
                    Vec3::new(offset.x, offset.y, 0.0),

                );

                let mat = model * mat;

                renderer.particle_at(ParticleInstance::new(mat, (cell.kind.colour() * (entity.fullness / entity.stats.max_fullness))));
            }
        }

        for y in 0..self.world_size.y {
            for x in 0..self.world_size.x {
                let model = Mat4::from_translation(Vec3::new(x as f32, y as f32, 0.0));
                let colour = self.world[(y * self.world_size.x + x) as usize];
                if !colour { continue }
                renderer.particle_at(ParticleInstance::new(model, Vec4::ONE));
            }
        }



        for food in &self.food {
            renderer.particle_at(ParticleInstance::new(Mat4::from_translation(Vec3::new(food.x, food.y, 0.0)), Vec4::new(1.0, 0.0, 0.0, 1.0)));
        }

    }
}



impl Entity {
    pub fn mutate(&mut self) {
        self.mutate_ex(false);
    }


    pub fn spawn(&mut self) {
        for i in 0..(6usize.min(self.cells.len())) {
            self.grow();
        }

    }


    pub fn mutate_ex(&mut self, change_top: bool) {
        let mut changed_topology = change_top;
        let mut rng = rand::rng();
        for i in 0..self.cells.len() {
            let cell = &mut self.cells[i];

            if rng.random_bool(0.8) {
                continue;
            }

            let change = rng.random_range(0..4);
            let new_cell = match change {
                0 => CellKind::BasicCell,
                1 => CellKind::SpeedCell,
                2 => CellKind::FatCell,
                3 => CellKind::Spike,
                _ => unreachable!(),
            };


            cell.kind = new_cell;
            changed_topology = true;
        }


        if rng.random_bool(0.3) && self.cells.len() > 6 {
            let count = if rng.random_bool(0.1) {
                self.cells.len() / 2
            } else {
                1
            };

            for i in 0..count {
                let mut new_cells = self.cells.clone();
                new_cells.remove(rng.random_range(0..self.cells.len()));

                if is_connected(&new_cells) {
                    changed_topology = true;
                }
            }

        }


        if rng.random_bool(0.5) {
            let count = if rng.random_bool(0.1) {
                self.cells.len() * 2
            } else {
                rng.random_range(0..3)
            };

            for i in 0..count {
                let cell = &self.cells[rng.random_range(0..self.cells.len())];
                let mut offset = rng.random::<IVec2>().signum();
                offset[rng.random_range(0..2)] = 0;
                let offset = cell.offset + offset;


                let mut can_place = true;
                for o in &self.cells {
                    if o.offset == offset { can_place = false; break }
                }


                if can_place {
                    let change = rng.random_range(0..4);
                    let kind = match change {
                        0 => CellKind::BasicCell,
                        1 => CellKind::SpeedCell,
                        2 => CellKind::FatCell,
                        3 => CellKind::Spike,
                        _ => unreachable!(),
                    };


                    let mat = Mat4::from_translation(
                        Vec3::new(offset.x as f32, offset.y as f32, 0.0),
                    );

                    let cell = Cell {
                        offset,
                        mat,
                        kind,
                    };


                    self.cells.push(cell);
                    changed_topology = true;
                }

            }
        }

        if changed_topology {
            order_cells_growth(&mut self.cells);
        }

    }


    fn grow(&mut self) {
        if self.active_cells.len() == self.cells.len() { return }

        self.active_cells.push(self.cells[self.active_cells.len()]);
        let mut stats = EntityStats::new();
        let mut max = f32::MIN;
        for cell in &self.active_cells {
            cell.kind.modify_stats(&mut stats);
            max = max.max(cell.offset.distance_squared(IVec2::ZERO) as f32);
        }

        self.stats = stats;
        self.radius_squared = max;
    }
}



fn raycast(
    origin: Vec2,
    dir: Vec2,
    max_distance: f32,
    items: impl Iterator<Item = (Vec2, f32)>,

    grid: UVec2,
    grid_data: &[bool],
) -> f32 {
    let mut min_dist = max_distance;
    let mut min_sig = 0.0;


    // check items
    for (food, signal) in items {
        let to_food = food - origin;
        let projection = to_food.dot(dir); // how far along the ray the food is

        if projection < 0.0 || projection > max_distance {
            continue; // behind the start or beyond max distance
        }

        // distance from food center to ray
        let closest_point = origin + dir * projection;
        let distance_to_food = (food - closest_point).length();

        if distance_to_food < min_dist {
            min_dist = distance_to_food;
            min_sig = signal;
        }
    }


    // check the grid

    // starting cell
    let grid_cell_size = 1.0;
    let mut x = (origin.x / grid_cell_size).floor() as isize;
    let mut y = (origin.y / grid_cell_size).floor() as isize;

    // step direction
    let step_x = if dir.x > 0.0 { 1 } else { -1 };
    let step_y = if dir.y > 0.0 { 1 } else { -1 };

    // tMax: distance along ray to next grid line
    let t_max_x = {
        let gx = x as f32 * grid_cell_size;
        let next_x = if step_x > 0 { gx + grid_cell_size } else { gx };
        (next_x - origin.x) / dir.x
    };
    let t_max_y = {
        let gy = y as f32 * grid_cell_size;
        let next_y = if step_y > 0 { gy + grid_cell_size } else { gy };
        (next_y - origin.y) / dir.y
    };

    // tDelta: distance along ray to cross a grid cell
    let t_delta_x = grid_cell_size / dir.x.abs();
    let t_delta_y = grid_cell_size / dir.y.abs();

    let mut t_max_x = t_max_x;
    let mut t_max_y = t_max_y;
    let mut distance = 0.0;

    while distance < min_dist {
        // check current cell
        if x >= 0 && y >= 0 && x < grid.x as isize && y < grid.y as isize {
            let idx = y as usize * grid.x as usize + x as usize;
            if grid_data[idx] {
                return 0.5; // obstacle signal
            }
        } else {
            break; // outside grid
        }

        // step to next cell
        if t_max_x < t_max_y {
            distance = t_max_x;
            t_max_x += t_delta_x;
            x += step_x;
        } else {
            distance = t_max_y;
            t_max_y += t_delta_y;
            y += step_y;
        }
    }



    min_sig
}



impl EntityStats {
    pub fn new() -> Self {
        Self {
            acceleration: 1.0,
            weight: 0.0,
            max_fullness: 10.0,
            basal_metabolic_rate: 0.001,
            max_hp: 25.0,
            max_speed: 1.0
        }
    }

}



impl CellKind {
    pub fn modify_stats(&self, stats: &mut EntityStats) {
        match self {
            CellKind::BasicCell => {
                stats.weight += 0.1;
                stats.max_hp += 0.5;
            },
            CellKind::SpeedCell => {
                stats.weight += 1.0;
                stats.acceleration += 0.07;
                stats.max_hp += 0.25;
                stats.max_speed += 0.125;
            },
            CellKind::FatCell => {
                stats.weight += 0.5;
                stats.basal_metabolic_rate *= 0.90;
                stats.max_fullness += 5.0;
                stats.max_hp += 1.0;
            },

            CellKind::Spike => {
                stats.weight += 0.25;
                stats.max_hp -= 2.0;
            }

            CellKind::HealthyCell => {
                stats.basal_metabolic_rate *= 1.1;
                stats.max_hp += 20.0;
                stats.weight += 0.5;
            }
        }
    }


    pub fn colour(&self) -> Vec4 {
        match self {
            CellKind::BasicCell => Vec4::new(0.4, 0.4, 0.4, 1.0),
            CellKind::SpeedCell => Vec4::new(0.0, 0.5, 1.0, 1.0),
            CellKind::FatCell => {
                Vec4::new(0.6, 0.6, 0.4, 1.0)
            },
            CellKind::Spike => {
                Vec4::new(1.3, 1.0, 1.3, 1.0)
            }
            CellKind::HealthyCell => {
                Vec4::new(1.0, 0.8, 0.8, 1.0)
            }
        }
    }
}


fn is_connected(cells: &[Cell]) -> bool {
    if cells.is_empty() { return true; }

    let mut visited = HashSet::new();
    let mut stack = vec![cells[0].offset];

    while let Some(pos) = stack.pop() {
        if !visited.insert(pos) { continue; }

        // push neighbors
        for dx in -1i32..=1 {
            for dy in -1i32..=1 {
                if dx == 0 && dy == 0 { continue; }
                if dx.abs() == 1 && dy.abs() == 1 { continue; }
                let neighbor = IVec2::new(pos.x + dx, pos.y + dy);
                if cells.iter().any(|c| c.offset == neighbor) {
                    stack.push(neighbor);
                }
            }
        }
    }

    visited.len() == cells.len()
}





/// Reorders `cells` in-place so that each cell is after at least one connected cell.
/// BFS starts from origin (0,0). Cells not connected to origin are moved to the end.
pub fn order_cells_growth(cells: &mut [Cell]) {
    let mut visited = HashSet::new();
    let mut queue = VecDeque::new();
    let mut ordered = Vec::with_capacity(cells.len());

    // Map offsets to their indices for quick lookup
    let mut offset_to_index = std::collections::HashMap::new();
    for (i, cell) in cells.iter().enumerate() {
        offset_to_index.insert(cell.offset, i);
    }

    let origin = IVec2 { x: 0, y: 0 };
    if offset_to_index.contains_key(&origin) {
        queue.push_back(origin);
        visited.insert(origin);
    }

    while let Some(pos) = queue.pop_front() {
        if let Some(&idx) = offset_to_index.get(&pos) {
            ordered.push(cells[idx].clone());
        }

        // Check 8 neighbors
        for dx in -1i32..=1 {
            for dy in -1i32..=1 {
                if dx == 0 && dy == 0 { continue; }
                if dx.abs() == 1 && dy.abs() == 1 { continue; }
                let neighbor = IVec2 { x: pos.x + dx, y: pos.y + dy };
                if visited.contains(&neighbor) { continue; }
                if offset_to_index.contains_key(&neighbor) {
                    queue.push_back(neighbor);
                    visited.insert(neighbor);
                }
            }
        }
    }

    // Copy back into original slice
    cells.copy_from_slice(&ordered);
}
