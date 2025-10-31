use std::f32::consts::TAU;

use glam::{IVec2, Mat4, Quat, Vec2, Vec3, Vec4};
use nalgebra::{UnitComplex, Vector2};
use rand::{random_range, rngs::SmallRng, Rng, SeedableRng};
use rapier2d::{parry::{query::DefaultQueryDispatcher, utils::{hashmap::HashMap, hashset::HashSet}}, prelude::*};
use rayon::iter::{ParallelBridge, ParallelIterator};
use serde_derive::{Deserialize, Serialize};

use crate::{beter_nn::{BetterNN, ForwardBuffers}, renderer::{ParticleInstance, Renderer}};


pub struct Sim {
    tick: u32,

    world: IVec2,

    entities: HashMap<u32, Entity>,
    foods: HashSet<ColliderHandle>,
    rng: SmallRng,
    entity_counter: u32,


    pub gravity: Vector2<f32>,
    integration_params: IntegrationParameters,
    physics_pipeline: PhysicsPipeline,
    island_manager: IslandManager,
    broad_phase: DefaultBroadPhase,
    narrow_phase: NarrowPhase,
    impulse_joint_set: ImpulseJointSet,
    multibody_joint_set: MultibodyJointSet,
    ccd_solver: CCDSolver,

    pub rigid_body_set: RigidBodySet,
    pub collider_set: ColliderSet,
}


#[derive(Debug, Clone)]
struct Entity {
    velocity: Vector2<f32>,
    angular_velocity: f32,

    health: f32,

    rb: RigidBodyHandle,
    col: ColliderHandle,

    brain: BetterNN,
    archetype: EntityArchetype,
}


#[derive(Debug, Clone)]
struct EntityArchetype {
    cells: Vec<Cell>,
}


#[derive(Debug, Clone)]
struct Cell {
    offset: IVec2,
    kind  : CellKind,
}


#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
enum CellKind {
    BasicCell,
    SpeedCell,
    FatCell,
    Spike,
    HealthyCell,
}


impl Sim {
    pub fn new() -> Self {
        let world = IVec2::new(100, 100);

        let bodies = RigidBodySet::new();
        let colliders = ColliderSet::new();
        let physics_pipeline = PhysicsPipeline::new();
        let integration_parameters = IntegrationParameters::default();

        let islands = IslandManager::new();
        let narrow_phase = NarrowPhase::new();
        let broad_phase = BroadPhaseBvh::new();
        let ccd_solver = CCDSolver::new();
        let impulse_joints = ImpulseJointSet::new();
        let multibody_joints = MultibodyJointSet::new();

        let mut this = Self {
            tick: 0,
            world,
            rng: SmallRng::seed_from_u64(6969696969),
            entities: HashMap::default(),
            entity_counter: 0,
            foods: HashSet::default(),
            gravity: Vector2::zeros(),
            integration_params: integration_parameters,
            physics_pipeline,
            island_manager: islands,
            broad_phase,
            narrow_phase,
            impulse_joint_set: impulse_joints,
            multibody_joint_set: multibody_joints,
            ccd_solver,
            rigid_body_set: bodies,
            collider_set: colliders,
        };


        for _ in 0..50 {
            this.new_entity();
        }

        for _ in 0..25 {
            this.new_food();
        }


        this
    }


    pub fn step(&mut self) {
        self.tick += 1;

        self.physics_pipeline.step(
            &mut self.gravity,
            &mut self.integration_params,
            &mut self.island_manager,
            &mut self.broad_phase,
            &mut self.narrow_phase,
            &mut self.rigid_body_set,
            &mut self.collider_set,
            &mut self.impulse_joint_set,
            &mut self.multibody_joint_set,
            &mut self.ccd_solver,
            &(), 
            &()
        );

        let qp = self.broad_phase.as_query_pipeline(&DefaultQueryDispatcher, &self.rigid_body_set, &self.collider_set, QueryFilter::new());

        self.entities.iter_mut()
            .for_each(|(_, entity)| {
                let eye_offsets = [
                    0.0f32,
                    5.0, 10.0, 15.0, 20.0,
                    -5.0, -10.0, -15.0, -20.0,
                ];

                let rb = self.rigid_body_set.get(entity.rb).unwrap();
                let qp = qp.with_filter(QueryFilter::new().exclude_collider(entity.col));
                let dir = rb.position().rotation;
                let forward = dir.transform_vector(&Vector2::x());

                let rays = core::array::from_fn::<f32, 9, _>(|i| {
                    let rads = eye_offsets[i].to_radians();
                    let offset_rot = UnitComplex::new(rads);
                    let dir = offset_rot.transform_vector(&forward).normalize();


                    let ray = Ray {
                        origin: Vector2::new(rb.position().translation.x, rb.position().translation.y).into(),
                        dir,
                    };

                    let ray = qp.cast_ray(&ray, 32.0, false);
                    if let Some((v, dist)) = ray {
                        let dist = if self.foods.contains(&v) { dist } else { -dist };
                        dist / 32.0
                    } else {
                        0.0
                    }
                });

                let mut buf = ForwardBuffers::new();
                let input = [
                    entity.health,
                    entity.angular_velocity,
                    entity.velocity.len() as f32 * 0.0001,
                    dir.angle(),

                    rays[0], rays[1], rays[2], rays[3],
                    rays[4], rays[5], rays[6], rays[7],
                    rays[8],
                ];

                let out = entity.brain.forward(&mut buf, &input);

                entity.velocity += Vector2::new(out[1], out[2]);
                entity.angular_velocity += 0.5 * out[0];
                entity.health -= 0.0004 + entity.angular_velocity * 0.002 + entity.velocity.len() as f32 * 0.008;
                
            });


        self.entities.iter_mut()
            .for_each(|(_, entity)| {
                let rb = self.rigid_body_set.get_mut(entity.rb).unwrap();
                rb.set_vels(RigidBodyVelocity { linvel: entity.velocity, angvel: (entity.angular_velocity) % TAU }, true);
            });


        self.foods
            .retain(|food| {
                let mut contact = false;
                for pair in self.narrow_phase.contact_pairs_with(*food) {
                    if !pair.has_any_active_contact { continue }

                    let (_, oth) = 
                        if pair.collider1 == *food { (pair.collider1, pair.collider2) }
                        else { (pair.collider2, pair.collider1) };

                    let coll = self.collider_set.get(oth).unwrap();
                    let parent = coll.parent().unwrap();
                    let rb = self.rigid_body_set.get(parent).unwrap();
                    let entity_id = rb.user_data as u32;
                    let entity = self.entities.get_mut(&entity_id).unwrap();
                    entity.health += 1.0;

                    let coll = coll.clone();
                    let coll = self.collider_set.insert(coll);
                    let mut rb = RigidBodyBuilder::new(RigidBodyType::KinematicVelocityBased)
                        .pose(*rb.position())
                        .user_data(self.entity_counter as u128)
                        .build();


                    let entity = Entity {
                        velocity: Vector2::zeros(),
                        angular_velocity: 0.0,
                        health: 1.0,
                        rb: self.rigid_body_set.insert(rb),
                        col: coll,
                        brain: {
                            let mut brain = entity.brain.clone();
                            brain.mutate(&mut self.rng, 0.1);
                            brain
                        },
                        archetype: entity.archetype.clone(),
                    };

                    self.collider_set.set_parent(coll, Some(entity.rb), &mut self.rigid_body_set);
                    self.entities.insert(self.entity_counter, entity);
                    self.entity_counter += 1;

                    contact = true;
                    break;
                }

                !contact
            });

        self.entities.retain(|_, x| {
            let cond = x.health > 0.0;
            if !cond {
                self.rigid_body_set.remove(x.rb, &mut self.island_manager, &mut self.collider_set, &mut self.impulse_joint_set, &mut self.multibody_joint_set, true);
            }
            cond
        });

        if (self.tick % 2) == 0 {
            self.new_food();
        }
    }



    pub fn render(&self, renderer: &mut Renderer) {
        for (_, entity) in &self.entities {
            let rb = self.rigid_body_set.get(entity.rb).unwrap();
            let model = Mat4::from_rotation_translation(
                Quat::from_rotation_z(rb.rotation().angle()),
                Vec3::new(rb.position().translation.x, rb.position().translation.y, 0.0)
            );

            for cell in &entity.archetype.cells {
                let offset = cell.offset.as_vec2();

                let mat = Mat4::from_translation(
                    Vec3::new(offset.x, offset.y, 0.0),
                );

                let mat = model * mat;

                renderer.particle_at(ParticleInstance::new(mat, cell.kind.colour().with_w(entity.health)));
            }
        }



        for food in &self.foods {
            let rb = self.collider_set.get(*food).unwrap().parent().unwrap();
            let pos = self.rigid_body_set.get(rb).unwrap().position();
            let mat = Mat4::from_translation(Vec3::new(pos.translation.x, pos.translation.y, 0.0));
            renderer.particle_at(ParticleInstance::new(
                mat,
                Vec4::new(1.0, 0.0, 0.0, 1.0)
            ));
        }
    }


    pub fn new_entity(&mut self) {
        let pos = Vector2::new(self.rng.random_range(0..self.world.x) as f32, self.rng.random_range(0..self.world.y) as f32);
        let body = RigidBodyBuilder::new(RigidBodyType::KinematicVelocityBased)
            .pose(Isometry::new(pos, 0.0))
            .user_data(self.entity_counter as u128)
            .build();

        let archetype = EntityArchetype::new();

        let mut shapes = vec![];
        for cell in &archetype.cells {
            shapes.push((Isometry::new(Vector2::new(cell.offset.x as _, cell.offset.y as _), 0.0), ColliderShape::cuboid(0.5, 0.5)));
        }

        let shape = Compound::new(shapes);
        let coll = ColliderBuilder::new(ColliderShape::new(shape)).build();
        let rb = self.rigid_body_set.insert(body);
        let coll = self.collider_set.insert(coll);

        self.collider_set.set_parent(coll, Some(rb), &mut self.rigid_body_set);

        let entity = Entity {
            velocity: Vector2::zeros(),
            angular_velocity: 0.0,
            rb,
            col: coll,
            brain: BetterNN::new(&mut self.rng, &[12, 32, 3]),
            health: 1.0,
            archetype,
        };

        self.entities.insert(self.entity_counter, entity);
        self.entity_counter += 1;
    }


    pub fn new_food(&mut self) {
        let pos = Vector2::new(self.rng.random_range(0..self.world.x) as f32, self.rng.random_range(0..self.world.y) as f32);
        let body = RigidBodyBuilder::new(RigidBodyType::KinematicPositionBased)
            .pose(Isometry::new(pos, 0.0))
            .user_data(1)
            .build();


        let collider = ColliderBuilder::new(ColliderShape::cuboid(0.5, 0.5))
            .sensor(true)
            .build();

        let rb = self.rigid_body_set.insert(body);
        let collider = self.collider_set.insert(collider);
        self.collider_set.set_parent(collider, Some(rb), &mut self.rigid_body_set);
        self.foods.insert(collider);
    }
}



impl CellKind {
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



impl EntityArchetype {
    pub fn new() -> Self {
        Self {
            cells: vec![
                Cell { offset: IVec2::ZERO, kind: CellKind::BasicCell },
            ],
        }
    }
}
