use std::{marker::PhantomData, num::NonZero};

use bytemuck::Pod;
use wgpu::{util::StagingBelt, BindGroupDescriptor, BindGroupLayout, BindGroupLayoutDescriptor, BindingType, BufferDescriptor, BufferUsages, ShaderStages};

#[derive(Debug)]
pub struct Uniform<T: Pod> {
    buffer: wgpu::Buffer,
    pub bind_group: wgpu::BindGroup,
    bind_group_layout: wgpu::BindGroupLayout,
    binding: u32,
    marker: PhantomData<T>,
}


impl<T: Pod> Uniform<T> {
    pub fn new(name: &str, device: &wgpu::Device, binding: u32, visibility: ShaderStages) -> Self {
        let buffer = device.create_buffer(&BufferDescriptor{
            label: Some(&format!("{name}-buffer")),
            size: size_of::<T>() as u64,
            usage: BufferUsages::UNIFORM.union(BufferUsages::COPY_DST),
            mapped_at_creation: false,
        });


        let bind_group_layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some(&format!("{name}-bind-group-descriptor")),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility,
                    ty: BindingType::Buffer { ty: wgpu::BufferBindingType::Uniform, has_dynamic_offset: false, min_binding_size: None },
                    count: None,
                }
            ],
        });

        
        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some(&format!("{name}-bind-group")),
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: buffer.as_entire_binding(),
                }
            ]
        });


        Self {
            buffer,
            bind_group,
            binding,
            bind_group_layout,
            marker: PhantomData,
        }
    }


    pub fn from_bgl(name: &str, device: &wgpu::Device, binding: u32, bgl: BindGroupLayout) -> Self {
        let buffer = device.create_buffer(&BufferDescriptor{
            label: Some(&format!("{name}-buffer")),
            size: size_of::<T>() as u64,
            usage: BufferUsages::UNIFORM.union(BufferUsages::COPY_DST),
            mapped_at_creation: false,
        });

        
        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some(&format!("{name}-bind-group")),
            layout: &bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: buffer.as_entire_binding(),
                }
            ]
        });


        Self {
            buffer,
            bind_group,
            binding,
            bind_group_layout: bgl,
            marker: PhantomData,
        }
    }


    pub fn update(&self, queue: &wgpu::Queue, data: &T) {
        queue.write_buffer(&self.buffer, 0, bytemuck::bytes_of(data));
    }


    pub fn update_belt(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        device: &wgpu::Device,
        belt: &mut StagingBelt,
        data: &T
    ) {

        let mut buf = belt.write_buffer(encoder, &self.buffer, 0, NonZero::new(size_of::<T>() as _).unwrap(), device);
        buf.copy_from_slice(bytemuck::bytes_of(data));
    }


    pub fn use_uniform(&self, render_pass: &mut wgpu::RenderPass) {
        render_pass.set_bind_group(self.binding, &self.bind_group, &[]);
    }


    pub fn bind_group_layout(&self) -> &wgpu::BindGroupLayout {
        &self.bind_group_layout
    }
}
