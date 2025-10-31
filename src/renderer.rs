use bytemuck::{Pod, Zeroable};
use glam::{Mat4, Quat, Vec2, Vec3, Vec4};
use wgpu::{util::{DeviceExt, StagingBelt}, wgc::id::markers::StagingBuffer};
use winit::window::Window;

use crate::{buffer::ResizableBuffer, uniform::Uniform};


const QUAD_VERTICES : &[Vec2] = &[
    Vec2::new(0.5, 0.5),
    Vec2::new(0.5, -0.5),
    Vec2::new(-0.5, -0.5),
    Vec2::new(-0.5, -0.5),
    Vec2::new(-0.5, 0.5),
    Vec2::new(0.5, 0.5),
];



#[derive(Debug, Clone, Copy, Pod, Zeroable)]
#[repr(C)]
struct ParticleVertex(Vec2);


#[derive(Debug, Clone, Copy, Pod, Zeroable)]
#[repr(C)]
pub struct ParticleInstance {
    pub colour: Vec4,
    pub mat0: Vec4,
    pub mat1: Vec4,
    pub mat2: Vec4,
    pub mat3: Vec4,
}


#[derive(Debug, Clone, Copy, Pod, Zeroable)]
#[repr(C)]
struct ParticleUniform {
    projection: Mat4,
}


struct ParticlePipeline {
    render_pipeline: wgpu::RenderPipeline,
    vertices: wgpu::Buffer,
    instances: ResizableBuffer<ParticleInstance>,
    uniform: Uniform<ParticleUniform>,
}


pub struct Renderer {
    pub window: &'static Window,

    pub device: wgpu::Device,
    queue: wgpu::Queue,
    surface: wgpu::Surface<'static>,
    config: wgpu::SurfaceConfiguration,

    framebuffer: wgpu::TextureView,

    particle_pipeline: ParticlePipeline,
    particle_instances: Vec<ParticleInstance>,
    belt: StagingBelt,

    pub camera_pos: Vec2,
    pub camera_ortho: f32,
}



impl Renderer {
    pub async fn new(window: Window) -> Self {
        let window = Box::leak(Box::new(window));
        let size = window.inner_size();
        
        let mut instance = wgpu::Instance::new(&wgpu::InstanceDescriptor::default());
        let surface = instance.create_surface(&*window).unwrap();

        let adapter = instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: Some(&surface),
            force_fallback_adapter: false,
        }).await.unwrap();



        let (device, queue) = adapter.request_device(&wgpu::DeviceDescriptor {
            label: Some("device"),
            required_features: wgpu::Features::empty(),
            required_limits: {
                let mut limits = wgpu::Limits::downlevel_defaults();
                limits.max_texture_dimension_2d = adapter.limits().max_texture_dimension_2d;
                limits
            },
            memory_hints: wgpu::MemoryHints::Performance,
            trace: wgpu::Trace::Off,
        }).await.unwrap();


        let surface_capabilities = surface.get_capabilities(&adapter);


        let surface_format = surface_capabilities.formats.iter()
            .find(|f| f.is_srgb())
            .copied()
            .unwrap_or(surface_capabilities.formats[0]);


        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format: surface_format,
            width: size.width,
            height: size.height,
            present_mode: wgpu::PresentMode::Fifo,
            desired_maximum_frame_latency: 2,
            alpha_mode: surface_capabilities.alpha_modes[0],
            view_formats: vec![],
        };


        surface.configure(&device, &config);


        let framebuffer = create_multisampled_framebuffer(&device, &config);


        let pipeline = {
            let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("particle-shader"),
                source: wgpu::ShaderSource::Wgsl(include_str!("../shader.wgsl").into()),
            });


            let uniform = Uniform::new(
                "particle-shader-uniform",
                &device,
                0,
                wgpu::ShaderStages::VERTEX_FRAGMENT
            );



            let rpl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("particle-shader-pipeline-desc"),
                bind_group_layouts: &[uniform.bind_group_layout()],
                push_constant_ranges: &[],
            });


            let targets = [Some(wgpu::ColorTargetState {
                format: config.format,
                blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                write_mask: wgpu::ColorWrites::ALL,
            })];

            let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("particle-render-pipeline"),
                layout: Some(&rpl),
                vertex: wgpu::VertexState {
                    module: &shader,
                    entry_point: Some("vs_main"),
                    compilation_options: wgpu::PipelineCompilationOptions::default(),
                    buffers: &[ParticleVertex::desc(), ParticleInstance::desc()],
                },


                fragment: Some(wgpu::FragmentState {
                    module: &shader,
                    entry_point: Some("fs_main"),
                    compilation_options: wgpu::PipelineCompilationOptions::default(),
                    targets: &targets,
                }),


                primitive: wgpu::PrimitiveState {
                    topology: wgpu::PrimitiveTopology::TriangleList,
                    strip_index_format: None,
                    front_face: wgpu::FrontFace::Ccw,
                    cull_mode: None,
                    polygon_mode: wgpu::PolygonMode::Fill,
                    unclipped_depth: false,
                    conservative: false,
                },


                depth_stencil: None,


                multisample: wgpu::MultisampleState {
                    count: 1,
                    mask: !0,
                    alpha_to_coverage_enabled: false,
                },


                multiview: None,
                cache: None,
            });


            let vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("particle-shader-quad-vertices"),
                contents: bytemuck::cast_slice(QUAD_VERTICES),
                usage: wgpu::BufferUsages::VERTEX,
            });


            let instance_buffer = ResizableBuffer::new(
                "particle-shader-instance-buffer", 
                &device,
                wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
                128
            );


            ParticlePipeline {
                render_pipeline: pipeline,
                vertices: vertex_buffer,
                instances: instance_buffer,
                uniform,
            }
        };


        Self {
            window,
            device,
            queue,
            surface,
            config,
            framebuffer,
            particle_pipeline: pipeline,
            belt: StagingBelt::new(1024 * 1024),
            particle_instances: vec![],
            camera_pos: Vec2::ZERO,
            camera_ortho: 3.0,
        }

    }


    pub fn resize(&mut self) {
        let size = self.window.inner_size();
        self.config.width = size.width;
        self.config.height = size.height;
        self.surface.configure(&self.device, &self.config);

    }


    pub fn vp(&self) -> Mat4 {
        let n = self.camera_ortho;
        let aspect_ratio = 16.0/9.0;
        let left = -n*0.5*aspect_ratio;
        let right = n*0.5*aspect_ratio;
        let down = -n*0.5;
        let up = n*0.5;

        let projection = Mat4::orthographic_rh(
            left + self.camera_pos.x, right + self.camera_pos.x,
            down + self.camera_pos.y, up + self.camera_pos.y,
            -1.0, 1.0
        );

        projection
    }


    pub fn screenspace_to_worldspace(&self, pos: Vec2) -> Vec2 {
        let size = self.window.inner_size();
        let size = Vec2::new(size.width as f32, size.height as f32);

        let ndc = (pos / size) * 2.0 - 1.0;
        let inv_vp = self.vp().inverse();

        let ndc_matrix = {
            let mut mat = Mat4::IDENTITY.to_cols_array_2d();

            mat[0][0] = ndc.x;
            mat[1][1] = ndc.y;
            Mat4::from_cols_array_2d(&mat)
        };

        let mat = (ndc_matrix * inv_vp).to_cols_array_2d();
        Vec2::new(mat[0][0], -mat[1][1])
    }


    pub fn render(&mut self, mut encoder: wgpu::CommandEncoder) {
        // prepare buffers
        if self.particle_instances.len() > 0 {
            self.particle_pipeline.instances.resize(
                &self.device,
                &mut encoder,
                self.particle_instances.len()
            );

            self.particle_pipeline.instances.write(
                &mut self.belt,
                &mut encoder,
                &self.device,
                0,
                &self.particle_instances
            );
        }
        

        // draw

        let output = self.surface.get_current_texture().unwrap();
        let view = output.texture.create_view(&wgpu::TextureViewDescriptor::default());

        let projection = self.vp();
        self.particle_pipeline.uniform.update(&self.queue, &ParticleUniform { projection });

        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("render-pass"),
            color_attachments: &[
                Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color { r: 0.1, g: 0.2, b: 0.1, a: 1.0 }),
                        store: wgpu::StoreOp::Store,
                    },
                })
            ],

            depth_stencil_attachment: None,
            ..Default::default()
        });

        pass.set_pipeline(&self.particle_pipeline.render_pipeline);

        pass.set_vertex_buffer(0, self.particle_pipeline.vertices.slice(..));
        pass.set_vertex_buffer(1, self.particle_pipeline.instances.buffer.slice(..));
        self.particle_pipeline.uniform.use_uniform(&mut pass);

        if self.particle_instances.len() > 0 {
            pass.draw(
                0..QUAD_VERTICES.len() as _,
                0..self.particle_instances.len() as _
            );
        }


        drop(pass);

        self.belt.finish();
        self.queue.submit(core::iter::once(encoder.finish()));
        self.belt.recall();

        output.present();

        self.particle_instances.clear();
    }


    pub fn particle_at(&mut self, particle: ParticleInstance) {
        self.particle_instances.push(particle);
    }
}


fn create_multisampled_framebuffer(
    device: &wgpu::Device,
    config: &wgpu::SurfaceConfiguration,
) -> wgpu::TextureView {
    let size = wgpu::Extent3d {
        width: config.width,
        height: config.height,
        depth_or_array_layers: 1,
    };


    let multisampled_frame_descriptor = &wgpu::TextureDescriptor {
        size,
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: config.format,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
        label: None,
        view_formats: &[],
    };

    device
        .create_texture(multisampled_frame_descriptor)
        .create_view(&wgpu::TextureViewDescriptor::default())
}




impl ParticleVertex {
    fn desc() -> wgpu::VertexBufferLayout<'static> {
        wgpu::VertexBufferLayout {
            array_stride: size_of::<Self>() as _,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &[
                wgpu::VertexAttribute {
                    format: wgpu::VertexFormat::Float32x2,
                    offset: 0,
                    shader_location: 0,
                }
            ],
        }
    }
}

impl ParticleInstance {
    pub fn new(mat: Mat4, colour: Vec4) -> Self {
        let vecs = unsafe { core::mem::transmute::<Mat4, [Vec4; 4]>(mat) };

        Self {
            colour,
            mat0: vecs[0],
            mat1: vecs[1],
            mat2: vecs[2],
            mat3: vecs[3],
        }

    }


    fn desc() -> wgpu::VertexBufferLayout<'static> {
        const ATTRS : &[wgpu::VertexAttribute] = &[
            wgpu::VertexAttribute {
                format: wgpu::VertexFormat::Float32x4,
                offset: core::mem::offset_of!(ParticleInstance, colour) as _,
                shader_location: 1,
            },

            wgpu::VertexAttribute {
                format: wgpu::VertexFormat::Float32x4,
                offset: core::mem::offset_of!(ParticleInstance, mat0) as _,
                shader_location: 2,
            },
            wgpu::VertexAttribute {
                format: wgpu::VertexFormat::Float32x4,
                offset: core::mem::offset_of!(ParticleInstance, mat1) as _,
                shader_location: 3,
            },
            wgpu::VertexAttribute {
                format: wgpu::VertexFormat::Float32x4,
                offset: core::mem::offset_of!(ParticleInstance, mat2) as _,
                shader_location: 4,
            },
            wgpu::VertexAttribute {
                format: wgpu::VertexFormat::Float32x4,
                offset: core::mem::offset_of!(ParticleInstance, mat3) as _,
                shader_location: 5,
            },
        ];

        wgpu::VertexBufferLayout {
            array_stride: size_of::<Self>() as _,
            step_mode: wgpu::VertexStepMode::Instance,
            attributes: ATTRS,
        }
    }
}
