#![feature(portable_simd)]


use std::env;

use evolutionary_biology::{new_sim::Sim, renderer::{ParticleInstance, Renderer}, Simulation};
use glam::Vec2;
use winit::{dpi::LogicalSize, event_loop::ActiveEventLoop, window::Cursor};


const MUTATION_FREQ : usize = 120;


struct AppData {
    renderer: Renderer,
    simulation: Simulation,
    tick_count: usize,
    mutation_count: usize,
    speed: bool,
    stopped: bool,

    is_middle_click_down: bool,
    prev_cursor: Vec2,
}


struct App {
    app: Option<AppData>,
}



impl winit::application::ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        let window = event_loop.create_window(
            winit::window::Window::default_attributes()
                .with_inner_size(LogicalSize::new(1920/2, 1080/2))
        ).unwrap();

        let sim = if let Some(str) = env::args().skip(1).next() {
            let str = std::fs::read(&str).unwrap();
            let sim = serde_json::from_slice(&str).unwrap();
            sim
        } else {
            Simulation::new()
        };

        self.app = Some(AppData {
            renderer: pollster::block_on(Renderer::new(window)),
            simulation: sim,
            tick_count: 0,
            speed: false,
            mutation_count: 0,
            prev_cursor: Vec2::ZERO,
            is_middle_click_down: false,
            stopped: false,
        });


    }

    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        window_id: winit::window::WindowId,
        event: winit::event::WindowEvent,
    ) {
        let Some(app) = &mut self.app
        else { unreachable!() };

        match event {
            winit::event::WindowEvent::CloseRequested => {
                let data = &app.simulation;
                let data = serde_json::to_string(data);
                std::fs::write("save.json", data.unwrap()).unwrap();
                event_loop.exit();
            },


            winit::event::WindowEvent::RedrawRequested => {
                //let world = box2d::world::World::new(box2d::math::Vec2::new_zero());


                if !app.stopped {
                    if app.speed {
                        app.simulation.tick();
                        app.simulation.tick();
                        app.simulation.tick();
                        app.simulation.tick();
                        app.simulation.tick();
                        app.simulation.tick();
                        app.simulation.tick();
                        app.simulation.tick();
                        app.simulation.tick();
                        app.simulation.tick();
                        app.simulation.tick();
                        app.simulation.tick();
                        app.simulation.tick();
                        app.simulation.tick();
                        app.simulation.tick();
                        app.simulation.tick();
                        app.simulation.tick();
                        app.simulation.tick();
                        app.simulation.tick();
                        app.simulation.tick();
                        app.simulation.tick();
                        app.simulation.tick();
                        app.simulation.tick();
                        app.simulation.tick();
                        app.simulation.tick();
                        app.simulation.tick();
                        app.simulation.tick();
                        app.simulation.tick();
                        app.simulation.tick();
                        app.simulation.tick();
                        app.simulation.tick();
                        app.simulation.tick();
                        app.simulation.tick();
                        app.simulation.tick();
                        app.simulation.tick();
                        app.simulation.tick();
                        app.simulation.tick();
                        app.simulation.tick();
                        app.simulation.tick();
                        app.simulation.tick();
                        app.simulation.tick();
                        app.simulation.tick();
                        app.simulation.tick();
                        app.simulation.tick();
                        app.simulation.tick();
                        app.simulation.tick();
                        app.simulation.tick();
                        app.simulation.tick();
                        app.simulation.tick();
                        app.simulation.tick();
                        app.simulation.tick();
                        app.simulation.tick();
                        app.simulation.tick();
                        app.simulation.tick();
                        app.simulation.tick();
                        app.simulation.tick();
                        app.simulation.tick();
                        app.simulation.tick();
                        app.simulation.tick();
                        app.simulation.tick();
                        app.simulation.tick();
                        app.simulation.tick();
                        app.simulation.tick();
                        app.simulation.tick();
                    }

                    app.simulation.tick();
                }


                app.simulation.render(&mut app.renderer);

                let command_encoder = app.renderer.device.create_command_encoder(
                    &wgpu::CommandEncoderDescriptor {
                        label: Some("command-encoder"),
                    }
                );

                app.renderer.render(command_encoder);


                app.renderer.window.request_redraw();
            },


            winit::event::WindowEvent::KeyboardInput { device_id, event, is_synthetic } => {
                match (event.physical_key, event.state) {
                    (winit::keyboard::PhysicalKey::Code(winit::keyboard::KeyCode::Enter), winit::event::ElementState::Pressed) => self.app.as_mut().unwrap().speed = !self.app.as_ref().unwrap().speed,
                    (winit::keyboard::PhysicalKey::Code(winit::keyboard::KeyCode::Space), winit::event::ElementState::Pressed) => self.app.as_mut().unwrap().stopped = !self.app.as_ref().unwrap().stopped,
                    _ => (),
                }
            }


            winit::event::WindowEvent::MouseInput { device_id, state, button } => {
                match button {
                    winit::event::MouseButton::Left => app.is_middle_click_down = state.is_pressed(),
                    _ => (),
                }
            }


            winit::event::WindowEvent::CursorMoved { device_id, position } => {
                let current_pos = Vec2::new(position.x as f32, position.y as f32);
                if !app.is_middle_click_down {
                    app.renderer.window.set_cursor(Cursor::Icon(winit::window::CursorIcon::Default));
                    app.prev_cursor = current_pos;
                    return;
                };

                app.renderer.window.set_cursor(Cursor::Icon(winit::window::CursorIcon::Grabbing));


                let old_wp = app.renderer.screenspace_to_worldspace(app.prev_cursor);
                let curr_wp = app.renderer.screenspace_to_worldspace(current_pos);

                let delta = old_wp - curr_wp;
                app.renderer.camera_pos += delta;
                app.prev_cursor = current_pos;
            }


            winit::event::WindowEvent::MouseWheel { device_id, delta, phase } => {
                let delta = match delta {
                    winit::event::MouseScrollDelta::LineDelta(x, y) => Vec2::new(x * 16.0, y * 16.0),
                    winit::event::MouseScrollDelta::PixelDelta(pos) => Vec2::new(pos.x as f32, pos.y as f32),
                };

                app.renderer.camera_ortho += delta.y;
                app.renderer.camera_ortho = app.renderer.camera_ortho.max(8.0);
            }


            winit::event::WindowEvent::Resized(_) => {
                app.renderer.resize();
            }

            _ => (),
        }
    }
}




fn main() {
    tracing_subscriber::fmt().init();

    let event_loop = winit::event_loop::EventLoop::builder()
        .build().unwrap();

    event_loop.set_control_flow(winit::event_loop::ControlFlow::Poll);

    let mut app = NewApp {
        app: None,
    };

    event_loop.run_app(&mut app).unwrap();
}



struct NewAppData {
    renderer: Renderer,
    simulation: Sim,
    tick_count: usize,
    mutation_count: usize,
    speed: bool,
    stopped: bool,

    is_middle_click_down: bool,
    prev_cursor: Vec2,
}


struct NewApp {
    app: Option<NewAppData>,
}


impl winit::application::ApplicationHandler for NewApp {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        let window = event_loop.create_window(
            winit::window::Window::default_attributes()
                .with_inner_size(LogicalSize::new(1920/2, 1080/2))
        ).unwrap();

        let sim = if let Some(str) = env::args().skip(1).next() {
            todo!()
        } else {
            Sim::new()
        };

        self.app = Some(NewAppData {
            renderer: pollster::block_on(Renderer::new(window)),
            simulation: sim,
            tick_count: 0,
            speed: false,
            mutation_count: 0,
            prev_cursor: Vec2::ZERO,
            is_middle_click_down: false,
            stopped: false,
        });


    }

    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        window_id: winit::window::WindowId,
        event: winit::event::WindowEvent,
    ) {
        let Some(app) = &mut self.app
        else { unreachable!() };

        match event {
            winit::event::WindowEvent::CloseRequested => {
                event_loop.exit();
            },


            winit::event::WindowEvent::RedrawRequested => {
                //let world = box2d::world::World::new(box2d::math::Vec2::new_zero());


                if !app.stopped {
                    if app.speed {
                        for _ in 0..30 {
                            app.simulation.step();
                        }
                    }

                    app.simulation.step();
                }


                app.simulation.render(&mut app.renderer);

                let command_encoder = app.renderer.device.create_command_encoder(
                    &wgpu::CommandEncoderDescriptor {
                        label: Some("command-encoder"),
                    }
                );

                app.renderer.render(command_encoder);


                app.renderer.window.request_redraw();
            },


            winit::event::WindowEvent::KeyboardInput { device_id, event, is_synthetic } => {
                match (event.physical_key, event.state) {
                    (winit::keyboard::PhysicalKey::Code(winit::keyboard::KeyCode::Enter), winit::event::ElementState::Pressed) => self.app.as_mut().unwrap().speed = !self.app.as_ref().unwrap().speed,
                    (winit::keyboard::PhysicalKey::Code(winit::keyboard::KeyCode::Space), winit::event::ElementState::Pressed) => self.app.as_mut().unwrap().stopped = !self.app.as_ref().unwrap().stopped,
                    _ => (),
                }
            }


            winit::event::WindowEvent::MouseInput { device_id, state, button } => {
                match button {
                    winit::event::MouseButton::Left => app.is_middle_click_down = state.is_pressed(),
                    _ => (),
                }
            }


            winit::event::WindowEvent::CursorMoved { device_id, position } => {
                let current_pos = Vec2::new(position.x as f32, position.y as f32);
                if !app.is_middle_click_down {
                    app.renderer.window.set_cursor(Cursor::Icon(winit::window::CursorIcon::Default));
                    app.prev_cursor = current_pos;
                    return;
                };

                app.renderer.window.set_cursor(Cursor::Icon(winit::window::CursorIcon::Grabbing));


                let old_wp = app.renderer.screenspace_to_worldspace(app.prev_cursor);
                let curr_wp = app.renderer.screenspace_to_worldspace(current_pos);

                let delta = old_wp - curr_wp;
                app.renderer.camera_pos += delta;
                app.prev_cursor = current_pos;
            }


            winit::event::WindowEvent::MouseWheel { device_id, delta, phase } => {
                let delta = match delta {
                    winit::event::MouseScrollDelta::LineDelta(x, y) => Vec2::new(x * 16.0, y * 16.0),
                    winit::event::MouseScrollDelta::PixelDelta(pos) => Vec2::new(pos.x as f32, pos.y as f32),
                };

                app.renderer.camera_ortho += delta.y;
                app.renderer.camera_ortho = app.renderer.camera_ortho.max(8.0);
            }


            winit::event::WindowEvent::Resized(_) => {
                app.renderer.resize();
            }

            _ => (),
        }
    }
}
