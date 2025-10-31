use wgpu::ShaderSource;

pub fn create_shader_module(device: &wgpu::Device, mut desc: wgpu::ShaderModuleDescriptor) -> wgpu::ShaderModule {
    let str = match desc.source {
        wgpu::ShaderSource::Wgsl(cow) => cow,
        _ => todo!(),
    };

    let data = parse_shader_data(&*str);

    desc.source = ShaderSource::Wgsl(std::borrow::Cow::Owned(data));
    device.create_shader_module(desc)
}



fn parse_shader_data(str: &str) -> String {
    let mut data = String::with_capacity(str.len());

    for line in str.lines() {
        if line.starts_with("#include") {
            let (_, path) = line.split_once(' ').unwrap();
            println!("opening {path}");
            let file = std::fs::read_to_string(path).unwrap();
            let file = parse_shader_data(&file);
            data.push_str("\n// INCLUDE START\n");
            data.push_str(file.as_str());
            data.push_str("\n// INCLUDE END\n");
            continue;
        }
        data.push_str(line);
        data.push('\n');
    }

    data
}
