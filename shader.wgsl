struct Vertex {
    @location(0) position: vec2<f32>,
}


struct Instance {
    @location(1) colour: vec4<f32>,
    @location(2) mat0: vec4<f32>,
    @location(3) mat1: vec4<f32>,
    @location(4) mat2: vec4<f32>,
    @location(5) mat3: vec4<f32>,
}


struct Fragment {
    @builtin(position) position: vec4<f32>,
    @location(0) modulate: vec4<f32>,
}


struct Uniforms {
    projection: mat4x4<f32>,
}


@group(0) @binding(0) var<uniform> u : Uniforms;


@vertex
fn vs_main(vertex: Vertex, instance: Instance) -> Fragment {
    var output : Fragment;

    let mat = mat4x4(instance.mat0, instance.mat1,
                     instance.mat2, instance.mat3);
    let pos = mat * vec4(vertex.position, 0.0, 1.0);
    output.position = u.projection * pos;
    output.modulate = instance.colour;


    return output;
}



@fragment
fn fs_main(fragment: Fragment) -> @location(0) vec4<f32> {
    return fragment.modulate;
}
