use wgpu::util::DeviceExt;
use std::future::Future;
use std::pin::Pin;

// Compute shader code (WGSL)
const COMPUTE_SHADER: &str = r#"
[[block]] struct Data {
    input: array<u32>;
    output: array<u32>;
};

[[group(0), binding(0)]] var<storage> data: Data;

[[stage(compute), workgroup_size(64)]] fn main([[builtin(global_invocation_id)]] global_id: vec3<u32>) -> void {
    let index: u32 = global_id.x;
    data.output[index] = data.input[index] * data.input[index];
    return;
}
"#;

pub struct WebGPUCompute {
    device: wgpu::Device,
    queue: wgpu::Queue,
    bind_group_layout: wgpu::BindGroupLayout,
    compute_pipeline: wgpu::ComputePipeline,
}

impl WebGPUCompute {
    // Initialize the WebGPUCompute struct
    pub async fn new() -> Self {
        // Create an instance of WebGPU
        let instance = wgpu::Instance::new(wgpu::Backends::PRIMARY);
        let adapter = instance.request_adapter(&wgpu::RequestAdapterOptions::default()).await.unwrap();
        let (device, queue) = adapter.request_device(&wgpu::DeviceDescriptor::default(), None).await.unwrap();

        // Create a shader module with the compute shader code
        let shader = device.create_shader_module(&wgpu::ShaderModuleDescriptor {
            label: Some("Compute Shader"),
            source: wgpu::ShaderSource::Wgsl(COMPUTE_SHADER.into()),
        });

        // Create a bind group layout
        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Bind Group Layout"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: wgpu::BufferSize::new(8),
                },
                count: None,
            }],
        });

        // Create a compute pipeline
        let compute_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Compute Pipeline"),
            layout: Some(&device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Pipeline Layout"),
                bind_group_layouts: &[&bind_group_layout],
                push_constant_ranges: &[],
            })),
            module: &shader,
            entry_point: "main",
        });

        Self {
            device,
            queue,
            bind_group_layout,
            compute_pipeline,
        }
    }

    // Run the WebGPU computation
    pub fn run_gpu_computation(
        &self,
        input_data: Vec<u32>,
    ) -> Pin<Box<dyn Future<Output = Result<Vec<u32>, String>> + '_>> {
        let device = self.device.clone();
        let queue = self.queue.clone();
        let bind_group_layout = self.bind_group_layout.clone();
        let compute_pipeline = self.compute_pipeline.clone();

        Box::pin(async         move {
            // Create a buffer with input data
            let input_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Input Buffer"),
                contents: bytemuck::cast_slice(&input_data),
                usage: wgpu::BufferUsages::STORAGE,
            });

            // Create a buffer to store the output data
            let output_buffer = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("Output Buffer"),
                size: (input_data.len() * std::mem::size_of::<u32>()) as wgpu::BufferAddress,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });

            // Create a bind group
            let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("Bind Group"),
                layout: &bind_group_layout,
                entries: &[wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer: &input_buffer,
                        offset: 0,
                        size: wgpu::BufferSize::new((input_data.len() * std::mem::size_of::<u32>()) as wgpu::BufferAddress),
                    }),
                }],
            });

            // Create a command encoder and record the compute pass
            let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Compute Encoder"),
            });
            {
                let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("Compute Pass"),
                });
                compute_pass.set_pipeline(&compute_pipeline);
                compute_pass.set_bind_group(0, &bind_group, &[]);
                compute_pass.dispatch(input_data.len() as u32, 1, 1);
            }

            // Copy the output data from the GPU to a buffer that can be read by the CPU
            encoder.copy_buffer_to_buffer(
                &output_buffer,
                0,
                &input_buffer,
                0,
                (input_data.len() * std::mem::size_of::<u32>()) as wgpu::BufferAddress,
            );

            // Submit the command buffer to the queue
            queue.submit(Some(encoder.finish()));

            // Read the output data from the buffer
            let buffer_slice = output_buffer.slice(..);
            let buffer_future = buffer_slice.map_async(wgpu::MapMode::Read);
            device.poll(wgpu::Maintain::Wait);
            if let Ok(()) = buffer_future.await {
                let data = buffer_slice.get_mapped_range();
                let result: Vec<u32> = data
                    .chunks_exact(4)
                    .map(|b| u32::from_ne_bytes(b.try_into().unwrap()))
                    .collect();
                Ok(result)
            } else {
                Err("Failed to run compute on the GPU.".to_string())
            }
        }
    )
}
}