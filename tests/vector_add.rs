use rust_cuda_runtime::{DeviceBuffer, launch_vector_add};

#[test]
fn test_vector_add()
{
  let n=1024;
  let a: Vec<f32> = (0..n).map(|i| i as f32).collect();
  let b: Vec<f32> = (0..n).map(|i| (2 * i) as f32).collect();
  let mut c = vec![0.0f32; n];
  
  let d_a = DeviceBuffer::new(n);
  let d_b = DeviceBuffer::new(n);
  let d_c = DeviceBuffer::new(n);

  d_a.copy_from_host(&a);
  d_b.copy_from_host(&b);

  unsafe{ launch_vector_add(d_a.as_ptr(), d_b.as_ptr(), d_c.as_ptr(), n as i32);}

  d_c.copy_to_host(&mut c);

  for i in 0..n
  {
    assert_eq!(c[i], a[i] + b[i]);
  }

}