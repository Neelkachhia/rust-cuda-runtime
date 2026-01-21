pub struct DeviceBuffer<T>
{
    ptr: *mut T, //raw GPU pointer (will come from CUDA)
    len: usize, // number of elements
}
extern "C"
{
    pub fn launch_vector_add(a: *const f32, b: *const f32, c: *mut f32, n: i32);
}

impl<T> DeviceBuffer<T>
{
    pub fn len(&self) -> usize
    {
        self.len
    }    
}