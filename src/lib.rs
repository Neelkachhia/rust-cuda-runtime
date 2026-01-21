pub struct DeviceBuffer<T>
{
    ptr: *mut T, //raw GPU pointer (will come from CUDA)
    len: usize, // number of elements
}

impl<T> DeviceBuffer<T>
{
    pub fn len(&self) -> usize
    {
        self.len
    }    
}