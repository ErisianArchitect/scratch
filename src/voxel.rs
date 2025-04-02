
// Contree implementation for voxel world



pub enum Node {
    Node(u32),
    Id(u32),
}

#[test]
fn node_size() {
    println!("{}", std::mem::size_of::<Node>());
}

pub struct Level0 {
    
}