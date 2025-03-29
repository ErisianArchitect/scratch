use std::{borrow::Borrow, collections::{HashMap, HashSet, VecDeque}, hash::Hash};

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct NodeId {
    index: u32,
}

impl NodeId {
    pub const ZERO: NodeId = NodeId::new(0);
    pub const MIN: NodeId = NodeId::new(u32::MIN);
    pub const MAX: NodeId = NodeId::new(u32::MAX);

    pub const fn new(index: u32) -> Self {
        Self { index }
    }

    pub fn fetch_add(&mut self, value: u32) -> Self {
        let old = *self;
        self.index += value;
        old
    }
}

#[derive(Debug, Default, Clone)]
pub struct Dependency {
    dependencies: HashSet<NodeId>,
    dependents: HashSet<NodeId>,
}

#[derive(Debug, Default, Clone)]
pub struct DependencyGraph {
    unique_id: NodeId,
    nodes: HashMap<NodeId, Dependency>,
    roots: HashSet<NodeId>,
}

impl DependencyGraph {
    pub fn new() -> Self {
        Self {
            unique_id: NodeId::ZERO,
            nodes: HashMap::new(),
            roots: HashSet::new(),
        }
    }

    pub fn create_node(&mut self, dependencies: &[NodeId]) -> NodeId {
        let id = self.unique_id.fetch_add(1);
        self.nodes.insert(id, Dependency::default());
        if dependencies.is_empty() {
            self.roots.insert(id);
        } else {
            self.add_dependencies(id, dependencies);
        }
        id
    }

    pub fn remove_node(&mut self, node: NodeId) {
        let mut dep_list = Vec::new();
        let mut out_list = Vec::new();
        if let Some(node_mut) = self.nodes.remove(&node) {
            dep_list.extend(node_mut.dependencies);
            out_list.extend(node_mut.dependents);
        }
        for dep_node in dep_list {
            let Some(dep_mut) = self.nodes.get_mut(&dep_node) else {
                continue;
            };
            dep_mut.dependents.remove(&node);
        }
        for out_node in out_list {
            let Some(node_mut) = self.nodes.get_mut(&out_node) else {
                continue;
            };
            node_mut.dependencies.remove(&node);
        }
        self.roots.remove(&node);
    }

    pub fn add_dependency(&mut self, node: NodeId, dependency: NodeId) {
        if let Some(node_mut) = self.nodes.get_mut(&node) {
            node_mut.dependencies.insert(dependency);
        } else {
            panic!("Node not found.");
        }
        if let Some(dependency_mut) = self.nodes.get_mut(&dependency) {
            dependency_mut.dependents.insert(node);
        } else {
            panic!("Dependency not found.")
        }
        self.roots.remove(&node);
    }

    pub fn add_dependencies(&mut self, node: NodeId, dependencies: &[NodeId]) {
        if dependencies.is_empty() {
            return;
        }
        if let Some(node_mut) = self.nodes.get_mut(&node) {
            node_mut.dependencies.extend(dependencies);
        } else {
            panic!("Node not found.");
        }
        for dep in dependencies {
            if let Some(dep_mut) = self.nodes.get_mut(dep) {
                dep_mut.dependents.insert(node);
            } else {
                panic!("Dependency not found.");
            }
        }
        self.roots.remove(&node);
    }

    pub fn remove_dependency(&mut self, node: NodeId, dependency: NodeId) {
        if let Some(node_mut) = self.nodes.get_mut(&node) {
            node_mut.dependencies.remove(&dependency);
            if node_mut.dependencies.is_empty() {
                self.roots.insert(node);
            }
        }
        if let Some(dependency) = self.nodes.get_mut(&dependency) {
            dependency.dependents.remove(&node);
        }
    }

    pub fn remove_all_dependencies(&mut self, node: NodeId) {
        let dependencies = if let Some(node_mut) = self.nodes.get_mut(&node) {
            node_mut.dependencies.drain().collect::<Vec<_>>()
        } else {
            return;
        };
        for dep in dependencies {
            let Some(dep_mut) = self.nodes.get_mut(&dep) else {
                // TODO: Or maybe panic?
                continue;
            };
            dep_mut.dependents.remove(&node);
        }
        self.roots.insert(node);
    }

    pub fn remove_dependent(&mut self, node: NodeId, dependent: NodeId) {
        if let Some(node_mut) = self.nodes.get_mut(&node) {
            node_mut.dependents.remove(&dependent);
        }
        if let Some(dep_mut) = self.nodes.get_mut(&dependent) {
            dep_mut.dependencies.remove(&node);
            if dep_mut.dependencies.is_empty() {
                self.roots.insert(dependent);
            }
        }
    }

    pub fn remove_all_dependents(&mut self, dependency: NodeId) {
        let dependents = if let Some(dep_mut) = self.nodes.get_mut(&dependency) {
            dep_mut.dependents.drain().collect::<Vec<_>>()
        } else {
            return;
        };
        for node in dependents {
            let Some(node_mut) = self.nodes.get_mut(&node) else {
                continue;
            };
            node_mut.dependencies.remove(&dependency);
            if node_mut.dependencies.is_empty() {
                self.roots.insert(node);
            }
        }
    }

    /// Insert `node` before `target`.
    pub fn insert_before(&mut self, node: NodeId, target: NodeId) {
        self.add_dependency(target, node);
    }

    /// Insert `node` after `target`.
    pub fn insert_after(&mut self, node: NodeId, target: NodeId) {
        self.add_dependency(node, target);
    }

    pub fn topological_sort_into(&self, out: &mut Vec<NodeId>) {
        out.reserve(self.nodes.len());
        let mut in_degrees = HashMap::with_capacity(self.nodes.len());
        let mut queue = VecDeque::from_iter(self.roots.iter().cloned());

        // let mut o = 0usize;

        for (&node, dep) in self.nodes.iter() {
            let degree = dep.dependencies.len() as u32;
            in_degrees.insert(node, degree);
        }

        while let Some(node) = queue.pop_front() {
            out.push(node);
            for dependent in self.nodes[&node].dependents.iter() {
                let Some(degree) = in_degrees.get_mut(dependent) else {
                    panic!("in_degree not found.");
                };
                *degree -= 1;
                if *degree == 0 {
                    queue.push_back(*dependent);
                }
            }
        }
    }

    pub fn topological_sort(&self) -> Vec<NodeId> {
        let mut order = Vec::new();
        self.topological_sort_into(&mut order);
        order
    }
}

pub struct DependencyGraphMap<K: std::hash::Hash + Eq> {
    graph: DependencyGraph,
    node_keys: bimap::BiHashMap<K, NodeId>,
}

impl<K: std::hash::Hash + Eq> DependencyGraphMap<K> {
    pub fn new() -> Self {
        Self {
            graph: DependencyGraph::new(),
            node_keys: bimap::BiHashMap::new(),
        }
    }

    pub fn create_node(&mut self, key: K) -> NodeId {
        let node = self.graph.create_node(&[]);
        self.node_keys.insert(key.into(), node);
        node
    }

    pub fn create_node_with<D: Borrow<K>>(&mut self, key: K, dependencies: &[D]) -> NodeId {
        let node = self.graph.create_node(&[]);
        for dep in dependencies {
            let Some(&dep_node) = self.node_keys.get_by_left(dep.borrow()) else {
                panic!("Dependency not found.");
            };
            self.graph.add_dependency(node, dep_node);
        }
        self.node_keys.insert(key.into(), node);
        node
    }

    pub fn remove_node<R: Borrow<K>>(&mut self, key: R) {
        let Some(&node_id) = self.node_keys.get_by_left(key.borrow()) else {
            panic!("Node not found.");
        };
        self.graph.remove_node(node_id);
    }

    pub fn add_dependency<NQ, DQ>(&mut self, node: NQ, dependency: DQ)
    where
        NQ: Hash + Eq,
        K: Borrow<NQ>,
        DQ: Hash + Eq,
        K: Borrow<DQ>,
    {
        let node_id = self.node_keys.get_by_left(node.borrow()).cloned().expect("Node not found.");
        let dep_id = self.node_keys.get_by_left(dependency.borrow()).cloned().expect("Dependency not found.");
        self.graph.add_dependency(node_id, dep_id);
    }

    pub fn add_dependencies<NQ, DQ>(&mut self, node: NQ, dependencies: &[DQ])
    where
        NQ: Hash + Eq,
        K: Borrow<NQ>,
        DQ: Hash + Eq,
        K: Borrow<DQ>,
    {
        let node_id = self.node_keys.get_by_left(node.borrow()).cloned().expect("Node not found.");
        for dep in dependencies {
            let Some(&dep_node) = self.node_keys.get_by_left(dep.borrow()) else {
                panic!("Dependency not found.");
            };
            self.graph.add_dependency(node_id, dep_node);
        }
    }

    pub fn remove_dependency<N: Borrow<K>, D: Borrow<K>>(&mut self, node: N, dependency: D) {
        let node_id = self.node_keys.get_by_left(node.borrow()).cloned().expect("Node not found.");
        let dep_id = self.node_keys.get_by_left(dependency.borrow()).cloned().expect("Dependency not found.");
        self.graph.remove_dependency(node_id, dep_id);
    }

    pub fn remove_all_dependencies<N: Borrow<K>>(&mut self, node: N) {
        let node_id = self.node_keys.get_by_left(node.borrow()).cloned().expect("Node not found.");
        self.graph.remove_all_dependencies(node_id);
    }

    pub fn remove_dependent<N: Borrow<K>, D: Borrow<K>>(&mut self, node: N, depednent: D) {
        let node_id = self.node_keys.get_by_left(node.borrow()).cloned().expect("Node not found.");
        let dep_id = self.node_keys.get_by_left(depednent.borrow()).cloned().expect("Dependent not found.");
        self.graph.remove_dependent(node_id, dep_id);
    }

    pub fn remove_all_dependents<D: Borrow<K>>(&mut self, dependency: D) {
        let dep_id = self.node_keys.get_by_left(dependency.borrow()).cloned().expect("Node not found.");
        self.graph.remove_all_dependents(dep_id);
    }

    /// Insert `node` before `target`.
    pub fn insert_before<N: Borrow<K>, T: Borrow<K>>(&mut self, node: N, target: T) {
        let node_id = self.node_keys.get_by_left(node.borrow()).cloned().expect("Node not found.");
        let target_id = self.node_keys.get_by_left(target.borrow()).cloned().expect("Dependency not found.");
        self.graph.insert_before(node_id, target_id);
    }

    /// Insert `node` after `target`.
    pub fn insert_after<N: Borrow<K>, T: Borrow<K>>(&mut self, node: N, target: T) {
        let node_id = self.node_keys.get_by_left(node.borrow()).cloned().expect("Node not found.");
        let target_id = self.node_keys.get_by_left(target.borrow()).cloned().expect("Dependency not found.");
        self.graph.insert_after(node_id, target_id);
    }

    pub fn topological_sort(&mut self) -> Vec<&K> {
        self.graph.topological_sort().into_iter().map(|node| {
            self.node_keys.get_by_right(&node).expect("Failed to get node key.")
        }).collect()
    }
}

pub struct SwapBuffer<T> {
    pub back: Vec<T>,
    pub front: Vec<T>,
}

impl<T> SwapBuffer<T> {
    pub fn new() -> Self {
        Self {
            back: vec![],
            front: vec![],
        }
    }

    /// Swap the back and front buffers, then clear the new front buffer.
    pub fn swap(&mut self) {
        std::mem::swap(&mut self.back, &mut self.front);
        self.front.clear();
    }
}

impl<T> std::ops::Deref for SwapBuffer<T> {
    type Target = Vec<T>;

    fn deref(&self) -> &Self::Target {
        &self.front
    }
}

impl<T> std::ops::DerefMut for SwapBuffer<T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.front
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn swap_buffer_test() {
        let mut buffer = SwapBuffer::new();
        buffer.push(1);
        buffer.push(2);
        buffer.push(3);
        buffer.swap();
        buffer.push(4);
        buffer.push(5);
        for n in buffer.back.iter() {
            println!("{n}");
        }
        println!("********************************");
        for n in buffer.front.iter() {
            println!("{n}");
        }
        println!("********************************");
        println!("Swap");
        buffer.swap();
        println!("********************************");
        for n in buffer.back.iter() {
            println!("{n}");
        }
        println!("********************************");
        for n in buffer.front.iter() {
            println!("{n}");
        }
    }
    
    #[test]
    fn graph_map_test() {
        let mut graph = DependencyGraphMap::new();
        // graph.create_node("root");
        // graph.create_node_with("level0_a", &["root"]);
        // graph.create_node("foo");
        // graph.insert_before("foo", "root");
        // graph.create_node("bar");
        // graph.insert_before("bar", "foo");
        // graph.create_node_with("test", &["root", "level0_a"]);

        graph.create_node("fred");
        graph.create_node("dog");
        graph.create_node_with("bob", &["fred"]);
        graph.create_node_with("dylan", &["fred"]);
        graph.create_node_with("marley", &["fred"]);
        graph.remove_all_dependents("fred");
        graph.add_dependencies("fred", &["bob", "dylan", "marley"]);
        graph.add_dependency("bob", "marley");
        graph.add_dependency("marley", "dylan");
        graph.remove_all_dependencies("fred");
        // graph.add_dependency("fred", "fred");
        graph.insert_before("fred", "dylan");
        let top_sort = graph.topological_sort();
        println!("Capacity: {}\nLength: {}", top_sort.capacity(), top_sort.len());
        for &key in top_sort {
            println!("{}", key);
        }

    }
}

#[cfg(test)]
mod testing_sandbox {
    // TODO: Remove this sandbox when it is no longer in use.
    use super::*;
    #[test]
    fn sandbox() {
        let mut dag = DependencyGraph::new();
        let mut node_keys = HashMap::<&str, NodeId>::new();
        let mut node_names = HashMap::<NodeId, &str>::new();
        macro_rules! create_node {
            ($name:ident$([$($dep:ident),*])?) => {
                let $name = dag.create_node(&[$($($dep),*)?]);
                let key = stringify!($name);
                node_names.insert($name, key);
                node_keys.insert(key, $name);
            };
        }
        create_node!(root);
        create_node!(level_0_a[root]);
        create_node!(level_0_b[root]);
        create_node!(level_0_c[root]);
        create_node!(level_0_d[root]);
        create_node!(level_1_a[level_0_a, level_0_b]);
        create_node!(level_1_b[level_0_b, level_0_c]);
        create_node!(level_1_c[level_0_c, level_0_d]);
        create_node!(level_2_a[level_1_a, level_1_b]);
        create_node!(level_2_b[level_1_b, level_1_c]);
        create_node!(level_3_a[level_2_a, level_2_b]);
        create_node!(outcast);
        create_node!(level_4_a[level_3_a, level_0_a, level_0_b, outcast]);

        create_node!(foo);
        create_node!(bar);
        create_node!(baz);
        // create_node!(baz);

        dag.insert_after(bar, foo);

        dag.insert_before(foo, level_2_b);
        dag.insert_after(foo, level_2_a);

        dag.insert_before(baz, foo);
        dag.insert_after(baz, level_1_a);

        dag.add_dependency(outcast, level_0_d);
        dag.remove_node(outcast);
        // dag.remove_dependency(level_4_a, level_3_a);
        let top_sort = dag.topological_sort();
        for node in top_sort.into_iter() {
            println!("Node: {}", node_names[&node]);
        }
        println!("Graph: {:?}", dag);
    }
}