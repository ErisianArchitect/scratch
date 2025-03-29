macro_rules! prototype { ($($_:tt)*) => {} }

prototype!(
    fn node0_a_activate() {

    }

    create_graph!(graph_name => {
        root => {},
        root2 => {},
        node0_a => {
            dependencies: root,
            on_activate: {
                node0_a_activate();
            }
        },
        node0_b => root,
        node0_c => root && root2,
        node1_a => node0_a || node0_b || node0_c,
    });
    // Creates graph_name macro
    let node_name = graph_name!().create_node("node_name");

);
