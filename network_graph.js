// Load the data from JSON file
// d3.json("./result/json_for_d3_graph_dummydata.json").then(function (graph) {
d3.json("./result/json_for_d3_graph_realdata.json").then(function (graph) {
    var svg = d3.select("svg"),
        width = +svg.attr("width"),
        height = +svg.attr("height");
    console.log(width, height)

    // Create a map of nodes by their id
    var nodesMap = new Map(graph.nodes.map(node => [node.id, node]));
    console.log(nodesMap);

    var color = d3.scaleOrdinal(d3.schemeCategory10);

    // Create a container for zooming and panning
    var container = svg.append("g");

    // Define arrow markers for graph links
    svg.append("defs").append("marker")
        .attr("id", "arrowhead")
        .attr("viewBox", "-0 -5 10 10")
        .attr("refX", 13)
        .attr("refY", 0)
        .attr("orient", "auto")
        .attr("markerWidth", 10)  // Reduced from 20 to 10
        .attr("markerHeight", 10)  // Reduced from 20 to 10
        .attr("xoverflow", "visible")
        .append("svg:path")
        .attr("d", "M 0,-3 L 6 ,0 L 0,3")  // Adjusted path for smaller arrowhead
        .attr("fill", "#999")
        .style("stroke", "none");

    var zoom = d3.zoom()
        .scaleExtent([0.1, 10])
        .on("zoom", function (event) {
            container.attr("transform", event.transform);
        });

    svg
        .call(zoom)
        .call(zoom.transform, d3.zoomIdentity.translate(width / 2, height / 2).scale(0.1));


    // Tooltip for displaying user_id
    var tooltip = d3.select("body").append("div")
        .attr("class", "tooltip")
        .style("position", "absolute")
        .style("visibility", "hidden")
        .style("background", "#f9f9f9")
        .style("border", "1px solid #ccc")
        .style("padding", "5px")
        .style("border-radius", "3px")
        .style("font-size", "12px");


    var node = container.append("g")
        .attr("class", "nodes")
        .selectAll("circle")
        .data(graph.nodes)
        .enter().append("circle")
        .attr("class", "node")
        .attr("r", 10)
        .attr("fill", d => color(d.group))
        .attr("cx", d => d.pca_embeddings_x)
        .attr("cy", d => d.pca_embeddings_y)
        .call(d3.drag()
            .on("start", dragstarted)
            .on("drag", dragged)
            .on("end", dragended))
        .attr("data-html", "true")
        .on("mouseover", function (event, d) {
            tooltip.style("visibility", "visible")
                .html(
                    `user_id: ${d.id}<br />`
                    + `inbound_trx_count: ${d.inbound_trx_count}<br />`
                    + `outbound_trx_count: ${d.outbound_trx_count}<br />`
                    + `risk_score: ${d.risk_score}<br />`
                    + `classified_type: ${d.classified_type}<br />`
                    + `betweenness_centrality: ${d.betweenness_centrality}<br />`
                    + `syndicate_score: ${d.syndicate_score}<br />`
                );
        })
        .on("mousemove", function (event) {
            tooltip.style("top", (event.pageY - 10) + "px")
                .style("left", (event.pageX + 10) + "px");
        })
        .on("mouseout", function () {
            tooltip.style("visibility", "hidden");
        });


    var link = container.append("g")
        .attr("class", "links")
        .selectAll("line")
        .data(graph.links)
        .enter().append("line")
        .attr("class", "link")
        .attr("x1", d => graph.nodes.find(node => node.id === d.source).pca_embeddings_x)
        .attr("y1", d => graph.nodes.find(node => node.id === d.source).pca_embeddings_y)
        .attr("x2", d => graph.nodes.find(node => node.id === d.target).pca_embeddings_x)
        .attr("y2", d => graph.nodes.find(node => node.id === d.target).pca_embeddings_y)
        .attr("stroke-width", d => Math.sqrt(d.value))
        .attr("stroke", "#999")
        .attr("stroke-opacity", 0.6)
        .attr("marker-end", "url(#arrowhead)");

    // Update node positions based on PCA embeddings
    node
        .attr("cx", d => d.pca_embeddings_x)
        .attr("cy", d => d.pca_embeddings_y);
    // node.append("title")
    //     .text(d => d.id);

    // Let's list the force we wanna apply on the network
    var simulation = d3.forceSimulation(graph.nodes)                 // Force algorithm is applied to graph.nodes
        .force("link", d3.forceLink()                               // This force provides links between nodes
            .id(function (d) { return d.id; })                     // This provide  the id of a node
            .links(graph.links)                                    // and this the list of links
        )
        .force("charge", d3.forceManyBody().strength(-400))         // This adds repulsion between nodes. Play with the -400 for the repulsion strength
        .force("center", d3.forceCenter(width / 2, height / 2))     // This force attracts nodes to the center of the svg area
        .on("tick", ticked)
        .on("end", () => {
            link
                .attr("x1", d => d.source.x)
                .attr("y1", d => d.source.y)
                .attr("x2", d => d.target.x)
                .attr("y2", d => d.target.y);

            node
                .attr("cx", d => d.x)
                .attr("cy", d => d.y);

            alert("Simulation finished");
        });

    var tickCount = -1;
    function ticked() {
        if (tickCount == -1) {
            node.attr("r", d => Math.max(2, Math.min(30, Math.log(d.inbound_trx_count + d.outbound_trx_count + 1) / Math.log(15) * 3)))
        }

        tickCount = (tickCount + 1) % 5;
        if (tickCount != 0) {
            return
        }

        link
            .attr("x1", d => d.source.x)
            .attr("y1", d => d.source.y)
            .attr("x2", d => d.target.x)
            .attr("y2", d => d.target.y);

        node
            .attr("cx", d => d.x)
            .attr("cy", d => d.y);
    }

    simulation.on("tick", () => {
        link
            .attr("x1", d => d.source.x)
            .attr("y1", d => d.source.y)
            .attr("x2", d => d.target.x)
            .attr("y2", d => d.target.y);

        node
            .attr("cx", d => d.x)
            .attr("cy", d => d.y);
    });

    function dragstarted(event, d) {
        if (!event.active) simulation.alphaTarget(0.3).restart();
        d.fx = d.x;
        d.fy = d.y;
    }

    function dragged(event, d) {
        d.fx = event.x;
        d.fy = event.y;
    }

    function dragended(event, d) {
        if (!event.active) simulation.alphaTarget(0);
        d.fx = null;
        d.fy = null;
    }

    // Add zoom functionality
    function zoomed(event) {
        container.attr("transform", event.transform);
    }
    // Function to zoom to a specific node and change its color
    function zoomToNode(userId) {
        const targetNode = graph.nodes.find(d => d.id == userId);

        if (!targetNode) {
            alert(`User ID ${userId} not found!`);
            return;
        }

        const transform = d3.zoomIdentity
            .translate(width / 2, height / 2)
            .scale(1.5)
            .translate(-targetNode.x, -targetNode.y);

        svg.transition()
            .duration(750)
            .call(zoom.transform, transform);

        // Change the color of the target node
        d3.selectAll(".node")
            .attr("fill", d => d.id === userId ? "red" : color(d.group)); // Change color to red for the searched node
    }

    // Search button event listener
    d3.select("#search-btn").on("click", () => {
        const userId = document.getElementById("search-input").value;
        zoomToNode(userId);
    });
});