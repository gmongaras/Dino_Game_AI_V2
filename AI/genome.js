//--------------------------------------------Helpers----------------------------------------------------

// Given a vector of size [n, 1], it returns a copied vector
function copyVector(v) {
	vCopy = [];
	for (let item of v) {
		vCopy.push(item);
	}
	return vCopy;
}


// Given a matrix of size [n, n], it returns a copied matrix
function copyMatrix(m) {
	mCopy = [];
	for (let v of m) {
		vCopy = [];
		for (let item of v) {
			vCopy.push(item);
		}
		mCopy.push(vCopy);
	}
	return mCopy;
}


// Given two variable which can be either a number or a node, compare them.
// The function returns true if both values are the same
function compareNodes(n1, n2) {
	// If both nodes are input nodes (numbers)
	if (isNaN(Number(n1)) == false && isNaN(Number(n2)) == false) {
		// If both values are equal, return true
		if (n1 == n2) {
			return true;
		}
		return false;
	}
	
	// If both nodes are not input nodes (node objects)
	if (isNaN(Number(n1)) == true && isNaN(Number(n2)) == true) {
		// Nodes are equal if the bias, weights, activation function, and previous nodes are equal.
		
		// First, compare the biases
		if (n1.bias != n2.bias) {
			return false;
		}
		
		// Now, compare the activation function
		if (n1.actiation != n2.actiation) {
			return false;
		}
		
		// Next, compare the weights
		// If the weight lengths are not equal, return false
		if (n1.prevNodesWeights.length != n2.prevNodesWeights.length) {
			return false;
		}
		// Iterate over each weight. If the weights are not the same, return false
		for (let i = 0; i < n1.prevNodesWeights.length; i++) {
			// If the ith weights are not the same, return false
			if (n1.prevNodesWeights[i] != n2.prevNodesWeights[i]) {
				return false;
			}
		}
		
		// Finally, compare the previous nodes
		// If the previous node array lengths are not equal, return false
		if (n1.prevNodes.length != n2.prevNodes.length) {
			return false;
		}
		// Iterate over each previous node. If the values are not the same, then
		// return false.
		for (let i = 0; i < n1.prevNodes.length; i++) {
			// If the nodes are not the same, return false
			if (compareNodes(n1.prevNodes[i], n2.prevNodes[i]) == false) {
				return false;
			}
		}
		
		// If all tests pass, return true
		return true;
	}
	
	// If the node types are different, return false
	return false;
}


// Given a vector of size [1, n] and a vector of size [1, m], compare them. If the vectors
// are equal, return true, otherwise return false.
function compareVectors(v1, v2) {
	// If the vectors have difference length, return false
	if (v1.length != v2.length) {
		return false;
	}
	
	// Iterate over every item in the vectors and compare them
	for (let i = 0; i < v1.length; i++) {
		// If the items are different, return false
		if (compareNodes(v1[i], v2[i]) == false) {
			return false;
		}
	}
	
	// if the vectors have the same items, return true
	return true;
}


// Given a matrix of size [a, b] and a matrix of size [n, m], compare them. If the matricies
// are equal, return true, otherwise return false.
function compareMatricies(m1, m2) {
	// If the matricies have difference length, return false
	if (m1.length != m2.length) {
		return false;
	}
	
	// Iterate over every vector in the matrix and compare them
	for (let i = 0; i < v1.length; i++) {
		// If the items are different, return false
		if (compareVectors(m1[i], m2[i]) == false) {
			return false;
		}
	}
	
	// if the matricies have the same items, return true
	return true;
}


// Given two vectors of size [1, n] and [1, m], return a new vector that includes
// items from both arrays without duplicates.
function concatVectorsDistinct(a1, a2) {
	// New array that stores a combination of both arrays. It is initialized to
	// the vector "a1"
	let newArray = copyVector(a1);
	
	// For every item in vector "a2".
	for (let i = 0; i < a2.length; i++) {
		// If the item is a node
		if (isNaN(Number(a2[i]))) {
			// Iterate over all items in newArray.
			let newArrayCopy = copyVector(newArray);
			for (let j = 0; j < newArrayCopy.length; j++) {
				// If the items are different, add them to the array.
				if (compareNodes(newArray[i], a2[i]) == false) {
					newArray.push(a2[i]);
				}
			}
		}
		
		// If the item is not a node, then add the item if it is not already in the new array.
		if (newArray.includes(a2[i]) == false) {
			newArray.push(a2[i]);
		}
	}
	
	// Return the new array
	return newArray;
}


// Given two matricies of size [a, b] and [n, m], return a new matrix that includes
// items from both arrays without duplicates.
function concatMatriciesDistinct(a1, a2) {
	// New array that stores a combination of both arrays. It is initialized to
	// the matrix "a1"
	let newArray = copyMatrix(a1);
	
	// For every vector in matrix "a2".
	for (let i = 0; i < a2.length; i++) {
		// Iterate over every value in newArray. If any of the items in
		// newArray are equal to the current vector, don't add the vector
		// to the array.
		let inArray = false;
		let temp = copyMatrix(newArray);
		for (let j = 0; j < temp.length; j++) {
			// Compare the jth vector in newArray to the ith vector in the a2 array.
			// If the arrays are the same, then set inArray to true since the
			// vector is alreay present in the array.
			if (compareVectors(temp[j], a2[i]) == true) {
				inArray = true;
				break;
			}
		}
		
		// If the current vector is not in "newArray", add it to "newArray"
		if (inArray == false) {
			newArray.push(a2[i]);
		}
	}
	
	// Return the new array
	return newArray;
}









//-------------------------------------------Activation Functions----------------------------------------
// Puts a t value through the sigmoid function and returns the value
function sigmoid(t) {
    return 1/(1+Math.pow(Math.E, -t));
}



// Returns the output when x is sent through the ReLU function
function ReLU(x) {
	if (x >= 0) {
		return x;
	}
	else {
		return 0;
	}
}




// Returns the output when x is sent through the tanh function.
function tanh(x) {
	(Math.pow(Math.E, x) - Math.pow(Math.E, -x))/(Math.pow(Math.E, x) + Math.pow(Math.E, -x))
}




// Calculates the softax given an array of values
// Link to softmax function: https://deepai.org/machine-learning-glossary-and-terms/softmax-layer
function softmax(inputs) {
	// Calculate the top half of the funciton.
	expVals = []; // The exponential values for each item
	
	for (let i = 0; i < inputs.length; i++) {
		expVals.push(Math.pow(Math.E, inputs[i]));
	}
	
	
	// Calculate the bottom half of the function
	summation = 0; // The summation of all exponential values
	
	for (let i = 0; i < inputs.length; i++) {
		summation += expVals[i];
	}
	
	
	// Calculate the final value for each input. Each input is it's exponential
	// value divided by the summation
	outputs = []; // The final value of each input
	
	for (let i = 0; i < inputs.length; i++) {
		outputs.push(expVals[i]/summation);
	}
	
	return outputs;
}









//-------------------------------------------Node----------------------------------------
class Node {
	// prevNodes - An array of nodes preceeding this node in the network .
	//			   This variable is a null array if there are no preceeding nodes.
	//			   If the previous node is an input, then a number will be stored instead of
	//			   a node. The number corresponds to the input value. A value
	//			   of 0 means the input is the first input in the inputs array.
	// prevNodesWeights - If there are any preceeding nodes, store the weights between
	// 					  the preceeding and current node.
	// activation - The activation function fpr this node.
	constructor(prevNodes, prevNodesWeights, activation, posInputs) {
		// If prevNodes was supplied, store it and it's weight.
		if (prevNodes != null) {
			// Convert the previous nodes and weights to an array if they aren't already one.
			if (prevNodes[0] == undefined) {
				prevNodes = [prevNodes];
			}
			if (prevNodesWeights[0] == undefined) {
				prevNodesWeights = [prevNodesWeights];
			}
			
			
			// Initialize the variables
			this.prevNodes = prevNodes;
			this.prevNodesWeights = prevNodesWeights;
			this.bias = 0;
			this.activation = activation;
			this.posInputs = posInputs;
			this.depth = 0;
		}
		// If nextNodes was not supplied, store null for both the next node and it's weight.
		else {
			// Initialize the variables
			this.prevNodes = [];
			this.prevNodesWeights = [];
			this.bias = 0;
			this.activation = activation;
			this.posInputs = posInputs;
			this.depth = 0;
		}
	}
	
	
	// Given the neural network inputs, the node returns an output
	forward(inputs) {
		// If prevNodes is not a null array, iterate over every previous node
		// and recursively call the forward method in that node
		if (this.prevNodes != []) {
			// Stored the output
			let z = 0;
			
			// Iterate over every previous node
			for (let i = 0; i < this.prevNodes.length; i++) {
				// If the previous node is a node, then recursively call the previous
				// node to get the input from that node.
				if (isNaN(Number(this.prevNodes[i]))) {
					z += this.prevNodesWeights[i] * this.prevNodes[i].forward(inputs);
				}
				
				
				// If the previous node is a numerical value, then the previous
				// node is an input. The number corresponds to an input where the
				// first input in the inputs array is input 0.
				else {
					z += this.prevNodesWeights[i] * inputs[this.prevNodes[i]];
				}
			}
			
			// Add the bias to the z value.
			z += this.bias;
			
			
			// Return the output of the activation function with the z parameter as input.
			if (this.activation == "sigmoid") {
				return sigmoid(z);
			}
			else if (this.activation == "ReLU") {
				return ReLU(z);
			}
			else if (this.activation == "sigmoid") {
				return sigmoid(z);
			}
			
			// If the activation function is not a given activation function, return the z value.
			return z;
		}
		
		// If prevNodes is a null array, then there are no previous nodes, so return 0
		return 0;
	}
	
	
	
	// Returns a copy of this node
	getCopy() {
		// Stores a copy of all previous nodes in this network
		let copiedNodes = [];
		// Stores a copy of all previous node weights
		let copiedWeights = [];
		
		// Iterate over every previous node
		for (let i = 0; i < this.prevNodes.length; i++) {
			// If the previous node is a node object, then add a copy
			// of the node object to the array.
			if (isNaN(Number(this.prevNodes[i]))) {
				copiedNodes.push(this.prevNodes[i].getCopy());
			}
			
			// If the previous node is an input node, then add the
			// index of the input node to the copiedNodes array.
			else {
				copiedNodes.push(this.prevNodes[i]);
			}
			
			// Copy the current nodes weight to the copiedWeights array.
			copiedWeights.push(this.prevNodesWeights[i]);
		}
		
		
		
		// Return a copy of the node
		return new Node(copiedNodes, copiedWeights, this.activation);
	}
	
	
	
	
	// Mutates the weights and biases in this node,
	mutateWeightsAndBiases() {
		const weightRate = 0.13; // The rate at which the weights are changed
		const weightChangeAmount = 1 // The range which the weights can be changed
		const biasRate = 0.1; // The rate at which the biases are changed
		const biasChangeAmount = 0.1 // The range which the biases can be changed
		
		
		// Iterate over every previous output
		for (let i = 0; i < this.prevNodes.length; i++) {
			// If the previous node is not an input node, then call the
			// mutateWeightsAndBiases function for that node.
			if (isNaN(Number(this.prevNodes[i]))) {
				this.prevNodes[i].mutateWeightsAndBiases()
			}
			
			// With a chance of "weightRate" change the weight by a value between -1 and 1.
			let v = Math.random(1)
			if (v < weightRate) {
				this.prevNodesWeights[i] += parseFloat((Math.random() * weightChangeAmount) * (Math.round(Math.random()) ? 1 : -1));
			}
		}
		
		// With a chance of "biasRate" change the weight by a value between -1 and 1.
		if (Math.random(1) < biasRate) {
			this.bias += parseFloat((Math.random() * biasChangeAmount) * (Math.round(Math.random()) ? 1 : -1));
		}
		
		;;
	}
	
	
	
	
	// Returns the location where new nodes can be placed. Basically, it returns
	// an array filled with two valued vectors which denote the location the node can
	// be placed into:
	// [right_node, prevNodeIndex]
	// right_node: The node that proceeds the new node that will be added.
	// prevNodeIndex: index in the "right_node"'s previous nodes to place the
	//				  new node between. So, the new node will be placed between right_node
	//				  and right_node.prevNodes[prevNodeIndex].
	getPossibleNodeLocations() {
		// Array that holds locations that new nodes can be added to.
		let posNodes = [];
		
		// Iterate over every previous node.
		for (let i = 0; i < this.prevNodes.length; i++) {
			// Add the location which the new node can be placed. This is bascially
			// just the location of a weight as when a node is added, it is added
			// and intersects the weight.
			posNodes.push([this, i]);
			
			// If the previous node is not an input node, then get the
			// possible node locations of the previous node and add it to
			// the posNodes array.
			if (isNaN(Number(this.prevNodes[i]))) {
				posNodes.concat(this.prevNodes[i].getPossibleNodeLocations());
			}
		}
		
		// Return the possible node locations
		return posNodes;
	}
	
	
	
	// Adds a node that interscts a given connection.
	// newNode - new node to add in the network
	// prevNodeIndex - index of previous node. The new node will be placed between the
	//				   previous node and this node. So, if prevNodeIndex is 0, then
	//				   the new node will be placed between prevNode[0] and this node
	// posInputs - Array of input layer
	addNode(newNode, prevNodeIndex, posInputs) {
		// When a node is added, it creates a connection between the previous node (which may
		// be a node or an input), the new node, and this node (as in this current object). The
		// connection from the previous node to the new node has a weight of 1 and the connection
		// between the new node and this node is equal to the old connection between the
		// previous node and this node.
		
		// Add a new node to the newNode. The node added will be the previousNode at index prevNodeIndex.
		newNode.prevNodes.push(this.prevNodes[prevNodeIndex]);
		// Add a new weight to the newNode which is the connection between the previous node at
		// index prevNodeIndex and the newNode. The weight is 1.
		newNode.prevNodesWeights.push(1);
		
		// Set the previous node of this current node at index prevNodeIndex to the new node
		this.prevNodes[prevNodeIndex] = newNode;
		// The weight between the newNode and this node is already correct and does not need to be changed.
		
		// Store the input layer in the new node
		newNode.posInputs = posInputs;
	}
	
	
	
	// Returns all node objects which preceed this node.
	getPrevNodes() {
		// Array that holds all previous nodes in the array.
		let prevNodes = [];
		
		// Iterate over all previous nodes connected to this node
		for (let i = 0; i < this.prevNodes.length; i++) {
			// If the previous node is a node object, then get the
			// previous nodes from that node. Store the previous nodes
			// in the "prevNodes" array we are building if the items are distinct.
			if (isNaN(Number(this.prevNodes[i]))) {
				prevNodes = concatDistinct(prevNodes, this.prevNodes[i].getPrevNodes());
			}
			
			// If the previous node is an input node, then add it to the
			// "prevNodes" array if it is not already in there.
			if (prevNodes.includes(this.prevNodes[i]) == false) {
				prevNodes.push(this.prevNodes[i]);
			}
		}
		
		// Return the nodes that preceed this node.
		console.log(prevNodes);
		return prevNodes;
	}
	
	
	
	// Returns an array filled with location where weights can be placed. Each weight
	// location is a two part vector:
	// [right_node, prevNode]
	// right_node: The node to attach to the previous node with a weight.
	// prevNode: The node to connect to the "right_node". When the weight is added, right_node
	// 			 will add "prevNode" to it's stored prevNodes. The weight will be
	//			 a random value. Note that this is the actual node, not the location.
	getPossibleWeightLocations() {
		// Stores all possible node connections
		let posNodeCon = [];
		// Stores all previous possible weight locations
		let prevPosWeights = [];
		
		// First hidden node flag. If the flag is true, then the node does not have any preceeding
		// nodes besides input layer nodes, if the flag is false, then the node has preceeding
		// nodes that aren't input nodes.
		let firstHidNode = true;
		
		// Iterate over every previous node.
		for (let i = 0; i < this.prevNodes.length; i++) {
			// Add the node to the "posNodeCon" array if it's not already
			// in it.
			
			// Flag that is true if the previous node already has a connection and
			// false if it doesn't
			let isConnection = false;
			// Iterate over all connections
			for (let n of this.prevNodes) {
				// If the nodes are equal, there is already a conneciton, so convert the flag to true.
				if (compareNodes(n, this.prevNodes[i]) == true) {
					console.log(n);
					console.log(this.prevNodes[i]);
					isConnection = true;
					break;
				}
			}
			// If the flag is false, add the node to the array.
			if (isConnection == false) {
				posNodeCon = concatVectorsDistinct(posNodeCon, [this.prevNodes[i]]);
			}
			
			
			// If the previous node is not an input node, then get the possible
			// nodes that can connect with this node.
			if (isNaN(Number(this.prevNodes[i]))) {
				prevPosWeights = concatMatriciesDistinct(prevPosWeights, this.prevNodes[i].getPossibleWeightLocations());
				
				// Since this node has a connection with a node preceeding it, it is not
				// in the first hidden layer, so the flag should now be false.
				firstHidNode = false;
			}
			
			// If the previous node was an input node, then do nothing
		}
		
		// If the node does not have any preceeding node connections, then it can be
		// connected to all inputs, not including the ones it's already connected to.
		for (let n of this.posInputs) {
			// If the there is not already a connection between this node and the nth
			// input node, add it to the posNodeCon array.
			if (this.prevNodes.includes(n) == false) {
				posNodeCon.push(n);
			}
		}
		
		
		// Array that holds locations that new weights can be added to.
		let posWeights = [];
		
		// Iterate over every possible node connection
		for (let n of posNodeCon) {
			// Add a new possible weight in teh correct form using the
			// possible node connection.
			posWeights.push([this, n]);
		}

		// Concat the old possible weights and new possible weights
		posWeights = concatMatriciesDistinct(posWeights, prevPosWeights);
		
		// Return the possible weight locations
		return posWeights;
	}
	
	
	
	// Adds a weight between a previous node and this node.
	// prevNode - previous node to connect the weight to.
	addWeight(prevNode) {
		// When a new weight is added, it is initialized to a random value. The weight connects
		// prevNode to this node so that this node has a new input.
		
		// Add node which we want to connect this node to.
		this.prevNodes.push(prevNode);
		// Add the weight that corresponds to the onnection between prevNode and this node.
		this.prevNodesWeights.push((Math.random() * 1) * (Math.round(Math.random()) ? 1 : -1));
	}
	
	
	
	// Given a depth d, update the current node's depth to d and update
	// all nodes connected to this one with a depth of d+1.
	// depth is defined by how far away this node is from the output layer
	updateDepths(d) {
		// If the given depth is greater than the current depth, update it
		if (d > this.depth) {
			this.depth = d;
		}
		
		
		// Iterate over all nodes connected to this node.
		for (let n of this.prevNodes) {
			// If the previous node is not an input node, update it's depth with d+1
			if (isNaN(Number(n))) {
				n.updateDepths(d+1);
			}
			// If the previous node is an input node, don't do anything with it
		}
	}
}








//-------------------------------------------Neural Network----------------------------------------
// A class to create a neural network structure.
class NeuralNetwork {
	constructor(inputs, outputs, outputNodes) {
		// inputs = number of inputs. If there are 6 inputs, inputs = 6
		// outputs = number of outputs. If there are 8 outputs, outputs = 8
		this.i = inputs;
		this.o = outputs;
		
		
		// Store all nodes in the neural network
		this.nodes = [];
		
		
		
		// If an array of output nodes was given, store the nodes in this.outputs
		if (outputNodes != undefined) {
			this.outputs = outputNodes;
		}
		// If an array of output nodes was not given,
		// initialize the outputs array to an array of "outputs" number of nodes
		// with random weights connecting it to the input layer.
		else {
			// Stores the nodes in the output layer
			this.outputs = [];
			// Create "outputs" number of new nodes
			for (let i = 0; i < outputs; i++) {
				// Array that holds "inputs" number of inputs ranging from 0 to "inputs"-1
				let inp = [...Array(inputs).keys()]
				// Array that holds "inputs" number of random inputs ranging from -1 to 1.
				// The activation function is none so that softmax can be used on the entire output layer.
				let w = [...Array(inputs)].map(() => parseFloat(((Math.random() * 1) * (Math.round(Math.random()) ? 1 : -1))))
				
				// Store the new node
				let newNode = new Node(inp, w, "", [...Array(inputs).keys()])
				this.outputs.push(newNode); // Add the new node to the output layer
				this.nodes.push(newNode); // Add the new node to the total nodes
			}
		}
	}
	
	
	// Given an input vector of size "this.i", the function returns the output from the neural network
	forward(inputs) {
		// If the inputs are the right size
		if (inputs.length == this.i) {
			// Stores each output from the neural network
			let outputs = [];
			
			// Iterate through each node and add the output of that node to the outputs array
			for (let n of this.outputs) {
				outputs.push(n.forward(inputs));
			}
			
			// Return the output of the outputs sent through the softmax actiation function.
			return softmax(outputs);
		}
		
		// Return null if the inputs are not the right size
		return null;
	}
	
	
	// Returns a copy of this neural network
	getCopy() {
		// Array to hold the output nodes
		let outputNodes = [];
		
		// Iterate over each node and copy it
		for (let n of this.outputs) {
			outputNodes.push(n.getCopy());
		}
		
		// Create a new neural network
		return new NeuralNetwork(this.i, this.o, outputNodes);
	}
	
	
	// Mutates the weights and biases in the neural network,
	mutateWeightsAndBiases() {
		// Iterate over every output node
		for (let n of this.outputs) {
			// Mutate the weights and biases of the current output node
			// being examined.
			n.mutateWeightsAndBiases();
		}
		
		
		// Randomly pick an output node to mutate one weight and bias. Then pick a single weight and
		// bias to mutate in that node. That should mutate only 1 per mutation.
		
		
	}
	
	
	
	// Returns an array of all possible nodes that can be added and their location. A node location is
	// a two part vector:
	// [right_node, prevNodeIndex]
	// right_node: The node that proceeds the new node that will be added.
	// prevNodeIndex: index in the "right_node"'s previous nodes to place the
	//				  new node between. So, the new node will be placed between right_node
	//				  and right_node.prevNodes[prevNodeIndex].
	getPossibleNodeLocations() {
		// Stores the possible node locations
		let posNodes = [];
		
		// Iterate over every node node in the output layer
		for (let n of this.outputs) {
			// Get all the possible node locations for each node and add it
			// to the cumulative node positions array.
			posNodes = posNodes.concat(n.getPossibleNodeLocations());
		}
		
		// Return the possible node locations.
		return posNodes;
	}
	
	
	
	// Given a node location from the getPossibleNodeLocations() function, add a node between the
	// two given nodes.
	// The input is a vector in the following form
	// [right_node, prevNodeIndex]
	// right_node: The node that proceeds the new node that will be added.
	// prevNodeIndex: index in the "right_node"'s previous nodes to place the
	//				  new node between. So, the new node will be placed between right_node
	//				  and right_node.prevNodes[prevNodeIndex].
	
	// This is done by calling the "right_node"'s addNode function.
	addNode(nodeLocation) {
		let newNode = new Node(); // Create the new node
		nodeLocation[0].addNode(newNode, nodeLocation[1], [...Array(this.i).keys()]); // add the new node
		this.nodes.push(newNode); // Store the new node
	}
	
	
	// Returns an array of all possible weights that can be added and their location. A weight location is
	// a two part vector:
	// [right_node, prevNode]
	// right_node: The node to attach to the previous node with a weight.
	// prevNode: The node to connect to the "right_node". When the weight is added, right_node
	// 			 will add "prevNode" to it's stored prevNodes. The weight will be
	//			 a random value. Note that this is the actual node, not the location.
	getPossibleWeightLocations() {
		// Update each node's max depth.
		this.updateDepths();
		
		// Stores all possible new weight locations
		let posWeights = [];
		
		// Iterate over every node in the network
		for (let n1 of this.nodes) {
			// Iterate over every input in the network
			for (let i of [...Array(this.i).keys()]) {
				// If the node is not already connected to the input, then added the node-input
				// weight to the posWeights array.
				if (n1.prevNodes.includes(i) == false) {
					posWeights.push([n1, i]);
				}
			}
			
			// Iterate over every node in the network again.
			for (let n2 of this.nodes) {
				// Add the node-node weight to the posWeighs array if:
				// - the nodes are different
				// - n1 doesn't include n2
				// - n2 doesn't include n1
				// - n2 has a greater depth than n1
				if (compareNodes(n1, n2) == false &&
					n1.prevNodes.includes(n2) == false &&
					n2.prevNodes.includes(n1) == false &&
					n2.depth > n1.depth) {
						posWeights.push([n1, n2]);
				}
			}
		}
		
		// Return the possible weight locations
		return posWeights;
		
		
		/*
		// Stores the possible weight locations
		let posWeights = [];
		
		// Iterate over every node node in the output layer
		for (let n of this.outputs) {
			// Get all the possible node locations for each node and add it
			// to the cumulative node positions array.
			posWeights = concatMatriciesDistinct(posWeights, n.getPossibleWeightLocations());
		}
		
		// Return the possible node locations.
		return posWeights;
		*/
	}
	
	// Given a weight location from the getPossibleWeightLocations() function, add a weight between the two nodes.
	// The input is a vector in the following form
	// [right_node, prevNode]
	// right_node: The node to attach to the previous node with a weight.
	// prevNode: The node to connect to the "right_node". When the weight is added, right_node
	// 			 will add "prevNode" to it's stored prevNodes. The weight will be
	//			 a random value. Note that this is the actual node, not the location.
	
	// This is done by calling the "right_node"'s addWeight function
	addWeight(weightLocation) {
		weightLocation[0].addWeight(weightLocation[1])
	}
	
	
	
	// Updates all the depths of each node in the neural network
	// Depth defined by how many nodes away from the output layer the node is.
	// A depth of 0 means the node is in the output layer
	updateDepths() {
		// Iterate over every node in the output layer and update its depth as well
		// as the nodes connected to it.
		for (let n of this.outputs) {
			n.updateDepths(0);
		}
	}
}