// Given a neural network, return a copied and mutated neural network,
function mutate(brain) {
	// Constants to be used during the mutation process
	const mutateWeights = 0.34;
	const addNode = 0.33;
	const addWeight = 0.33;
	
	//33% chance to add node, 33% chance to add weight 33% chance to mutate weights nromally.
	
	// Get a random number between 0 and 1.
	let num = Math.random(1);
	
	
	
	// Get a copy of the brain
	mutBrain = brain.getCopy();
	
	
	// If the number is less than "mutateWeights", mutate the weights and biases in the neural network
	if (num <= mutateWeights) {
		mutBrain.mutateWeightsAndBiases();
	}
	// If the number if greater than "mutateWeights", but less than "mutateWeights"+"addNode",
	// add a node to the network.
	else if (num > mutateWeights && num <= mutateWeights+addNode) {
		// Get the possible nodes that can be added.
		posNodes = mutBrain.getPossibleNodeLocations();
		
		// Add a random node from the nodes in the node array.
		mutBrain.addNode(posNodes[Math.floor(Math.random() * posNodes.length)]);
	}
	// If the number is greater than "mutateWeights"+"addNode" and less than 
	// "mutateWeights"+"addNode"+"addWeight", add a new weight to the network.
	else {
		// Get the possible weights that can be added.
		posWeights = mutBrain.getPossibleWeightLocations();
		
		// If there are no possible weights to add, mutate the weights
		if (posWeights.length == 0) {
			mutBrain.mutateWeightsAndBiases();
		}
		// If there are possible weights, randomly choose one and add it to the network
		else {
			mutBrain.addWeight(posWeights[Math.floor(Math.random() * posWeights.length)]);
		}
	}
	
	
	// Return the mutated brain
	return mutBrain;
}




function nextGeneration() {
	// Calculate the fitness for each dino and normalize the fitness values
	// between 0 and 1 and add up to 1.
	console.log('next generation');
	calculateFitness();
	
	// Create a new population of dinos -------------------
	
	
	// Get the dino with the highest fitness
	let bestFitness = 0;
	let bestBrain;
	
	for (let i = 0; i < POPULATION; i++) {
		if (dinos[i].fitness > bestFitness) {
			bestFitness = dinos[i].fitness;
			bestBrain = dinos[i].brain;
		}
	}
	
	// Copy the dino with the highest fitness to the array without mutating it
	dinos[0].brain = bestBrain.getCopy();
	
	// Copy and mutate the rest of the dinos
	for (let i = 1; i < POPULATION; i++) {
		// Pick a dino based on the probabiloty mapped to it's fitness
		dinos[i].brain = pickOne();
	}
}


// Pick a dino based on it's fitness. Those with a higher fitness are more liekly
// to pass on their genes to the next population
function pickOne() {
	// Put each dino's fitness in the array and sum up all the weights
	dinoFitness = [];
	sum = 0;
	for (let w = 0; w < POPULATION; w++) {
		if (w == 0) {
			dinoFitness.push(dinos[w].fitness * 100);
			sum += dinos[w].fitness * 100;
		}
		else {
			dinoFitness.push(dinos[w].fitness);
			sum += dinos[w].fitness;
		}
	}
	// Get the threshold
	const threshold = Math.random() * sum;
	
	// Now we just need to loop through the main data one more time
	// until we discover which value would live within this
	// particular threshold. We need to keep a running count of
	// weights as we go, so let's just reuse the "total" variable
	// since it was already declared.
	let total = 0;
	for (let w = 0; w < POPULATION; ++w) {
		// Add the weight to our running total.
		total += dinoFitness[w];

		// If this value falls within the threshold, we're done!
		if (total >= threshold) {
			let bestDino = dinoFitness[w];
			break;
		}
	}
	
	
	
	/*
	let bestDino;
	let bestScore = 0;
	for (let w = 0; w < POPULATION; w++) {
		if (dinos[w].fitness > bestScore) {
			bestDino = dinos[w];
			bestScore = dinos[w].fitness;
		}
	}
	*/
	
	/*
	var index = 0;
	var r = Math.random(1);
	
	// Pcik a dino based on it's fitness value
	while(r > 0) {
		r = r - dinos[index].fitness;
		index++;
	}
	index--;
	*/
	
	
	// Return a mutated brain of the chosen dinos brain
	return mutate(bestDino.brain);
}


// Calculate the fitness for each dino and normalize the fitness values
function calculateFitness() {
	// Get the highest distance ran
	let bestDistanceRan = 0;
	let bestDino;
	
	for (let i = 0; i < POPULATION; i++) {
		if (dinos[i].distanceRan > bestDistanceRan) {
			bestDistanceRan = dinos[i].distanceRan;
			bestDino = dinos[i];
		}
	}
	
	// Multiply the furthest distance by rate so that it has a higher chance of being chosen
	const rate = 1;
	bestDino.distanceRan *= rate;
	
	
	// Get the sum of all dino scores
	let s = 0;
	for (let d of dinos) {
		s += d.distanceRan;
	}
	
	
	
	// Create the fitness for each dino between 0 and 1
	for (let d of dinos) {
		d.fitness = d.distanceRan / s;
	}
}