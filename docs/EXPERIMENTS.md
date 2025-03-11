# Base Experiment

- We first initialize 2N random genomes. We split them in half, keeping half in group A and half in group B.
- Randomly select one genome from group A and B and enter them into a fighting simulation, or an **iteration**. Output who won, time left on the timer and player HP. We will fix settings and character selection in the fighting simulation as control variables. ![iteration process picture](../images/iteration.png)
- Repeat N times, sampling without replacement from both pools. When both pools are exhausted, this record a **generation**.
- For each of the N outcomes recorded in the generation, calculate individual fitness for each genome. The details of the fitness function is to be discussed, but we'll likely use some weighted sum of win/lose status, player HP and timer.
- Having two sets of fitness calculated, we select from these fitness K individual genomes (likely using a randomized elitist policy) for survival, yielding 2K individuals (K < N).
- Have these 2K individuals mutate until the population is 2N again. I'm not certain of the mutation policy we'll use, but it's likely going to be similar to the NEAT paper.
- Repeat from step 2 for G **generations**. ![evolution process picture](../images/evolution.png)

# Single Evolving Species vs. Adversarially Evolving Species

If the game as an AI mode, we can do a single-player genome evaluation instead of multiplayer. Then, we can pitch the end result of this process with a species that evolved from the adversarial process to see who's better.

# Holistic Evolution vs. Species Evolution

In Holisitc Evolution (HE), we allow for extinction by applying the selection process on both populations as a whole instead of each population individually. With this, we can track population change and observe any equilibrium or species collapse. With this, we would have to implement sampling with replacement for the population with less species and average out repeated fitness calculation for an individual if they are sampled more than once. Of course, at the end, we can also pitch individuals from this process with the regular selection process to see who's better.

# Different Architecture

Either we allow for the genome to encode the NN architecture, or manually just pick two different architecures for population A and B. This would mean that they can end up with different genome structure, which is fine, but out fitness function need to accomodate that. With this, we can extract what network architecture would be better for our purpose.