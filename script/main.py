import random
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch

# Define inheritance rules for blood types
"""AA:0, AO:1, BB:2, BO:3 ,OO:4, AB:5"""
BLOOD_TYPE_RULES = {
    (0, 0): 0, 
    (0, 1): random.choice([0,1]), 
    (0, 4): 1,
    (1, 1): random.choice([0, 1, 4]),
    (1, 4): random.choice([1, 4]),
    (4, 4): 4,
    (2, 2): 2, 
    (2, 3): random.choice([2,3]), 
    (2, 4): 3,
    (3, 3): random.choice([2, 3, 4]),
    (3, 4): random.choice([3, 4]),
    (5, 0): random.choice([0, 5]),
    (5, 1): random.choice([1, 5]),
    (5, 2): random.choice([2, 5]),
    (5, 3): random.choice([3, 5]),
    (5, 4): random.choice([1, 3]),
    (5, 5): random.choice([0, 2, 5]),
}

# Grouping for higher-level view
GROUP_MAPPING = torch.tensor([0, 0, 1, 1, 2, 3])  # AA:A, AO:A, BB:B, BO:B, OO:O, AB:AB

class Simulator:
    def __init__(self):
        self.time = 0
        self.history = []

        # Initialize the population with 6 men and 6 women evenly distributed
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        initial_blood_types = torch.tensor([0, 2, 4, 4] * 2, device=self.device)
        self.population = initial_blood_types

        # Move GROUP_MAPPING to the appropriate device
        self.group_mapping = GROUP_MAPPING.to(self.device)

        """    
        def step(self):
        # Aging: Each generation lasts 3 time steps
        current_population = self.population

        # Pair and reproduce
        parents = current_population[torch.randint(len(current_population), (len(current_population),))]
        partners = current_population[torch.randint(len(current_population), (len(current_population),))]
        offspring = torch.tensor([
            BLOOD_TYPE_RULES.get((int(parents[i]), int(partners[i])), random.randint(0, 5))
            for i in range(len(parents))
        ], device=self.device)

        # Update population: remove oldest generation and add new offspring
        self.population = torch.cat((current_population, offspring))[-len(current_population):]

        # Record statistics
        self.record_statistics()

        # Increment time
        self.time += 1"""

    def step(self):
        # Memory manangement 
        max_population_size = 100000
        if len(self.population) > max_population_size: 
                # run population reduction 
                indices = torch.randperm(len(self.population))[:max_population_size]
                self.population = self.population[indices] 
                self.population_ages = self.population_ages[indices]


        # Add an age tracking tensor alongside population
        if not hasattr(self, 'population_ages'):
            self.population_ages = torch.zeros_like(self.population, dtype=torch.int)

        # Increment ages
        self.population_ages += 1

        # Find individuals older than 2 steps
        age_mask = self.population_ages >= 4

        # Remove aged individuals and replace with offspring
        current_population = self.population[~age_mask]
        current_ages = self.population_ages[~age_mask]

        # Pair and reproduce (similar to original code)
        men = current_population[:len(current_population)//2]
        women = current_population[len(current_population)//2:]
        
        offspring = torch.tensor([
            BLOOD_TYPE_RULES.get((int(men[i]), int(women[i])), random.randint(0, 5))
            for i in range(len(men))
        ], device=self.device)

        # Reset ages for new offspring to 0
        offspring_ages = torch.zeros_like(offspring, dtype=torch.int)

        # Update population and ages
        self.population = torch.cat((current_population, offspring))
        self.population_ages = torch.cat((current_ages, offspring_ages))

        # Record statistics and increment time
        self.record_statistics()
        self.time += 1

    def record_statistics(self):
        grouped_population = self.group_mapping[self.population]
        blood_type_counts = torch.bincount(grouped_population, minlength=4).float()
        blood_type_ratios = blood_type_counts / blood_type_counts.sum()
        age_dist = torch.bincount(self.population_ages, minlength=4).float()  
        
        population_stats = {
            'total_population': len(self.population),
            'blood_type_counts': blood_type_counts.cpu().tolist(),
            'blood_type_ratios': blood_type_ratios.cpu().tolist(),
            'age_distribution': age_dist.cpu().tolist() 
        }

        self.history.append(population_stats)

    def simulate(self, steps):
        for _ in tqdm(range(steps), desc="Simulating", unit="step"):
            self.step()

    def plot_statistics(self):
        times = range(len(self.history))
        total_populations = [stats['total_population'] for stats in self.history]
        blood_type_ratios = torch.tensor([stats['blood_type_ratios'] for stats in self.history]).T  # Transpose to access ratios for each blood type
        age_distributions = torch.tensor([stats['age_distribution'] for stats in self.history]).T

        # Create subplots
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 15))
        
        # Population plot
        ax1.plot(times, total_populations)
        ax1.set_title('Total Population Over Time')
        ax1.set_xlabel('Time')
        ax1.set_ylabel('Population')
        
        # Blood type ratios plot
        for i, label in enumerate(['A', 'B', 'O', 'AB']):
            ax2.plot(times, blood_type_ratios[i], label=label)
        ax2.set_title('Blood Type Ratios Over Time')
        ax2.set_xlabel('Time')
        ax2.set_ylabel('Ratio')
        ax2.legend()
        
        # Age distribution plot
        age_labels = ['0', '1', '2', '3']
        for i, label in enumerate(age_labels):
            ax3.plot(times, age_distributions[i], label=label)
        ax3.set_title('Population Trend Breakdown by Age')
        ax3.set_xlabel('Time')
        ax3.set_ylabel('Count')
        ax3.legend()

        plt.tight_layout()
        plt.show()

# Run the simulator
sim = Simulator()
sim.simulate(100)
sim.plot_statistics()