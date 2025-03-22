# Evolutionary Simulation Using a GA

This repository contains a Python-based evolutionary simulation that models sequence evolution using a genetic algorithm (GA). The simulation explores the effects of mutation and horizontal gene transfer (HGT) on nucleic acid and protein sequence evolution under selection pressures. It is designed to simulate a population of nucleotide sequences, enforce conservation in specific regions, and evaluate fitness based on hydrophobicity profiles of translated amino acid sequences.

This project was developed as part of a computational biology assignment by Matthew Kinahan, Mohammad Abbaspour, and Dawson Burgess at the University of Idaho, March 2025.

## Features

- **Population Evolution**: Simulates a population of 100 individuals with 99-nucleotide sequences over 500 generations.
- **Mutation**: Introduces point mutations at a 1% rate per nucleotide per generation.
- **Horizontal Gene Transfer (HGT)**: Optionally transfers 3-base segments between high- and low-fitness individuals.
- **Conservation Enforcement**: Maintains 97% similarity to the ancestor in a designated region (positions 80-99).
- **Fitness Function**: Selects for hydrophobic middles and hydrophilic ends in translated amino acid sequences.
- **Alignment Analysis**: Compares evolved sequences to the ancestor at both nucleic acid (Needleman-Wunsch) and protein (BLOSUM50) levels.
- **Visualization**: Generates plots for fitness, deletions, and sequence similarity over generations.

## Installation

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/your-username/evolutionary-simulation-using-a-ga.git
   cd evolutionary-simulation-using-a-ga
   ```
   
2. Set Up a Virtual Environment (optional but recommended):
  ```bash 
  python -m venv venv
  source venv/bin/activate
  ```
3. Install Dependencies:
  ```bash
  pip install -r requirements.txt
  ```

## Usage 
Run the simulation with the default configuration (3 iterations of two experimental conditions: mutation only and mutation + HGT):
  ```bash
  python3 main.py
  ```
The script will:
- Execute the evolutionary simulation.
- Save results as CSV files in `experiment_X` directories and averaged results in the root directory.
- Generate plots (e.g., fitness over generations, nucleic vs. protein alignment) saved as PNG files.

To modify parameters (e.g., population size, mutation rate), edit the `EvolutionSimulator` class in `main.py`.

## Project Structure
evolutionary-simulation-using-a-ga/
│
├── main.py                 # Main simulation script
├── README.md              # Project documentation (this file)
├── LICENSE                # MIT License file
├── requirements.txt       # Python dependencies
├── experiment_X/          # Output directories for each experiment run
│   ├── experiment1_data.csv       # Data for mutation-only condition
│   ├── experiment2_data.csv       # Data for mutation + HGT condition
│   ├── fitness_over_generations.png    # Fitness plot
│   ├── protein_vs_nucleic_alignment.png # Alignment comparison plot
│   └── deletions_boxplot.png          # Deletions distribution plot
├── avg_experiment1_data.csv     # Averaged data for mutation-only condition
├── avg_experiment2_data.csv     # Averaged data for mutation + HGT condition
├── avg_fitness_over_generations.png  # Averaged fitness plot
├── avg_protein_vs_nucleic_alignment.png  # Averaged alignment plot
├── avg_deletions_boxplot.png         # Averaged deletions plot
├── all_experiments_fitness.png       # Fitness plot for all runs
└── all_experiments_alignment.png     # Alignment plot for all runs


## Dependencies

See `requirements.txt` for a full list of required Python packages.

## Results

The simulation compares two conditions:

1. **Mutation Only**: Evolution driven by random mutations and selection.
2. **Mutation + HGT**: Adds horizontal gene transfer to spread beneficial mutations.

Key findings (based on 3 runs):
- HGT increases final average fitness (e.g., 5.22–6.10 vs. 4.26–5.22 for mutation only).
- Nucleic acid similarity stabilizes at 55–66%, while protein alignment scores vary more with HGT (4.96–20.69 vs. -19.58–11.69).
- HGT maintains higher protein-level similarity, suggesting better preservation of function.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Contributing

Feel free to fork this repository, submit issues, or create pull requests. Contributions are welcome!

## Acknowledgments

- Developed by Matthew Kinahan, Mohammad Abbaspour, and Dawson Burgess.
- Codon table sourced from [GenScript](https://www.genscript.com/tools/codon-table).
- Hydrophobicity scales from [CLC Genomics Workbench](https://www.qiagenbioinformatics.com/).

For more details, refer to the project report included in the repository or contact the authors.
