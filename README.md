# lgp

![LGP Evolution](media/lgp_evolution.gif)

This is a simple Rust library which implements [linear genetic programming](https://en.wikipedia.org/wiki/Linear_genetic_programming). 

### Details
The library currently only supports single register input and output. This means the possible use cases would include evolving programs which encode single variable functions given a set of training data (i.e., (x, y) pairs).

This library uses tournament selection as well as self-adaptation of certain mutation parameters.

The data flow is similar to the classic LGP: 
- Initialize population of random programs (only effective code, no semantic introns)
- Run data through each program and assign each program a fitness value based on MSE (mean-squared error)
- Perform tournament selection on population based on their fitness values
- Perform cross-over mutation on winners of tournament selection
- Replace previous population with crossover/mutation resulting programs
- Repeat for fixed number of generations or until best fitness falls below set threshold

### Performance
Currently needs a lot of work to make it faster. Many generations are needed to converge on exact solution or nearly exact solution.

### TODOs
- Re-write library to support variable input and output registers (this would make the use cases far more vast - e.g., time series)
- Perform testing to see which methods are most efficient (e.g., to allow introns or not), adding structural intron removal, etc
- Update and complete documentation
- Update and improve experience for user to set the configuration values
- **TOP PRIORITY**: Add sick GIF showing evolution of single-variable polynomial regression graph. Each frame is the graph of the current best program from generation 1 to the last generation.
