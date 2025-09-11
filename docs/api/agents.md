# Quantum Agents API Reference

SuperQuantX's quantum agents represent the next evolution of AI - autonomous systems that leverage quantum computing to achieve enhanced performance in specialized domains. This reference documents all quantum agent classes and their capabilities.

## Overview

Quantum agents are specialized combinations of quantum algorithms designed for specific problem domains. They provide high-level interfaces for complex quantum machine learning workflows and autonomous decision-making.

### Agent Types

- **[QuantumTradingAgent](#quantumtradingagent)**: Autonomous financial trading with quantum advantages
- **[QuantumResearchAgent](#quantumresearchagent)**: Scientific discovery and hypothesis generation  
- **[QuantumOptimizationAgent](#quantumoptimizationagent)**: General combinatorial and continuous optimization
- **[QuantumClassificationAgent](#quantumclassificationagent)**: Automated ML with ensemble methods
- **[QuantumPortfolioAgent](#quantumportfolioagent)**: Portfolio optimization and risk management

## Base Classes

### QuantumAgent

Base class for all quantum agents providing common functionality.

```python
from superquantx.algorithms.quantum_agents import QuantumAgent

class MyQuantumAgent(QuantumAgent):
    def _initialize_agent(self):
        # Initialize agent-specific algorithms
        pass
    
    def solve(self, problem_instance, **kwargs):
        # Implement problem-solving logic
        pass
```

#### Constructor

##### __init__(backend, agent_config=None, shots=1024, **kwargs)

**Parameters:**
- `backend` (str|Backend): Quantum backend for circuit execution
- `agent_config` (dict, optional): Configuration dictionary for the agent
- `shots` (int): Number of measurement shots (default: 1024)
- `**kwargs`: Additional parameters

#### Methods

##### _initialize_agent()

Abstract method to initialize agent-specific algorithms and configurations.

**Must be implemented by subclasses.**

##### solve(problem_instance, **kwargs)

Abstract method to solve a problem using the quantum agent.

**Parameters:**
- `problem_instance` (Any): Problem data or specification
- `**kwargs`: Additional solving parameters

**Returns:**
- `QuantumResult`: Solution result with metadata

##### get_agent_info()

Get information about the agent and its algorithms.

**Returns:**
- `dict`: Agent information including algorithms, config, and performance metrics

#### Attributes

- `algorithms` (dict): Dictionary of initialized quantum algorithms
- `results_history` (list): History of problem-solving results
- `performance_metrics` (dict): Performance metrics for different algorithms
- `agent_config` (dict): Agent configuration parameters

## Specialized Agents

### QuantumTradingAgent

Autonomous quantum trading agent for financial markets.

```python
from superquantx import QuantumTradingAgent

# Create trading agent
agent = QuantumTradingAgent(
    backend='pennylane',
    strategy='quantum_portfolio',
    risk_tolerance=0.6,
    markets=['stocks', 'crypto']
)

# Deploy agent
result = agent.deploy(market_data)
print(f"Performance: {result.result['performance']}")
print(f"Quantum advantage: {result.result['quantum_advantage']}")
```

#### Constructor

##### __init__(backend='auto', strategy='quantum_portfolio', risk_tolerance=0.5, quantum_advantage_threshold=0.05, markets=None, **kwargs)

**Parameters:**
- `backend` (str): Quantum backend ('auto' for automatic selection)
- `strategy` (str): Trading strategy type
  - `'quantum_portfolio'`: Quantum portfolio optimization
  - `'momentum'`: Quantum-enhanced momentum trading
  - `'mean_reversion'`: Mean reversion with quantum analysis
- `risk_tolerance` (float): Risk tolerance level (0.0 to 1.0)
- `quantum_advantage_threshold` (float): Minimum quantum advantage required
- `markets` (list): Markets to trade in (['stocks', 'crypto', 'forex'])
- `**kwargs`: Additional parameters

#### Methods

##### deploy(market_data=None, **kwargs)

Deploy the trading agent and return performance metrics.

**Parameters:**
- `market_data` (Any, optional): Historical market data for analysis
- `**kwargs`: Deployment parameters

**Returns:**
- `QuantumResult`: Trading performance and quantum advantage metrics

##### fit(X, y=None, **kwargs)

Fit the trading agent (optional for pre-configured agents).

**Parameters:**
- `X` (numpy.ndarray): Historical market data
- `y` (numpy.ndarray, optional): Target returns
- `**kwargs`: Fitting parameters

**Returns:**
- `QuantumTradingAgent`: Self

##### predict(X, **kwargs)

Predict trading decisions based on market data.

**Parameters:**
- `X` (numpy.ndarray): Market data for prediction

**Returns:**
- `numpy.ndarray`: Trading signals (0=hold, 1=buy, -1=sell)

#### Strategy Types

- **quantum_portfolio**: Uses QAOA for portfolio optimization
- **momentum**: Quantum-enhanced momentum detection
- **mean_reversion**: Quantum analysis of market reversals

#### Example

```python
import numpy as np
from superquantx import QuantumTradingAgent

# Create sample market data
market_data = {
    'prices': np.random.randn(100, 10),  # 100 days, 10 assets
    'returns': np.random.randn(100, 10),
    'volatility': np.random.rand(10)
}

# Create and deploy trading agent
agent = QuantumTradingAgent(
    strategy='quantum_portfolio',
    risk_tolerance=0.4,
    markets=['stocks']
)

result = agent.deploy(market_data)

print(f"Expected Return: {result.result['performance']['expected_return']:.3f}")
print(f"Risk: {result.result['performance']['risk']:.3f}")
print(f"Sharpe Ratio: {result.result['performance']['sharpe_ratio']:.3f}")
print(f"Quantum Advantage: {result.result['quantum_advantage']:.3f}")
```

### QuantumResearchAgent

Autonomous quantum research agent for scientific discovery.

```python
from superquantx import QuantumResearchAgent

# Create research agent
agent = QuantumResearchAgent(
    domain='materials_science',
    hypothesis_generation=True,
    experiment_design=True
)

# Investigate research question
result = agent.investigate(
    "How can quantum simulation accelerate discovery of superconducting materials?"
)
print(result.result['research_plan'])
```

#### Constructor

##### __init__(backend='auto', domain='materials_science', hypothesis_generation=True, experiment_design=True, literature_synthesis=False, **kwargs)

**Parameters:**
- `backend` (str): Quantum backend for simulations
- `domain` (str): Research domain
  - `'materials_science'`: Materials and condensed matter research
  - `'drug_discovery'`: Pharmaceutical and biochemical research
  - `'physics'`: Fundamental physics research
- `hypothesis_generation` (bool): Enable automated hypothesis generation
- `experiment_design` (bool): Enable quantum experiment design
- `literature_synthesis` (bool): Enable literature review and synthesis
- `**kwargs`: Additional parameters

#### Methods

##### investigate(research_question, constraints=None, **kwargs)

Investigate a research question and return research plan.

**Parameters:**
- `research_question` (str): The research question to investigate
- `constraints` (dict, optional): Research constraints (timeline, budget, etc.)
- `**kwargs`: Investigation parameters

**Returns:**
- `QuantumResult`: Research plan, hypotheses, and experiment designs

##### fit(X, y=None, **kwargs)

Fit the research agent to domain-specific data.

##### predict(X, **kwargs)

Predict research outcomes based on data.

#### Research Domains

- **materials_science**: Quantum simulation of materials, electronic structure calculations
- **drug_discovery**: Molecular simulation, protein folding, drug-target interactions
- **physics**: Fundamental physics simulations, quantum many-body systems

#### Example

```python
from superquantx import QuantumResearchAgent

# Create research agent for drug discovery
agent = QuantumResearchAgent(
    domain='drug_discovery',
    hypothesis_generation=True,
    experiment_design=True
)

# Investigate research question
research_question = "Can quantum simulation identify novel COVID-19 drug targets?"
constraints = {'timeline': '18_months', 'budget': 500000}

result = agent.investigate(research_question, constraints)

# Access results
research_plan = result.result['research_plan']
hypothesis = result.result['hypothesis']
experiments = result.result['experiments']

print("Research Plan Phases:", research_plan['phases'])
print("Primary Hypothesis:", hypothesis['primary_hypothesis'])
print("Number of Experiments:", len(experiments))

for exp in experiments:
    print(f"- {exp['experiment_name']}: {exp['expected_duration']}")
```

### QuantumOptimizationAgent

General-purpose quantum optimization agent.

```python
from superquantx import QuantumOptimizationAgent

# Create optimization agent
agent = QuantumOptimizationAgent(
    backend='qiskit',
    problem_type='combinatorial',
    algorithms=['qaoa', 'vqe']
)

# Solve optimization problem
result = agent.solve(problem_instance)
```

#### Constructor

##### __init__(backend, problem_type='combinatorial', algorithms=None, **kwargs)

**Parameters:**
- `backend` (str): Quantum backend
- `problem_type` (str): Type of optimization
  - `'combinatorial'`: Discrete optimization (TSP, Max-Cut, etc.)
  - `'continuous'`: Continuous optimization (VQE, portfolio, etc.)
  - `'mixed'`: Mixed integer optimization
- `algorithms` (list): Algorithms to use (['qaoa', 'vqe'])
- `**kwargs`: Additional parameters

#### Methods

##### solve(problem_instance, **kwargs)

Solve optimization problem using appropriate quantum algorithm.

**Parameters:**
- `problem_instance` (Any): Problem specification or data
- `**kwargs`: Solving parameters

**Returns:**
- `QuantumResult`: Optimization result with solution and metadata

##### fit(X, y=None, **kwargs)

Fit optimization algorithms to problem characteristics.

##### predict(X, **kwargs)

Predict optimal solutions for new problem instances.

#### Supported Algorithms

- **QAOA**: Quantum Approximate Optimization Algorithm for combinatorial problems
- **VQE**: Variational Quantum Eigensolver for continuous optimization

#### Example

```python
import numpy as np
from superquantx import QuantumOptimizationAgent

# Create Max-Cut problem (combinatorial)
edges = [(0, 1), (1, 2), (2, 3), (3, 0), (0, 2)]
weights = [1, 1, 1, 1, 2]

problem = {
    'type': 'max_cut',
    'edges': edges,
    'weights': weights,
    'num_nodes': 4
}

# Create and use optimization agent
agent = QuantumOptimizationAgent(
    problem_type='combinatorial',
    algorithms=['qaoa']
)

result = agent.solve(problem)
print(f"Optimal solution: {result.result}")
print(f"Algorithm used: {result.metadata['algorithm_used']}")
```

### QuantumClassificationAgent

Ensemble quantum machine learning agent for classification.

```python
from superquantx import QuantumClassificationAgent

# Create classification agent
agent = QuantumClassificationAgent(
    backend='pennylane',
    algorithms=['quantum_svm', 'quantum_nn', 'hybrid'],
    ensemble_method='weighted'
)

# Fit and predict
agent.fit(X_train, y_train)
predictions = agent.predict(X_test)
```

#### Constructor

##### __init__(backend, algorithms=None, ensemble_method='voting', auto_tune=False, **kwargs)

**Parameters:**
- `backend` (str): Quantum backend
- `algorithms` (list): Algorithms to include
  - `'quantum_svm'`: Quantum Support Vector Machine
  - `'quantum_nn'`: Quantum Neural Network
  - `'hybrid'`: Hybrid quantum-classical classifier
- `ensemble_method` (str): How to combine predictions
  - `'voting'`: Majority voting
  - `'weighted'`: Performance-weighted voting
  - `'stacking'`: Meta-learning ensemble
- `auto_tune` (bool): Automatic hyperparameter tuning
- `**kwargs`: Additional parameters

#### Methods

##### fit(X, y, **kwargs)

Train all classification algorithms and compute ensemble weights.

**Parameters:**
- `X` (numpy.ndarray): Training features
- `y` (numpy.ndarray): Training labels
- `**kwargs`: Training parameters

**Returns:**
- `QuantumClassificationAgent`: Self

##### predict(X, **kwargs)

Make ensemble predictions on new data.

**Parameters:**
- `X` (numpy.ndarray): Test features
- `**kwargs`: Prediction parameters

**Returns:**
- `numpy.ndarray`: Predicted labels

##### solve(problem_instance, **kwargs)

Solve complete classification problem with performance evaluation.

**Parameters:**
- `problem_instance` (tuple): (X, y) data tuple
- `**kwargs`: Solving parameters

**Returns:**
- `QuantumResult`: Classification results with ensemble performance

#### Ensemble Methods

- **voting**: Simple majority voting among algorithms
- **weighted**: Weight votes by individual algorithm performance
- **stacking**: Train meta-learner to combine algorithm predictions

#### Example

```python
from sklearn.datasets import make_classification
from superquantx import QuantumClassificationAgent

# Generate dataset
X, y = make_classification(n_samples=200, n_features=8, n_classes=2, random_state=42)
X_train, X_test = X[:160], X[160:]
y_train, y_test = y[:160], y[160:]

# Create ensemble agent
agent = QuantumClassificationAgent(
    algorithms=['quantum_svm', 'quantum_nn'],
    ensemble_method='weighted',
    backend='simulator'
)

# Train ensemble
agent.fit(X_train, y_train)

# Make predictions
predictions = agent.predict(X_test)
accuracy = agent.score(X_test, y_test)

print(f"Ensemble accuracy: {accuracy:.3f}")
print("Individual performances:", agent.performance_metrics)
```

### QuantumPortfolioAgent

Specialized quantum agent for portfolio optimization and risk management.

```python
from superquantx import QuantumPortfolioAgent

# Create portfolio agent
agent = QuantumPortfolioAgent(
    backend='pennylane',
    risk_model='mean_variance',
    optimization_objective='sharpe',
    constraints=[
        {'type': 'budget'},
        {'type': 'long_only'},
        {'type': 'max_weight', 'value': 0.2}
    ]
)

# Fit to historical returns
agent.fit(returns_data)
optimal_weights = agent.predict(current_data)
```

#### Constructor

##### __init__(backend, risk_model='mean_variance', optimization_objective='sharpe', constraints=None, rebalancing_frequency='monthly', **kwargs)

**Parameters:**
- `backend` (str): Quantum backend
- `risk_model` (str): Risk model type
  - `'mean_variance'`: Classic mean-variance optimization
  - `'black_litterman'`: Black-Litterman model
  - `'factor'`: Factor-based risk model
- `optimization_objective` (str): Objective function
  - `'return'`: Maximize expected return
  - `'sharpe'`: Maximize Sharpe ratio
  - `'risk_parity'`: Risk parity allocation
- `constraints` (list): Portfolio constraints (see below)
- `rebalancing_frequency` (str): Rebalancing frequency
- `**kwargs`: Additional parameters

#### Constraint Types

Portfolio constraints can be specified as dictionaries:

```python
constraints = [
    {'type': 'budget'},                                    # Sum of weights = 1
    {'type': 'long_only'},                                # All weights >= 0
    {'type': 'max_weight', 'value': 0.3},                 # Max weight per asset
    {'type': 'sector_limit', 'assets': [0,1,2], 'value': 0.5}  # Sector exposure limit
]
```

#### Methods

##### fit(X, y=None, **kwargs)

Fit portfolio optimization to historical returns data.

**Parameters:**
- `X` (numpy.ndarray): Historical returns (samples × assets)
- `y` (numpy.ndarray, optional): Not used
- `**kwargs`: Additional parameters including `risk_aversion`

**Returns:**
- `QuantumPortfolioAgent`: Self

##### predict(X, **kwargs)

Predict optimal portfolio weights for current market conditions.

**Parameters:**
- `X` (numpy.ndarray): Current market data
- `**kwargs`: Prediction parameters

**Returns:**
- `numpy.ndarray`: Optimal portfolio weights

##### solve(problem_instance, **kwargs)

Solve complete portfolio optimization problem.

**Parameters:**
- `problem_instance` (numpy.ndarray): Returns data
- `**kwargs`: Solving parameters including `method` ('vqe' or 'qaoa')

**Returns:**
- `QuantumResult`: Portfolio optimization results

#### Example

```python
import numpy as np
from superquantx import QuantumPortfolioAgent

# Generate sample returns data
np.random.seed(42)
returns = np.random.multivariate_normal(
    mean=[0.08, 0.12, 0.10, 0.15],  # Expected returns
    cov=[[0.16, 0.05, 0.03, 0.02],   # Covariance matrix
         [0.05, 0.25, 0.04, 0.03],
         [0.03, 0.04, 0.20, 0.05],
         [0.02, 0.03, 0.05, 0.30]],
    size=252  # One year of daily returns
)

# Create portfolio agent with constraints
agent = QuantumPortfolioAgent(
    risk_model='mean_variance',
    optimization_objective='sharpe',
    constraints=[
        {'type': 'budget'},
        {'type': 'long_only'},
        {'type': 'max_weight', 'value': 0.4}
    ]
)

# Optimize portfolio
agent.fit(returns, risk_aversion=2.0)
result = agent.solve(returns, method='vqe')

weights = result.result['weights']
expected_return = result.result['expected_return']
risk = result.result['risk']
sharpe = result.result['sharpe_ratio']

print(f"Optimal weights: {weights}")
print(f"Expected return: {expected_return:.3f}")
print(f"Risk (volatility): {risk:.3f}")
print(f"Sharpe ratio: {sharpe:.3f}")
```

## Agent Workflows

### Multi-Agent Systems

Combine multiple agents for complex problems:

```python
from superquantx import QuantumTradingAgent, QuantumResearchAgent

# Research agent discovers new strategies
research_agent = QuantumResearchAgent(domain='finance')
research_result = research_agent.investigate(
    "What quantum advantages exist in high-frequency trading?"
)

# Trading agent implements the strategies
trading_agent = QuantumTradingAgent(
    strategy='quantum_momentum',
    risk_tolerance=0.3
)
trading_result = trading_agent.deploy(market_data)

# Compare results
quantum_advantage = (
    trading_result.result['quantum_advantage'] + 
    research_result.result.get('expected_quantum_advantage', 0)
) / 2
```

### Agent Orchestration

Coordinate agents for complex workflows:

```python
class QuantumAgentOrchestrator:
    def __init__(self):
        self.agents = {}
    
    def add_agent(self, name, agent):
        self.agents[name] = agent
    
    def run_workflow(self, problem_data, workflow_steps):
        results = {}
        
        for step in workflow_steps:
            agent_name = step['agent']
            method = step['method']
            params = step.get('params', {})
            
            agent = self.agents[agent_name]
            result = getattr(agent, method)(problem_data, **params)
            results[agent_name] = result
            
            # Pass results to next step
            if 'output_to' in step:
                problem_data = result.result
        
        return results

# Use orchestrator
orchestrator = QuantumAgentOrchestrator()
orchestrator.add_agent('research', QuantumResearchAgent())
orchestrator.add_agent('optimization', QuantumOptimizationAgent())

workflow = [
    {'agent': 'research', 'method': 'investigate', 'params': {'research_question': 'Optimize supply chain'}},
    {'agent': 'optimization', 'method': 'solve', 'output_to': 'final_result'}
]

results = orchestrator.run_workflow(supply_chain_data, workflow)
```

### Performance Monitoring

Monitor agent performance across deployments:

```python
class AgentPerformanceMonitor:
    def __init__(self):
        self.performance_history = {}
    
    def track_agent_performance(self, agent_name, result):
        if agent_name not in self.performance_history:
            self.performance_history[agent_name] = []
        
        performance_data = {
            'timestamp': time.time(),
            'execution_time': result.execution_time,
            'success': result.error is None,
            'quantum_advantage': result.result.get('quantum_advantage', 0),
            'metadata': result.metadata
        }
        
        self.performance_history[agent_name].append(performance_data)
    
    def get_performance_summary(self, agent_name):
        history = self.performance_history.get(agent_name, [])
        if not history:
            return None
        
        success_rate = sum(1 for h in history if h['success']) / len(history)
        avg_execution_time = np.mean([h['execution_time'] for h in history])
        avg_quantum_advantage = np.mean([h['quantum_advantage'] for h in history])
        
        return {
            'success_rate': success_rate,
            'avg_execution_time': avg_execution_time,
            'avg_quantum_advantage': avg_quantum_advantage,
            'total_deployments': len(history)
        }

# Monitor agent performance
monitor = AgentPerformanceMonitor()

# Deploy agents and track performance
trading_agent = QuantumTradingAgent()
result = trading_agent.deploy(market_data)
monitor.track_agent_performance('trading_agent', result)

# Get performance summary
summary = monitor.get_performance_summary('trading_agent')
print(f"Success rate: {summary['success_rate']:.2%}")
print(f"Avg execution time: {summary['avg_execution_time']:.2f}s")
```

## Error Handling

All quantum agents implement robust error handling:

```python
try:
    agent = QuantumTradingAgent(strategy='invalid_strategy')
    result = agent.deploy(market_data)
    
    if result.error:
        print(f"Agent error: {result.error}")
        print(f"Metadata: {result.metadata}")
    else:
        print(f"Success: {result.result}")
        
except Exception as e:
    print(f"Unexpected error: {e}")
```

### Common Error Types

- **AlgorithmError**: Issues with quantum algorithm execution
- **ConfigurationError**: Invalid agent configuration
- **ConvergenceError**: Optimization failed to converge
- **BackendError**: Quantum backend issues
- **DataError**: Invalid input data format

### Error Recovery

Agents can implement automatic error recovery:

```python
class ResilientQuantumAgent(QuantumAgent):
    def solve(self, problem_instance, max_retries=3, **kwargs):
        for attempt in range(max_retries):
            try:
                return super().solve(problem_instance, **kwargs)
            except Exception as e:
                if attempt == max_retries - 1:
                    raise e
                else:
                    logger.warning(f"Attempt {attempt + 1} failed: {e}. Retrying...")
                    # Modify strategy for retry
                    kwargs['fallback_mode'] = True
```

## Best Practices

### Agent Selection

Choose agents based on problem characteristics:

```python
def select_optimal_agent(problem_type, data_size, quantum_resources):
    if problem_type == 'portfolio_optimization':
        return QuantumPortfolioAgent(
            backend='pennylane' if quantum_resources == 'high' else 'simulator'
        )
    elif problem_type == 'classification' and data_size > 1000:
        return QuantumClassificationAgent(
            algorithms=['quantum_svm', 'hybrid'],
            ensemble_method='weighted'
        )
    elif problem_type == 'research':
        return QuantumResearchAgent(
            domain=infer_domain(problem_type),
            hypothesis_generation=True
        )
    else:
        return QuantumOptimizationAgent(
            problem_type='combinatorial' if is_discrete(problem_type) else 'continuous'
        )
```

### Configuration Management

Use configuration files for agent setup:

```python
import yaml

# agent_config.yaml
config = yaml.safe_load("""
trading_agent:
  strategy: quantum_portfolio
  risk_tolerance: 0.4
  markets: [stocks, crypto]
  constraints:
    - type: long_only
    - type: max_weight
      value: 0.3

research_agent:
  domain: materials_science
  hypothesis_generation: true
  experiment_design: true
""")

# Load agents from config
agents = {}
for agent_name, agent_config in config.items():
    if 'trading' in agent_name:
        agents[agent_name] = QuantumTradingAgent(**agent_config)
    elif 'research' in agent_name:
        agents[agent_name] = QuantumResearchAgent(**agent_config)
```

### Performance Optimization

Optimize agent performance:

```python
# 1. Use appropriate backends
agent = QuantumTradingAgent(
    backend='pennylane' if complex_problem else 'simulator'
)

# 2. Tune shot counts
agent.shots = 2048 if high_precision_needed else 1024

# 3. Enable parallel execution
results = []
with ThreadPoolExecutor() as executor:
    futures = [
        executor.submit(agent.solve, problem_instance) 
        for problem_instance in problem_batch
    ]
    results = [f.result() for f in futures]

# 4. Cache expensive computations
from functools import lru_cache

@lru_cache(maxsize=128)
def cached_agent_solve(agent_hash, problem_hash):
    return agent.solve(problem_instance)
```

## Examples

### Financial Portfolio Management

Complete portfolio management workflow:

```python
import pandas as pd
import numpy as np
from superquantx import QuantumPortfolioAgent, QuantumTradingAgent

# Load historical data
returns_data = pd.read_csv('stock_returns.csv').values

# Create portfolio optimization agent
portfolio_agent = QuantumPortfolioAgent(
    optimization_objective='sharpe',
    constraints=[
        {'type': 'budget'},
        {'type': 'long_only'},
        {'type': 'max_weight', 'value': 0.25}
    ]
)

# Optimize portfolio
portfolio_agent.fit(returns_data)
optimal_weights = portfolio_agent.predict(returns_data[-30:])

# Create trading agent for execution
trading_agent = QuantumTradingAgent(
    strategy='quantum_portfolio',
    risk_tolerance=0.3
)

# Deploy trading strategy
trading_result = trading_agent.deploy({
    'optimal_weights': optimal_weights,
    'current_prices': current_market_data
})

print(f"Portfolio Performance:")
print(f"Expected Return: {trading_result.result['performance']['expected_return']:.2%}")
print(f"Risk: {trading_result.result['performance']['risk']:.2%}")
print(f"Quantum Advantage: {trading_result.result['quantum_advantage']:.2%}")
```

### Scientific Research Acceleration

Accelerate materials science research:

```python
from superquantx import QuantumResearchAgent

# Create research agent for materials science
research_agent = QuantumResearchAgent(
    domain='materials_science',
    hypothesis_generation=True,
    experiment_design=True
)

# Investigate superconductor research
research_question = (
    "How can quantum simulation discover room-temperature superconductors "
    "with transition temperatures above 300K?"
)

constraints = {
    'timeline': '24_months',
    'budget': 1000000,
    'quantum_hardware': 'available'
}

result = research_agent.investigate(research_question, constraints)

# Access research plan
plan = result.result['research_plan']
hypothesis = result.result['hypothesis']
experiments = result.result['experiments']

print("Research Plan:")
for i, phase in enumerate(plan['phases'], 1):
    print(f"{i}. {phase}")

print(f"\nPrimary Hypothesis:")
print(hypothesis['primary_hypothesis'])

print(f"\nProposed Experiments:")
for exp in experiments:
    print(f"- {exp['experiment_name']}")
    print(f"  Duration: {exp['expected_duration']}")
    print(f"  Resources: {exp['quantum_resources_needed']}")
```

### Multi-Modal Classification

Complex classification with ensemble methods:

```python
from sklearn.datasets import fetch_openml
from superquantx import QuantumClassificationAgent

# Load complex dataset
X, y = fetch_openml('wine-quality-red', return_X_y=True, as_frame=False)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create ensemble classification agent
classification_agent = QuantumClassificationAgent(
    algorithms=['quantum_svm', 'quantum_nn', 'hybrid'],
    ensemble_method='weighted',
    auto_tune=True
)

# Solve classification problem
result = classification_agent.solve((X_train, y_train))

# Evaluate on test set
test_predictions = classification_agent.predict(X_test)
test_accuracy = accuracy_score(y_test, test_predictions)

print("Classification Results:")
print(f"Training Accuracy: {result.result['accuracy']:.3f}")
print(f"Test Accuracy: {test_accuracy:.3f}")

print("\nIndividual Algorithm Performance:")
for algo, performance in result.result['individual_performances'].items():
    print(f"  {algo}: {performance:.3f}")

print(f"\nEnsemble Method: {result.result['ensemble_method']}")
print(f"Algorithms Used: {result.metadata['algorithms_used']}")
```

---

*The quantum agent API represents the cutting edge of quantum AI - autonomous systems that leverage quantum advantages for real-world problem solving. These agents combine multiple quantum algorithms with classical AI to create powerful, specialized problem-solving capabilities.*

!!! tip "Agent Selection"
    Choose agents based on your problem domain and quantum resource availability. Start with simulation backends and upgrade to quantum hardware when advantages are demonstrated.

!!! warning "Performance Considerations"
    Quantum agents may require significant computational resources. Monitor performance and use appropriate backends for your use case.

!!! info "Extensibility"
    The agent framework is designed for extensibility. Create custom agents by inheriting from `QuantumAgent` and implementing domain-specific logic.