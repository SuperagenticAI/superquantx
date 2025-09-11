#!/usr/bin/env python3
"""
SuperQuantX Quantum Agentic AI Demo

This example demonstrates the new quantum agents capabilities,
showcasing autonomous quantum-enhanced AI systems that can
make decisions and solve complex problems independently.
"""

import numpy as np

import superquantx as sqx


def demo_quantum_trading_agent():
    """Demonstrate quantum trading agent capabilities."""
    print("🤖 Quantum Trading Agent Demo")
    print("=" * 50)

    # Create a quantum trading agent
    trader = sqx.QuantumTradingAgent(
        backend='simulator',
        strategy='quantum_portfolio',
        risk_tolerance=0.3,
        markets=['crypto', 'stocks']
    )

    # Deploy the agent
    print("Deploying quantum trading agent...")
    result = trader.deploy()

    if result.result:
        performance = result.result['performance']
        quantum_advantage = result.result['quantum_advantage']

        print("✅ Agent deployed successfully!")
        print(f"Expected Return: {performance['expected_return']:.2%}")
        print(f"Risk Level: {performance['risk']:.2%}")
        print(f"Sharpe Ratio: {performance['sharpe_ratio']:.2f}")
        print(f"Quantum Advantage: {quantum_advantage:.2%}")
        print(f"Execution Time: {result.execution_time:.2f}s")
    else:
        print(f"❌ Deployment failed: {result.error}")

    print()

def demo_quantum_research_agent():
    """Demonstrate quantum research agent capabilities."""
    print("🔬 Quantum Research Agent Demo")
    print("=" * 50)

    # Create a quantum research agent for materials science
    researcher = sqx.QuantumResearchAgent(
        backend='simulator',
        domain='materials_science',
        hypothesis_generation=True,
        experiment_design=True
    )

    # Investigate a research question
    print("Investigating superconductor research question...")
    result = researcher.investigate(
        research_question="Novel room-temperature superconductor discovery",
        constraints={"budget": 500000, "timeline": "18_months"}
    )

    if result.result:
        research_plan = result.result['research_plan']
        hypothesis = result.result['hypothesis']
        experiments = result.result['experiments']

        print("✅ Research investigation completed!")
        print(f"Research Domain: {result.result['domain']}")
        print(f"Timeline: {research_plan['timeline']}")
        print(f"Budget Estimate: ${research_plan['budget_estimate']:,}")
        print(f"Quantum Simulations: {research_plan['quantum_simulations_required']}")
        print(f"Primary Hypothesis: {hypothesis['primary_hypothesis'][:100]}...")
        print(f"Experiments Designed: {len(experiments)}")
        print(f"Execution Time: {result.execution_time:.2f}s")
    else:
        print(f"❌ Investigation failed: {result.error}")

    print()

def demo_quantum_optimization_agent():
    """Demonstrate quantum optimization agent capabilities."""
    print("⚡ Quantum Optimization Agent Demo")
    print("=" * 50)

    # Create a quantum optimization agent
    optimizer = sqx.QuantumOptimizationAgent(
        backend='simulator',
        problem_type='combinatorial'
    )

    # Create a simple optimization problem (traveling salesman)
    problem_data = {
        'num_cities': 4,
        'distances': np.random.rand(4, 4)
    }

    print("Solving traveling salesman problem...")
    result = optimizer.solve(problem_data)

    if result.result:
        print("✅ Optimization completed!")
        print("Problem Type: combinatorial")
        print(f"Algorithm Used: {result.metadata.get('algorithm_used', 'Unknown')}")
        print(f"Execution Time: {result.execution_time:.2f}s")
    else:
        print(f"❌ Optimization failed: {result.error}")

    print()

def demo_traditional_quantum_ml():
    """Demonstrate traditional quantum ML algorithms for comparison."""
    print("📊 Traditional Quantum ML Demo (For Comparison)")
    print("=" * 50)

    # Generate sample data
    np.random.seed(42)
    X = np.random.rand(100, 4)
    y = np.random.randint(0, 2, 100)

    # Train a quantum SVM
    print("Training Quantum SVM...")
    qsvm = sqx.QuantumSVM(backend='simulator')
    qsvm.fit(X, y)

    # Get predictions
    predictions = qsvm.predict(X[:10])
    accuracy = qsvm.score(X, y)

    print("✅ Quantum SVM trained!")
    print(f"Accuracy: {accuracy:.2%}")
    print(f"Sample predictions: {predictions[:5]}")

    print()

def main():
    """Run all quantum agent demos."""
    print("🚀 SuperQuantX: Quantum Agentic AI Platform Demo")
    print("Building the Foundation for Quantum Agentic AI")
    print("=" * 60)
    print()

    print("This demo showcases autonomous quantum agents that combine")
    print("quantum computing with AI to solve complex real-world problems.")
    print()

    try:
        # Demo quantum agents (the new hotness!)
        demo_quantum_trading_agent()
        demo_quantum_research_agent()
        demo_quantum_optimization_agent()

        # Demo traditional quantum ML (still supported)
        demo_traditional_quantum_ml()

        print("🎯 Summary")
        print("=" * 50)
        print("✅ Quantum Trading Agent: Autonomous financial optimization")
        print("✅ Quantum Research Agent: Scientific discovery acceleration")
        print("✅ Quantum Optimization Agent: General problem solving")
        print("✅ Traditional Quantum ML: Core algorithm support")
        print()
        print("SuperQuantX bridges quantum computing and agentic AI,")
        print("enabling developers to deploy quantum-enhanced intelligent")
        print("systems in minutes, not months.")
        print()
        print("🌟 Ready to build quantum agents? Try:")
        print("   pip install superquantx")
        print("   import superquantx as sqx")
        print("   agent = sqx.QuantumTradingAgent()")

    except Exception as e:
        print(f"❌ Demo failed: {e}")
        print("This is expected if quantum backends are not fully configured.")
        print("The agents are designed to work with actual quantum hardware")
        print("and simulators when properly set up.")

if __name__ == "__main__":
    main()
