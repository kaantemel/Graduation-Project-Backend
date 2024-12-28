
# Backend for Decision Support System

## Overview
This backend application powers the Decision Support System for Route Planning in Last-Mile Delivery Operations. It uses Gurobi optimization software and the Tabu Search algorithm to solve the Vehicle Routing Problem (VRP).

## Technology Stack
- **Programming Language:** Python
- **Framework:** Flask
- **Optimization Software:** Gurobi
- **Algorithm:** Tabu Search

## Getting Started

### Prerequisites
- Python 3.8+
- Flask
- Gurobi (license required)

### Installation
1. Clone the repository:
   ```bash
   git clone <backend-repo-url>
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Set up the Gurobi license on your machine.

### Running the Application
1. Start the server:
   ```bash
   python app.py
   ```
2. The server will be available at `http://localhost:5000`.

## API Endpoints
- `/calculate-route` - POST: Receives input data and returns the optimized route.

## License
[MIT](https://choosealicense.com/licenses/mit/)
