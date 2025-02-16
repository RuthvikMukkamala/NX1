
# **NX1 Case Study: SEC 8-K Event Classification Pipeline**

## **Overview**
**LLM-powered event classification system** designed to process **SEC 8-K filings** and assess their impact on financial markets. The system uses **multiple reasoning techniques**, and **self-refinement mechanisms** to enhance classification accuracy. 

The architecture integrates **Comet (Opik)** for real-time model evaluation and experiment tracking, ensuring continuous improvement in classification performance.

---

## **Architecture Overview**

### **SEC 8-K Data Retrieval**
- Retrieves and processes **SEC 8-K filings** to extract market-moving events.
- Supports **specific ticker-based retrieval** - just add a ticker, start/end date to automatically retrieve the 8-K files.
- Uses FinnHub for 8-K link retrival.
- Initially, filings are stored **locally for development**, with a future building block to **AWS S3 for scalable access**.

### **2️⃣ Event Classification Models**
I implement a range of **LLM-powered classification models**, each optimized for different reasoning approaches:

| **Methodology**      | **Purpose**                                                                                      |
|----------------------|--------------------------------------------------------------------------------------------------|
| **Zero-Shot**        | Basic classification without prior examples.                                                     |
| **Few-Shot**         | Leverages examples to improve classification accuracy.                                           |
| **Chain-of-Thought** | Step-by-step logical reasoning for complex events.                                               |
| **Self-Consistency** | Uses multiple reasoning paths to establish consensus.                                            |
| **Tree-of-Thought**  | Expands on parallel reasoning structures.                                                        |
| **Self-Refinement**  | Iteratively improves classifications based on prior outputs.                                     |
| **Decomposition**    | Breaks complex events into granular sub-events for better categorization. (Still in development) |
| **ReAct**            | Reason and Act Model - Use system context to provide a better final assessment                   |

---

### **3️⃣ Model Evaluation and Performance Tracking**
I implement **DSPy’s assessment framework** to ensure high classification accuracy:

- **Significance Validation**:  
  - Utilizes an **Assess module** to verify classification correctness.
  - Constructs evaluation questions such as:  
    *"Is the event 'Acquisition' correctly classified as 'Significant'?"*
  
- **Majority Voting Mechanism**:  
  - Aggregates multiple classification outputs to determine the most accurate event significance.

- **Real-Time Experiment Tracking with Comet (Opik)**:  
  - Logs model decisions, prompting strategies, and classification results.
  - Allows continuous monitoring and comparison of different classification techniques.

---

## ** Setup and Execution**

### **Create a Conda Environment**
Ensure that you have **Anaconda** or **Miniconda** installed before proceeding.

```sh
conda env create -f environment.yml
conda activate nx1
