trigger_connect.py talks to knowledgebase.py, dataset_tools.py, and dataset_definitions.py.

knowledgebase.py internally uses kb_weaviate.py + info_retrieval_prompt.py.

---

# 👥 Team Responsibilities

### **Jerry & Fergie – Orchestrator & System Integration**
**File:** `trigger_connect.py`  (Jerry combine later, use jerry prompt for explain answer and ask proactive ask)
- Main entry point  
- Loads all tools  
- Connects Trigger Agent → Planner → Tools  
- Defines system + role-adaptive prompts  
- Runs the Gradio interface  

---

### **Jessie – Column Definitions + Retrieval (RAG)**
#### A. Column Definitions  
**File:** `dataset_definitions.py`  
- Loads the “column definition” dataset  
- Implements:  
  - `get_column_definition(column_name)`  

#### B. Retrieval-Augmented Search (RAG)  
**Files:**  
- `kb_weaviate.py`  
- `knowledgebase.py`  
- `info_retrieval_prompt.py`  

**Purpose:**  
- Provide semantic search across knowledgebase  
- Support planner reasoning  
- Allow natural language → structured query transformation  

---

### **Sutha – Dataset Metrics & Calculations**  
**File:** `dataset_tools.py`  
- Loads dataset (`mj44442022/dataset_synthetic_v2`)  
- Implements tools:  
  - `get_dataset_columns()`  
  - `count_active_clients_last_month()`  
  - `build_dataset_summary()`  
- Future expansion: churn, balances by segment, trends

---

### **MJ – External Search Agent** 
(Combine later on)
- Search Agent for real-world information  
- Integrated into `trigger_connect.py` via `.as_tool()`

---

# 🔌 Tool Integration

Inside **`trigger_connect.py`**, all teammate modules are registered:

```python
tools = [
    search_agent.as_tool(...),     # MJ
    dataset_columns_tool,          # Sutha
    active_clients_tool,           # Sutha
    column_definition_tool,        # Jessie
    # Future: RAG search tool
]
