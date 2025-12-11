# ============================================================================
# SCOTIABANK AI INSIGHTS AGENT - INTELLIGENT REASONING v5
# Vector Institute Agent Bootcamp
# ============================================================================
# ARCHITECTURE:
# Each question goes through 5 layers:
# 1. UNDERSTAND - Extract intent and relevant data
# 2. PLAN - Design analysis methodology (like senior analyst)
# 3. CODE - Generate custom Python code for analysis
# 4. EXECUTE - Run code safely and capture results
# 5. INTERPRET - Generate executive-friendly business insights
# ============================================================================

import sys
import os
from pathlib import Path
import json
import pandas as pd
import numpy as np
import gradio as gr
import requests
from dotenv import load_dotenv
from typing import Dict, Any, List

# LangChain
#from langchain_openai import ChatOpenAI
#from langchain_core.messages import HumanMessage
from openai import OpenAI                      # NEW
from dataset_tools import BANK_DATASET_ID      # NEW
from datasets import load_dataset              # NEW

# ============================================================================
# CONFIGURATION
# ============================================================================
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://generativelanguage.googleapis.com/v1beta/openai/")

if not OPENAI_API_KEY:
    print("❌ ERROR: OPENAI_API_KEY not found")
    sys.exit(1)


# Initialize OpenAI-compatible LLMs (no LangChain)
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

llm_flash_model = os.getenv("GEMINI_FLASH_MODEL", "gemini-2.5-flash")
llm_pro_model   = os.getenv("GEMINI_PRO_MODEL", "gemini-2.5-pro")

def llm_flash(prompt: str):
    """Lightweight model call"""
    resp = client.chat.completions.create(
        model=llm_flash_model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    )
    return resp.choices[0].message.content

def llm_pro(prompt: str):
    """High-reasoning model call"""
    resp = client.chat.completions.create(
        model=llm_pro_model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
    )
    return resp.choices[0].message.content
'''
# Initialize LLMs
llm_flash = ChatOpenAI(
    model=os.getenv("GEMINI_FLASH_MODEL", "gemini-2.0-flash-exp"),
    api_key=OPENAI_API_KEY,
    base_url=OPENAI_BASE_URL,
    temperature=0
)

llm_pro = ChatOpenAI(
    model=os.getenv("GEMINI_PRO_MODEL", "gemini-2.0-flash-exp"),
    api_key=OPENAI_API_KEY,
    base_url=OPENAI_BASE_URL,
    temperature=0.2
)
'''

print("✅ Scotiabank AI Insights Agent v5 initialized")

# ============================================================================
# DATA LOADING
# ============================================================================
DATA_CACHE = None

def load_banking_data():
    """Load data with caching"""
    global DATA_CACHE

    if DATA_CACHE is not None:
        return DATA_CACHE

    print("📊 Loading banking data...")
    hf_url = "https://huggingface.co/datasets/mj44442022/dataset_synthetic_v2/resolve/main/banking_data_final_complete_flags(1).csv"
    temp_path = Path("temp_banking_data.csv")

    try:
        if temp_path.exists():
            print("   ✅ Using cached data")
            DATA_CACHE = pd.read_csv(temp_path)
            return DATA_CACHE

        print("   ⬇️ Downloading from HuggingFace...")
        response = requests.get(hf_url, timeout=60)
        response.raise_for_status()

        with open(temp_path, 'wb') as f:
            f.write(response.content)

        DATA_CACHE = pd.read_csv(temp_path)
        print(f"   ✅ Loaded {len(DATA_CACHE):,} rows, {len(DATA_CACHE.columns)} columns")
        return DATA_CACHE

    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return pd.DataFrame()

# ============================================================================
# LAYER 1: UNDERSTAND QUESTION
# ============================================================================

def understand_question(user_question: str, df: pd.DataFrame) -> Dict[str, Any]:
    """Extract intent and identify relevant data columns"""

    print("\n" + "="*60)
    print("🧠 LAYER 1: UNDERSTANDING QUESTION")
    print("="*60)

    cols_info = {
        'numeric': df.select_dtypes(include=[np.number]).columns.tolist(),
        'boolean': [c for c in df.columns if df[c].dtype == 'bool'],
        'categorical': [c for c in df.select_dtypes(include=['object']).columns if df[c].dtype != 'bool']
    }

    prompt = f"""You are a Senior Data Analyst at Scotiabank. A business leader asked:

"{user_question}"

AVAILABLE DATA:
- Numeric metrics: {', '.join(cols_info['numeric'])}
- Boolean flags: {', '.join(cols_info['boolean'])}
- Categorical: {', '.join(cols_info['categorical'])}

TASK: Understand what they're asking for.

Return JSON:
{{
  "intent": "what they want to know (1 sentence)",
  "relevant_columns": ["list", "of", "column", "names"],
  "analysis_type": "comparison | correlation | profiling | segmentation | drivers",
  "business_context": "why this matters to executives"
}}"""

    #response = llm_flash.invoke([HumanMessage(content=prompt)])
    #content = response.content.replace("```json", "").replace("```", "").strip()
    content = llm_flash(prompt)
    try:
        understanding = json.loads(content)
        print(f"✅ Intent: {understanding['intent']}")
        print(f"✅ Relevant columns: {understanding['relevant_columns']}")
        print(f"✅ Analysis type: {understanding['analysis_type']}")
        return understanding
    except:
        return {
            "intent": user_question,
            "relevant_columns": cols_info['numeric'][:5],
            "analysis_type": "profiling",
            "business_context": "General data exploration"
        }

# ============================================================================
# LAYER 2: PLAN ANALYSIS
# ============================================================================

def plan_analysis(understanding: Dict[str, Any], df: pd.DataFrame) -> Dict[str, Any]:
    """Design the analysis methodology like a seasoned analyst would"""

    print("\n" + "="*60)
    print("📋 LAYER 2: PLANNING ANALYSIS")
    print("="*60)

    prompt = f"""You are a Senior Data Analyst with 15 years at Scotiabank. Design the analysis approach.

QUESTION INTENT: {understanding['intent']}
ANALYSIS TYPE: {understanding['analysis_type']}
RELEVANT COLUMNS: {understanding['relevant_columns']}
BUSINESS CONTEXT: {understanding['business_context']}

DATASET: {len(df):,} rows

TASK: Design a clear, methodical analysis plan.

Return JSON:
{{
  "approach": "2-3 sentence methodology description",
  "steps": [
    "Step 1: Specific action",
    "Step 2: Specific action",
    "Step 3: Specific action"
  ],
  "expected_insights": "What we expect to learn",
  "pandas_operations": ["groupby", "pivot", "corr", etc]
}}

Think like an experienced analyst. Be specific and methodical."""

    #response = llm_pro.invoke([HumanMessage(content=prompt)])
    #content = response.content.replace("```json", "").replace("```", "").strip()
    
    content = llm_pro(prompt)

    try:
        plan = json.loads(content)
        print(f"✅ Approach: {plan['approach']}")
        print(f"✅ Steps: {len(plan['steps'])} planned")
        return plan
    except Exception as e:
        print(f"⚠️ Planning error: {e}")
        return {
            "approach": "Standard profiling analysis",
            "steps": ["Calculate summary statistics", "Group by key dimensions"],
            "expected_insights": "Key patterns in the data",
            "pandas_operations": ["describe", "groupby"]
        }

# ============================================================================
# LAYER 3: GENERATE CODE
# ============================================================================

def generate_analysis_code(understanding: Dict[str, Any], plan: Dict[str, Any], df: pd.DataFrame) -> str:
    """Generate custom Python code for this specific analysis"""

    print("\n" + "="*60)
    print("💻 LAYER 3: GENERATING CODE")
    print("="*60)

    # Simplify based on analysis type
    analysis_type = understanding.get('analysis_type', 'profiling')
    relevant_cols = understanding['relevant_columns'][:15]

    if analysis_type == 'profiling':
        # Simple profiling - just show schema
        code = f"""
# Dataset overview
numeric_cols = {[c for c in relevant_cols if c in df.select_dtypes(include=[np.number]).columns.tolist()]}
boolean_cols = {[c for c in relevant_cols if df[c].dtype == 'bool']}
categorical_cols = {[c for c in relevant_cols if c in df.select_dtypes(include=['object']).columns.tolist()]}

results = {{
    'total_rows': len(df),
    'total_columns': len(df.columns),
    'numeric_columns': numeric_cols,
    'boolean_flags': boolean_cols,
    'categorical_columns': categorical_cols
}}
"""
    else:
        # For other types, generate targeted code
        prompt = f"""Write SHORT Python code (max 10 lines) for this analysis.

ANALYSIS TYPE: {analysis_type}
COLUMNS: {relevant_cols}

REQUIREMENTS:
- DataFrame is named 'df'
- Store final result in 'results' (DataFrame or dict)
- THINK CLIENT-CENTRIC: Use .mean() for per-client metrics, .sum() for portfolio totals
- Include BOTH averages (per client) AND counts (portfolio size) when comparing groups
- For comparisons: use df.groupby(['col1', 'col2'])['metric'].agg(['mean', 'count'])
- For correlations: use df[cols].corr()
- Keep it SIMPLE - max 10 lines
- NO functions, NO imports, NO prints

EXAMPLES:

Comparison by country and mortgage (CLIENT-CENTRIC):
results = df.groupby(['country_name', 'has_open_mortgage'])['total_revenues'].agg([('avg_revenue', 'mean'), ('client_count', 'count')]).round(2).reset_index()

Correlation:
results = df[['total_revenues', 'total_loans_balance', 'total_deposit_balance']].corr()

Return ONLY the Python code."""

        #response = llm_flash.invoke([HumanMessage(content=prompt)])
        #code = response.content
        
        content = llm_flash(prompt)

        # Extract code
        if "```python" in code:
            code = code.split("```python")[1].split("```")[0].strip()
        elif "```" in code:
            code = code.split("```")[1].split("```")[0].strip()

    print(f"✅ Generated {len(code)} characters of code")
    print(f"Code:\n{code}")

    return code

# ============================================================================
# LAYER 4: EXECUTE CODE
# ============================================================================

def execute_analysis_code(code: str, df: pd.DataFrame) -> Dict[str, Any]:
    """Safely execute the generated code"""

    print("\n" + "="*60)
    print("⚡ LAYER 4: EXECUTING CODE")
    print("="*60)

    # Create safe execution environment
    exec_globals = {
        'pd': pd,
        'np': np,
        'df': df,
        'results': None
    }

    try:
        # Execute code
        exec(code, exec_globals)
        results = exec_globals.get('results')

        if results is None:
            print("⚠️ No results variable found")
            return {"error": "Code didn't produce 'results' variable"}

        # Convert results to serializable format
        if isinstance(results, pd.DataFrame):
            # Limit rows
            results = results.head(15)
            results_dict = {
                "type": "dataframe",
                "data": results.to_dict('records'),
                "columns": results.columns.tolist(),
                "shape": results.shape,
                "preview": results.to_string()
            }
        elif isinstance(results, dict):
            results_dict = {
                "type": "dict",
                "data": results,
                "preview": json.dumps(results, indent=2, default=str)
            }
        elif isinstance(results, (pd.Series, np.ndarray)):
            results_dict = {
                "type": "series",
                "data": results.to_dict() if hasattr(results, 'to_dict') else str(results),
                "preview": str(results)
            }
        else:
            results_dict = {
                "type": "other",
                "data": str(results),
                "preview": str(results)
            }

        print(f"✅ Execution successful")
        print(f"Results type: {results_dict['type']}")
        print(f"Preview:\n{results_dict['preview'][:300]}...")

        return results_dict

    except Exception as e:
        print(f"❌ Execution error: {e}")
        import traceback
        traceback.print_exc()
        return {"error": str(e), "traceback": traceback.format_exc()}

# ============================================================================
# LAYER 5: INTERPRET & FORMAT INSIGHTS
# ============================================================================

def generate_insights(understanding: Dict[str, Any], plan: Dict[str, Any],
                     execution_results: Dict[str, Any], user_question: str) -> str:
    """Generate executive-friendly business insights"""

    print("\n" + "="*60)
    print("📊 LAYER 5: GENERATING INSIGHTS")
    print("="*60)

    if "error" in execution_results:
        return f"⚠️ I encountered an issue analyzing that: {execution_results['error']}\n\nCould you rephrase your question?"

    results_preview = execution_results.get('preview', '')
    results_type = execution_results.get('type', 'unknown')

    # For simple profiling, format differently
    if results_type == 'dict' and 'total_rows' in str(results_preview):
        data = execution_results.get('data', {})
        output = f"""### 📊 Dataset Overview

**{data.get('total_rows', 0):,} customers** across **{data.get('total_columns', 0)} data points**

**Numeric Metrics** ({len(data.get('numeric_columns', []))}):
{', '.join(data.get('numeric_columns', [])[:10])}{"..." if len(data.get('numeric_columns', [])) > 10 else ""}

**Customer Flags** ({len(data.get('boolean_flags', []))}):
{', '.join(data.get('boolean_flags', [])[:10])}{"..." if len(data.get('boolean_flags', [])) > 10 else ""}

**Segmentation** ({len(data.get('categorical_columns', []))}):
{', '.join(data.get('categorical_columns', []))}

💡 *Ask follow-up questions to analyze specific metrics or compare customer segments.*"""
        return output

    # For analysis results, generate insights
    prompt = f"""You are presenting to Scotiabank executives who need ACTIONABLE INSIGHTS.

QUESTION: "{user_question}"

ANALYSIS RESULTS:
{results_preview}

TASK: Write executive-friendly insights.

FORMAT:

**Methodology**: [1 sentence: what analysis was done]

### Key Findings
- **Finding 1**: [Insight with numbers and % difference - per client average]
- **Finding 2**: [Insight with numbers - portfolio level when relevant]
- **Finding 3**: [Insight comparing segments]

### Business Implication
[What this means and what to do about it]

RULES - CRITICAL:
- Start with brief methodology (e.g., "Analyzed average revenue per client across X segments")
- NEVER use casual language ("Okay", "Here's", "Let me", "So")
- Be PROFESSIONAL and DIRECT
- Report AVERAGES for client-level insights (how a typical client looks)
- Report TOTALS/COUNTS for portfolio-level context (how many clients, total value)
- Calculate and highlight DIFFERENCES (X is Y% higher than Z)
- Use business language (clients, average revenue per client, not rows)
- Keep it BRIEF (3-5 bullets max)
- Focus on SO WHAT not WHAT

BAD: "Okay, here's the analysis. total_revenues: 230.66"
GOOD: "**Methodology**: Compared average revenue per client by payroll status.

**Non-payroll clients generate $231 average revenue** (9% higher than payroll clients at $212 avg)"
"""

    #response = llm_pro.invoke([HumanMessage(content=prompt)])
    #insights = response.content

    content = llm_pro(prompt)

    print(f"✅ Generated {len(insights)} characters of insights")

    return insights

# ============================================================================
# MAIN CONVERSATIONAL AGENT
# ============================================================================

CONVERSATION_HISTORY = []
MAX_HISTORY = 8

def conversational_agent(user_input: str, history):
    """Main agent with 5-layer intelligent reasoning"""
    global CONVERSATION_HISTORY

    # Load data
    df = load_banking_data()
    if df.empty:
        yield "❌ Error: Could not load data"
        return

    # Memory management
    if len(CONVERSATION_HISTORY) >= MAX_HISTORY:
        summary = f"📝 **Memory limit reached.** Starting fresh conversation."
        CONVERSATION_HISTORY = []
        yield summary

    try:
        # Layer 1: Understand
        understanding = understand_question(user_input, df)

        # Layer 2: Plan
        plan = plan_analysis(understanding, df)

        # Layer 3: Generate code
        code = generate_analysis_code(understanding, plan, df)

        # Layer 4: Execute
        execution_results = execute_analysis_code(code, df)

        # Layer 5: Interpret & Present
        insights = generate_insights(understanding, plan, execution_results, user_input)

        # Store in history
        CONVERSATION_HISTORY.append({
            "user": user_input,
            "agent": insights
        })

        yield insights

    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        yield f"⚠️ I encountered an unexpected error. Please try rephrasing your question.\n\nError details logged to console."

# ============================================================================
# GRADIO INTERFACE
# ============================================================================

demo = gr.ChatInterface(
    fn=conversational_agent,
    title="🏦 Scotiabank AI Insights Agent v5 - Intelligent Reasoning",
    description="""
    **5-Layer Intelligent Analysis**

    Each question goes through:
    1. 🧠 Understanding - What are you asking?
    2. 📋 Planning - Design the methodology
    3. 💻 Coding - Generate custom analysis
    4. ⚡ Execution - Run the analysis
    5. 📊 Insights - Executive-friendly results

    **Try asking:**
    - "What data is available?"
    - "How do clients with payroll compare to those without?"
    - "Does mortgage ownership affect revenue by country?"
    """,
    examples=[
        "What information is available?",
        "Compare clients with payroll vs without payroll",
        "How does mortgage ownership affect revenue by country?",
        "Show me correlation between revenue and loans"
    ],
    type="messages"
)

# ============================================================================
# SELF-TEST
# ============================================================================

def run_self_test():
    """Test the system before launching"""
    print("\n" + "="*80)
    print("🧪 RUNNING SELF-TEST")
    print("="*80)

    test_questions = [
        "What information is available?",
        "How does mortgage ownership affect revenue by country?"
    ]

    df = load_banking_data()
    if df.empty:
        print("❌ SELF-TEST FAILED: Could not load data")
        return False

    for i, question in enumerate(test_questions, 1):
        print(f"\n--- Test {i}/{len(test_questions)} ---")
        print(f"Question: {question}")

        try:
            # Test each layer
            understanding = understand_question(question, df)
            plan = plan_analysis(understanding, df)
            code = generate_analysis_code(understanding, plan, df)
            results = execute_analysis_code(code, df)
            insights = generate_insights(understanding, plan, results, question)

            if "error" in results or len(insights) < 50:
                print(f"❌ Test {i} FAILED")
                return False

            print(f"✅ Test {i} PASSED")
            print(f"Insight preview: {insights[:150]}...")

        except Exception as e:
            print(f"❌ Test {i} FAILED: {e}")
            return False

    print("\n" + "="*80)
    print("✅ ALL SELF-TESTS PASSED")
    print("="*80)
    return True

# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    print("🚀 Scotiabank AI Insights Agent v5")
    print("   5-layer intelligent reasoning system")
    print("   Designed for executive-level business insights\n")

    # Run self-test
    if run_self_test():
        print("\n🌐 Launching Gradio interface...")
        demo.launch(server_name="0.0.0.0", server_port=7865, share=True)
    else:
        print("\n❌ Self-test failed. Fix issues before launching.")
        sys.exit(1)

def run_mj_scotia(question: str) -> str:
    """
    Wrapper: runs the full 5-layer MJ_SCOTIA reasoning pipeline
    and returns final insights as a single string.
    """
    df = load_banking_data()

    understanding = understand_question(question, df)
    plan = plan_analysis(understanding, df)
    code = generate_analysis_code(understanding, plan, df)
    execution = execute_analysis_code(code, df)
    insights = generate_insights(understanding, plan, execution, question)

    return insights
