"""Business Insight Worker Agent - Layers 2-5 ONLY.

This worker agent handles the core business insight generation process:
- Layer 2: PLAN - Design analysis methodology
- Layer 3: CODE - Generate custom Python code
- Layer 4: EXECUTE - Run code safely and capture results
- Layer 5: INTERPRET - Generate executive-friendly insights

NOTE: Layer 1 (intent understanding) is handled by the Main Planner.
This worker focuses on execution only.
"""

import os
import json
import pandas as pd
import numpy as np
from typing import Dict, Any
from dotenv import load_dotenv

'''
# LangChain for LLM operations
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

# Import shared data loading
from multi_agent.data_overview_agent import load_banking_data
'''

from openai import OpenAI                      # NEW
from dataset_tools import BANK_DATASET_ID      # NEW
from datasets import load_dataset              # NEW

# ============================================================================
# CONFIGURATION
# ============================================================================
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://generativelanguage.googleapis.com/v1beta/openai/")

# Initialize LLMs

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


# ============================================================================
# MINIMAL COLUMN IDENTIFICATION (Internal Helper)
# ============================================================================

def identify_relevant_columns(user_question: str, df: pd.DataFrame) -> Dict[str, Any]:
    """Lightweight column identification (not full intent understanding).

    This is a minimal helper to identify which columns are relevant
    for the analysis. Intent understanding is done by the Main Planner.

    Args:
        user_question: The business question
        df: The dataframe

    Returns:
        Dict with relevant columns and basic analysis type
    """
    # Get column categories
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    boolean_cols = [c for c in df.columns if df[c].dtype == 'bool']
    categorical_cols = [c for c in df.select_dtypes(include=['object']).columns if df[c].dtype != 'bool']

    # Simple keyword-based column identification
    question_lower = user_question.lower()

    relevant_cols = []

    # Check for mentioned columns
    for col in df.columns:
        if col.lower().replace('_', ' ') in question_lower:
            relevant_cols.append(col)

    # Identify analysis type from keywords
    if any(word in question_lower for word in ['compare', 'comparison', 'vs', 'versus', 'between']):
        analysis_type = 'comparison'
    elif any(word in question_lower for word in ['correlate', 'correlation', 'relationship', 'related']):
        analysis_type = 'correlation'
    elif any(word in question_lower for word in ['segment', 'group', 'category', 'breakdown']):
        analysis_type = 'segmentation'
    else:
        analysis_type = 'profiling'

    # If no specific columns identified, use numeric columns
    if not relevant_cols:
        relevant_cols = numeric_cols[:10]

    return {
        "relevant_columns": relevant_cols,
        "analysis_type": analysis_type,
        "all_numeric": numeric_cols,
        "all_boolean": boolean_cols,
        "all_categorical": categorical_cols
    }


# ============================================================================
# LAYER 2: PLAN ANALYSIS
# ============================================================================

def plan_analysis(user_question: str, column_info: Dict[str, Any], df: pd.DataFrame) -> Dict[str, Any]:
    """Design the analysis methodology like a seasoned analyst would.

    Args:
        user_question: The business question
        column_info: Information about relevant columns
        df: The dataframe

    Returns:
        Analysis plan with approach, steps, and expected insights
    """
    print("\n" + "="*60)
    print("📋 LAYER 2: PLANNING ANALYSIS")
    print("="*60)

    prompt = f"""You are a Senior Data Analyst with 15 years at Scotiabank. Design the analysis approach.

QUESTION: {user_question}
ANALYSIS TYPE: {column_info['analysis_type']}
RELEVANT COLUMNS: {column_info['relevant_columns']}

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

def generate_analysis_code(user_question: str, column_info: Dict[str, Any],
                          plan: Dict[str, Any], df: pd.DataFrame) -> str:
    """Generate custom Python code for this specific analysis.

    Args:
        user_question: The business question
        column_info: Information about relevant columns
        plan: The analysis plan
        df: The dataframe

    Returns:
        Python code as a string
    """
    print("\n" + "="*60)
    print("💻 LAYER 3: GENERATING CODE")
    print("="*60)

    analysis_type = column_info.get('analysis_type', 'profiling')
    relevant_cols = column_info['relevant_columns'][:15]

    # Generate targeted code based on analysis type
    prompt = f"""Write SHORT Python code (max 10 lines) for this analysis.

QUESTION: {user_question}
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
    code = llm_flash(prompt)

    # Extract code
    if "```python" in code:
        code = code.split("```python")[1].split("```")[0].strip()
    elif "```" in code:
        code = code.split("```")[1].split("```")[0].strip()

    # 🔧 HOTFIX: map generic 'revenue' → actual column name
    if "revenue" in code and "total_revenues" in df.columns and "total_revenues" not in code:
        code = code.replace("revenue", "total_revenues")

    print(f"✅ Generated {len(code)} characters of code")
    print(f"Code:\n{code}")

    return code



# ============================================================================
# LAYER 4: EXECUTE CODE
# ============================================================================

def execute_analysis_code(code: str, df: pd.DataFrame) -> Dict[str, Any]:
    """Safely execute the generated code.

    Args:
        code: Python code to execute
        df: The dataframe

    Returns:
        Execution results with type, data, and preview
    """
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

def generate_insights(plan: Dict[str, Any], execution_results: Dict[str, Any],
                     user_question: str) -> str:
    """Generate executive-friendly business insights.

    Args:
        plan: The analysis plan
        execution_results: Results from code execution
        user_question: The original business question

    Returns:
        Formatted insights as markdown string
    """
    print("\n" + "="*60)
    print("📊 LAYER 5: GENERATING INSIGHTS")
    print("="*60)

    if "error" in execution_results:
        return f"⚠️ I encountered an issue analyzing that: {execution_results['error']}\n\nCould you rephrase your question?"

    results_preview = execution_results.get('preview', '')

    # Generate insights
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

    insights = llm_pro(prompt)


    print(f"✅ Generated {len(insights)} characters of insights")

    return insights


# ============================================================================
# MAIN WORKER FUNCTION
# ============================================================================

def analyze_business_insight(question: str) -> str:
    """Main worker function that runs layers 2-5 of analysis.

    This function is exposed as a tool to the main planner agent.
    It assumes the planner has already understood the intent.

    Args:
        question: User's business question about the banking data

    Returns:
        Executive-friendly business insights as a formatted string
    """
    print("\n" + "="*80)
    print("🏦 BUSINESS INSIGHT WORKER AGENT - STARTING ANALYSIS")
    print("="*80)
    print(f"Question: {question}\n")

    # Load data
    ds = load_dataset(BANK_DATASET_ID)
    split_name = "train" if "train" in ds else list(ds.keys())[0]
    df = ds[split_name].to_pandas()

    if df.empty:
        return "❌ Error: Could not load banking data. Please check data source."

    try:
        # Minimal column identification (internal, not full Layer 1)
        column_info = identify_relevant_columns(question, df)
        print(f"📝 Identified columns: {column_info['relevant_columns'][:5]}...")
        print(f"📝 Analysis type: {column_info['analysis_type']}")

        # Layer 2: Plan
        plan = plan_analysis(question, column_info, df)

        # Layer 3: Generate code
        code = generate_analysis_code(question, column_info, plan, df)

        # Layer 4: Execute
        execution_results = execute_analysis_code(code, df)

        # Layer 5: Interpret & Present
        insights = generate_insights(plan, execution_results, question)

        print("\n" + "="*80)
        print("✅ BUSINESS INSIGHT WORKER AGENT - ANALYSIS COMPLETE")
        print("="*80)

        return insights

    except Exception as e:
        print(f"\n❌ ERROR in worker agent: {e}")
        import traceback
        traceback.print_exc()
        return f"⚠️ I encountered an unexpected error during analysis.\n\nError: {str(e)}\n\nPlease try rephrasing your question or contact support."