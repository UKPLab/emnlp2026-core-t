"""Prompt strings used by CORE-T scripts."""

from __future__ import annotations

from typing import Any, Dict, List

purpose_prompt = (
    "Given the following table, describe the purpose of this table in layman's terms in one paragraph. "
    "If you do not think the text is semantically meaningful, output None.\n{table}"
)

summary_prompt = (
    "Given the following table, summarize this table in layman's terms in one paragraph. "
    "If you do not think the text is semantically meaningful, output None.\n{table}"
)

qa_prompt = (
    "Given the following table, generate at most 20 distinct question-answer pairs on this table that includes both "
    "simple questions and those requiring summarization or aggregation. The questions should be phrased in layman's "
    "terms, using explicit or implicit knowledge from the table. The question and answer should avoid using exact "
    "terms, such as shortened forms, from the table but can instead use naturally phrased language. Each question-answer "
    "pair should be formatted as a list where the first element is the question and the second element is the answer. "
    "The output should be a list of lists in JSON format. If the table is empty or you do not think the text is "
    "semantically meaningful, output an empty list: []\n{table}"
)

def get_selection_prompt_lrm_few_shot(query: str, tables_content: str, compatibility_analysis: str) -> str:
    """Generate the combined validation and selection prompt with query and tables context."""
    return f"""You are a SQL schema analyst.
Your task: From a set of retrieved tables, identify a comprehensive set of tables that are BOTH:
(1) Relevant to the given query, and  
(2) Compatible (joinable) with each other to answer the query.

IMPORTANT:  
- Do NOT aggressively eliminate tables.  
- If there is a reasonable probability that a table is relevant and compatible, keep it.  
- When uncertain, prefer to keep the table rather than remove it — it is better to have slightly more tables than to risk removing a necessary one.  
- Only remove a table if it is clearly irrelevant or incompatible.

---

### Information Provided:
- **Query**: {query}
- **Tables**: {tables_content}  
  Each table includes:
  - Table name
  - Table header and sample content in markdown format (5 rows)
- **Compatibility analysis (restricted to valid key–foreign key pairs)**: {compatibility_analysis}  
  For each pair of tables, compatibility scores are included **only if** at least one column of the first table is completely unique and at least one column of the second table is a subset of it.  
  If no such relationship exists, that pair is omitted (since all scores would be zero).  

  For included pairs, the following metrics are provided:
    - `overall_compatibility`: Highest weighted score between all possible column pairs that satisfy the constraint: one column is unique, the other is a subset of it.
    - `best_join_columns`: The specific column pair with the highest overall compatibility score.

---

### Step-by-step reasoning policy (YOU MUST FOLLOW THIS ORDER):

**Step 1 – Understand the query**  
- Identify the core entities and relationships.  
- Determine what type of data is required to answer it.

**Step 2 – Evaluate individual table relevance**  
- Use table name, column names, and sample data to decide if each table is relevant.  
- When unsure, treat the table as potentially relevant.

**Step 3 – Evaluate pairwise compatibility**  
For each pair of retrieved tables:  
- Interpret the compatibility scores.  
- Cross-check with table semantics from names, sample values.  
- When in doubt about compatibility, keep the pair as potentially relevant.

**Step 4 – Group formation**  
- Form one or more groups of tables where all members are mutually joinable.  
- Groups must form connected join graphs (no isolated tables).  
- Prefer forming larger groups when there is uncertainty rather than splitting unnecessarily.

**Step 5 – Group selection**  
- Select the single most relevant and compatible group for the query.  
- High recall is as important as precision in this step — include tables that are possibly relevant to ensure coverage.

---

### Output Format:
Return the output as valid JSON in the following format:

{{
  "overall_reasoning": "Your general approach and observations about the tables and query",
  "group_formation": {{
    "reasoning": "How groups were formed based on provided quantitative and qualitative information",
    "groups_formed": [
      {{
        "group_index": 0,
        "table_indices": [0, 1, 2],
        "group_description": "Description of what this group represents"
      }}
    ]
  }},
  "group_selection": {{
    "selected_group_index": 0,
    "reasoning": "Detailed explanation of why this group was selected for the query",
    "group_analysis": [
      {{
        "group_index": 0,
        "reasoning": "Why this group is/isn't suitable for the query"
      }}
    ]
  }}
}}

---

### Few-shot Example

**Example Input**:
Query:
"In campaigns with exactly 2 events, how many of the events have clicks equal to 0?"

Tables:
Table 0:
Table name: campaigns  
Example table content:
| campaign_id | owner_id | name              | created_at           | event_count |
|------------:|---------:|-------------------|----------------------|------------:|
| 10          | 1        | Winter Launch     | 2024-01-05 10:00:00  | 2           |
| 11          | 2        | Spring Promo      | 2024-02-10 09:30:00  | 1           |
| 12          | 1        | Summer Teaser     | 2024-03-01 12:15:00  | 2           |

Table 1:
Table name: campaign_events  
Example table content:
| event_id | campaign_id | event_type | clicks | impressions | created_at           |
|---------:|------------:|-----------|-------:|------------:|----------------------|
| 100      | 10          | email     | 0      | 500         | 2024-01-05 10:05:00  |
| 101      | 10          | banner    | 12     | 1000        | 2024-01-05 10:06:00  |
| 102      | 11          | email     | 5      | 300         | 2024-02-10 09:35:00  |
| 103      | 12          | social    | 0      | 800         | 2024-03-01 12:20:00  |
| 104      | 12          | banner    | 7      | 900         | 2024-03-01 12:21:00  |

Table 2:
Table name: cities  
Example table content:
| city_id | name    | country | population |
|--------:|---------|---------|-----------:|
| 1       | Berlin  | DE      | 3600000    |
| 2       | Munich  | DE      | 1500000    |
| 3       | Hamburg | DE      | 1800000    |

Compatibility analysis:
Pair (Table 0 <-> Table 1):
  overall_compatibility: 0.96
  best_join_columns: "campaign_id ↔ campaign_id"

**Example Output**:
{{
  "overall_reasoning": "The query is about campaigns and their events. The 'campaigns' table holds campaign-level data including event_count, while 'campaign_events' holds per-event data including clicks and campaign_id for linking. The 'cities' table is unrelated to the query and has no compatible join key with the other tables.",
  "group_formation": {{
    "reasoning": "Formed one group with 'campaigns' and 'campaign_events' because they are both relevant to the query and strongly joinable via campaign_id ↔ campaign_id. 'cities' is excluded due to lack of relevance and join compatibility.",
    "groups_formed": [
      {{
        "group_index": 0,
        "table_indices": [0, 1],
        "group_description": "Campaigns and their associated events, enabling filtering by event_count and counting events with clicks = 0."
      }}
    ]
  }},
  "group_selection": {{
    "selected_group_index": 0,
    "reasoning": "This group contains all and only the tables needed to answer the query: campaigns to identify those with exactly 2 events, and campaign_events to count events with clicks equal to 0.",
    "group_analysis": [
      {{
        "group_index": 0,
        "reasoning": "Fully suitable and sufficient for the query; no other table contributes necessary information."
      }}
    ]
  }}
}}"""

def get_sql_generation_prompt_lrm(query: str, table_schemas: Dict[str, Dict[str, str]], table_relationships: List[Dict[str, Any]] = None, column_value_hints: Dict[str, Dict[str, List[Any]]] = None, evidence: str = "") -> str:
    """Generate the SQL generation prompt with query and database schema context.

    `column_value_hints` optionally provides, for each table, per-column example values
    as a list (e.g., top-5 most frequent non-null values). Structure:
      { table_name: { column_name: [v1, v2, ...] } }
    """
    
    # Format table schemas for the prompt
    schema_text = ""
    for table_name, columns in table_schemas.items():
        schema_text += f"\nTable: {table_name}\n"
        for col_name, col_type in columns.items():
            hint_str = ""
            if column_value_hints and table_name in column_value_hints and col_name in column_value_hints[table_name]:
                hints = column_value_hints[table_name][col_name]
                # Render hints as a compact list, quoting strings
                def _render(v: Any) -> str:
                    if isinstance(v, str):
                        return f"'{v}'"
                    return str(v)
                hint_items = ", ".join(_render(v) for v in hints)
                hint_str = f" [values: {hint_items}]"
            schema_text += f"  - {col_name}: {col_type}{hint_str}\n"
    
    # Format relationships if provided
    relationships_text = ""
    if table_relationships:
        relationships_text = "\nTABLE RELATIONSHIPS:\n"
        for rel in table_relationships:
            relationships_text += f"  - {rel.get('from_table', 'unknown')}.{rel.get('from_column', 'unknown')} → {rel.get('to_table', 'unknown')}.{rel.get('to_column', 'unknown')}\n"
    
    return f"""You are an expert SQL query generator. Given the following tables and a natural language question, generate a precise SQL query that answers the question. The target dialect is SQLite.

AVAILABLE TABLES:
{schema_text}

External knowledge: {evidence}

QUESTION: {query}

INSTRUCTIONS:
1. Analyze the question carefully to understand what information is being requested.
2. Identify which tables and columns are needed from the available tables. Use only the provided tables and columns; never invent schema or values. Treat sample values in the schema as illustrative, not exhaustive.
3. Generate a syntactically correct SQL query that answers the question.
4. Use explicit JOIN ... ON ... with appropriate JOINs when multiple tables are needed; avoid cartesian products.
5. Apply proper filtering, grouping, ordering, LIMIT, DISTINCT, HAVING, and aggregates as required.
6. Use the exact column names as shown in the schema; qualify ambiguous column names with table names or aliases.
7. Be careful with column names that contain spaces—use backticks or quotes as needed.
8. Return ONLY the SQL query without any explanation, comments, or markdown formatting.

SQL QUERY:"""
