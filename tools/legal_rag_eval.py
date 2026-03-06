import os
import json
import re
import csv
import numpy as np
from langchain_groq import ChatGroq
from tools.legal_rag import legal_chat_flow 

def judge_answer(query, system_answer, expected_answer, llm):
    """
    Evaluates the system's answer against the expected golden answer.
    """
    judge_prompt = f"""
    <system>
        You are a strict Indian Legal Auditor. 
        Evaluate the System Answer based ONLY on Indian Law (BNS, IPC, etc.).
    </system>
    
    <query>{query}</query>
    <expected_golden_answer>{expected_answer}</expected_golden_answer>
    <system_answer>{system_answer}</system_answer>
    
    <instructions>
        Provide a structured evaluation in this EXACT format:
        Domain: [Yes/No]
        Factual: [Yes/No]
        Accuracy: [Score between 0.0 and 1.0]
        Rank: [Integer 1 to 5, where 1 is the most relevant legal match]
        Reason: [One sentence explanation]
    </instructions>
    """
    try:
        judgment = llm.invoke(judge_prompt).content
        return judgment
    except Exception as e:
        return f"Judge Error: {str(e)}"

def aggregate_metrics(results):
    """
    Calculates Accuracy, Domain Relevance, Precision, Recall, and MRR.
    """
    total = len(results)
    if total == 0: return {}

    domain_correct = 0
    factual_correct = 0
    accuracy_scores = []
    true_positives = 0  
    false_positives = 0 
    reciprocal_ranks = []
    
    for r in results:
        text = r["judgment"].lower()
        is_domain = "domain: yes" in text
        if is_domain: domain_correct += 1
        if "factual: yes" in text: factual_correct += 1
        
        acc_match = re.search(r"accuracy[: ]*(\d?\.?\d+)", text)
        acc = 0.0
        if acc_match:
            acc = min(float(acc_match.group(1)), 1.0)
            accuracy_scores.append(acc)

        if is_domain:
            if acc >= 0.7:
                true_positives += 1
            else:
                false_positives += 1

        rank_match = re.search(r"rank[: ]*(\d+)", text)
        if rank_match:
            rank = int(rank_match.group(1))
            if acc >= 0.5:
                reciprocal_ranks.append(1.0 / rank)
            else:
                reciprocal_ranks.append(0.0)
        else:
            reciprocal_ranks.append(0.0)

    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / total if total > 0 else 0
    mrr = np.mean(reciprocal_ranks) if reciprocal_ranks else 0.0

    return {
        "total_queries": total,
        "domain_relevance_rate": domain_correct / total,
        "factual_grounding_rate": factual_correct / total,
        "average_accuracy": sum(accuracy_scores) / len(accuracy_scores) if accuracy_scores else 0.0,
        "precision": precision,
        "recall": recall,
        "mrr": mrr
    }

def add_golden_entry(query, expected, golden_file="golden.json"):
    """
    Appends a single Q&A pair to the golden file if the query is unique.
    """
    data = []
    if os.path.exists(golden_file):
        with open(golden_file, "r", encoding="utf-8") as f:
            try:
                data = json.load(f)
            except:
                data = []

    if any(item.get("query") == query for item in data):
        return False

    data.append({"query": query, "expected": expected})
    with open(golden_file, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    return True

def import_from_nested_history(history_data, golden_file="golden.json"):
    """
    Parses the specific nested UUID chat history format into golden.json.
    """
    added_count = 0
    for session_id in history_data:
        messages = history_data[session_id].get("messages", [])
        for i in range(len(messages) - 1):
            if messages[i]["role"] == "user" and messages[i+1]["role"] == "assistant":
                query = messages[i]["content"].strip()
                expected = messages[i+1]["content"].strip()
                if add_golden_entry(query, expected, golden_file):
                    added_count += 1
    return added_count

def run_evaluation_cycle(llm, golden_file="golden.json"):
    """
    Executes the full RAG evaluation pipeline.
    """
    if not os.path.exists(golden_file):
        print(f"❌ Error: {golden_file} not found.")
        return None

    # ✅ Force UTF-8 decoding
    with open(golden_file, "r", encoding="utf-8") as f:
        golden_data = json.load(f)

    results = []
    for entry in golden_data:
        q = entry['query']
        exp = entry['expected']
        print(f"⚖️ Testing: {q[:50]}...")
        sys_ans = legal_chat_flow(q)
        judgment = judge_answer(q, sys_ans, exp, llm)
        results.append({
            "query": q,
            "system_answer": sys_ans,
            "expected": exp,
            "judgment": judgment
        })

    metrics = aggregate_metrics(results)

    # ✅ Also force UTF-8 for CSV output
    with open("eval_results.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["query", "system_answer", "expected", "judgment"])
        writer.writeheader()
        writer.writerows(results)
        
    return metrics