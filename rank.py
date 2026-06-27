import argparse
import gzip
import json
import csv
import heapq
import re
from datetime import datetime

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", required=True, help="Path to candidates.jsonl")
    parser.add_argument("--out", required=True, help="Output CSV path")
    return parser.parse_args()

def is_honeypot(candidate):
    skills = candidate.get("skills", [])
    # 1. Expert proficiency with 0 months used
    expert_zero_months = sum(1 for s in skills if s.get("proficiency") in ["expert", "advanced"] and s.get("duration_months", 1) == 0)
    if expert_zero_months >= 3:
        return True
    
    # 2. Career history doesn't match total YOE
    history = candidate.get("career_history", [])
    total_months = sum(job.get("duration_months", 0) for job in history)
    prof_yoe = candidate.get("profile", {}).get("years_of_experience", 0)
    
    if prof_yoe > 0 and (total_months / 12.0) > (prof_yoe + 5):
        return True
    
    if (total_months / 12.0) > 0 and prof_yoe > ((total_months / 12.0) + 15):
        return True
        
    return False

def score_candidate(candidate):
    score = 0.0
    reasoning_points = []
    
    profile = candidate.get("profile", {})
    history = candidate.get("career_history", [])
    signals = candidate.get("redrob_signals", {})
    skills = candidate.get("skills", [])
    
    # Text pool for quick matching
    full_text = (
        profile.get("headline", "") + " " +
        profile.get("summary", "") + " " +
        " ".join([job.get("description", "") + " " + job.get("title", "") for job in history])
    ).lower()
    
    # 1. Reject criteria
    consulting_firms = {"tcs", "infosys", "wipro", "accenture", "cognizant", "capgemini"}
    if history:
        all_consulting = all(any(c in job.get("company", "").lower() for c in consulting_firms) for job in history)
        if all_consulting:
            return 0, "Rejected: Consulting only."
            
    # Pure research check
    if history:
        current_job = history[0]
        if "research" in current_job.get("title", "").lower() and "engineer" not in current_job.get("title", "").lower():
            score -= 1.0
            
    # YOE check (5-9 years preferred, but flexible)
    yoe = profile.get("years_of_experience", 0)
    if 4 <= yoe <= 10:
        score += 1.0
        reasoning_points.append(f"{yoe} years of experience fits the optimal window.")
    elif yoe > 10:
        score += 0.5
    else:
        score -= 0.5

    # 2. Skills Match (from JD)
    # Embeddings/Retrieval
    if any(k in full_text for k in ["embedding", "retrieval", "sentence-transformers", "sentence transformer", "openai", "bge", "e5"]):
        score += 2.0
        reasoning_points.append("Has strong embeddings and retrieval background.")
        
    # Vector DBs
    if any(k in full_text for k in ["pinecone", "weaviate", "qdrant", "milvus", "opensearch", "elasticsearch", "faiss"]):
        score += 2.0
        reasoning_points.append("Experience with production vector databases/search.")
        
    # Evaluation
    if any(k in full_text for k in ["ndcg", "mrr", "map", "a/b test", "evaluation framework", "offline benchmark"]):
        score += 1.5
        reasoning_points.append("Familiar with rigorous ranking evaluation metrics.")
        
    # Python
    if any(s.get("name", "").lower() == "python" for s in skills) or "python" in full_text:
        score += 1.0
    
    # Product/Shipper mindset vs just frameworks
    if "langchain" in full_text and score < 2.0:
        score -= 0.5 # Penalize if langchain is the only AI signal
        
    # 3. Behavioral Signals Multipliers
    response_rate = signals.get("recruiter_response_rate", 0.5)
    score *= (0.5 + response_rate) # Reward high response rate
    
    if signals.get("interview_completion_rate", 1.0) < 0.5:
        score *= 0.5
        
    # Notice period
    notice_days = signals.get("notice_period_days", 90)
    if notice_days <= 30:
        score += 0.5
        reasoning_points.append("Immediate joiner/short notice period.")
    elif notice_days > 60:
        score -= 0.5
        
    last_active = signals.get("last_active_date", "2000-01-01")
    try:
        la_dt = datetime.strptime(last_active, "%Y-%m-%d")
        days_inactive = (datetime(2024, 6, 1) - la_dt).days # Arbitrary fixed current date for stable eval
        if days_inactive > 90:
            score *= 0.8
    except:
        pass

    if score < 1.0:
        return score, "Not a strong fit based on profile semantics."
        
    reasoning = " ".join(reasoning_points)
    if not reasoning:
        reasoning = f"Solid general AI engineering profile with {yoe} YOE, but lacks specific vector/retrieval keywords."
        
    return score, reasoning

def main():
    args = parse_args()
    
    top_candidates = []
    
    # Support both gz and standard jsonl
    open_fn = gzip.open if args.candidates.endswith('.gz') else open
    
    with open_fn(args.candidates, 'rt', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line: continue
            
            candidate = json.loads(line)
            
            if is_honeypot(candidate):
                continue
                
            score, reasoning = score_candidate(candidate)
            
            # Use negative score for min-heap to keep top 100
            # Tie breaker: candidate_id
            heapq.heappush(top_candidates, (score, candidate["candidate_id"], reasoning))
            if len(top_candidates) > 100:
                heapq.heappop(top_candidates)
                
    # Sort descending
    top_candidates.sort(key=lambda x: (-x[0], x[1]))
    
    with open(args.out, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["candidate_id", "rank", "score", "reasoning"])
        for i, (score, cid, reasoning) in enumerate(top_candidates, 1):
            writer.writerow([cid, i, round(score, 4), reasoning])
            
if __name__ == "__main__":
    main()
