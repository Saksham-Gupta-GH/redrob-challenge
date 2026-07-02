# Redrob Hackathon: Intelligent Candidate Discovery

## Team: Road Roller
**Members**: Saksham Gupta, Kushagra Gupta, Abhishek Patro

## Overview
This repository contains the top-100 ranking pipeline for the Intelligent Candidate Discovery Challenge.
Our solution strictly adheres to the < 5 minute runtime and 16GB memory constraints by utilizing streaming data parsing and a highly optimized CPU-bound heuristic ranker that avoids the overhead of runtime LLM inference.

### Key Features
1. **Honeypot Detection**: We immediately disqualify logically impossible profiles.
2. **Behavioral Multipliers**: Strong semantic fits are further enhanced if the candidate has recent activity and high recruiter response rates.
3. **Deterministic Memory Use**: Uses streaming generators and a fixed-size min-heap for the Top 100, requiring < 1GB RAM.
4. **Transparent Explainability**: Synthesizes the precise facts that led to a candidate's high score.

## How to Run

1. Place `candidates.jsonl` (or `candidates.jsonl.gz`) in the same directory.
2. Run the ranker:
```bash
python3 rank.py --candidates ./candidates.jsonl --out ./submission.csv
```
3. Validate:
```bash
python3 validate_submission.py submission.csv
```

## Requirements
- Python 3.9+
- No external libraries required (pure Python standard library).
