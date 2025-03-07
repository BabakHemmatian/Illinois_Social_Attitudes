import os
import argparse

def parse_range(value):
    """Parses a single integer or a range (e.g., '2007' or '2008-2010') into a list of integers,
    ensuring the start is ≥ 2007 and the end is ≤ 2023."""
    try:
        
        if '-' in value:  # Handling a range like "2008-2010"
            start, end = map(int, value.split('-'))
            if start > end:
                raise argparse.ArgumentTypeError(f"Invalid range '{value}': start must be ≤ end.")
        else:  # Handling a single integer
            start = end = int(value)

        # Enforce constraints on the range
        if start < 2007:
            raise argparse.ArgumentTypeError(f"Invalid value '{value}': years must be ≥ 2007.")
        if end > 2023:
            raise argparse.ArgumentTypeError(f"Invalid value '{value}': years must be ≤ 2023.")

        if start == end:
            return int(value)
        else:
            return list(range(start, end + 1))
    except ValueError:
        raise argparse.ArgumentTypeError(f"Invalid value '{value}': must be an integer or a range (e.g., 2007 or 2008-2010).")

# Helper function to load terms from a file
def load_terms(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return [line.lower().strip() for line in f if line.strip()]
    
# The list of social groups. Marginalized groups always listed first.
groups = {"sexuality":['gay','straight'],'age':['old','young'],'weight':['fat','thin'],'ability':['disabled','abled'],'race':['black','white'],'skin_tone':['dark','light']}

# the information we store for each comment in ISAAC
headers = ["id", "parent id", "text", "author", "time", "subreddit", "score", "matched patterns"]